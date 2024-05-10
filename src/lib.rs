use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use features::extract_features;

use haec_io::HAECRecord;

use pbars::{
    get_parse_reads_spinner, set_parse_reads_spinner_finish, track_progress, PBarNotification,
};

use std::{
    fs::File,
    io::{prelude::*, BufWriter},
    path::Path,
    thread::{self},
};

use crate::{
    consensus::consensus_worker,
    features::{FeatsGenOutput, InferenceOutput},
    inference::inference_worker,
    overlaps::alignment_reader,
};

mod aligners;
mod consensus;
mod features;
mod haec_io;
mod inference;
mod mm2;
mod overlaps;
mod pbars;
mod windowing;

pub(crate) const READS_BATCH_SIZE: usize = 100_000;
pub(crate) const ALN_CHANNEL_CAPACITY: usize = 50_000;
pub(crate) const LINE_ENDING: u8 = b'\n';
pub(crate) const INFER_CHANNEL_CAP_FACTOR: usize = 4;

pub enum AlnMode<V: AsRef<Path>> {
    None,
    Read(V),
    Write(V),
}

pub fn generate_features<T, U, V>(
    reads_path: T,
    output_path: U,
    threads: usize,
    window_size: u32,
    aln_mode: AlnMode<V>,
) where
    T: AsRef<Path> + Send + Sync,
    U: AsRef<Path> + Send + Sync + Clone,
    V: AsRef<Path> + Send,
{
    // Get fastq reads
    let reads = parse_reads(&reads_path, window_size);
    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();

    let (alns_sender, alns_receiver) = bounded(ALN_CHANNEL_CAPACITY);
    let (pbar_sender, pbar_receiver) = unbounded();
    thread::scope(|s| {
        let pbar_s = pbar_sender.clone();
        s.spawn(|| alignment_reader(&reads, &reads_path, aln_mode, threads, alns_sender, pbar_s));

        for _ in 0..threads {
            let pbar_s = pbar_sender.clone();

            s.spawn(|| {
                let mut feats_output = FeatsGenOutput::new(&output_path, pbar_s);
                let mut tbuf = vec![0; max_len];
                let mut qbuf = vec![0; max_len];

                loop {
                    let (rid, alns) = match alns_receiver.recv() {
                        Ok(out) => out,
                        Err(_) => break,
                    };

                    extract_features(
                        rid,
                        &reads,
                        alns,
                        window_size,
                        (&mut tbuf, &mut qbuf),
                        &mut feats_output,
                    );
                }
            });
        }

        drop(pbar_sender);

        track_progress(pbar_receiver);
    });
}

pub fn error_correction<T, U, V>(
    reads_path: T,
    model_path: &str,
    output_path: U,
    threads: usize,
    window_size: u32,
    devices: Vec<usize>,
    batch_size: usize,
    aln_mode: AlnMode<V>,
) where
    T: AsRef<Path> + Send + Sync,
    U: AsRef<Path> + Send + Sync,
    V: AsRef<Path> + Send,
{
    tch::set_num_threads(1);

    let reads = parse_reads(&reads_path, window_size);
    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();

    let (alns_sender, alns_receiver) = bounded(ALN_CHANNEL_CAPACITY);
    let (writer_sender, writer_receiver) = unbounded();
    let (pbar_sender, pbar_receiver) = unbounded();
    thread::scope(|s| {
        let pbar_s = pbar_sender.clone();
        s.spawn(|| alignment_reader(&reads, &reads_path, aln_mode, threads, alns_sender, pbar_s));
        s.spawn(|| correction_writer(&reads, output_path, writer_receiver, pbar_sender));

        for device in devices {
            let (infer_sender, infer_recv) = bounded(INFER_CHANNEL_CAP_FACTOR * threads);
            let (cons_sender, cons_recv) = unbounded();
            let writer_s = writer_sender.clone();

            for _ in 0..threads {
                let alns_r = alns_receiver.clone();
                let infer_s = infer_sender.clone();

                let ref_reads = &reads;
                s.spawn(move || {
                    let _guard = tch::no_grad_guard();

                    let mut feats_output = InferenceOutput::new(infer_s, batch_size);
                    let mut tbuf = vec![0; max_len];
                    let mut qbuf = vec![0; max_len];

                    loop {
                        let (rid, alns) = match alns_r.recv() {
                            Ok(out) => out,
                            Err(_) => break,
                        };

                        extract_features(
                            rid,
                            ref_reads,
                            alns,
                            window_size,
                            (&mut tbuf, &mut qbuf),
                            &mut feats_output,
                        );
                    }
                });
            }

            s.spawn(move || {
                inference_worker(
                    model_path,
                    tch::Device::Cuda(device),
                    infer_recv,
                    cons_sender,
                )
            });

            s.spawn(move || consensus_worker(cons_recv, writer_s));
        }

        drop(writer_sender);

        track_progress(pbar_receiver);
    });
}

fn parse_reads<P: AsRef<Path>>(reads_path: P, window_size: u32) -> Vec<HAECRecord> {
    // Get fastq reads
    let spinner = get_parse_reads_spinner(None);
    let reads = haec_io::get_reads(&reads_path, window_size);
    set_parse_reads_spinner_finish(reads.len(), spinner);

    reads
}

fn correction_writer<U: AsRef<Path>>(
    reads: &[HAECRecord],
    output_path: U,
    consensus_recv: Receiver<(usize, Vec<Vec<u8>>)>,
    pbar_sender: Sender<PBarNotification>,
) {
    let file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(file);

    loop {
        let (rid, seqs) = match consensus_recv.recv() {
            Ok(out) => out,
            Err(_) => break,
        };

        if seqs.len() == 1 {
            write!(&mut writer, ">").unwrap();
            writer.write_all(&reads[rid].id).unwrap();
            write!(&mut writer, "\n").unwrap();

            writer.write_all(&seqs[0]).unwrap();
            write!(&mut writer, "\n").unwrap();
        } else {
            for (i, seq) in seqs.into_iter().enumerate() {
                write!(&mut writer, ">").unwrap();
                writer.write_all(&reads[rid].id).unwrap();
                write!(&mut writer, ":{}\n", i).unwrap();

                writer.write_all(&seq).unwrap();
                write!(&mut writer, "\n").unwrap();
            }
        }

        pbar_sender.send(PBarNotification::Inc).unwrap();
    }
}
