use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use features::extract_features;

use haec_io::HAECRecord;
use indicatif::{ParallelProgressIterator, ProgressBar};

use overlaps::Alignment;
//use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use thread_local::ThreadLocal;

use std::{
    cell::RefCell,
    fs::File,
    io::{prelude::*, BufWriter},
    path::Path,
    sync::Arc,
    thread::{self, Scope},
};

use crate::{
    consensus::consensus_worker,
    features::{FeatsGenOutput, InferenceOutput},
    inference::inference_worker,
    overlaps::{generate_batches, read_batches},
};

mod aligners;
mod consensus;
mod features;
mod haec_io;
mod inference;
mod mm2;
mod overlaps;
mod windowing;

pub(crate) const READS_BATCH_SIZE: usize = 100_000;
pub(crate) const LINE_ENDING: u8 = b'\n';

pub enum AlnMode<V: AsRef<Path>> {
    None,
    Read(V),
    Write(V),
}

/*pub fn generate_features<T, U, V>(
    reads_path: T,
    output_path: U,
    threads: usize,
    window_size: u32,
    aln_mode: AlnMode<V>,
) where
    T: AsRef<Path>,
    U: AsRef<Path> + Send + Sync + Clone,
    V: AsRef<Path>,
{
    // Get fastq reads
    let reads = haec_io::get_reads(&reads_path, window_size);
    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (&*e.id, i as u32))
        .collect();
    eprintln!("Parsed {} reads.", reads.len());

    let sequences = Arc::new(ThreadLocal::new());
    let fo_tl = Arc::new(ThreadLocal::new());
    let feats_output = FeatsGenOutput::new(output_path);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let batches: Box<dyn Iterator<Item = (HashSet<u32>, Vec<Alignment>)>> = match aln_mode {
        AlnMode::None => {
            let batches = generate_batches(&reads, &name_to_id, &reads_path, threads, None::<T>);
            Box::new(batches)
        }
        AlnMode::Read(path) => {
            let batches = read_batches(&name_to_id, path);
            Box::new(batches)
        }
        AlnMode::Write(path) => {
            let batches = generate_batches(&reads, &name_to_id, &reads_path, threads, Some(path));
            Box::new(batches)
        }
    };

    for (tids, alignments) in batches {
        let mut read_to_alns = HashMap::default();
        alignments.into_iter().for_each(|aln| {
            if tids.contains(&aln.overlap.tid) {
                read_to_alns
                    .entry(aln.overlap.tid)
                    .or_insert_with(|| Vec::new())
                    .push(aln);
            }

            /*if tids.contains(&aln.overlap.qid) {
                read_to_alns
                    .entry(aln.overlap.qid)
                    .or_insert_with(|| Vec::new())
                    .push(aln);
            }*/
        });

        let n_targets = read_to_alns.len();
        read_to_alns
            .into_par_iter()
            .progress_count(n_targets as u64)
            .for_each(|(rid, alns)| {
                let (target, query) = sequences.get_or(|| {
                    (
                        RefCell::new(vec![0u8; max_len]),
                        RefCell::new(vec![0u8; max_len]),
                    )
                });
                let fo = fo_tl.get_or(|| RefCell::new(feats_output.clone()));

                extract_features(
                    rid,
                    &reads,
                    alns,
                    window_size,
                    (&mut target.borrow_mut(), &mut query.borrow_mut()),
                    &mut *fo.borrow_mut(),
                )
            });
    }
}*/

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
    //tch::set_num_interop_threads(1);
    //tch::maybe_init_cuda();

    // Get fastq reads
    let reads = haec_io::get_reads(&reads_path, window_size);
    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();
    eprintln!("Parsed {} reads.", reads.len());

    let (alns_sender, alns_receiver) = bounded(10_000);
    let (writer_sender, writer_receiver) = bounded(10_000);
    let (pbar_sender, pbar_receiver) = unbounded();
    let pbar = ProgressBar::new(reads.len() as u64);
    thread::scope(|s| {
        s.spawn(|| alignment_reader(&reads, &reads_path, aln_mode, threads, alns_sender));
        s.spawn(|| correction_writer(&reads, output_path, writer_receiver, pbar_sender));

        for device in devices {
            let (infer_sender, infer_recv) = bounded(10_000);
            let (cons_sender, cons_recv) = bounded(10_000);
            let writer_s = writer_sender.clone();

            for _ in 0..threads {
                let alns_r = alns_receiver.clone();
                let infer_s = infer_sender.clone();

                let ref_reads = &reads;
                s.spawn(move || {
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
            s.spawn(|| consensus_worker(&reads, cons_recv, writer_s, window_size));

            //drop(infer_sender);
        }

        drop(alns_receiver);
        drop(writer_sender);

        loop {
            match pbar_receiver.recv() {
                Ok(_) => pbar.inc(1),
                Err(_) => break,
            }
        }

        drop(pbar_receiver);
    });
}

fn alignment_reader<T: AsRef<Path>, U: AsRef<Path>>(
    reads: &[HAECRecord],
    reads_path: &T,
    aln_mode: AlnMode<U>,
    n_threads: usize,
    alns_sender: Sender<(u32, Vec<Alignment>)>,
) {
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (&*e.id, i as u32))
        .collect();

    let batches: Box<dyn Iterator<Item = (HashSet<u32>, Vec<Alignment>)>> = match aln_mode {
        AlnMode::None => {
            let batches = generate_batches(&reads, &name_to_id, &reads_path, n_threads, None::<T>);
            Box::new(batches)
        }
        AlnMode::Read(path) => {
            let batches = read_batches(&name_to_id, path);
            Box::new(batches)
        }
        AlnMode::Write(path) => {
            let batches = generate_batches(&reads, &name_to_id, &reads_path, n_threads, Some(path));
            Box::new(batches)
        }
    };

    for (tids, alignments) in batches {
        let mut read_to_alns = HashMap::default();
        alignments.into_iter().for_each(|aln| {
            if tids.contains(&aln.overlap.tid) {
                read_to_alns
                    .entry(aln.overlap.tid)
                    .or_insert_with(|| Vec::new())
                    .push(aln);
            }
        });

        read_to_alns
            .into_iter()
            .for_each(|example| alns_sender.send(example).unwrap())
    }
}

fn correction_writer<U: AsRef<Path>>(
    reads: &[HAECRecord],
    output_path: U,
    consensus_recv: Receiver<(usize, Vec<Vec<u8>>)>,
    pbar_sender: Sender<()>,
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
                write!(&mut writer, "_{}\n", i).unwrap();

                writer.write_all(&seq).unwrap();
                write!(&mut writer, "\n").unwrap();
            }
        }

        pbar_sender.send(());
    }
}

fn create_gpu_pipeline<'a, 'b>(
    reads: &'a [HAECRecord],
    window_size: u32,
    scope: &'b Scope,
    device: usize,
    model_path: &str,
    batch_size: usize,
    n_threads: usize,
    alns_recv: Receiver<(u32, Vec<Alignment>)>,
    writer_sender: Sender<(u32, Vec<Vec<u8>>)>,
) where
    'a: 'b,
{
}
