use crossbeam_channel::bounded;
use features::extract_features;

use glob::glob;
use haec_io::HAECRecord;
use indicatif::ParallelProgressIterator;

use overlaps::{parse_paf, Alignment};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thread_local::ThreadLocal;

use std::{
    cell::RefCell,
    fs::File,
    io::{prelude::*, BufRead, BufReader, BufWriter},
    path::Path,
    sync::{Arc, RwLock},
    thread::{self},
};

use crate::{
    features::{FeatsGenOutput, InferenceOutput},
    inference::{consensus_worker, inference_worker},
};

mod aligners;
mod features;
mod haec_io;
mod inference;
mod mm2;
mod overlaps;
mod windowing;

pub type ReadOverlaps = Vec<Arc<RwLock<Alignment>>>;
const READS_BATCH_SIZE: usize = 100_000;
pub(crate) const BATCH_SIZE: usize = 32;
pub(crate) const LINE_ENDING: u8 = b'\n';

pub fn generate_features<T, U, V>(
    reads_path: T,
    overlaps: Option<U>,
    output_path: V,
    threads: usize,
    window_size: u32,
) where
    T: AsRef<Path>,
    U: AsRef<Path>,
    V: AsRef<Path> + Send + Sync + Clone,
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

    let batches: Box<dyn Iterator<Item = (HashSet<u32>, Vec<Alignment>)>> =
        if let Some(overlaps) = overlaps {
            let batches = read_batches(&name_to_id, overlaps);
            Box::new(batches)
        } else {
            let batches = generate_batches(&reads, &name_to_id, &reads_path, threads);
            Box::new(batches)
        };

    for (tids, alignments) in batches {
        let mut read_to_alns = HashMap::default();
        alignments.iter().for_each(|aln| {
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
                    &alns,
                    window_size,
                    (&mut target.borrow_mut(), &mut query.borrow_mut()),
                    &mut *fo.borrow_mut(),
                )
            });
    }
}

pub fn error_correction<T, U, V>(
    reads_path: T,
    paf_path: Option<U>,
    model_path: &str,
    output_path: V,
    threads: usize,
    window_size: u32,
    devices: &[usize],
) where
    T: AsRef<Path>,
    U: AsRef<Path>,
    V: AsRef<Path> + Send + Sync,
{
    // Get fastq reads
    let mut reads = haec_io::get_reads(&reads_path, window_size);
    let mut rng = StdRng::seed_from_u64(42);
    reads.shuffle(&mut rng);

    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (&*e.id, i as u32))
        .collect();
    eprintln!("Parsed {} reads.", reads.len());

    let (feats_sender, feats_receiver) = bounded(1000);
    let (pred_sender, pred_receiver) = bounded(1000);
    let (consensus_sender, consesus_receiver) = bounded(1000);

    let sequences = Arc::new(ThreadLocal::new());
    let fo_tl = Arc::new(ThreadLocal::new());
    let feats_output = InferenceOutput::new(feats_sender);

    thread::scope(|s| {
        // Create inference thread for every GPU
        devices.iter().for_each(|d| {
            let fr = feats_receiver.clone();
            let ps = pred_sender.clone();
            s.spawn(|| inference_worker(model_path, tch::Device::Cuda(*d), fr, ps));

            let pr = pred_receiver.clone();
            let cs = consensus_sender.clone();
            s.spawn(|| consensus_worker(&reads, pr, cs, window_size));
        });

        // Drop the handles so that the inference threads can exit
        drop(feats_receiver);
        drop(pred_sender);

        // Drop the handles so that the consensus threads can exit
        drop(pred_receiver);
        drop(consensus_sender);

        // Create writer thread
        let reads_ref = &reads;
        s.spawn(move || {
            let file = File::create(output_path).unwrap();
            let mut writer = BufWriter::new(file);

            loop {
                let (rid, seq) = match consesus_receiver.recv() {
                    Ok(out) => out,
                    Err(_) => break,
                };

                write!(&mut writer, ">").unwrap();
                writer.write_all(&reads_ref[rid].id).unwrap();
                write!(&mut writer, "\n").unwrap();

                writer.write_all(&seq).unwrap();
                write!(&mut writer, "\n").unwrap();
            }
        });

        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();

        let batches: Box<dyn Iterator<Item = (HashSet<u32>, Vec<Alignment>)>> =
            if let Some(batches) = paf_path {
                let batches = read_batches(&name_to_id, batches);
                Box::new(batches)
            } else {
                let batches = generate_batches(&reads, &name_to_id, &reads_path, threads);
                Box::new(batches)
            };

        for (tids, alignments) in batches {
            let mut read_to_alns = HashMap::default();
            alignments.iter().for_each(|aln| {
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

            let batch_size = read_to_alns.len();
            read_to_alns
                .into_par_iter()
                .progress_count(batch_size as u64)
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
                        &alns,
                        window_size,
                        (&mut target.borrow_mut(), &mut query.borrow_mut()),
                        &mut *fo.borrow_mut(),
                    )
                });
        }

        drop(fo_tl);
        drop(feats_output);
    });
}

fn generate_batches<'a, P>(
    reads: &'a [HAECRecord],
    name_to_id: &'a HashMap<&[u8], u32>,
    reads_path: P,
    threads: usize,
) -> impl Iterator<Item = (HashSet<u32>, Vec<Alignment>)> + 'a
where
    P: AsRef<Path>,
    P: 'a,
{
    reads.chunks(READS_BATCH_SIZE).filter_map(move |batch| {
        let mm2_out = BufReader::new(mm2::call_mm2(batch, &reads_path, threads));
        let alignments = overlaps::parse_paf(mm2_out, &name_to_id);

        let tids = batch
            .iter()
            .map(|r| *name_to_id.get(&*r.id).unwrap())
            .collect();

        Some((tids, alignments))
    })
}

fn read_batches<'a, P>(
    name_to_id: &'a HashMap<&[u8], u32>,
    batches: P,
) -> impl Iterator<Item = (HashSet<u32>, Vec<Alignment>)> + 'a
where
    P: AsRef<Path>,
    P: 'a,
{
    let g = batches.as_ref().join("*.oec.zst");
    glob(g.to_str().unwrap()).unwrap().map(|p| {
        let mut reader = {
            let file = File::open(p.unwrap()).unwrap();
            let reader = zstd::Decoder::new(file).unwrap();
            BufReader::new(reader)
        };

        // Read number of target reads
        let mut buf = Vec::new();
        let len = reader.read_until(LINE_ENDING, &mut buf).unwrap();
        let n_targets = buf[..len - 1]
            .iter()
            .fold(0, |acc, &d| acc * 10 + (d - b'0') as u32);

        let tids: HashSet<_> = (0..n_targets)
            .map(|_| {
                buf.clear();
                let len = reader.read_until(LINE_ENDING, &mut buf).unwrap();

                *name_to_id.get(&buf[..len - 1]).unwrap()
            })
            .collect();

        let alignments = parse_paf(&mut reader, name_to_id);

        (tids, alignments)
    })
}
