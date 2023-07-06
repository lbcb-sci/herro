use aligners::align_overlaps;
use crossbeam_channel::{bounded, Sender};
use features::{extract_features, FeaturesOutput};

use haec_io::HAECRecord;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use inference::InputData;
use overlaps::{Alignment, Overlap};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thread_local::ThreadLocal;

use std::{
    cell::RefCell,
    collections::VecDeque,
    path::Path,
    process::exit,
    sync::{Arc, Mutex, RwLock},
    thread,
};

use crate::{
    aligners::wfa::WFAAligner,
    features::{FeatsGenOutput, InferenceOutput},
    inference::{consensus_worker, inference_worker},
};

mod aligners;
mod features;
mod haec_io;
mod inference;
mod overlaps;
mod windowing;

pub type ReadOverlaps = Vec<Arc<RwLock<Alignment>>>;
const READS_BATCH_SIZE: usize = 50_000;

pub fn generate_features<T, U, V>(
    reads: T,
    overlaps: U,
    output_path: V,
    feat_gen_threads: usize,
    window_size: u32,
) where
    T: AsRef<Path>,
    U: AsRef<Path>,
    V: AsRef<Path> + Send + Sync + Clone,
{
    // Get fastq reads
    let reads = haec_io::get_reads(reads, window_size);
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id.as_str(), i as u32))
        .collect();
    eprintln!("Parsed {} reads.", reads.len());

    // Parse, filteer and extend overlaps
    let read_to_overlaps = overlaps::parse_paf(overlaps, &name_to_id);

    // Build batches
    let batches = build_batches(read_to_overlaps);

    let feats_output = FeatsGenOutput::new(output_path);

    align_and_extract(batches, &reads, feats_output, window_size, feat_gen_threads);
}

pub fn error_correction<T, U, V>(
    reads_path: T,
    paf_path: U,
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
    let reads = haec_io::get_reads(reads_path, window_size);
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id.as_str(), i as u32))
        .collect();
    eprintln!("Parsed {} reads.", reads.len());

    // Parse, filteer and extend overlaps
    let read_to_overlaps = overlaps::parse_paf(paf_path, &name_to_id);

    // Build batches
    let batches = build_batches(read_to_overlaps);

    let (feats_sender, feats_receiver) = bounded(1000);
    let feats_output = InferenceOutput::new(feats_sender);

    let (pred_sender, pred_receiver) = bounded(1000);
    thread::scope(|s| {
        // Create inference thread for every GPU
        devices.iter().for_each(|d| {
            let fr = feats_receiver.clone();
            let ps = pred_sender.clone();

            s.spawn(|| inference_worker(model_path, tch::Device::Cuda(*d), fr, ps));
        });

        // Drop the handles so that the inference threads can exit
        drop(feats_receiver);
        drop(pred_sender);

        // Create consensus thread
        s.spawn(|| consensus_worker(output_path, &reads, pred_receiver, window_size));

        align_and_extract(batches, &reads, feats_output, window_size, threads);
    });

    //extract_features(&reads, &overlaps, window_size, feats_sender);
}

fn align_and_extract<'a, T>(
    batches: Vec<Vec<(u32, ReadOverlaps)>>,
    reads: &'a [HAECRecord],
    feats_output: T,
    window_size: u32,
    threads: usize,
) where
    T: FeaturesOutput<'a> + Clone + Send + Sync,
{
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();

    let aligners = Arc::new(ThreadLocal::new());
    let sequences = Arc::new(ThreadLocal::new());
    let fo_tl = Arc::new(ThreadLocal::new());

    let n_batches = batches.len();
    batches
        .into_iter()
        .progress_count(n_batches as u64)
        .for_each(|batch| {
            let batch_len = batch.len();

            batch
                .into_par_iter()
                .progress_count(batch_len as u64)
                .for_each(|(rid, overlaps)| {
                    let aligner = aligners.get_or(|| RefCell::new(WFAAligner::new()));
                    let (target, query) = sequences.get_or(|| {
                        (
                            RefCell::new(vec![0u8; max_len]),
                            RefCell::new(vec![0u8; max_len]),
                        )
                    });
                    let fo = fo_tl.get_or(|| RefCell::new(feats_output.clone()));

                    align_overlaps(
                        &overlaps,
                        &reads,
                        &mut aligner.borrow_mut(),
                        (&mut target.borrow_mut(), &mut query.borrow_mut()),
                    );

                    extract_features(
                        rid,
                        &reads,
                        &overlaps,
                        window_size,
                        (&mut target.borrow_mut(), &mut query.borrow_mut()),
                        &mut *fo.borrow_mut(),
                    )
                });
        });
}

fn build_batches(
    mut read_to_overlaps: HashMap<u32, ReadOverlaps>,
) -> Vec<Vec<(u32, ReadOverlaps)>> {
    let mut unprocessed: HashSet<_> = read_to_overlaps.keys().map(|k| *k).collect();
    let mut queue = VecDeque::new();

    let mut batches = Vec::new();
    let mut batch = Vec::with_capacity(READS_BATCH_SIZE);

    // While all the reads haven't been processed
    while unprocessed.len() > 0 {
        // No more reads in the queue, get a new one
        if queue.len() == 0 {
            let rid = *unprocessed.iter().next().unwrap();
            queue.push_back(rid);
        }

        let tid = queue.pop_front().unwrap();
        if unprocessed.contains(&tid) {
            // Add all the reads that overlap with this one
            read_to_overlaps.get(&tid).unwrap().iter().for_each(|aln| {
                let qid = aln.read().unwrap().overlap.return_other_id(tid);
                queue.push_back(qid);
            });

            batch.push((tid, read_to_overlaps.remove(&tid).unwrap()));
            unprocessed.remove(&tid);

            // Emit a new batch
            if batch.len() >= READS_BATCH_SIZE {
                batches.push(batch);
                batch = Vec::with_capacity(READS_BATCH_SIZE);
            }
        }
    }

    if batch.len() > 0 {
        batches.push(batch);
    }

    batches
}
