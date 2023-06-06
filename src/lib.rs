use aligners::align_overlaps;
use crossbeam_channel::{bounded, Sender};
use features::extract_features;

use haec_io::HAECRecord;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use inference::InputData;
use overlaps::Overlap;
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
    inference::{consensus_worker, inference_worker},
};

mod aligners;
mod features;
mod haec_io;
mod inference;
mod overlaps;
mod windowing;

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
    let batches = build_batches(&read_to_overlaps);

    let (feats_sender, feats_receiver) = bounded(1000);
    let (pred_sender, pred_receiver) = bounded(1000);

    thread::scope(|s| {
        // Create inference thread for every GPU
        devices.iter().for_each(|d| {
            let fr = feats_receiver.clone();
            let ps = pred_sender.clone();

            s.spawn(|| inference_worker(model_path, tch::Device::Cuda(*d), fr, ps));
        });

        // Create consensus thread
        s.spawn(|| consensus_worker(output_path, &reads, pred_receiver, window_size));

        align_and_extract(
            batches,
            &reads,
            &read_to_overlaps,
            feats_sender,
            window_size,
            threads,
        );
    });

    drop(feats_receiver);
    drop(pred_sender);

    //extract_features(&reads, &overlaps, window_size, feats_sender);
}

fn align_and_extract(
    batches: Vec<HashSet<u32>>,
    reads: &[HAECRecord],
    read_to_overlaps: &HashMap<u32, Vec<Arc<RwLock<Overlap>>>>,
    feats_sender: Sender<InputData>,
    window_size: u32,
    threads: usize,
) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();

    let aligners = Arc::new(ThreadLocal::new());
    let sequences = Arc::new(ThreadLocal::new());

    let n_batches = batches.len();
    batches
        .into_iter()
        .progress_count(n_batches as u64)
        .for_each(|batch| {
            let batch_len = batch.len();

            let fs_batch = feats_sender.clone();

            batch
                .into_par_iter()
                .progress_count(batch_len as u64)
                .for_each(|rid| {
                    let aligner = aligners.get_or(|| RefCell::new(WFAAligner::new()));
                    let (target, query) = sequences.get_or(|| {
                        (
                            RefCell::new(vec![0u8; max_len]),
                            RefCell::new(vec![0u8; max_len]),
                        )
                    });

                    let overlaps = read_to_overlaps.get(&rid).unwrap();
                    align_overlaps(
                        overlaps,
                        &reads,
                        &mut aligner.borrow_mut(),
                        (&mut target.borrow_mut(), &mut query.borrow_mut()),
                    );

                    extract_features(
                        rid,
                        &reads,
                        overlaps,
                        window_size,
                        (&mut target.borrow_mut(), &mut query.borrow_mut()),
                        fs_batch.clone(),
                    )
                });
        });
}

fn build_batches(read_to_overlaps: &HashMap<u32, Vec<Arc<RwLock<Overlap>>>>) -> Vec<HashSet<u32>> {
    let mut unprocessed: HashSet<_> = read_to_overlaps.keys().map(|k| *k).collect();
    let mut queue = VecDeque::new();

    let mut batches = Vec::new();
    let mut batch = HashSet::default();

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
            read_to_overlaps.get(&tid).unwrap().iter().for_each(|o| {
                let qid = o.read().unwrap().return_other_id(tid);
                queue.push_back(qid);
            });

            batch.insert(tid);
            unprocessed.remove(&tid);

            // Emit a new batch
            if batch.len() >= 50_000 {
                batches.push(batch);
                batch = HashSet::default();
            }
        }
    }

    if batch.len() > 0 {
        batches.push(batch);
    }

    batches
}
