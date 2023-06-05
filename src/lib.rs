use aligners::align_overlaps;
use crossbeam_channel::bounded;
use features::extract_features;
use haec_io::HAECRecord;
use overlaps::Overlap;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use std::{collections::VecDeque, path::Path, process::exit, thread};

use crate::{
    inference::{consensus_worker, inference_worker},
    overlaps::extend_overlaps,
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
    let (mut overlaps, read_to_overlaps) = overlaps::parse_paf(paf_path, &name_to_id);
    extend_overlaps(&mut overlaps);
    eprintln!("Parsed {} overlaps", overlaps.len());

    // Build batches
    let batches = build_batches(&overlaps, &read_to_overlaps);
    batches.iter().for_each(|b| println!("{:?}", b));

    exit(-1);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let overlaps = align_overlaps(overlaps, &reads);

    let (feats_sender, feats_receiver) = bounded(1000);
    let (pred_sender, pred_receiver) = bounded(1000);

    //extract_features(&reads, &overlaps, window_size, feats_sender);

    thread::scope(|s| {
        // Create inference thread for every GPU
        devices.iter().for_each(|d| {
            let fr = feats_receiver.clone();
            let ps = pred_sender.clone();

            s.spawn(|| inference_worker(model_path, tch::Device::Cuda(*d), fr, ps));
        });

        // Create consensus thread
        s.spawn(|| consensus_worker(output_path, &reads, pred_receiver, window_size));

        extract_features(&reads, &overlaps, window_size, feats_sender);
    });
}

fn build_batches(
    overlaps: &[Overlap],
    read_to_overlaps: &HashMap<u32, HashSet<u32>>,
) -> Vec<HashSet<u32>> {
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
            read_to_overlaps.get(&tid).unwrap().iter().for_each(|oid| {
                let qid = overlaps[*oid as usize].return_other_id(tid);
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
