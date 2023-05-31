use aligners::align_overlaps;
use crossbeam_channel::bounded;
use features::extract_features;
use rustc_hash::FxHashMap as HashMap;

use std::{path::Path, thread};

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
    let reads = haec_io::get_reads(reads_path, window_size);
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id.as_str(), i as u32))
        .collect();
    eprintln!("Parsed {} reads.", reads.len());

    let mut overlaps = overlaps::parse_paf(paf_path, &name_to_id);
    extend_overlaps(&mut overlaps);
    eprintln!("Parsed {} overlaps", overlaps.len());

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
