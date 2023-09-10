use crossbeam_channel::bounded;
use features::extract_features;

use indicatif::{ParallelProgressIterator, ProgressIterator};

use overlaps::Alignment;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thread_local::ThreadLocal;

use std::{
    cell::RefCell,
    path::Path,
    sync::{Arc, RwLock},
    thread,
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
const READS_BATCH_SIZE: usize = 50_000;

pub fn generate_features<T, U, V>(
    reads_path: T,
    overlaps: Option<U>,
    output_path: V,
    feat_gen_threads: usize,
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
        .map(|(i, e)| (e.id.as_str(), i as u32))
        .collect();
    eprintln!("Parsed {} reads.", reads.len());

    let sequences = Arc::new(ThreadLocal::new());
    let fo_tl = Arc::new(ThreadLocal::new());
    let feats_output = FeatsGenOutput::new(output_path);

    rayon::ThreadPoolBuilder::new()
        .num_threads(feat_gen_threads)
        .build_global()
        .unwrap();

    for batch in reads.chunks(READS_BATCH_SIZE).progress() {
        let alignments = if overlaps.is_none() {
            let mm2_out = mm2::call_mm2(batch, &reads_path, feat_gen_threads);
            overlaps::parse_paf(mm2_out, &name_to_id)
        } else {
            /*let file = File::open(&overlaps.as_ref().unwrap()).unwrap();
            overlaps::parse_paf(file, &name_to_id)*/

            unimplemented!()
        };

        // Parse, filter and extend overlaps
        let tids: HashSet<_> = batch
            .iter()
            .map(|r| *name_to_id.get(&*r.id).unwrap())
            .collect();

        let mut read_to_alns = HashMap::default();
        alignments.iter().for_each(|aln| {
            if tids.contains(&aln.overlap.tid) {
                read_to_alns
                    .entry(aln.overlap.tid)
                    .or_insert_with(|| Vec::new())
                    .push(aln);
            }

            if tids.contains(&aln.overlap.qid) {
                read_to_alns
                    .entry(aln.overlap.qid)
                    .or_insert_with(|| Vec::new())
                    .push(aln);
            }
        });

        read_to_alns
            .into_par_iter()
            .progress_count(batch.len() as u64)
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
    let reads = haec_io::get_reads(&reads_path, window_size);
    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id.as_str(), i as u32))
        .collect();
    eprintln!("Parsed {} reads.", reads.len());

    let sequences = Arc::new(ThreadLocal::new());
    let fo_tl = Arc::new(ThreadLocal::new());

    let (feats_sender, feats_receiver) = bounded(1000);
    let (pred_sender, pred_receiver) = bounded(1000);
    let feats_output = InferenceOutput::new(feats_sender);

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

        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();

        for batch in reads.chunks(READS_BATCH_SIZE).progress() {
            let alignments = if paf_path.is_none() {
                let mm2_out = mm2::call_mm2(batch, &reads_path, threads);
                overlaps::parse_paf(mm2_out, &name_to_id)
            } else {
                /*let file = File::open(&overlaps.as_ref().unwrap()).unwrap();
                overlaps::parse_paf(file, &name_to_id)*/

                unimplemented!()
            };

            // Parse, filter and extend overlaps
            let tids: HashSet<_> = batch
                .iter()
                .map(|r| *name_to_id.get(&*r.id).unwrap())
                .collect();

            let mut read_to_alns = HashMap::default();
            alignments.iter().for_each(|aln| {
                if tids.contains(&aln.overlap.tid) {
                    read_to_alns
                        .entry(aln.overlap.tid)
                        .or_insert_with(|| Vec::new())
                        .push(aln);
                }

                if tids.contains(&aln.overlap.qid) {
                    read_to_alns
                        .entry(aln.overlap.qid)
                        .or_insert_with(|| Vec::new())
                        .push(aln);
                }
            });

            read_to_alns
                .into_par_iter()
                .progress_count(batch.len() as u64)
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
    });
}
