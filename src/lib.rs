use std::{
    collections::{HashMap, VecDeque},
    path::Path,
};

use aligners::CigarOp;

use aligners::align_overlaps;
use features::extract_features;

mod aligners;
mod features;
mod haec_io;
mod overlaps;

fn calculate_accuracy(cigar: &VecDeque<CigarOp>) -> f32 {
    let (mut matches, mut subs, mut ins, mut dels) = (0u32, 0u32, 0u32, 0u32);
    for op in cigar {
        match op {
            CigarOp::MATCH(l) => matches += l,
            CigarOp::MISMATCH(l) => subs += l,
            CigarOp::INSERTION(l) => ins += l,
            CigarOp::DELETION(l) => dels += l,
        };
    }

    let length = (matches + subs + ins + dels) as f32;
    matches as f32 / length
}

pub fn error_correction<P: AsRef<Path>>(reads_path: P, paf_path: P, threads: usize) {
    let reads = haec_io::get_reads(reads_path);
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id.as_str(), i as u32))
        .collect();
    eprintln!("Parsed {} reads.", reads.len());

    let mut overlaps = overlaps::process_overlaps(overlaps::parse_paf(paf_path, &name_to_id));
    eprintln!("Parsed {} overlaps", overlaps.len());

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    align_overlaps(&mut overlaps, &reads);
    extract_features(&reads, &overlaps);
}
