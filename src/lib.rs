use std::{collections::HashMap, path::Path};

use aligners::align_overlaps;
use features::extract_features;

use crate::aligners::CigarOp;

mod aligners;
mod features;
mod haec_io;
mod inference;
mod overlaps;
mod windowing;

pub fn error_correction<T, U, V>(
    reads_path: T,
    paf_path: U,
    output_path: V,
    threads: usize,
    window_size: u32,
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

    let mut overlaps = overlaps::process_overlaps(overlaps::parse_paf(paf_path, &name_to_id));
    eprintln!("Parsed {} overlaps", overlaps.len());

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    align_overlaps(&mut overlaps, &reads);
    overlaps.retain(|o| o.cigar.is_some());

    extract_features(&reads, &overlaps, window_size, output_path);
}
