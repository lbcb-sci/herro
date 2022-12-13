use std::{collections::HashMap, path::Path};

use aligners::align_overlaps;
use features::extract_features;

mod aligners;
mod features;
mod haec_io;
mod overlaps;
mod windowing;

pub fn error_correction<P: AsRef<Path>>(
    reads_path: P,
    paf_path: P,
    threads: usize,
    window_size: u32,
) {
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
    extract_features(&reads, &overlaps, window_size);
}
