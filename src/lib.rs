use std::io::prelude::*;
use std::{collections::HashMap, fs::File, io::BufWriter, path::Path};

use aligners::align_overlaps;
use features::extract_features;

use crate::aligners::cigar_to_string;

mod aligners;
mod features;
mod haec_io;
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

    let mut overlap_file = BufWriter::new(
        File::create("/raid/scratch/stanojevicd/aisg/hg002/chr20/overlap_analysis/p_longer_than_4096/overlaps_cigar_haec.paf")
            .expect("Uh Oh"),
    );
    for overlap in &overlaps {
        if overlap.cigar.is_none() {
            continue;
        }

        writeln!(
            overlap_file,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            &reads[overlap.qid as usize].id,
            overlap.qlen,
            overlap.qstart,
            overlap.qend,
            overlap.strand,
            &reads[overlap.tid as usize].id,
            overlap.tlen,
            overlap.tstart,
            overlap.tend,
            cigar_to_string(overlap.cigar.as_ref().unwrap())
        )
        .unwrap();
    }

    // extract_features(&reads, &overlaps, window_size, output_path);
}
