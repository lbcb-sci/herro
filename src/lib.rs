use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    fs::File,
    io::{BufWriter, Write},
    ops::Index,
    path::Path,
    process::exit,
    sync::Arc,
};

use aligners::CigarOp;
use indicatif::ParallelProgressIterator;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use thread_local::ThreadLocal;

mod aligners;
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

pub fn error_correction<P: AsRef<Path>>(reads_path: P, paf_path: P) {
    let mut reads = haec_io::get_reads(reads_path);
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id.as_str(), i as u32))
        .collect();

    println!("Parsed {} reads.", reads.len());

    let overlaps = overlaps::process_overlaps(overlaps::parse_paf(paf_path, &name_to_id));
    println!("Parsed {} overlaps", overlaps.len());

    rayon::ThreadPoolBuilder::new()
        .num_threads(7)
        .build_global()
        .unwrap();

    let aligners = Arc::new(ThreadLocal::new());

    overlaps
        .par_iter()
        .progress_count(overlaps.len() as u64)
        .for_each_with(aligners, |aligners, o| {
            //let o = &overlaps[i];
            let aligner = aligners.get_or(|| aligners::wfa::WFAAligner::new());

            let query = &reads[o.qid as usize].seq[o.qstart as usize..o.qend as usize];
            let query = match o.strand {
                overlaps::Strand::FORWARD => Cow::Borrowed(query),
                overlaps::Strand::REVERSE => Cow::Owned(aligners::reverse_complement(query)),
            };

            let target = &reads[o.tid as usize].seq[o.tstart as usize..o.tend as usize];

            let cigar = aligner.align(&query, target).unwrap();
        });

    exit(-1);

    let aligner = aligners::wfa::WFAAligner::new();
    let id_to_name: HashMap<_, _> = name_to_id.into_iter().map(|(k, v)| (v, k)).collect();
    for i in 0..100000 {
        let o = &overlaps[i];
        let query = &reads[o.qid as usize].seq[o.qstart as usize..o.qend as usize];
        let query = match o.strand {
            overlaps::Strand::FORWARD => Cow::Borrowed(query),
            overlaps::Strand::REVERSE => Cow::Owned(aligners::reverse_complement(query)),
        };

        let target = &reads[o.tid as usize].seq[o.tstart as usize..o.tend as usize];

        let cigar = aligner.align(&query, target).unwrap();
        /*if i % 100 == 0 {
            println!(
                "Cigar len {}, accuracy {}, cigar: {:?}",
                cigar.len(),
                calculate_accuracy(&cigar),
                cigar
            );
        }

        let accuracy = calculate_accuracy(&cigar);
        if accuracy > 1.9 {
            println!(
                "Query: {}, target: {}, accuracy: {}, cigar: {:?}",
                id_to_name.get(&o.qid).unwrap(),
                id_to_name.get(&o.tid).unwrap(),
                accuracy,
                cigar
            );
        }*/
    }

    //println!("Result: {}", cigar);

    /*let id_to_name: HashMap<_, _> = name_to_id.into_iter().map(|(k, v)| (v, k)).collect();
    let file = File::create("test.txt").unwrap();
    let mut writer = BufWriter::new(file);
    for o in overlaps {
        writeln!(
            &mut writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            id_to_name.get(&o.qid).unwrap(),
            o.qlen,
            o.qstart,
            o.qend,
            o.strand,
            id_to_name.get(&o.tid).unwrap(),
            o.tlen,
            o.tstart,
            o.tend
        )
        .unwrap();
    }*/
}
