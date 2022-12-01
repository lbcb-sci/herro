use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    ops::Index,
    path::Path,
};

mod aligners;
mod haec_io;
mod overlaps;

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

    let aligner = aligners::wfa::WFAAligner::new();
    for i in 0..1000 {
        let o = &overlaps[i];
        let query = &reads[o.qid as usize].seq[o.qstart as usize..o.qend as usize];
        let target = &reads[o.tid as usize].seq[o.tstart as usize..o.tend as usize];

        let cigar = aligner.align(query, target).unwrap();
        if i % 100 == 0 {
            println!("Cigar len {}", cigar.len());
        }
    }

    let id_to_name: HashMap<_, _> = name_to_id.into_iter().map(|(k, v)| (v, k)).collect();
    println!(
        "Query: {}, target: {}",
        id_to_name.get(&overlaps[0].qid).unwrap(),
        id_to_name.get(&overlaps[0].tid).unwrap()
    );
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
