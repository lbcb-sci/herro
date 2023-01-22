use std::fmt::{self};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::aligners::CigarOp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    Forward,
    Reverse,
}

impl fmt::Display for Strand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Self::Forward => '+',
            Self::Reverse => '-',
        };

        write!(f, "{}", s)
    }
}

#[derive(Debug)]
pub struct Overlap {
    pub qid: u32,
    pub qlen: u32,
    pub qstart: u32,
    pub qend: u32,
    pub strand: Strand,
    pub tid: u32,
    pub tlen: u32,
    pub tstart: u32,
    pub tend: u32,
    pub cigar: Option<Vec<CigarOp>>,
    pub accuracy: Option<f32>,
}

impl Overlap {
    pub fn new(
        qid: u32,
        qlen: u32,
        qstart: u32,
        qend: u32,
        strand: Strand,
        tid: u32,
        tlen: u32,
        tstart: u32,
        tend: u32,
    ) -> Self {
        Overlap {
            qid,
            qlen,
            qstart,
            qend,
            strand,
            tid,
            tlen,
            tstart,
            tend,
            cigar: None,
            accuracy: None,
        }
    }

    fn target_overlap_length(&self) -> u32 {
        return self.tend - self.tstart;
    }

    pub fn return_other_id(&self, id: u32) -> u32 {
        if self.qid == id {
            return self.tid;
        } else {
            return self.qid;
        }
    }
}

impl PartialEq for Overlap {
    fn eq(&self, other: &Self) -> bool {
        self.qid == other.qid
            && self.qstart == other.qstart
            && self.qend == other.qend
            && self.strand == other.strand
            && self.tid == other.tid
            && self.tstart == other.tstart
            && self.tend == other.tend
    }
}

impl Eq for Overlap {}

const EXTEND_LENGTH: u32 = 500;
const OL_THRESHOLD: u32 = 500;
const DOUBLE_OL_THRESHOLD: u32 = 2 * OL_THRESHOLD;

pub fn parse_paf<P: AsRef<Path>>(path: P, name_to_id: &HashMap<&str, u32>) -> Vec<Overlap> {
    let file = File::open(path).expect("Cannot open overlap file.");
    let reader = BufReader::new(file);

    let mut overlaps = Vec::new();
    let mut ratio_removed = 0u32;
    let mut processed = HashSet::new();
    for line in reader.lines() {
        let line = line.unwrap();

        let mut data = line.split("\t");

        let qid = match name_to_id.get(data.next().unwrap()) {
            Some(qid) => *qid,
            None => continue,
        };
        let qlen: u32 = data.next().unwrap().parse().unwrap();
        let qstart: u32 = data.next().unwrap().parse().unwrap();
        let qend: u32 = data.next().unwrap().parse().unwrap();

        let strand = match data.next().unwrap() {
            "+" => Strand::Forward,
            "-" => Strand::Reverse,
            _ => panic!("Invalid strand character."),
        };

        let tid = match name_to_id.get(data.next().unwrap()) {
            Some(tid) => *tid,
            None => continue,
        };
        let tlen: u32 = data.next().unwrap().parse().unwrap();
        let tstart: u32 = data.next().unwrap().parse().unwrap();
        let tend: u32 = data.next().unwrap().parse().unwrap();

        if tid == qid {
            // Cannot have self-overlaps
            continue;
        }

        if processed.contains(&(qid, tid)) {
            continue; // We assume the first one is the best one
        }
        processed.insert((qid, tid));

        let ratio = (tend - tstart) as f64 / (qend - qstart) as f64;
        if ratio < 0.9 || ratio > 1.111 {
            // Filter based on length ratio
            ratio_removed += 1;
            continue;
        }

        let overlap = Overlap::new(qid, qlen, qstart, qend, strand, tid, tlen, tstart, tend);
        overlaps.push(overlap);
    }

    eprintln!("Removed overlaps due to ratio {}", ratio_removed);
    eprintln!("Total overlaps {}", overlaps.len());
    overlaps
}

fn find_primary_overlaps(overlaps: &[Overlap]) -> HashSet<usize> {
    let mut ovlps_for_pairs = HashMap::new();
    for i in 0..overlaps.len() {
        ovlps_for_pairs
            .entry((overlaps[i].qid, overlaps[i].tid))
            .or_insert_with(|| Vec::new())
            .push(i);
    }

    let mut kept_overlap_ids = HashSet::new();
    for ((_, _), ovlps) in ovlps_for_pairs {
        let kept_id = match ovlps.len() {
            1 => ovlps[0],
            _ => ovlps
                .into_iter()
                .max_by_key(|id| overlaps[*id].target_overlap_length())
                .unwrap(),
        };

        kept_overlap_ids.insert(kept_id);
    }

    kept_overlap_ids
}

fn extend_overlap(overlap: &mut Overlap) {
    overlap.qstart = overlap.qstart.checked_sub(EXTEND_LENGTH).unwrap_or(0);
    overlap.qend = overlap.qlen.min(overlap.qend + EXTEND_LENGTH); // Should not overflow

    overlap.tstart = overlap.tstart.checked_sub(EXTEND_LENGTH).unwrap_or(0);
    overlap.tend = overlap.tlen.min(overlap.tend + EXTEND_LENGTH);
}

fn is_valid_overlap(overlap: &Overlap) -> bool {
    // Query contained in target
    if (overlap.qlen - (overlap.qend - overlap.qstart)) <= OL_THRESHOLD {
        return true;
    }

    // Target contained in query
    if (overlap.tlen - (overlap.tend - overlap.tstart)) <= OL_THRESHOLD {
        return true;
    }

    let (qstart, qend) = match overlap.strand {
        Strand::Forward => (overlap.qstart, overlap.qend),
        Strand::Reverse => (overlap.qlen - overlap.qend, overlap.qlen - overlap.qstart),
    };

    // Prefix overlap between query and target
    if qstart > OL_THRESHOLD
        && overlap.tstart <= OL_THRESHOLD
        && (overlap.qlen - qend) <= OL_THRESHOLD
    {
        return true;
    }

    // Suffix overlap between query and target
    if overlap.tstart > OL_THRESHOLD
        && qstart <= OL_THRESHOLD
        && (overlap.tlen - overlap.tend) <= OL_THRESHOLD
    {
        return true;
    }

    false
}

pub fn process_overlaps(overlaps: Vec<Overlap>) -> Vec<Overlap> {
    //let primary_overlaps = find_primary_overlaps(&overlaps);
    //println!("Number of primary overlaps {}", primary_overlaps.len());

    /*overlaps
    .into_iter()
    .enumerate()
    //.filter(|(i, _)| primary_overlaps.contains(i)) // Keep only primary overlaps
    .map(|(_, o)| {
        //extend_overlap(&mut o);
        o
    })
    .filter(|o| {
        let b = is_valid_overlap(o);
        b
    })
    .collect()*/

    overlaps
        .into_iter()
        .filter(|o| {
            let b = is_valid_overlap(o);
            b
        })
        .collect()
}
