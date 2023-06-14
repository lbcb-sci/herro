use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::aligners::{cigar_to_string, CigarOp};
use crate::haec_io::HAECRecord;

const OL_THRESHOLD: u32 = 2500;

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
pub enum CigarStatus {
    Unprocessed,
    Unmapped,
    Mapped(Vec<CigarOp>),
}

impl CigarStatus {
    pub fn as_ref(&self) -> Option<&[CigarOp]> {
        match self {
            Self::Mapped(cigar) => Some(cigar),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug)]
pub struct Alignment {
    pub overlap: Overlap,
    pub cigar: CigarStatus,
}

impl Alignment {
    pub fn new(overlap: Overlap) -> Self {
        Alignment {
            overlap: overlap,
            cigar: CigarStatus::Unprocessed,
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

pub fn parse_paf<P: AsRef<Path>>(
    path: P,
    name_to_id: &HashMap<&str, u32>,
) -> HashMap<u32, Vec<Arc<RwLock<Alignment>>>> {
    let file = File::open(path).expect("Cannot open overlap file.");
    let mut reader = BufReader::new(file);

    let mut buffer = String::new();
    let mut processed = HashSet::default();
    let mut read_to_overlaps = HashMap::default();
    while let Ok(len) = reader.read_line(&mut buffer) {
        if len == 0 {
            break;
        }

        let mut data = buffer[..len - 1].split("\t");

        let qid = match name_to_id.get(data.next().unwrap()) {
            Some(qid) => *qid,
            None => {
                buffer.clear();
                continue;
            }
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
            None => {
                buffer.clear();
                continue;
            }
        };
        let tlen: u32 = data.next().unwrap().parse().unwrap();
        let tstart: u32 = data.next().unwrap().parse().unwrap();
        let tend: u32 = data.next().unwrap().parse().unwrap();

        buffer.clear();
        if tid == qid {
            // Cannot have self-overlaps
            continue;
        }

        if processed.contains(&(qid, tid)) {
            continue; // We assume the first overlap between two reads is the best one
        }
        processed.insert((qid, tid));

        if is_valid_overlap(qlen, qstart, qend, strand, tlen, tstart, tend) {
            let mut alignment = Alignment::new(Overlap::new(
                qid, qlen, qstart, qend, strand, tid, tlen, tstart, tend,
            ));
            extend_overlap(&mut alignment.overlap);
            let overlap = Arc::new(RwLock::new(alignment));

            read_to_overlaps
                .entry(tid)
                .or_insert_with(|| Vec::new())
                .push(Arc::clone(&overlap));

            read_to_overlaps
                .entry(qid)
                .or_insert_with(|| Vec::new())
                .push(overlap);
        }
    }

    read_to_overlaps
}

fn is_valid_overlap(
    qlen: u32,
    qstart: u32,
    qend: u32,
    strand: Strand,
    tlen: u32,
    tstart: u32,
    tend: u32,
) -> bool {
    let ratio = (tend - tstart) as f64 / (qend - qstart) as f64;
    if ratio < 0.9 || ratio > 1.111 {
        return false;
    }

    if (qlen - (qend - qstart)) <= OL_THRESHOLD {
        return true;
    }

    // Target contained in query
    if (tlen - (tend - tstart)) <= OL_THRESHOLD {
        return true;
    }

    let (qstart, qend) = match strand {
        Strand::Forward => (qstart, qend),
        Strand::Reverse => (qlen - qend, qlen - qstart),
    };

    // Prefix overlap between query and target
    if qstart > OL_THRESHOLD && tstart <= OL_THRESHOLD && (qlen - qend) <= OL_THRESHOLD {
        return true;
    }

    // Suffix overlap between query and target
    if tstart > OL_THRESHOLD && qstart <= OL_THRESHOLD && (tlen - tend) <= OL_THRESHOLD {
        return true;
    }

    false
}

fn extend_overlap(overlap: &mut Overlap) {
    match overlap.strand {
        Strand::Forward => {
            let beginning = overlap.tstart.min(overlap.qstart).min(2500);
            overlap.tstart -= beginning;
            overlap.qstart -= beginning;
            let end = (overlap.tlen - overlap.tend)
                .min(overlap.qlen - overlap.qend)
                .min(2500);
            overlap.tend += end;
            overlap.qend += end;
        }
        Strand::Reverse => {
            let beginning = overlap.tstart.min(overlap.qlen - overlap.qend).min(2500);
            overlap.tstart -= beginning;
            overlap.qend += beginning;

            let end = (overlap.tlen - overlap.tend).min(overlap.qstart).min(2500);
            overlap.tend += end;
            overlap.qstart -= end;
        }
    }
}

#[allow(dead_code)]
pub(crate) fn print_alignments(alignments: &[Alignment], reads: &[HAECRecord]) {
    for aln in alignments {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            reads[aln.overlap.qid as usize].id,
            aln.overlap.qlen,
            aln.overlap.qstart,
            aln.overlap.qend,
            aln.overlap.strand,
            reads[aln.overlap.tid as usize].id,
            aln.overlap.tlen,
            aln.overlap.tstart,
            aln.overlap.tend,
            match aln.cigar {
                CigarStatus::Mapped(ref c) => cigar_to_string(c),
                CigarStatus::Unprocessed | CigarStatus::Unmapped => "".to_string(),
            }
        )
    }
}
