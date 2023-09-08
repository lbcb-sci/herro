use lazy_static::lazy_static;
use regex::bytes::RegexBuilder;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use regex::bytes::Regex;
use serde::Deserialize;
use serde::Serialize;
use std::fmt;
use std::io::Read;

use std::io::{BufRead, BufReader};

use crate::aligners::{cigar_to_string, CigarOp};
use crate::haec_io::HAECRecord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn return_other_id(&self, id: u32) -> u32 {
        if self.qid == id {
            return self.tid;
        } else {
            return self.qid;
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Alignment {
    pub overlap: Overlap,
    pub cigar: Vec<CigarOp>,
}

impl Alignment {
    pub fn new(overlap: Overlap, cigar: Vec<CigarOp>) -> Self {
        Alignment { overlap, cigar }
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

pub fn parse_paf(read: impl Read, name_to_id: &HashMap<&str, u32>) -> Vec<Alignment> {
    let mut reader = BufReader::new(read);

    let mut buffer = String::new();
    let mut processed = HashSet::default();

    let mut alignments = Vec::new();
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

        let cigar = data.last().unwrap();
        let cigar = parse_cigar(&cigar[5..]);

        buffer.clear();
        if tid == qid {
            // Cannot have self-overlaps
            continue;
        }

        if processed.contains(&(qid, tid)) {
            continue; // We assume the first overlap between two reads is the best one
        }
        processed.insert((qid, tid));

        let overlap = Overlap::new(qid, qlen, qstart, qend, strand, tid, tlen, tstart, tend);
        let alignment = Alignment::new(overlap, cigar);
        alignments.push(alignment);
    }

    alignments
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
            cigar_to_string(&aln.cigar),
        )
    }
}

lazy_static! {
    static ref CIGAR_PATTERN: Regex = RegexBuilder::new(r"(\d+)([MIDNSHP=X])")
        .unicode(false)
        .build()
        .unwrap();
}

fn parse_cigar(string: &str) -> Vec<CigarOp> {
    CIGAR_PATTERN
        .captures_iter(string.as_bytes())
        .map(|c| {
            let (_, [l, op]) = c.extract();

            let l = l.iter().fold(0, |acc, &d| acc * 10 + (d - b'0') as u32);
            match op {
                b"M" => CigarOp::Match(l),
                b"I" => CigarOp::Insertion(l),
                b"D" => CigarOp::Deletion(l),
                b"X" => CigarOp::Mismatch(l),
                b"=" => CigarOp::Match(l),
                _ => panic!("Invalid CIGAR operation."),
            }
        })
        .collect()
}
