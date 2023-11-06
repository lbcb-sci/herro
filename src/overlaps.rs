use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use zstd::stream::AutoFinishEncoder;

use std::fmt;

use std::fs::File;
use std::io::BufRead;
use std::io::BufWriter;
use std::io::Write;

use crate::aligners::{cigar_to_string, CigarOp};
use crate::haec_io::bytes_to_u32;
use crate::haec_io::HAECRecord;
use crate::LINE_ENDING;

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

pub fn parse_paf(
    mut reader: impl BufRead,
    name_to_id: &HashMap<&[u8], u32>,
    mut alns_writer: Option<&mut AutoFinishEncoder<BufWriter<File>>>,
) -> Vec<Alignment> {
    //let mut reader = BufReader::new(read);

    let mut buffer = Vec::new();
    let mut processed = HashSet::default();

    let mut alignments = Vec::new();
    while let Ok(len) = reader.read_until(LINE_ENDING, &mut buffer) {
        if len == 0 {
            break;
        }

        let mut data = buffer[..len - 1].split(|&c| c == b'\t');

        let qid = match name_to_id.get(data.next().unwrap()) {
            Some(qid) => *qid,
            None => {
                buffer.clear();
                continue;
            }
        };
        let qlen = bytes_to_u32(data.next().unwrap());
        let qstart = bytes_to_u32(data.next().unwrap());
        let qend = bytes_to_u32(data.next().unwrap());

        let strand = match data.next().unwrap()[0] {
            b'+' => Strand::Forward,
            b'-' => Strand::Reverse,
            _ => panic!("Invalid strand character."),
        };

        let tid = match name_to_id.get(data.next().unwrap()) {
            Some(tid) => *tid,
            None => {
                buffer.clear();
                continue;
            }
        };
        let tlen: u32 = bytes_to_u32(data.next().unwrap());
        let tstart: u32 = bytes_to_u32(data.next().unwrap());
        let tend: u32 = bytes_to_u32(data.next().unwrap());

        let cigar = data.last().unwrap();
        let cigar = parse_cigar(&cigar[5..]);

        if tid == qid {
            // Cannot have self-overlaps
            buffer.clear();
            continue;
        }

        if processed.contains(&(qid, tid)) {
            buffer.clear();
            continue; // We assume the first overlap between two reads is the best one
        }
        processed.insert((qid, tid));

        let overlap = Overlap::new(qid, qlen, qstart, qend, strand, tid, tlen, tstart, tend);
        let alignment = Alignment::new(overlap, cigar);
        alignments.push(alignment);

        if let Some(ref mut aw) = alns_writer {
            aw.write_all(&buffer[..len]).unwrap();
        }

        buffer.clear();
    }

    alignments
}

#[allow(dead_code)]
pub(crate) fn print_alignments(alignments: &[Alignment], reads: &[HAECRecord]) {
    for aln in alignments {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            std::str::from_utf8(&reads[aln.overlap.qid as usize].id).unwrap(),
            aln.overlap.qlen,
            aln.overlap.qstart,
            aln.overlap.qend,
            aln.overlap.strand,
            std::str::from_utf8(&reads[aln.overlap.tid as usize].id).unwrap(),
            aln.overlap.tlen,
            aln.overlap.tstart,
            aln.overlap.tend,
            cigar_to_string(&aln.cigar),
        )
    }
}

fn parse_cigar(cigar: &[u8]) -> Vec<CigarOp> {
    let mut ops = Vec::new();

    let mut l = 0;
    for &c in cigar {
        if c.is_ascii_digit() {
            l = l * 10 + (c - b'0') as u32;
        } else {
            match c {
                b'M' => ops.push(CigarOp::Match(l)),
                b'I' => ops.push(CigarOp::Insertion(l)),
                b'D' => ops.push(CigarOp::Deletion(l)),
                _ => panic!("Invalid CIGAR character."),
            }

            l = 0;
        }
    }

    ops
}
