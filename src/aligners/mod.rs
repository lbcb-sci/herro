use std::{borrow::Cow, sync::Arc};

use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use thread_local::ThreadLocal;

use crate::{
    haec_io::HAECRecord,
    overlaps::{self, Overlap},
};

pub mod wfa;

#[derive(Debug, PartialEq, Clone, Eq)]
pub enum CigarOp {
    Match(u32),
    Mismatch(u32),
    Insertion(u32),
    Deletion(u32),
}

impl CigarOp {
    pub fn reverse(&self) -> Self {
        match self {
            Self::Insertion(l) => Self::Deletion(*l),
            Self::Deletion(l) => Self::Insertion(*l),
            _ => self.clone(),
        }
    }
}

impl From<(u32, char)> for CigarOp {
    fn from(cigar: (u32, char)) -> Self {
        match cigar.1 {
            'M' => CigarOp::Match(cigar.0),
            'X' => CigarOp::Mismatch(cigar.0),
            'I' => CigarOp::Insertion(cigar.0),
            'D' => CigarOp::Deletion(cigar.0),
            _ => panic!("Invalid cigar op {}", cigar.1),
        }
    }
}

impl ToString for CigarOp {
    fn to_string(&self) -> String {
        match self {
            CigarOp::Match(l) => format!("{}{}", l, '='),
            CigarOp::Mismatch(l) => format!("{}{}", l, 'X'),
            CigarOp::Deletion(l) => format!("{}{}", l, 'D'),
            CigarOp::Insertion(l) => format!("{}{}", l, 'I'),
        }
    }
}

pub fn cigar_to_string(cigar: &[CigarOp]) -> String {
    cigar.iter().map(|op| op.to_string()).collect()
}

#[inline]
fn complement(base: u8) -> u8 {
    match base {
        b'A' => b'T', // A -> T
        b'C' => b'G', // C -> G
        b'G' => b'C', // G -> C
        b'T' => b'A', // T -> A
        _ => panic!("Invalid base."),
    }
}

pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|c| complement(*c)).collect()
}

pub fn align_overlaps(overlaps: &mut [Overlap], reads: &[HAECRecord]) {
    let n_overlaps = overlaps.len();
    let aligners = Arc::new(ThreadLocal::new());

    overlaps
        .par_iter_mut()
        //.with_min_len(10)
        .progress_count(n_overlaps as u64)
        .for_each_with(aligners, |aligners, o| {
            let aligner = aligners.get_or(|| wfa::WFAAligner::default());

            let query = &reads[o.qid as usize].seq[o.qstart as usize..o.qend as usize];
            let query = match o.strand {
                overlaps::Strand::Forward => Cow::Borrowed(query),
                overlaps::Strand::Reverse => Cow::Owned(reverse_complement(query)),
            };

            let target = &reads[o.tid as usize].seq[o.tstart as usize..o.tend as usize];

            let align_result = aligner.align(&query, target).unwrap();
            o.cigar = Some(align_result.cigar);

            o.tstart += align_result.tstart;
            o.tend -= align_result.tend;

            match o.strand {
                overlaps::Strand::Forward => {
                    o.qstart += align_result.qstart;
                    o.qend -= align_result.qend;
                }
                overlaps::Strand::Reverse => {
                    o.qstart += align_result.qend;
                    o.qend -= align_result.qstart;
                }
            }

            o.accuracy = Some(calculate_accuracy(o.cigar.as_ref().unwrap()));
        });
}

fn calculate_accuracy(cigar: &[CigarOp]) -> f32 {
    let (mut matches, mut subs, mut ins, mut dels) = (0u32, 0u32, 0u32, 0u32);
    for op in cigar {
        match op {
            CigarOp::Match(l) => matches += l,
            CigarOp::Mismatch(l) => subs += l,
            CigarOp::Insertion(l) => ins += l,
            CigarOp::Deletion(l) => dels += l,
        };
    }

    let length = (matches + subs + ins + dels) as f32;
    matches as f32 / length
}

pub struct AlignmentResult {
    cigar: Vec<CigarOp>,
    tstart: u32,
    tend: u32,
    qstart: u32,
    qend: u32,
}

impl AlignmentResult {
    fn new(cigar: Vec<CigarOp>, tstart: u32, tend: u32, qstart: u32, qend: u32) -> Self {
        AlignmentResult {
            cigar,
            tstart,
            tend,
            qstart,
            qend,
        }
    }
}
