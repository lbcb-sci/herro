#![allow(dead_code)]
use itertools::Itertools;

use crate::overlaps::Strand;

#[derive(Debug, PartialEq, Copy, Clone, Eq)]
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
            _ => *self,
        }
    }

    pub fn get_length(&self) -> u32 {
        match self {
            Self::Match(l) => *l,
            Self::Mismatch(l) => *l,
            Self::Insertion(l) => *l,
            Self::Deletion(l) => *l,
        }
    }

    pub fn with_length(&self, length: u32) -> Self {
        match self {
            Self::Match(_) => CigarOp::Match(length),
            Self::Mismatch(_) => CigarOp::Mismatch(length),
            Self::Insertion(_) => CigarOp::Insertion(length),
            Self::Deletion(_) => CigarOp::Deletion(length),
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

#[allow(dead_code)]
pub fn cigar_to_string(cigar: &[CigarOp]) -> String {
    cigar.iter().map(|op| op.to_string()).collect()
}

/*pub fn align_overlaps(
    overlaps: &[Arc<RwLock<Alignment>>],
    reads: &[HAECRecord],
    aligner: &mut WFAAligner,
    (tbuf, qbuf): (&mut [u8], &mut [u8]),
) {
    overlaps.iter().for_each(|overlap| {
        let mut aln = overlap.write().unwrap();
        if !matches!(aln.cigar, CigarStatus::Unprocessed) {
            return;
        }

        let qlen = aln.overlap.qend as usize - aln.overlap.qstart as usize;
        if aln.overlap.strand == overlaps::Strand::Forward {
            reads[aln.overlap.qid as usize]
                .seq
                .get_subseq(aln.overlap.qstart as usize..aln.overlap.qend as usize, qbuf);
        } else {
            reads[aln.overlap.qid as usize]
                .seq
                .get_rc_subseq(aln.overlap.qstart as usize..aln.overlap.qend as usize, qbuf);
        };

        let tlen = aln.overlap.tend as usize - aln.overlap.tstart as usize;
        reads[aln.overlap.tid as usize]
            .seq
            .get_subseq(aln.overlap.tstart as usize..aln.overlap.tend as usize, tbuf);

        let align_result = aligner.align(&qbuf[..qlen], &tbuf[..tlen]);
        if let Some(result) = align_result {
            aln.cigar = CigarStatus::Mapped(result.cigar);

            aln.overlap.tstart += result.tstart;
            aln.overlap.tend -= result.tend;

            match aln.overlap.strand {
                overlaps::Strand::Forward => {
                    aln.overlap.qstart += result.qstart;
                    aln.overlap.qend -= result.qend;
                }
                overlaps::Strand::Reverse => {
                    aln.overlap.qstart += result.qend;
                    aln.overlap.qend -= result.qstart;
                }
            }
        } else {
            aln.cigar = CigarStatus::Unmapped;
            return;
        }
    })
}*/

pub(crate) fn calculate_accuracy(cigar: &[CigarOp]) -> f32 {
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

pub(crate) fn get_proper_cigar(cigar: &[CigarOp], is_target: bool, strand: Strand) -> Vec<CigarOp> {
    // Replace mismatch with match
    let cigar = cigar.iter().map(|op| {
        if let CigarOp::Mismatch(l) = op {
            CigarOp::Match(*l)
        } else {
            *op
        }
    });

    let iter: Box<dyn Iterator<Item = CigarOp>>;
    if !is_target {
        let iter_rev = cigar.map(|c| c.reverse());
        if let Strand::Reverse = strand {
            iter = Box::new(iter_rev.rev());
        } else {
            iter = Box::new(iter_rev);
        }
    } else {
        iter = Box::new(cigar);
    }

    // Merge ops -> because of converting mismatches to matches
    iter.coalesce(|prev_op, curr_op| {
        if std::mem::discriminant(&prev_op) == std::mem::discriminant(&curr_op) {
            Ok(prev_op.with_length(prev_op.get_length() + curr_op.get_length()))
        } else {
            Err((prev_op, curr_op))
        }
    })
    .collect()
}

pub(crate) fn fix_cigar(cigar: &mut Vec<CigarOp>, target: &[u8], query: &[u8]) -> (u32, u32) {
    // Left-alignment of indels
    // https://github.com/lh3/minimap2/blob/master/align.c#L91

    let (mut tpos, mut qpos) = (0usize, 0usize);
    for i in 0..cigar.len() {
        if let CigarOp::Match(l) | CigarOp::Mismatch(l) = &cigar[i] {
            tpos += *l as usize;
            qpos += *l as usize;
        } else {
            if i > 0
                && i < cigar.len() - 1
                && matches!(cigar[i - 1], CigarOp::Match(_) | CigarOp::Mismatch(_))
                && matches!(cigar[i + 1], CigarOp::Match(_) | CigarOp::Mismatch(_))
            {
                let prev_len = match &cigar[i - 1] {
                    CigarOp::Match(pl) => *pl as usize,
                    CigarOp::Mismatch(pl) => *pl as usize,
                    _ => unreachable!(),
                };
                let mut l = 0;

                if let CigarOp::Insertion(len) = &cigar[i] {
                    while l < prev_len {
                        if query[qpos - 1 - l] != query[qpos + *len as usize - 1 - l] {
                            break;
                        }

                        l += 1;
                    }
                } else {
                    let len = cigar[i].get_length() as usize;

                    while l < prev_len {
                        if target[tpos - 1 - l] != target[tpos + len - 1 - l] {
                            break;
                        }

                        l += 1;
                    }
                }

                if l > 0 {
                    cigar[i - 1] = match &cigar[i - 1] {
                        CigarOp::Match(v) => CigarOp::Match(*v - l as u32),
                        CigarOp::Mismatch(v) => CigarOp::Mismatch(*v - l as u32),
                        _ => unreachable!(),
                    };

                    cigar[i + 1] = match &cigar[i + 1] {
                        CigarOp::Match(v) => CigarOp::Match(*v + l as u32),
                        CigarOp::Mismatch(v) => CigarOp::Mismatch(*v + l as u32),
                        _ => unreachable!(),
                    };

                    tpos -= l;
                    qpos -= l;
                }
            }

            match &cigar[i] {
                CigarOp::Insertion(len) => qpos += *len as usize,
                CigarOp::Deletion(len) => tpos += *len as usize,
                _ => unreachable!(),
            }
        }
    }

    let mut is_start = true;
    let (mut tshift, mut qshift) = (0, 0);
    cigar.retain(|op| {
        if is_start {
            match op {
                CigarOp::Match(l) | CigarOp::Mismatch(l) if *l > 0 => {
                    is_start = false;
                    return true;
                }
                CigarOp::Match(_) | CigarOp::Mismatch(_) => return false,
                CigarOp::Insertion(ref l) => {
                    is_start = false;
                    qshift = *l;
                    return false;
                }
                CigarOp::Deletion(ref l) => {
                    is_start = false;
                    tshift = *l;
                    return false;
                }
            }
        }

        if op.get_length() > 0 {
            return true;
        } else {
            return false;
        };
    });

    let mut l = 0;
    for i in 0..cigar.len() {
        if i == cigar.len() - 1
            || std::mem::discriminant(&cigar[i]) != std::mem::discriminant(&cigar[i + 1])
        {
            cigar[l] = cigar[i];
            l += 1;
        } else {
            cigar[i + 1] = cigar[i].with_length(cigar[i].get_length() + cigar[i + 1].get_length());
        }
    }
    cigar.drain(l..);

    (tshift, qshift)
}

#[cfg(test)]
mod tests {
    use super::{fix_cigar, CigarOp};

    #[test]
    fn fix_cigar_test1() {
        let target = "TTTTGTTTTTTTTTTCTTTTTTTTTTTTTTTTTTTGCT".as_bytes();
        let query = "TTTTGTTTTTTTTTTCTTTTTTTTTTTTTTTGCT".as_bytes();
        let mut cigar = vec![CigarOp::Match(31), CigarOp::Deletion(4), CigarOp::Match(3)];

        fix_cigar(&mut cigar, target, query);
        assert_eq!(
            cigar,
            [CigarOp::Match(16), CigarOp::Deletion(4), CigarOp::Match(18)]
        )
    }

    #[test]
    fn fix_cigar_test2() {
        let target = "AGCAAAAAAAAAAAAAAAGAAAAAAAAAACAAAA".as_bytes();
        let query = "AGCAAAAAAAAAAAAAAAAAAAGAAAAAAAAAACAAAA".as_bytes();
        let mut cigar = vec![
            CigarOp::Match(18),
            CigarOp::Insertion(4),
            CigarOp::Match(16),
        ];

        fix_cigar(&mut cigar, target, query);
        assert_eq!(
            cigar,
            [CigarOp::Match(3), CigarOp::Insertion(4), CigarOp::Match(31)]
        )
    }

    #[test]
    fn fix_cigar_test3() {
        let target = "CACCAGGCCA".as_bytes();
        let query = "CACCAGCCA".as_bytes();
        let mut cigar = vec![CigarOp::Match(6), CigarOp::Deletion(1), CigarOp::Match(3)];

        fix_cigar(&mut cigar, target, query);
        assert_eq!(
            cigar,
            [CigarOp::Match(5), CigarOp::Deletion(1), CigarOp::Match(4)]
        )
    }
}
