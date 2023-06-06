use std::ptr::NonNull;

use itertools::Itertools;

use super::{AlignmentResult, CigarOp};
use wfa2_sys as wfa;

const MISMATCH_COST: i32 = 5;
const GAP_OPEN1_COST: i32 = 6;
const GAP_EXT1_COST: i32 = 2;
const GAP_OPEN2_COST: i32 = 24;
const GAP_EXT2_COST: i32 = 1;

const MIN_WAVEFRONT_LENGTH: i32 = 10;
const MAX_DISTANCE_THRESHOLD: i32 = 200;
const STEPS_BETWEEN_CUTOFFS: i32 = 1;

const MIN_END_MATCHES: u8 = 8;

pub struct WFAAligner {
    aligner: NonNull<wfa::_wavefront_aligner_t>,
}

impl WFAAligner {
    pub fn new() -> Self {
        unsafe {
            let mut attributes = wfa::wavefront_aligner_attr_default;

            attributes.distance_metric = wfa::distance_metric_t_gap_affine_2p;
            attributes.affine2p_penalties.mismatch = MISMATCH_COST;
            attributes.affine2p_penalties.gap_opening1 = GAP_OPEN1_COST;
            attributes.affine2p_penalties.gap_extension1 = GAP_EXT1_COST;
            attributes.affine2p_penalties.gap_opening2 = GAP_OPEN2_COST;
            attributes.affine2p_penalties.gap_extension2 = GAP_EXT2_COST;

            attributes.alignment_scope = wfa::alignment_scope_t_compute_alignment;

            attributes.alignment_form.span = wfa::alignment_span_t_alignment_end2end;

            attributes.memory_mode = wfa::wavefront_memory_t_wavefront_memory_ultralow;

            attributes.heuristic.strategy = wfa::wf_heuristic_strategy_wf_heuristic_wfadaptive;
            attributes.heuristic.min_wavefront_length = MIN_WAVEFRONT_LENGTH;
            attributes.heuristic.max_distance_threshold = MAX_DISTANCE_THRESHOLD;
            attributes.heuristic.steps_between_cutoffs = STEPS_BETWEEN_CUTOFFS;

            let aligner = wfa::wavefront_aligner_new(&mut attributes);
            let aligner = NonNull::new(aligner).expect("Cannot create WFA aligner.");
            WFAAligner { aligner }
        }
    }

    pub fn align(&mut self, query: &[u8], target: &[u8]) -> Option<AlignmentResult> {
        unsafe {
            let q_seq: &[i8] = std::mem::transmute(query);
            let t_seq: &[i8] = std::mem::transmute(target);

            let status = wfa::wavefront_align(
                self.aligner.as_ptr(),
                t_seq.as_ptr(),
                target.len() as i32,
                q_seq.as_ptr(),
                query.len() as i32,
            );

            if status == 0 {
                let cigar_start = (*(*self.aligner.as_ptr()).cigar)
                    .operations
                    .offset((*(*self.aligner.as_ptr()).cigar).begin_offset as isize);
                let size = (*(*self.aligner.as_ptr()).cigar).end_offset
                    - (*(*self.aligner.as_ptr()).cigar).begin_offset;

                let cigar = std::slice::from_raw_parts(cigar_start as *mut u8, size as usize);

                // Alignment successful
                get_alignment_results(cigar)
            } else {
                None // Unsuccessful
            }
        }
    }
}

fn get_alignment_results(cigar: &[u8]) -> Option<AlignmentResult> {
    let (mut tstart, mut qstart) = (0, 0);
    let mut i = 0;
    let mut cm = 0; // consecutive matches

    loop {
        match cigar.get(i).map(|c| *c as char) {
            Some('M') => {
                cm += 1;
                if cm == MIN_END_MATCHES {
                    break;
                }
            }
            Some('X') => {
                tstart += cm as u32;
                qstart += cm as u32;
                cm = 0;

                tstart += 1;
                qstart += 1;
            }
            Some('D') => {
                tstart += cm as u32;
                qstart += cm as u32;
                cm = 0;

                tstart += 1;
            }
            Some('I') => {
                tstart += cm as u32;
                qstart += cm as u32;
                cm = 0;

                qstart += 1;
            }
            Some(_) => panic!("Invalid cigar op"),
            None => {
                return None;
            }
        }

        i += 1;
    }

    i -= MIN_END_MATCHES as usize - 1;

    let (mut tend, mut qend) = (0, 0);
    let mut j = cigar.len() - 1;
    cm = 0;

    loop {
        match cigar.get(j).map(|c| *c as char) {
            Some('M') => {
                cm += 1;
                if cm == MIN_END_MATCHES {
                    break;
                }
            }
            Some('X') => {
                tstart += cm as u32;
                qstart += cm as u32;
                cm = 0;

                tend += 1;
                qend += 1;
            }
            Some('D') => {
                tstart += cm as u32;
                qstart += cm as u32;
                cm = 0;

                tend += 1;
            }
            Some('I') => {
                tstart += cm as u32;
                qstart += cm as u32;
                cm = 0;

                qend += 1;
            }
            Some(_) => panic!("Invalid cigar op"),
            None => {
                return None;
            }
        }

        j -= 1;
    }
    j += MIN_END_MATCHES as usize; // Move to exclusive

    let cigar = cigar_merge_ops(&cigar[i..j]);
    Some(AlignmentResult::new(cigar, tstart, tend, qstart, qend))
}

impl Drop for WFAAligner {
    fn drop(&mut self) {
        unsafe { wfa::wavefront_aligner_delete(self.aligner.as_ptr()) }
    }
}

/// SAFETY: It is safe to send WFAAligner to another thread since it doesn't share the ownership
/// of aligner
unsafe impl Send for WFAAligner {}

fn cigar_merge_ops(cigar: &[u8]) -> Vec<CigarOp> {
    cigar
        .iter()
        .map(|c| (1u32, *c as char))
        .coalesce(|(l1, c1), (_, c2)| {
            if c1 == c2 {
                Ok((l1 + 1, c1))
            } else {
                Err(((l1, c1), (1, c2)))
            }
        })
        .map_into::<CigarOp>()
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::aligners::{cigar_to_string, wfa::WFAAligner};

    #[test]
    fn test_aligner() {
        let target = "AAAACTCTTTCCTGA";
        let query = "AAAAACCTTCTGA";

        let mut aligner = WFAAligner::new();
        let cigar = aligner.align(query.as_bytes(), target.as_bytes()).unwrap();

        println!("{}", cigar_to_string(&cigar.cigar));
    }

    #[test]
    fn test_aligner2() {
        let target = "TCAGAAGGTTTTT";
        let query = "TCAGGAAAGAGTTTT";

        let mut aligner = WFAAligner::new();
        let cigar = aligner.align(query.as_bytes(), target.as_bytes()).unwrap();

        println!("{}", cigar_to_string(&cigar.cigar));
    }
}

/*
#[cfg(test)]
mod tests {
    use crate::aligners::{
        wfa::{WFAAlignerBuilder, WFAAlignmentScope, WFAAlignmentSpan, WFADistanceMetric},
        CigarOp,
    };

    #[test]
    fn test_aligner() {
        let query = "AACGTTAGAT";
        let target = "TTAGAT";

        let aligner = WFAAlignerBuilder::new()
            .set_distance_metric(WFADistanceMetric::GapAffine {
                match_: 0,
                mismatch: 6,
                gap_opening: 4,
                gap_extension: 2,
            })
            .build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(cigar, [CigarOp::Insertion(4), CigarOp::Match(6)]);
    }

    #[test]
    fn test_aligner_indel() {
        let query = "AACTAAGTGTCGGTGGCTACTATATATCAGGTCCT";
        let target = "AGCTAGTGTCAATGGCTACTTTTCAGGTCCT";

        let aligner = WFAAlignerBuilder::new()
            .set_distance_metric(WFADistanceMetric::Indel)
            .build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(
            cigar,
            [
                CigarOp::Match(1),
                CigarOp::Insertion(1),
                CigarOp::Deletion(1),
                CigarOp::Match(3),
                CigarOp::Insertion(1),
                CigarOp::Match(5),
                CigarOp::Insertion(2),
                CigarOp::Deletion(2),
                CigarOp::Match(8),
                CigarOp::Insertion(1),
                CigarOp::Match(1),
                CigarOp::Insertion(1),
                CigarOp::Match(1),
                CigarOp::Insertion(1),
                CigarOp::Match(9)
            ]
        );
    }

    #[test]
    fn test_aligner_edit() {
        let query = "AACTAAGTGTCGGTGGCTACTATATATCAGGTCCT";
        let target = "AGCTAGTGTCAATGGCTACTTTTCAGGTCCT";

        let aligner = WFAAlignerBuilder::new()
            .set_distance_metric(WFADistanceMetric::Edit)
            .build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(
            cigar,
            [
                CigarOp::Match(1),
                CigarOp::Mismatch(1),
                CigarOp::Match(3),
                CigarOp::Insertion(1),
                CigarOp::Match(5),
                CigarOp::Mismatch(2),
                CigarOp::Match(8),
                CigarOp::Insertion(1),
                CigarOp::Match(1),
                CigarOp::Insertion(1),
                CigarOp::Match(1),
                CigarOp::Insertion(1),
                CigarOp::Match(9)
            ]
        );
    }

    #[test]
    fn test_aligner_gap_linear() {
        let query = "AACTAAGTGTCGGTGGCTACTATATATCAGGTCCT";
        let target = "AGCTAGTGTCAATGGCTACTTTTCAGGTCCT";

        let aligner = WFAAlignerBuilder::new()
            .set_distance_metric(WFADistanceMetric::GapLinear {
                match_: 0,
                mismatch: 6,
                indel: 2,
            })
            .build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(
            cigar,
            [
                CigarOp::Match(1),
                CigarOp::Insertion(1),
                CigarOp::Deletion(1),
                CigarOp::Match(3),
                CigarOp::Insertion(1),
                CigarOp::Match(5),
                CigarOp::Insertion(2),
                CigarOp::Deletion(2),
                CigarOp::Match(8),
                CigarOp::Insertion(1),
                CigarOp::Match(1),
                CigarOp::Insertion(1),
                CigarOp::Match(1),
                CigarOp::Insertion(1),
                CigarOp::Match(9)
            ]
        );
    }

    #[test]
    fn test_aligner_gap_affine() {
        let query = "AACTAAGTGTCGGTGGCTACTATATATCAGGTCCT";
        let target = "AGCTAGTGTCAATGGCTACTTTTCAGGTCCT";

        let aligner = WFAAlignerBuilder::new()
            .set_distance_metric(WFADistanceMetric::GapAffine {
                match_: (0),
                mismatch: (6),
                gap_opening: (4),
                gap_extension: (2),
            })
            .build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(
            cigar,
            [
                CigarOp::Match(1),
                CigarOp::Mismatch(1),
                CigarOp::Match(3),
                CigarOp::Insertion(1),
                CigarOp::Match(5),
                CigarOp::Mismatch(2),
                CigarOp::Match(8),
                CigarOp::Insertion(3),
                CigarOp::Match(1),
                CigarOp::Mismatch(1),
                CigarOp::Match(9)
            ]
        );
    }

    #[test]
    fn test_aligner_gap_affine2p() {
        let query = "AACTAAGTGTCGGTGGCTACTATATATCAGGTCCT";
        let target = "AGCTAGTGTCAATGGCTACTTTTCAGGTCCT";

        let aligner = WFAAlignerBuilder::new()
            .set_distance_metric(WFADistanceMetric::GapAffine2p {
                match_: (0),
                mismatch: (6),
                gap_opening1: (4),
                gap_extension1: (2),
                gap_opening2: (12),
                gap_extension2: (1),
            })
            .build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(
            cigar,
            [
                CigarOp::Match(1),
                CigarOp::Mismatch(1),
                CigarOp::Match(3),
                CigarOp::Insertion(1),
                CigarOp::Match(5),
                CigarOp::Mismatch(2),
                CigarOp::Match(8),
                CigarOp::Insertion(3),
                CigarOp::Match(1),
                CigarOp::Mismatch(1),
                CigarOp::Match(9)
            ]
        );
    }

    #[test]
    fn test_aligner_ends_free() {
        let query = "AATTTAAGTCTAGGCTACTTTCGGTACTTTCTT";
        let target = "AATTAATTTAAGTCTAGGCTACTTTCGGTACTTTGTTCTT";

        let aligner = WFAAlignerBuilder::new()
            .set_distance_metric(WFADistanceMetric::Indel)
            .set_alignment_span(WFAAlignmentSpan::EndsFree {
                pattern_begin_free: (10),
                pattern_end_free: (10),
                text_begin_free: (10),
                text_end_free: (10),
            })
            .build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(
            cigar,
            [
                CigarOp::Deletion(4),
                CigarOp::Match(30),
                CigarOp::Insertion(1),
                CigarOp::Deletion(1),
                CigarOp::Match(2),
                CigarOp::Deletion(3)
            ]
        );
    }

    #[test]
    fn test_aligner2() {
        let query = "AACGTTAGAT";
        let target = "TTAGTTGAT";

        let aligner = WFAAlignerBuilder::new()
            .set_distance_metric(WFADistanceMetric::GapAffine {
                match_: 0,
                mismatch: 6,
                gap_opening: 4,
                gap_extension: 2,
            })
            .build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(
            cigar,
            [
                CigarOp::Insertion(4),
                CigarOp::Match(4),
                CigarOp::Deletion(3),
                CigarOp::Match(2),
            ]
        );
    }

    #[test]
    fn test_aligner_mis_ends() {
        let query = "AATTAGATTCACACCCTTTTTTTTT";
        let target = "GGGGATCCCGGGG";

        let aligner = WFAAlignerBuilder::new()
            .set_distance_metric(WFADistanceMetric::GapAffine {
                match_: 0,
                mismatch: 6,
                gap_opening: 4,
                gap_extension: 2,
            })
            .build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(
            cigar,
            [
                CigarOp::Insertion(5),
                CigarOp::Match(1),
                CigarOp::Deletion(3),
                CigarOp::Match(2),
                CigarOp::Insertion(5),
                CigarOp::Match(3),
                CigarOp::Insertion(9),
                CigarOp::Deletion(4)
            ]
        );
    }
}*/
