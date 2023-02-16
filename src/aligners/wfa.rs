use std::str::from_utf8;

use itertools::Itertools;

use super::{AlignmentResult, CigarOp};

#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unused)]
mod wfa {
    include!(concat!(env!("OUT_DIR"), "/bindings_wfa.rs"));
}

pub struct WFAAlignerBuilder {
    attributes: wfa::wavefront_aligner_attr_t,
}

impl WFAAlignerBuilder {
    pub fn new() -> Self {
        unsafe {
            WFAAlignerBuilder {
                attributes: wfa::wavefront_aligner_attr_default,
            }
        }
    }

    pub fn set_distance_metric(&mut self, metric: WFADistanceMetric) -> &mut Self {
        match metric {
            WFADistanceMetric::Indel => {
                self.attributes.distance_metric = wfa::distance_metric_t_indel;
            }
            WFADistanceMetric::Edit => {
                self.attributes.distance_metric = wfa::distance_metric_t_edit;
            }
            WFADistanceMetric::GapLinear {
                match_,
                mismatch,
                indel,
            } => {
                self.attributes.distance_metric = wfa::distance_metric_t_gap_linear;
                self.attributes.linear_penalties.match_ = match_;
                self.attributes.linear_penalties.mismatch = mismatch;
                self.attributes.linear_penalties.indel = indel;
            }
            WFADistanceMetric::GapAffine {
                match_,
                mismatch,
                gap_opening,
                gap_extension,
            } => {
                self.attributes.distance_metric = wfa::distance_metric_t_gap_affine;
                self.attributes.affine_penalties.match_ = match_;
                self.attributes.affine_penalties.mismatch = mismatch;
                self.attributes.affine_penalties.gap_opening = gap_opening;
                self.attributes.affine_penalties.gap_extension = gap_extension;
            }
            WFADistanceMetric::GapAffine2p {
                match_,
                mismatch,
                gap_opening1,
                gap_extension1,
                gap_opening2,
                gap_extension2,
            } => {
                self.attributes.distance_metric = wfa::distance_metric_t_gap_affine_2p;
                self.attributes.affine2p_penalties.match_ = match_;
                self.attributes.affine2p_penalties.mismatch = mismatch;
                self.attributes.affine2p_penalties.gap_opening1 = gap_opening1;
                self.attributes.affine2p_penalties.gap_extension1 = gap_extension1;
                self.attributes.affine2p_penalties.gap_opening2 = gap_opening2;
                self.attributes.affine2p_penalties.gap_extension2 = gap_extension2;
            }
        }
        self
    }

    pub fn set_alignment_scope(&mut self, scope: WFAAlignmentScope) -> &mut Self {
        self.attributes.alignment_scope = scope as u32;
        self
    }

    pub fn set_alignment_span(&mut self, span: WFAAlignmentSpan) -> &mut Self {
        match span {
            WFAAlignmentSpan::EndToEnd => {
                self.attributes.alignment_form.span = wfa::alignment_span_t_alignment_end2end;
            }
            WFAAlignmentSpan::EndsFree {
                pattern_begin_free,
                pattern_end_free,
                text_begin_free,
                text_end_free,
            } => {
                self.attributes.alignment_form.span = wfa::alignment_span_t_alignment_endsfree;
                self.attributes.alignment_form.pattern_begin_free = pattern_begin_free;
                self.attributes.alignment_form.pattern_end_free = pattern_end_free;
                self.attributes.alignment_form.text_begin_free = text_begin_free;
                self.attributes.alignment_form.text_end_free = text_end_free;
            }
        }
        self
    }

    pub fn set_memory_mode(&mut self, mode: WFAMemoryMode) -> &Self {
        self.attributes.memory_mode = mode as u32;
        self
    }

    pub fn build(mut self) -> WFAAligner {
        unsafe {
            self.attributes.heuristic.strategy = wfa::wf_heuristic_strategy_wf_heuristic_none;

            let aligner = wfa::wavefront_aligner_new(&mut self.attributes); // TODO Handle null possibility
            WFAAligner { aligner }
        }
    }
}

pub enum WFADistanceMetric {
    Indel,
    Edit,
    GapLinear {
        match_: i32,
        mismatch: i32,
        indel: i32,
    },
    GapAffine {
        match_: i32,
        mismatch: i32,
        gap_opening: i32,
        gap_extension: i32,
    },
    GapAffine2p {
        match_: i32,
        mismatch: i32,
        gap_opening1: i32,
        gap_extension1: i32,
        gap_opening2: i32,
        gap_extension2: i32,
    },
}

#[repr(u32)]
pub enum WFAAlignmentScope {
    Score = wfa::alignment_scope_t_compute_score,
    Alignment = wfa::alignment_scope_t_compute_alignment,
}

pub enum WFAAlignmentSpan {
    EndToEnd,
    EndsFree {
        pattern_begin_free: i32,
        pattern_end_free: i32,
        text_begin_free: i32,
        text_end_free: i32,
    },
}

#[repr(u32)]
pub enum WFAMemoryMode {
    HIGH = wfa::wavefront_memory_t_wavefront_memory_high,
    MED = wfa::wavefront_memory_t_wavefront_memory_med,
    LOW = wfa::wavefront_memory_t_wavefront_memory_low,
    ULTRALOW = wfa::wavefront_memory_t_wavefront_memory_ultralow,
}
pub struct WFAAligner {
    aligner: *mut wfa::_wavefront_aligner_t,
}

impl WFAAligner {
    pub fn default() -> Self {
        let mut builder = WFAAlignerBuilder::new();

        //BiWFA
        builder.set_memory_mode(WFAMemoryMode::ULTRALOW);

        // Similar to minimap2 defaults
        /*builder.set_distance_metric(WFADistanceMetric::GapAffine {
            match_: 0,
            mismatch: 5,
            gap_opening: 4,
            gap_extension: 2,
        });*/

        builder.set_distance_metric(WFADistanceMetric::GapAffine2p {
            match_: 0,
            mismatch: 5,
            gap_opening1: 6,
            gap_extension1: 2,
            gap_opening2: 12,
            gap_extension2: 1,
        });

        builder.build()
    }

    pub fn align(&self, query: &[u8], target: &[u8]) -> Option<AlignmentResult> {
        unsafe {
            let q_seq: &[i8] = std::mem::transmute(query);
            let t_seq: &[i8] = std::mem::transmute(target);

            let status = wfa::wavefront_align(
                self.aligner,
                t_seq.as_ptr(),
                target.len() as i32,
                q_seq.as_ptr(),
                query.len() as i32,
            );

            if status == 0 {
                let cigar_start = (*(*self.aligner).cigar)
                    .operations
                    .offset((*(*self.aligner).cigar).begin_offset as isize);
                let size =
                    (*(*self.aligner).cigar).end_offset - (*(*self.aligner).cigar).begin_offset;

                let cigar = std::slice::from_raw_parts(cigar_start as *mut u8, size as usize);

                let (mut tstart, mut qstart) = (0, 0);
                let mut i = 0;
                loop {
                    match cigar[i] as char {
                        'M' => break,
                        'X' => {
                            tstart += 1;
                            qstart += 1;
                        }
                        'D' => {
                            tstart += 1;
                        }
                        'I' => {
                            qstart += 1;
                        }
                        _ => panic!("Invalid cigar op"),
                    }

                    i += 1;
                }

                let (mut tend, mut qend) = (0, 0);
                let mut j = cigar.len() - 1;
                loop {
                    match cigar[j] as char {
                        'M' => break,
                        'X' => {
                            tend += 1;
                            qend += 1;
                        }
                        'D' => {
                            tend += 1;
                        }
                        'I' => {
                            qend += 1;
                        }
                        _ => panic!("Invalid cigar op"),
                    }

                    j -= 1;
                }
                j += 1; // Move to exclusive

                // Alignment successful
                let cigar = cigar_merge_ops(&cigar[i..j]);
                Some(AlignmentResult::new(cigar, tstart, tend, qstart, qend))
                //Some(AlignmentResult::new(cigar, 0, 0, 0, 0))
            } else {
                None // Unsuccessful
            }
        }
    }
}

impl Drop for WFAAligner {
    fn drop(&mut self) {
        unsafe { wfa::wavefront_aligner_delete(self.aligner) }
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

        let aligner = WFAAligner::default();
        let cigar = aligner.align(query.as_bytes(), target.as_bytes()).unwrap();

        println!("{}", cigar_to_string(&cigar.cigar));
    }

    #[test]
    fn test_aligner2() {
        let target = "TCAGAAGGTTTTT";
        let query = "TCAGGAAAGAGTTTT";

        let aligner = WFAAligner::default();
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
