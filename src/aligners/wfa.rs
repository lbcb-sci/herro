use std::collections::VecDeque;

use itertools::Itertools;

use super::CigarOp;

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

    pub fn set_distance_metric(&mut self, metric: WFADistanceMetric) {
        match metric {
            WFADistanceMetric::Indel => {
                self.attributes.distance_metric = wfa::distance_metric_t_indel;
            }
            WFADistanceMetric::Edit => {
                self.attributes.distance_metric = wfa::distance_metric_t_edit;
            }
            WFADistanceMetric::GapLinear => {
                self.attributes.distance_metric = wfa::distance_metric_t_gap_linear;
            }
            WFADistanceMetric::GapAffine {
                match_,
                mismatch,
                gap_opening,
                gap_extension,
            }
            | WFADistanceMetric::GapAffine2p {
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
        }
    }

    pub fn set_alignment_scope(&mut self, scope: WFAAlignmentScope) {
        self.attributes.alignment_scope = scope as u32;
    }

    pub fn set_alignment_span(&mut self, span: WFAAlignmentSpan) {
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
                self.attributes.alignment_form.span = wfa::alignment_span_t_alignment_end2end;
                self.attributes.alignment_form.pattern_begin_free = pattern_begin_free;
                self.attributes.alignment_form.pattern_end_free = pattern_end_free;
                self.attributes.alignment_form.text_begin_free = text_begin_free;
                self.attributes.alignment_form.text_end_free = text_end_free;
            }
        }
    }

    pub fn set_memory_mode(&mut self, mode: WFAMemoryMode) {
        self.attributes.memory_mode = mode as u32;
    }

    pub fn build(&mut self) -> WFAAligner {
        unsafe {
            let aligner = wfa::wavefront_aligner_new(&mut self.attributes);
            WFAAligner { aligner }
        }
    }
}

pub enum WFADistanceMetric {
    Indel,
    Edit,
    GapLinear,
    GapAffine {
        match_: i32,
        mismatch: i32,
        gap_opening: i32,
        gap_extension: i32,
    },
    GapAffine2p {
        match_: i32,
        mismatch: i32,
        gap_opening: i32,
        gap_extension: i32,
    },
}

#[repr(u32)]
pub enum WFAAlignmentScope {
    SCORE = wfa::alignment_scope_t_compute_score,
    ALIGNMENT = wfa::alignment_scope_t_compute_alignment,
}

#[repr(u32)]
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
    pub fn align(&self, query: &str, target: &str) -> Option<VecDeque<CigarOp>> {
        unsafe {
            let q_seq: &[i8] = std::mem::transmute(query.as_bytes());
            let t_seq: &[i8] = std::mem::transmute(target.as_bytes());

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

                let cigar = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    cigar_start as *mut u8,
                    size as usize,
                ));

                Some(cigar_merge_ops(cigar)) // Alignment successful
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

fn cigar_merge_ops(cigar: &str) -> VecDeque<CigarOp> {
    cigar
        .chars()
        .map(|c| (1u32, c))
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
    use crate::aligners::{wfa::WFAAlignerBuilder, CigarOp};

    #[test]
    fn test_aligner() {
        let query = "AACGTTAGAT";
        let target = "TTAGAT";

        let aligner = WFAAlignerBuilder::new().build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(cigar, [CigarOp::INSERTION(4), CigarOp::MATCH(6)]);
    }

    #[test]
    fn test_aligner2() {
        let query = "AACGTTAGAT";
        let target = "TTAGTTGAT";

        let aligner = WFAAlignerBuilder::new().build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(
            cigar,
            [
                CigarOp::INSERTION(4),
                CigarOp::MATCH(4),
                CigarOp::MISMATCH(1),
                CigarOp::MATCH(1),
                CigarOp::DELETION(3),
            ]
        );
    }

    #[test]
    fn test_aligner_mis_ends() {
        let query = "AATTAGATTCACACCCTTTTTTTTT";
        let target = "GGGGCCCGGGG";

        let aligner = WFAAlignerBuilder::new().build();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(
            cigar,
            [
                CigarOp::MISMATCH(4),
                CigarOp::MATCH(3),
                CigarOp::MISMATCH(4)
            ]
        );
    }
}
