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

pub struct WFAAligner {
    aligner: *mut wfa::_wavefront_aligner_t,
}

impl WFAAligner {
    pub fn new() -> Self {
        unsafe {
            let mut attributes = wfa::wavefront_aligner_attr_default;

            // Distance metrics - gap affine
            attributes.distance_metric = wfa::distance_metric_t_gap_affine;
            attributes.affine_penalties.mismatch = 6;
            attributes.affine_penalties.gap_opening = 4;
            attributes.affine_penalties.gap_extension = 2;

            // Alignment scope - alignment scope
            attributes.alignment_scope = wfa::alignment_scope_t_compute_alignment;

            // Alignment span - ends-free
            attributes.alignment_form.span = wfa::alignment_span_t_alignment_end2end;
            //attributes.alignment_form.pattern_begin_free = 5;
            //attributes.alignment_form.pattern_end_free = 5;
            //attributes.alignment_form.text_begin_free = 5;
            //attributes.alignment_form.text_end_free = 5;

            // Memory mode - high
            attributes.memory_mode = wfa::wavefront_memory_t_wavefront_memory_high;

            let aligner = wfa::wavefront_aligner_new(&mut attributes);
            wfa::wavefront_aligner_set_max_num_threads(aligner, 7);

            WFAAligner { aligner }
        }
    }

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
    use crate::aligners::CigarOp;

    use super::WFAAligner;

    #[test]
    fn test_aligner() {
        let query = "AACGTTAGAT";
        let target = "TTAGAT";

        let aligner = WFAAligner::new();
        let cigar = aligner.align(query, target).unwrap();

        assert_eq!(cigar, [CigarOp::INSERTION(4), CigarOp::MATCH(6)]);
    }

    #[test]
    fn test_aligner2() {
        let query = "AACGTTAGAT";
        let target = "TTAGTTGAT";

        let aligner = WFAAligner::new();
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

        let aligner = WFAAligner::new();
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
