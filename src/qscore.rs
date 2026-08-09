// Q-score assignment for corrected-read output.
//
// The corrected read carries the target read's own qual bytes, unchanged, regardless of whether
// a base was copied verbatim, replaced by the model, or replaced by the majority vote. The QUAL
// column therefore describes the original read's confidence at that position, not confidence in
// the emitted base.
//
// Insertions are the one case with no original qual to copy: the base exists in the corrected
// read but not in the target, so no qual byte was ever measured for it. Those get the median of
// the target's quals in a small window around the anchoring base (the target base the insertion
// follows). A median, being a rank statistic, is the same whether taken in Phred space or in
// error-probability space, which sidesteps having to pick between the two -- the means differ by
// ~15 Phred points on the same input.

/// Number of the target's own qual bytes the insertion median is taken over. Odd, so there is no
/// tie to break.
const INSERTION_WINDOW: usize = 5;

/// Quality for an inserted base, which has no counterpart in the target read: the median of the
/// target's quals over `INSERTION_WINDOW` positions centred on `anchor`, the target base the
/// insertion follows. The window is clipped at the read's ends; `anchor` is always in range, so
/// it never comes out empty.
///
/// Every base of a multi-base insertion shares one anchor and so receives the same value: the
/// estimator has nothing that distinguishes them. Note this makes a long insertion look exactly
/// as trustworthy as a single-base one, which is optimistic, since long insertions are enriched
/// for alignment artifacts.
pub(crate) fn insertion_qual(qual: &[u8], anchor: usize) -> u8 {
    debug_assert!(anchor < qual.len(), "insertion anchor outside the read");

    let half = INSERTION_WINDOW / 2;
    let start = anchor.saturating_sub(half);
    let end = (anchor + half + 1).min(qual.len());

    let mut window = [0u8; INSERTION_WINDOW];
    let n = end - start;
    window[..n].copy_from_slice(&qual[start..end]);
    window[..n].sort_unstable();

    window[n / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_a_full_interior_window() {
        //                 window over [1..=5] -> 50 20 40 10 30, median 30
        let qual = vec![99, 50, 20, 40, 10, 30, 99];
        assert_eq!(insertion_qual(&qual, 3), 30);
    }

    #[test]
    fn window_is_centred_on_the_anchor_not_offset() {
        // Only a window centred on index 4 sees both 10s and returns 10; a window shifted either
        // way picks up a 99 and returns something higher.
        let qual = vec![99, 99, 10, 10, 10, 99, 99];
        assert_eq!(insertion_qual(&qual, 4), 10);
    }

    #[test]
    fn window_clips_at_the_read_start() {
        // anchor 0 -> [0..=2] only, median of 30 10 20 is 20.
        let qual = vec![30, 10, 20, 99, 99];
        assert_eq!(insertion_qual(&qual, 0), 20);
    }

    #[test]
    fn window_clips_at_the_read_end() {
        // anchor 4 (last) -> [2..=4] only, median of 30 10 20 is 20.
        let qual = vec![99, 99, 30, 10, 20];
        assert_eq!(insertion_qual(&qual, 4), 20);
    }

    #[test]
    fn single_base_read_returns_that_base() {
        assert_eq!(insertion_qual(&[42], 0), 42);
    }

    #[test]
    fn even_sized_clipped_window_takes_the_upper_middle() {
        // anchor 0 on a 2-base read -> [0..=1], n = 2, index n/2 = 1 of the sorted pair.
        assert_eq!(insertion_qual(&[40, 20], 0), 40);
    }

    #[test]
    fn median_ignores_a_single_low_outlier() {
        // The robustness the median is chosen for: one Q2 neighbour among Q30s does not move it,
        // where a Phred-space mean would give ~Q24 and a probability-space mean ~Q9.
        let q30 = 33 + 30;
        let q2 = 33 + 2;
        assert_eq!(insertion_qual(&[q30, q30, q2, q30, q30], 2), q30);
    }
}
