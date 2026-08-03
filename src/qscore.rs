// Q-score estimation for corrected-read output.
//
// Three cases produce a Phred quality byte for each output base:
//   1. Insufficient support -> caller copies the read's own qual byte (see `clamp_qual_byte`).
//   2. Model-corrected base -> `model_qscore` from the base-head logits.
//   3. Majority vote -> `vote_qscore` from a naive-Bayes posterior over per-row vote weights.
//
// Case 3's math: treat each aligned row's vote as evidence with error probability e_i (derived
// from its qual, or Q_GAP for a gap), giving log-likelihood-ratio weight
// w_i = ln(4(1-e_i)/e_i) for the base it voted for. Summing w_i per candidate base and taking
// softmax is exactly the naive-Bayes posterior under a uniform prior; LAMBDA is a temperature
// compensating for the fact that aligned rows are not actually independent evidence.

pub(crate) const LAMBDA: f32 = 0.25;
pub(crate) const EPS_SYS: f32 = 1e-4;
pub(crate) const Q_GAP: u8 = 10;
pub(crate) const Q_MIN: u8 = 1;
pub(crate) const Q_MAX: u8 = 40;
pub(crate) const MODEL_TEMPERATURE: f32 = 1.0;

const QUAL_ASCII_OFFSET: i32 = 33; // Phred+33 encoding used throughout this crate

/// exp() overflows f32 for inputs > ~88; clamp each term below that so a sum of such terms
/// stays finite (and the resulting 1-p correctly saturates to ~1.0 instead of becoming NaN
/// from an inf/inf division).
const MAX_EXP_TERM: f32 = 1e30;

/// Precomputed `LAMBDA * ln(4(1-e)/e)` per ASCII qual byte, with the underlying Phred value
/// clamped to `[Q_MIN, Q_MAX]` before the error probability is derived from it.
pub(crate) struct WeightTable {
    table: [f32; 256],
}

impl WeightTable {
    pub(crate) fn new() -> Self {
        let mut table = [0f32; 256];
        for (byte, slot) in table.iter_mut().enumerate() {
            let q = (byte as i32 - QUAL_ASCII_OFFSET).clamp(Q_MIN as i32, Q_MAX as i32);
            let e = 10f32.powf(-(q as f32) / 10.0);
            let w = (4.0 * (1.0 - e) / e).ln();
            *slot = LAMBDA * w;
        }
        Self { table }
    }

    /// Weight contributed by a row with this raw ASCII (Phred+33) qual byte.
    pub(crate) fn weight_for_qual_byte(&self, qual_byte: u8) -> f32 {
        self.table[qual_byte as usize]
    }

    /// Weight contributed by a gap ('*'/'#') row, which has no real qual value attached.
    pub(crate) fn weight_for_gap(&self) -> f32 {
        self.table[(Q_GAP as i32 + QUAL_ASCII_OFFSET) as usize]
    }
}

/// Clamp a raw FASTQ qual byte's Phred value to `[Q_MIN, Q_MAX]`, for Case 1 (copied bases),
/// so its scale matches the bounded output of Case 2 / Case 3.
pub(crate) fn clamp_qual_byte(ascii_qual: u8) -> u8 {
    let q = (ascii_qual as i32 - QUAL_ASCII_OFFSET).clamp(Q_MIN as i32, Q_MAX as i32);
    (QUAL_ASCII_OFFSET + q) as u8
}

/// `P_err -> Phred ASCII byte`, `Q = -10*log10(P_err)`, clamped to `[Q_MIN, Q_MAX]`.
pub(crate) fn err_to_phred(p_err: f32) -> u8 {
    let p_err = p_err.clamp(f32::MIN_POSITIVE, 1.0);
    let q = (-10.0 * p_err.log10())
        .round()
        .clamp(Q_MIN as f32, Q_MAX as f32);
    (QUAL_ASCII_OFFSET + q as i32) as u8
}

/// Numerically stable `1 - P(called)` for a softmax over `scores`, without ever forming `p`
/// itself (which loses precision once p is very close to 1, the common case here).
fn stable_one_minus_p(scores: &[f32], called_idx: usize) -> f32 {
    let called_score = scores[called_idx];
    let r: f32 = scores
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != called_idx)
        .map(|(_, &s)| (s - called_score).exp().min(MAX_EXP_TERM))
        .sum();

    r / (1.0 + r)
}

/// Case 3: majority vote. `bin_scores` are the summed `LAMBDA*w_i` per candidate base
/// (including gap), `called_bin` is the index of the base actually being emitted (the
/// existing tie-break vote's call, not necessarily argmax of `bin_scores`).
pub(crate) fn vote_qscore(bin_scores: &[f32; 5], called_bin: usize) -> u8 {
    let one_minus_p = stable_one_minus_p(bin_scores, called_bin);
    let p_err = EPS_SYS + (1.0 - EPS_SYS) * one_minus_p;
    err_to_phred(p_err)
}

/// Case 2: model-corrected base. `logits` are the base head's per-class outputs (softmax
/// invariant to whether they're raw logits or log-probs), `called_idx` is the argmax already
/// computed by the caller.
pub(crate) fn model_qscore(logits: &[f32], called_idx: usize) -> u8 {
    let scaled: Vec<f32> = logits.iter().map(|&l| l / MODEL_TEMPERATURE).collect();
    let one_minus_p = stable_one_minus_p(&scaled, called_idx);
    let p_err = EPS_SYS + (1.0 - EPS_SYS) * one_minus_p;
    err_to_phred(p_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q_byte(q: u8) -> u8 {
        (QUAL_ASCII_OFFSET + q as i32) as u8
    }

    #[test]
    fn weight_table_matches_hand_computed_value_at_q20() {
        let table = WeightTable::new();
        let e = 10f32.powf(-20.0 / 10.0); // 0.01
        let expected = LAMBDA * (4.0 * (1.0 - e) / e).ln();
        assert!((table.weight_for_qual_byte(q_byte(20)) - expected).abs() < 1e-6);
    }

    #[test]
    fn weight_table_is_monotonic_increasing_in_q() {
        let table = WeightTable::new();
        let mut prev = table.weight_for_qual_byte(q_byte(Q_MIN));
        for q in (Q_MIN + 1)..=Q_MAX {
            let w = table.weight_for_qual_byte(q_byte(q));
            assert!(w > prev, "weight should strictly increase with q");
            prev = w;
        }
    }

    #[test]
    fn weight_table_clamps_below_q_min_and_above_q_max() {
        let table = WeightTable::new();
        let below = table.weight_for_qual_byte(q_byte(0));
        let at_min = table.weight_for_qual_byte(q_byte(Q_MIN));
        assert!((below - at_min).abs() < 1e-6);

        let above = table.weight_for_qual_byte(q_byte(90));
        let at_max = table.weight_for_qual_byte(q_byte(Q_MAX));
        assert!((above - at_max).abs() < 1e-6);
    }

    #[test]
    fn weight_for_gap_matches_q_gap_lookup() {
        let table = WeightTable::new();
        assert_eq!(
            table.weight_for_gap(),
            table.weight_for_qual_byte(q_byte(Q_GAP))
        );
    }

    #[test]
    fn vote_qscore_saturates_at_q_max_for_ten_agreeing_reads_at_q20() {
        let table = WeightTable::new();
        let w = table.weight_for_qual_byte(q_byte(20));
        let bin_scores = [10.0 * w, 0.0, 0.0, 0.0, 0.0]; // 10 reads all vote bin 0
        let q = vote_qscore(&bin_scores, 0);
        assert_eq!(
            q,
            q_byte(Q_MAX),
            "known saturation ceiling at these settings"
        );
    }

    #[test]
    fn vote_qscore_eight_two_split_at_q20_is_overconfident_by_design() {
        // Documents the known naive-Bayes overconfidence at a real 80/20 minor-allele split:
        // this is a property of the algorithm as specified, not a bug to "fix" here.
        let table = WeightTable::new();
        let w = table.weight_for_qual_byte(q_byte(20));
        let bin_scores = [8.0 * w, 2.0 * w, 0.0, 0.0, 0.0];
        let q = vote_qscore(&bin_scores, 0);
        let phred_q = q - QUAL_ASCII_OFFSET as u8;
        // p_err = eps_sys + (1-eps_sys)*(1-p) with p = softmax(8w, 2w, 0, 0, 0)[0], w = w(q=20)
        // -> p_err ~= 2.4600e-4 -> Q = -10*log10(p_err) ~= 36.09 -> rounds to 36.
        assert_eq!(phred_q, 36, "expected high-Q overconfidence at exactly Q36");
    }

    #[test]
    fn vote_qscore_single_read_reflects_its_own_quality_not_inflated() {
        let table = WeightTable::new();
        let w_low = table.weight_for_qual_byte(q_byte(5));
        let bin_scores = [w_low, 0.0, 0.0, 0.0, 0.0];
        let q = vote_qscore(&bin_scores, 0);
        let phred_q = q - QUAL_ASCII_OFFSET as u8;
        // p_err = eps_sys + (1-eps_sys)*(1-p) with p = softmax(w(q=5), 0, 0, 0, 0)[0]
        // -> p_err ~= 0.700061 -> Q = -10*log10(p_err) ~= 1.549 -> rounds to 2.
        assert_eq!(
            phred_q, 2,
            "single low-quality vote should reflect its own low quality"
        );
    }

    #[test]
    fn vote_qscore_uses_q_gap_for_gap_rows_not_a_real_qual() {
        let table = WeightTable::new();
        let w_gap = table.weight_for_gap();
        let w_q1 = table.weight_for_qual_byte(q_byte(Q_GAP)); // Q_GAP treated as a real qual
        assert_eq!(
            w_gap, w_q1,
            "gap weight must come from Q_GAP, matching a q=Q_GAP lookup"
        );
    }

    #[test]
    fn vote_qscore_all_zero_scores_does_not_panic_and_is_low_confidence() {
        // No observed evidence for any bin (e.g. a fully-unaligned column reaching this path
        // unexpectedly): must not panic, and must not report high confidence.
        let bin_scores = [0.0; 5];
        let q = vote_qscore(&bin_scores, 0);
        let phred_q = q - QUAL_ASCII_OFFSET as u8;
        // All scores tied at 0 -> uniform softmax over 5 bins -> p = 0.2, p_err ~= 0.80002
        // -> Q = -10*log10(p_err) ~= 0.969 -> rounds to 1.
        assert_eq!(
            phred_q, 1,
            "no evidence should yield the uniform-prior score, not a confident one"
        );
    }

    #[test]
    fn model_qscore_peaked_logits_near_q_max() {
        let logits = [10.0, -10.0, -10.0, -10.0, -10.0];
        let q = model_qscore(&logits, 0);
        assert_eq!(q, q_byte(Q_MAX));
    }

    #[test]
    fn model_qscore_uniform_logits_near_zero() {
        let logits = [0.0, 0.0, 0.0, 0.0, 0.0];
        let q = model_qscore(&logits, 0);
        let phred_q = q - QUAL_ASCII_OFFSET as u8;
        assert!(
            phred_q <= 2,
            "uniform logits over 5 classes should score near 0, got Q{}",
            phred_q
        );
    }

    #[test]
    fn model_qscore_called_idx_not_argmax_reflects_lower_probability() {
        let logits = [10.0, 0.0, -10.0, -10.0, -10.0];
        let q_argmax = model_qscore(&logits, 0);
        let q_other = model_qscore(&logits, 1);
        let phred_argmax = q_argmax - QUAL_ASCII_OFFSET as u8;
        let phred_other = q_other - QUAL_ASCII_OFFSET as u8;
        assert!(phred_other < phred_argmax);
    }

    #[test]
    fn err_to_phred_boundary_values() {
        // p_err = 1.0 -> Q = 0 exactly, clamped up to the Q_MIN floor
        assert_eq!(err_to_phred(1.0), q_byte(Q_MIN));
        // p_err = EPS_SYS floor -> Q = 40 exactly
        assert_eq!(err_to_phred(EPS_SYS), q_byte(Q_MAX));
    }

    #[test]
    fn clamp_qual_byte_clamps_high_and_low_and_passes_through_midrange() {
        assert_eq!(clamp_qual_byte(q_byte(93)), q_byte(Q_MAX));
        assert_eq!(clamp_qual_byte(q_byte(0)), q_byte(Q_MIN));
        assert_eq!(clamp_qual_byte(q_byte(20)), q_byte(20));
    }

    #[test]
    fn stable_one_minus_p_does_not_overflow_to_nan_for_extreme_scores() {
        // A score gap far beyond exp()'s f32 overflow threshold (~88) must still saturate to
        // a finite 1-p (~1.0), not NaN from an inf/inf division.
        let bin_scores = [0.0, 1000.0, 0.0, 0.0, 0.0];
        let q = vote_qscore(&bin_scores, 0);
        assert_eq!(
            q,
            q_byte(Q_MIN),
            "overwhelming contrary evidence should saturate to the Q_MIN floor, not NaN"
        );
    }

    #[test]
    fn gap_vote_contributes_q_gap_weight_to_gap_bin() {
        let table = WeightTable::new();

        let w20 = table.weight_for_qual_byte(q_byte(20));
        let w30 = table.weight_for_qual_byte(q_byte(30));
        let w10 = table.weight_for_qual_byte(q_byte(10));
        let w_gap = table.weight_for_gap();

        let bin_scores = [
            w20 + w30, // two A votes
            w10,       // one C vote
            0.0,       // no G votes
            0.0,       // no T votes
            w_gap,     // one gap vote
        ];

        assert_eq!(vote_qscore(&bin_scores, 0), q_byte(8));
    }
}
