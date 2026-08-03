use ndarray::Array2;
use ordered_float::OrderedFloat;

use std::{cmp::Reverse, collections::BinaryHeap};

use crossbeam_channel::{Receiver, Sender};
use itertools::Itertools;

use ndarray::{s, Axis};
use rustc_hash::FxHashMap as HashMap;

use crate::features::SupportedPos;

use crate::haec_io::HAECRecord;
use crate::inference::BASES_MAP;
use crate::qscore::{clamp_qual_byte, model_qscore, vote_qscore, WeightTable};

const BASES_UPPER: [u8; 10] = [b'A', b'C', b'G', b'T', b'*', b'A', b'C', b'G', b'T', b'*'];
const BASES_UPPER_COUNTER: [usize; 10] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
const GAP_BIN: usize = 4;

// Bases, tidx, supported, logits
pub(crate) struct ConsensusWindow {
    pub(crate) rid: u32,
    pub(crate) wid: u16,
    pub(crate) n_alns: u8,
    pub(crate) n_total_wins: u16,
    pub(crate) bases: Array2<u8>,
    pub(crate) quals: Array2<u8>,
    pub(crate) indices: Vec<usize>,
    pub(crate) supported: Vec<SupportedPos>,
    pub(crate) info_logits: Option<Vec<f32>>,
    pub(crate) bases_logits: Option<Vec<Vec<f32>>>,
}

impl ConsensusWindow {
    pub(crate) fn new(
        rid: u32,
        wid: u16,
        n_alns: u8,
        n_total_wins: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        indices: Vec<usize>,
        supported: Vec<SupportedPos>,
        info_logits: Option<Vec<f32>>,
        bases_logits: Option<Vec<Vec<f32>>>,
    ) -> Self {
        Self {
            rid,
            wid,
            n_alns,
            n_total_wins,
            bases,
            quals,
            indices,
            supported,
            info_logits,
            bases_logits,
        }
    }
}

pub type ConsensusData = Vec<ConsensusWindow>;

#[allow(dead_code)]
fn two_most_frequent<'a, I>(elements: I) -> Vec<(usize, u8)>
where
    I: Iterator<Item = u8>,
{
    let mut map = HashMap::default();
    for x in elements {
        *map.entry(x).or_default() += 1;
    }

    let mut heap = BinaryHeap::with_capacity(3);
    for (x, count) in map.into_iter() {
        heap.push(Reverse((count, x)));
        if heap.len() > 2 {
            heap.pop();
        }
    }

    heap.into_sorted_vec().into_iter().map(|r| r.0).collect()
}

fn consensus(
    data: ConsensusData,
    counts: &mut [u8],
    weights: &WeightTable,
    read: &HAECRecord,
    window_size: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut corrected: Vec<u8> = Vec::new();
    let mut corrected_quals: Vec<u8> = Vec::new();

    for window in data.iter() {
        if window.n_alns < 2 {
            let start = window.wid as usize * window_size;
            let end = ((window.wid as usize + 1) * window_size).min(read.seq.len());
            let old_len = corrected.len();
            corrected.resize(old_len + (end - start), 0);
            read.seq.get_subseq(start..end, &mut corrected[old_len..]);

            corrected_quals.extend(read.qual[start..end].iter().map(|&q| clamp_qual_byte(q)));
            continue;
        }

        // Don't analyze empty rows: LxR -> LxN
        //let n_rows = (window.n_alns + 1).min(TOP_K + 1);
        let n_rows = window.n_alns + 1;
        let bases = window.bases.slice(s![.., ..n_rows as usize]);
        let quals = window.quals.slice(s![.., ..n_rows as usize]);
        let maybe_info = match window.supported.len() {
            0 => HashMap::default(),
            _ => window
                .supported
                .iter()
                .zip(window.info_logits.as_ref().unwrap().iter())
                .zip(window.bases_logits.as_ref().unwrap().iter())
                .map(|((supp, il), bl)| (*supp, (*il, bl)))
                .collect(),
        };

        let (mut pos, mut ins) = (-1i32, 0);
        for (col, qcol) in bases.axis_iter(Axis(0)).zip(quals.axis_iter(Axis(0))) {
            if col[0] == BASES_MAP[b'*' as usize] {
                ins += 1;
            } else {
                pos += 1;
                ins = 0;
            }

            if let Some((_, b)) = maybe_info.get(&SupportedPos::new(pos as u16, ins)) {
                let argmax = b
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &v)| OrderedFloat(v))
                    .unwrap()
                    .0 as u8;
                let base = match argmax {
                    0 => b'A',
                    1 => b'C',
                    2 => b'G',
                    3 => b'T',
                    4 => b'*',
                    _ => panic!("Unrecognized base"),
                };

                /*if *il > 0.0 {
                    println!(
                        "{}\t{}\t{}",
                        std::str::from_utf8(&read.id).unwrap(),
                        corrected_seqs.len(),
                        corrected.len(),
                    );
                }*/

                /*println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{}",
                    std::str::from_utf8(&read.id).unwrap(),
                    wid,
                    pos,
                    ins,
                    corrected_seqs.len(),
                    corrected.len(),
                    'S',
                    b,
                    base
                );*/

                if base != b'*' {
                    corrected.push(base);
                    corrected_quals.push(model_qscore(b, argmax as usize));
                }
            } else {
                // Count bases and accumulate each row's vote weight into its base's bin.
                // Rows with no alignment ('.') contribute neither a count nor a weight.
                counts.iter_mut().for_each(|c| *c = 0);
                let mut bin_scores = [0f32; 5];
                col.iter().zip(qcol.iter()).for_each(|(&b, &q)| {
                    if b != BASES_MAP[b'.' as usize] {
                        let bin = BASES_UPPER_COUNTER[b as usize];
                        counts[bin] += 1;

                        // Gaps carry no real quality (the quals array only holds a b'!'
                        // placeholder there), so they are weighted at Q_GAP instead.
                        bin_scores[bin] += if bin == GAP_BIN {
                            weights.weight_for_gap()
                        } else {
                            weights.weight_for_qual_byte(q)
                        };
                    }
                });

                // Get two most common bases and counts - (c, b)
                let (mc0, mc1) = counts
                    .iter()
                    .enumerate()
                    .sorted_by_key(|(_, c)| Reverse(*c))
                    .take(2)
                    .map(|(i, c)| (*c, BASES_UPPER[i]))
                    .collect_tuple()
                    .unwrap();
                let tbase = BASES_UPPER[col[0] as usize];

                let base = if mc0.0 < 2 || (mc0.0 == mc1.0 && (mc0.1 == tbase || mc1.1 == tbase)) {
                    tbase
                } else {
                    mc0.1
                };

                /*println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{}",
                    std::str::from_utf8(&read.id).unwrap(),
                    wid,
                    pos,
                    ins,
                    corrected_seqs.len(),
                    corrected.len(),
                    "N",
                    counts,
                    base,
                );*/

                if base != b'*' {
                    corrected.push(base);

                    // `base` is uppercase ACGT here, for which BASES_MAP is the inverse of
                    // BASES_UPPER and so yields the bin index directly.
                    let called_bin = BASES_MAP[base as usize] as usize;
                    corrected_quals.push(vote_qscore(&bin_scores, called_bin));
                }
            }
        }
    }

    (corrected, corrected_quals)
}

pub(crate) fn consensus_worker(
    reads: &[HAECRecord],
    receiver: Receiver<ConsensusData>,
    sender: Sender<(usize, Vec<u8>, Vec<u8>)>,
    window_size: usize,
) {
    let mut consensus_data = HashMap::default();
    let mut counts = [0u8; 5];
    let weights = WeightTable::new();
    loop {
        let output = match receiver.recv() {
            Ok(output) => output,
            Err(_) => break,
        };

        for cw in output {
            let rid = cw.rid;
            let n_total_wins = cw.n_total_wins;

            let entry = consensus_data.entry(cw.rid).or_insert_with(|| Vec::new());
            entry.push(cw);

            if entry.len() == (n_total_wins as usize) {
                let mut windows = consensus_data.remove(&rid).unwrap();
                windows.sort_by_key(|cw| cw.wid);

                let (seq, quals) = consensus(
                    windows,
                    &mut counts,
                    &weights,
                    &reads[rid as usize],
                    window_size,
                );
                sender.send((rid as usize, seq, quals)).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::haec_io::HAECSeq;
    use crate::qscore::clamp_qual_byte;
    use ndarray::Array2;

    const Q10: u8 = 33 + 10;
    const Q20: u8 = 33 + 20;
    const Q30: u8 = 33 + 30;
    const Q_PLACEHOLDER: u8 = b'!'; // what features.rs writes where no base aligned

    fn q_byte(q: u8) -> u8 {
        33 + q
    }

    /// Single-position window with no model support, so every test below takes the Case 3 path.
    fn vote_window(row_bases: Vec<u8>, row_quals: Vec<u8>) -> ConsensusWindow {
        let n_rows = row_bases.len();
        ConsensusWindow::new(
            0,
            0,
            (n_rows - 1) as u8, // n_rows == n_alns + 1
            1,
            Array2::from_shape_vec((1, n_rows), row_bases).unwrap(),
            Array2::from_shape_vec((1, n_rows), row_quals).unwrap(),
            Vec::new(),
            Vec::new(),
            None,
            None,
        )
    }

    fn make_read(seq: &str, qual: Vec<u8>) -> HAECRecord {
        HAECRecord {
            id: b"test".to_vec(),
            description: None,
            seq: HAECSeq::from(seq.as_bytes()),
            qual,
        }
    }

    #[test]
    fn case1_copies_uncorrected_bases_and_clamped_quals() {
        let window_size = 4;
        let seq = "ACGTACGT";
        // Chosen to exercise: below Q_MIN (33), passthrough (40, 60), above Q_MAX (126)
        let raw_quals = vec![33, 40, 60, 126, 50, 55, 33, 70];
        let read = make_read(seq, raw_quals.clone());

        let window = ConsensusWindow::new(
            0,
            0,
            1, // n_alns < 2 -> Case 1
            1,
            Array2::from_elem((1, 1), 0u8),
            Array2::from_elem((1, 1), 0u8),
            Vec::new(),
            Vec::new(),
            None,
            None,
        );

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, window_size);

        assert_eq!(corrected, seq.as_bytes()[..window_size].to_vec());

        let expected_quals: Vec<u8> = raw_quals[..window_size]
            .iter()
            .map(|&q| clamp_qual_byte(q))
            .collect();
        assert_eq!(quals, expected_quals);
        assert_eq!(corrected.len(), quals.len());
    }

    #[test]
    fn case1_last_window_truncates_to_read_length() {
        let window_size = 5;
        let seq = "ACGTACG"; // 7 bases: window 1 covers [5,7)
        let raw_quals = vec![40, 40, 40, 40, 40, 60, 70];
        let read = make_read(seq, raw_quals);

        let window = ConsensusWindow::new(
            0,
            1, // second window
            1,
            2,
            Array2::from_elem((1, 1), 0u8),
            Array2::from_elem((1, 1), 0u8),
            Vec::new(),
            Vec::new(),
            None,
            None,
        );

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, window_size);

        assert_eq!(corrected, b"CG".to_vec());
        assert_eq!(quals, vec![clamp_qual_byte(60), clamp_qual_byte(70)]);
    }

    #[test]
    fn case2_model_corrected_base_uses_model_qscore() {
        use crate::features::SupportedPos;
        use crate::qscore::model_qscore;

        let read = make_read("A", vec![40]);

        // One position, 3 pileup rows; only col[0] (not a gap) drives pos/ins tracking here.
        let bases = Array2::from_elem((1, 3), 0u8); // 0 = BASES_MAP['A'], i.e. not a gap
        let quals = Array2::from_elem((1, 3), 0u8);
        let logits = vec![10.0, -10.0, -10.0, -10.0, -10.0]; // argmax -> 'A' (index 0)

        let window = ConsensusWindow::new(
            0,
            0,
            2, // n_alns >= 2 -> skip Case 1
            1,
            bases,
            quals,
            Vec::new(),
            vec![SupportedPos::new(0, 0)],
            Some(vec![0.0]),
            Some(vec![logits.clone()]),
        );

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, 1);

        assert_eq!(corrected, b"A".to_vec());
        assert_eq!(quals, vec![model_qscore(&logits, 0)]);
    }

    #[test]
    fn case2_gap_called_base_emits_no_base_and_no_qual() {
        use crate::features::SupportedPos;

        let read = make_read("A", vec![40]);

        let bases = Array2::from_elem((1, 3), 0u8); // pos 0, not a gap column
        let quals = Array2::from_elem((1, 3), 0u8);
        let logits = vec![-10.0, -10.0, -10.0, -10.0, 10.0]; // argmax -> index 4 == '*'

        let window = ConsensusWindow::new(
            0,
            0,
            2,
            1,
            bases,
            quals,
            Vec::new(),
            vec![SupportedPos::new(0, 0)],
            Some(vec![0.0]),
            Some(vec![logits]),
        );

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, 1);

        assert!(corrected.is_empty(), "gap call must not emit a base");
        assert!(quals.is_empty(), "gap call must not emit a qual either");
    }

    #[test]
    fn case2_multiple_supported_positions_each_get_independent_qscore() {
        use crate::features::SupportedPos;
        use crate::qscore::model_qscore;

        let read = make_read("AG", vec![40, 40]);

        // Two consecutive real positions (pos 0 and pos 1, both ins 0).
        let bases = Array2::from_elem((2, 3), 0u8);
        let quals = Array2::from_elem((2, 3), 0u8);
        let logits_a = vec![10.0, -10.0, -10.0, -10.0, -10.0]; // confident 'A'
        let logits_g = vec![-5.0, -5.0, 3.0, -5.0, -5.0]; // less confident 'G'

        let window = ConsensusWindow::new(
            0,
            0,
            2,
            1,
            bases,
            quals,
            Vec::new(),
            vec![SupportedPos::new(0, 0), SupportedPos::new(1, 0)],
            Some(vec![0.0, 0.0]),
            Some(vec![logits_a.clone(), logits_g.clone()]),
        );

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, 2);

        assert_eq!(corrected, b"AG".to_vec());
        assert_eq!(
            quals,
            vec![model_qscore(&logits_a, 0), model_qscore(&logits_g, 2)]
        );
        // The two positions have different confidence, so they must not collapse to the same Q.
        assert_ne!(quals[0], quals[1]);
    }

    #[test]
    fn case2_supported_insertion_position_uses_correct_pos_ins_key() {
        use crate::features::SupportedPos;
        use crate::qscore::model_qscore;

        let read = make_read("AC", vec![40, 40]);

        // Row 0: real position (pos 0, ins 0), left unsupported -> falls through to Case 3.
        // Row 1: insertion column (col[0] == gap code) -> pos stays 0, ins becomes 1.
        let mut bases = Array2::from_elem((2, 3), 0u8);
        bases.row_mut(1).fill(4); // BASES_MAP['*'] == 4, marks an insertion column
        let quals = Array2::from_elem((2, 3), 0u8);
        let logits_c = vec![-5.0, 10.0, -5.0, -5.0, -5.0]; // confident 'C', at the insertion slot

        let window = ConsensusWindow::new(
            0,
            0,
            2,
            1,
            bases,
            quals,
            Vec::new(),
            vec![SupportedPos::new(0, 1)], // keyed on the insertion (pos=0, ins=1), not (0, 0)
            Some(vec![0.0]),
            Some(vec![logits_c.clone()]),
        );

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, 2);

        // Row 0 (unsupported) goes through Case 3's majority vote over all-'A' rows -> 'A'.
        // Row 1 (supported at (0,1)) is scored by the model, independent of its gap-coded col[0].
        assert_eq!(corrected, b"AC".to_vec());
        assert_eq!(corrected.len(), quals.len(), "bases and quals must stay in sync across cases");
        assert_eq!(quals[1], model_qscore(&logits_c, 1));
    }

    #[test]
    fn case3_eight_two_split_matches_pinned_vote_qscore() {
        // 8 rows vote 'A', 2 vote 'C', all at Q20 -> S = [8w20, 2w20, 0, 0, 0], called bin 0.
        // Same scores as qscore's vote_qscore_eight_two_split_at_q20 test, which pins Q36.
        let read = make_read("A", vec![q_byte(40)]);
        let window = vote_window(vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1], vec![Q20; 10]);

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, 1);

        assert_eq!(corrected, b"A".to_vec());
        assert_eq!(quals, vec![q_byte(36)]);
    }

    #[test]
    fn case3_equal_vote_counts_weighted_by_each_rows_own_quality() {
        // 5 rows vote 'A' at Q30, 5 vote 'C' at Q10: equal COUNTS (5-5), so this only
        // distinguishes the two bases via per-row quality. A implementation that hardcoded a
        // single quality (e.g. Q20) for every row -- rather than reading each row's own qual
        // byte -- would still pass every other Case 3 test here, since they all use one
        // uniform quality across every voting row.
        let read = make_read("A", vec![q_byte(40)]);
        let mut row_quals = vec![Q30; 5];
        row_quals.extend(vec![Q10; 5]);
        let window = vote_window(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1], row_quals);

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, 1);

        // 5-5 tie -> resolved to the target base ('A', col[0]).
        // S = [5*w(Q30), 5*w(Q10), 0, 0, 0] -> p_err ~= 0.0029609 -> Q ~= 25.29 -> rounds to 25.
        // A fixed-Q20 bug would instead score both bins identically (S = [5w20, 5w20, ...]),
        // giving Q3 instead of Q25 -- so this catches that class of bug.
        assert_eq!(corrected, b"A".to_vec());
        assert_eq!(quals, vec![q_byte(25)]);
    }

    #[test]
    fn case3_gap_rows_weighted_at_q_gap_not_the_placeholder_qual() {
        // 3 rows vote 'A' at Q20, 3 vote gap. The quals array holds only b'!' (Q0) for the gap
        // rows, so reading it instead of substituting Q_GAP would under-weight the gap bin and
        // inflate the score to Q14 rather than the correct Q8.
        let read = make_read("A", vec![q_byte(40)]);
        let window = vote_window(
            vec![0, 0, 0, 4, 4, 4],
            vec![Q20, Q20, Q20, Q_PLACEHOLDER, Q_PLACEHOLDER, Q_PLACEHOLDER],
        );

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, 1);

        // 3-3 tie with the target base among the winners -> target base 'A' is kept.
        assert_eq!(corrected, b"A".to_vec());
        assert_eq!(quals, vec![q_byte(8)]);
    }

    #[test]
    fn case3_unaligned_rows_contribute_no_weight() {
        // 8 rows vote 'A' at Q20; the 2 '.' rows carry a high qual byte but must be skipped
        // entirely (they are also outside BASES_UPPER_COUNTER's range, so including them
        // would panic rather than merely skew the score).
        let read = make_read("A", vec![q_byte(40)]);
        let window = vote_window(
            vec![0, 0, 0, 0, 0, 0, 0, 0, 10, 10],
            vec![Q20, Q20, Q20, Q20, Q20, Q20, Q20, Q20, b'~', b'~'],
        );

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, 1);

        // S = [8*w20, 0, 0, 0, 0] -> the unanimous-8 score, unaffected by the '.' rows.
        assert_eq!(corrected, b"A".to_vec());
        assert_eq!(quals, vec![q_byte(39)]);
    }

    #[test]
    fn case3_reverse_strand_rows_fold_into_forward_bins() {
        // 4 'A' + 4 'a' + 2 'C' must score identically to 8 'A' + 2 'C': lowercase
        // (reverse-strand) rows contribute their weight to the same bin as uppercase.
        let read = make_read("A", vec![q_byte(40)]);
        let mixed = vote_window(vec![0, 0, 0, 0, 5, 5, 5, 5, 1, 1], vec![Q20; 10]);

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![mixed], &mut counts, &weights, &read, 1);

        assert_eq!(corrected, b"A".to_vec());
        assert_eq!(quals, vec![q_byte(36)]);
    }

    #[test]
    fn case3_gap_called_emits_no_base_and_no_qual() {
        // 4 gap rows outvote the single 'A' target row -> the column is deleted, so neither a
        // base nor a qual byte may be emitted.
        let read = make_read("A", vec![q_byte(40)]);
        let window = vote_window(
            vec![0, 4, 4, 4, 4],
            vec![Q20, Q_PLACEHOLDER, Q_PLACEHOLDER, Q_PLACEHOLDER, Q_PLACEHOLDER],
        );

        let mut counts = [0u8; 5];
        let weights = WeightTable::new();
        let (corrected, quals) = consensus(vec![window], &mut counts, &weights, &read, 1);

        assert!(corrected.is_empty(), "gap call must not emit a base");
        assert!(quals.is_empty(), "gap call must not emit a qual either");
    }
}
