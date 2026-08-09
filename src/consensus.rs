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
use crate::qscore::insertion_qual;

const BASES_UPPER: [u8; 10] = [b'A', b'C', b'G', b'T', b'*', b'A', b'C', b'G', b'T', b'*'];
const BASES_UPPER_COUNTER: [usize; 10] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];

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

/// The qual byte the corrected read carries for one emitted base: the target's own, copied from
/// pileup row 0, or -- for an insertion, which the target has no base and hence no qual for --
/// the local median around `anchor`, the target position the insertion follows.
fn column_qual(read: &HAECRecord, is_insertion: bool, anchor: usize, tqual: u8) -> u8 {
    if is_insertion {
        insertion_qual(&read.qual, anchor)
    } else {
        tqual
    }
}

fn consensus(
    data: ConsensusData,
    counts: &mut [u8],
    read: &HAECRecord,
    window_size: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut corrected: Vec<u8> = Vec::new();
    let mut corrected_quals: Vec<u8> = Vec::new();

    for window in data.iter() {
        let win_start = window.wid as usize * window_size;

        if window.n_alns < 2 {
            let end = ((window.wid as usize + 1) * window_size).min(read.seq.len());
            let old_len = corrected.len();
            corrected.resize(old_len + (end - win_start), 0);
            read.seq
                .get_subseq(win_start..end, &mut corrected[old_len..]);

            corrected_quals.extend_from_slice(&read.qual[win_start..end]);
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
            // A column the target has no base in: whatever gets emitted here is an insertion,
            // with no qual of its own to copy.
            let is_insertion = col[0] == BASES_MAP[b'*' as usize];
            if is_insertion {
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
                    corrected_quals.push(column_qual(
                        read,
                        is_insertion,
                        win_start + pos as usize,
                        qcol[0],
                    ));
                }
            } else {
                // Count bases. Rows with no alignment ('.') do not contribute a count.
                counts.iter_mut().for_each(|c| *c = 0);
                col.iter().for_each(|&b| {
                    if b != BASES_MAP[b'.' as usize] {
                        counts[BASES_UPPER_COUNTER[b as usize]] += 1;
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
                    corrected_quals.push(column_qual(
                        read,
                        is_insertion,
                        win_start + pos as usize,
                        qcol[0],
                    ));
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

                let (seq, quals) =
                    consensus(windows, &mut counts, &reads[rid as usize], window_size);
                sender.send((rid as usize, seq, quals)).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::haec_io::HAECSeq;
    use ndarray::Array2;

    const Q20: u8 = 33 + 20;
    const Q30: u8 = 33 + 30;
    const Q_PLACEHOLDER: u8 = b'!'; // what features.rs writes where no base aligned
    const GAP: u8 = 4; // BASES_MAP['*'], a column the target read has no base in

    fn q_byte(q: u8) -> u8 {
        33 + q
    }

    /// Pileup arrays from one `(bases, quals)` pair per column, each holding one entry per row.
    /// Row 0 is the target read.
    fn pileup(cols: &[(&[u8], &[u8])]) -> (Array2<u8>, Array2<u8>) {
        let n_rows = cols[0].0.len();
        let bases: Vec<u8> = cols.iter().flat_map(|(b, _)| b.iter().copied()).collect();
        let quals: Vec<u8> = cols.iter().flat_map(|(_, q)| q.iter().copied()).collect();
        (
            Array2::from_shape_vec((cols.len(), n_rows), bases).unwrap(),
            Array2::from_shape_vec((cols.len(), n_rows), quals).unwrap(),
        )
    }

    /// Single-position window with no model support, so it takes the Case 3 path.
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
    fn case1_copies_uncorrected_bases_and_original_quals() {
        let window_size = 4;
        let seq = "ACGTACGT";
        // Includes the two bytes the old [Q_MIN, Q_MAX] clamp used to move (33 = Q0 and
        // 126 = Q93): they must now pass through untouched.
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
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, window_size);

        assert_eq!(corrected, seq.as_bytes()[..window_size].to_vec());
        assert_eq!(quals, raw_quals[..window_size].to_vec());
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
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, window_size);

        assert_eq!(corrected, b"CG".to_vec());
        assert_eq!(quals, vec![60, 70]);
    }

    #[test]
    fn case2_model_corrected_base_keeps_the_targets_original_qual() {
        // The model confidently replaces the target's 'A' with 'C'. The emitted qual is still
        // the target's own byte at that position -- the model's confidence does not enter it.
        let read = make_read("A", vec![q_byte(10)]);

        // One position, 3 pileup rows; only col[0] (not a gap) drives pos/ins tracking here.
        let (bases, quals) = pileup(&[(&[0, 0, 0], &[q_byte(10), Q30, Q30])]);
        let logits = vec![-10.0, 10.0, -10.0, -10.0, -10.0]; // argmax -> 'C' (index 1)

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
            Some(vec![logits]),
        );

        let mut counts = [0u8; 5];
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 1);

        assert_eq!(corrected, b"C".to_vec());
        assert_eq!(quals, vec![q_byte(10)]);
    }

    #[test]
    fn case2_gap_called_base_emits_no_base_and_no_qual() {
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
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 1);

        assert!(corrected.is_empty(), "gap call must not emit a base");
        assert!(quals.is_empty(), "gap call must not emit a qual either");
    }

    #[test]
    fn case2_each_position_takes_its_own_targets_qual_not_the_model_confidence() {
        let read = make_read("AG", vec![q_byte(10), q_byte(30)]);

        // Two consecutive real positions (pos 0 and pos 1, both ins 0).
        let (bases, quals) = pileup(&[
            (&[0, 0, 0], &[q_byte(10), Q30, Q30]),
            (&[2, 2, 2], &[q_byte(30), Q30, Q30]),
        ]);
        let logits_a = vec![10.0, -10.0, -10.0, -10.0, -10.0]; // confident 'A'
        let logits_g = vec![-5.0, -5.0, 3.0, -5.0, -5.0]; // much less confident 'G'

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
            Some(vec![logits_a, logits_g]),
        );

        let mut counts = [0u8; 5];
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 2);

        // The quals track the target read (Q10 then Q30) and run opposite to the model's
        // confidence (very high then low), which is exactly the point: they are unrelated.
        assert_eq!(corrected, b"AG".to_vec());
        assert_eq!(quals, vec![q_byte(10), q_byte(30)]);
    }

    #[test]
    fn case2_insertion_takes_the_local_median_of_the_targets_quals() {
        // The window covers only the read's first position; the rest of the read exists so the
        // insertion's median window has real neighbours to draw on.
        let read = make_read(
            "ACGTA",
            vec![q_byte(10), q_byte(30), q_byte(20), q_byte(40), q_byte(50)],
        );

        // Column 0: real position (pos 0, ins 0), left unsupported -> falls through to Case 3.
        // Column 1: insertion column (col[0] == gap code) -> pos stays 0, ins becomes 1.
        let (bases, quals) = pileup(&[
            (&[0, 0, 0], &[q_byte(10), Q30, Q30]),
            (&[GAP, 1, 1], &[Q_PLACEHOLDER, Q30, Q30]),
        ]);
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
            Some(vec![logits_c]),
        );

        let mut counts = [0u8; 5];
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 5);

        // Column 0 (unsupported) goes through Case 3's majority vote over all-'A' rows -> 'A',
        // carrying the target's own Q10. Column 1 is a model-called insertion anchored at read
        // position 0, so it takes the median of Phred [10, 30, 20] = Q20.
        assert_eq!(corrected, b"AC".to_vec());
        assert_eq!(quals, vec![q_byte(10), q_byte(20)]);
        assert_ne!(
            quals[1], Q_PLACEHOLDER,
            "must not emit row 0's gap placeholder"
        );
    }

    #[test]
    fn case3_vote_corrected_base_keeps_the_targets_original_qual() {
        // Four Q30 rows outvote the target's own 'A' -> 'C' is emitted, but carrying the
        // target's Q10, not the quality of the rows that outvoted it.
        let read = make_read("A", vec![q_byte(10)]);
        let window = vote_window(vec![0, 1, 1, 1, 1], vec![q_byte(10), Q30, Q30, Q30, Q30]);

        let mut counts = [0u8; 5];
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 1);

        assert_eq!(corrected, b"C".to_vec());
        assert_eq!(quals, vec![q_byte(10)]);
    }

    #[test]
    fn case3_unaligned_rows_are_not_counted() {
        // '.' rows must be skipped entirely: BASES_MAP['.'] is 10, past the end of the 5-bin
        // counts array, so counting one would panic rather than merely skew the call.
        let read = make_read("A", vec![q_byte(10)]);
        let window = vote_window(
            vec![0, 0, 10, 10, 10],
            vec![q_byte(10), Q20, b'~', b'~', b'~'],
        );

        let mut counts = [0u8; 5];
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 1);

        assert_eq!(corrected, b"A".to_vec());
        assert_eq!(quals, vec![q_byte(10)]);
    }

    #[test]
    fn case3_reverse_strand_rows_fold_into_forward_bins() {
        // Lowercase (reverse-strand) rows count towards the same base as uppercase: 2 'G' plus
        // 2 'g' is a 4-count that outvotes the target's 'A'.
        let read = make_read("A", vec![q_byte(10)]);
        let window = vote_window(vec![0, 2, 2, 7, 7], vec![q_byte(10), Q20, Q20, Q20, Q20]);

        let mut counts = [0u8; 5];
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 1);

        assert_eq!(corrected, b"G".to_vec());
        assert_eq!(quals, vec![q_byte(10)]);
    }

    #[test]
    fn case3_gap_called_emits_no_base_and_no_qual() {
        // 4 gap rows outvote the single 'A' target row -> the column is deleted, so neither a
        // base nor a qual byte may be emitted.
        let read = make_read("A", vec![q_byte(10)]);
        let window = vote_window(
            vec![0, GAP, GAP, GAP, GAP],
            vec![
                q_byte(10),
                Q_PLACEHOLDER,
                Q_PLACEHOLDER,
                Q_PLACEHOLDER,
                Q_PLACEHOLDER,
            ],
        );

        let mut counts = [0u8; 5];
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 1);

        assert!(corrected.is_empty(), "gap call must not emit a base");
        assert!(quals.is_empty(), "gap call must not emit a qual either");
    }

    // --- Insertions: the one case with no original qual to copy ---

    #[test]
    fn run_of_four_insertions_all_share_the_anchors_median() {
        // Four insertion columns following one target base. They share an anchor, so they all
        // get the same qual: the estimator has nothing that distinguishes them. That makes a
        // 4-base insertion look exactly as trustworthy as a 1-base one, which is optimistic and
        // is a known, accepted property rather than a bug.
        let phred = [10u8, 12, 35, 14, 16, 18, 20, 22];
        let read = make_read("ACGTACGT", phred.iter().map(|&q| q_byte(q)).collect());

        let (bases, quals) = pileup(&[
            (&[0, 0, 0], &[q_byte(10), Q30, Q30]),      // pos 0, 'A'
            (&[1, 1, 1], &[q_byte(12), Q30, Q30]),      // pos 1, 'C'
            (&[2, 2, 2], &[q_byte(35), Q30, Q30]),      // pos 2, 'G'
            (&[3, 3, 3], &[q_byte(14), Q30, Q30]),      // pos 3, 'T' <- the anchor
            (&[GAP, 1, 1], &[Q_PLACEHOLDER, Q30, Q30]), // ins 1
            (&[GAP, 1, 1], &[Q_PLACEHOLDER, Q30, Q30]), // ins 2
            (&[GAP, 1, 1], &[Q_PLACEHOLDER, Q30, Q30]), // ins 3
            (&[GAP, 1, 1], &[Q_PLACEHOLDER, Q30, Q30]), // ins 4
        ]);

        let window =
            ConsensusWindow::new(0, 0, 2, 1, bases, quals, Vec::new(), Vec::new(), None, None);

        let mut counts = [0u8; 5];
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 8);

        // Anchor is read position 3, so the window is Phred [12, 35, 14, 16, 18] -> median 16.
        // Note that differs from the anchor's own Q14, so this also rules out simply copying
        // the anchor's qual.
        assert_eq!(corrected, b"ACGTCCCC".to_vec());
        assert_eq!(
            quals,
            vec![
                q_byte(10),
                q_byte(12),
                q_byte(35),
                q_byte(14),
                q_byte(16),
                q_byte(16),
                q_byte(16),
                q_byte(16),
            ]
        );
        assert_eq!(corrected.len(), quals.len());
    }

    #[test]
    fn insertion_anchor_accounts_for_the_window_offset() {
        // wid 1 at window_size 4 -> the anchor is read position 4, not 0. The quals are shaped
        // so the two choices give different medians (Q35 vs Q1).
        let phred = [1u8, 1, 40, 40, 10, 12, 35, 14];
        let read = make_read("ACGTACGT", phred.iter().map(|&q| q_byte(q)).collect());

        let (bases, quals) = pileup(&[
            (&[0, 0, 0], &[q_byte(10), Q30, Q30]), // read position 4, 'A'
            (&[GAP, 1, 1], &[Q_PLACEHOLDER, Q30, Q30]), // insertion after it
        ]);

        let window = ConsensusWindow::new(
            0,
            1, // second window
            2,
            2,
            bases,
            quals,
            Vec::new(),
            Vec::new(),
            None,
            None,
        );

        let mut counts = [0u8; 5];
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 4);

        // Anchor 4 -> window Phred [40, 40, 10, 12, 35] -> median 35. Ignoring the window
        // offset would anchor at 0 and give [1, 1, 40] -> median 1.
        assert_eq!(corrected, b"AC".to_vec());
        assert_eq!(quals, vec![q_byte(10), q_byte(35)]);
    }
}
