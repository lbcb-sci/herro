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
use crate::qscore::{clamp_qual_byte, model_qscore};

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

fn consensus(
    data: ConsensusData,
    counts: &mut [u8],
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
        for col in bases.axis_iter(Axis(0)) {
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
                // Count bases
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
                    corrected_quals.push(clamp_qual_byte(0)); // TODO(Step 4): real vote_qscore
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

                let (seq, quals) = consensus(windows, &mut counts, &reads[rid as usize], window_size);
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
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, window_size);

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
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, window_size);

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
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 1);

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
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 1);

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
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 2);

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
        let (corrected, quals) = consensus(vec![window], &mut counts, &read, 2);

        // Row 0 (unsupported) goes through Case 3's majority vote over all-'A' rows -> 'A'.
        // Row 1 (supported at (0,1)) is scored by the model, independent of its gap-coded col[0].
        assert_eq!(corrected, b"AC".to_vec());
        assert_eq!(corrected.len(), quals.len(), "bases and quals must stay in sync across cases");
        assert_eq!(quals[1], model_qscore(&logits_c, 1));
    }
}
