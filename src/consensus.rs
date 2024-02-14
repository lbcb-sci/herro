use ndarray::Array2;
use ordered_float::OrderedFloat;

use std::{cmp::Reverse, collections::BinaryHeap};

use crossbeam_channel::{Receiver, Sender};
use itertools::Itertools;
use itertools::MinMaxResult::*;

use ndarray::{s, Axis};
use rustc_hash::FxHashMap as HashMap;

use crate::features::SupportedPos;

use crate::inference::BASES_MAP;
use crate::inference::QUAL_MAX_VAL;
use crate::inference::QUAL_MIN_VAL;

const BASES_UPPER: [u8; 10] = [b'A', b'C', b'G', b'T', b'*', b'A', b'C', b'G', b'T', b'*'];
const BASES_UPPER_COUNTER: [usize; 10] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];

// Bases, tidx, supported, logits
pub(crate) struct ConsensusWindow {
    pub(crate) rid: u32,
    pub(crate) wid: u16,
    pub(crate) n_alns: u8,
    pub(crate) n_total_wins: u16,
    pub(crate) bases: Array2<u8>,
    pub(crate) quals: Array2<f32>,
    pub(crate) indices: Vec<usize>,
    pub(crate) supported: Vec<SupportedPos>,
    pub(crate) info_logits: Option<Vec<f32>>,
    pub(crate) bases_logits: Option<Array2<f32>>,
}

impl ConsensusWindow {
    pub(crate) fn new(
        rid: u32,
        wid: u16,
        n_alns: u8,
        n_total_wins: u16,
        bases: Array2<u8>,
        quals: Array2<f32>,
        indices: Vec<usize>,
        supported: Vec<SupportedPos>,
        info_logits: Option<Vec<f32>>,
        bases_logits: Option<Array2<f32>>,
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
    scores: &mut [f32],
) -> Option<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut corrected_seqs = Vec::new();
    let mut corrected_bases: Vec<u8> = Vec::new();
    let mut corrected_quals: Vec<u8> = Vec::new();

    let minmax = data
        .iter()
        .enumerate()
        .filter_map(|(idx, win)| if win.n_alns > 1 { Some(idx) } else { None })
        .minmax();
    let (wid_st, wid_en) = match minmax {
        NoElements => {
            return None;
        }
        OneElement(wid) => (wid, wid + 1),
        MinMax(st, en) => (st, en + 1),
    };

    for window in data[wid_st..wid_en].iter() {
        /*if window.n_alns < 2 {
            let start = wid * window_size;
            let end = ((wid + 1) * window_size).min(uncorrected.len());

            corrected.extend(&uncorrected[start..end]);
            continue;
        }*/
        if window.n_alns < 2 {
            if corrected_bases.len() > 0 {
                corrected_seqs.push((corrected_bases, corrected_quals));

                corrected_bases = Vec::new();
                corrected_quals = Vec::new();
            }

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
                .zip(window.bases_logits.as_ref().unwrap().axis_iter(Axis(0)))
                .map(|((supp, il), bl)| (*supp, (*il, bl)))
                .collect(),
        };

        let (mut pos, mut ins) = (-1i32, 0);
        for (bases_col, quals_col) in bases.axis_iter(Axis(0)).zip(quals.axis_iter(Axis(0))) {
            if bases_col[0] == BASES_MAP[b'*' as usize] {
                ins += 1;
            } else {
                pos += 1;
                ins = 0;
            }

            if let Some((_, logits)) = maybe_info.get(&SupportedPos::new(pos as u16, ins)) {
                let idx = logits
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &l)| OrderedFloat(l))
                    .map(|(i, _)| i)
                    .unwrap();

                let base = match idx as u8 {
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
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    std::str::from_utf8(&read.id).unwrap(),
                    wid,
                    pos,
                    ins,
                    corrected_seqs.len(),
                    corrected.len(),
                    'S',
                    base
                );*/
                if base != b'*' {
                    corrected_bases.push(base);

                    let qual = (10. * (logits[idx].exp() + 1.).log10()).round() as u8;
                    corrected_quals.push(qual);
                }
            } else {
                // Count bases
                counts.iter_mut().for_each(|c| *c = 0);
                scores.iter_mut().for_each(|s| *s = -1.);

                bases_col.iter().zip(quals_col.iter()).for_each(|(&b, &q)| {
                    if b != BASES_MAP[b'.' as usize] {
                        counts[BASES_UPPER_COUNTER[b as usize]] += 1;

                        if b != BASES_MAP[b'*' as usize] && b != BASES_MAP[b'#' as usize] {
                            let v = &mut scores[BASES_UPPER_COUNTER[b as usize]];
                            *v = v.max(q);
                        }
                    }
                });

                // Get indices of the two most common bases
                let (idx1, idx2) = counts
                    .iter()
                    .enumerate()
                    .sorted_by_key(|(_, &c)| Reverse(c))
                    .take(2)
                    .map(|(i, _)| i)
                    .collect_tuple()
                    .unwrap();
                let tbase = BASES_UPPER[bases_col[0] as usize];

                let (base, qual) = if counts[idx1] < 2
                    || (counts[idx1] == counts[idx2]
                        && (BASES_UPPER[idx1] == tbase || BASES_UPPER[idx2] == tbase))
                {
                    let tqual = qual_transform_inv(quals_col[0]);
                    (tbase, tqual)
                } else if counts[idx1] == counts[idx2] && BASES_UPPER[idx1] == tbase {
                    let tqual = match idx1 {
                        4 => 0,
                        _ => qual_transform_inv(scores[idx1]),
                    };

                    (tbase, tqual)
                } else if counts[idx1] == counts[idx2] && BASES_UPPER[idx2] == tbase {
                    let tqual = match idx2 {
                        4 => 0,
                        _ => qual_transform_inv(scores[idx2]),
                    };

                    (tbase, tqual)
                } else {
                    let tqual = match idx1 {
                        4 => 0,
                        _ => qual_transform_inv(scores[idx1]),
                    };

                    (BASES_UPPER[idx1], tqual)
                };

                /*println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    std::str::from_utf8(&read.id).unwrap(),
                    wid,
                    pos,
                    ins,
                    corrected_seqs.len(),
                    corrected.len(),
                    "N",
                    base,
                );*/
                if base != b'*' {
                    corrected_bases.push(base);
                    corrected_quals.push(qual);
                }
            }
        }
    }

    if corrected_bases.len() > 0 {
        corrected_seqs.push((corrected_bases, corrected_quals));
    }

    Some(corrected_seqs)
}

pub(crate) fn consensus_worker(
    receiver: Receiver<ConsensusData>,
    sender: Sender<(usize, Vec<(Vec<u8>, Vec<u8>)>)>,
) {
    let mut consensus_data = HashMap::default();
    let mut counts = [0u8; 5];
    let mut scores = [0f32; 4];

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

                let seq = consensus(windows, &mut counts, &mut scores);

                if let Some(s) = seq {
                    sender.send((rid as usize, s)).unwrap();
                }
            }
        }

        //println!("Consensus device: {}, in {}", device, receiver.len());
    }
}

fn qual_transform_inv(x: f32) -> u8 {
    ((x + 1.) * (QUAL_MAX_VAL - QUAL_MIN_VAL) / 2.).round() as u8
}
