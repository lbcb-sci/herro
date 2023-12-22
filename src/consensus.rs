use ndarray::Array2;

use std::{cmp::Reverse, collections::BinaryHeap};

use crossbeam_channel::{Receiver, Sender};
use itertools::Itertools;
use itertools::MinMaxResult::*;
use lazy_static::lazy_static;
use ndarray::{s, Axis};
use rustc_hash::FxHashMap as HashMap;

use crate::features::SupportedPos;
use crate::features::TOP_K;
use crate::haec_io::HAECRecord;
use crate::inference::BASES_MAP;

lazy_static! {
    static ref BASES_UPPER: [u8; 10] = {
        let mut map = [0; 10];
        map[0] = b'A';
        map[1] = b'C';
        map[2] = b'G';
        map[3] = b'T';
        map[4] = b'*';
        map[5] = b'A';
        map[6] = b'C';
        map[7] = b'G';
        map[8] = b'T';
        map[9] = b'*';
        map
    };
}

// Bases, tidx, supported, logits
pub(crate) struct ConsensusWindow {
    pub(crate) n_alns: usize,
    pub(crate) bases: Array2<u8>,
    pub(crate) quals: Array2<f32>,
    pub(crate) indices: Vec<usize>,
    pub(crate) supported: Vec<SupportedPos>,
    pub(crate) info_logits: Option<Vec<f32>>,
    pub(crate) bases_logits: Option<Vec<u8>>,
}

impl ConsensusWindow {
    pub(crate) fn new(
        n_alns: usize,
        bases: Array2<u8>,
        quals: Array2<f32>,
        indices: Vec<usize>,
        supported: Vec<SupportedPos>,
        info_logits: Option<Vec<f32>>,
        bases_logits: Option<Vec<u8>>,
    ) -> Self {
        Self {
            n_alns,
            bases,
            quals,
            indices,
            supported,
            info_logits,
            bases_logits,
        }
    }
}

pub(crate) struct ConsensusData {
    pub(crate) rid: u32,
    pub(crate) windows: Vec<ConsensusWindow>,
}

impl ConsensusData {
    pub(crate) fn new(rid: u32, windows: Vec<ConsensusWindow>) -> Self {
        Self { rid, windows }
    }
}

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
    read: &HAECRecord,
    window_size: usize,
    buffer: &mut Vec<u8>,
) -> Option<Vec<Vec<u8>>> {
    read.seq.get_sequence(buffer);
    let uncorrected = &buffer[..read.seq.len()];

    let mut corrected_seqs = Vec::new();
    let mut corrected: Vec<u8> = Vec::new();

    let minmax = data
        .windows
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

    for (wid, window) in (wid_st..wid_en).zip(data.windows[wid_st..wid_en].iter()) {
        /*if window.n_alns < 2 {
            let start = wid * window_size;
            let end = ((wid + 1) * window_size).min(uncorrected.len());

            corrected.extend(&uncorrected[start..end]);
            continue;
        }*/
        if window.n_alns < 2 {
            corrected_seqs.push(corrected);
            corrected = Vec::new();
            continue;
        }

        // Don't analyze empty rows: LxR -> LxN
        let n_rows = (window.n_alns + 1).min(TOP_K + 1);
        let bases = window.bases.slice(s![.., ..n_rows]);
        let maybe_info = match window.supported.len() {
            0 => HashMap::default(),
            _ => window
                .supported
                .iter()
                .zip(window.info_logits.as_ref().unwrap().iter())
                .zip(window.bases_logits.as_ref().unwrap().iter())
                .map(|((supp, il), bl)| (*supp, (*il, *bl)))
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
                let base = match *b {
                    0 => b'A',
                    1 => b'C',
                    2 => b'G',
                    3 => b'T',
                    4 => b'*',
                    _ => panic!("Unrecognized base"),
                };

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
                ); */
                if base != b'*' {
                    corrected.push(base);
                }
            } else {
                let most_common = two_most_frequent(col.iter().filter_map(|b| {
                    if *b != BASES_MAP[b'.' as usize] {
                        Some(BASES_UPPER[*b as usize])
                    } else {
                        None
                    }
                }));
                let tbase = BASES_UPPER[col[0] as usize];

                let base = if most_common.len() == 2
                    && most_common[0].0 == most_common[1].0
                    && (most_common[0].1 == tbase || most_common[1].1 == tbase)
                {
                    tbase
                } else {
                    most_common[0].1
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
                    corrected.push(base);
                }
            }
        }
    }

    corrected_seqs.push(corrected);
    Some(corrected_seqs)
}

pub(crate) fn consensus_worker(
    reads: &[HAECRecord],
    receiver: Receiver<ConsensusData>,
    sender: Sender<(usize, Vec<Vec<u8>>)>,
    window_size: u32,
) {
    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();
    let mut buffer = vec![0; max_len];
    loop {
        let output = match receiver.recv() {
            Ok(output) => output,
            Err(_) => break,
        };

        let rid = output.rid as usize;
        let seq = consensus(output, &reads[rid], window_size as usize, &mut buffer);

        if let Some(s) = seq {
            sender.send((rid, s)).unwrap();
        }
    }
}
