use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crossbeam_channel::{Receiver, Sender};
use itertools::Itertools;
use itertools::MinMaxResult::*;
use lazy_static::lazy_static;
use ndarray::{s, Array2, Array3, Axis};
use rustc_hash::FxHashMap as HashMap;

use tch::{CModule, IValue, IndexOp, Tensor};

use crate::{
    features::{get_target_indices, SupportedPos, TOP_K},
    haec_io::HAECRecord,
};

const BASE_PADDING: u8 = 11;
const QUAL_MIN_VAL: f32 = 33.;
const QUAL_MAX_VAL: f32 = 126.;

lazy_static! {
    static ref BASES_MAP: [u8; 128] = {
        let mut map = [0; 128];
        map[b'A' as usize] = 0;
        map[b'C' as usize] = 1;
        map[b'G' as usize] = 2;
        map[b'T' as usize] = 3;
        map[b'*' as usize] = 4;
        map[b'a' as usize] = 5;
        map[b'c' as usize] = 6;
        map[b'g' as usize] = 7;
        map[b't' as usize] = 8;
        map[b'#' as usize] = 9;
        map[b'.' as usize] = 10;
        map
    };
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
    n_alns: usize,
    bases: Array2<u8>,
    quals: Array2<f32>,
    indices: Vec<usize>,
    supported: Vec<SupportedPos>,
    info_logits: Option<Vec<f32>>,
    bases_logits: Option<Vec<u8>>,
}

impl ConsensusWindow {
    fn new(
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
    rid: u32,
    windows: Vec<ConsensusWindow>,
}

impl ConsensusData {
    fn new(rid: u32, windows: Vec<ConsensusWindow>) -> Self {
        Self { rid, windows }
    }
}

pub(crate) struct InferenceBatch {
    wids: Vec<u32>,
    bases: Tensor,
    quals: Tensor,
    lens: Tensor,
    indices: Vec<Tensor>,
}

impl InferenceBatch {
    fn new(
        wids: Vec<u32>,
        bases: Tensor,
        quals: Tensor,
        lens: Tensor,
        indices: Vec<Tensor>,
    ) -> Self {
        Self {
            wids,
            bases,
            quals,
            lens,
            indices,
        }
    }
}

pub(crate) struct InferenceData {
    consensus_data: ConsensusData,
    batches: Vec<InferenceBatch>,
}

impl InferenceData {
    fn new(consensus_data: ConsensusData, batches: Vec<InferenceBatch>) -> Self {
        Self {
            consensus_data,
            batches,
        }
    }
}

fn collate<'a>(batch: &[(u32, &ConsensusWindow)]) -> InferenceBatch {
    // Get longest sequence
    let length = batch
        .iter()
        .map(|(_, f)| f.bases.len_of(Axis(0)))
        .max()
        .unwrap();
    let size = [
        batch.len() as i64,
        length as i64,
        batch[0].1.bases.len_of(Axis(1)) as i64,
    ]; // [B, L, R]

    let bases = Tensor::full(
        &size,
        BASE_PADDING as i64,
        (tch::Kind::Int, tch::Device::Cpu),
    );

    let quals = Tensor::ones(&size, (tch::Kind::Float, tch::Device::Cpu));

    let mut lens = Vec::with_capacity(batch.len());
    let mut indices = Vec::with_capacity(batch.len());
    let mut wids = Vec::with_capacity(batch.len());

    for (idx, (wid, f)) in batch.iter().enumerate() {
        wids.push(*wid);
        let l = f.bases.len_of(Axis(0));

        bases
            .i((idx as i64, ..l as i64, ..))
            .copy_(&Tensor::try_from(&f.bases).unwrap());
        quals
            .i((idx as i64, ..l as i64, ..))
            .copy_(&Tensor::try_from(&f.quals).unwrap());

        lens.push(f.supported.len() as i32);

        let tidx: Vec<_> = f
            .supported
            .iter()
            .map(|&sp| (f.indices[sp.pos as usize] + sp.ins as usize) as i32)
            .collect();
        indices.push(Tensor::try_from(tidx).unwrap());
    }

    InferenceBatch::new(wids, bases, quals, Tensor::try_from(lens).unwrap(), indices)
}

fn inference(
    batch: InferenceBatch,
    model: &CModule,
    device: tch::Device,
) -> (Vec<u32>, Vec<Tensor>, Vec<Tensor>) {
    let inputs = [
        IValue::Tensor(batch.bases.to(device)),
        IValue::Tensor(batch.quals.to(device)),
        IValue::Tensor(batch.lens),
        IValue::TensorList(batch.indices),
    ];

    let (info_logits, bases_logits) =
        <(Tensor, Tensor)>::try_from(model.forward_is(&inputs).unwrap()).unwrap();

    // Get number of target positions for each window
    let lens: Vec<i64> = match inputs[2] {
        IValue::Tensor(ref t) => Vec::try_from(t).unwrap(),
        _ => unreachable!(),
    };

    let info_logits = info_logits.to(tch::Device::Cpu).split_with_sizes(&lens, 0);
    let bases_logits = bases_logits
        .argmax(1, false)
        .to(tch::Device::Cpu)
        .split_with_sizes(&lens, 0);

    (batch.wids, info_logits, bases_logits)
}

pub(crate) fn inference_worker<P: AsRef<Path>>(
    model_path: P,
    device: tch::Device,
    input_channel: Receiver<InferenceData>,
    output_channel: Sender<ConsensusData>,
) {
    let mut model = tch::CModule::load_on_device(model_path, device).expect("Cannot load model.");
    model.set_eval();

    let _no_grad = tch::no_grad_guard();

    loop {
        let mut data = match input_channel.recv() {
            Ok(data) => data,
            Err(_) => break,
        };

        for batch in data.batches {
            let (wids, info_logits, bases_logits) = inference(batch, &model, device);
            wids.into_iter()
                .zip(info_logits.into_iter())
                .zip(bases_logits.into_iter())
                .for_each(|((wid, il), bl)| {
                    data.consensus_data.windows[wid as usize]
                        .info_logits
                        .replace(Vec::try_from(il).unwrap());

                    data.consensus_data.windows[wid as usize]
                        .bases_logits
                        .replace(Vec::try_from(bl).unwrap());
                });
        }

        output_channel.send(data.consensus_data).unwrap();
    }
}

pub(crate) fn prepare_examples(
    rid: u32,
    features: impl IntoIterator<Item = (usize, Array3<u8>, Vec<SupportedPos>)>,
    batch_size: usize,
) -> InferenceData {
    let windows: Vec<_> = features
        .into_iter()
        .map(|(n_alns, ref mut feats, supported)| {
            // Transpose: [R, L, 2] -> [L, R, 2]
            feats.swap_axes(1, 0);

            // Transform bases (encode) and quals (normalize)
            let bases = feats.index_axis(Axis(2), 0).mapv(|b| BASES_MAP[b as usize]);
            let quals = feats
                .index_axis(Axis(2), 1)
                .mapv(|q| 2. * (f32::from(q) - QUAL_MIN_VAL) / (QUAL_MAX_VAL - QUAL_MIN_VAL) - 1.);

            let tidx = get_target_indices(&feats.index_axis(Axis(2), 0));

            //TODO: Start here.
            ConsensusWindow::new(n_alns, bases, quals, tidx, supported, None, None)
        })
        .collect();

    let batches: Vec<_> = (0u32..)
        .zip(windows.iter())
        .filter(|(_, features)| features.supported.len() > 0)
        .chunks(batch_size)
        .into_iter()
        .map(|v| {
            let batch = v.collect::<Vec<_>>();
            collate(&batch)
        })
        .collect();

    let consensus_data = ConsensusData::new(rid, windows);
    InferenceData::new(consensus_data, batches)
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use approx::assert_relative_eq;
    use ndarray::{Array1, Array3};

    use super::{inference, prepare_examples};

    #[test]
    fn test() {
        let _guard = tch::no_grad_guard();
        let device = tch::Device::Cpu;
        let resources = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Load model
        let mut model =
            tch::CModule::load_on_device(&resources.join("resources/mm2-attn.pt"), device).unwrap();
        model.set_eval();

        // Get files list
        /*let mut files: Vec<_> = resources
            .join("resources/example_feats")
            .read_dir()
            .unwrap()
            .filter_map(|p| {
                let p = p.unwrap().path();
                p.
                match p.extension() {
                    Some(ext) if ext == "npy" => Some(p),
                    _ => None,
                }
            })
            .collect();
        files.sort();*/

        // Create input data
        let features: Vec<_> = (0..4)
            .into_iter()
            .map(|wid| {
                let feats: Array3<u8> = read_npy(format!(
                "/home/stanojevicd/projects/ont-haec-rs/resources/example_feats/{}.features.npy",
                wid
            ))
                .unwrap();
                let supported: Array1<u16> = read_npy(format!(
                "/home/stanojevicd/projects/ont-haec-rs/resources/example_feats/{}.supported.npy",
                wid
            ))
                .unwrap();
                (feats, supported.iter().map(|s| *s as usize).collect())
            })
            .collect();
        let mut input_data = prepare_examples(0, features);
        let batch = input_data.batches.remove(0);

        let output = inference(batch, &model, device);
        let predicted: Array1<f32> = output
            .1
            .into_iter()
            .flat_map(|l| Vec::try_from(l).unwrap().into_iter())
            .collect();

        let target: Array1<f32> =
            read_npy(resources.join("resources/example_feats_tch_out.npy")).unwrap();

        assert_relative_eq!(predicted, target, epsilon = 1e-5);
    }

    /*#[test]
    fn test2() {
        let _guard = tch::no_grad_guard();
        let device = tch::Device::Cpu;
        let resources = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Load model
        let mut model =
            tch::CModule::load_on_device(&resources.join("resources/model.pt"), device).unwrap();
        model.set_eval();

        // Get files list
        let mut files: Vec<_> =
            PathBuf::from("/scratch/users/astar/gis/stanojev/haec/chr19_py_inference/test_rs")
                .read_dir()
                .unwrap()
                .filter_map(|p| {
                    let p = p.unwrap().path();
                    match p.extension() {
                        Some(ext) if ext == "npy" => Some(p),
                        _ => None,
                    }
                })
                .collect();
        files.sort();

        // Create input data
        let mut features: Vec<_> = files
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let feats: Array3<u8> = read_npy(p).unwrap();
                (i as u16, feats)
            })
            .collect();
        let input_data = prepare_examples(0, &mut features);

        let output = inference(input_data, &model, device);
        let predicted: Array1<f32> = output
            .windows
            .into_iter()
            .flat_map(|(_, _, l)| l.into_iter())
            .collect();

        println!("{:?}", &predicted.to_vec()[4056 - 5..4056 + 5]);
    }*/
}
