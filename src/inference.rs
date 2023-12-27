use std::path::Path;

use crossbeam_channel::{Receiver, Sender};
use itertools::Itertools;

use lazy_static::lazy_static;
use ndarray::{Array2, Axis};

use tch::{CModule, IValue, IndexOp, Tensor};

use crate::{
    consensus::{ConsensusData, ConsensusWindow},
    features::{get_target_indices, SupportedPos},
};

const BASE_PADDING: u8 = 11;
const QUAL_MIN_VAL: f32 = 33.;
const QUAL_MAX_VAL: f32 = 126.;

lazy_static! {
    pub(crate) static ref BASES_MAP: [u8; 128] = {
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

        for p in 0..l {
            for r in 0..f.bases.len_of(Axis(1)) {
                let _ = bases
                    .i((idx as i64, p as i64, r as i64))
                    .fill_(f.bases[[p, r]] as i64);

                let _ = quals
                    .i((idx as i64, p as i64, r as i64))
                    .fill_(f.quals[[p, r]] as f64);
            }
        }

        //println!("Bases shape: {:?}", f.bases.shape());
        //println!("Quals shape: {:?}", f.quals.shape());

        /*bases
            .i((idx as i64, ..l as i64, ..))
            .copy_(&Tensor::try_from(&f.bases).unwrap());
        quals
            .i((idx as i64, ..l as i64, ..))
            .copy_(&Tensor::try_from(&f.quals).unwrap());*/

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
    let d = match device {
        tch::Device::Cuda(d) => d,
        _ => unreachable!(),
    };

    let _no_grad = tch::no_grad_guard();

    let mut model = tch::CModule::load_on_device(model_path, device).expect("Cannot load model.");
    model.set_eval();

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

        /*println!(
            "Device {}, in: {}, out: {}",
            d,
            input_channel.len(),
            output_channel.len()
        );*/

        output_channel.send(data.consensus_data).unwrap();
    }
}

pub(crate) fn prepare_examples(
    rid: u32,
    features: impl IntoIterator<Item = (usize, (Array2<u8>, Array2<f32>), Vec<SupportedPos>)>,
    batch_size: usize,
) -> InferenceData {
    let windows: Vec<_> = features
        .into_iter()
        .map(|(n_alns, (mut bases, mut quals), supported)| {
            // Transform bases (encode) and quals (normalize)
            bases.mapv_inplace(|b| BASES_MAP[b as usize]);
            quals.mapv_inplace(|q| {
                2. * (f32::from(q) - QUAL_MIN_VAL) / (QUAL_MAX_VAL - QUAL_MIN_VAL) - 1.
            });

            // Transpose: [R, L] -> [L, R]
            //bases.swap_axes(1, 0);
            //quals.swap_axes(1, 0);

            let tidx = get_target_indices(&bases);

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
