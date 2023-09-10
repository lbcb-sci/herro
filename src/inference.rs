use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crossbeam_channel::{Receiver, Sender};
use lazy_static::lazy_static;
use ndarray::{s, Array2, Array3, Axis};
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use tch::{CModule, IValue, IndexOp, Tensor};

use crate::haec_io::HAECRecord;

const BATCH_SIZE: usize = 32;
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

pub(crate) struct InputData {
    rid: u32,
    windows: Vec<Features>,
}

impl InputData {
    fn new(rid: u32, windows: Vec<Features>) -> Self {
        Self { rid, windows }
    }
}

pub(crate) struct Features {
    wid: u16,
    bases: Array2<u8>,  // [L, R]
    quals: Array2<f32>, // [L, R]
    target_positions: Vec<i32>,
}

impl Features {
    fn new(wid: u16, bases: Array2<u8>, quals: Array2<f32>, target_positions: Vec<i32>) -> Self {
        Self {
            wid,
            bases,
            quals,
            target_positions,
        }
    }
}

pub(crate) struct OutputData {
    rid: u32,
    windows: Vec<Output>,
}

type Output = (u16, Array2<u8>, Vec<i32>, Vec<f32>); // wid, bases, logits

impl OutputData {
    fn new(rid: u32, windows: Vec<Output>) -> Self {
        Self { rid, windows }
    }
}

fn collate(batch: &[Features], device: tch::Device) -> Vec<IValue> {
    // Get longest sequence
    let length = batch.iter().map(|f| f.bases.len_of(Axis(0))).max().unwrap();
    let size = [
        batch.len() as i64,
        length as i64,
        batch[0].bases.len_of(Axis(1)) as i64,
    ]; // [B, L, R]

    let bases = Tensor::full(&size, BASE_PADDING as i64, (tch::Kind::Int, device));
    let quals = Tensor::ones(&size, (tch::Kind::Float, device));
    let mut tps = Vec::new();

    let mut lens = Vec::with_capacity(batch.len());
    for (idx, f) in batch.iter().enumerate() {
        if f.target_positions.len() == 0 {
            continue; // No suported positions
        }

        let l = f.bases.len_of(Axis(0));
        lens.push(l as i32);

        bases
            .i((idx as i64, ..l as i64, ..))
            .copy_(&Tensor::try_from(&f.bases).unwrap());
        quals
            .i((idx as i64, ..l as i64, ..))
            .copy_(&Tensor::try_from(&f.quals).unwrap());

        tps.push(Tensor::from_slice(&f.target_positions));
    }

    let inputs = vec![
        IValue::Tensor(bases),
        IValue::Tensor(quals),
        IValue::Tensor(Tensor::try_from(lens).unwrap()),
        IValue::TensorList(tps),
    ];

    inputs
}

fn inference(data: InputData, model: &CModule, device: tch::Device) -> OutputData {
    let mut logits = Vec::new();

    for i in (0..data.windows.len()).step_by(BATCH_SIZE) {
        let batch = &data.windows[i..(i + BATCH_SIZE).min(data.windows.len())];
        let inputs = collate(batch, device);

        let out = model
            .forward_is(&inputs)
            .map(|v| Tensor::try_from(v).unwrap().to(tch::Device::Cpu))
            .unwrap();

        // Get number of target positions for each window
        let lens: Vec<_> = match inputs[3] {
            IValue::TensorList(ref tps) => tps.iter().map(|t| t.size1().unwrap()).collect(),
            _ => unreachable!(),
        };

        let logits_batch: Vec<_> = out.split_with_sizes(&lens, 0);
        logits.extend(logits_batch);
    }

    let outputs: Vec<_> = data
        .windows
        .into_iter()
        .zip(logits.into_iter())
        .map(|(f, t)| (f.wid, f.bases, f.target_positions, t.try_into().unwrap()))
        .collect();

    OutputData::new(data.rid, outputs)
}

pub(crate) fn inference_worker<P: AsRef<Path>>(
    model_path: P,
    device: tch::Device,
    input_channel: Receiver<InputData>,
    output_channel: Sender<OutputData>,
) {
    let mut model = tch::CModule::load_on_device(model_path, device).expect("Cannot load model.");
    model.set_eval();

    tch::no_grad(|| loop {
        let data = match input_channel.recv() {
            Ok(data) => data,
            Err(_) => break,
        };

        let output = inference(data, &model, device);
        output_channel.send(output).unwrap();
    });
}

pub(crate) fn prepare_examples(
    rid: u32,
    features: impl IntoIterator<Item = (u16, Array3<u8>, Vec<u32>)>,
) -> InputData {
    let windows: Vec<_> = features
        .into_iter()
        .map(|(wid, ref mut feats, supported)| {
            // Transpose: [R, L, 2] -> [L, R, 2]
            feats.swap_axes(1, 0);

            // Transform bases (encode) and quals (normalize)
            let bases = feats.index_axis(Axis(2), 0).mapv(|b| BASES_MAP[b as usize]);
            let quals = feats
                .index_axis(Axis(2), 1)
                .mapv(|q| (f32::from(q) - QUAL_MIN_VAL) / (QUAL_MAX_VAL - QUAL_MIN_VAL));

            Features::new(
                wid,
                bases,
                quals,
                supported.into_iter().map(|i| i as i32).collect(),
            )
        })
        .collect();

    InputData::new(rid, windows)
}

fn consensus(
    data: OutputData,
    read: &HAECRecord,
    window_size: usize,
    buffer: &mut Vec<u8>,
) -> Vec<u8> {
    let windows: HashMap<_, (_, _)> = data
        .windows
        .into_iter()
        .map(|(w, b, _, l)| (w, (b, l)))
        .collect();
    let n_windows = ((read.seq.len() - 1) / window_size) + 1;

    read.seq.get_sequence(buffer);
    let uncorrected = &buffer[..read.seq.len()];
    let mut corrected: Vec<u8> = Vec::new();
    for wid in 0..n_windows {
        if let Some((bases, logits)) = windows.get(&(wid as u16)) {
            // Get target positions
            let target = bases.slice(s![.., 0]);
            let positions: Vec<_> = target
                .iter()
                .enumerate()
                .filter_map(|(idx, b)| {
                    if BASES_UPPER[*b as usize] != b'*' {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .chain(Some(bases.len_of(Axis(0))).into_iter())
                .collect();

            (0..positions.len() - 1).zip(logits).for_each(|(i, l)| {
                if *l <= 0. {
                    // Non-informative -> correct position
                    bases
                        .slice(s![positions[i]..positions[i + 1], ..])
                        .axis_iter(Axis(0))
                        .enumerate()
                        .for_each(|(pos, pos_bases)| {
                            let mut counts: HashMap<u8, u8> = HashMap::default();
                            pos_bases.iter().for_each(|b| {
                                if *b != BASES_MAP[b'.' as usize] {
                                    *counts.entry(BASES_UPPER[*b as usize]).or_insert(0) += 1;
                                }
                            });

                            let max_occ = counts.iter().map(|(_, c)| *c).max().unwrap();
                            let most_freq_bases: HashSet<_> = counts
                                .into_iter()
                                .filter_map(|(b, c)| if c == max_occ { Some(b) } else { None })
                                .collect();

                            let tgt_base = BASES_UPPER[target[positions[i] + pos] as usize];

                            if most_freq_bases.len() > 1 {
                                let base = if most_freq_bases.contains(&tgt_base) {
                                    tgt_base
                                } else {
                                    most_freq_bases.into_iter().min().unwrap()
                                };

                                if base != b'*' {
                                    corrected.push(base);
                                }

                                return;
                            }

                            /*let base =
                             *counts.iter().max_by_key(|&(_, count)| count).unwrap().0;*/
                            let base = most_freq_bases.into_iter().next().unwrap();

                            if base != b'*' {
                                corrected.push(base);
                            }
                        });
                } else {
                    // Keep the original base
                    corrected.push(uncorrected[wid * window_size + i]);
                }
            });
        } else {
            let start = wid * window_size;
            let end = (start + window_size).min(read.seq.len());
            corrected.extend(&uncorrected[start..end]);
        }
    }

    corrected
}

pub(crate) fn consensus_worker<P: AsRef<Path>>(
    output_path: P,
    reads: &[HAECRecord],
    receiver: Receiver<OutputData>,
    window_size: u32,
) {
    let file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(file);

    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();
    let mut buffer = vec![0; max_len];
    loop {
        let output = match receiver.recv() {
            Ok(output) => output,
            Err(_) => break,
        };

        let rid = output.rid as usize;

        output
            .windows
            .iter()
            .for_each(|(wid, _, supported, logits)| {
                supported
                    .iter()
                    .zip(logits.iter())
                    .for_each(|(pos, l)| println!("{}\t{}\t{}\t{}", &reads[rid].id, wid, pos, l));
            });

        /*let seq = consensus(output, &reads[rid], window_size as usize, &mut buffer);

        writeln!(&mut writer, ">{}", &reads[rid].id).unwrap();
        writer.write(&seq).unwrap();
        write!(&mut writer, "\n").unwrap();*/
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array3};
    use ndarray_npy::read_npy;

    use super::{inference, prepare_examples};

    #[test]
    fn test() {
        let _guard = tch::no_grad_guard();
        let device = tch::Device::Cpu;
        let resources = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Load model
        let mut model = tch::CModule::load_on_device(
            &resources.join("resources/patched_transformer-run2.pt"),
            device,
        )
        .unwrap();
        model.set_eval();

        // Get files list
        let mut files: Vec<_> = resources
            .join("resources/example_feats")
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
        let input_data = prepare_examples(0, features);

        let output = inference(input_data, &model, device);
        let predicted: Array1<f32> = output
            .windows
            .into_iter()
            .flat_map(|(_, _, l)| l.into_iter())
            .collect();

        let target: Array1<f32> =
            read_npy(resources.join("resources/example_feats_tch_out.npy")).unwrap();

        assert_abs_diff_eq!(predicted, target);
    }

    #[test]
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
    }
}
