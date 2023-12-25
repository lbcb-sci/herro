use npyz::WriterBuilder;
use rustc_hash::FxHashMap as HashMap;
use std::fs::{create_dir_all, File};
use std::io::prelude::*;
use std::io::{BufWriter, Result};
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use crossbeam_channel::Sender;

use lazy_static::lazy_static;
use ndarray::{s, Array, Array2, Array3, ArrayBase, ArrayViewMut1, ArrayViewMut2, Axis, Data, Ix2};
use ordered_float::OrderedFloat;

use crate::aligners::{fix_cigar, get_proper_cigar, CigarOp};
use crate::haec_io::HAECRecord;
use crate::inference::{prepare_examples, InferenceData};
use crate::overlaps::{self, Alignment, Strand};
use crate::windowing::{extract_windows, OverlapWindow};

pub(crate) const TOP_K: usize = 30;

lazy_static! {
    static ref BASE_LOWER: [u8; 128] = {
        let mut arr = [255; 128];
        arr[b'A' as usize] = b'a';
        arr[b'C' as usize] = b'c';
        arr[b'G' as usize] = b'g';
        arr[b'T' as usize] = b't';
        arr
    };
    static ref BASE_FORWARD: [u8; 128] = {
        let mut arr = [255; 128];
        arr[b'A' as usize] = b'A';
        arr[b'C' as usize] = b'C';
        arr[b'G' as usize] = b'G';
        arr[b'T' as usize] = b'T';
        arr[b'*' as usize] = b'*';
        arr[b'a' as usize] = b'A';
        arr[b'c' as usize] = b'C';
        arr[b'g' as usize] = b'G';
        arr[b't' as usize] = b'T';
        arr[b'#' as usize] = b'*';
        arr
    };
}

fn get_max_ins_for_window(
    overlaps: &[OverlapWindow], // Sorted overlaps
    ovlps_cigar_map: &HashMap<u32, Vec<CigarOp>>,
    tid: u32,
    tstart: usize,
    window_length: usize,
) -> Vec<u16> {
    let mut max_ins = vec![0; window_length];
    for ow in overlaps.iter().take(TOP_K) {
        let mut tpos = ow.tstart as usize - tstart;

        // Handle cigar
        let qid = ow.overlap.return_other_id(tid);
        let cigar = ovlps_cigar_map.get(&qid).unwrap()[ow.cigar_start_idx..].iter();
        let cigar_len = ow.cigar_end_idx - ow.cigar_start_idx + 1;

        cigar.take(cigar_len).enumerate().for_each(|(i, op)| {
            let l = match op {
                CigarOp::Match(l) | CigarOp::Mismatch(l) | CigarOp::Deletion(l) => *l as usize,
                CigarOp::Insertion(l) => {
                    max_ins[tpos - 1] = max_ins[tpos - 1].max(*l as u16);
                    return;
                }
            };

            if cigar_len == 1 {
                tpos += (ow.cigar_end_offset - ow.cigar_start_offset) as usize;
            } else if i == 0 {
                tpos += l - ow.cigar_start_offset as usize;
            } else if i == cigar_len - 1 {
                tpos += ow.cigar_end_offset as usize;
            } else {
                tpos += l;
            }
        });
    }

    max_ins
}

fn get_query_region(window: &OverlapWindow, tid: u32) -> (u32, u32) {
    let (qstart, qend) = if window.overlap.tid == tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    match window.overlap.strand {
        Strand::Forward => (qstart + window.qstart, qstart + window.qend),
        Strand::Reverse => (qend - window.qend, qend - window.qstart),
    }
}

fn get_features_for_ol_window(
    mut bases: ArrayViewMut1<'_, u8>,
    mut quals: ArrayViewMut1<'_, u8>,
    window: &OverlapWindow,
    cigar: &[CigarOp],
    query: &HAECRecord,
    offset: usize,
    tid: u32,
    max_ins: &[u16],
    qbuffer: &mut [u8],
) {
    // Handle query sequence
    let (qstart, qend) = if window.overlap.tid == tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    let mut query_iter: Box<dyn DoubleEndedIterator<Item = (&u8, &u8)>> =
        match window.overlap.strand {
            Strand::Forward => {
                let range = (qstart + window.qstart) as usize..(qstart + window.qend) as usize;
                let qlen = (window.qend - window.qstart) as usize;

                query.seq.get_subseq(range.clone(), qbuffer);
                let quals = &query.qual[range];

                Box::new(qbuffer[..qlen].iter().zip(quals))
            }
            Strand::Reverse => {
                let range = (qend - window.qend) as usize..(qend - window.qstart) as usize;
                let qlen = (window.qend - window.qstart) as usize;

                query.seq.get_rc_subseq(range.clone(), qbuffer);
                let quals = &query.qual[range];

                Box::new(
                    qbuffer[..qlen]
                        .iter()
                        .zip(quals.iter().rev())
                        .map(|(b, q)| (&BASE_LOWER[*b as usize], q)),
                )
            }
        };
    //let mut query_iter = query_iter.skip(window.qstart as usize);

    // Number of cigars for the window
    // TODO get error when we calculate correct number for end -> (idx, 0)
    // Works for this expression but unecessarily iterates through (idx, 0)
    let cigar_len = window.cigar_end_idx - window.cigar_start_idx + 1;
    let cigar_end = cigar.len().min((window.cigar_end_idx + 1) as usize);

    // Handle cigar
    let cigar = cigar[window.cigar_start_idx as usize..cigar_end].iter();

    // Get features
    let gap = if let Strand::Forward = window.overlap.strand {
        b'*'
    } else {
        b'#'
    };
    bases.fill(gap); // Initialize with gap token

    let mut tpos = offset; // position in the target read (excluding insertions)
    let mut idx = offset + max_ins[..offset].iter().map(|v| *v as usize).sum::<usize>(); // position in the features (including insertions)

    if idx > 0 {
        // No alignment at the start
        bases.slice_mut(s![..idx]).fill(b'.');
    }

    cigar
        .take(cigar_len)
        .enumerate()
        .for_each(|(cigar_idx, op)| {
            let mut l = match op {
                CigarOp::Match(l)
                | CigarOp::Mismatch(l)
                | CigarOp::Deletion(l)
                | CigarOp::Insertion(l) => *l as usize,
            };

            // Calculate true length
            if cigar_len == 1 {
                l = (window.cigar_end_offset - window.cigar_start_offset) as usize;
            } else if cigar_idx == 0 {
                l -= window.cigar_start_offset as usize;
            } else if cigar_idx == cigar_len - 1 {
                l = window.cigar_end_offset as usize;
            }

            // Write features
            match op {
                CigarOp::Match(_) | CigarOp::Mismatch(_) => {
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");
                        bases[idx] = *base;
                        quals[idx] = *qual;

                        idx += 1 + max_ins[tpos + i] as usize;
                    }

                    tpos += l;
                }
                CigarOp::Deletion(_) => {
                    for i in 0..l {
                        // No need to write gap, gap is already written
                        idx += 1 + max_ins[tpos + i] as usize;
                    }

                    tpos += l;
                }
                CigarOp::Insertion(_) => {
                    /*assert!(
                        max_ins[tpos - 1] as usize >= l,
                        "Insertion length is bigger than max_ins"
                    );*/

                    idx -= max_ins[tpos - 1] as usize; // Return to first insertion for the previous base
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");

                        bases[idx + i] = *base;
                        quals[idx + i] = *qual;
                    }
                    idx += max_ins[tpos - 1] as usize; // Move back to the last base
                }
            }
        });

    if idx < bases.shape()[0] {
        // No alignment at the end
        bases.slice_mut(s![idx..]).fill(b'.');
    }
}

fn write_target_for_window(
    tstart: usize,
    target: &HAECRecord,
    max_ins: &[u16],
    mut bases: ArrayViewMut1<'_, u8>,
    mut quals: ArrayViewMut1<'_, u8>,
    window_length: usize,
    tbuffer: &mut [u8],
) {
    bases.fill(b'*'); // Fill like forward

    let tlen = tstart + window_length - tstart;
    target
        .seq
        .get_subseq(tstart..tstart + window_length, tbuffer);

    let mut tpos = 0;
    tbuffer[..tlen]
        .iter()
        .zip(target.qual[tstart..tstart + window_length].iter())
        .enumerate()
        .for_each(|(i, (b, q))| {
            bases[tpos] = *b;
            quals[tpos] = *q;

            tpos += 1 + max_ins[i] as usize;
        });
}

fn get_features_for_window(
    overlaps: &mut [OverlapWindow],
    ovlps_cigar_map: &HashMap<u32, Vec<CigarOp>>,
    tid: u32,
    reads: &[HAECRecord],
    max_ins: &[u16],
    tstart: usize,
    window_length: usize, // Full window length
    tbuffer: &mut [u8],
    qbuffer: &mut [u8],
) -> (Array2<u8>, Array2<u8>) {
    //Get features
    let length = max_ins.iter().map(|v| *v as usize).sum::<usize>() + max_ins.len();

    let mut bases = Array::from_elem((1 + TOP_K, length), b'.');
    let mut quals = Array::from_elem((1 + TOP_K, length), b'!');

    // First write the target
    write_target_for_window(
        tstart,
        &reads[tid as usize],
        &max_ins,
        bases.index_axis_mut(Axis(0), 0),
        quals.index_axis_mut(Axis(0), 0),
        window_length,
        tbuffer,
    );

    // Write top-k overlaps for the window
    overlaps.iter().take(TOP_K).enumerate().for_each(|(i, ow)| {
        let qid = ow.overlap.return_other_id(tid);
        get_features_for_ol_window(
            bases.index_axis_mut(Axis(0), i + 1),
            quals.index_axis_mut(Axis(0), i + 1),
            ow,
            ovlps_cigar_map.get(&qid).unwrap(),
            &reads[qid as usize],
            ow.tstart as usize - tstart,
            tid,
            &max_ins,
            qbuffer,
        )
    });

    (bases, quals)
}

fn overlap_window_filter(cigar: &[CigarOp]) -> bool {
    let long_indel = cigar.iter().any(|op| match op {
        CigarOp::Insertion(l) | CigarOp::Deletion(l) if *l >= 30 => true,
        _ => false,
    });

    //accuracy >= 0.80 && !long_indel
    //calculate_accuracy(cigar) >= 0.85 &&
    !long_indel
}

pub(crate) fn extract_features<'a, T: FeaturesOutput<'a>>(
    rid: u32,
    reads: &'a [HAECRecord],
    overlaps: Vec<Alignment>,
    window_size: u32,
    (tbuf, qbuf): (&mut [u8], &mut [u8]),
    feats_output: &mut T,
) {
    let read = &reads[rid as usize];

    // Get overlaps for windows
    let n_windows = (read.seq.len() + window_size as usize - 1) / window_size as usize;
    let mut windows = vec![Vec::new(); n_windows];

    let mut ovlps_cigar_map = HashMap::default();
    for alignment in overlaps {
        let overlap = Rc::new(alignment.overlap.clone());
        let qid = overlap.return_other_id(rid);

        let mut cigar = get_proper_cigar(&alignment.cigar, overlap.tid == rid, overlap.strand);

        // TODO - get proper target and query
        let (tstart, tend, qstart, qend) = if overlap.tid == rid {
            (overlap.tstart, overlap.tend, overlap.qstart, overlap.qend)
        } else {
            (overlap.qstart, overlap.qend, overlap.tstart, overlap.tend)
        };

        let tlen = tend as usize - tstart as usize;
        reads[rid as usize]
            .seq
            .get_subseq(tstart as usize..tend as usize, tbuf);

        let qlen = qend as usize - qstart as usize;
        if overlaps::Strand::Forward == overlap.strand {
            reads[qid as usize]
                .seq
                .get_subseq(qstart as usize..qend as usize, qbuf);
        } else {
            reads[qid as usize]
                .seq
                .get_rc_subseq(qstart as usize..qend as usize, qbuf);
        }
        let (tshift, qshift) = fix_cigar(&mut cigar, &tbuf[..tlen], &qbuf[..qlen]);

        //Extract windows
        let is_target = overlap.tid == rid;
        extract_windows(
            &mut windows,
            overlap,
            &cigar,
            tshift,
            qshift,
            is_target,
            window_size,
        );

        ovlps_cigar_map.insert(qid, cigar);
    }

    // Create directory for the read
    //let output_path = Path::new("features").join(&read.id);
    //create_dir_all(&output_path).expect("Cannot create directory");

    feats_output.init(rid, &read.id);
    for i in 0..n_windows {
        /*if windows[i].len() == 0 {
            continue;
        }*/

        let win_len = if i == n_windows - 1 {
            read.seq.len() - i * window_size as usize
        } else {
            window_size as usize
        };

        // Filter windows
        windows[i].retain(|ow| {
            let qid = ow.overlap.return_other_id(rid);

            // TODO: Handle CIGAR offsets
            let cigar = ovlps_cigar_map.get(&qid).unwrap();
            let cigar_end = (ow.cigar_end_idx + 1).min(cigar.len());
            overlap_window_filter(&cigar[ow.cigar_start_idx..cigar_end])
        });

        // Sort window to take TOP-K
        windows[i].sort_by_key(|ow| {
            let cigar = ovlps_cigar_map
                .get(&ow.overlap.return_other_id(rid))
                .unwrap();

            let tstart = ow.tstart as usize;
            let tend = i * window_size as usize + win_len;
            let tlen = tend - tstart;
            reads[rid as usize].seq.get_subseq(tstart..tend, tbuf);

            let qid = ow.overlap.return_other_id(rid);
            let (qstart, qend) = get_query_region(ow, rid);
            let qlen = (qend - qstart) as usize;
            match ow.overlap.strand {
                Strand::Forward => reads[qid as usize]
                    .seq
                    .get_subseq(qstart as usize..qend as usize, qbuf),
                Strand::Reverse => reads[qid as usize]
                    .seq
                    .get_rc_subseq(qstart as usize..qend as usize, qbuf),
            }

            let acc = calculate_accuracy(ow, cigar, &tbuf[..tlen], &qbuf[..qlen]);
            OrderedFloat(-acc)
        });

        let max_ins = get_max_ins_for_window(
            &windows[i],
            &ovlps_cigar_map,
            rid,
            i * window_size as usize,
            win_len,
        );

        let (bases, quals) = get_features_for_window(
            &mut windows[i],
            &ovlps_cigar_map,
            rid,
            reads,
            &max_ins,
            i * window_size as usize,
            win_len,
            tbuf,
            qbuf,
        );

        let qids: Vec<&str> = windows[i]
            .iter()
            .map(|ow| {
                std::str::from_utf8(&reads[ow.overlap.return_other_id(rid) as usize].id).unwrap()
            })
            .collect();

        let supported = get_supported(&bases);

        feats_output.update(bases, quals, supported, qids, i as u16);
    }

    feats_output.emit();
}

fn calculate_accuracy(window: &OverlapWindow, cigar: &[CigarOp], tseq: &[u8], qseq: &[u8]) -> f32 {
    let (mut tpos, mut qpos) = (0, 0);
    let (mut m, mut s, mut i, mut d) = (0, 0, 0, 0);
    for idx in window.cigar_start_idx..=window.cigar_end_idx {
        let len = if window.cigar_start_idx == window.cigar_end_idx {
            (window.cigar_end_offset - window.cigar_start_offset) as usize
        } else if idx == window.cigar_start_idx {
            (cigar[idx].get_length() - window.cigar_start_offset) as usize
        } else if idx == window.cigar_end_idx {
            window.cigar_end_offset as usize
        } else {
            cigar[idx].get_length() as usize
        };

        if len == 0 {
            break;
        }

        match cigar[idx] {
            CigarOp::Match(_) => {
                for j in 0..len {
                    let tbase = tseq[tpos + j];
                    let qbase = qseq[qpos + j];

                    if tbase == qbase {
                        m += 1;
                    } else {
                        s += 1;
                    }
                }

                tpos += len;
                qpos += len;
            }
            CigarOp::Mismatch(_) => unreachable!(),
            CigarOp::Insertion(_) => {
                i += len;
                qpos += len;
            }
            CigarOp::Deletion(_) => {
                d += len;
                tpos += len;
            }
        }
    }

    (m as f32) / ((m + s + i + d) as f32)
}

pub(crate) fn get_target_indices<S: Data<Elem = u8>>(bases: &ArrayBase<S, Ix2>) -> Vec<usize> {
    bases
        .slice(s![.., 0])
        .iter()
        .enumerate()
        .filter_map(|(idx, b)| if *b != b'*' { Some(idx) } else { None })
        .collect()
}

fn get_supported<S>(bases: &ArrayBase<S, Ix2>) -> Vec<SupportedPos>
where
    S: Data<Elem = u8>,
{
    // bases -> [R, L]

    let mut counter: HashMap<u8, u8> = HashMap::default();
    counter.insert(b'A', 0);
    counter.insert(b'C', 0);
    counter.insert(b'G', 0);
    counter.insert(b'T', 0);
    counter.insert(b'*', 0);

    let mut supporeted = Vec::new();

    let (mut tpos, mut ins) = (-1i16, 0);
    for col in bases.axis_iter(Axis(1)) {
        if col[0] == b'*' {
            ins += 1;
        } else {
            tpos += 1;
            ins = 0;
        }

        counter.iter_mut().for_each(|(_, c)| *c = 0);
        col.iter().for_each(|&b| {
            if b == b'.' {
                return;
            }

            *counter.get_mut(&BASE_FORWARD[b as usize]).unwrap() += 1;
        });

        let n_supported = counter
            .iter()
            .fold(0u8, |acc, (_, &c)| if c >= 3 { acc + 1 } else { acc });
        if n_supported >= 2 {
            supporeted.push(SupportedPos::new(tpos as u16, ins));
        }
    }

    /*for l in 1..len + 1 {
        if l != len && bases[[0, l]] == b'*' {
            // Gap in target -> do not test
            continue;
        }

        let subseq = bases.slice(s![.., start..l]);
        counter.clear();
        for read_subseq in subseq.axis_iter(Axis(0)) {
            let mut hasher = FxHasher::default();
            let result = read_subseq.iter().try_for_each(|&v| {
                if v == b'.' {
                    return Err(()); // No alignment position present
                } else {
                    hasher.write_u8(BASE_FORWARD[v as usize]);
                    return Ok(());
                }
            });

            if result.is_ok() {
                // Check if alignment is really aligned
                let entry = counter.entry(hasher.finish()).or_insert(0);
                *entry += 1;
            }
        }

        let n_supported = counter
            .iter()
            .fold(0u8, |acc, (_, &c)| if c >= 3 { acc + 1 } else { acc });
        if n_supported >= 2 {
            supporeted.push(tpos);
        }

        start = l;
        tpos += 1;
    }*/

    supporeted
}

fn output_features<P: AsRef<Path>>(
    path: P,
    window_id: u16,
    ids: &[&str],
    features: &Array3<u8>,
    supported: impl IntoIterator<Item = SupportedPos>,
) -> Result<()> {
    let ids_path = path.as_ref().join(format!("{}.ids.txt", window_id));
    let ids_file = File::create(ids_path)?;
    let mut ids_writer = BufWriter::new(ids_file);
    for id in ids {
        writeln!(&mut ids_writer, "{}", id)?
    }

    let features_path = path.as_ref().join(format!("{}.features.npy", window_id));
    let shape: Vec<_> = features.shape().iter().map(|&s| s as u64).collect();
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(BufWriter::new(File::create(features_path)?))
        .begin_nd()?;
    writer.extend(features.iter())?;
    writer.finish()?;

    let supported_path = path.as_ref().join(format!("{}.supported.npy", window_id));
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .writer(BufWriter::new(File::create(supported_path)?))
        .begin_1d()?;
    writer.extend(supported)?;
    writer.finish()?;

    Ok(())
}

pub(crate) trait FeaturesOutput<'a> {
    fn init<'b>(&mut self, rid: u32, rname: &'b [u8])
    where
        'b: 'a;
    fn update(
        &mut self,
        bases: Array2<u8>,
        quals: Array2<u8>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        wid: u16,
    );
    fn emit(&mut self);
}

#[derive(Clone)]
pub(crate) struct FeatsGenOutput<'a, T>
where
    T: AsRef<Path> + Clone,
{
    base_path: T,
    rname: Option<&'a [u8]>,
}

impl<T> FeatsGenOutput<'_, T>
where
    T: AsRef<Path> + Clone,
{
    pub(crate) fn new(path: T) -> Self {
        Self {
            base_path: path,
            rname: None,
        }
    }
}

impl<'a, T> FeaturesOutput<'a> for FeatsGenOutput<'a, T>
where
    T: AsRef<Path> + Clone,
{
    fn init<'b>(&mut self, _rid: u32, rname: &'b [u8])
    where
        'b: 'a,
    {
        self.rname.replace(rname);
    }

    fn update(
        &mut self,
        bases: Array2<u8>,
        quals: Array2<u8>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        wid: u16,
    ) {
        let rid = std::str::from_utf8(self.rname.unwrap()).unwrap();
        let output_path = self.base_path.as_ref().join(rid);
        create_dir_all(&output_path).expect("Cannot create directory");

        //output_features(&output_path, wid, &ids, &bases, supported.into_iter());
    }

    fn emit(&mut self) {
        self.rname = None;
    }
}

#[derive(Clone)]
pub(crate) struct InferenceOutput<'a> {
    sender: Sender<InferenceData>,
    rid: u32,
    rname: Option<&'a [u8]>,
    features: Vec<(usize, (Array2<u8>, Array2<u8>), Vec<SupportedPos>)>,
    batch_size: usize,
}

impl InferenceOutput<'_> {
    pub(crate) fn new(sender: Sender<InferenceData>, batch_size: usize) -> Self {
        Self {
            sender,
            rid: u32::MAX,
            rname: None,
            features: Vec::new(),
            batch_size: batch_size,
        }
    }
}

impl<'a> FeaturesOutput<'a> for InferenceOutput<'a> {
    fn init<'b>(&mut self, rid: u32, rname: &'b [u8])
    where
        'b: 'a,
    {
        self.rid = rid;
        self.rname.replace(rname);
        self.features.clear();
    }

    fn update(
        &mut self,
        bases: Array2<u8>,
        quals: Array2<u8>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        _wid: u16,
    ) {
        self.features.push((ids.len(), (bases, quals), supported));
    }

    fn emit(&mut self) {
        let data = prepare_examples(self.rid, self.features.drain(..), self.batch_size);
        self.sender.send(data).unwrap();
    }
}

#[derive(npyz::AutoSerialize, npyz::Serialize, PartialEq, Eq, Hash, Clone, Copy)]
pub(crate) struct SupportedPos {
    pub pos: u16,
    pub ins: u8,
}

impl SupportedPos {
    pub fn new(pos: u16, ins: u8) -> Self {
        SupportedPos { pos, ins }
    }
}
