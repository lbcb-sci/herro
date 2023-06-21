use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::prelude::*;
use std::io::{BufWriter, Result};
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};

use crossbeam_channel::Sender;

use lazy_static::lazy_static;
use ndarray::{s, Array, Array3, ArrayViewMut2, Axis};
use ndarray_npy::WriteNpyExt;
use ordered_float::OrderedFloat;

use crate::aligners::{calculate_accuracy, fix_cigar, get_proper_cigar, CigarOp};
use crate::haec_io::HAECRecord;
use crate::inference::{prepare_examples, InputData};
use crate::overlaps::{self, Alignment, CigarStatus, Overlap, Strand};
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

fn get_features_for_ol_window(
    mut features: ArrayViewMut2<'_, u8>,
    window: &OverlapWindow,
    cigar: &[CigarOp],
    query: &HAECRecord,
    offset: usize,
    tid: u32,
    max_ins: &[u16],
    qbuffer: &mut [u8],
) {
    // Handle query sequence
    let overlap = &window.overlap;
    let (qstart, qend) = if overlap.tid == tid {
        (overlap.qstart, overlap.qend)
    } else {
        (overlap.tstart, overlap.tend)
    };

    let mut query_iter: Box<dyn DoubleEndedIterator<Item = (&u8, &u8)>> = match overlap.strand {
        Strand::Forward => {
            let range = (qstart + window.qstart) as usize..qend as usize;
            let qlen = qend as usize - (qstart + window.qstart) as usize;

            query.seq.get_subseq(range.clone(), qbuffer);
            let quals = &query.qual[range];

            Box::new(qbuffer[..qlen].iter().zip(quals))
        }
        Strand::Reverse => {
            let range = qstart as usize..(qend - window.qstart) as usize;
            let qlen = (qend - window.qstart) as usize - qstart as usize;

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
    let gap = if let Strand::Forward = overlap.strand {
        b'*'
    } else {
        b'#'
    };
    features.column_mut(0).fill(gap); // Initialize with gap token

    let mut tpos = offset; // position in the target read (excluding insertions)
    let mut idx = offset + max_ins[..offset].iter().map(|v| *v as usize).sum::<usize>(); // position in the features (including insertions)

    if idx > 0 {
        // No alignment at the start
        features.slice_mut(s![..idx, 0]).fill(b'.');
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
                        features[[idx, 0]] = *base;
                        features[[idx, 1]] = *qual;

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

                        features[[idx + i, 0]] = *base;
                        features[[idx + i, 1]] = *qual;
                    }
                    idx += max_ins[tpos - 1] as usize; // Move back to the last base
                }
            }
        });

    if idx < features.shape()[0] {
        // No alignment at the end
        features.slice_mut(s![idx.., 0]).fill(b'.');
    }
}

fn write_target_for_window(
    tstart: usize,
    target: &HAECRecord,
    max_ins: &[u16],
    mut features: ArrayViewMut2<'_, u8>,
    window_length: usize,
    tbuffer: &mut [u8],
) {
    features.column_mut(0).fill(b'*'); // Fill like forward

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
            features[[tpos, 0]] = *b;
            features[[tpos, 1]] = *q;

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
) -> Array3<u8> {
    //Get features
    let length = max_ins.iter().map(|v| *v as usize).sum::<usize>() + max_ins.len();
    let mut features = Array::zeros((1 + TOP_K, length, 2));
    features.index_axis_mut(Axis(2), 0).fill(b'.'); // Set '.' as marker for empty row
    features.index_axis_mut(Axis(2), 1).fill(b'!'); // Set default quality to 0

    // First write the target
    write_target_for_window(
        tstart,
        &reads[tid as usize],
        &max_ins,
        features.index_axis_mut(Axis(0), 0),
        window_length,
        tbuffer,
    );

    // Write top-k overlaps for the window
    overlaps.iter().take(TOP_K).enumerate().for_each(|(i, ow)| {
        let qid = ow.overlap.return_other_id(tid);
        get_features_for_ol_window(
            features.index_axis_mut(Axis(0), i + 1),
            ow,
            ovlps_cigar_map.get(&qid).unwrap(),
            &reads[qid as usize],
            ow.tstart as usize - tstart,
            tid,
            &max_ins,
            qbuffer,
        )
    });

    features
}

fn overlap_window_filter(cigar: &[CigarOp]) -> bool {
    let long_indel = cigar.iter().any(|op| match op {
        CigarOp::Insertion(l) | CigarOp::Deletion(l) if *l >= 30 => true,
        _ => false,
    });

    //accuracy >= 0.80 && !long_indel
    calculate_accuracy(cigar) >= 0.85 && !long_indel
}

pub(crate) fn extract_features(
    rid: u32,
    reads: &[HAECRecord],
    overlaps: &[Arc<RwLock<Alignment>>],
    window_size: u32,
    (tbuf, qbuf): (&mut [u8], &mut [u8]),
    sender: Sender<InputData>,
) {
    let read = &reads[rid as usize];

    // Get overlaps for windows
    let n_windows = (read.seq.len() + window_size as usize - 1) / window_size as usize;
    let mut windows = vec![Vec::new(); n_windows];

    let mut ovlps_cigar_map = HashMap::new();
    for ovlp in overlaps {
        let alignment = ovlp.read().unwrap();
        if let CigarStatus::Unmapped = alignment.cigar {
            continue;
        }

        let overlap = Rc::new(alignment.overlap.clone());
        let qid = overlap.return_other_id(rid);

        let mut cigar = get_proper_cigar(
            alignment.cigar.as_ref().unwrap(),
            overlap.tid == rid,
            overlap.strand,
        );

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

    let mut features = Vec::new();
    for i in 0..n_windows {
        if windows[i].len() == 0 {
            continue;
        }

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

            let cigar_end = (ow.cigar_end_idx + 1).min(cigar.len());
            let acc = calculate_accuracy(&cigar[ow.cigar_start_idx..cigar_end]);
            OrderedFloat(-acc)
        });

        let max_ins = get_max_ins_for_window(
            &windows[i],
            &ovlps_cigar_map,
            rid,
            i * window_size as usize,
            win_len,
        );

        let window = get_features_for_window(
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

        /*let qids: Vec<&str> = windows[i]
        .iter()
        .map(|ow| reads[ow.overlap.return_other_id(rid) as usize].id.as_str())
        .collect();*/
        features.push((i as u16, window));

        //TODO handle Result
        //output_features(&output_path, i, &qids, &window);
    }

    let features = prepare_examples(rid, features);
    sender.send(features).unwrap();
}

fn output_features<P: AsRef<Path>>(
    path: P,
    window_id: usize,
    ids: &[&str],
    features: &Array3<u8>,
) -> Result<()> {
    let ids_path = path.as_ref().join(format!("{}.ids.txt", window_id));
    let ids_file = File::create(ids_path)?;
    let mut ids_writer = BufWriter::new(ids_file);
    for id in ids {
        writeln!(&mut ids_writer, "{}", id)?
    }

    let features_path = path.as_ref().join(format!("{}.features.npy", window_id));
    let file = File::create(features_path)?;
    features.write_npy(file).unwrap();

    Ok(())
}
