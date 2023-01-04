use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::prelude::*;
use std::io::{BufWriter, Result};
use std::path::Path;

use indicatif::ParallelProgressIterator;
use lazy_static::lazy_static;
use ndarray::{Array, Array3, ArrayViewMut2, Axis};
use ndarray_npy::WriteNpyExt;
use ordered_float::OrderedFloat;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::aligners::CigarOp;
use crate::haec_io::HAECRecord;
use crate::overlaps::{Overlap, Strand};
use crate::windowing::{extract_windows, OverlapWindow};

const TOP_K: usize = 30;

lazy_static! {
    static ref BASE_COMP: HashMap<u8, u8> = {
        let mut m = HashMap::new();
        m.insert(b'A', b't');
        m.insert(b'C', b'g');
        m.insert(b'G', b'c');
        m.insert(b'T', b'a');
        m
    };
}

fn get_reads_to_overlaps(overlaps: &[Overlap]) -> HashMap<u32, Vec<&Overlap>> {
    let mut read_to_overlaps = HashMap::new();
    for overlap in overlaps {
        read_to_overlaps
            .entry(overlap.qid)
            .or_insert_with(Vec::new)
            .push(overlap);

        read_to_overlaps
            .entry(overlap.tid)
            .or_insert_with(Vec::new)
            .push(overlap);
    }

    read_to_overlaps
}

pub(crate) fn get_cigar_iterator(
    cigar: &[CigarOp],
    is_target: bool,
    strand: Strand,
) -> Box<dyn DoubleEndedIterator<Item = Cow<CigarOp>> + '_> {
    let iter = cigar.iter().map(move |c| {
        if is_target {
            Cow::Borrowed(c)
        } else {
            Cow::Owned(c.reverse())
        }
    });

    match strand {
        Strand::Reverse if !is_target => Box::new(iter.rev()),
        _ => Box::new(iter),
    }
}

fn get_max_ins_for_window(
    overlaps: &[OverlapWindow], // Sorted overlaps
    tid: u32,
    tstart: usize,
    window_length: usize,
) -> Vec<u16> {
    let mut max_ins = vec![0; window_length];
    for ow in overlaps.iter().take(TOP_K) {
        let mut tpos = ow.tstart as usize - tstart;

        // Handle cigar
        let cigar = get_cigar_iterator(
            ow.overlap.cigar.as_ref().unwrap(),
            ow.overlap.tid == tid,
            ow.overlap.strand,
        )
        .skip(ow.cigar_start_idx);
        let cigar_len = ow.cigar_end_idx - ow.cigar_start_idx + 1;

        cigar.take(cigar_len).enumerate().for_each(|(i, op)| {
            let l = match op.as_ref() {
                CigarOp::Match(l) | CigarOp::Mismatch(l) | CigarOp::Deletion(l) => *l as usize,
                CigarOp::Insertion(l) => {
                    /*max_ins.get(tpos - 1).expect(&format!(
                        "{} {} {} {} {} {} {:?}",
                        ow.tstart as usize % window_length,
                        ow.qstart,
                        ow.cigar_start_idx,
                        ow.cigar_start_offset,
                        ow.cigar_end_idx,
                        ow.cigar_end_offset,
                        cigar_to_string(ow.overlap.cigar.as_ref().unwrap())
                    ));*/

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
    query: &HAECRecord,
    offset: usize,
    tid: u32,
    max_ins: &[u16],
) {
    // Handle query sequence
    let (qstart, qend) = if window.overlap.tid == tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    let mut query_iter: Box<dyn DoubleEndedIterator<Item = (u8, u8)>> = match window.overlap.strand
    {
        Strand::Forward => Box::new(
            query
                .subseq_iter((qstart + window.qstart) as usize..qend as usize)
                .map(|(b, q)| (*b, *q)),
        ),
        Strand::Reverse => Box::new(
            query
                .subseq_iter(qstart as usize..(qend - window.qstart) as usize)
                .rev()
                .map(|(b, q)| (*BASE_COMP.get(&b).unwrap(), *q)),
        ),
    };
    //let mut query_iter = query_iter.skip(window.qstart as usize);

    // Handle cigar
    let cigar = get_cigar_iterator(
        window.overlap.cigar.as_ref().unwrap(),
        window.overlap.tid == tid,
        window.overlap.strand,
    )
    .skip(window.cigar_start_idx as usize);

    // Number of cigars for the window
    // TODO get error when we calculate correct number for end -> (idx, 0)
    // Works for this expression but unecessarily iterates through (idx, 0)
    let cigar_len = window.cigar_end_idx - window.cigar_start_idx + 1;

    // Get features
    let gap = if let Strand::Forward = window.overlap.strand {
        b'*'
    } else {
        b'#'
    };
    features.fill(gap);

    let mut idx = 0;
    let mut tpos = offset;
    cigar
        .take(cigar_len)
        .enumerate()
        .for_each(|(cigar_idx, op)| {
            let mut l = match op.as_ref() {
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
            match op.as_ref() {
                CigarOp::Match(_) | CigarOp::Mismatch(_) => {
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");
                        features[[idx, 0]] = base;
                        features[[idx, 1]] = qual;

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

                        features[[idx + i, 0]] = base;
                        features[[idx + i, 1]] = qual;
                    }
                    idx += max_ins[tpos - 1] as usize; // Move back to the last base
                }
            }
        });
}

fn write_target_for_window(
    tstart: usize,
    target: &HAECRecord,
    max_ins: &[u16],
    mut features: ArrayViewMut2<'_, u8>,
    window_length: usize,
) {
    features.fill(b'*'); // Fill like forward

    let mut tpos = 0;
    target.seq[tstart..tstart + window_length]
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
    tid: u32,
    reads: &[HAECRecord],
    max_ins: &[u16],
    tstart: usize,
    window_length: usize, // Full window length
) -> Array3<u8> {
    //Get features
    let length = max_ins.iter().map(|v| *v as usize).sum::<usize>() + max_ins.len();
    let mut features = Array::zeros((31, length, 2));
    features.index_axis_mut(Axis(2), 0).fill(b'.'); // Set '.' as marker for empty row

    // First write the target
    write_target_for_window(
        tstart,
        &reads[tid as usize],
        &max_ins,
        features.index_axis_mut(Axis(0), 0),
        window_length,
    );

    // Write top-k overlaps for the window
    overlaps.iter().take(TOP_K).enumerate().for_each(|(i, ow)| {
        get_features_for_ol_window(
            features.index_axis_mut(Axis(0), i + 1),
            ow,
            &reads[ow.overlap.return_other_id(tid) as usize],
            ow.tstart as usize - tstart,
            tid,
            &max_ins,
        )
    });

    features
}

pub fn extract_features<P: AsRef<Path>>(
    reads: &[HAECRecord],
    overlaps: &[Overlap],
    window_size: u32,
    output_path: P,
) where
    P: AsRef<Path> + Send + Sync,
{
    let read_to_overlaps = get_reads_to_overlaps(overlaps);

    eprintln!("Extracting features...");
    read_to_overlaps
        .par_iter()
        .progress_count(read_to_overlaps.len() as u64)
        .for_each(|(rid, ovlps)| {
            let read = &reads[*rid as usize];

            // Get overlaps for windows
            let n_windows = (read.seq.len() + window_size as usize - 1) / window_size as usize;
            let mut windows = vec![Vec::new(); n_windows];

            for overlap in ovlps {
                let mut cigar_iter = get_cigar_iterator(
                    overlap.cigar.as_ref().unwrap(),
                    overlap.tid == *rid,
                    overlap.strand,
                )
                .enumerate()
                .peekable();

                //Extract windows
                extract_windows(
                    &mut windows,
                    &overlap,
                    &mut cigar_iter,
                    overlap.tid == *rid,
                    window_size,
                );
            }

            // Create directory for the read
            let output_path = output_path.as_ref().join(&read.id);
            create_dir_all(&output_path).expect("Cannot create directory");

            for i in 0..n_windows {
                if windows[i].len() == 0 {
                    continue;
                }

                let win_len = if i == n_windows - 1 {
                    read.seq.len() - i * window_size as usize
                } else {
                    window_size as usize
                };

                // Sort window to take TOP-K
                windows[i].sort_by_key(|ow| OrderedFloat(-ow.overlap.accuracy.unwrap()));

                let max_ins =
                    get_max_ins_for_window(&windows[i], *rid, i * window_size as usize, win_len);

                let window = get_features_for_window(
                    &mut windows[i],
                    *rid,
                    reads,
                    &max_ins,
                    i * window_size as usize,
                    win_len,
                );

                let qids: Vec<&str> = windows[i]
                    .iter()
                    .map(|ow| reads[ow.overlap.return_other_id(*rid) as usize].id.as_str())
                    .collect();

                //TODO handle Result
                output_features(&output_path, i, qids, window, max_ins);
            }
        });
}

fn output_features<P: AsRef<Path>>(
    path: P,
    window_id: usize,
    ids: Vec<&str>,
    features: Array3<u8>,
    max_ins: Vec<u16>,
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

    let ins_path = path.as_ref().join(format!("{}.ins.npy", window_id));
    let file = File::create(ins_path)?;
    Array::from(max_ins).write_npy(file).unwrap();

    Ok(())
}
