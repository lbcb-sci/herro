use std::borrow::Cow;
use std::collections::HashMap;
use std::process::exit;

use lazy_static::lazy_static;
use ndarray::{Array2, ArrayViewMut1};
use ordered_float::OrderedFloat;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::aligners::CigarOp;
use crate::haec_io::HAECRecord;
use crate::overlaps::{Overlap, Strand};
use crate::windowing::{extract_windows, OverlapWindow};

lazy_static! {
    static ref BASE_COMP: HashMap<char, char> = {
        let mut m = HashMap::new();
        m.insert('A', 't');
        m.insert('C', 'g');
        m.insert('G', 'c');
        m.insert('T', 'a');
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
    window_length: usize,
) -> Vec<u16> {
    let mut max_ins = vec![0; window_length];
    for ow in overlaps.iter().take(30) {
        let mut tpos = 0;

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
    mut features: ArrayViewMut1<'_, char>,
    window: &OverlapWindow,
    query: &HAECRecord,
    tid: u32,
    max_ins: &[u16],
) {
    // Handle query sequence
    let (qstart, qend) = if window.overlap.tid == tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };
    let query_iter: Box<dyn DoubleEndedIterator<Item = char>> = match window.overlap.strand {
        Strand::Forward => Box::new(query.seq[qstart as usize..qend as usize].chars()),
        Strand::Reverse => Box::new(
            query.seq[qstart as usize..qend as usize]
                .chars()
                .rev()
                .map(|c| *BASE_COMP.get(&c).unwrap()),
        ),
    };
    let mut query_iter = query_iter.skip(window.qstart as usize);

    // Handle cigar
    let cigar = get_cigar_iterator(
        window.overlap.cigar.as_ref().unwrap(),
        window.overlap.tid == tid,
        window.overlap.strand,
    )
    .skip(window.cigar_start_idx as usize);
    let cigar_len = window.cigar_end_idx - window.cigar_start_idx + 1;

    // Get features
    let gap = if let Strand::Forward = window.overlap.strand {
        '*'
    } else {
        '#'
    };
    features.fill(gap);
    let mut idx = 0;
    let mut tpos = 0;
    cigar.take(cigar_len).enumerate().for_each(|(i, op)| {
        let mut l = match op.as_ref() {
            CigarOp::Match(l)
            | CigarOp::Mismatch(l)
            | CigarOp::Deletion(l)
            | CigarOp::Insertion(l) => *l as usize,
        };

        // Calculate true length
        if cigar_len == 1 {
            l = (window.cigar_end_offset - window.cigar_start_offset) as usize;
        } else if i == 0 {
            l -= window.cigar_start_offset as usize;
        } else if i == cigar_len - 1 {
            l = window.cigar_end_offset as usize;
        }

        // Write features
        match op.as_ref() {
            CigarOp::Match(_) | CigarOp::Mismatch(_) => {
                for i in 0..l {
                    features[idx] = query_iter.next().unwrap();

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
                idx -= max_ins[tpos - 1] as usize; // Return to first insertion for the previous base
                for i in 0..l {
                    features[idx + i] = query_iter.next().unwrap();
                }
                idx += max_ins[tpos - 1] as usize; // Move back to the last base
            }
        }
    });
}

fn get_features_for_window(
    overlaps: &mut [OverlapWindow],
    tid: u32,
    reads: &[HAECRecord],
    window_length: usize,
) {
    overlaps.sort_by_key(|ow| OrderedFloat(-ow.overlap.accuracy.unwrap()));
    let max_ins = get_max_ins_for_window(overlaps, tid, window_length);

    //Get features
    let length = max_ins.iter().map(|v| *v as usize).sum::<usize>() + max_ins.len();
    let mut features = Array2::from_elem((length, 30), '.');

    overlaps.iter().take(30).enumerate().for_each(|(i, ow)| {
        get_features_for_ol_window(
            features.row_mut(i),
            ow,
            &reads[ow.overlap.return_other_id(tid) as usize],
            tid,
            &max_ins,
        )
    });

    println!("{:?}", features);
    exit(-1);
}

fn generate_features_for_read(
    read: &HAECRecord,
    tid: u32,
    overlaps: &[&Overlap],
    reads: &[HAECRecord],
    window_size: u32,
) {
    // Get overlaps for windows
    let n_windows = (read.seq.len() + window_size as usize - 1) / window_size as usize;
    let mut windows = vec![Vec::new(); n_windows];

    for overlap in overlaps {
        let mut cigar_iter = get_cigar_iterator(
            overlap.cigar.as_ref().unwrap(),
            overlap.tid == tid,
            overlap.strand,
        )
        .enumerate()
        .peekable();

        //Extract windows
        extract_windows(
            &mut windows,
            &overlap,
            &mut cigar_iter,
            overlap.tid == tid,
            window_size,
        );
    }

    for i in (0..read.seq.len()).step_by(window_size as usize) {
        let win_len = read.seq.len().min(i + window_size as usize) - i;
        get_features_for_window(&mut windows[i / window_size as usize], tid, reads, win_len);
    }
}

pub fn extract_features(reads: &[HAECRecord], overlaps: &[Overlap], window_size: u32) {
    let mut read_to_overlaps = get_reads_to_overlaps(overlaps);

    read_to_overlaps.par_iter_mut().for_each(|(rid, ovlps)| {
        generate_features_for_read(&reads[*rid as usize], *rid, ovlps, reads, window_size)
    });
}
