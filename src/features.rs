use std::borrow::Cow;
use std::collections::HashMap;
use std::iter::Peekable;

use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::aligners::CigarOp;
use crate::haec_io::HAECRecord;
use crate::overlaps::{Overlap, Strand};

const WINDOW_SIZE: u32 = 1024;

#[derive(Clone)]
struct OverlapWindow<'a> {
    overlap: &'a Overlap,
    cigar_start_idx: usize,
    cigar_start_offset: u32,
    cigar_end_idx: usize,
    cigar_end_offset: u32,
}

impl<'a> OverlapWindow<'a> {
    fn new(
        overlap: &'a Overlap,
        cigar_start_idx: usize,
        cigar_start_offset: u32,
        cigar_end_idx: usize,
        cigar_end_offset: u32,
    ) -> Self {
        OverlapWindow {
            overlap,
            cigar_start_idx,
            cigar_start_offset,
            cigar_end_idx,
            cigar_end_offset,
        }
    }
}

struct WindowingState<'a> {
    windows: Vec<Vec<OverlapWindow<'a>>>,
    max_ins: Vec<u16>,
}

impl<'a> WindowingState<'a> {
    fn new(seq_len: usize) -> Self {
        let n_windows = (seq_len + WINDOW_SIZE as usize - 1) / WINDOW_SIZE as usize;
        let windows = vec![Vec::new(); n_windows];
        let max_ins = vec![0; seq_len];

        WindowingState { windows, max_ins }
    }
}

fn calculate_accuracy(cigar: &[CigarOp]) -> f32 {
    let (mut matches, mut subs, mut ins, mut dels) = (0u32, 0u32, 0u32, 0u32);
    for op in cigar {
        match op {
            CigarOp::Match(l) => matches += l,
            CigarOp::Mismatch(l) => subs += l,
            CigarOp::Insertion(l) => ins += l,
            CigarOp::Deletion(l) => dels += l,
        };
    }

    let length = (matches + subs + ins + dels) as f32;
    matches as f32 / length
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

fn get_cigar_iterator(
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

    if let Strand::Reverse = strand {
        return Box::new(iter.rev());
    }

    Box::new(iter)
}

fn extract_windows<'a, 'b, I>(
    state: &mut WindowingState<'a>,
    overlap: &'a Overlap,
    cigar_iter: &mut Peekable<I>,
    tstart: u32,
    tend: u32,
) where
    I: Iterator<Item = (usize, Cow<'b, CigarOp>)>,
{
    let first_window = (tstart + WINDOW_SIZE - 1) / WINDOW_SIZE;
    let last_window = tend / WINDOW_SIZE; // semi-closed interval
    let mut tpos = tstart;

    let mut cigar_start_idx = None;
    let mut cigar_start_offset = None;
    if tpos % WINDOW_SIZE == 0 {
        cigar_start_idx = Some(0);
        cigar_start_offset = Some(0);
    }

    while let Some((cigar_idx, op)) = cigar_iter.next() {
        let tnew = match op.as_ref() {
            CigarOp::Match(l) | CigarOp::Mismatch(l) | CigarOp::Deletion(l) => tpos + *l,
            CigarOp::Insertion(l) => {
                let prev_op_win = (tpos - 1) / WINDOW_SIZE;
                if first_window <= prev_op_win && prev_op_win < last_window {
                    state.max_ins[(tpos - 1) as usize] =
                        state.max_ins[(tpos - 1) as usize].max(*l as u16);
                }
                continue;
            }
        };

        let current_w = tpos / WINDOW_SIZE;
        let new_w = tnew / WINDOW_SIZE;
        let diff_w = new_w - current_w; // Can span more than one window

        if diff_w == 0 {
            // Still in the same window - continue with looping
            tpos = tnew;
            continue;
        }

        // Handle first diff_w - 1 windows
        for i in 1..diff_w {
            let offset = (current_w + i) * WINDOW_SIZE - tpos;

            // If there was full window -> emit it, else label start
            if cigar_start_idx.is_some() {
                state.windows[(current_w + i) as usize - 1].push(OverlapWindow::new(
                    overlap,
                    cigar_start_idx.unwrap(),
                    cigar_start_offset.unwrap(),
                    cigar_idx,
                    offset,
                ));

                cigar_start_idx.replace(cigar_idx);
                cigar_start_offset.replace(offset);
            } else {
                cigar_start_idx = Some(cigar_idx);
                cigar_start_offset = Some(offset)
            }
        }

        // Handle the last one
        let offset = new_w * WINDOW_SIZE - tpos;

        let cigar_end_idx;
        let cigar_end_offset;
        if tnew == new_w * WINDOW_SIZE {
            if let Some(CigarOp::Insertion(_)) = cigar_iter.peek().map(|(_, op)| op.as_ref()) {
                cigar_end_idx = cigar_idx + 2;
            } else {
                cigar_end_idx = cigar_idx + 1;
            }

            cigar_end_offset = 0; // Beginning of the new op
        } else {
            cigar_end_idx = cigar_idx;
            cigar_end_offset = offset;
        }

        if cigar_start_idx.is_some() {
            state.windows[new_w as usize - 1].push(OverlapWindow::new(
                overlap,
                cigar_start_idx.unwrap(),
                cigar_start_offset.unwrap(),
                cigar_end_idx,
                cigar_end_offset,
            ));

            cigar_start_idx.replace(cigar_end_idx);
            cigar_start_offset.replace(cigar_end_offset);
        } else {
            cigar_start_idx = Some(cigar_end_idx);
            cigar_start_offset = Some(cigar_end_offset);
        }

        tpos = tnew;
    }
}

fn generate_features_for_read(read: &HAECRecord, overlaps: &mut [&Overlap], tid: u32) {
    // Get alignment accuracy for every overlap
    let accuracies: Vec<f32> = overlaps
        .iter()
        .map(|o| calculate_accuracy(o.cigar.as_ref().unwrap()))
        .collect();

    // Get overlaps for windows
    let mut state = WindowingState::new(read.seq.len());

    for overlap in overlaps {
        // Read is the target -> tstart
        // Read is the query  -> qstart
        let (tstart, tend) = if overlap.tid == tid {
            (overlap.tstart, overlap.tend)
        } else {
            (overlap.qstart, overlap.qend)
        };

        let mut cigar_iter = get_cigar_iterator(
            overlap.cigar.as_ref().unwrap(),
            overlap.tid == tid,
            overlap.strand,
        )
        .enumerate()
        .peekable();

        // Move to first full window
        extract_windows(&mut state, &overlap, &mut cigar_iter, tstart, tend);

        todo!("Work on extracting features from windows")
    }
}

pub fn extract_features(reads: &[HAECRecord], overlaps: &[Overlap]) {
    let mut read_to_overlaps = get_reads_to_overlaps(overlaps);

    read_to_overlaps
        .par_iter_mut()
        .map(|(rid, ovlps)| generate_features_for_read(&reads[*rid as usize], ovlps, *rid));
}
