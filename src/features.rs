use std::borrow::Cow;
use std::collections::HashMap;
use std::iter::Peekable;

use ordered_float::OrderedFloat;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::aligners::CigarOp;
use crate::haec_io::HAECRecord;
use crate::overlaps::{Overlap, Strand};

// const WINDOW_SIZE: u32 = 1024;
const WINDOW_SIZE: u32 = 5;

#[derive(Clone, Eq, PartialEq, Debug)]
struct OverlapWindow<'a> {
    overlap: &'a Overlap,
    qstart: u32,
    cigar_start_idx: usize,
    cigar_start_offset: u32,
    cigar_end_idx: usize,
    cigar_end_offset: u32,
}

impl<'a> OverlapWindow<'a> {
    fn new(
        overlap: &'a Overlap,
        qstart: u32,
        cigar_start_idx: usize,
        cigar_start_offset: u32,
        cigar_end_idx: usize,
        cigar_end_offset: u32,
    ) -> Self {
        OverlapWindow {
            overlap,
            qstart,
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

    match strand {
        Strand::Reverse if !is_target => Box::new(iter.rev()),
        _ => Box::new(iter),
    }
}

fn extract_windows<'a, 'b, I>(
    state: &mut WindowingState<'a>,
    overlap: &'a Overlap,
    cigar_iter: &mut Peekable<I>,
    is_target: bool,
) where
    I: Iterator<Item = (usize, Cow<'b, CigarOp>)>,
{
    let first_window;
    let last_window;
    let mut tpos;
    let mut qpos = 0;

    // Read is the target -> tstart
    // Read is the query  -> qstart
    if is_target {
        first_window = (overlap.tstart + WINDOW_SIZE - 1) / WINDOW_SIZE;
        last_window = overlap.tend / WINDOW_SIZE; // semi-closed interval
        tpos = overlap.tstart;
    } else {
        first_window = (overlap.qstart + WINDOW_SIZE - 1) / WINDOW_SIZE;
        last_window = overlap.qend / WINDOW_SIZE; // semi-closed interval
        tpos = overlap.qstart;
    }

    let mut qstart = None;
    let mut cigar_start_idx = None;
    let mut cigar_start_offset = None;
    if tpos % WINDOW_SIZE == 0 {
        qstart = Some(0);
        cigar_start_idx = Some(0);
        cigar_start_offset = Some(0);
    }

    while let Some((cigar_idx, op)) = cigar_iter.next() {
        let (tnew, qnew) = match op.as_ref() {
            CigarOp::Match(l) | CigarOp::Mismatch(l) => (tpos + *l, qpos + *l),
            CigarOp::Deletion(l) => (tpos + *l, qpos),
            CigarOp::Insertion(l) => {
                let prev_op_win = (tpos - 1) / WINDOW_SIZE;
                if first_window <= prev_op_win && prev_op_win < last_window {
                    state.max_ins[(tpos - 1) as usize] =
                        state.max_ins[(tpos - 1) as usize].max(*l as u16);
                }
                qpos += *l;
                continue;
            }
        };

        let current_w = tpos / WINDOW_SIZE;
        let new_w = tnew / WINDOW_SIZE;
        let diff_w = new_w - current_w; // Can span more than one window

        if diff_w == 0 {
            // Still in the same window - continue with looping
            tpos = tnew;
            qpos = qnew;
            continue;
        }

        // Handle first diff_w - 1 windows
        for i in 1..diff_w {
            let offset = (current_w + i) * WINDOW_SIZE - tpos;

            // If there was full window -> emit it, else label start
            if cigar_start_idx.is_some() {
                state.windows[(current_w + i) as usize - 1].push(OverlapWindow::new(
                    overlap,
                    qstart.unwrap(),
                    cigar_start_idx.unwrap(),
                    cigar_start_offset.unwrap(),
                    cigar_idx,
                    offset,
                ));
                //handle qpos
                if let CigarOp::Match(_) | CigarOp::Mismatch(_) = op.as_ref() {
                    qstart.replace(qpos + offset);
                }

                cigar_start_idx.replace(cigar_idx);
                cigar_start_offset.replace(offset);
            } else {
                qstart = Some(qpos + offset);
                cigar_start_idx = Some(cigar_idx);
                cigar_start_offset = Some(offset)
            }
        }

        // Handle the last one
        let offset = new_w * WINDOW_SIZE - tpos;

        let mut qend = if let CigarOp::Match(_) | CigarOp::Mismatch(_) = op.as_ref() {
            qpos + offset
        } else {
            qpos
        };

        let cigar_end_idx;
        let cigar_end_offset;
        if tnew == new_w * WINDOW_SIZE {
            if let Some(CigarOp::Insertion(l)) = cigar_iter.peek().map(|(_, op)| op.as_ref()) {
                qend += *l;
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
                qstart.unwrap(),
                cigar_start_idx.unwrap(),
                cigar_start_offset.unwrap(),
                cigar_end_idx,
                cigar_end_offset,
            ));
            qstart.replace(qend);
            cigar_start_idx.replace(cigar_end_idx);
            cigar_start_offset.replace(cigar_end_offset);
        } else {
            qstart = Some(qend);
            cigar_start_idx = Some(cigar_end_idx);
            cigar_start_offset = Some(cigar_end_offset);
        }

        tpos = tnew;
        qpos = qnew;
    }
}

fn get_features_for_window(
    tstart: usize,
    overlaps: &mut [OverlapWindow],
    max_ins: &[u16],
    tid: u32,
    //reads: &[HAECRecord],
) -> Vec<u32> {
    overlaps.sort_by_key(|ow| OrderedFloat(-ow.overlap.accuracy.unwrap()));

    let mut all_max = Vec::new();
    for ow in overlaps.iter().take(30) {
        let mut cigar = get_cigar_iterator(
            ow.overlap.cigar.as_ref().unwrap(),
            ow.overlap.tid == tid,
            ow.overlap.strand,
        );

        for _ in 0..ow.cigar_start_idx {
            cigar.next();
        }

        let m = cigar
            .take(ow.cigar_end_idx - ow.cigar_start_idx)
            .filter_map(|c| match c.as_ref() {
                CigarOp::Insertion(l) => Some(*l),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        all_max.push(m);
    }

    all_max
}

fn generate_features_for_read(
    read: &HAECRecord,
    tid: u32,
    overlaps: &[&Overlap],
    reads: &[HAECRecord],
) {
    // Get overlaps for windows
    let mut state = WindowingState::new(read.seq.len());

    for overlap in overlaps {
        let mut cigar_iter = get_cigar_iterator(
            overlap.cigar.as_ref().unwrap(),
            overlap.tid == tid,
            overlap.strand,
        )
        .enumerate()
        .peekable();

        // Move to first full window
        extract_windows(&mut state, &overlap, &mut cigar_iter, overlap.tid == tid);

        //state.windows.iter().for_each(|w|get)

        //todo!("Work on extracting features from windows")
    }

    let mut data = Vec::new();
    for i in (0..read.seq.len()).step_by(WINDOW_SIZE as usize) {
        let mut m = get_features_for_window(
            i,
            &mut state.windows[i / WINDOW_SIZE as usize],
            &state.max_ins,
            //&reads,
            tid,
        );
        data.append(&mut m);
    }

    let mean = data.iter().sum::<u32>() as f64 / data.len() as f64;
    let max = data.iter().max().unwrap();

    println!("Avg: {mean}, Max: {max}");
}

pub fn extract_features(reads: &[HAECRecord], overlaps: &[Overlap]) {
    let mut read_to_overlaps = get_reads_to_overlaps(overlaps);

    read_to_overlaps.par_iter_mut().for_each(|(rid, ovlps)| {
        generate_features_for_read(&reads[*rid as usize], *rid, ovlps, reads)
    });
}

#[cfg(test)]
mod tests {

    use std::borrow::Cow;

    use crate::{
        aligners::{
            wfa::{WFAAlignerBuilder, WFAAlignmentScope, WFAAlignmentSpan, WFADistanceMetric},
            CigarOp,
        },
        features::OverlapWindow,
        overlaps::{Overlap, Strand},
    };

    use super::{extract_windows, get_cigar_iterator, WindowingState};

    #[test]
    fn test_extract_windows1() {
        let query_seq = "AACTAAGTGTCGGTGGCTACTATATATCAGGTCCT";
        let target_seq = "AGCTAGTGTCAATGGCTACTTTTCAGGTCCT";
        let q_len = query_seq.len() as u32;
        let t_len = target_seq.len() as u32;

        let mut builder = WFAAlignerBuilder::new();
        builder.set_distance_metric(WFADistanceMetric::Edit);
        let aligner = builder.build();
        let cigar = aligner.align(query_seq, target_seq);
        let mut overlap = Overlap::new(
            0,
            q_len,
            0,
            q_len - 1,
            Strand::Forward,
            1,
            t_len,
            0,
            t_len - 1,
        );
        overlap.cigar = cigar;
        let mut cigar_iter =
            get_cigar_iterator(overlap.cigar.as_ref().unwrap(), true, Strand::Forward)
                .enumerate()
                .peekable();

        let mut state = WindowingState::new(target_seq.len());
        extract_windows(&mut state, &overlap, &mut cigar_iter, true);

        let mut expected_state = WindowingState::new(target_seq.len());
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 4, 0));
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 6, 4, 0, 5, 0));
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 11, 5, 0, 6, 3));
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 16, 6, 3, 8, 0));
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 22, 8, 0, 12, 3));
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 29, 12, 3, 12, 8));

        for i in 0..state.windows[0].len() {
            assert_eq!(state.windows[0][i], expected_state.windows[0][i]);
        }
    }

    #[test]
    fn test_extract_windows2() {
        let query_seq = "AATTTTTTTTTTTTTTTTTTTTGCACC";
        let target_seq = "AAGCTTTTTTTTTTTTTTTTTTTTCGTCC";
        let q_len = query_seq.len() as u32;
        let t_len = target_seq.len() as u32;

        let mut builder = WFAAlignerBuilder::new();
        builder.set_distance_metric(WFADistanceMetric::GapAffine {
            match_: (0),
            mismatch: (6),
            gap_opening: (4),
            gap_extension: (2),
        });
        let aligner = builder.build();
        let cigar = aligner.align(query_seq, target_seq);
        println!("{:?}", cigar);

        let mut overlap = Overlap::new(
            0,
            q_len,
            0,
            q_len - 1,
            Strand::Forward,
            1,
            t_len,
            0,
            t_len - 1,
        );
        overlap.cigar = cigar;
        let mut cigar_iter =
            get_cigar_iterator(overlap.cigar.as_ref().unwrap(), true, Strand::Forward)
                .enumerate()
                .peekable();

        let mut state = WindowingState::new(target_seq.len());
        extract_windows(&mut state, &overlap, &mut cigar_iter, true);

        let mut expected_state = WindowingState::new(target_seq.len());
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 2, 1));
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 3, 2, 1, 2, 6));
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 8, 2, 6, 2, 11));
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 13, 2, 11, 2, 16));
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 18, 2, 16, 3, 1));

        for i in 0..state.windows[0].len() {
            assert_eq!(state.windows[0][i], expected_state.windows[0][i]);
        }
    }
}
