use std::borrow::Cow;
use std::collections::HashMap;
use std::iter::Peekable;

use ordered_float::OrderedFloat;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::aligners::CigarOp;
use crate::haec_io::HAECRecord;
use crate::overlaps::{Overlap, Strand};

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
}

impl<'a> WindowingState<'a> {
    fn new(seq_len: usize, window_size: u32) -> Self {
        let n_windows = (seq_len + window_size as usize - 1) / window_size as usize;
        let windows = vec![Vec::new(); n_windows];

        WindowingState { windows }
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
    window_size: u32,
) where
    I: Iterator<Item = (usize, Cow<'b, CigarOp>)>,
{
    let first_window;
    let last_window;
    let mut tpos;
    let mut qpos = 0;
    let tlen;

    // Read is the target -> tstart
    // Read is the query  -> qstart
    if is_target {
        first_window = (overlap.tstart + window_size - 1) / window_size;
        last_window = (overlap.tend - 1) / window_size; // semi-closed interval
        tpos = overlap.tstart;
        tlen = overlap.tlen;
    } else {
        first_window = (overlap.qstart + window_size - 1) / window_size;
        last_window = overlap.qend / window_size; // semi-closed interval
        tpos = overlap.qstart;
        tlen = overlap.qlen;
    }

    let mut qstart = None;
    let mut cigar_start_idx = None;
    let mut cigar_start_offset = None;
    if tpos % window_size == 0 {
        qstart = Some(0);
        cigar_start_idx = Some(0);
        cigar_start_offset = Some(0);
    }

    while let Some((cigar_idx, op)) = cigar_iter.next() {
        let (tnew, qnew) = match op.as_ref() {
            CigarOp::Match(l) | CigarOp::Mismatch(l) => (tpos + *l, qpos + *l),
            CigarOp::Deletion(l) => (tpos + *l, qpos),
            CigarOp::Insertion(l) => {
                /*let prev_op_win = (tpos - 1) / window_size;
                if first_window <= prev_op_win && prev_op_win < last_window {
                    state.max_ins[(tpos - 1) as usize] =
                        state.max_ins[(tpos - 1) as usize].max(*l as u16);
                }*/
                qpos += *l;
                continue;
            }
        };

        let current_w = tpos / window_size;
        let new_w = tnew / window_size;
        let diff_w = new_w - current_w; // Can span more than one window

        if diff_w == 0 {
            // Still in the same window - continue with looping
            tpos = tnew;
            qpos = qnew;
            continue;
        }

        // Handle first diff_w - 1 windows
        for i in 1..diff_w {
            let offset = (current_w + i) * window_size - tpos;

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
        let offset = new_w * window_size - tpos;

        let mut qend = if let CigarOp::Match(_) | CigarOp::Mismatch(_) = op.as_ref() {
            qpos + offset
        } else {
            qpos
        };

        let cigar_end_idx;
        let cigar_end_offset;
        if tnew == new_w * window_size {
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

    if tpos == tlen && tlen % window_size != 0 {
        state.windows[last_window as usize].push(OverlapWindow::new(
            overlap,
            qstart.unwrap(),
            cigar_start_idx.unwrap(),
            cigar_start_offset.unwrap(),
            overlap.cigar.as_ref().unwrap().len(),
            0,
        ))
    }
}

fn get_features_for_window(
    window_length: usize,
    overlaps: &mut [OverlapWindow],
    tid: u32,
    //reads: &[HAECRecord],
) {
    overlaps.sort_by_key(|ow| OrderedFloat(-ow.overlap.accuracy.unwrap()));

    let mut max_ins = vec![0; window_length];
    for ow in overlaps.iter().take(30) {
        let mut tpos = 0;

        // Handle max ins
        let mut cigar = get_cigar_iterator(
            ow.overlap.cigar.as_ref().unwrap(),
            ow.overlap.tid == tid,
            ow.overlap.strand,
        );

        // Handle start index and offset
        for _ in 0..ow.cigar_start_idx {
            cigar.next();
        }

        let cigar_len = ow.cigar_end_idx - ow.cigar_start_idx + 1;

        cigar.take(cigar_len).enumerate().for_each(|(i, op)| {
            let l = match op.as_ref() {
                CigarOp::Match(l) | CigarOp::Mismatch(l) | CigarOp::Deletion(l) => *l as usize,
                CigarOp::Insertion(l) => {
                    max_ins[tpos - 1] = max_ins[tpos - 1].max(*l);
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

    let total = max_ins.into_iter().sum::<u32>() as usize + window_length;
    println!(
        "Total number of positions {total}, number of overlaps {}",
        overlaps.len()
    );
}

fn generate_features_for_read(
    read: &HAECRecord,
    tid: u32,
    overlaps: &[&Overlap],
    reads: &[HAECRecord],
    window_size: u32,
) {
    // Get overlaps for windows
    let mut state = WindowingState::new(read.seq.len(), window_size);

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
            &mut state,
            &overlap,
            &mut cigar_iter,
            overlap.tid == tid,
            window_size,
        );

        //state.windows.iter().for_each(|w|get)

        //todo!("Work on extracting features from windows")
    }

    for i in (0..read.seq.len()).step_by(window_size as usize) {
        let win_len = read.seq.len().min(i + window_size as usize) - i;
        get_features_for_window(win_len, &mut state.windows[i / window_size as usize], tid);
    }
}

pub fn extract_features(reads: &[HAECRecord], overlaps: &[Overlap], window_size: u32) {
    let mut read_to_overlaps = get_reads_to_overlaps(overlaps);

    read_to_overlaps.par_iter_mut().for_each(|(rid, ovlps)| {
        generate_features_for_read(&reads[*rid as usize], *rid, ovlps, reads, window_size)
    });
}

#[cfg(test)]
mod tests {

    const WINDOW_SIZE: u32 = 5;

    use crate::{
        aligners::wfa::{WFAAlignerBuilder, WFADistanceMetric},
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
        let mut overlap = Overlap::new(0, q_len, 0, q_len, Strand::Forward, 1, t_len, 0, t_len);

        overlap.cigar = cigar;
        let mut cigar_iter =
            get_cigar_iterator(overlap.cigar.as_ref().unwrap(), true, Strand::Forward)
                .enumerate()
                .peekable();

        let mut state = WindowingState::new(target_seq.len(), WINDOW_SIZE);
        extract_windows(&mut state, &overlap, &mut cigar_iter, true, WINDOW_SIZE);

        let mut expected_state = WindowingState::new(target_seq.len(), WINDOW_SIZE);
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 4, 0));
        expected_state.windows[1].push(OverlapWindow::new(&overlap, 6, 4, 0, 5, 0));
        expected_state.windows[2].push(OverlapWindow::new(&overlap, 11, 5, 0, 6, 3));
        expected_state.windows[3].push(OverlapWindow::new(&overlap, 16, 6, 3, 8, 0));
        expected_state.windows[4].push(OverlapWindow::new(&overlap, 22, 8, 0, 12, 3));
        expected_state.windows[5].push(OverlapWindow::new(&overlap, 29, 12, 3, 12, 8));
        expected_state.windows[6].push(OverlapWindow::new(&overlap, 34, 12, 8, 13, 0));

        for i in 0..state.windows.len() {
            assert_eq!(state.windows[i][0], expected_state.windows[i][0]);
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

        let mut overlap = Overlap::new(0, q_len, 0, q_len, Strand::Forward, 1, t_len, 0, t_len);
        overlap.cigar = cigar;
        let mut cigar_iter =
            get_cigar_iterator(overlap.cigar.as_ref().unwrap(), true, Strand::Forward)
                .enumerate()
                .peekable();

        let mut state = WindowingState::new(target_seq.len(), WINDOW_SIZE);
        extract_windows(&mut state, &overlap, &mut cigar_iter, true, WINDOW_SIZE);

        let mut expected_state = WindowingState::new(target_seq.len(), WINDOW_SIZE);
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 2, 1));
        expected_state.windows[1].push(OverlapWindow::new(&overlap, 3, 2, 1, 2, 6));
        expected_state.windows[2].push(OverlapWindow::new(&overlap, 8, 2, 6, 2, 11));
        expected_state.windows[3].push(OverlapWindow::new(&overlap, 13, 2, 11, 2, 16));
        expected_state.windows[4].push(OverlapWindow::new(&overlap, 18, 2, 16, 3, 1));
        expected_state.windows[5].push(OverlapWindow::new(&overlap, 23, 3, 1, 5, 0));

        for i in 0..state.windows.len() {
            assert_eq!(state.windows[i][0], expected_state.windows[i][0]);
        }
    }

    #[test]
    fn test_extract_windows3() {
        // long insertion
        let query_seq = "ATCGTTTTTTTTTTTTTTTTTTTTATCGAAAAAAAAAAAA"; // 40
        let target_seq = "ATCGATCGAAAAAAAAAAAA"; // 20
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

        let mut overlap = Overlap::new(0, q_len, 0, q_len, Strand::Forward, 1, t_len, 0, t_len);
        overlap.cigar = cigar;
        let mut cigar_iter =
            get_cigar_iterator(overlap.cigar.as_ref().unwrap(), true, Strand::Forward)
                .enumerate()
                .peekable();

        let mut state = WindowingState::new(target_seq.len(), WINDOW_SIZE);
        extract_windows(&mut state, &overlap, &mut cigar_iter, true, WINDOW_SIZE);

        let mut expected_state = WindowingState::new(target_seq.len(), WINDOW_SIZE);
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 2, 1));
        expected_state.windows[1].push(OverlapWindow::new(&overlap, 25, 2, 1, 2, 6));
        expected_state.windows[2].push(OverlapWindow::new(&overlap, 30, 2, 6, 2, 11));
        expected_state.windows[3].push(OverlapWindow::new(&overlap, 35, 2, 11, 3, 0));

        for i in 0..state.windows.len() {
            assert_eq!(state.windows[i][0], expected_state.windows[i][0]);
        }
    }

    #[test]
    fn test_extract_windows4() {
        // reverse
        let target_seq = "AGCTAGTGTCAATGGCTACTTTTCAGGTCCT";
        let query_seq = "AACTAAGTGTCGGTGGCTACTATATATCAGGTCCT";
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

        let mut overlap = Overlap::new(0, q_len, 0, q_len, Strand::Reverse, 1, t_len, 0, t_len);
        overlap.cigar = cigar;
        let mut cigar_iter =
            get_cigar_iterator(overlap.cigar.as_ref().unwrap(), false, Strand::Reverse)
                .enumerate()
                .peekable();

        let mut state = WindowingState::new(target_seq.len(), WINDOW_SIZE);
        extract_windows(&mut state, &overlap, &mut cigar_iter, false, WINDOW_SIZE);
        for s in state.windows.iter() {
            for window in s {
                println!(
                    "{} {} {} {} {}",
                    window.qstart,
                    window.cigar_start_idx,
                    window.cigar_start_offset,
                    window.cigar_end_idx,
                    window.cigar_end_offset
                );
            }
        }
        let mut expected_state = WindowingState::new(target_seq.len(), WINDOW_SIZE);
        expected_state.windows[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 0, 5));
        expected_state.windows[1].push(OverlapWindow::new(&overlap, 5, 0, 5, 2, 0));
        expected_state.windows[2].push(OverlapWindow::new(&overlap, 10, 2, 0, 4, 1));
        expected_state.windows[3].push(OverlapWindow::new(&overlap, 12, 4, 1, 4, 6));
        expected_state.windows[4].push(OverlapWindow::new(&overlap, 17, 4, 6, 6, 1));
        expected_state.windows[5].push(OverlapWindow::new(&overlap, 22, 6, 1, 8, 0));
        expected_state.windows[6].push(OverlapWindow::new(&overlap, 26, 8, 0, 11, 0));

        for i in 0..state.windows.len() {
            assert_eq!(state.windows[i][0], expected_state.windows[i][0]);
        }
    }
}
