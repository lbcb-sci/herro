use std::ops::Range;
use std::u32;

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use crate::haec_io::BASE_ENCODING;
use crate::{
    aligners::{CigarIter, CigarOp},
    haec_io::HAECRecord,
    overlaps::{Alignment, Overlap, Strand},
};

#[derive(Clone, Debug)]
pub struct OverlapWindow<'a> {
    pub overlap: &'a Overlap,
    pub tstart: u32,
    pub tend: u32,
    pub qstart: u32,
    pub qend: u32,
    pub cigar_start_idx: usize,
    pub cigar_start_offset: u32,
    pub cigar_end_idx: usize,
    pub cigar_end_offset: u32,
}

impl<'a> OverlapWindow<'a> {
    pub fn new(
        overlap: &'a Overlap,
        tstart: u32,
        tend: u32,
        qstart: u32,
        qend: u32,
        cigar_start_idx: usize,
        cigar_start_offset: u32,
        cigar_end_idx: usize,
        cigar_end_offset: u32,
    ) -> Self {
        OverlapWindow {
            overlap,
            tstart,
            tend,
            qstart,
            qend,
            cigar_start_idx,
            cigar_start_offset,
            cigar_end_idx,
            cigar_end_offset,
        }
    }
}

type Windows<'a> = Vec<Vec<OverlapWindow<'a>>>;

pub(crate) fn extract_windows_new<'a>(
    alignments: &'a [Alignment],
    reads: &[HAECRecord],
    window_size: u32,
    (tbuf, qbuf): (&[u8], &mut [u8]),
) -> (Vec<AlnFeatsInfo<'a>>, HashMap<(u32, u16), [u32; 5]>) {
    let mut counts: HashMap<(u32, u16), [u32; 5]> = HashMap::default();

    let aln_feats_info = alignments
        .iter()
        .map(|a| {
            let aln_feats_info =
                extract_windows_from_alignment(a, &mut counts, reads, tbuf, qbuf, window_size);

            aln_feats_info
        })
        .collect::<Vec<_>>();

    (aln_feats_info, counts)
}

struct WinState {
    tpos: u32,               // tpos
    qpos: u32,               // qpos
    t_win_start: u32,        // t_win_start
    q_win_start: u32,        // q_win_start
    cigar_start_idx: usize,  // cigar_start_idx
    cigar_start_offset: u32, // cigar_start_offset
}

impl WinState {
    pub fn new(
        tpos: u32,
        qpos: u32,
        t_win_start: u32,
        q_win_start: u32,
        cigar_start_idx: usize,
        cigar_start_offset: u32,
    ) -> Self {
        WinState {
            tpos,
            qpos,
            t_win_start,
            q_win_start,
            cigar_start_idx,
            cigar_start_offset,
        }
    }
}

pub(crate) struct AlnFeatsInfo<'a> {
    pub windows: HashMap<u32, OverlapWindow<'a>>,
    pub first_win_tstart: u32,
    pub last_win_tend: u32,
    pub diffs: HashSet<(u32, u16)>,
}

impl<'a> AlnFeatsInfo<'a> {
    pub fn new(
        windows: HashMap<u32, OverlapWindow<'a>>,
        first_win_tstart: u32,
        last_win_tend: u32,
        diffs: HashSet<(u32, u16)>,
    ) -> Self {
        AlnFeatsInfo {
            windows,
            first_win_tstart,
            last_win_tend,
            diffs,
        }
    }
}

fn extract_windows_from_alignment<'a>(
    alignment: &'a Alignment,
    counts: &mut HashMap<(u32, u16), [u32; 5]>,
    reads: &[HAECRecord],
    tbuf: &[u8],
    qbuf: &mut [u8],
    window_size: u32,
) -> AlnFeatsInfo<'a> {
    let mut wins = HashMap::default();
    let mut diffs = HashSet::default();

    let mut win_state = WinState::new(
        alignment.overlap.tstart,
        0,
        u32::MAX,
        u32::MAX,
        usize::MAX,
        u32::MAX,
    );

    // Load query into buffer
    if alignment.overlap.strand == Strand::Forward {
        reads[alignment.overlap.qid as usize].seq.get_subseq(
            alignment.overlap.qstart as usize..alignment.overlap.qend as usize,
            qbuf,
        );
    } else {
        reads[alignment.overlap.qid as usize].seq.get_rc_subseq(
            alignment.overlap.qstart as usize..alignment.overlap.qend as usize,
            qbuf,
        );
    }

    let zeroth_window_thresh = (0.1 * window_size as f32) as u32;
    let last_window_thresh = alignment.overlap.tlen - zeroth_window_thresh;

    let mut first_win_tstart = u32::MAX;
    let mut last_win_tend = u32::MAX;
    if win_state.tpos % window_size == 0 || win_state.tpos < zeroth_window_thresh {
        win_state.t_win_start = win_state.tpos;
        win_state.q_win_start = 0;
        win_state.cigar_start_idx = 0;
        win_state.cigar_start_offset = 0;
    }

    let mut cigar_iter = CigarIter::new(&alignment.cigar).peekable();
    while let Some((op, range)) = cigar_iter.next() {
        match op {
            CigarOp::Match(l) => {
                for i in 0..l {
                    let tbase = tbuf[(win_state.tpos + i) as usize];
                    let qbase = qbuf[(win_state.qpos + i) as usize];
                    counts.entry((win_state.tpos + i, 0)).or_default()
                        [BASE_ENCODING[qbase as usize] as usize] += 1;

                    if tbase != qbase {
                        diffs.insert((win_state.tpos + i, 0));
                    };

                    if (win_state.tpos + i + 1) % window_size == 0 {
                        if let Some(ow) = emit_window_check(
                            i,
                            op,
                            &range,
                            cigar_iter.peek(),
                            &mut win_state,
                            &alignment.overlap,
                        ) {
                            if first_win_tstart == u32::MAX {
                                first_win_tstart = ow.tstart;
                            }
                            last_win_tend = ow.tend;

                            wins.insert(win_state.tpos / window_size, ow);
                        }
                    }
                }

                win_state.tpos += l;
                win_state.qpos += l;
            }
            CigarOp::Deletion(l) => {
                for i in 0..l {
                    counts.entry((win_state.tpos + i, 0)).or_default()[4] += 1;
                    diffs.insert((win_state.tpos + i, 0));

                    if (win_state.tpos + i + 1) % window_size == 0 {
                        if let Some(ow) = emit_window_check(
                            i,
                            op,
                            &range,
                            cigar_iter.peek(),
                            &mut win_state,
                            &alignment.overlap,
                        ) {
                            if first_win_tstart == u32::MAX {
                                first_win_tstart = win_state.t_win_start;
                            }
                            last_win_tend = ow.tend;

                            wins.insert(win_state.tpos / window_size, ow);
                        }
                    }
                }

                win_state.tpos += l;
            }
            CigarOp::Insertion(l) => {
                // TODO indels >= 256
                for ins in 0..l as u16 {
                    let qbase = qbuf[(win_state.qpos + ins as u32) as usize];
                    counts.entry((win_state.tpos, ins)).or_default()
                        [BASE_ENCODING[qbase as usize] as usize] += 1;

                    diffs.insert((win_state.tpos, ins as u16));
                }

                if win_state.tpos % window_size == 0 {
                    if let Some(ow) = emit_window_check(
                        l - 1,
                        op,
                        &range,
                        cigar_iter.peek(),
                        &mut win_state,
                        &alignment.overlap,
                    ) {
                        if first_win_tstart == u32::MAX {
                            first_win_tstart = win_state.t_win_start;
                        }
                        last_win_tend = ow.tend;

                        wins.insert((win_state.tpos - 1) / window_size, ow);
                    }
                }

                win_state.qpos += l;
            }
            _ => panic!("Invalid CIGAR operation encountered"),
        }
    }

    if win_state.tpos > last_window_thresh && win_state.tpos % window_size != 0 {
        let ow = OverlapWindow::new(
            &alignment.overlap,
            win_state.t_win_start,
            win_state.tpos,
            win_state.q_win_start,
            win_state.qpos,
            win_state.cigar_start_idx,
            win_state.cigar_start_offset,
            alignment.cigar.len(),
            get_last_cigar_op(&alignment.cigar).get_length(),
        );

        if first_win_tstart == u32::MAX {
            first_win_tstart = win_state.t_win_start;
        }
        last_win_tend = ow.tend;

        wins.insert(win_state.tpos / window_size, ow);
    }

    assert!(win_state.tpos == alignment.overlap.tend);
    assert!(win_state.qpos == alignment.overlap.qend - alignment.overlap.qstart);

    AlnFeatsInfo::new(wins, first_win_tstart, last_win_tend, diffs)
}

fn emit_window_check<'a>(
    curr_idx: u32,
    curr_op: CigarOp,
    range: &Range<usize>,
    next_op: Option<&(CigarOp, Range<usize>)>,
    win_state: &mut WinState,
    overlap: &'a Overlap,
) -> Option<OverlapWindow<'a>> {
    let op_len = curr_op.get_length() as u32;
    let is_ins_next = next_op.map_or(false, |(op, _)| matches!(op, CigarOp::Insertion(_)));

    if curr_idx + 1 == op_len && is_ins_next {
        // Emit the window in the insertion block
        return None;
    }

    let (t_win_end, q_win_end) = match curr_op {
        CigarOp::Match(_) => (win_state.tpos + curr_idx + 1, win_state.qpos + curr_idx + 1),
        CigarOp::Insertion(_) => (win_state.tpos, win_state.qpos + curr_idx + 1),
        CigarOp::Deletion(_) => (win_state.tpos + curr_idx + 1, win_state.qpos),
        CigarOp::Mismatch(_) => panic!("Mismatch not supported in emit_window"),
    };

    let (cigar_end_idx, cigar_end_offset) = if curr_idx + 1 == op_len {
        (range.end, op_len as u32)
    } else {
        (range.end, curr_idx + 1)
    };

    // Only emit if the window has a valid start
    let ow = if win_state.cigar_start_idx != usize::MAX {
        Some(OverlapWindow::new(
            overlap,
            win_state.t_win_start,
            t_win_end,
            win_state.q_win_start,
            q_win_end,
            win_state.cigar_start_idx,
            win_state.cigar_start_offset,
            cigar_end_idx,
            cigar_end_offset,
        ))
    } else {
        None
    };

    win_state.t_win_start = t_win_end;
    win_state.q_win_start = q_win_end;
    if curr_idx + 1 == op_len {
        win_state.cigar_start_idx = cigar_end_idx;
        win_state.cigar_start_offset = 0;
    } else {
        win_state.cigar_start_idx = range.start;
        win_state.cigar_start_offset = cigar_end_offset;
    };

    ow
}

/*
pub(crate) fn extract_windows<'a>(
    windows: &mut Windows<'a>,
    overlap: &'a Overlap,
    cigar: &[u8],
    tshift: u32,
    qshift: u32,
    is_target: bool,
    window_size: u32,
) {
    if (is_target && (overlap.tend - overlap.tstart) < window_size)
        || ((overlap.qend - overlap.qstart) < window_size)
    {
        return;
    }

    let first_window;
    let last_window; // Should be exclusive
    let tstart;
    let mut tpos;
    let mut qpos = 0;

    let zeroth_window_thresh = (0.1 * window_size as f32) as u32;
    let nth_window_thresh = if is_target {
        overlap.tlen - zeroth_window_thresh
    } else {
        overlap.qlen - zeroth_window_thresh
    };

    // Read is the target -> tstart
    // Read is the query  -> qstart
    if is_target {
        first_window = if overlap.tstart < zeroth_window_thresh {
            0
        } else {
            (overlap.tstart + window_size - 1) / window_size
        };

        last_window = if overlap.tend > nth_window_thresh {
            (overlap.tend - 1) / window_size + 1
        } else {
            overlap.tend / window_size
        };

        tstart = overlap.tstart;
        tpos = overlap.tstart;
    } else {
        first_window = if overlap.qstart < zeroth_window_thresh {
            0
        } else {
            (overlap.qstart + window_size - 1) / window_size
        };

        last_window = if overlap.qend > nth_window_thresh {
            (overlap.qend - 1) / window_size + 1
        } else {
            overlap.qend / window_size
        };

        tstart = overlap.qstart;
        tpos = overlap.qstart;
    }

    if last_window - first_window < 1 {
        return;
    }

    let mut t_window_start = None;
    let mut q_window_start = None;
    let mut cigar_start_idx = None;
    let mut cigar_start_offset = None;

    // Shift target or query due to cigar fixing
    tpos += tshift;
    qpos += qshift;

    // Start of the window OR beginning of the target
    if tpos % window_size == 0 || tstart < zeroth_window_thresh {
        t_window_start = Some(tpos);
        q_window_start = Some(qpos);
        cigar_start_idx = Some(0);
        cigar_start_offset = Some(0);
    }

    let mut cigar_iter = CigarIter::new(cigar).peekable();
    while let Some((op, range)) = cigar_iter.next() {
        let (tnew, qnew) = match op {
            CigarOp::Match(l) | CigarOp::Mismatch(l) => (tpos + l, qpos + l),
            CigarOp::Deletion(l) => (tpos + l, qpos),
            CigarOp::Insertion(l) => {
                qpos += l;
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

            let q_start_new = if let CigarOp::Match(_) | CigarOp::Mismatch(_) = op {
                qpos + offset
            } else {
                qpos
            };

            // If there was full window -> emit it, else label start
            if cigar_start_idx.is_some() {
                windows[(current_w + i) as usize - 1].push(OverlapWindow::new(
                    overlap,
                    q_window_start.unwrap(),
                    q_start_new,
                    cigar_start_idx.unwrap(),
                    cigar_start_offset.unwrap(),
                    range.end,
                    offset,
                ));

                t_window_start.replace(tpos + offset);

                //handle qpos
                if let CigarOp::Match(_) | CigarOp::Mismatch(_) = op {
                    q_window_start.replace(qpos + offset);
                } else {
                    q_window_start.replace(qpos);
                }

                cigar_start_idx.replace(range.start);
                cigar_start_offset.replace(offset);
            } else {
                t_window_start = Some(tpos + offset);

                if let CigarOp::Match(_) | CigarOp::Mismatch(_) = op {
                    q_window_start = Some(qpos + offset);
                } else {
                    q_window_start = Some(qpos);
                }

                cigar_start_idx = Some(range.start);
                cigar_start_offset = Some(offset)
            }
        }

        // Handle the last one
        let offset = new_w * window_size - tpos;

        let mut qend = if let CigarOp::Match(_) | CigarOp::Mismatch(_) = op {
            qpos + offset
        } else {
            qpos
        };

        let cigar_end_idx;
        let cigar_end_offset;
        let next_cigar_start_idx;
        let next_cigar_start_offset;
        if tnew == new_w * window_size {
            if let Some((CigarOp::Insertion(l), range_next)) = cigar_iter.peek() {
                qend += *l;
                //cigar_end_idx = cigar_idx + 2;
                cigar_end_idx = range_next.end;
                cigar_end_offset = *l;
            } else {
                //cigar_end_idx = cigar_idx + 1;
                cigar_end_idx = range.end;
                cigar_end_offset = op.get_length();
            }

            next_cigar_start_idx = cigar_end_idx;
            next_cigar_start_offset = 0;
        } else {
            cigar_end_idx = range.end;
            cigar_end_offset = offset;

            next_cigar_start_idx = range.start;
            next_cigar_start_offset = cigar_end_offset;
        }

        if cigar_start_idx.is_some() {
            windows[new_w as usize - 1].push(OverlapWindow::new(
                overlap,
                q_window_start.unwrap(),
                qend,
                cigar_start_idx.unwrap(),
                cigar_start_offset.unwrap(),
                cigar_end_idx,
                cigar_end_offset,
            ));

            t_window_start.replace(tpos + offset);
            q_window_start.replace(qend);

            cigar_start_idx.replace(next_cigar_start_idx);
            cigar_start_offset.replace(next_cigar_start_offset);
        } else {
            t_window_start = Some(tpos + offset);
            q_window_start = Some(qend);
            cigar_start_idx = Some(next_cigar_start_idx);
            cigar_start_offset = Some(next_cigar_start_offset);
        }

        tpos = tnew;
        qpos = qnew;
    }

    // End of the target, emitted already for tlen % W = 0
    if tpos > nth_window_thresh && tpos % window_size != 0 {
        windows[last_window as usize - 1].push(OverlapWindow::new(
            overlap,
            q_window_start.unwrap(),
            qpos,
            cigar_start_idx.unwrap(),
            cigar_start_offset.unwrap(),
            cigar.len(),
            get_last_cigar_op(&cigar).get_length(),
        ));
    }
}*/

fn get_last_cigar_op(cigar: &[u8]) -> CigarOp {
    let op = cigar[cigar.len() - 1];

    let len = cigar[..cigar.len() - 1]
        .iter()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .enumerate()
        .fold(0, |len, (i, c)| {
            len + (c - b'0') as u32 * 10u32.pow(i as u32)
        });

    match op {
        b'M' => CigarOp::Match(len),
        b'I' => CigarOp::Insertion(len),
        b'D' => CigarOp::Deletion(len),
        _ => panic!("Invalid cigar op"),
    }
}

/*
#[cfg(test)]
mod tests {

    const WINDOW_SIZE: u32 = 5;

    use super::{extract_windows, OverlapWindow};

    use crate::{
        aligners::wfa::{WFAAlignerBuilder, WFADistanceMetric},
        features::get_cigar_iterator,
        overlaps::{Overlap, Strand},
    };

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

        let n_windows = (target_seq.len() + WINDOW_SIZE as usize - 1) / WINDOW_SIZE as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, &mut cigar_iter, true, WINDOW_SIZE);

        let mut expected_windows = vec![Vec::new(); n_windows];
        expected_windows[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 4, 0));
        expected_windows[1].push(OverlapWindow::new(&overlap, 6, 4, 0, 5, 0));
        expected_windows[2].push(OverlapWindow::new(&overlap, 11, 5, 0, 6, 3));
        expected_windows[3].push(OverlapWindow::new(&overlap, 16, 6, 3, 8, 0));
        expected_windows[4].push(OverlapWindow::new(&overlap, 22, 8, 0, 12, 3));
        expected_windows[5].push(OverlapWindow::new(&overlap, 29, 12, 3, 12, 8));
        expected_windows[6].push(OverlapWindow::new(&overlap, 34, 12, 8, 13, 0));

        for i in 0..windows.len() {
            assert_eq!(windows[i][0], expected_windows[i][0]);
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

        let n_windows = (target_seq.len() + WINDOW_SIZE as usize - 1) / WINDOW_SIZE as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, &mut cigar_iter, true, WINDOW_SIZE);

        let mut expected_windows = vec![Vec::new(); n_windows];
        expected_windows[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 2, 1));
        expected_windows[1].push(OverlapWindow::new(&overlap, 3, 2, 1, 2, 6));
        expected_windows[2].push(OverlapWindow::new(&overlap, 8, 2, 6, 2, 11));
        expected_windows[3].push(OverlapWindow::new(&overlap, 13, 2, 11, 2, 16));
        expected_windows[4].push(OverlapWindow::new(&overlap, 18, 2, 16, 3, 1));
        expected_windows[5].push(OverlapWindow::new(&overlap, 23, 3, 1, 5, 0));

        for i in 0..windows.len() {
            assert_eq!(windows[i][0], expected_windows[i][0]);
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

        let n_windows = (target_seq.len() + WINDOW_SIZE as usize - 1) / WINDOW_SIZE as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, &mut cigar_iter, true, WINDOW_SIZE);

        let mut expected_windows = vec![Vec::new(); n_windows];
        expected_windows[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 2, 1));
        expected_windows[1].push(OverlapWindow::new(&overlap, 25, 2, 1, 2, 6));
        expected_windows[2].push(OverlapWindow::new(&overlap, 30, 2, 6, 2, 11));
        expected_windows[3].push(OverlapWindow::new(&overlap, 35, 2, 11, 3, 0));

        for i in 0..windows.len() {
            assert_eq!(windows[i][0], expected_windows[i][0]);
        }
    }

    #[test]
    fn test_extract_windows4() {
        // reverse + not target
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

        let n_windows = (target_seq.len() + WINDOW_SIZE as usize - 1) / WINDOW_SIZE as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, &mut cigar_iter, false, WINDOW_SIZE);

        let mut expected_state = vec![Vec::new(); n_windows];
        expected_state[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 0, 5));
        expected_state[1].push(OverlapWindow::new(&overlap, 5, 0, 5, 2, 0));
        expected_state[2].push(OverlapWindow::new(&overlap, 10, 2, 0, 4, 1));
        expected_state[3].push(OverlapWindow::new(&overlap, 12, 4, 1, 4, 6));
        expected_state[4].push(OverlapWindow::new(&overlap, 17, 4, 6, 6, 1));
        expected_state[5].push(OverlapWindow::new(&overlap, 22, 6, 1, 8, 0));
        expected_state[6].push(OverlapWindow::new(&overlap, 26, 8, 0, 11, 0));

        for i in 0..windows.len() {
            assert_eq!(windows[i][0], expected_state[i][0]);
        }
    }

    #[test]
    fn test_extract_windows5() {
        // reverse + target
        let query_seq = "AACTAAGTGTCGGTGGCTACTATATATCAGGTCCT";
        let target_seq = "AGCTAGTGTCAATGGCTACTTTTCAGGTCCT";
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
            get_cigar_iterator(overlap.cigar.as_ref().unwrap(), true, Strand::Reverse)
                .enumerate()
                .peekable();

        let n_windows = (target_seq.len() + WINDOW_SIZE as usize - 1) / WINDOW_SIZE as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, &mut cigar_iter, true, WINDOW_SIZE);

        let mut expected_state = vec![Vec::new(); n_windows];
        expected_state[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 4, 0));
        expected_state[1].push(OverlapWindow::new(&overlap, 6, 4, 0, 5, 0));
        expected_state[2].push(OverlapWindow::new(&overlap, 11, 5, 0, 6, 3));
        expected_state[3].push(OverlapWindow::new(&overlap, 16, 6, 3, 8, 0));
        expected_state[4].push(OverlapWindow::new(&overlap, 24, 8, 0, 10, 3));
        expected_state[5].push(OverlapWindow::new(&overlap, 29, 10, 3, 10, 8));
        expected_state[6].push(OverlapWindow::new(&overlap, 34, 10, 8, 11, 0));

        for i in 0..windows.len() {
            assert_eq!(windows[i][0], expected_state[i][0]);
        }
    }

    #[test]
    fn test_extract_windows6() {
        // forward + not target
        let query_seq = "AACTAAGTGTCGGTGGCTACTATATATCAGGTCCT";
        let target_seq = "AGCTAGTGTCAATGGCTACTTTTCAGGTCCT";
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
            get_cigar_iterator(overlap.cigar.as_ref().unwrap(), false, Strand::Forward)
                .enumerate()
                .peekable();

        let n_windows = (target_seq.len() + WINDOW_SIZE as usize - 1) / WINDOW_SIZE as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, &mut cigar_iter, false, WINDOW_SIZE);

        let mut expected_state = vec![Vec::new(); n_windows];
        expected_state[0].push(OverlapWindow::new(&overlap, 0, 0, 0, 3, 0));
        expected_state[1].push(OverlapWindow::new(&overlap, 5, 3, 0, 4, 4));
        expected_state[2].push(OverlapWindow::new(&overlap, 9, 4, 4, 6, 2));
        expected_state[3].push(OverlapWindow::new(&overlap, 14, 6, 2, 6, 7));
        expected_state[4].push(OverlapWindow::new(&overlap, 19, 6, 7, 9, 0));
        expected_state[5].push(OverlapWindow::new(&overlap, 21, 9, 0, 10, 4));
        expected_state[6].push(OverlapWindow::new(&overlap, 26, 10, 4, 11, 0));

        for i in 0..windows.len() {
            assert_eq!(windows[i][0], expected_state[i][0]);
        }
    }

    #[test]
    fn test_extract_windows7() {
        // ends are not 0 or len
        let query_seq = "AACTAAGTGTCGGTGGCTACTATATATCAGGTCCT";
        let target_seq = "TTTTTAGCTAGTGTCAATGGCTACTTTTCAGGTCCT";
        let q_len = query_seq.len() as u32;
        let t_len = target_seq.len() as u32;
        let overlap_target = "AGCTAGTGTCAATGGCTACTTTTCAGGTCCT"; // target starts at pos 5

        let mut builder = WFAAlignerBuilder::new();
        builder.set_distance_metric(WFADistanceMetric::GapAffine {
            match_: (0),
            mismatch: (6),
            gap_opening: (4),
            gap_extension: (2),
        });
        let aligner = builder.build();
        let cigar = aligner.align(query_seq, overlap_target);
        println!("{:?}", cigar);

        let mut overlap = Overlap::new(0, q_len, 0, q_len, Strand::Forward, 1, t_len, 5, t_len);
        overlap.cigar = cigar;
        let mut cigar_iter =
            get_cigar_iterator(overlap.cigar.as_ref().unwrap(), true, Strand::Forward)
                .enumerate()
                .peekable();

        let n_windows = (target_seq.len() + WINDOW_SIZE as usize - 1) / WINDOW_SIZE as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, &mut cigar_iter, true, WINDOW_SIZE);

        let mut expected_state = vec![Vec::new(); n_windows];
        expected_state[1].push(OverlapWindow::new(&overlap, 0, 0, 0, 4, 0));
        expected_state[2].push(OverlapWindow::new(&overlap, 6, 4, 0, 5, 0));
        expected_state[3].push(OverlapWindow::new(&overlap, 11, 5, 0, 6, 3));
        expected_state[4].push(OverlapWindow::new(&overlap, 16, 6, 3, 8, 0));
        expected_state[5].push(OverlapWindow::new(&overlap, 24, 8, 0, 10, 3));
        expected_state[6].push(OverlapWindow::new(&overlap, 29, 10, 3, 10, 8));
        expected_state[7].push(OverlapWindow::new(&overlap, 34, 10, 8, 11, 0));

        println!("{}", windows.len());
        for i in 1..windows.len() {
            assert_eq!(windows[i][0], expected_state[i][0]);
        }
    }
}
*/
