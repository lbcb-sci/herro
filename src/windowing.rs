use crate::{aligners::CigarOp, overlaps::Overlap};

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct OverlapWindow<'a> {
    pub overlap: &'a Overlap,
    pub tstart: u32,
    pub qstart: u32,
    pub cigar_start_idx: usize,
    pub cigar_start_offset: u32,
    pub cigar_end_idx: usize,
    pub cigar_end_offset: u32,
}

impl<'a> OverlapWindow<'a> {
    pub fn new(
        overlap: &'a Overlap,
        tstart: u32,
        qstart: u32,
        cigar_start_idx: usize,
        cigar_start_offset: u32,
        cigar_end_idx: usize,
        cigar_end_offset: u32,
    ) -> Self {
        OverlapWindow {
            overlap,
            tstart,
            qstart,
            cigar_start_idx,
            cigar_start_offset,
            cigar_end_idx,
            cigar_end_offset,
        }
    }
}

type Windows<'a> = Vec<Vec<OverlapWindow<'a>>>;

pub(crate) fn extract_windows<'a, 'b>(
    windows: &mut Windows<'a>,
    overlap: &'a Overlap,
    cigar: &[CigarOp],
    tshift: u32,
    qshift: u32,
    is_target: bool,
    window_size: u32,
) {
    if is_target && (overlap.tend - overlap.tstart) < window_size {
        return;
    } else if (overlap.qend - overlap.qstart) < window_size {
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

    let mut cigar_iter = cigar.iter().enumerate().peekable();
    while let Some((cigar_idx, op)) = cigar_iter.next() {
        let (tnew, qnew) = match op {
            CigarOp::Match(l) | CigarOp::Mismatch(l) => (tpos + *l, qpos + *l),
            CigarOp::Deletion(l) => (tpos + *l, qpos),
            CigarOp::Insertion(l) => {
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
                windows[(current_w + i) as usize - 1].push(OverlapWindow::new(
                    overlap,
                    t_window_start.unwrap(),
                    q_window_start.unwrap(),
                    cigar_start_idx.unwrap(),
                    cigar_start_offset.unwrap(),
                    cigar_idx,
                    offset,
                ));

                t_window_start.replace(tpos + offset);

                //handle qpos
                if let CigarOp::Match(_) | CigarOp::Mismatch(_) = op {
                    q_window_start.replace(qpos + offset);
                } else {
                    q_window_start.replace(qpos);
                }

                cigar_start_idx.replace(cigar_idx);
                cigar_start_offset.replace(offset);
            } else {
                t_window_start = Some(tpos + offset);

                if let CigarOp::Match(_) | CigarOp::Mismatch(_) = op {
                    q_window_start = Some(qpos + offset);
                } else {
                    q_window_start = Some(qpos);
                }

                cigar_start_idx = Some(cigar_idx);
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
        if tnew == new_w * window_size {
            if let Some(CigarOp::Insertion(l)) = cigar_iter.peek().map(|(_, op)| op) {
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
            windows[new_w as usize - 1].push(OverlapWindow::new(
                overlap,
                t_window_start.unwrap(),
                q_window_start.unwrap(),
                cigar_start_idx.unwrap(),
                cigar_start_offset.unwrap(),
                cigar_end_idx,
                cigar_end_offset,
            ));

            t_window_start.replace(tpos + offset);
            q_window_start.replace(qend);
            cigar_start_idx.replace(cigar_end_idx);
            cigar_start_offset.replace(cigar_end_offset);
        } else {
            t_window_start = Some(tpos + offset);
            q_window_start = Some(qend);
            cigar_start_idx = Some(cigar_end_idx);
            cigar_start_offset = Some(cigar_end_offset);
        }

        tpos = tnew;
        qpos = qnew;
    }

    // End of the target, emitted already for tlen % W = 0
    if tpos > nth_window_thresh && tpos % window_size != 0 {
        windows[last_window as usize - 1].push(OverlapWindow::new(
            overlap,
            t_window_start.unwrap(),
            q_window_start.unwrap(),
            cigar_start_idx.unwrap(),
            cigar_start_offset.unwrap(),
            cigar.len(),
            0,
        ));
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
