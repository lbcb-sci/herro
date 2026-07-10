use crate::{
    aligners::{CigarIter, CigarOp},
    overlaps::Overlap,
};

#[derive(Clone, Debug)]
pub struct OverlapWindow<'a> {
    pub overlap: &'a Overlap,
    pub tstart: u32,
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

pub(crate) fn extract_windows<'a>(
    windows: &mut Windows<'a>,
    overlap: &'a Overlap,
    cigar: &[u8],
    tshift: u32,
    qshift: u32,
    is_target: bool,
    window_size: u32,
) {
    let (ostart, oend, _olen) = if is_target {
        (overlap.tstart, overlap.tend, overlap.tlen)
    } else {
        (overlap.qstart, overlap.qend, overlap.qlen)
    };

    if oend == ostart {
        return;
    }

    let last_window = (oend + window_size - 1) / window_size;

    let mut tpos = ostart + tshift;
    let mut qpos = qshift;

    let mut t_window_start = Some(tpos);
    let mut q_window_start = Some(qpos);
    let mut cigar_start_idx = Some(0usize);
    let mut cigar_start_offset = Some(0u32);

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
        let diff_w = new_w - current_w;

        if diff_w == 0 {
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

            if cigar_start_idx.is_some() {
                windows[(current_w + i) as usize - 1].push(OverlapWindow::new(
                    overlap,
                    t_window_start.unwrap(),
                    q_window_start.unwrap(),
                    q_start_new,
                    cigar_start_idx.unwrap(),
                    cigar_start_offset.unwrap(),
                    range.end,
                    offset,
                ));
            }

            t_window_start = Some(tpos + offset);
            q_window_start = Some(q_start_new);
            cigar_start_idx = Some(range.start);
            cigar_start_offset = Some(offset);
        }

        // Handle the last window boundary
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
                cigar_end_idx = range_next.end;
                cigar_end_offset = *l;
            } else {
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
                t_window_start.unwrap(),
                q_window_start.unwrap(),
                qend,
                cigar_start_idx.unwrap(),
                cigar_start_offset.unwrap(),
                cigar_end_idx,
                cigar_end_offset,
            ));
        }

        t_window_start = Some(tpos + offset);
        q_window_start = Some(qend);
        cigar_start_idx = Some(next_cigar_start_idx);
        cigar_start_offset = Some(next_cigar_start_offset);

        tpos = tnew;
        qpos = qnew;
    }

    // End of the overlap, emitted already for oend % W = 0
    if tpos % window_size != 0 {
        windows[last_window as usize - 1].push(OverlapWindow::new(
            overlap,
            t_window_start.unwrap(),
            q_window_start.unwrap(),
            qpos,
            cigar_start_idx.unwrap(),
            cigar_start_offset.unwrap(),
            cigar.len(),
            get_last_cigar_op(&cigar).get_length(),
        ));
    }
}

pub(crate) fn filter_partial_overlap_windows<'a>(
    windows: &mut Windows<'a>,
    window_size: u32,
    rid: u32,
) {
    for (i, window_overlaps) in windows.iter_mut().enumerate() {
        window_overlaps.retain(|ow| !is_partial_overlap_window(ow, i, window_size, rid));
    }
}

fn is_partial_overlap_window(ow: &OverlapWindow, window_idx: usize, window_size: u32, rid: u32) -> bool {
    let is_target = ow.overlap.tid == rid;
    let (ostart, oend, olen) = if is_target {
        (ow.overlap.tstart, ow.overlap.tend, ow.overlap.tlen)
    } else {
        (ow.overlap.qstart, ow.overlap.qend, ow.overlap.qlen)
    };

    let zeroth_window_thresh = (0.1 * window_size as f32) as u32;
    let nth_window_thresh = olen - zeroth_window_thresh;
    let n_windows = (olen + window_size - 1) / window_size;

    // Partial first window: overlap starts inside window 0 (not on boundary) and too far from target start
    if window_idx == 0 && ostart % window_size != 0 && ostart >= zeroth_window_thresh {
        return true;
    }
    // Partial last window: overlap ends inside the last window (not on boundary) and too far from target end
    if window_idx + 1 == n_windows as usize && oend % window_size != 0 && oend <= nth_window_thresh {
        return true;
    }
    false
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlaps::{Overlap, Strand};

    fn make_overlap(
        tid: u32,
        tlen: u32,
        tstart: u32,
        tend: u32,
        qid: u32,
        qlen: u32,
        qstart: u32,
        qend: u32,
    ) -> Overlap {
        Overlap::new(qid, qlen, qstart, qend, Strand::Forward, tid, tlen, tstart, tend)
    }

    // ---- Group A: extract_windows now includes partial windows ----

    #[test]
    fn test_short_overlap_produces_window() {
        // Overlap length (3) < window_size (5): old code returned early, new code extracts.
        let window_size = 5u32;
        let overlap = make_overlap(1, 3, 0, 3, 0, 3, 0, 3);
        let cigar = b"3M";

        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);

        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[0][0].tstart, 0);
        assert_eq!(windows[0][0].qend, 3);
    }

    #[test]
    fn test_partial_first_window_included() {
        // Overlap starts at tstart=2, inside window 0 but not at its boundary.
        // Old code skipped window 0; new code must include it.
        let window_size = 5u32;
        let overlap = make_overlap(1, 20, 2, 20, 0, 18, 0, 18);
        let cigar = b"18M";

        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);

        assert_eq!(windows[0].len(), 1, "window 0 must have an entry");
        assert_eq!(windows[0][0].tstart, 2);
    }

    #[test]
    fn test_partial_last_window_included() {
        // Overlap ends at 17, inside window 3 (positions 15-19) but not at its boundary.
        // Old code omitted window 3; new code must include it.
        let window_size = 5u32;
        let overlap = make_overlap(1, 20, 0, 17, 0, 17, 0, 17);
        let cigar = b"17M";

        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);

        assert_eq!(windows[3].len(), 1, "window 3 must have an entry");
        assert_eq!(windows[3][0].tstart, 15);
        assert_eq!(windows[3][0].qend, 17);
    }

    // ---- Group B: filter_partial_overlap_windows ----

    #[test]
    fn test_filter_removes_partial_first_window() {
        // zeroth_window_thresh = (0.1 * 100) = 10; ostart=15 >= 10 → partial → removed
        let window_size = 100u32;
        let rid = 1;
        let overlap = make_overlap(rid, 200, 15, 200, 0, 185, 0, 185);
        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        windows[0].push(OverlapWindow::new(&overlap, 15, 0, 100, 0, 0, 5, 85));
        filter_partial_overlap_windows(&mut windows, window_size, rid);

        assert_eq!(windows[0].len(), 0);
    }

    #[test]
    fn test_filter_keeps_non_partial_first_window() {
        // ostart=5 < zeroth_window_thresh=10 → not partial → kept
        let window_size = 100u32;
        let rid = 1;
        let overlap = make_overlap(rid, 200, 5, 200, 0, 195, 0, 195);
        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        windows[0].push(OverlapWindow::new(&overlap, 5, 0, 100, 0, 0, 5, 95));
        filter_partial_overlap_windows(&mut windows, window_size, rid);

        assert_eq!(windows[0].len(), 1);
    }

    #[test]
    fn test_filter_removes_partial_last_window() {
        // nth_window_thresh=190; oend=90 <= 190 → partial → removed from last window
        let window_size = 100u32;
        let rid = 1;
        let overlap = make_overlap(rid, 200, 0, 90, 0, 90, 0, 90);
        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        windows[n_windows - 1].push(OverlapWindow::new(&overlap, 0, 0, 90, 0, 0, 5, 90));
        filter_partial_overlap_windows(&mut windows, window_size, rid);

        assert_eq!(windows[n_windows - 1].len(), 0);
    }

    #[test]
    fn test_filter_keeps_non_partial_last_window() {
        // oend=195 > nth_window_thresh=190 → not partial → kept
        let window_size = 100u32;
        let rid = 1;
        let overlap = make_overlap(rid, 200, 0, 195, 0, 195, 0, 195);
        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        windows[n_windows - 1].push(OverlapWindow::new(&overlap, 100, 0, 195, 0, 0, 5, 95));
        filter_partial_overlap_windows(&mut windows, window_size, rid);

        assert_eq!(windows[n_windows - 1].len(), 1);
    }

    #[test]
    fn test_filter_keeps_interior_window() {
        // Interior windows (not first, not last) are never filtered regardless of ostart/oend.
        let window_size = 100u32;
        let rid = 1;
        let overlap = make_overlap(rid, 300, 50, 200, 0, 150, 0, 150);
        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        // window index 1 is interior (not first=0, not last=2)
        windows[1].push(OverlapWindow::new(&overlap, 100, 0, 100, 0, 0, 5, 100));
        filter_partial_overlap_windows(&mut windows, window_size, rid);

        assert_eq!(windows[1].len(), 1);
    }

    #[test]
    fn test_filter_short_overlap_not_filtered() {
        // olen=50 < window_size=100: single window that is both first and last.
        // ostart=0 < zeroth_window_thresh=10 → window 0 kept.
        // oend=50 > nth_window_thresh=40 → last window kept.
        let window_size = 100u32;
        let rid = 1;
        let overlap = make_overlap(rid, 50, 0, 50, 0, 50, 0, 50);
        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        windows[0].push(OverlapWindow::new(&overlap, 0, 0, 50, 0, 0, 3, 50));
        filter_partial_overlap_windows(&mut windows, window_size, rid);

        assert_eq!(windows[0].len(), 1);
    }

    // ---- Group C: integration — extract + filter matches old monolithic behavior ----

    #[test]
    fn test_integration_full_coverage_no_filtering() {
        // Full-coverage overlap (tstart=0, tend=olen): extract + filter keeps all windows.
        let window_size = 5u32;
        let overlap = make_overlap(1, 20, 0, 20, 0, 20, 0, 20);
        let cigar = b"20M";
        let rid = 1;

        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        filter_partial_overlap_windows(&mut windows, window_size, rid);

        // All 4 windows (covering 0-4, 5-9, 10-14, 15-19) should be present.
        for (i, w) in windows.iter().enumerate() {
            assert_eq!(w.len(), 1, "window {i} should have an entry");
        }
    }

    #[test]
    fn test_integration_partial_first_filtered_out() {
        // Overlap starts at tstart=2 (inside window 0, not on boundary).
        // With window_size=5, zeroth_window_thresh=0: ostart=2 >= 0 and 2%5!=0 → partial → removed.
        let window_size = 5u32;
        let overlap = make_overlap(1, 20, 2, 20, 0, 18, 0, 18);
        let cigar = b"18M";
        let rid = 1;

        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        filter_partial_overlap_windows(&mut windows, window_size, rid);

        assert_eq!(windows[0].len(), 0, "partial window 0 must be filtered");
        assert_eq!(windows[1].len(), 1, "full window 1 must remain");
        assert_eq!(windows[2].len(), 1, "full window 2 must remain");
        assert_eq!(windows[3].len(), 1, "full window 3 must remain");
    }

    #[test]
    fn test_integration_partial_last_filtered_out() {
        // Overlap ends at 17, inside window 3 (15-19). With zeroth_window_thresh=0,
        // nth_window_thresh=20; oend=17 <= 20 and 17%5!=0 → partial last window removed.
        let window_size = 5u32;
        let overlap = make_overlap(1, 20, 0, 17, 0, 17, 0, 17);
        let cigar = b"17M";
        let rid = 1;

        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        filter_partial_overlap_windows(&mut windows, window_size, rid);

        assert_eq!(windows[0].len(), 1, "full window 0 must remain");
        assert_eq!(windows[1].len(), 1, "full window 1 must remain");
        assert_eq!(windows[2].len(), 1, "full window 2 must remain");
        assert_eq!(windows[3].len(), 0, "partial last window must be filtered");
    }

    #[test]
    fn test_integration_with_deletion() {
        // Overlap with a deletion: 5M3D10M (target length 18, query length 15).
        // Full coverage from 0 to 18 in a tlen=20 target.
        // With zeroth_window_thresh=0, nth_window_thresh=20:
        // ostart=0 (on boundary → not partial), oend=18 <= 20 and 18%5!=0 → last window partial.
        let window_size = 5u32;
        let overlap = make_overlap(1, 20, 0, 18, 0, 15, 0, 15);
        let cigar = b"5M3D10M";
        let rid = 1;

        let n_windows =
            (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];

        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        filter_partial_overlap_windows(&mut windows, window_size, rid);

        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_eq!(windows[2].len(), 1);
        assert_eq!(windows[3].len(), 0, "partial last window (18 of 20) must be filtered");
    }

    // ---- Group D: cigar idx/offset correctness ----

    fn assert_window(
        w: &OverlapWindow,
        tstart: u32,
        qstart: u32,
        qend: u32,
        cigar_start_idx: usize,
        cigar_start_offset: u32,
        cigar_end_idx: usize,
        cigar_end_offset: u32,
    ) {
        assert_eq!(w.tstart, tstart, "tstart mismatch");
        assert_eq!(w.qstart, qstart, "qstart mismatch");
        assert_eq!(w.qend, qend, "qend mismatch");
        assert_eq!(w.cigar_start_idx, cigar_start_idx, "cigar_start_idx mismatch");
        assert_eq!(w.cigar_start_offset, cigar_start_offset, "cigar_start_offset mismatch");
        assert_eq!(w.cigar_end_idx, cigar_end_idx, "cigar_end_idx mismatch");
        assert_eq!(w.cigar_end_offset, cigar_end_offset, "cigar_end_offset mismatch");
    }

    // T1: "10M" window_size=5 — single op split into two equal windows.
    // Both windows share the same cigar byte range (0..3) but at increasing offsets.
    // Window 1 is emitted via the "last window boundary" path, not the trailing path,
    // because tpos % window_size == 0 after the loop.
    #[test]
    fn test_cigar_single_op_two_exact_windows() {
        let window_size = 5u32;
        let overlap = make_overlap(1, 10, 0, 10, 0, 10, 0, 10);
        let cigar = b"10M"; // op0 Match(10): range 0..3
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_window(&windows[0][0], 0, 0, 5,  0, 0,  3, 5);
        assert_window(&windows[1][0], 5, 5, 10, 0, 5,  3, 10);
    }

    // T2: "15M" window_size=5 — single op split into three windows (diff_w=3).
    // Windows 0 and 1 come from the inner `for i in 1..diff_w` loop;
    // window 2 comes from the "last window boundary" block.
    // All three share op0's byte range (0..3).
    #[test]
    fn test_cigar_single_op_three_windows_diff_w_gt_1() {
        let window_size = 5u32;
        let overlap = make_overlap(1, 15, 0, 15, 0, 15, 0, 15);
        let cigar = b"15M"; // op0 Match(15): range 0..3
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_eq!(windows[2].len(), 1);
        assert_window(&windows[0][0], 0,  0,  5,  0, 0,  3, 5);
        assert_window(&windows[1][0], 5,  5,  10, 0, 5,  3, 10);
        assert_window(&windows[2][0], 10, 10, 15, 0, 10, 3, 15);
    }

    // T3: "7M" window_size=5 — single op, boundary mid-op.
    // Window 0 uses the main boundary path; window 1 is a trailing window
    // (tpos % window_size != 0), so cigar_end_idx = cigar.len() = 2 and
    // cigar_end_offset = get_last_cigar_op("7M").get_length() = 7.
    #[test]
    fn test_cigar_single_op_trailing_window() {
        let window_size = 5u32;
        let overlap = make_overlap(1, 7, 0, 7, 0, 7, 0, 7);
        let cigar = b"7M"; // op0 Match(7): range 0..2; cigar.len()=2
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_window(&windows[0][0], 0, 0, 5, 0, 0, 2, 5);
        assert_window(&windows[1][0], 5, 5, 7, 0, 5, 2, 7);
    }

    // T4: "5M5M" window_size=5 — window boundary falls exactly at the op seam.
    // Window 0: cigar_end_idx=2 (end of op0). Window 1: cigar_start_idx=2 (start of op1),
    // cigar_start_offset resets to 0.
    #[test]
    fn test_cigar_two_ops_boundary_at_op_seam() {
        let window_size = 5u32;
        let overlap = make_overlap(1, 10, 0, 10, 0, 10, 0, 10);
        let cigar = b"5M5M"; // op0 Match(5): 0..2, op1 Match(5): 2..4
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_window(&windows[0][0], 0, 0, 5,  0, 0, 2, 5);
        assert_window(&windows[1][0], 5, 5, 10, 2, 0, 4, 5);
    }

    // T5: "5M5D5M" window_size=5 — boundaries at Match→Deletion and Deletion→Match seams.
    // Window 1 covers the pure Deletion: qstart == qend because deletions don't advance query.
    #[test]
    fn test_cigar_match_deletion_match_exact_seams() {
        let window_size = 5u32;
        // tlen=15 (5+5+5 target advance); qlen=10 (5+0+5, deletion skips query)
        let overlap = make_overlap(1, 15, 0, 15, 0, 10, 0, 10);
        let cigar = b"5M5D5M"; // op0: 0..2, op1 Deletion(5): 2..4, op2: 4..6
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_eq!(windows[2].len(), 1);
        assert_window(&windows[0][0], 0,  0, 5,  0, 0, 2, 5);
        assert_window(&windows[1][0], 5,  5, 5,  2, 0, 4, 5); // qstart == qend: deletion
        assert_window(&windows[2][0], 10, 5, 10, 4, 0, 6, 5);
    }

    // T6: "3M7M" window_size=5 — boundary falls 2 bases into op1.
    // Window 0's cigar_end points to the END of op1 (idx=4) at offset 2, not to op0,
    // because the window boundary lies inside op1.
    // Window 1 starts mid-op1: cigar_start_idx=2 (range.start of op1), offset=2.
    #[test]
    fn test_cigar_boundary_mid_second_op() {
        let window_size = 5u32;
        let overlap = make_overlap(1, 10, 0, 10, 0, 10, 0, 10);
        let cigar = b"3M7M"; // op0 Match(3): 0..2, op1 Match(7): 2..4
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_window(&windows[0][0], 0, 0, 5,  0, 0, 4, 2);
        assert_window(&windows[1][0], 5, 5, 10, 2, 2, 4, 7);
    }

    // T7: "3M5D5M" window_size=5 — Deletion crosses a window boundary (mid-op split).
    // Window 0: ends 2 bases into the Deletion (cigar_end_idx=4, cigar_end_offset=2).
    // Window 1: starts mid-Deletion (idx=2=range.start, offset=2), ends 2 bases into op2.
    // Window 2 (trailing): starts mid-op2; cigar_end_idx=cigar.len()=6, cigar_end_offset=5.
    #[test]
    fn test_cigar_deletion_crosses_window_boundary() {
        let window_size = 5u32;
        // tlen=13 (3+5+5); qlen=8 (3+0+5)
        let overlap = make_overlap(1, 13, 0, 13, 0, 8, 0, 8);
        let cigar = b"3M5D5M"; // op0: 0..2, op1 Deletion(5): 2..4, op2: 4..6; cigar.len()=6
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_eq!(windows[2].len(), 1);
        assert_window(&windows[0][0], 0,  0, 3, 0, 0, 4, 2);
        assert_window(&windows[1][0], 5,  3, 5, 2, 2, 6, 2);
        assert_window(&windows[2][0], 10, 5, 8, 4, 2, 6, 5);
    }

    // T8: "8D" window_size=5 — pure Deletion spanning one boundary.
    // qstart == qend == 0 for both windows (deletion never advances query).
    // Trailing window: cigar_end_idx=cigar.len()=2, cigar_end_offset=get_last_op(8D)=8.
    #[test]
    fn test_cigar_pure_deletion_spanning_boundary() {
        let window_size = 5u32;
        let overlap = make_overlap(1, 8, 0, 8, 0, 0, 0, 0);
        let cigar = b"8D"; // op0 Deletion(8): 0..2; cigar.len()=2
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_window(&windows[0][0], 0, 0, 0, 0, 0, 2, 5);
        assert_window(&windows[1][0], 5, 0, 0, 0, 5, 2, 8);
    }

    // T9: "5M3I5M" window_size=5 — Insertion sits exactly at a window boundary.
    // The peek branch absorbs the Insertion into window 0:
    //   qend=8 (5 from Match + 3 from Insertion), cigar_end_idx=4 (end of Insertion op),
    //   cigar_end_offset=3 (Insertion length).
    // Window 1 starts right after the Insertion: cigar_start_idx=4 (start of second Match).
    #[test]
    fn test_cigar_insertion_at_window_boundary() {
        let window_size = 5u32;
        // tlen=10; qlen=5+3+5=13
        let overlap = make_overlap(1, 10, 0, 10, 0, 13, 0, 13);
        let cigar = b"5M3I5M"; // op0: 0..2, op1 Insertion(3): 2..4, op2: 4..6
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_window(&windows[0][0], 0, 0, 8,  0, 0, 4, 3);
        assert_window(&windows[1][0], 5, 8, 13, 4, 0, 6, 5);
    }

    // T10: "3M3I4M" window_size=5 — Insertion falls mid-window (not at a boundary).
    // The Insertion is consumed silently via `continue`; it shifts qpos but does NOT
    // trigger the peek logic, so cigar indices reference only the Match ops around it.
    #[test]
    fn test_cigar_insertion_mid_window_no_peek() {
        let window_size = 5u32;
        // tlen=7; qlen=3+3+4=10
        let overlap = make_overlap(1, 7, 0, 7, 0, 10, 0, 10);
        let cigar = b"3M3I4M"; // op0: 0..2, op1 Insertion(3): 2..4, op2 Match(4): 4..6
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 0, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        // qend=8: qpos after 3M=3, after Insertion=6, +2 bases into 4M = 8
        // cigar_end_idx=6 (end of op2), cigar_end_offset=2 (2 bases into 4M)
        assert_window(&windows[0][0], 0, 0, 8,  0, 0, 6, 2);
        // Trailing: cigar_end_idx=cigar.len()=6, cigar_end_offset=get_last_op(4M)=4
        assert_window(&windows[1][0], 5, 8, 10, 4, 2, 6, 4);
    }

    // T11: "10M" window_size=5 qshift=10 — non-zero qshift shifts all q-coordinates.
    // cigar idx/offsets are identical to T1; only qstart and qend differ.
    #[test]
    fn test_cigar_nonzero_qshift() {
        let window_size = 5u32;
        let overlap = make_overlap(1, 10, 0, 10, 0, 10, 0, 10);
        let cigar = b"10M";
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 0, 10, true, window_size);
        assert_eq!(windows[0].len(), 1);
        assert_eq!(windows[1].len(), 1);
        assert_window(&windows[0][0], 0, 10, 15, 0, 0, 3, 5);
        assert_window(&windows[1][0], 5, 15, 20, 0, 5, 3, 10);
    }

    // T12: "10M" window_size=5 tshift=5 — non-zero tshift places windows in higher slots.
    // tpos starts at ostart+tshift=5, so windows land in slots 1 and 2 of a tlen=20 vector.
    // cigar idx/offsets are identical to T1; tstart values reflect the shifted origin.
    #[test]
    fn test_cigar_nonzero_tshift() {
        let window_size = 5u32;
        // tlen=20 → 4 window slots; overlap covers 10 bases starting at tstart=0 with tshift=5
        let overlap = make_overlap(1, 20, 0, 10, 0, 10, 0, 10);
        let cigar = b"10M";
        let n_windows = (overlap.tlen as usize + window_size as usize - 1) / window_size as usize;
        let mut windows = vec![Vec::new(); n_windows];
        extract_windows(&mut windows, &overlap, cigar, 5, 0, true, window_size);
        assert_eq!(windows[0].len(), 0, "slot 0 must be empty");
        assert_eq!(windows[1].len(), 1);
        assert_eq!(windows[2].len(), 1);
        assert_eq!(windows[3].len(), 0, "slot 3 must be empty");
        assert_window(&windows[1][0], 5,  0, 5,  0, 0, 3, 5);
        assert_window(&windows[2][0], 10, 5, 10, 0, 5, 3, 10);
    }
}
