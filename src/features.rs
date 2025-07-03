use npyz::WriterBuilder;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::cmp::Reverse;
use std::fs::{create_dir_all, File};
use std::hash::Hash;
use std::io::prelude::*;
use std::io::{BufWriter, Result};
use std::path::Path;

use crossbeam_channel::Sender;

use ndarray::{s, stack, Array, Array2, ArrayBase, ArrayViewMut1, Axis, Data, Ix2};
use ordered_float::OrderedFloat;

use crate::aligners::{CigarIter, CigarOp};
use crate::haec_io::HAECRecord;
use crate::inference::{prepare_examples, InferenceData, WindowExample};
use crate::overlaps::{Alignment, Strand};
use crate::pbars::PBarNotification;
use crate::windowing::extract_windows_new;
use crate::windowing::AlnFeatsInfo;
use crate::windowing::OverlapWindow;

pub(crate) const TOP_K_SORT: usize = 30;
const REL_FILTER_THRESHOLD: f32 = 0.1;

const BASE_LOWER: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 97, 255, 99, 255, 255, 255, 103, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 116, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

const BASE_FORWARD: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 42, 255, 255,
    255, 255, 255, 255, 42, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 65, 255, 67, 255, 255, 255, 71, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 84, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 65, 255, 67, 255, 255, 255, 71, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 84, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

fn get_max_ins_for_window(
    overlaps: &[OverlapWindow], // Sorted overlaps
    tstart: usize,
    window_length: usize,
) -> Vec<u16> {
    let mut max_ins = vec![0; window_length];
    for ow in overlaps.iter() {
        let mut tpos = ow.tstart as usize - tstart;

        // Handle cigar
        let cigar = &ow.alignment.cigar;
        //let cigar_len = ow.cigar_end_idx - ow.cigar_start_idx + 1;
        let cigar_iter = CigarIter::new(&cigar[ow.cigar_start_idx..ow.cigar_end_idx]);

        cigar_iter.for_each(|(op, range)| {
            let l = match op {
                CigarOp::Match(l) | CigarOp::Mismatch(l) | CigarOp::Deletion(l) => l as usize,
                CigarOp::Insertion(l) => {
                    assert!(
                        tpos <= max_ins.len(),
                        "Window start {}, target start {} - this is bigger than the tseq {}. {} {} {} {:?} {:?}",
                        ow.tstart,
                        tstart,
                        max_ins.len(),
                        ow.cigar_start_offset,
                        op.get_length(),
                        std::str::from_utf8(cigar).unwrap(),
                        range,
                        ow.cigar_start_idx..ow.cigar_end_idx
                    );

                    max_ins[tpos - 1] = max_ins[tpos - 1].max(l as u16);
                    return;
                }
            };

            if range == (0..ow.cigar_end_idx - ow.cigar_start_idx) {
                tpos += (ow.cigar_end_offset - ow.cigar_start_offset) as usize;
            } else if range.start == 0 {
                tpos += l - ow.cigar_start_offset as usize;
            } else if range.end == ow.cigar_end_idx - ow.cigar_start_idx {
                tpos += ow.cigar_end_offset as usize;
            } else {
                tpos += l;
            }
        });
    }

    max_ins
}

fn get_features_for_ol_window(
    mut bases: ArrayViewMut1<'_, u8>,
    mut quals: ArrayViewMut1<'_, u8>,
    window: &OverlapWindow,
    cigar: &[u8],
    query: &HAECRecord,
    offset: usize,
    max_ins: &[u16],
    qbuffer: &mut [u8],
) {
    // Handle query sequence
    let qstart = window.alignment.overlap.qstart;
    let qend = window.alignment.overlap.qend;
    let strand = window.alignment.overlap.strand;

    let mut query_iter: Box<dyn DoubleEndedIterator<Item = (&u8, &u8)>> = match strand {
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
    //let cigar_len = window.cigar_end_idx - window.cigar_start_idx + 1;
    //let cigar_end = cigar.len().min((window.cigar_end_idx + 1) as usize);

    // Handle cigar
    //let cigar = cigar[window.cigar_start_idx as usize..cigar_end].iter();
    let cigar = CigarIter::new(&cigar[window.cigar_start_idx..window.cigar_end_idx]);

    // Get features
    let gap = if let Strand::Forward = strand {
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

    cigar.for_each(|(op, range)| {
        let mut l = match op {
            CigarOp::Match(l)
            | CigarOp::Mismatch(l)
            | CigarOp::Deletion(l)
            | CigarOp::Insertion(l) => l as usize,
        };

        // Calculate true length
        if (0..window.cigar_end_idx - window.cigar_start_idx) == range {
            l = (window.cigar_end_offset - window.cigar_start_offset) as usize;
        } else if range.start == 0 {
            l -= window.cigar_start_offset as usize;
        } else if range.end == window.cigar_end_idx - window.cigar_start_idx {
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
    tbuffer: &[u8],
) {
    bases.fill(b'*'); // Fill like forward

    /*let tlen = tstart + window_length - tstart;
    target
        .seq
        .get_subseq(tstart..tstart + window_length, tbuffer);*/

    let mut tpos = 0;
    tbuffer[tstart..tstart + window_length]
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
    tid: u32,
    reads: &[HAECRecord],
    max_ins: &[u16],
    tstart: usize,
    window_length: usize, // Full window length
    tbuffer: &[u8],
    qbuffer: &mut [u8],
) -> (Array2<u8>, Array2<u8>) {
    //Get features
    let length = max_ins.iter().map(|v| *v as usize).sum::<usize>() + max_ins.len();

    let mut bases = Array::from_elem((length, 1 + TOP_K_SORT), b'.');
    let mut quals = Array::from_elem((length, 1 + TOP_K_SORT), b'!');

    // First write the target
    write_target_for_window(
        tstart,
        &reads[tid as usize],
        &max_ins,
        bases.index_axis_mut(Axis(1), 0),
        quals.index_axis_mut(Axis(1), 0),
        window_length,
        tbuffer,
    );

    // Write top-k overlaps for the window
    overlaps.iter().enumerate().for_each(|(i, ow)| {
        get_features_for_ol_window(
            bases.index_axis_mut(Axis(1), i + 1),
            quals.index_axis_mut(Axis(1), i + 1),
            ow,
            &ow.alignment.cigar,
            &reads[ow.alignment.overlap.qid as usize],
            ow.tstart as usize - tstart,
            &max_ins,
            qbuffer,
        )
    });

    (bases, quals)
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
    read.seq.get_sequence(tbuf);

    let (windows, aln_feats_info, counts) =
        extract_windows_new(&overlaps, reads, window_size, (tbuf, qbuf));

    let scores = score_alignments(&overlaps, &aln_feats_info, counts);

    let n_windows = windows.len();
    feats_output.init(rid, &read.id);
    for (i, mut window) in windows.into_iter().enumerate() {
        window.sort_unstable_by_key(|w| Reverse(scores[&w.alignment.overlap.qid]));
        window.truncate(TOP_K_SORT);

        let win_len = if i == n_windows - 1 {
            read.seq.len() - i * window_size as usize
        } else {
            window_size as usize
        };

        let max_ins = get_max_ins_for_window(&window, i * window_size as usize, win_len);

        let (bases, quals) = get_features_for_window(
            &mut window,
            rid,
            reads,
            &max_ins,
            i * window_size as usize,
            win_len,
            tbuf,
            qbuf,
        );

        let supported = get_supported(&bases);
        let qids = window
            .iter()
            .map(|ow| std::str::from_utf8(&reads[ow.alignment.overlap.qid as usize].id).unwrap())
            .collect::<Vec<_>>();

        feats_output.update(
            rid,
            i as u16,
            bases,
            quals,
            supported,
            qids,
            n_windows as u16,
        );
    }

    feats_output.emit();
}

fn score_alignments(
    alignments: &[Alignment],
    aln_feats_info: &HashMap<u32, AlnFeatsInfo>,
    counts: Vec<[u32; 5]>,
) -> HashMap<u32, OrderedFloat<f32>> {
    let mut candidates = find_all_candidates(counts);
    candidates.sort_unstable(); // We have to sort it for partitioning

    /*let output_path = format!("{}.txt", tname);
    let mut file = std::fs::File::create(&output_path).expect("Cannot create overlap windows file");
    writeln!(file, "qname\tsupported\tcandidates\tscore").unwrap();*/

    let scores = alignments
        .iter()
        .map(|a| {
            let tid = a.overlap.tid;
            let qid = a.overlap.qid;
            let feats_info = &aln_feats_info[&qid];

            let start = candidates.partition_point(|&x| x < feats_info.first_win_tstart);
            let end = candidates.partition_point(|&x| x < feats_info.last_win_tend);

            let cset = HashSet::from_iter(candidates[start..end].iter().cloned());
            let n_diff = feats_info
                .diffs
                .keys()
                .filter(|(pos, ins)| *ins == 0 && cset.contains(pos))
                .count();

            let n_candidates = end - start;
            assert!(
                n_diff <= n_candidates,
                "{tid}:{qid} n_diff {n_diff} cannot be bigger than n_candidates {n_candidates}",
            );

            if n_candidates == 0 {
                //let qname = std::str::from_utf8(&reads[qid as usize].id).unwrap();
                //writeln!(file, "{qname}\t0\t0\t0").unwrap();

                return (qid, OrderedFloat(0.0));
            }

            let score = (n_candidates - n_diff) as f32 / n_candidates as f32
                * f32::ln(n_candidates as f32 + 1.);

            /*let qname = std::str::from_utf8(&reads[qid as usize].id).unwrap();
            writeln!(
                file,
                "{qname}\t{}\t{}\t{}",
                n_candidates - n_diff,
                n_candidates,
                score
            )
            .unwrap();*/

            (qid, OrderedFloat(score))
        })
        .collect();

    scores
}

fn find_all_candidates(counts: Vec<[u32; 5]>) -> Vec<u32> {
    let candidates = counts
        .into_iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let n_alns = c.iter().sum::<u32>();
            let thresh = ((n_alns as f32 * REL_FILTER_THRESHOLD) as u32).max(3);

            if c.into_iter().filter(|&count| count >= thresh).count() >= 2 {
                Some(i as u32)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    candidates
}

/*pub(crate) fn extract_features_old<'a, T: FeaturesOutput<'a>>(
    rid: u32,
    reads: &'a [HAECRecord],
    overlaps: Vec<Alignment>,
    window_size: u32,
    (tbuf, qbuf): (&mut [u8], &mut [u8]),
    feats_output: &mut T,
) {
    let read = &reads[rid as usize];
    reads[rid as usize].seq.get_sequence(tbuf);

    // Get overlaps for windows
    let n_windows = (read.seq.len() + window_size as usize - 1) / window_size as usize;
    //let mut windows = vec![Vec::new(); n_windows];

    let (mut windows, counts, diffs) =
        extract_windows_new(rid, &overlaps, reads, window_size, (tbuf, qbuf));

    let ovlps_cigar_map = overlaps
        .iter()
        .map(|a| (a.overlap.qid, &a.cigar))
        .collect::<HashMap<u32, &Vec<u8>>>();
    let mut all_features = Vec::new();
    /*for alignment in overlaps.iter() {
        let qid = alignment.overlap.return_other_id(rid);

        //Extract windows
        let is_target = alignment.overlap.tid == rid;
        extract_windows_new(
            &mut windows,
            &alignment.overlap,
            &alignment.cigar,
            tshift,
            qshift,
            is_target,
            window_size,
        );

        ovlps_cigar_map.insert(qid, &alignment.cigar);
    }*/

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
            //let cigar_end = (ow.cigar_end_idx + 1).min(cigar.len());
            overlap_window_filter(&cigar[ow.cigar_start_idx..ow.cigar_end_idx])
        });

        // Sort window to take TOP-K
        windows[i].sort_by_key(|ow| {
            let cigar = ovlps_cigar_map
                .get(&ow.overlap.return_other_id(rid))
                .unwrap();

            let tstart = ow.tstart as usize;
            let tend = i * window_size as usize + win_len;
            //reads[rid as usize].seq.get_subseq(tstart..tend, tbuf);

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

            let acc = calculate_accuracy(ow, cigar, &tbuf[tstart..tend], &qbuf[..qlen]);
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

        let qids: Vec<_> = windows[i]
            .iter()
            .map(|ow| {
                std::str::from_utf8(&reads[ow.overlap.return_other_id(rid) as usize].id).unwrap()
            })
            .collect();

        let supported = get_supported(&bases);

        /*feats_output.update(
            rid,
            i as u16,
            bases,
            quals,
            supported,
            qids,
            n_windows as u16,
        );*/

        all_features.push((
            rid,
            i as u16,
            bases,
            quals,
            supported,
            qids,
            n_windows as u16,
        ));
    }

    // Calculate ratios
    let mut ratios = HashMap::default();
    for (rid, i, bases, quals, supported, qids, n_windows) in all_features.iter() {
        let pos_to_idx = bases
            .slice(s![.., 0])
            .iter()
            .enumerate()
            .filter(|&(_, b)| *b != b'*')
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let indices = supported
            .iter()
            .map(|s| pos_to_idx[s.pos as usize] + s.ins as usize)
            .collect::<HashSet<_>>();

        let tgt = bases.slice(s![.., 0]);

        for (qid, row_idx) in qids.iter().zip(1..) {
            let qry = bases.slice(s![.., row_idx]);

            for (pos, (tgt_b, qry_b)) in tgt.iter().zip(qry.iter()).enumerate() {
                if !indices.contains(&pos) {
                    continue;
                }

                let t = (*tgt_b as char).to_ascii_uppercase();
                let q = (*qry_b as char).to_ascii_uppercase();

                if t == '*' {
                    continue;
                }

                if q == t {
                    ratios.entry(qid.to_string()).or_insert((0., 0.)).0 += 1.;
                } else {
                    ratios.entry(qid.to_string()).or_insert((0., 0.)).1 += 1.;
                }
            }
        }
    }

    for (rid, i, bases, quals, supported, qids, n_windows) in all_features {
        let mut iden = vec![OrderedFloat(f64::MAX)];
        for &qry_rid in qids.iter() {
            let s = ratios
                .get(qry_rid)
                .map_or(0., |(n, d)| n / (n + d) * f64::ln(n + d + 1.));

            iden.push(OrderedFloat(s));
        }

        let mut sr = (0..iden.len()).collect::<Vec<_>>();
        sr.sort_by_key(|&i| Reverse(iden[i]));

        let mut new_bases = Vec::new();
        let mut new_quals = Vec::new();

        for &i in sr.iter().take(TOP_K_SORT + 1) {
            new_bases.push(bases.index_axis(Axis(1), i));
            new_quals.push(quals.index_axis(Axis(1), i));
        }
        for i in sr.len()..TOP_K_SORT + 1 {
            new_bases.push(bases.index_axis(Axis(1), i));
            new_quals.push(quals.index_axis(Axis(1), i));
        }

        assert_eq!(new_bases.len(), TOP_K_SORT + 1);
        assert_eq!(new_quals.len(), TOP_K_SORT + 1);

        let new_bases = stack(Axis(1), &new_bases).unwrap();
        let retain_idx: Vec<_> = new_bases
            .axis_iter(Axis(0))
            .enumerate()
            .filter_map(|(pos, col)| {
                let all_gap = col
                    .iter()
                    .filter(|&b| *b != b'.')
                    .all(|&b| b == b'*' || b == b'#');
                if all_gap {
                    None
                } else {
                    Some(pos)
                }
            })
            .collect();

        let new_bases = new_bases
            .select(Axis(0), &retain_idx)
            .as_standard_layout()
            .to_owned();

        let new_quals = stack(Axis(1), &new_quals)
            .unwrap()
            .select(Axis(0), &retain_idx)
            .as_standard_layout()
            .to_owned();

        let supported = get_supported(&new_bases);

        /*let new_bases = stack(Axis(1), &new_bases)
        .unwrap()
        .as_standard_layout()
        .to_owned();
        let new_quals = stack(Axis(1), &new_quals)
            .unwrap()
            .as_standard_layout()
            .to_owned();*/

        let qids = sr.iter().skip(1).map(|&i| qids[i - 1]).collect();

        feats_output.update(
            rid,
            i as u16,
            new_bases,
            new_quals,
            supported,
            qids,
            n_windows as u16,
        );
    }

    feats_output.emit();
}*/

fn calculate_accuracy(window: &OverlapWindow, cigar: &[u8], tseq: &[u8], qseq: &[u8]) -> f32 {
    let (mut tpos, mut qpos) = (0, 0);
    let (mut m, mut s, mut i, mut d) = (0, 0, 0, 0);

    let cigar_iter = CigarIter::new(&cigar[window.cigar_start_idx..window.cigar_end_idx]);
    for (op, range) in cigar_iter {
        let len = if (0..window.cigar_end_idx - window.cigar_start_idx) == range {
            assert!(
                window.cigar_end_offset > window.cigar_start_offset,
                "{}, {}",
                window.cigar_start_idx,
                window.cigar_end_idx
            );
            (window.cigar_end_offset - window.cigar_start_offset) as usize
        } else if range.start == 0 {
            assert!(
                op.get_length() > window.cigar_start_offset,
                "{}, {}, {}, op range{:?}, window range {:?}",
                window.cigar_start_offset,
                op.get_length(),
                std::str::from_utf8(cigar).unwrap(),
                range,
                window.cigar_start_idx..window.cigar_end_idx
            );
            (op.get_length() - window.cigar_start_offset) as usize
        } else if range.end == window.cigar_end_idx - window.cigar_start_idx {
            window.cigar_end_offset as usize
        } else {
            op.get_length() as usize
        };

        assert!(len > 0, "Operation length cannot be 0");

        // Not insertion -> consume tseq -> check bounds
        if !matches!(op, CigarOp::Insertion(_)) {
            assert!(
                tpos + len <= tseq.len(),
                "Length {} + {} is bigger than the tseq {}. {} {} {} {:?} {:?}",
                len,
                tpos,
                tseq.len(),
                window.cigar_start_offset,
                op.get_length(),
                std::str::from_utf8(cigar).unwrap(),
                range,
                window.cigar_start_idx..window.cigar_end_idx
            );
        }

        // Not deletion -> consume qseq -> check bounds
        if !matches!(op, CigarOp::Deletion(_)) {
            assert!(
                qpos + len <= qseq.len(),
                "Length {} + {} is bigger than the qseq {}. {} {} {} {:?} {:?}",
                len,
                qpos,
                qseq.len(),
                window.cigar_start_offset,
                op.get_length(),
                std::str::from_utf8(cigar).unwrap(),
                range,
                window.cigar_start_idx..window.cigar_end_idx
            );
        }

        match op {
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

fn get_supported<S>(bases: &ArrayBase<S, Ix2>) -> Vec<SupportedPos>
where
    S: Data<Elem = u8>,
{
    let mut counter: HashMap<u8, usize> = HashMap::default();
    counter.insert(b'A', 0);
    counter.insert(b'C', 0);
    counter.insert(b'G', 0);
    counter.insert(b'T', 0);
    counter.insert(b'*', 0);

    let mut supporeted = Vec::new();

    let (mut tpos, mut ins) = (-1i16, 0);
    for col in bases.axis_iter(Axis(0)) {
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

        let thresh = (bases.len_of(Axis(1)) as f64 * 0.1) as usize;
        let n_supported = counter
            .iter()
            .fold(0u8, |acc, (_, &c)| if c >= thresh { acc + 1 } else { acc });
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
    bases: Array2<u8>,
    quals: Array2<u8>,
    supported: impl IntoIterator<Item = SupportedPos>,
) -> Result<()> {
    let ids_path = path.as_ref().join(format!("{}.ids.txt", window_id));
    let ids_file = File::create(ids_path)?;
    let mut ids_writer = BufWriter::new(ids_file);
    for id in ids {
        writeln!(&mut ids_writer, "{}", id)?
    }

    let features_path = path.as_ref().join(format!("{}.features.npy", window_id));

    // Convert quals to u8 + stack feats
    let quals = quals.mapv(|q| q as u8);
    let features = stack![Axis(0), bases, quals];

    // Write feats
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
        rid: u32,
        wid: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        n_wids: u16,
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
    pbar_sender: Sender<PBarNotification>,
}

impl<T> FeatsGenOutput<'_, T>
where
    T: AsRef<Path> + Clone,
{
    pub(crate) fn new(path: T, pbar_sender: Sender<PBarNotification>) -> Self {
        Self {
            base_path: path,
            rname: None,
            pbar_sender: pbar_sender,
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
        _rid: u32,
        wid: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        _n_wids: u16,
    ) {
        let rid = std::str::from_utf8(self.rname.unwrap()).unwrap();
        let output_path = self.base_path.as_ref().join(rid);
        create_dir_all(&output_path).expect("Cannot create directory");

        output_features(&output_path, wid, &ids, bases, quals, supported.into_iter()).unwrap();
    }

    fn emit(&mut self) {
        self.pbar_sender.send(PBarNotification::Inc).unwrap();

        self.rname = None;
    }
}

pub(crate) struct InferenceOutput {
    sender: Sender<InferenceData>,
    features: Vec<WindowExample>,
    batch_size: usize,
}

impl InferenceOutput {
    pub(crate) fn new(sender: Sender<InferenceData>, batch_size: usize) -> Self {
        Self {
            sender,
            features: Vec::with_capacity(batch_size),
            batch_size: batch_size,
        }
    }
}

impl<'a> FeaturesOutput<'a> for InferenceOutput {
    fn init<'b>(&mut self, _rid: u32, _rname: &'b [u8])
    where
        'b: 'a,
    {
    }

    fn update(
        &mut self,
        rid: u32,
        wid: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        n_wids: u16,
    ) {
        self.features.push(WindowExample::new(
            rid,
            wid,
            ids.len().min(TOP_K_SORT) as u8,
            bases,
            quals,
            supported,
            n_wids,
        ));

        if self.features.len() == self.batch_size {
            let data = prepare_examples(self.features.drain(..), self.batch_size);
            self.sender.send(data).unwrap();
        }
    }

    fn emit(&mut self) {
        let data = prepare_examples(self.features.drain(..), self.batch_size);
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
