use std::borrow::Cow;
use std::collections::HashMap;

use ordered_float::OrderedFloat;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::aligners::CigarOp;
use crate::haec_io::HAECRecord;
use crate::overlaps::{Overlap, Strand};
use crate::windowing::{extract_windows, OverlapWindow};

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

        //state.windows.iter().for_each(|w|get)

        //todo!("Work on extracting features from windows")
    }

    for i in (0..read.seq.len()).step_by(window_size as usize) {
        let win_len = read.seq.len().min(i + window_size as usize) - i;
        get_features_for_window(win_len, &mut windows[i / window_size as usize], tid);
    }
}

pub fn extract_features(reads: &[HAECRecord], overlaps: &[Overlap], window_size: u32) {
    let mut read_to_overlaps = get_reads_to_overlaps(overlaps);

    read_to_overlaps.par_iter_mut().for_each(|(rid, ovlps)| {
        generate_features_for_read(&reads[*rid as usize], *rid, ovlps, reads, window_size)
    });
}
