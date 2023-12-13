use std::ffi::CString;

use crate::{features::TOP_K, haec_io::HAECRecord, overlaps::Strand, windowing::OverlapWindow};

use lazy_static::lazy_static;
use ndarray::ArrayViewMut2;
use spoa::{AlignmentEngine, AlignmentType, Graph};

lazy_static! {
    static ref BASE_REVERSE: [u8; 128] = {
        let mut arr = [255; 128];
        arr[b'A' as usize] = b'a';
        arr[b'C' as usize] = b'c';
        arr[b'G' as usize] = b'g';
        arr[b'T' as usize] = b't';
        arr[b'-' as usize] = b'#';
        arr
    };
    static ref BASE_FORWARD: [u8; 128] = {
        let mut arr = [255; 128];
        arr[b'A' as usize] = b'A';
        arr[b'C' as usize] = b'C';
        arr[b'G' as usize] = b'G';
        arr[b'T' as usize] = b'T';
        arr[b'-' as usize] = b'*';
        arr
    };
}

pub(crate) fn get_msa(
    overlaps: &mut [OverlapWindow],
    tid: u32,
    reads: &[HAECRecord],
    tstart: usize,
    window_length: usize, // Full window length
    tbuffer: &mut [u8],
    qbuffer: &mut [u8],
) -> Vec<CString> {
    let mut engine = AlignmentEngine::new(AlignmentType::kNW, 5, -4, -8, -6, -10, -4);
    let mut graph = Graph::new();

    let target = &reads[tid as usize];
    target
        .seq
        .get_subseq(tstart..tstart + window_length, tbuffer);

    let target = CString::new(&tbuffer[..window_length as usize]).unwrap();
    let alignment = engine.align(&target, &graph);
    graph.add_alignment(&alignment, &target);

    overlaps.iter().take(TOP_K).for_each(|o| {
        let qid = o.overlap.return_other_id(tid) as usize;
        let seq = get_seq_for_ol_window(o, &reads[qid], tid, qbuffer);

        let alignment = engine.align(&seq, &graph);
        graph.add_alignment(&alignment, &seq);
    });

    graph.multiple_sequence_alignment(false)
}

fn get_seq_for_ol_window(
    window: &OverlapWindow,
    query: &HAECRecord,
    tid: u32,
    qbuffer: &mut [u8],
) -> CString {
    // Handle query sequence
    let (qstart, qend) = if window.overlap.tid == tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    if window.overlap.strand == Strand::Forward {
        let range = (qstart + window.qstart) as usize..(qstart + window.qend) as usize;
        let qlen = (window.qend - window.qstart) as usize;

        query.seq.get_subseq(range.clone(), qbuffer);
        return CString::new(&qbuffer[..qlen]).unwrap();
    } else {
        let range = (qend - window.qend) as usize..(qend - window.qstart) as usize;
        let qlen = (window.qend - window.qstart) as usize;

        query.seq.get_rc_subseq(range.clone(), qbuffer);
        return CString::new(&qbuffer[..qlen]).unwrap();
    }
}

pub(crate) fn write_target_for_window(
    tstart: usize,
    target: &HAECRecord,
    mut features: ArrayViewMut2<'_, u8>,
    window_length: usize,
    seq: &[u8],
) {
    let quals = &target.qual[tstart..tstart + window_length];

    let mut tpos = 0;
    seq.iter().enumerate().for_each(|(i, &b)| {
        features[[i, 0]] = BASE_FORWARD[b as usize];

        if b != b'-' {
            features[[i, 1]] = quals[tpos];
            tpos += 1;
        }
    });
}

pub(crate) fn write_query_for_window(
    mut features: ArrayViewMut2<'_, u8>,
    window: &OverlapWindow,
    query: &HAECRecord,
    tid: u32,
    seq: &[u8],
) {
    // Handle query sequence
    let (qstart, qend) = if window.overlap.tid == tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    let mut quals: Box<dyn DoubleEndedIterator<Item = &u8>> = match window.overlap.strand {
        Strand::Forward => {
            let range = (qstart + window.qstart) as usize..(qstart + window.qend) as usize;
            let quals = &query.qual[range];

            Box::new(quals.into_iter())
        }
        Strand::Reverse => {
            let range = (qend - window.qend) as usize..(qend - window.qstart) as usize;
            let quals = &query.qual[range];

            Box::new(quals.into_iter().rev())
        }
    };

    seq.iter().enumerate().for_each(|(i, &b)| {
        if window.overlap.strand == Strand::Forward {
            features[[i, 0]] = BASE_FORWARD[b as usize];
        } else {
            features[[i, 0]] = BASE_REVERSE[b as usize];
        }

        if b != b'-' {
            features[[i, 1]] = *quals.next().unwrap();
        }
    });

    assert!(quals.next().is_none());
}
