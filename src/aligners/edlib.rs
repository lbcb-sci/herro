use std::borrow::Cow;

use edlib_rs::edlibrs::edlibAlignRs;
use edlib_rs::edlibrs::edlibAlignmentToCigarRs;
use edlib_rs::edlibrs::EdlibAlignConfigRs;
use edlib_rs::edlibrs::EdlibCigarFormatRs;
use lazy_static::lazy_static;

use crate::haec_io::HAECRecord;
use crate::overlaps::Overlap;
use crate::overlaps::Strand;

lazy_static! {
    static ref CONFIG: EdlibAlignConfigRs<'static> = EdlibAlignConfigRs::new(
        -1,
        edlib_rs::edlibrs::EdlibAlignModeRs::EDLIB_MODE_NW,
        edlib_rs::edlibrs::EdlibAlignTaskRs::EDLIB_TASK_PATH,
        &[],
    );
}

pub fn align(overlaps: &mut [Overlap], reads: &[HAECRecord]) {
    for overlap in overlaps {
        let tseq = &reads[overlap.tid as usize].seq[overlap.tstart as usize..overlap.tend as usize];

        let qseq = match overlap.strand {
            Strand::FORWARD => Cow::Borrowed(
                &reads[overlap.tid as usize].seq[overlap.tstart as usize..overlap.tend as usize],
            ),
            Strand::REVERSE => {
                let seq = &reads[overlap.tid as usize].seq
                    [overlap.tstart as usize..overlap.tend as usize];
                Cow::Owned(reverse_complement(seq))
            }
        };

        let result = edlibAlignRs(qseq.as_bytes(), tseq.as_bytes(), &CONFIG);
        overlap.cigar = Some(edlibAlignmentToCigarRs(
            &result.alignment.unwrap(),
            &EdlibCigarFormatRs::EDLIB_CIGAR_EXTENDED,
        ));
    }
}
