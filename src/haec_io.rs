use core::panic;
use std::{ops::RangeBounds, path::Path};

use needletail::parse_fastx_file;
use rustc_hash::FxHashSet;

const BASE_ENCODING: [u64; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 1, 255, 255, 255, 2, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 0, 255, 1, 255, 255, 255, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

const BASE_DECODING: [u8; 4] = [b'A', b'C', b'G', b'T'];

pub struct HAECRecord {
    pub id: Vec<u8>,
    pub description: Option<Vec<u8>>,
    pub seq: HAECSeq,
    pub qual: Vec<u8>,
}

impl HAECRecord {
    fn new(id: Vec<u8>, description: Option<Vec<u8>>, seq: HAECSeq, qual: Vec<u8>) -> Self {
        HAECRecord {
            id,
            description,
            seq,
            qual,
        }
    }
}

pub fn get_reads<P: AsRef<Path>>(
    path: P,
    min_length: u32,
    core: &Option<FxHashSet<String>>,
    neighbour: &Option<FxHashSet<String>>,
) -> Vec<HAECRecord> {
    let mut reader = parse_fastx_file(path).expect("Cannot open file containing reads.");

    let mut reads = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.expect("Error parsing fastx file.");
        if record.num_bases() < min_length as usize {
            continue;
        }

        let mut split = record.id().splitn(2, |c| *c == b' ' || *c == b'\t');
        let id = split.next().expect("Invalid read id.").to_owned();
        let description = split.next().map(|d| d.to_owned());

        let seq = HAECSeq::from(&*record.seq());
        let qual = record
            .qual()
            .expect("Qualities should be present.")
            .to_owned();

        if let (Some(core), Some(neighbour)) = (&core, &neighbour) {
            if !neighbour.contains(std::str::from_utf8(&id).unwrap())
                && !core.contains(std::str::from_utf8(&id).unwrap())
            {
                continue;
            }
        }

        reads.push(HAECRecord::new(id, description, seq, qual));
    }

    reads.shrink_to_fit();
    reads
}

#[derive(PartialEq, Debug)]
pub struct HAECSeq {
    data: Vec<u64>,
    length: usize,
}

impl HAECSeq {
    pub fn new(data: Vec<u64>, length: usize) -> Self {
        HAECSeq { data, length }
    }

    pub fn len(&self) -> usize {
        return self.length;
    }

    pub fn get_sequence(&self, buffer: &mut [u8]) {
        decode(&self.data, self.length, .., false, buffer)
    }

    pub fn get_subseq<R: RangeBounds<usize>>(&self, range: R, buffer: &mut [u8]) {
        decode(&self.data, self.length, range, false, buffer)
    }

    pub fn get_rc_subseq<R: RangeBounds<usize>>(&self, range: R, buffer: &mut [u8]) {
        decode(&self.data, self.length, range, true, buffer)
    }
}

impl From<&[u8]> for HAECSeq {
    fn from(value: &[u8]) -> Self {
        let (data, length) = encode(value);
        HAECSeq::new(data, length)
    }
}

impl From<&HAECSeq> for Vec<u8> {
    fn from(value: &HAECSeq) -> Self {
        let mut seq = vec![0; value.length];
        decode(&value.data, value.length, .., false, &mut seq);

        seq
    }
}

fn encode(sequence: &[u8]) -> (Vec<u64>, usize) {
    let mut data = Vec::with_capacity((sequence.len() + 31) / 32);
    let mut block = 0;

    for (i, b) in sequence.iter().enumerate() {
        let c = BASE_ENCODING[*b as usize];

        block |= c << ((i << 1) & 63);
        if (i + 1) & 31 == 0 || i == sequence.len() - 1 {
            data.push(block);
            block = 0;
        }
    }

    (data, sequence.len())
}

fn decode<R: RangeBounds<usize>>(
    sequence: &[u64],
    length: usize,
    range: R,
    is_reversed: bool,
    buffer: &mut [u8],
) {
    let start = match range.start_bound() {
        std::ops::Bound::Unbounded => 0,
        std::ops::Bound::Included(s) => *s,
        std::ops::Bound::Excluded(s) => *s + 1,
    };

    let end = match range.end_bound() {
        std::ops::Bound::Unbounded => length,
        std::ops::Bound::Included(e) => *e + 1,
        std::ops::Bound::Excluded(e) => *e,
    };

    assert!(end <= length, "Out of bounds for 2-bit sequence decoding.");

    if start >= end {
        return;
    }

    let rc_mask = if is_reversed { 3 } else { 0 };
    (start..end).into_iter().for_each(|mut i| {
        let idx = i - start;
        if is_reversed {
            i = end - idx - 1;
        }

        let code = ((sequence[i >> 5] >> ((i << 1) & 63)) & 3) ^ rc_mask;
        buffer[idx] = BASE_DECODING[code as usize];
    });
}

pub(crate) fn bytes_to_u32(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0, |acc, &d| {
        if d.is_ascii_digit() {
            acc * 10 + (d - b'0') as u32
        } else {
            panic!("Character is not a valid digit");
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::haec_io::HAECSeq;

    use super::{decode, encode};

    #[test]
    fn encode_sequence1() {
        let sequence = "ACGT";
        let result = encode(sequence.as_bytes());

        assert_eq!(result, (vec![0b11100100], 4));
    }

    #[test]
    fn encode_sequence2() {
        let haec_seq = HAECSeq::from("ACGTACG".as_bytes());
        assert_eq!(haec_seq, HAECSeq::new(vec![0b10010011100100], 7));
    }

    #[test]
    fn decode_sequence1() {
        let encoded = vec![0b11100100];
        let length = 4;

        let mut buffer = vec![0; 4];
        decode(&encoded, length, .., false, &mut buffer);
        assert_eq!(&buffer[..4], "ACGT".as_bytes());
    }

    #[test]
    fn decode_sequence2() {
        let haec_seq = HAECSeq::new(vec![0b10010011100100], 7);
        assert_eq!(Vec::from(&haec_seq), "ACGTACG".as_bytes());
    }

    #[test]
    fn test_range1() {
        let haec_seq = HAECSeq::from("ACGTACGTACGT".as_bytes());

        let mut buffer = vec![0; 100];
        haec_seq.get_subseq(3..10, &mut buffer);

        assert_eq!(&buffer[..7], "TACGTAC".as_bytes())
    }

    #[test]
    fn test_range2() {
        let haec_seq = HAECSeq::from("ACGTACGTACGT".as_bytes());

        let mut buffer = vec![0; 100];
        haec_seq.get_subseq(3.., &mut buffer);

        assert_eq!(&buffer[..9], "TACGTACGT".as_bytes())
    }

    #[test]
    fn test_range3() {
        let haec_seq = HAECSeq::from("ACGTACGTACGT".as_bytes());

        let mut buffer = vec![0; 100];
        haec_seq.get_subseq(3..haec_seq.len(), &mut buffer);

        assert_eq!(&buffer[..9], "TACGTACGT".as_bytes())
    }

    #[test]
    fn test_range4() {
        let haec_seq = HAECSeq::from("ACGTACGTACGT".as_bytes());

        let mut buffer = vec![0; 100];
        haec_seq.get_subseq(.., &mut buffer);

        assert_eq!(&buffer[..12], "ACGTACGTACGT".as_bytes())
    }

    #[test]
    fn test_range5() {
        let haec_seq = HAECSeq::from("ACGTACGTACGT".as_bytes());

        let mut buffer = vec![0; 100];
        haec_seq.get_subseq(8..11, &mut buffer);

        assert_eq!(&buffer[..3], "ACG".as_bytes())
    }

    #[test]
    fn test_rc1() {
        let haec_seq = HAECSeq::from("ATCGATCGATCG".as_bytes());

        let mut buffer = vec![0; 100];
        haec_seq.get_rc_subseq(.., &mut buffer);

        assert_eq!(&buffer[..12], "CGATCGATCGAT".as_bytes())
    }

    #[test]
    fn test_rc2() {
        let haec_seq = HAECSeq::from("ATCGATCGATCG".as_bytes());

        let mut buffer = vec![0; 100];
        haec_seq.get_rc_subseq(3.., &mut buffer);

        assert_eq!(&buffer[..9], "CGATCGATC".as_bytes())
    }

    #[test]
    fn test_rc3() {
        let haec_seq = HAECSeq::from("ATCGATCGATCG".as_bytes());

        let mut buffer = vec![0; 100];
        haec_seq.get_rc_subseq(..9, &mut buffer);

        assert_eq!(&buffer[..9], "TCGATCGAT".as_bytes())
    }
}
