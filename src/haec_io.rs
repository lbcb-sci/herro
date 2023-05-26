use core::panic;
use std::{ops::RangeBounds, path::Path, str::from_utf8};

use needletail::parse_fastx_file;

pub struct HAECRecord {
    pub id: String,
    pub description: Option<String>,
    pub seq: HAECSeq,
    pub qual: Vec<u8>,
}

impl HAECRecord {
    fn new(id: String, description: Option<String>, seq: Vec<u8>, qual: Vec<u8>) -> Self {
        let len = seq.len();
        HAECRecord {
            id,
            description,
            seq: HAECSeq::new(seq, len), //seq: HAECSeq::from(&*seq),
            qual,
        }
    }
}

pub fn get_reads<P: AsRef<Path>>(path: P, min_length: u32) -> Vec<HAECRecord> {
    let mut reader = parse_fastx_file(path).expect("Cannot open file containing reads.");

    let mut reads = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.expect("Error parsing fastx file.");
        if record.num_bases() < min_length as usize {
            continue;
        }

        let mut split = from_utf8(record.id())
            .expect("Cannot parse the sequence id")
            .splitn(2, " ");
        let id = split.next().expect("Cannot be empty").to_owned();
        let description = split.next().map(|s| s.to_owned());

        let seq = record.seq().into_owned();
        let qual = record
            .qual()
            .expect("Qualities should be present.")
            .to_owned();

        reads.push(HAECRecord::new(id, description, seq, qual));
    }

    reads
}

#[derive(PartialEq, Debug)]
pub struct HAECSeq {
    data: Vec<u8>,
    length: usize,
}

impl HAECSeq {
    pub fn new(data: Vec<u8>, length: usize) -> Self {
        HAECSeq { data, length }
    }

    pub fn len(&self) -> usize {
        return self.length;
    }

    pub fn get_sequence(&self) -> Vec<u8> {
        Vec::from(self)
    }

    pub fn get_subsequence<R: RangeBounds<usize>>(&self, range: R) -> Vec<u8> {
        //decode(&self.data, self.length, range)

        let start = match range.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Included(s) => *s,
            std::ops::Bound::Excluded(s) => *s + 1,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Unbounded => self.length,
            std::ops::Bound::Included(e) => *e + 1,
            std::ops::Bound::Excluded(e) => *e,
        };

        Vec::from(&self.data[start..end])
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
        decode(&value.data, value.length, ..)
    }
}

fn encode(sequence: &[u8]) -> (Vec<u8>, usize) {
    let length = sequence.len();
    let mut data = vec![0; (length + 3) / 4];

    for (i, base) in sequence.iter().enumerate() {
        data[i / 4] |= ((*base >> 1) & 0b11) << (6 - 2 * (i % 4));
    }

    (data, length)
}

fn decode_base(value: u8) -> u8 {
    match value {
        0b00 => b'A',
        0b01 => b'C',
        0b11 => b'G',
        0b10 => b'T',
        _ => unreachable!(),
    }
}

fn decode<R: RangeBounds<usize>>(sequence: &[u8], length: usize, range: R) -> Vec<u8> {
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

    if end > length {
        panic!("Out of bounds for 2-bit sequence decoding.")
    }

    let mut data = Vec::with_capacity(end - start);
    let st_bin = start / 4;
    let en_bin = end / 4;

    // If there is only one window
    if st_bin == en_bin {
        (start % 4..end % 4).for_each(|i| {
            let base = decode_base((sequence[st_bin] >> (6 - 2 * (i % 4))) & 0b11);
            data.push(base);
        });

        return data;
    }

    // First handle the first bin
    (start % 4..4).for_each(|i| {
        let base = decode_base((sequence[st_bin] >> (6 - 2 * (i % 4))) & 0b11);
        data.push(base);
    });

    // Handle full bins
    (st_bin + 1..en_bin).for_each(|bin| {
        (0..4).for_each(|i| {
            let base = decode_base((sequence[bin] >> (6 - 2 * (i % 4))) & 0b11);
            data.push(base);
        });
    });

    // Handle last bin
    (0..end % 4).for_each(|i| {
        let base = decode_base((sequence[en_bin] >> (6 - 2 * (i % 4))) & 0b11);
        data.push(base);
    });

    data
}

#[cfg(test)]
mod tests {
    use crate::haec_io::HAECSeq;

    use super::{decode, encode};

    #[test]
    fn encode_sequence1() {
        let sequence = "ACGT";
        let result = encode(sequence.as_bytes());

        assert_eq!(result, (vec![0b00011110], 4));
    }

    #[test]
    fn encode_sequence2() {
        let haec_seq = HAECSeq::from("ACGTACG".as_bytes());
        assert_eq!(haec_seq, HAECSeq::new(vec![0b00011110, 0b00011100], 7));
    }

    #[test]
    fn decode_sequence1() {
        let encoded = vec![0b00011110];
        let length = 4;

        let decoded = decode(&encoded, length, ..);
        assert_eq!(decoded, "ACGT".as_bytes());
    }

    #[test]
    fn decode_sequence2() {
        let haec_seq = HAECSeq::new(vec![0b00011110, 0b00011100], 7);
        assert_eq!(Vec::from(&haec_seq), "ACGTACG".as_bytes());
    }

    #[test]
    fn test_range1() {
        let haec_seq = HAECSeq::from("ACGTACGTACGT".as_bytes());
        let subseq = haec_seq.get_subsequence(3..10);

        assert_eq!(&subseq, "TACGTAC".as_bytes())
    }

    #[test]
    fn test_range2() {
        let haec_seq = HAECSeq::from("ACGTACGTACGT".as_bytes());
        let subseq = haec_seq.get_subsequence(3..);

        assert_eq!(&subseq, "TACGTACGT".as_bytes())
    }

    #[test]
    fn test_range3() {
        let haec_seq = HAECSeq::from("ACGTACGTACGT".as_bytes());
        let subseq = haec_seq.get_subsequence(3..haec_seq.len());

        assert_eq!(&subseq, "TACGTACGT".as_bytes())
    }

    #[test]
    fn test_range4() {
        let haec_seq = HAECSeq::from("ACGTACGTACGT".as_bytes());
        let subseq = haec_seq.get_subsequence(..);

        assert_eq!(&subseq, "ACGTACGTACGT".as_bytes())
    }

    #[test]
    fn test_range5() {
        let haec_seq = HAECSeq::from("ACGTACGTACGT".as_bytes());
        let subseq = haec_seq.get_subsequence(8..11);

        assert_eq!(&subseq, "ACG".as_bytes())
    }
}
