use std::{ops::Range, path::Path, str::from_utf8};

use needletail::parse_fastx_file;

pub struct HAECRecord {
    pub id: String,
    pub description: Option<String>,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
}

impl HAECRecord {
    fn new(id: String, description: Option<String>, seq: Vec<u8>, qual: Vec<u8>) -> Self {
        HAECRecord {
            id,
            description,
            seq,
            qual,
        }
    }

    pub fn subseq_iter(&self, range: Range<usize>) -> impl DoubleEndedIterator<Item = (&u8, &u8)> {
        self.seq[range.clone()].iter().zip(self.qual[range].iter())
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
