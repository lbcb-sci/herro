use std::{path::Path, str::from_utf8};

use needletail::parse_fastx_file;

pub struct HAECRecord {
    pub id: String,
    pub description: Option<String>,
    pub seq: String,
}

impl HAECRecord {
    fn new(id: String, description: Option<String>, seq: String) -> Self {
        HAECRecord {
            id,
            description,
            seq,
        }
    }
}

pub fn get_reads<P: AsRef<Path>>(path: P) -> Vec<HAECRecord> {
    let mut reader = parse_fastx_file(path).expect("Cannot open file containing reads.");

    let mut reads = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.expect("Error parsing fastx file.");

        let mut split = from_utf8(record.id())
            .expect("Cannot parse the sequence id")
            .splitn(2, " ");
        let id = split.next().expect("Cannot be empty").to_owned();
        let description = split.next().map(|s| s.to_owned());

        let seq = from_utf8(&record.seq())
            .expect("Cannot parse the sequnce")
            .to_owned();

        reads.push(HAECRecord::new(id, description, seq));
    }

    reads
}
