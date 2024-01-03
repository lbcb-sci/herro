use crossbeam_channel::Sender;
use glob::glob;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use zstd::stream::AutoFinishEncoder;
use zstd::Encoder;

use std::fmt;

use std::fs::create_dir_all;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::Path;

use crate::aligners::{cigar_to_string, CigarOp};
use crate::haec_io::bytes_to_u32;
use crate::haec_io::HAECRecord;
use crate::mm2;
use crate::AlnMode;
use crate::LINE_ENDING;
use crate::READS_BATCH_SIZE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    Forward,
    Reverse,
}

impl fmt::Display for Strand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Self::Forward => '+',
            Self::Reverse => '-',
        };

        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone)]
pub struct Overlap {
    pub qid: u32,
    pub qlen: u32,
    pub qstart: u32,
    pub qend: u32,
    pub strand: Strand,
    pub tid: u32,
    pub tlen: u32,
    pub tstart: u32,
    pub tend: u32,
}

impl Overlap {
    pub fn new(
        qid: u32,
        qlen: u32,
        qstart: u32,
        qend: u32,
        strand: Strand,
        tid: u32,
        tlen: u32,
        tstart: u32,
        tend: u32,
    ) -> Self {
        Overlap {
            qid,
            qlen,
            qstart,
            qend,
            strand,
            tid,
            tlen,
            tstart,
            tend,
        }
    }

    pub fn return_other_id(&self, id: u32) -> u32 {
        if self.qid == id {
            return self.tid;
        } else {
            return self.qid;
        }
    }
}

#[derive(Debug)]
pub struct Alignment {
    pub overlap: Overlap,
    pub cigar: Vec<CigarOp>,
}

impl Alignment {
    pub fn new(overlap: Overlap, cigar: Vec<CigarOp>) -> Self {
        Alignment { overlap, cigar }
    }
}

impl PartialEq for Overlap {
    fn eq(&self, other: &Self) -> bool {
        self.qid == other.qid
            && self.qstart == other.qstart
            && self.qend == other.qend
            && self.strand == other.strand
            && self.tid == other.tid
            && self.tstart == other.tstart
            && self.tend == other.tend
    }
}

impl Eq for Overlap {}

pub fn parse_paf(
    mut reader: impl BufRead,
    name_to_id: &HashMap<&[u8], u32>,
    mut alns_writer: Option<&mut AutoFinishEncoder<BufWriter<File>>>,
) -> HashMap<u32, Vec<Alignment>> {
    //let mut reader = BufReader::new(read);

    let mut buffer = Vec::new();
    let mut processed = HashSet::default();

    //let mut alignments = Vec::new();
    let mut tid_to_alns = HashMap::default();
    while let Ok(len) = reader.read_until(LINE_ENDING, &mut buffer) {
        if len == 0 {
            break;
        }

        let mut data = buffer[..len - 1].split(|&c| c == b'\t');

        let qid = match name_to_id.get(data.next().unwrap()) {
            Some(qid) => *qid,
            None => {
                buffer.clear();
                continue;
            }
        };
        let qlen = bytes_to_u32(data.next().unwrap());
        let qstart = bytes_to_u32(data.next().unwrap());
        let qend = bytes_to_u32(data.next().unwrap());

        let strand = match data.next().unwrap()[0] {
            b'+' => Strand::Forward,
            b'-' => Strand::Reverse,
            _ => panic!("Invalid strand character."),
        };

        let tid = match name_to_id.get(data.next().unwrap()) {
            Some(tid) => *tid,
            None => {
                buffer.clear();
                continue;
            }
        };
        let tlen: u32 = bytes_to_u32(data.next().unwrap());
        let tstart: u32 = bytes_to_u32(data.next().unwrap());
        let tend: u32 = bytes_to_u32(data.next().unwrap());

        let cigar = data.last().unwrap();
        let cigar = parse_cigar(&cigar[5..]);

        if tid == qid {
            // Cannot have self-overlaps
            buffer.clear();
            continue;
        }

        if processed.contains(&(qid, tid)) {
            buffer.clear();
            continue; // We assume the first overlap between two reads is the best one
        }
        processed.insert((qid, tid));

        let overlap = Overlap::new(qid, qlen, qstart, qend, strand, tid, tlen, tstart, tend);
        let alignment = Alignment::new(overlap, cigar);
        tid_to_alns
            .entry(tid)
            .or_insert_with(|| Vec::new())
            .push(alignment);

        if let Some(ref mut aw) = alns_writer {
            aw.write_all(&buffer[..len]).unwrap();
        }

        buffer.clear();
    }

    tid_to_alns
}

#[allow(dead_code)]
pub(crate) fn print_alignments(alignments: &[Alignment], reads: &[HAECRecord]) {
    for aln in alignments {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            std::str::from_utf8(&reads[aln.overlap.qid as usize].id).unwrap(),
            aln.overlap.qlen,
            aln.overlap.qstart,
            aln.overlap.qend,
            aln.overlap.strand,
            std::str::from_utf8(&reads[aln.overlap.tid as usize].id).unwrap(),
            aln.overlap.tlen,
            aln.overlap.tstart,
            aln.overlap.tend,
            cigar_to_string(&aln.cigar),
        )
    }
}

fn parse_cigar(cigar: &[u8]) -> Vec<CigarOp> {
    let n_ops = cigar.iter().filter(|c| c.is_ascii_alphabetic()).count();
    let mut ops = Vec::with_capacity(n_ops);

    let mut l = 0;
    for &c in cigar {
        if c.is_ascii_digit() {
            l = l * 10 + (c - b'0') as u32;
        } else {
            match c {
                b'M' => ops.push(CigarOp::Match(l)),
                b'I' => ops.push(CigarOp::Insertion(l)),
                b'D' => ops.push(CigarOp::Deletion(l)),
                _ => panic!("Invalid CIGAR character."),
            }

            l = 0;
        }
    }

    ops
}

pub(crate) fn generate_batches<'a, P, T>(
    reads: &'a [HAECRecord],
    name_to_id: &'a HashMap<&[u8], u32>,
    reads_path: P,
    threads: usize,
    alns_path: Option<T>,
) -> impl Iterator<Item = HashMap<u32, Vec<Alignment>>> + 'a
where
    P: AsRef<Path>,
    P: 'a,
    T: AsRef<Path> + 'a,
{
    if let Some(ref ap) = alns_path {
        create_dir_all(ap).unwrap();
    }

    reads
        .chunks(READS_BATCH_SIZE)
        .enumerate()
        .map(move |(batch_idx, batch)| {
            let mm2_out = BufReader::new(mm2::call_mm2(batch, &reads_path, threads));

            let mut writer = alns_path.as_ref().map(|ap| {
                let batch_path = ap.as_ref().join(format!("{batch_idx}.oec.zst"));
                let file = File::create(batch_path).unwrap();
                let mut w = Encoder::new(BufWriter::new(file), 0).unwrap().auto_finish();

                // Write header
                writeln!(&mut w, "{}", batch.len()).unwrap();
                batch.iter().for_each(|r| {
                    writeln!(&mut w, "{}", std::str::from_utf8(&r.id).unwrap()).unwrap()
                });

                w
            });

            parse_paf(mm2_out, &name_to_id, writer.as_mut())
        })
}

pub(crate) fn read_batches<'a, P>(
    name_to_id: &'a HashMap<&[u8], u32>,
    batches: P,
) -> impl Iterator<Item = HashMap<u32, Vec<Alignment>>> + 'a
where
    P: AsRef<Path>,
    P: 'a,
{
    let g = batches.as_ref().join("*.oec.zst");
    glob(g.to_str().unwrap()).unwrap().map(|p| {
        let mut reader = {
            let file = File::open(p.unwrap()).unwrap();
            let reader = zstd::Decoder::new(file).unwrap();
            BufReader::with_capacity(65_536, reader)
        };

        // Read number of target reads
        let mut buf = Vec::new();
        let len = reader.read_until(LINE_ENDING, &mut buf).unwrap();
        let n_targets = buf[..len - 1]
            .iter()
            .fold(0, |acc, &d| acc * 10 + (d - b'0') as u32);

        let _tids: HashSet<_> = (0..n_targets)
            .map(|_| {
                buf.clear();
                let len = reader.read_until(LINE_ENDING, &mut buf).unwrap();

                *name_to_id.get(&buf[..len - 1]).unwrap()
            })
            .collect();

        parse_paf(&mut reader, name_to_id, None)
    })
}

pub(crate) fn alignment_reader<T: AsRef<Path>, U: AsRef<Path>>(
    reads: &[HAECRecord],
    reads_path: &T,
    aln_mode: AlnMode<U>,
    n_threads: usize,
    alns_sender: Sender<(u32, Vec<Alignment>)>,
) {
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (&*e.id, i as u32))
        .collect();

    let batches: Box<dyn Iterator<Item = HashMap<u32, Vec<Alignment>>>> = match aln_mode {
        AlnMode::None => {
            let batches = generate_batches(&reads, &name_to_id, &reads_path, n_threads, None::<T>);
            Box::new(batches)
        }
        AlnMode::Read(path) => {
            let batches = read_batches(&name_to_id, path);
            Box::new(batches)
        }
        AlnMode::Write(path) => {
            let batches = generate_batches(&reads, &name_to_id, &reads_path, n_threads, Some(path));
            Box::new(batches)
        }
    };

    for alignments in batches {
        /*let mut read_to_alns = HashMap::default();
        alignments.into_iter().for_each(|aln| {
            if tids.contains(&aln.overlap.tid) {
                read_to_alns
                    .entry(aln.overlap.tid)
                    .or_insert_with(|| Vec::new())
                    .push(aln);
            }
        });*/

        alignments.into_iter().for_each(|example| {
            //println!("Aln reader: {}", alns_sender.len());
            alns_sender.send(example).unwrap();
        });
    }
}

/*pub(crate) fn aln_reader_worker<T, U>(
    reads: &[HAECRecord],
    reads_path: &T,
    aln_mode: AlnMode<U>,
    n_threads: usize,
    alns_sender: Sender<(u32, Vec<Alignment>)>,
) where
    T: AsRef<Path>,
    U: AsRef<Path>,
{
    let name_to_id: HashMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(i, e)| (&*e.id, i as u32))
        .collect();

    let path = match aln_mode {
        AlnMode::Read(p) => p,
        _ => unreachable!(),
    };

    let g = path.as_ref().join("*.oec.zst");
    let mut alignments = Vec::new();
    glob(g.to_str().unwrap()).unwrap().for_each(|p| {
        let mut reader = {
            let file = File::open(p.unwrap()).unwrap();
            let reader = zstd::Decoder::new(file).unwrap();
            BufReader::with_capacity(65_536, reader)
        };

        // Read number of target reads
        let mut buf = Vec::new();
        let len = reader.read_until(LINE_ENDING, &mut buf).unwrap();
        let n_targets = buf[..len - 1]
            .iter()
            .fold(0, |acc, &d| acc * 10 + (d - b'0') as u32);

        let tids: HashSet<_> = (0..n_targets)
            .map(|_| {
                buf.clear();
                let len = reader.read_until(LINE_ENDING, &mut buf).unwrap();

                *name_to_id.get(&buf[..len - 1]).unwrap()
            })
            .collect();

        parse_paf(&mut reader, &name_to_id, None, &mut alignments);

        let mut read_to_alns = HashMap::default();
        alignments.drain(..).for_each(|aln| {
            if tids.contains(&aln.overlap.tid) {
                read_to_alns
                    .entry(aln.overlap.tid)
                    .or_insert_with(|| Vec::new())
                    .push(aln);
            }
        });

        read_to_alns.into_iter().for_each(|example| {
            //println!("Aln reader: {}", alns_sender.len());
            alns_sender.send(example).unwrap();
        });
    });
}*/
