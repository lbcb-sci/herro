pub mod wfa;

#[derive(Debug, PartialEq)]
pub enum CigarOp {
    MATCH(u32),
    MISMATCH(u32),
    INSERTION(u32),
    DELETION(u32),
}

impl From<(u32, char)> for CigarOp {
    fn from(cigar: (u32, char)) -> Self {
        match cigar.1 {
            'M' => CigarOp::MATCH(cigar.0),
            'X' => CigarOp::MISMATCH(cigar.0),
            'I' => CigarOp::INSERTION(cigar.0),
            'D' => CigarOp::DELETION(cigar.0),
            _ => panic!("Invalid cigar op {}", cigar.1),
        }
    }
}

#[inline]
fn complement(base: char) -> char {
    match base {
        'A' => 'T',
        'C' => 'G',
        'G' => 'C',
        'T' => 'A',
        _ => panic!("Unknown base"),
    }
}

pub fn reverse_complement(seq: &str) -> String {
    seq.chars().rev().map(|c| complement(c)).collect()
}
