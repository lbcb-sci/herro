pub mod wfa;

#[derive(Debug)]
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
