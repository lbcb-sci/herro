use clap::Parser;

use haec_baseline::error_correction;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    reads_path: String,
    paf_path: String,
}

fn main() {
    let cli = Cli::parse();

    error_correction(&cli.reads_path, &cli.paf_path)
}
