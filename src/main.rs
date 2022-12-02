use clap::Parser;

use ont_haec_rs::error_correction;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    reads_path: String,
    paf_path: String,
    threads: usize,
}

fn main() {
    let cli = Cli::parse();

    error_correction(&cli.reads_path, &cli.paf_path, cli.threads)
}
