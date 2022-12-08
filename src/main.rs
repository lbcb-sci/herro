use clap::Parser;

use ont_haec_rs::error_correction;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    reads_path: String,
    paf_path: String,
    #[arg(short = 'w', default_value = "1024")]
    window_size: u32,
    #[arg(short = 't', default_value = "1")]
    threads: usize,
}

fn main() {
    let cli = Cli::parse();

    error_correction(&cli.reads_path, &cli.paf_path, cli.threads)
}
