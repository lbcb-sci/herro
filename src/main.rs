use clap::Parser;

use ont_haec_rs::error_correction;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    reads_path: String,
    paf_path: String,
    #[arg(short = 'w', default_value = "4096")]
    window_size: u32,
    #[arg(short = 't', default_value = "1")]
    threads: usize,
    #[arg(short = 'd', value_delimiter = ',', default_value = "0")]
    devices: Vec<usize>,
    #[arg(short = 'o', default_value = "features")]
    output: String,
    #[arg(short = 'm')]
    model: String,
}

fn main() {
    let cli = Cli::parse();

    error_correction(
        &cli.reads_path,
        &cli.paf_path,
        &cli.model,
        &cli.output,
        cli.threads,
        cli.window_size,
        &cli.devices,
    );
}
