use clap::{Args, Parser, Subcommand};

use ont_haec_rs::{error_correction, AlnMode};

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Subcommand used for generating features")]
    Features(FeatGenArgs),
    #[command(about = "Subcommand used for error-correcting reads")]
    Inference(InferenceArgs),
}

#[derive(Args)]
#[group(required = false, multiple = false)]
struct AlignmentsIO {
    #[arg(long, help = "Path to the folder containing *.oec.zst alignments")]
    read_alns: Option<String>,

    #[arg(
        long,
        help = "Path to the folder where *.oec.zst alignments will be saved"
    )]
    write_alns: Option<String>,
}

#[derive(Args)]
struct FeatGenArgs {
    #[command(flatten)]
    alns: AlignmentsIO,

    #[arg(
        short = 'w',
        default_value = "4096",
        help = "Size of the window used for target chunking (default 4096)"
    )]
    window_size: u32,

    #[arg(
        short = 't',
        default_value = "1",
        help = "Number of feature generation threads (default 1)"
    )]
    feat_gen_threads: usize,

    #[arg(help = "Path to the fastq reads (can be gzipped)")]
    reads: String,

    #[arg(help = "Path to the folder where features will be stored")]
    output: String,
}

#[derive(Args)]
struct InferenceArgs {
    #[command(flatten)]
    alns: AlignmentsIO,

    #[arg(
        short = 'w',
        default_value = "4096",
        help = "Size of the window used for target chunking (default 4096)"
    )]
    window_size: u32,

    #[arg(
        short = 't',
        default_value = "1",
        help = "Number of feature generation threads (default 1)"
    )]
    feat_gen_threads: usize,

    #[arg(short = 'm', help = "Path to the model file")]
    model: String,

    #[arg(
        short = 'd',
        value_delimiter = ',',
        default_value = "0",
        help = "List of cuda devices in format d0,d1... (e.g 0,1,3) (default 0)"
    )]
    devices: Vec<usize>,

    #[arg(short = 'b', help = "Batch size per device.")]
    batch_size: usize,

    #[arg(help = "Path to the fastq reads (can be gzipped)")]
    reads: String,

    #[arg(help = "Path to the corrected reads")]
    output: String,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Features(args) => {
            let mode = match (args.alns.read_alns, args.alns.write_alns) {
                (None, None) => AlnMode::None,
                (Some(p), None) => AlnMode::Read(p),
                (None, Some(p)) => AlnMode::Write(p),
                _ => unreachable!(),
            };

            /*generate_features(
                args.reads,
                args.output,
                args.feat_gen_threads,
                args.window_size,
                mode,
            );*/
        }
        Commands::Inference(args) => {
            let mode = match (args.alns.read_alns, args.alns.write_alns) {
                (None, None) => AlnMode::None,
                (Some(p), None) => AlnMode::Read(p),
                (None, Some(p)) => AlnMode::Write(p),
                _ => unreachable!(),
            };

            error_correction(
                args.reads,
                &args.model,
                args.output,
                args.feat_gen_threads,
                args.window_size,
                args.devices,
                args.batch_size,
                mode,
            );
        }
    }
}
