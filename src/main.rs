use clap::{Args, Parser, Subcommand};

use ont_haec_rs::{error_correction, generate_features, AlnMode};

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Features(FeatGenArgs),
    Inference(InferenceArgs),
}

#[derive(Args)]
#[group(required = false, multiple = false)]
struct AlignmentsIO {
    #[arg(long)]
    read_alns: Option<String>,

    #[arg(long)]
    write_alns: Option<String>,
}

#[derive(Args)]
struct FeatGenArgs {
    #[command(flatten)]
    alns: AlignmentsIO,

    #[arg(short = 'w', default_value = "4096")]
    window_size: u32,

    #[arg(short = 't', default_value = "1")]
    feat_gen_threads: usize,

    reads: String,

    output: String,
}

#[derive(Args)]
struct InferenceArgs {
    #[command(flatten)]
    alns: AlignmentsIO,

    #[arg(short = 'w', default_value = "4096")]
    window_size: u32,

    #[arg(short = 't', default_value = "1")]
    feat_gen_threads: usize,

    #[arg(short = 'm')]
    model: String,

    #[arg(short = 'd', value_delimiter = ',', default_value = "0")]
    devices: Vec<usize>,

    reads: String,

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

            generate_features(
                args.reads,
                args.output,
                args.feat_gen_threads,
                args.window_size,
                mode,
            );
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
                &args.devices,
                mode,
            );
        }
    }
}
