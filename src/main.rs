use clap::{Args, Parser, Subcommand};

use ont_haec_rs::{error_correction, generate_features};

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
struct FeatGenArgs {
    #[arg(short = 'o')]
    overlaps: Option<String>,

    #[arg(short = 'w', default_value = "4096")]
    window_size: u32,

    #[arg(short = 't', default_value = "1")]
    feat_gen_threads: usize,

    reads: String,

    output: String,
}

#[derive(Args)]
struct InferenceArgs {
    #[arg(short = 'o')]
    overlaps: Option<String>,

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
            generate_features(
                args.reads,
                args.overlaps,
                args.output,
                args.feat_gen_threads,
                args.window_size,
            );
        }
        Commands::Inference(args) => error_correction(
            args.reads,
            args.overlaps,
            &args.model,
            args.output,
            args.feat_gen_threads,
            args.window_size,
            &args.devices,
        ),
    }
}
