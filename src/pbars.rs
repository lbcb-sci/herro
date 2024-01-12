use std::time::Duration;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

pub(super) enum PBarNotification {
    BatchLen(u64),
    Inc,
}

pub(super) fn get_parse_reads_spinner(multi: Option<&MultiProgress>) -> ProgressBar {
    let spinner = multi.map_or_else(
        || ProgressBar::new_spinner(),
        |m| m.add(ProgressBar::new_spinner()),
    );
    spinner.enable_steady_tick(Duration::from_millis(80));
    spinner.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {msg} {spinner:.blue}")
            .unwrap()
            // For more spinners check out the cli-spinners project:
            // https://github.com/sindresorhus/cli-spinners/blob/master/spinners.json
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", ""]),
    );
    spinner.set_message("Parsing reads");

    spinner
}

pub(super) fn set_parse_reads_spinner_finish(n_reads: usize, spinner: ProgressBar) {
    spinner.finish_with_message(format!("Parsed {} reads.", n_reads));
}

pub(super) fn get_alns_batches_pbar(multi: Option<&MultiProgress>) -> ProgressBar {
    let spinner = multi.map_or_else(
        || ProgressBar::new_spinner(),
        |m| m.add(ProgressBar::new_spinner()),
    );

    let spinner_style = ProgressStyle::with_template("[{elapsed_precise}] {msg} {spinner:.blue}")
        .unwrap()
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ");
    spinner.set_style(spinner_style);

    spinner.set_message("Processing 1/? batch");

    spinner
}
