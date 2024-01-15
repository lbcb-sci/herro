use std::time::Duration;

use crossbeam_channel::Receiver;
use indicatif::{FormattedDuration, MultiProgress, ProgressBar, ProgressStyle};

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

fn get_alns_batches_pbar(multi: Option<&MultiProgress>) -> ProgressBar {
    let spinner = multi.map_or_else(
        || ProgressBar::new_spinner(),
        |m| m.add(ProgressBar::new_spinner()),
    );

    spinner.enable_steady_tick(Duration::from_millis(100));
    let spinner_style = ProgressStyle::with_template("[{elapsed_precise}] {msg} {spinner:.blue}")
        .unwrap()
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ");
    spinner.set_style(spinner_style);

    spinner.set_message("Processing 1/? batch");

    spinner
}

fn get_in_batch_pbar() -> ProgressBar {
    let pbar = ProgressBar::hidden();

    pbar.set_style(
        ProgressStyle::with_template("[{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("#>-"),
    );

    pbar
}

pub(super) fn track_progress(pbar_receiver: Receiver<PBarNotification>) {
    let mbar = MultiProgress::new();
    let batches_bar = get_alns_batches_pbar(Some(&mbar));
    let mut pbar = get_in_batch_pbar();

    let mut n_batch = 0;
    while let Ok(notification) = pbar_receiver.recv() {
        match notification {
            PBarNotification::BatchLen(l) => {
                n_batch += 1;
                batches_bar.set_message(format!("Processing {}/? batch", n_batch));

                pbar.inc_length(l);

                if n_batch == 1 {
                    pbar = mbar.add(pbar);
                    pbar.set_length(l);
                }
            }
            PBarNotification::Inc => {
                pbar.inc(1);
            }
        }
    }

    batches_bar.finish_and_clear();
    pbar.finish_and_clear();

    eprintln!(
        "[{}] Processed {} reads.",
        FormattedDuration(batches_bar.elapsed()),
        pbar.position()
    );
}
