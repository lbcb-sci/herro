use std::{
    io::Write,
    path::Path,
    process::{ChildStdout, Command, Stdio},
    thread,
};

use crate::haec_io::HAECRecord;

pub(crate) fn call_mm2<P: AsRef<Path>>(
    target: &[HAECRecord],
    query: P,
    threads: usize,
) -> ChildStdout {
    let mut child = Command::new("minimap2")
        .args([
            "-t",
            &threads.to_string(),
            "-K8g",
            "-cx",
            "ava-ont",
            "-k25",
            "-w17",
            "-e200",
            "-r150",
            "-m2500",
            "-f0.005",
            "-z200",
            "--dual=yes",
            "-",
            query.as_ref().to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start echo process");

    let mut stdin = child.stdin.take().expect("Failed to get minimap2 stdin");

    thread::scope(|s| {
        s.spawn(move || {
            let mut buffer = vec![0u8; target.iter().map(|r| r.seq.len()).max().unwrap()];

            for read in target {
                write!(stdin, ">").unwrap();
                stdin.write_all(&read.id).unwrap();
                writeln!(stdin, "\n").unwrap();

                read.seq.get_sequence(&mut buffer);
                stdin
                    .write_all(&buffer[..read.seq.len()])
                    .expect("Failed to write to minimap2 stdin");

                writeln!(stdin, "").expect("Failed to write to minimap2 stdin");
            }
        });
    });

    let stdout = child.stdout.take().expect("Failed to get minimap2 stdout");
    stdout
}
