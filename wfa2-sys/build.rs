// Similar to https://github.com/pairwise-alignment/rust-wfa2

use std::{fs, path::Path};

const ROOT: &str = "WFA2-lib";

const DIRS: &[&str] = &["utils", "system", "alignment", "wavefront"];

fn main() {
    let mut cfg = cc::Build::new();

    cfg.flag("-march=native");
    cfg.warnings(false);

    let root = Path::new(ROOT);
    DIRS.iter()
        .flat_map(|d| fs::read_dir(root.join(d)).unwrap())
        .for_each(|p| {
            let p = p.unwrap().path();
            if p.extension().map(|e| e == "c").unwrap_or(false) {
                cfg.file(&p);
                println!("cargo:rerun-if-changed={}", p.display());
            }
        });

    cfg.include(ROOT);

    cfg.compile("wfa2");
}
