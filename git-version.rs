//! Build script: embeds `git describe` of the source tree, so that a binary
//! reports the code it was built from, not whatever the checkout holds when
//! it runs.

use std::process::Command;

fn main() {
    let describe = Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=LLMINUS_GIT_DESCRIBE={}", describe);
    for path in [".git/HEAD", ".git/index", ".git/refs", "src"] {
        println!("cargo:rerun-if-changed={}", path);
    }
}
