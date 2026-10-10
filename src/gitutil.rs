//! Running git.

use anyhow::{bail, Context, Result};
use std::io::Write;
use std::process::{Command, Stdio};

/// Run a git command and return stdout
pub fn git(args: &[&str]) -> Result<String> {
    Ok(String::from_utf8_lossy(&git_bytes(args)?).to_string())
}

/// Run a git command and return its raw stdout (for -z output and blobs)
pub fn git_bytes(args: &[&str]) -> Result<Vec<u8>> {
    git_bytes_env(args, &[])
}

/// git_bytes() with extra environment (e.g. GIT_INDEX_FILE)
pub fn git_bytes_env(args: &[&str], env: &[(&str, &str)]) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .args(args)
        .envs(env.iter().copied())
        .output()
        .context("Failed to run git")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git {} failed: {}", args.join(" "), stderr.trim());
    }

    Ok(output.stdout)
}

/// Run a git command with `input` on stdin and return stdout
pub fn git_with_input(args: &[&str], input: &[u8]) -> Result<String> {
    let mut child = Command::new("git")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to run git")?;
    let mut stdin = child.stdin.take().expect("piped stdin");
    let input = input.to_vec();
    // Write from a thread so a large input can't deadlock against stdout
    let writer = std::thread::spawn(move || stdin.write_all(&input));
    let output = child.wait_with_output().context("Failed to wait for git")?;
    writer.join().expect("stdin writer panicked").context("Failed to write to git")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git {} failed: {}", args.join(" "), stderr.trim());
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Run a git command, return stdout, allow failure
pub fn git_allow_fail(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
}

/// The object name `rev` resolves to, if it exists
pub fn rev_parse(rev: &str) -> Option<String> {
    git_allow_fail(&["rev-parse", "--verify", "-q", rev])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Path of a file in the git directory. In a worktree .git is a file, so
/// per-worktree files like MERGE_MSG don't live under .git/.
pub fn git_path(name: &str) -> String {
    git_allow_fail(&["rev-parse", "--git-path", name])
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| format!(".git/{}", name))
}

/// The top of the working tree
pub fn toplevel() -> Result<String> {
    Ok(git(&["rev-parse", "--show-toplevel"])?.trim().to_string())
}

/// Check we're in a git repository
pub fn check_repo() -> Result<()> {
    git(&["rev-parse", "--git-dir"])?;
    Ok(())
}
