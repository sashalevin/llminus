//! Build testing a resolution without touching the merge in progress.
//!
//! The staged result is turned into a scratch commit (write-tree +
//! commit-tree with the merge's parents), checked out in a temporary
//! worktree, and the user's build command runs there. HEAD, the index and the
//! MERGE_* state of the real worktree are never touched, and no LLM has to
//! commit and undo anything.
//!
//! The build command is the user's (--build-cmd, $LLMINUS_BUILD_CMD or `git
//! config llminus.buildCmd`). It runs at the top of the scratch worktree,
//! whose HEAD is the commit to build, with `{commit}` and `{baseline}` in it
//! replaced (and LLMINUS_BUILD_COMMIT / LLMINUS_BUILD_BASELINE set): errors
//! the baseline commit already has are not the resolution's. It exits 0 if
//! the commit builds without new errors, 1 if it has new errors, 75 if it
//! could not tell (build machines unavailable, ...). A command that builds the
//! committed HEAD elsewhere and doesn't need the files (Sasha's `stable build
//! log`) can skip the checkout (--build-no-checkout).

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::time::Duration;

use crate::gitutil::git;

/// What a build test found
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildVerdict {
    /// Built, no new errors
    Pass,
    /// The build tool had already built this exact tree clean
    CachedPass,
    /// New errors; the interesting part of the output
    Fail(String),
    /// The build could not be judged (build hosts failed, timed out)
    Infra(String),
    /// The build command itself failed (bad usage, configuration)
    ToolError(String),
}

impl BuildVerdict {
    pub fn describe(&self) -> &'static str {
        match self {
            BuildVerdict::Pass => "PASS",
            BuildVerdict::CachedPass => "PASS (cached: this tree had already built clean)",
            BuildVerdict::Fail(_) => "FAIL",
            BuildVerdict::Infra(_) => "NOT JUDGED (the build hosts failed)",
            BuildVerdict::ToolError(_) => "NOT JUDGED (the build command failed)",
        }
    }

    /// For the result file
    pub fn tag(&self) -> &'static str {
        match self {
            BuildVerdict::Pass => "PASS",
            BuildVerdict::CachedPass => "CACHED_PASS",
            BuildVerdict::Fail(_) => "FAIL",
            BuildVerdict::Infra(_) => "INFRA",
            BuildVerdict::ToolError(_) => "TOOL_ERROR",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BuildSpec {
    /// The build command, with {commit} and {baseline} placeholders
    pub cmd: String,
    /// Check the files out in the scratch worktree
    pub checkout: bool,
    /// What the build covers, for the prompt (e.g. "x86_64 and arm64
    /// allmodconfig, -Werror")
    pub desc: Option<String>,
    pub timeout: Duration,
}

impl BuildSpec {
    pub fn parse(cmd: &str, checkout: bool, desc: Option<String>, timeout: Duration) -> Result<Self> {
        let words = shlex::split(cmd).with_context(|| format!("Can't parse build command: {}", cmd))?;
        if words.is_empty() {
            bail!("empty build command");
        }
        Ok(BuildSpec { cmd: cmd.to_string(), checkout, desc, timeout })
    }

    pub fn describe(&self) -> String {
        self.cmd.clone()
    }

    /// The command's words, with the placeholders filled in
    fn argv(&self, commit: &str, baseline: &str) -> Result<Vec<String>> {
        let cmd = self.cmd.replace("{commit}", commit).replace("{baseline}", baseline);
        shlex::split(&cmd).with_context(|| format!("Can't parse build command: {}", cmd))
    }
}

/// A temporary worktree, removed on drop (only this one: other worktrees of
/// the repository may belong to parallel runs)
struct ScratchWorktree {
    path: PathBuf,
    _dir: tempfile::TempDir,
}

impl ScratchWorktree {
    fn add(commit: &str, checkout: bool) -> Result<Self> {
        let dir = tempfile::Builder::new().prefix("llminus-build.").tempdir()?;
        let path = dir.path().join("wt");
        let wt = ScratchWorktree { path, _dir: dir };
        let path = wt.path.to_string_lossy().to_string();
        let mut args = vec!["worktree", "add", "--detach", "--quiet"];
        if !checkout {
            args.push("--no-checkout");
        }
        args.extend([path.as_str(), commit]);
        // constructed first, so a half-done registration is removed too
        git(&args).context("Failed to add a scratch worktree for the build")?;
        Ok(wt)
    }
}

impl Drop for ScratchWorktree {
    fn drop(&mut self) {
        if let Err(e) = git(&["worktree", "remove", "--force", &self.path.to_string_lossy()])
            && self.path.exists()
        {
            eprintln!("Warning: could not remove the scratch worktree {}: {:#}", self.path.display(), e);
        }
    }
}

/// Make a commit of the staged result with `parents`, without touching HEAD
pub fn scratch_commit(parents: &[String]) -> Result<String> {
    if crate::unmerged::any() {
        bail!("can't build: there are still unmerged paths");
    }
    let tree = git(&["write-tree"])?.trim().to_string();
    let mut args = vec!["commit-tree", tree.as_str(), "-m", "llminus scratch build"];
    for p in parents {
        args.push("-p");
        args.push(p);
    }
    Ok(git(&args)?.trim().to_string())
}

/// The lines of a build log worth showing an LLM: compiler/linker errors and
/// warnings. Where the build command reports the errors that decide its
/// verdict as "<arch>: ..." lines (as `stable build log` and ~/build.sh do,
/// ahead of full logs that also hold errors the baseline has), only those.
pub fn error_lines(output: &str) -> String {
    const ARCHES: [&str; 5] = ["x86_64: ", "x86: ", "aarch64: ", "arm64: ", "arm: "];
    let mut seen = std::collections::BTreeSet::new();
    let (mut decisive, mut lines) = (Vec::new(), Vec::new());
    for l in output.lines() {
        let low = l.to_lowercase();
        let interesting = low.contains("error") || low.contains("undefined reference") || low.contains("warning:")
            || low.contains("multiple definition of") || low.contains("no rule to make target");
        if interesting && !low.contains("waiting for unfinished jobs") && seen.insert(l.to_string()) {
            if ARCHES.iter().any(|a| l.starts_with(a)) {
                decisive.push(l);
            }
            lines.push(l);
        }
    }
    if !decisive.is_empty() {
        lines = decisive;
    }
    let mut text = lines.join("\n");
    if text.is_empty() {
        // Nothing recognisable: the tail of the log, then
        let all: Vec<&str> = output.lines().collect();
        text = all[all.len().saturating_sub(80)..].join("\n");
    }
    if text.len() > 32 * 1024 {
        text.truncate(32 * 1024);
        text.push_str("\n[... truncated]");
    }
    text
}

/// Build `commit` with `spec`, counting only errors that `baseline` doesn't
/// already have
pub fn build_commit(commit: &str, baseline: &str, spec: &BuildSpec) -> Result<BuildVerdict> {
    let argv = spec.argv(commit, baseline)?;
    // No worktree (a full disk, say): nothing was judged
    let wt = match ScratchWorktree::add(commit, spec.checkout) {
        Ok(wt) => wt,
        Err(e) => return Ok(BuildVerdict::Infra(format!("could not set up the build worktree: {:#}", e))),
    };
    eprintln!("Build testing {} with: {} (baseline {})", &commit[..12.min(commit.len())], argv.join(" "),
              &baseline[..12.min(baseline.len())]);

    let run = crate::proc::Spec {
        argv: &argv,
        cwd: &wt.path,
        env: vec![("LLMINUS_BUILD_COMMIT".into(), commit.into()),
                  ("LLMINUS_BUILD_BASELINE".into(), baseline.into())],
        env_remove: vec![],
    };
    let f = crate::proc::run(&run, crate::proc::Io { stdin: None, capture_stdout: true, capture_stderr: true },
                             spec.timeout)
        .with_context(|| format!("Failed to run the build command ({})", spec.describe()))?;
    let output = format!("{}\n{}", String::from_utf8_lossy(&f.stdout), String::from_utf8_lossy(&f.stderr));

    if f.timed_out {
        return Ok(BuildVerdict::Infra(format!("the build timed out after {}s", spec.timeout.as_secs())));
    }
    Ok(match f.status {
        Some(0) if output.lines().any(|l| l.starts_with("Build cached")) => BuildVerdict::CachedPass,
        Some(0) => BuildVerdict::Pass,
        Some(1) => BuildVerdict::Fail(error_lines(&output)),
        Some(75) => BuildVerdict::Infra(error_lines(&output)),
        Some(c) if c >= 128 => BuildVerdict::Infra(error_lines(&output)),
        None => BuildVerdict::Infra("the build was killed by a signal".to_string()),
        Some(c) => BuildVerdict::ToolError(format!("exit {}: {}", c, error_lines(&output))),
    })
}

/// Build the staged result as a commit with `parents`
pub fn build_staged(parents: &[String], baseline: &str, spec: &BuildSpec) -> Result<BuildVerdict> {
    let commit = scratch_commit(parents)?;
    build_commit(&commit, baseline, spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_lines_keeps_signatures() {
        let log = "YOU MUST WAIT\nx86_64: drivers/foo.c: error: implicit declaration of 'bar'\nmake[2]: *** Waiting for unfinished jobs....\nok\nx86_64: drivers/foo.c: error: implicit declaration of 'bar'\n";
        assert_eq!(error_lines(log), "x86_64: drivers/foo.c: error: implicit declaration of 'bar'");
    }

    #[test]
    fn error_lines_prefers_the_decisive_ones() {
        // ~/build.sh with a baseline: the new error, then the full log that
        // also holds an error the baseline has
        let log = "x86: rc 2 W: 0 ERRORS\n\n=== errors the baseline 123 does not have ===\n\
                   x86: drivers/baz.c:3:3: error: brand new\n\n=== x86 build log ===\n\
                   drivers/foo.c:12:5: error: old problem\ndrivers/baz.c:3:3: error: brand new\n";
        assert_eq!(error_lines(log), "x86: rc 2 W: 0 ERRORS\nx86: drivers/baz.c:3:3: error: brand new");
        // no arch-prefixed lines: every error line
        assert_eq!(error_lines("a.c:1: error: x\nok\nb.c:2: warning: y\n"), "a.c:1: error: x\nb.c:2: warning: y");
        // linker and make failures that don't say "error"
        let log = "x86: rc 2 W: 0 ERRORS\nx86: ld: a.o: multiple definition of `f'; b.o: first defined here\n\
                   x86: make[2]: *** No rule to make target 'c.o', needed by 'built-in.a'.  Stop.\n";
        assert_eq!(error_lines(log), log.trim_end());
    }
}
