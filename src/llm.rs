//! Running the LLM.
//!
//! Built-in backends (`--llm claude`, `--llm codex`) are run with the
//! restrictions llminus relies on: file edits plus read-only git and lei
//! searches, nothing that stages, commits or builds. A generic command
//! (`--llm-cmd`, or the old positional command) gets the prompt on stdin, or
//! in a file wherever its arguments say `{prompt}`; it only gets the PATH
//! wrappers, so its restrictions are not enforced beyond that.

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::time::Duration;

use crate::proc;
use crate::shims::Shims;

#[derive(Debug, Clone)]
pub enum Backend {
    Claude { model: String, effort: String },
    /// Model and effort default to the user's Codex configuration
    Codex { model: Option<String>, effort: Option<String> },
    Command(String),
}

impl Backend {
    pub fn parse(llm: Option<&str>, llm_cmd: Option<&str>, model: Option<&str>, effort: Option<&str>) -> Result<Self> {
        match (llm, llm_cmd) {
            (Some(_), Some(_)) => bail!("use either --llm or --llm-cmd, not both"),
            (Some("claude"), None) => Ok(Backend::Claude {
                model: model.unwrap_or("opus").to_string(),
                effort: effort.unwrap_or("xhigh").to_string(),
            }),
            (Some("codex"), None) => Ok(Backend::Codex {
                model: model.map(str::to_string),
                effort: effort.map(str::to_string),
            }),
            (Some(other), None) => bail!("unknown --llm backend '{}' (claude, codex)", other),
            (None, Some(cmd)) => {
                if cmd.trim().is_empty() {
                    bail!("empty LLM command");
                }
                Ok(Backend::Command(cmd.to_string()))
            }
            (None, None) => bail!("no LLM given (--llm claude|codex, or --llm-cmd '<command>')"),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Backend::Claude { model, effort } => format!("claude ({}, effort {})", model, effort),
            Backend::Codex { model, effort } => format!("codex ({}, effort {})",
                model.as_deref().unwrap_or("model from ~/.codex/config.toml"),
                effort.as_deref().unwrap_or("from ~/.codex/config.toml")),
            Backend::Command(c) => c.clone(),
        }
    }

    /// Whether the backend can reach lore itself (codex runs without network)
    pub fn has_network(&self) -> bool {
        !matches!(self, Backend::Codex { .. })
    }
}

/// What the LLM run produced
#[derive(Debug, Default)]
pub struct LlmRun {
    /// Exit status (None if killed)
    pub status: Option<i32>,
    pub timed_out: bool,
    /// The LLM's final message, when the backend reports one
    pub final_message: String,
    pub cost_usd: Option<f64>,
}

const CLAUDE_TOOLS: &str = "Read,Edit,Write,Glob,Grep,Bash";

/// Read-only git, as Claude Code permission rules. User settings may allow
/// more (Sasha's allow `Bash(git *)`); the deny rules below win over those, and
/// the git wrapper on PATH refuses the rest.
const CLAUDE_ALLOW: &[&str] = &[
    "Read", "Edit", "Write", "Glob", "Grep",
    "Bash(git show)", "Bash(git log)", "Bash(git diff)", "Bash(git status)", "Bash(git symbolic-ref *)",
    "Bash(git show *)", "Bash(git log *)", "Bash(git diff *)", "Bash(git blame *)",
    "Bash(git grep *)", "Bash(git ls-files *)", "Bash(git ls-tree *)", "Bash(git rev-parse *)",
    "Bash(git merge-base *)", "Bash(git cat-file *)", "Bash(git describe *)", "Bash(git status *)",
    "Bash(git rev-list *)", "Bash(git name-rev *)", "Bash(git range-diff *)",
    "Bash(lei q *)",
    // to filter lei/git output in a pipeline (Claude Code requires every
    // part of a compound command to be allowed)
    "Bash(grep *)", "Bash(head *)", "Bash(tail *)", "Bash(wc *)",
];

const CLAUDE_DENY: &[&str] = &[
    "Bash(git commit *)", "Bash(git add *)", "Bash(git rm *)", "Bash(git reset *)",
    "Bash(git checkout *)", "Bash(git switch *)", "Bash(git restore *)", "Bash(git stash *)",
    "Bash(git merge *)", "Bash(git rebase *)", "Bash(git cherry-pick *)", "Bash(git revert *)",
    "Bash(git update-ref *)", "Bash(git branch *)", "Bash(git tag *)", "Bash(git push *)",
    "Bash(git fetch *)", "Bash(git pull *)", "Bash(git worktree *)", "Bash(git config *)",
    "Bash(git -c *)", "Bash(git -C *)", "Bash(/usr/bin/git *)", "Bash(/usr/bin/lei *)",
    "Bash(lei up *)", "Bash(lei edit-search *)", "Bash(lei forget-search *)",
    "Bash(stable *)", "Bash(make *)", "Bash(ssh *)", "Bash(curl *)", "Bash(wget *)", "Bash(sudo *)",
    "WebFetch", "Agent", "Task", "NotebookEdit",
];

/// Deny rules for the real binaries by their absolute paths (the PATH
/// wrappers only catch the bare names)
fn absolute_denies() -> Vec<String> {
    ["git", "lei"].iter()
        .filter_map(|n| crate::shims::which_real(n))
        .flat_map(|p| [format!("Bash({} *)", p.display()), format!("Bash({})", p.display())])
        .collect()
}

fn claude_argv(model: &str, effort: &str) -> Vec<String> {
    let mut a: Vec<String> = ["claude", "-p", "--output-format", "json", "--model", model, "--effort", effort,
        // dontAsk: anything not allowed is denied instead of prompting
        "--permission-mode", "dontAsk", "--strict-mcp-config", "--disable-slash-commands",
        "--tools", CLAUDE_TOOLS, "--allowedTools"]
        .iter().map(|s| s.to_string()).collect();
    a.extend(CLAUDE_ALLOW.iter().map(|s| s.to_string()));
    a.push("--disallowedTools".to_string());
    a.extend(CLAUDE_DENY.iter().map(|s| s.to_string()));
    a.extend(absolute_denies());
    a
}

fn codex_argv(model: Option<&str>, effort: Option<&str>, cwd: &Path, last_message: &Path) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "codex".into(), "exec".into(), "--ephemeral".into(), "--skip-git-repo-check".into(),
        "--color".into(), "never".into(),
    ];
    // Unless asked for, whatever the user's Codex configuration says
    if let Some(m) = model {
        a.extend(["-m".into(), m.into()]);
    }
    if let Some(e) = effort {
        a.extend(["-c".into(), format!("model_reasoning_effort={}", e)]);
    }
    a.extend([
        // approval_policy=never: a user config's auto-approver must never let
        // a command out of the sandbox
        "-s".into(), "workspace-write".into(), "-c".into(), "approval_policy=never".into(),
        "-c".into(), "sandbox_workspace_write.network_access=false".into(),
        "-o".into(), last_message.to_string_lossy().into(),
        "-C".into(), cwd.to_string_lossy().into(), "-".into(),
    ]);
    a
}

/// A short failed run whose message says the CLI hit a usage limit
/// (rather than failed); claude also says "(no content)" when that happens
fn is_rate_limited(status: Option<i32>, text: &str) -> bool {
    let t = text.trim();
    if t == "(no content)" {
        return true;
    }
    status != Some(0) && t.len() < 300
        && ["usage limit reached", "Session limit reached", "5-hour limit reached", "rate limit exceeded",
            "Rate limit reached"].iter().any(|m| t.contains(m))
}

pub struct RunOpts<'a> {
    pub cwd: &'a Path,
    pub timeout: Duration,
    /// How often to wait out a usage limit and try again
    pub rate_limit_retries: u32,
    pub rate_limit_wait: Duration,
    /// Extra environment for the LLM (e.g. LLMINUS_VERDICT)
    pub env: Vec<(&'a str, String)>,
    /// Files removed before every attempt (the verdict file)
    pub clear_before_attempt: Vec<std::path::PathBuf>,
}

fn spec<'a>(argv: &'a [String], opts: &'a RunOpts, path_env: &std::ffi::OsStr) -> proc::Spec<'a> {
    let mut env = vec![
        ("PATH".to_string(), path_env.to_os_string()),
        ("GIT_PAGER".to_string(), "cat".into()),
        ("PAGER".to_string(), "cat".into()),
    ];
    env.extend(opts.env.iter().map(|(k, v)| (k.to_string(), v.into())));
    proc::Spec { argv, cwd: opts.cwd, env, env_remove: vec!["GIT_EXTERNAL_DIFF"] }
}

/// Run the LLM with `prompt`
pub fn run(backend: &Backend, prompt: &str, opts: &RunOpts) -> Result<LlmRun> {
    let shims = Shims::create()?;
    let path_env = shims.path_env();
    let scratch = tempfile::Builder::new().prefix("llminus-llm.").tempdir()?;

    let mut attempt = 0;
    loop {
        attempt += 1;
        for f in &opts.clear_before_attempt {
            let _ = std::fs::remove_file(f);
        }
        let run = match backend {
            Backend::Claude { model, effort } => {
                let argv = claude_argv(model, effort);
                let f = proc::run(&spec(&argv, opts, &path_env),
                                  proc::Io { stdin: Some(prompt.as_bytes()), capture_stdout: true, capture_stderr: false },
                                  opts.timeout)?;
                let raw = String::from_utf8_lossy(&f.stdout).to_string();
                let json: Option<serde_json::Value> = serde_json::from_str(raw.trim()).ok();
                let final_message = json.as_ref()
                    .and_then(|j| j.get("result")).and_then(|r| r.as_str())
                    .map(str::to_string)
                    .unwrap_or(raw);
                let cost_usd = json.as_ref().and_then(|j| j.get("total_cost_usd")).and_then(|c| c.as_f64());
                LlmRun { status: f.status, timed_out: f.timed_out, final_message, cost_usd }
            }
            Backend::Codex { model, effort } => {
                let last = scratch.path().join("last-message");
                let _ = std::fs::remove_file(&last);
                let argv = codex_argv(model.as_deref(), effort.as_deref(), opts.cwd, &last);
                let f = proc::run(&spec(&argv, opts, &path_env),
                                  proc::Io { stdin: Some(prompt.as_bytes()), capture_stdout: false, capture_stderr: false },
                                  opts.timeout)?;
                let final_message = std::fs::read_to_string(&last).unwrap_or_default();
                LlmRun { status: f.status, timed_out: f.timed_out, final_message, cost_usd: None }
            }
            Backend::Command(cmd) => {
                let mut argv = shlex::split(cmd).with_context(|| format!("Can't parse LLM command: {}", cmd))?;
                if argv.is_empty() {
                    bail!("empty LLM command");
                }
                let prompt_file = scratch.path().join("prompt.md");
                let uses_file = argv.iter().any(|a| a.contains("{prompt}"));
                if uses_file {
                    std::fs::write(&prompt_file, prompt)?;
                    for a in &mut argv {
                        *a = a.replace("{prompt}", &prompt_file.to_string_lossy());
                    }
                }
                let stdin = (!uses_file).then_some(prompt.as_bytes());
                let f = proc::run(&spec(&argv, opts, &path_env),
                                  proc::Io { stdin, capture_stdout: true, capture_stderr: false }, opts.timeout)
                    .with_context(|| format!("Failed to run {}", argv[0]))?;
                let out = String::from_utf8_lossy(&f.stdout).to_string();
                eprint!("{}", out);
                LlmRun { status: f.status, timed_out: f.timed_out, final_message: out, cost_usd: None }
            }
        };

        if !run.timed_out && attempt <= opts.rate_limit_retries && is_rate_limited(run.status, &run.final_message) {
            eprintln!("LLM usage limit hit ({}), retrying in {}s ({}/{})",
                      run.final_message.trim(), opts.rate_limit_wait.as_secs(), attempt, opts.rate_limit_retries);
            std::thread::sleep(opts.rate_limit_wait);
            continue;
        }
        return Ok(run);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn backend_parse() {
        assert!(matches!(Backend::parse(Some("claude"), None, None, None).unwrap(), Backend::Claude { .. }));
        assert!(matches!(Backend::parse(None, Some("my-llm --x"), None, None).unwrap(), Backend::Command(_)));
        assert!(Backend::parse(Some("claude"), Some("x"), None, None).is_err());
        assert!(Backend::parse(Some("gpt"), None, None, None).is_err());
        assert!(Backend::parse(None, None, None, None).is_err());
    }

    #[test]
    fn codex_follows_the_user_config_unless_asked() {
        let (cwd, last) = (Path::new("/r"), Path::new("/t/last"));
        let a = codex_argv(None, None, cwd, last);
        assert!(!a.iter().any(|x| x == "-m" || x.starts_with("model_reasoning_effort")));
        assert!(a.windows(2).any(|w| w == ["-s", "workspace-write"]));
        assert!(a.iter().any(|x| x == "sandbox_workspace_write.network_access=false"));
        let a = codex_argv(Some("gpt-x"), Some("low"), cwd, last);
        assert!(a.windows(2).any(|w| w == ["-m", "gpt-x"]));
        assert!(a.iter().any(|x| x == "model_reasoning_effort=low"));
    }

    #[test]
    fn claude_argv_denies_mutating_git() {
        let a = claude_argv("opus", "high");
        let deny = a.iter().position(|x| x == "--disallowedTools").unwrap();
        assert!(a[deny..].iter().any(|x| x == "Bash(git commit *)"));
        assert!(a[deny..].iter().any(|x| x == "Bash(stable *)"));
        assert!(a.iter().any(|x| x == "dontAsk"));
    }

    fn opts(dir: &Path) -> RunOpts<'_> {
        RunOpts { cwd: dir, timeout: Duration::from_secs(20), rate_limit_retries: 1,
                  rate_limit_wait: Duration::from_millis(10), env: vec![("LLMINUS_TEST", "yes".into())],
                  clear_before_attempt: vec![] }
    }

    #[test]
    fn command_gets_prompt_on_stdin_or_in_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let r = run(&Backend::Command("cat".into()), "the prompt", &opts(dir.path())).unwrap();
        assert_eq!(r.final_message, "the prompt");
        let r = run(&Backend::Command("cat {prompt}".into()), "in a file", &opts(dir.path())).unwrap();
        assert_eq!(r.final_message, "in a file");
        let r = run(&Backend::Command("sh -c 'echo $LLMINUS_TEST; exit 3'".into()), "", &opts(dir.path())).unwrap();
        assert_eq!(r.final_message.trim(), "yes");
        assert_eq!(r.status, Some(3));
    }

    #[test]
    fn command_sees_the_shims() {
        let dir = tempfile::tempdir().unwrap();
        let r = run(&Backend::Command("sh -c 'git commit -m x; echo rc=$?'".into()), "", &opts(dir.path())).unwrap();
        assert!(r.final_message.contains("rc=128"), "{}", r.final_message);
    }

    #[test]
    fn a_successful_answer_mentioning_limits_is_not_retried() {
        assert!(!is_rate_limited(Some(0), "I checked the rate limit exceeded path"));
        assert!(is_rate_limited(Some(1), "Claude AI usage limit reached|1760000000"));
        assert!(is_rate_limited(Some(0), "(no content)"));
        assert!(!is_rate_limited(Some(1), &"x".repeat(400)));
    }

    #[test]
    fn timeout_kills_the_whole_group() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("survived");
        let cmd = format!("sh -c '(sleep 3; touch {}) & sleep 60'", marker.display());
        let mut o = opts(dir.path());
        o.timeout = Duration::from_secs(1);
        let start = Instant::now();
        let r = run(&Backend::Command(cmd), "", &o).unwrap();
        assert!(r.timed_out);
        assert!(start.elapsed() < Duration::from_secs(20));
        std::thread::sleep(Duration::from_secs(4));
        assert!(!marker.exists(), "a child of the LLM outlived the timeout");
    }

    #[test]
    fn rate_limit_is_retried() {
        let dir = tempfile::tempdir().unwrap();
        let count = dir.path().join("count");
        let cmd = format!("sh -c 'echo x >> {c}; [ $(wc -l < {c}) -ge 2 ] && echo done || {{ echo \"Session limit reached\"; exit 1; }}'",
                          c = count.display());
        let r = run(&Backend::Command(cmd), "", &opts(dir.path())).unwrap();
        assert_eq!(r.final_message.trim(), "done");
    }
}
