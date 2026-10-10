//! llminus - LLM-assisted merge conflict resolution for the Linux kernel

mod build;
mod context;
mod eval;
mod fixups;
mod gitutil;
mod history;
mod llm;
mod nextref;
mod lore;
mod outcome;
mod proc;
mod prompt;
mod pull;
mod resolve;
mod shims;
mod state;
mod unmerged;
mod verdict;
#[cfg(test)]
mod testutil;

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use std::time::Duration;

use crate::build::BuildSpec;
use crate::llm::Backend;
use crate::outcome::Exit;

/// Default maximum tokens for prompt (conservative for broad provider compatibility)
/// Most providers support at least 128K; we use 100K as a safe default.
const DEFAULT_MAX_TOKENS: usize = 100_000;

#[derive(Parser)]
#[command(name = "llminus")]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("LLMINUS_GIT_DESCRIBE"), ")"))]
#[command(about = "LLM-assisted merge conflict resolution for the Linux kernel")]
#[command(after_help = "Exit status (resolve, pull): 0 resolved and staged, 1 error, \
                        2 needs a human, 3 the build hosts failed (nothing judged)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// How to run the LLM and the build
#[derive(Args, Debug, Clone)]
struct RunArgs {
    /// Built-in LLM backend: claude or codex (restricted to file edits,
    /// read-only git and lei searches)
    #[arg(long)]
    llm: Option<String>,
    /// LLM command instead: gets the prompt on stdin, or in a file wherever an
    /// argument says {prompt}. Only the PATH wrappers restrict it.
    #[arg(long)]
    llm_cmd: Option<String>,
    /// Model for --llm [claude: opus; codex: the one in ~/.codex/config.toml]
    #[arg(long)]
    model: Option<String>,
    /// Reasoning effort for --llm [claude: xhigh; codex: the one in ~/.codex/config.toml]
    #[arg(long)]
    effort: Option<String>,
    /// Build command, run at the top of a scratch worktree whose HEAD is the
    /// commit to build; {commit} and {baseline} are replaced, and errors the
    /// baseline already has don't count. Exit 0 clean, 1 new errors, 75
    /// couldn't tell. [default: $LLMINUS_BUILD_CMD, git config
    /// llminus.buildCmd, or ~/build.sh; with none, nothing is build tested]
    #[arg(long)]
    build_cmd: Option<String>,
    /// Don't check the files out for the build command (it builds the
    /// committed HEAD elsewhere) [or $LLMINUS_BUILD_NO_CHECKOUT=1, git config
    /// llminus.buildNoCheckout]
    #[arg(long)]
    build_no_checkout: bool,
    /// What the build covers, for the prompt (e.g. "x86_64 and arm64
    /// allmodconfig, -Werror") [or $LLMINUS_BUILD_DESC, git config
    /// llminus.buildDesc]
    #[arg(long)]
    build_desc: Option<String>,
    /// Don't build test the resolution
    #[arg(long)]
    no_build: bool,
    /// How many times the LLM gets to fix new build errors
    #[arg(long, default_value_t = 2)]
    repair_rounds: u32,
    /// Time limit for each LLM run (e.g. 45m, 2h, 600s)
    #[arg(long, default_value = "45m")]
    timeout: String,
    /// Time limit for each build
    #[arg(long, default_value = "3h")]
    build_timeout: String,
    /// How often to wait out an LLM usage limit
    #[arg(long, default_value_t = 12)]
    rate_limit_retries: u32,
    /// Maximum tokens for prompt (reduces historical examples if exceeded)
    #[arg(long, default_value_t = DEFAULT_MAX_TOKENS)]
    max_tokens: usize,
    /// Don't record the resolution for later replay
    #[arg(long)]
    no_record: bool,
    /// Accept a resolution that differs from linux-next's resolution of the
    /// very same conflict (by default that goes to a human)
    #[arg(long)]
    allow_next_differ: bool,
    /// The pull request mail being merged, for the prompt (message-id, or a
    /// file with the raw mail)
    #[arg(long)]
    pr_mail: Option<String>,
    /// Fetch linux-next's "manual merge of" reports for the conflicted files
    /// from lore into the prompt (always on for --llm codex, which has no network)
    #[arg(long)]
    prefetch_lore: bool,
    /// Leave each side's commits and the one-sided identifiers out of the
    /// prompt (to measure what they're worth)
    #[arg(long)]
    no_side_context: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Learn from historical merge conflict resolutions
    Learn {
        /// Git revision range (e.g., "v6.0..v6.1"). If not specified, learns from entire history.
        range: Option<String>,
    },
    /// Generate embeddings for stored resolutions (for RAG similarity search)
    Vectorize {
        /// How many resolutions to embed between saves (default: 64)
        #[arg(short, long, default_value = "64")]
        batch_size: usize,
    },
    /// Find similar historical conflict resolutions for current conflicts
    Find {
        /// Number of similar resolutions to show (default: 1)
        #[arg(default_value = "1")]
        n: usize,
    },
    /// Resolve the merge in progress with an LLM, or fix the build of the
    /// merge at HEAD when no merge is in progress
    Resolve {
        /// LLM command (same as --llm-cmd; kept for older callers)
        command: Option<String>,
        #[command(flatten)]
        run: RunArgs,
    },
    /// Replay historical merges that needed a manual resolution and compare
    /// llminus's resolutions with the real ones (spends LLM budget)
    Eval {
        /// Revision range to take merges from (e.g. v7.2..v7.3-rc1)
        range: String,
        /// Take all merges, not just first-parent (mainline) ones
        #[arg(long)]
        all_merges: bool,
        /// Only merges committed by this person (e.g. "Linus Torvalds")
        #[arg(long)]
        committer: Option<String>,
        /// Only these merges
        #[arg(long)]
        only: Vec<String>,
        /// At most this many merges
        #[arg(long)]
        limit: Option<usize>,
        /// Where the results, logs and report go [default: ./llminus-eval-<time>]
        #[arg(long)]
        out: Option<std::path::PathBuf>,
        /// Build test the resolutions too (a full build per merge)
        #[arg(long)]
        build: bool,
        /// Keep the scratch repositories
        #[arg(long)]
        keep: bool,
        /// linux-next reference: "tags" uses the newest next-* tag older than
        /// each merge, "none" none
        #[arg(long, default_value = "none")]
        next_ref: String,
        /// Required: each merge is a full LLM run
        #[arg(long)]
        i_understand_this_spends_ai_budget: bool,
        #[command(flatten)]
        run: RunArgs,
    },
    /// Pull a pull request from lore.kernel.org and merge it
    Pull {
        /// Message ID of the pull request (e.g., "98b74397-05bc-dbee-cab4-3f40d643eaac@kernel.org")
        message_id: String,
        /// LLM command (same as --llm-cmd)
        #[arg(short, long)]
        command: Option<String>,
        #[command(flatten)]
        run: RunArgs,
    },
}

/// Parse "45m", "2h", "600s" or plain seconds
fn parse_duration(s: &str) -> Result<Duration> {
    let s = s.trim();
    let (num, mult) = match s.char_indices().last() {
        Some((i, 'h')) => (&s[..i], 3600),
        Some((i, 'm')) => (&s[..i], 60),
        Some((i, 's')) => (&s[..i], 1),
        _ => (s, 1),
    };
    let n: u64 = num.trim().parse().map_err(|_| anyhow::anyhow!("bad duration '{}'", s))?;
    Ok(Duration::from_secs(n * mult))
}

/// A setting: the command line's, else the environment's, else git config's
fn setting(arg: Option<&String>, env: &str, git_key: &str) -> Option<String> {
    arg.cloned()
        .or_else(|| std::env::var(env).ok())
        .or_else(|| gitutil::git_allow_fail(&["config", "--get", git_key]))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// The build command used when none is configured
const DEFAULT_BUILD_CMD: &str = "~/build.sh";

/// `~/...` in the command's first word is the home directory
fn expand_home(cmd: &str) -> String {
    match (cmd.strip_prefix("~/"), std::env::var("HOME")) {
        (Some(rest), Ok(home)) => format!("{}/{}", home.trim_end_matches('/'), rest),
        _ => cmd.to_string(),
    }
}

/// The build command, if one is configured (or ~/build.sh exists)
fn build_spec(run: &RunArgs) -> Result<Option<BuildSpec>> {
    let cmd = match setting(run.build_cmd.as_ref(), "LLMINUS_BUILD_CMD", "llminus.buildCmd") {
        Some(cmd) => expand_home(&cmd),
        None => {
            let default = expand_home(DEFAULT_BUILD_CMD);
            if !shims::is_executable(std::path::Path::new(&default)) {
                return Ok(None);
            }
            default
        }
    };
    let no_checkout = run.build_no_checkout
        || setting(None, "LLMINUS_BUILD_NO_CHECKOUT", "llminus.buildNoCheckout")
            .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"));
    let desc = setting(run.build_desc.as_ref(), "LLMINUS_BUILD_DESC", "llminus.buildDesc");
    Ok(Some(BuildSpec::parse(&cmd, !no_checkout, desc, parse_duration(&run.build_timeout)?)?))
}

fn resolve_opts(run: &RunArgs, command: Option<&str>) -> Result<resolve::Opts> {
    if command.is_some() && run.llm_cmd.is_some() {
        bail!("give the LLM command once (positional/-c or --llm-cmd)");
    }
    let llm_cmd = run.llm_cmd.as_deref().or(command);
    let backend = Backend::parse(run.llm.as_deref(), llm_cmd, run.model.as_deref(), run.effort.as_deref())?;
    let build = if run.no_build { None } else { build_spec(run)? };
    Ok(resolve::Opts {
        backend,
        llm_timeout: parse_duration(&run.timeout)?,
        rate_limit_retries: run.rate_limit_retries,
        rate_limit_wait: Duration::from_secs(300),
        build,
        repair_rounds: run.repair_rounds,
        max_tokens: run.max_tokens,
        record_fixups: !run.no_record,
        allow_next_differ: run.allow_next_differ,
        pr_mail: run.pr_mail.clone(),
        prefetch_lore: run.prefetch_lore || run.llm.as_deref() == Some("codex"),
        side_context: !run.no_side_context,
    })
}

fn warn_if_unbuilt(opts: &resolve::Opts, run: &RunArgs) {
    if opts.build.is_none() && !run.no_build {
        eprintln!("No build command (--build-cmd, $LLMINUS_BUILD_CMD, git config llminus.buildCmd, or an \
                   executable ~/build.sh): the resolution won't be build tested.");
    }
}

fn run_resolve(command: Option<&str>, run: &RunArgs) -> Result<Exit> {
    outcome::clear_result();
    let opts = resolve_opts(run, command)?;
    warn_if_unbuilt(&opts, run);
    let res = resolve::resolve(&opts, None)?;
    match res.exit {
        Exit::Done => {
            let fields = [
                ("mode", res.mode.to_string()),
                ("staged", if res.staged { "yes" } else { "no" }.to_string()),
                ("build", res.build.as_ref().map(|b| b.tag()).unwrap_or("NOT_BUILT").to_string()),
                ("parents", res.parents.join(" ")),
                ("next", res.next.unwrap_or("NONE").to_string()),
                // no success without the tree it is for
                ("tree", gitutil::git(&["write-tree"])?.trim().to_string()),
            ];
            outcome::write_result(&fields, &outcome::commit_text(&res.notes))?;
            eprintln!("\nResolved. Notes for the commit message are in {}", gitutil::git_path(outcome::RESULT_FILE));
        }
        _ => eprintln!("\nNot resolved ({:?}): {}", res.exit, res.reason),
    }
    Ok(res.exit)
}

/// `llminus resolve` arguments that reproduce `run` (for eval's child runs,
/// which run in scratch repositories without the user's git config)
fn resolve_args(run: &RunArgs, build: Option<&BuildSpec>) -> Vec<String> {
    let mut a = Vec::new();
    let mut opt = |k: &str, v: &Option<String>| if let Some(v) = v { a.push(k.to_string()); a.push(v.clone()); };
    opt("--llm", &run.llm);
    opt("--llm-cmd", &run.llm_cmd);
    opt("--model", &run.model);
    opt("--effort", &run.effort);
    if let Some(b) = build {
        opt("--build-cmd", &Some(b.cmd.clone()));
        opt("--build-desc", &b.desc);
        if !b.checkout {
            a.push("--build-no-checkout".to_string());
        }
    }
    for (k, v) in [("--repair-rounds", run.repair_rounds.to_string()), ("--timeout", run.timeout.clone()),
                   ("--build-timeout", run.build_timeout.clone()), ("--max-tokens", run.max_tokens.to_string()),
                   ("--rate-limit-retries", run.rate_limit_retries.to_string())] {
        a.push(k.to_string());
        a.push(v);
    }
    if build.is_none() {
        a.push("--no-build".to_string());
    }
    if run.allow_next_differ {
        a.push("--allow-next-differ".to_string());
    }
    if run.prefetch_lore {
        a.push("--prefetch-lore".to_string());
    }
    if run.no_side_context {
        a.push("--no-side-context".to_string());
    }
    a.push("--no-record".to_string());
    a
}

#[allow(clippy::too_many_arguments)]
fn run_eval(range: String, all_merges: bool, committer: Option<String>, only: Vec<String>, limit: Option<usize>,
            out: Option<std::path::PathBuf>, build: bool, keep: bool, next_ref: &str, budget_ok: bool,
            run: RunArgs) -> Result<Exit> {
    let next_tags = match next_ref {
        "none" => false,
        "tags" => true,
        other => bail!("--next-ref must be tags or none, not {}", other),
    };
    if !budget_ok {
        bail!("eval runs the LLM once per merge (or more, with repair rounds); \
               pass --i-understand-this-spends-ai-budget");
    }
    // check the backend arguments before replaying anything
    let opts = resolve_opts(&run, None)?;
    if build && opts.build.is_none() {
        bail!("eval --build needs a build command (--build-cmd, $LLMINUS_BUILD_CMD, git config llminus.buildCmd, \
               or ~/build.sh)");
    }
    let build_spec = if build { opts.build.clone() } else { None };
    let out_dir = out.unwrap_or_else(|| {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        std::path::PathBuf::from(format!("llminus-eval-{}", t))
    });
    let out_dir = std::path::absolute(&out_dir)?;
    let opts = eval::EvalOpts {
        range: range.clone(),
        first_parent: !all_merges,
        committer,
        only,
        limit,
        resolve_args: resolve_args(&run, build_spec.as_ref()),
        copy_remotes: build,
        out_dir: out_dir.clone(),
        keep,
        next_tags,
    };
    let results = eval::run(&opts)?;
    let header = format!("- range: `{}`{}\n- llminus {} resolve {}\n- history database: {}\n- linux-next reference: {}\n",
        range, if all_merges { "" } else { " (first-parent merges)" },
        env!("LLMINUS_GIT_DESCRIBE"),
        opts.resolve_args.join(" "),
        std::env::var("LLMINUS_DB").unwrap_or_else(|_| "none".into()), next_ref);
    let report_path = out_dir.join("report.md");
    std::fs::write(&report_path, eval::report(&results, &header))?;
    eprintln!("\nReport: {}", report_path.display());
    Ok(Exit::Done)
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Learn { range } => history::learn(range.as_deref()).map(|_| Exit::Done),
        Commands::Vectorize { batch_size } => history::vectorize(batch_size).map(|_| Exit::Done),
        Commands::Find { n } => history::find(n).map(|_| Exit::Done),
        Commands::Resolve { command, run } => run_resolve(command.as_deref(), &run),
        Commands::Eval { range, all_merges, committer, only, limit, out, build, keep, next_ref,
                         i_understand_this_spends_ai_budget, run } =>
            run_eval(range, all_merges, committer, only, limit, out, build, keep, &next_ref,
                     i_understand_this_spends_ai_budget, run),
        Commands::Pull { message_id, command, run } => {
            outcome::clear_result();
            resolve_opts(&run, command.as_deref()).and_then(|opts| {
                warn_if_unbuilt(&opts, &run);
                pull::pull(&message_id, &opts)
            })
        }
    };

    let code = match result {
        Ok(exit) => exit as i32,
        Err(e) => {
            eprintln!("Error: {:#}", e);
            Exit::Error as i32
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn verify_cli() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_learn_command_parses() {
        let cli = Cli::try_parse_from(["llminus", "learn"]).unwrap();
        match cli.command {
            Commands::Learn { range } => assert!(range.is_none()),
            _ => panic!("Expected Learn command"),
        }
    }

    #[test]
    fn test_learn_command_with_range() {
        let cli = Cli::try_parse_from(["llminus", "learn", "v6.0..v6.1"]).unwrap();
        match cli.command {
            Commands::Learn { range } => assert_eq!(range, Some("v6.0..v6.1".to_string())),
            _ => panic!("Expected Learn command"),
        }
    }

    #[test]
    fn test_vectorize_command_parses() {
        let cli = Cli::try_parse_from(["llminus", "vectorize", "-b", "128"]).unwrap();
        match cli.command {
            Commands::Vectorize { batch_size } => assert_eq!(batch_size, 128),
            _ => panic!("Expected Vectorize command"),
        }
    }

    #[test]
    fn test_find_command_with_n() {
        let cli = Cli::try_parse_from(["llminus", "find", "5"]).unwrap();
        match cli.command {
            Commands::Find { n } => assert_eq!(n, 5),
            _ => panic!("Expected Find command"),
        }
    }

    #[test]
    fn test_resolve_positional_command_still_works() {
        let cli = Cli::try_parse_from(["llminus", "resolve", "my-llm --model fancy"]).unwrap();
        match cli.command {
            Commands::Resolve { command, run } => {
                let opts = resolve_opts(&run, command.as_deref()).unwrap();
                assert!(matches!(opts.backend, Backend::Command(ref c) if c == "my-llm --model fancy"));
                assert_eq!(opts.repair_rounds, 2);
            }
            _ => panic!("Expected Resolve command"),
        }
    }

    #[test]
    fn test_resolve_options() {
        let cli = Cli::try_parse_from(["llminus", "resolve", "--llm", "claude", "--no-build", "--timeout", "10m"]).unwrap();
        match cli.command {
            Commands::Resolve { command, run } => {
                let opts = resolve_opts(&run, command.as_deref()).unwrap();
                assert!(matches!(opts.backend, Backend::Claude { .. }));
                assert!(opts.build.is_none());
                assert_eq!(opts.llm_timeout, Duration::from_secs(600));
            }
            _ => panic!("Expected Resolve command"),
        }
        let cli = Cli::try_parse_from(["llminus", "resolve", "x", "--llm-cmd", "y"]).unwrap();
        match cli.command {
            Commands::Resolve { command, run } => assert!(resolve_opts(&run, command.as_deref()).is_err()),
            _ => panic!("Expected Resolve command"),
        }
    }

    #[test]
    fn test_expand_home() {
        let home = std::env::var("HOME").unwrap();
        assert_eq!(expand_home("~/build.sh --x"), format!("{}/build.sh --x", home.trim_end_matches('/')));
        assert_eq!(expand_home("make -j8"), "make -j8");
    }

    #[test]
    fn test_build_command_options() {
        let cli = Cli::try_parse_from(["llminus", "resolve", "--llm", "claude", "--build-cmd",
                                       "env BASE={baseline} stable build log", "--build-no-checkout",
                                       "--build-desc", "x86_64 allmodconfig"]).unwrap();
        match cli.command {
            Commands::Resolve { command, run } => {
                let b = resolve_opts(&run, command.as_deref()).unwrap().build.unwrap();
                assert_eq!(b.cmd, "env BASE={baseline} stable build log");
                assert!(!b.checkout);
                assert_eq!(b.desc.as_deref(), Some("x86_64 allmodconfig"));
                let args = resolve_args(&run, Some(&b));
                assert!(args.windows(2).any(|w| w == ["--build-cmd", "env BASE={baseline} stable build log"]));
                assert!(args.iter().any(|a| a == "--build-no-checkout"));
                assert!(!args.iter().any(|a| a == "--no-build"));
            }
            _ => panic!("Expected Resolve command"),
        }
    }

    #[test]
    fn test_pull_command_parses() {
        let cli = Cli::try_parse_from(["llminus", "pull", "test@kernel.org", "-c", "my-llm --model fancy"]).unwrap();
        match cli.command {
            Commands::Pull { message_id, command, .. } => {
                assert_eq!(message_id, "test@kernel.org");
                assert_eq!(command.as_deref(), Some("my-llm --model fancy"));
            }
            _ => panic!("Expected Pull command"),
        }
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(parse_duration("45m").unwrap(), Duration::from_secs(2700));
        assert_eq!(parse_duration("2h").unwrap(), Duration::from_secs(7200));
        assert_eq!(parse_duration("90").unwrap(), Duration::from_secs(90));
        assert!(parse_duration("soon").is_err());
    }
}
