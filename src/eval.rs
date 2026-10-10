//! `llminus eval`: replay historical merges that needed a manual resolution
//! and compare what llminus produces with what was actually committed.
//!
//! Every merge is redone in its own scratch repository that borrows the
//! objects of the current one (alternates) but has refs only for the merge's
//! two parents, so the answer is not reachable from any ref the LLM can list
//! (`git log --all` won't show it). The resolution database, if used, is
//! limited to merges that are ancestors of the first parent. Builds are off
//! unless --build is given (each one is a full build on the build hosts).

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use crate::gitutil::{git, git_allow_fail, rev_parse};

/// A merge to replay
#[derive(Debug, Clone, Serialize)]
pub struct Case {
    pub merge: String,
    pub subject: String,
    pub date: String,
    pub committer: String,
    /// Paths git can't merge automatically
    pub conflicted: Vec<String>,
}

/// How llminus's result for one path compares with the real merge
#[derive(Debug, Clone, Serialize)]
pub struct PathScore {
    pub path: String,
    /// exact, whitespace (same apart from whitespace), reordered (the same
    /// lines in another order), different, or unresolved (llminus didn't
    /// produce a result)
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub case: Case,
    pub exit: i32,
    /// The "Not resolved" reason, when not resolved
    pub reason: String,
    pub seconds: u64,
    pub cost_usd: Option<f64>,
    pub paths: Vec<PathScore>,
    /// The whole staged tree equals the real merge's tree
    pub tree_match: bool,
    /// Files outside the conflicts that llminus or the real merge changed
    /// (relative to git's automatic merge): "same" (as the real merge),
    /// "different", "llminus only" or "missed" (only the real merge did)
    pub fixups: Vec<PathScore>,
    /// Diffs of the paths that differ (llminus's result vs the real merge)
    #[serde(skip_serializing_if = "String::is_empty")]
    pub divergence: String,
}

pub struct EvalOpts {
    pub range: String,
    /// Only first-parent (mainline) merges
    pub first_parent: bool,
    pub committer: Option<String>,
    pub only: Vec<String>,
    pub limit: Option<usize>,
    /// Arguments for `llminus resolve` (backend, timeouts, --no-build, ...)
    pub resolve_args: Vec<String>,
    /// Copy these remotes' configuration into the scratch repos (the build
    /// hosts `stable build log` pushes to)
    pub copy_remotes: bool,
    pub out_dir: PathBuf,
    pub keep: bool,
    /// Give each merge the newest next-* tag older than it as the
    /// linux-next reference
    pub next_tags: bool,
}

/// (commit date, commit) of every next-* tag, newest first
fn next_tags() -> Vec<(i64, String)> {
    git_allow_fail(&["for-each-ref", "--sort=-creatordate", "--format=%(creatordate:unix) %(refname)", "refs/tags/next-*"])
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.split_once(' '))
        .filter_map(|(t, r)| Some((t.parse().ok()?, rev_parse(&format!("{}^{{commit}}", r))?)))
        .collect()
}

/// The merges in the range that git can't merge automatically
pub fn select_cases(opts: &EvalOpts) -> Result<Vec<Case>> {
    let mut args = vec!["rev-list", "--merges", "--reverse"];
    if opts.first_parent {
        args.push("--first-parent");
    }
    args.push(&opts.range);
    let merges: Vec<String> = if opts.only.is_empty() {
        git(&args)?.lines().map(str::to_string).collect()
    } else {
        opts.only.iter().filter_map(|m| rev_parse(m)).collect()
    };

    let mut cases = Vec::new();
    for m in merges {
        let meta = git(&["log", "-1", "--format=%s%x00%cI%x00%cn%x00%P", &m])?;
        let mut f = meta.trim_end().split('\0');
        let (subject, date, committer, parents) = (f.next().unwrap_or(""), f.next().unwrap_or(""),
                                                   f.next().unwrap_or(""), f.next().unwrap_or(""));
        let parents: Vec<&str> = parents.split_whitespace().collect();
        if parents.len() != 2 {
            continue;
        }
        if opts.committer.as_ref().is_some_and(|c| committer != c) {
            continue;
        }
        // merge-tree exits 1 on conflicts; with --name-only the conflicted
        // paths follow the tree id, up to the blank line
        let out = Command::new("git")
            .args(["merge-tree", "--write-tree", "--name-only", "--no-messages", parents[0], parents[1]])
            .output()?;
        if out.status.code() != Some(1) {
            continue;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let conflicted: Vec<String> = text.lines().skip(1).take_while(|l| !l.is_empty()).map(str::to_string).collect();
        cases.push(Case { merge: m.clone(), subject: subject.into(), date: date.into(), committer: committer.into(),
                          conflicted });
        if opts.limit.is_some_and(|l| cases.len() >= l) {
            break;
        }
    }
    Ok(cases)
}

fn git_in(dir: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git").current_dir(dir).args(args).output().context("Failed to run git")?;
    if !out.status.success() {
        bail!("git {} failed in {}: {}", args.join(" "), dir.display(), String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// A scratch repository with the merge redone up to its conflicts
fn prepare(case: &Case, dir: &Path, copy_remotes: bool) -> Result<()> {
    let objects = Path::new(git(&["rev-parse", "--git-common-dir"])?.trim()).canonicalize()?.join("objects");
    let p1 = rev_parse(&format!("{}^1", case.merge)).context("no first parent")?;
    let p2 = rev_parse(&format!("{}^2", case.merge)).context("no second parent")?;

    std::fs::create_dir_all(dir)?;
    git_in(dir, &["init", "-q", "-b", "eval"])?;
    std::fs::write(dir.join(".git/objects/info/alternates"), format!("{}\n", objects.display()))?;
    for (k, v) in [("user.name", "llminus eval"), ("user.email", "eval@llminus.invalid"),
                   ("rerere.enabled", "false"), ("merge.conflictStyle", "diff3"), ("gc.auto", "0")] {
        git_in(dir, &["config", k, v])?;
    }
    if copy_remotes {
        // only the build hosts' remotes (stable-build pushes to them)
        if let Some(cfg) = git_allow_fail(&["config", "--get-regexp", r"^remote\.build[^.]*\.url$"]) {
            for l in cfg.lines() {
                if let Some((k, v)) = l.split_once(' ') {
                    git_in(dir, &["config", k, v])?;
                }
            }
        }
    }
    git_in(dir, &["update-ref", "refs/heads/eval", &p1])?;
    git_in(dir, &["reset", "-q", "--hard", "eval"])?;
    // the merge message as the real merge had it (llminus reads the source from it)
    let msg = git(&["log", "-1", "--format=%s", &case.merge])?;
    let merged = Command::new("git").current_dir(dir)
        .args(["merge", "--no-ff", "-q", "-m", msg.trim(), &p2])
        .stdout(Stdio::null()).stderr(Stdio::null())
        .status()?;
    if merged.success() {
        bail!("{} merged cleanly when redone", &case.merge[..12]);
    }
    Ok(())
}

fn blob(dir: &Path, rev: &str) -> Option<String> {
    git_in(dir, &["rev-parse", "--verify", "-q", rev]).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn normalized(dir: &Path, oid: &str) -> String {
    git_in(dir, &["cat-file", "blob", oid]).unwrap_or_default()
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Compare the staged result in `dir` with the real merge
fn score(case: &Case, dir: &Path, resolved: bool) -> (Vec<PathScore>, bool, Vec<PathScore>, String) {
    let mut divergence = String::new();
    let paths = case.conflicted.iter().map(|p| {
        let real = blob(dir, &format!("{}:{}", case.merge, p));
        let ours = if resolved { blob(dir, &format!(":0:{}", p)) } else { None };
        let status = if !resolved {
            "unresolved"
        } else if real == ours {
            "exact"
        } else if let (Some(r), Some(o)) = (&real, &ours) {
            let (nr, no) = (normalized(dir, r), normalized(dir, o));
            let sorted = |s: &str| { let mut v: Vec<&str> = s.lines().collect(); v.sort_unstable(); v.join("\n") };
            if nr == no {
                "whitespace"
            } else if sorted(&nr) == sorted(&no) {
                if let Ok(d) = git_in(dir, &["diff", r, o]) {
                    divergence.push_str(&format!("### {} (real merge -> llminus; same lines, other order)\n\n```diff\n{}```\n\n", p, d));
                }
                "reordered"
            } else {
                if let Ok(d) = git_in(dir, &["diff", r, o]) {
                    divergence.push_str(&format!("### {} (real merge -> llminus)\n\n```diff\n{}```\n\n", p, d));
                }
                "different"
            }
        } else {
            divergence.push_str(&format!("### {}\n\nreal merge: {}, llminus: {}\n\n", p,
                                         if real.is_some() { "kept" } else { "deleted" },
                                         if ours.is_some() { "kept" } else { "deleted" }));
            "different"
        };
        PathScore { path: p.clone(), status: status.into() }
    }).collect();

    let mut tree_match = false;
    let mut fixups = Vec::new();
    if resolved && let Ok(tree) = git_in(dir, &["write-tree"]) {
        tree_match = Some(tree.trim().to_string()) == blob(dir, &format!("{}^{{tree}}", case.merge));
        // What llminus and the real merge changed besides the conflicts,
        // relative to git's automatic merge
        let outside = |args: &[&str]| -> Vec<String> {
            git_in(dir, args).map(|names| names.lines()
                .filter(|n| !case.conflicted.iter().any(|c| c == n)).map(str::to_string).collect())
                .unwrap_or_default()
        };
        let ours_changed = outside(&["diff", "--cached", "--name-only", "AUTO_MERGE"]);
        let real_changed = outside(&["diff", "--name-only", "AUTO_MERGE", &case.merge]);
        let all: std::collections::BTreeSet<&String> = ours_changed.iter().chain(&real_changed).collect();
        for p in all {
            let real = blob(dir, &format!("{}:{}", case.merge, p));
            let ours = blob(dir, &format!(":0:{}", p));
            let status = if real == ours {
                "same"
            } else if !ours_changed.contains(p) {
                "missed"
            } else if !real_changed.contains(p) {
                "llminus only"
            } else {
                "different"
            };
            if status != "same" {
                let what = format!("{} (outside the conflicts, {}; real merge -> llminus)", p, status);
                match (&real, &ours) {
                    (Some(r), Some(o)) => if let Ok(d) = git_in(dir, &["diff", r, o]) {
                        divergence.push_str(&format!("### {}\n\n```diff\n{}```\n\n", what, d));
                    },
                    _ => divergence.push_str(&format!("### {}\n\nreal merge: {}, llminus: {}\n\n", what,
                                                      if real.is_some() { "has it" } else { "doesn't have it" },
                                                      if ours.is_some() { "has it" } else { "doesn't have it" })),
                }
            }
            fixups.push(PathScore { path: p.clone(), status: status.into() });
        }
    }
    (paths, tree_match, fixups, divergence)
}

/// "LLM finished in 150s (exit Some(0), $0.59)" -> 0.59, summed over rounds
fn cost_from_log(log: &str) -> Option<f64> {
    let costs: Vec<f64> = log.lines()
        .filter(|l| l.starts_with("LLM finished in"))
        .filter_map(|l| l.rsplit_once('$').and_then(|(_, c)| c.trim_end_matches(')').parse().ok()))
        .collect();
    (!costs.is_empty()).then(|| costs.iter().sum())
}

fn run_case(case: &Case, opts: &EvalOpts, exe: &Path, tags: &[(i64, String)]) -> Result<CaseResult> {
    let dir = opts.out_dir.join(format!("repo-{}", &case.merge[..12]));
    let _ = std::fs::remove_dir_all(&dir);
    prepare(case, &dir, opts.copy_remotes)?;

    let start = Instant::now();
    let log_path = opts.out_dir.join(format!("{}.log", &case.merge[..12]));
    let log = std::fs::File::create(&log_path)?;
    // linux-next as it was before this merge (or none)
    let merge_time: i64 = git(&["log", "-1", "--format=%ct", &case.merge])?.trim().parse().unwrap_or(0);
    let next = if opts.next_tags {
        tags.iter().find(|(t, _)| *t < merge_time).map(|(_, c)| c.clone()).unwrap_or_else(|| "none".into())
    } else {
        "none".into()
    };
    let status = Command::new(exe)
        .current_dir(&dir)
        .arg("resolve")
        .args(&opts.resolve_args)
        .env("LLMINUS_NEXT_REF", &next)
        // only history that precedes this merge
        .env("LLMINUS_DB_ANCESTOR_OF", format!("{}^1", case.merge))
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .status()
        .context("Failed to run llminus resolve")?;
    let seconds = start.elapsed().as_secs();
    let log_text = std::fs::read_to_string(&log_path).unwrap_or_default();
    let exit = status.code().unwrap_or(-1);
    let reason = log_text.lines().rev().find(|l| l.starts_with("Not resolved")).unwrap_or("").to_string();

    let (paths, tree_match, fixups, divergence) = score(case, &dir, exit == 0);
    if !opts.keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
    Ok(CaseResult { case: case.clone(), exit, reason, seconds, cost_usd: cost_from_log(&log_text), paths,
                    tree_match, fixups, divergence })
}

fn count(results: &[CaseResult], status: &str) -> usize {
    results.iter().flat_map(|r| &r.paths).filter(|p| p.status == status).count()
}

/// The markdown report
pub fn report(results: &[CaseResult], header: &str) -> String {
    let mut s = String::new();
    s.push_str("# llminus eval\n\n");
    s.push_str(header);
    let n = results.len();
    let paths: usize = results.iter().map(|r| r.paths.len()).sum();
    let resolved = results.iter().filter(|r| r.exit == 0).count();
    let all_exact = results.iter().filter(|r| r.exit == 0 && r.paths.iter().all(|p| p.status == "exact")).count();
    // (an empty float sum is -0.0)
    let cost: f64 = results.iter().filter_map(|r| r.cost_usd).fold(0.0, |a, c| a + c);
    let secs: u64 = results.iter().map(|r| r.seconds).sum();
    s.push_str(&format!("\n## Summary\n\n- merges: {}, resolved: {} ({} with every conflicted path identical to the real merge)\n", n, resolved, all_exact));
    s.push_str(&format!("- needs a human: {}, errors: {}, build hosts failed: {}\n",
                        results.iter().filter(|r| r.exit == 2).count(),
                        results.iter().filter(|r| r.exit == 1 || r.exit < 0).count(),
                        results.iter().filter(|r| r.exit == 3).count()));
    s.push_str(&format!("- conflicted paths: {} - exact {}, whitespace-only differences {}, same lines in another \
                         order {}, different {}, unresolved {}\n",
                        paths, count(results, "exact"), count(results, "whitespace"), count(results, "reordered"),
                        count(results, "different"), count(results, "unresolved")));
    s.push_str(&format!("- whole merge identical to the real one: {}\n", results.iter().filter(|r| r.tree_match).count()));
    let fixups = |status: &str| results.iter().flat_map(|r| &r.fixups).filter(|f| f.status == status).count();
    s.push_str(&format!("- changes outside the conflicts: same as the real merge {}, different {}, llminus only {}, \
                         missed {}\n", fixups("same"), fixups("different"), fixups("llminus only"), fixups("missed")));
    s.push_str(&format!("- LLM cost ${:.2}, time {}m\n\n", cost, secs / 60));

    s.push_str("| merge | subject | exit | paths (exact/ws/reordered/diff) | outside (same/other) | min |\n|---|---|---|---|---|---|\n");
    for r in results {
        let c = |st: &str| r.paths.iter().filter(|p| p.status == st).count();
        s.push_str(&format!("| {} | {} | {} | {}/{}/{}/{} | {}/{} | {} |\n", &r.case.merge[..12],
                            r.case.subject.replace('|', "/").chars().take(70).collect::<String>(), r.exit,
                            c("exact"), c("whitespace"), c("reordered"), c("different"),
                            r.fixups.iter().filter(|f| f.status == "same").count(),
                            r.fixups.iter().filter(|f| f.status != "same").count(),
                            r.seconds / 60));
    }

    s.push_str("\n## Divergences and failures\n\n");
    for r in results.iter().filter(|r| r.exit != 0 || r.paths.iter().any(|p| p.status == "different" || p.status == "reordered")
                                   || r.fixups.iter().any(|f| f.status != "same")) {
        s.push_str(&format!("### {} {}\n\n", &r.case.merge[..12], r.case.subject));
        if r.exit != 0 {
            s.push_str(&format!("Exit {}: {}\n\n", r.exit, r.reason));
        }
        if !r.fixups.is_empty() {
            s.push_str(&format!("Changed outside the conflicts: {}\n\n", r.fixups.iter()
                .map(|f| format!("{} ({})", f.path, f.status)).collect::<Vec<_>>().join(", ")));
        }
        s.push_str(&r.divergence);
    }
    s
}

pub fn run(opts: &EvalOpts) -> Result<Vec<CaseResult>> {
    std::fs::create_dir_all(&opts.out_dir)?;
    let cases = select_cases(opts)?;
    eprintln!("{} merges with conflicts in {}", cases.len(), opts.range);
    let exe = std::env::current_exe()?;
    let tags = if opts.next_tags { next_tags() } else { Vec::new() };
    let jsonl_path = opts.out_dir.join("results.jsonl");
    let mut jsonl = std::fs::OpenOptions::new().create(true).append(true).open(&jsonl_path)?;
    let mut results = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        eprintln!("[{}/{}] {} {} ({} conflicted)", i + 1, cases.len(), &case.merge[..12], case.subject,
                  case.conflicted.len());
        let r = match run_case(case, opts, &exe, &tags) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  skipped: {:#}", e);
                continue;
            }
        };
        eprintln!("  exit {} in {}s: {}", r.exit, r.seconds,
                  r.paths.iter().map(|p| format!("{} {}", p.path, p.status)).collect::<Vec<_>>().join(", "));
        writeln!(jsonl, "{}", serde_json::to_string(&r)?)?;
        results.push(r);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::*;

    #[test]
    fn cost_is_summed_over_rounds() {
        let log = "LLM finished in 150s (exit Some(0), $0.59)\nBuild: FAIL\nLLM finished in 60s (exit Some(0), $0.21)\n";
        assert!((cost_from_log(log).unwrap() - 0.80).abs() < 1e-9);
        assert_eq!(cost_from_log("LLM finished in 3s (exit Some(0))"), None);
    }

    #[test]
    fn replays_a_merge_and_scores_it() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        let base = numbered_lines(10);
        create_commit(&dir, "file.c", &base, "base");
        create_branch(&dir, "feature");
        create_commit(&dir, "file.c", &base.replace("line 5\n", "feature 5\n"), "feature");
        checkout(&dir, "master");
        create_commit(&dir, "file.c", &base.replace("line 5\n", "master 5\n"), "master");
        merge(&dir, "feature", "Merge feature");
        // an evil merge: it also adds evil.c
        std::fs::write(dir.path().join("evil.c"), "fix\n").unwrap();
        git_in(dir.path(), &["add", "evil.c"]).unwrap();
        create_commit(&dir, "file.c", &base.replace("line 5\n", "resolved 5\n"), "Merge feature");
        create_commit(&dir, "later.c", "x\n", "later");

        let out = tempfile::tempdir().unwrap();
        let opts = EvalOpts { range: "master".into(), first_parent: true, committer: None, only: vec![],
                              limit: None, resolve_args: vec![], copy_remotes: false,
                              out_dir: out.path().to_path_buf(), keep: true, next_tags: false };
        let cases = select_cases(&opts).unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].conflicted, ["file.c"]);

        let repo = out.path().join("repo");
        prepare(&cases[0], &repo, false).unwrap();
        // the answer is not reachable from any ref in the scratch repo
        let all = git_in(&repo, &["log", "--all", "--format=%H"]).unwrap();
        assert!(!all.contains(&cases[0].merge), "the real merge leaks into the scratch repo");
        assert!(git_in(&repo, &["ls-files", "-u"]).unwrap().contains("file.c"));

        // unresolved
        let (paths, tree_match, _, _) = score(&cases[0], &repo, false);
        assert_eq!(paths[0].status, "unresolved");
        assert!(!tree_match);

        let statuses = |f: &[PathScore]| f.iter().map(|f| format!("{} {}", f.path, f.status)).collect::<Vec<_>>();

        // the conflict resolved like the real merge, but not its fixup
        std::fs::write(repo.join("file.c"), base.replace("line 5\n", "resolved 5\n")).unwrap();
        git_in(&repo, &["add", "file.c"]).unwrap();
        let (paths, tree_match, fixups, divergence) = score(&cases[0], &repo, true);
        assert_eq!(paths[0].status, "exact");
        assert!(!tree_match);
        assert_eq!(statuses(&fixups), ["evil.c missed"]);
        assert!(divergence.contains("evil.c (outside the conflicts, missed"), "{}", divergence);

        // and the fixup too, plus one of llminus's own
        std::fs::write(repo.join("evil.c"), "fix\n").unwrap();
        std::fs::write(repo.join("extra.c"), "more\n").unwrap();
        git_in(&repo, &["add", "evil.c", "extra.c"]).unwrap();
        let (_, tree_match, fixups, _) = score(&cases[0], &repo, true);
        assert!(!tree_match);
        assert_eq!(statuses(&fixups), ["evil.c same", "extra.c llminus only"]);

        // exactly the real merge
        git_in(&repo, &["rm", "-q", "--cached", "extra.c"]).unwrap();
        let (_, tree_match, fixups, _) = score(&cases[0], &repo, true);
        assert!(tree_match);
        assert_eq!(statuses(&fixups), ["evil.c same"]);

        // resolved differently
        std::fs::write(repo.join("file.c"), base.replace("line 5\n", "other 5\n")).unwrap();
        git_in(&repo, &["add", "file.c"]).unwrap();
        let (paths, _, _, divergence) = score(&cases[0], &repo, true);
        assert_eq!(paths[0].status, "different");
        assert!(divergence.contains("+other 5"), "{}", divergence);
    }
}
