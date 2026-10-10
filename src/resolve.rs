//! `llminus resolve`: resolve the merge in progress, or fix the build of the
//! merge just committed, with the LLM.
//!
//! The LLM edits files and writes a verdict; llminus checks that the LLM left
//! git alone and changed only what it declared, stages the result, builds it
//! as a scratch commit (never touching HEAD or the merge in progress) and
//! hands the LLM any new build errors for at most `repair_rounds` more rounds.

use anyhow::{bail, Result};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::build::{self, BuildSpec, BuildVerdict};
use crate::context::get_merge_context;
use crate::gitutil::{git, rev_parse, toplevel};
use crate::history;
use crate::llm::{self, Backend};
use crate::nextref::{self, Agreement};
use crate::outcome::{format_notes, Exit};
use crate::prompt::{self, ConflictPrompt};
use crate::pull::PullRequest;
use crate::shims;
use crate::state;
use crate::unmerged::{self, UnmergedPath};
use crate::verdict::{self, Decision, Manifest, Mode, Verdict};

/// Approximate characters per token (for English text)
const CHARS_PER_TOKEN: usize = 4;

/// Estimate the number of tokens in a text string
pub fn estimate_tokens(text: &str) -> usize {
    text.len() / CHARS_PER_TOKEN
}

pub struct Opts {
    pub backend: Backend,
    pub llm_timeout: Duration,
    pub rate_limit_retries: u32,
    pub rate_limit_wait: Duration,
    /// None: don't build test (conflict mode only)
    pub build: Option<BuildSpec>,
    pub repair_rounds: u32,
    pub max_tokens: usize,
    pub record_fixups: bool,
    /// Accept a resolution that differs from linux-next's resolution of the
    /// very same conflict (otherwise that needs a human)
    pub allow_next_differ: bool,
    /// The pull request mail being merged (message-id or file)
    pub pr_mail: Option<String>,
    /// Fetch linux-next's conflict reports for the conflicted files from lore
    pub prefetch_lore: bool,
    /// Each side's commits and one-sided identifiers in the prompt
    pub side_context: bool,
}

/// How a resolve run ended
#[derive(Debug)]
pub struct Resolution {
    pub exit: Exit,
    /// Notes for the commit message (exit Done only)
    pub notes: String,
    /// Why it ended this way
    pub reason: String,
    /// "conflict" or "build-fix"
    pub mode: &'static str,
    /// Whether llminus staged changes
    pub staged: bool,
    /// The build verdict of the result, if it was built
    pub build: Option<BuildVerdict>,
    /// The parents the result is for
    pub parents: Vec<String>,
    /// How the result compares with linux-next's resolutions (nextref::summary)
    pub next: Option<&'static str>,
}

impl Resolution {
    fn ended(exit: Exit, reason: impl Into<String>) -> Self {
        Resolution { exit, notes: String::new(), reason: reason.into(), mode: "conflict", staged: false,
                     build: None, parents: Vec::new(), next: None }
    }
    fn needs_human(reason: impl Into<String>) -> Self {
        Self::ended(Exit::NeedsHuman, reason)
    }
    fn infra(reason: impl Into<String>) -> Self {
        Self::ended(Exit::Infra, reason)
    }
    fn error(reason: impl Into<String>) -> Self {
        Self::ended(Exit::Error, reason)
    }
}

/// One LLM round: returns the validated verdict, or how the run ends
fn llm_round(opts: &Opts, prompt: &str, mode: Mode, unmerged: &[UnmergedPath], prior: &Manifest)
             -> Result<std::result::Result<Verdict, Resolution>> {
    let top = PathBuf::from(toplevel()?);
    let verdict_path = verdict::verdict_path()?;
    let _ = std::fs::remove_file(&verdict_path);

    let before = state::take()?;
    let merge_state = state::save_merge_state();
    let searches_before = shims::lei_saved_searches();

    eprintln!("Invoking: {} (prompt: {} bytes, ~{} tokens)", opts.backend.describe(), prompt.len(),
              estimate_tokens(prompt));
    eprintln!("{}", "=".repeat(80));
    let start = Instant::now();
    let run = llm::run(&opts.backend, prompt, &llm::RunOpts {
        cwd: &top,
        timeout: opts.llm_timeout,
        rate_limit_retries: opts.rate_limit_retries,
        rate_limit_wait: opts.rate_limit_wait,
        env: vec![("LLMINUS_VERDICT", verdict_path.to_string_lossy().into())],
        clear_before_attempt: vec![verdict_path.clone()],
    })?;
    if matches!(opts.backend, Backend::Claude { .. } | Backend::Codex { .. }) {
        eprintln!("{}", run.final_message.trim());
    }
    eprintln!("{}", "=".repeat(80));
    eprintln!("LLM finished in {}s (exit {:?}{}{})", start.elapsed().as_secs(), run.status,
              if run.timed_out { ", TIMED OUT" } else { "" },
              run.cost_usd.map(|c| format!(", ${:.2}", c)).unwrap_or_default());

    // An LLM that committed and soft-reset during the merge: put the merge
    // state back for whoever looks at it next - and fail, it committed
    let restored = merge_state.as_ref().is_some_and(state::restore_merge_state);
    let after = state::take()?;

    if let (Some(b), Some(a)) = (searches_before, shims::lei_saved_searches()) {
        let new: Vec<&String> = a.iter().filter(|s| !b.contains(s)).collect();
        if !new.is_empty() {
            eprintln!("Warning: new lei saved searches appeared during the LLM run (not removed, \
                       they may be someone else's): {:?}", new);
        }
    }

    let mut violations = before.git_violations(&after);
    if restored {
        violations.insert(0, "it committed during the merge (llminus put the merge state back)".into());
    }
    if !violations.is_empty() {
        let _ = std::fs::remove_file(&verdict_path);
        return Ok(Err(Resolution::error(format!("the LLM changed git state it must leave alone: {}",
                                                  violations.join("; ")))));
    }
    if run.timed_out {
        let _ = std::fs::remove_file(&verdict_path);
        return Ok(Err(Resolution::needs_human(format!("the LLM timed out after {}s", opts.llm_timeout.as_secs()))));
    }
    if run.status != Some(0) {
        let _ = std::fs::remove_file(&verdict_path);
        let msg: String = run.final_message.trim().chars().take(300).collect();
        return Ok(Err(Resolution::error(format!("the LLM command failed (exit {:?}): {}", run.status, msg))));
    }

    let decision = verdict::load()?;
    let _ = std::fs::remove_file(&verdict_path);
    let v = match decision {
        Decision::NeedsHuman(reason) => return Ok(Err(Resolution::needs_human(reason))),
        Decision::Resolved(v) => v,
    };
    if let Err(e) = verdict::validate(&v, mode, unmerged, &before.changed_paths(&after), prior) {
        return Ok(Err(Resolution::needs_human(format!("unusable verdict: {}", e))));
    }
    Ok(Ok(v))
}

/// Stage the manifest, and check that nothing changed is left unstaged
fn stage(manifest: &Manifest) -> Result<std::result::Result<(), Resolution>> {
    if let Err(problems) = verdict::apply(manifest)? {
        return Ok(Err(Resolution::needs_human(problems)));
    }
    let unstaged = state::unstaged_tracked()?;
    if !unstaged.is_empty() {
        return Ok(Err(Resolution::needs_human(format!("changes left unstaged: {}", unstaged.join(", ")))));
    }
    Ok(Ok(()))
}

fn manifest_lines(manifest: &Manifest) -> Vec<String> {
    manifest.values().map(|pa| {
        if pa.rationale.is_empty() {
            format!("{} {}", pa.path, pa.action)
        } else {
            format!("{} {} ({})", pa.path, pa.action, pa.rationale)
        }
    }).collect()
}

/// Build the staged result; on new errors give the LLM up to
/// `opts.repair_rounds` more rounds. The manifest and notes are updated.
#[allow(clippy::too_many_arguments)]
fn build_and_repair(opts: &Opts, spec: &BuildSpec, parents: &[String], baseline: &str, mode: Mode,
                    unmerged: &[UnmergedPath], task: &str, manifest: &mut Manifest, notes: &mut String)
                    -> Result<std::result::Result<BuildVerdict, Resolution>> {
    let mut round = 0;
    loop {
        let bv = build::build_staged(parents, baseline, spec)?;
        eprintln!("Build: {}", bv.describe());
        match bv {
            BuildVerdict::Pass | BuildVerdict::CachedPass => return Ok(Ok(bv)),
            BuildVerdict::Infra(why) => return Ok(Err(Resolution::infra(format!("the build could not be judged: {}", why)))),
            BuildVerdict::ToolError(why) => return Ok(Err(Resolution::error(format!("the build command failed: {}", why)))),
            BuildVerdict::Fail(errors) => {
                if round >= opts.repair_rounds {
                    return Ok(Err(Resolution::needs_human(format!(
                        "the result still has new build errors after {} repair round(s):\n{}", round, errors))));
                }
                round += 1;
                eprintln!("New build errors; repair round {} of {}", round, opts.repair_rounds);
                let p = prompt::repair_prompt(&errors, round, opts.repair_rounds, &manifest_lines(manifest),
                                              notes, task);
                let v = match llm_round(opts, &p, mode, unmerged, manifest)? {
                    Ok(v) => v,
                    Err(res) => return Ok(Err(res)),
                };
                verdict::extend_manifest(manifest, &v);
                if let Err(res) = stage(manifest)? {
                    return Ok(Err(res));
                }
                if !v.notes.trim().is_empty() {
                    *notes = v.notes.clone();
                }
            }
        }
    }
}

fn build_fact(spec: Option<&BuildSpec>, bv: Option<&BuildVerdict>) -> String {
    match (spec, bv) {
        (Some(s), Some(v)) => format!("Build tested by llminus ({}): {}", s.describe(), v.describe()),
        _ => "Not build tested by llminus.".to_string(),
    }
}

/// Resolve the unmerged paths of the merge of `merge_commit` into HEAD
fn resolve_conflicts(opts: &Opts, pull_req: Option<&PullRequest>, merge_commit: &str) -> Result<Resolution> {
    let Some(head) = rev_parse("HEAD") else { bail!("HEAD does not name a commit") };
    let parents = vec![head.clone(), merge_commit.to_string()];
    let done = |notes: String, staged: bool, build: Option<BuildVerdict>| Resolution {
        exit: Exit::Done, notes, reason: "resolved".into(), mode: "conflict", staged, build,
        parents: parents.clone(), next: None,
    };
    let merge_ctx = get_merge_context();
    if let Some(ref source) = merge_ctx.merge_source {
        eprintln!("Merging: {}", source);
    }
    if let Some(ref h) = merge_ctx.head_branch {
        eprintln!("Into: {}", h);
    }

    let unmerged = unmerged::current()?;
    if unmerged.is_empty() {
        // A merge in progress with nothing left unmerged (rerere resolved it,
        // say): just check that it builds
        eprintln!("No unmerged paths.");
        let Some(spec) = &opts.build else {
            return Ok(done(format_notes("", &[build_fact(None, None)]), false, None));
        };
        let mut manifest = Manifest::new();
        let mut notes = String::new();
        let task = "Build fix for a merge in progress that has no conflicts left.";
        return Ok(match build_and_repair(opts, spec, &parents, &head, Mode::BuildFix, &[], task,
                                         &mut manifest, &mut notes)? {
            Ok(bv) => done(format_notes(&notes, &[build_fact(Some(spec), Some(&bv))]), !manifest.is_empty(), Some(bv)),
            Err(res) => res,
        });
    }

    for u in &unmerged {
        eprintln!("Unmerged: {} ({}{})", u.path, u.kind().describe(), if u.binary { ", binary" } else { "" });
    }

    let next_refs = match nextref::next_tip() {
        Some(next) => {
            let paths: Vec<String> = unmerged.iter().map(|u| u.path.clone()).collect();
            let refs = nextref::find(&next, &head, merge_commit, &paths);
            eprintln!("linux-next ({}): hand resolutions found for {} of {} paths", &next[..12], refs.len(), paths.len());
            refs
        }
        None => Vec::new(),
    };

    let pr_mail = match (pull_req, &opts.pr_mail) {
        (None, Some(spec)) => match crate::pull::load_mail(spec) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("Warning: could not load the pull request mail {}: {:#}", spec, e);
                None
            }
        },
        _ => None,
    };
    let pull_req = pull_req.or(pr_mail.as_ref());
    let lore_reports = if opts.prefetch_lore {
        let paths: Vec<String> = unmerged.iter().map(|u| u.path.clone()).collect();
        let r = crate::lore::manual_merge_reports(&paths, 2);
        eprintln!("lore: {} linux-next conflict report(s) for these files", r.len());
        r
    } else {
        Vec::new()
    };

    let regions: Vec<_> = unmerged.iter().flat_map(|u| u.regions.clone()).collect();
    let all_similar = if regions.is_empty() { Vec::new() } else { history::try_similar_resolutions(3, &regions) };
    if all_similar.is_empty() {
        eprintln!("No historical resolutions used (no database, or nothing similar).");
    } else {
        eprintln!("Found {} similar historical resolutions", all_similar.len());
    }

    // Build the prompt, dropping historical examples until it fits
    let mut similar = all_similar.clone();
    let make_prompt = |similar: &[history::SimilarResolution]| prompt::conflict_prompt(&ConflictPrompt {
        unmerged: &unmerged,
        similar,
        merge_ctx: &merge_ctx,
        pull_req,
        ours: Some(&head),
        theirs: Some(merge_commit),
        lore_access: opts.backend.has_network(),
        build: opts.build.as_ref().map(|b| b.desc.clone().unwrap_or_else(|| format!("`{}`", b.describe()))),
        next_refs: &next_refs,
        lore_reports: &lore_reports,
        side_context: opts.side_context,
    });
    let mut text = make_prompt(&similar);
    while estimate_tokens(&text) > opts.max_tokens && !similar.is_empty() {
        similar.pop();
        text = make_prompt(&similar);
        eprintln!("Reduced historical examples from {} to {} to fit the token limit (~{} tokens, limit: {})",
                  all_similar.len(), similar.len(), estimate_tokens(&text), opts.max_tokens);
    }
    if estimate_tokens(&text) > opts.max_tokens {
        eprintln!("Warning: the prompt still exceeds the token limit (~{} tokens, limit: {})",
                  estimate_tokens(&text), opts.max_tokens);
    }

    let mut manifest = Manifest::new();
    let v = match llm_round(opts, &text, Mode::Conflict, &unmerged, &manifest)? {
        Ok(v) => v,
        Err(res) => return Ok(res),
    };
    verdict::extend_manifest(&mut manifest, &v);
    if let Err(res) = stage(&manifest)? {
        return Ok(res);
    }
    let mut notes = v.notes.clone();

    // Compare with linux-next before spending a build on it
    let (mut next_facts, differ, mut next_summary) = compare_with_next(&next_refs);
    if !differ.is_empty() {
        if !opts.allow_next_differ {
            return Ok(Resolution::needs_human(format!(
                "the resolution of {} differs from linux-next's resolution of the very same conflict \
                 (review it, or pass --allow-next-differ)", differ.join(", "))));
        }
        eprintln!("Warning: {} resolved differently from linux-next (allowed by --allow-next-differ)", differ.join(", "));
    }

    let tree_before = git(&["write-tree"])?;
    let bv = match &opts.build {
        None => None,
        Some(spec) => match build_and_repair(opts, spec, &parents, &head, Mode::Conflict, &unmerged, &text,
                                             &mut manifest, &mut notes)? {
            Ok(bv) => Some(bv),
            Err(res) => return Ok(res),
        },
    };
    // A repair round may have changed a conflicted file: what linux-next is
    // compared with is what gets committed
    if !next_refs.is_empty() && git(&["write-tree"])? != tree_before {
        eprintln!("Comparing with linux-next again after the repair rounds");
        let (facts, differ, summary) = compare_with_next(&next_refs);
        if !differ.is_empty() {
            if !opts.allow_next_differ {
                return Ok(Resolution::needs_human(format!(
                    "after the repair rounds, the resolution of {} differs from linux-next's resolution of the \
                     very same conflict (review it, or pass --allow-next-differ)", differ.join(", "))));
            }
            eprintln!("Warning: {} resolved differently from linux-next (allowed by --allow-next-differ)",
                      differ.join(", "));
        }
        (next_facts, next_summary) = (facts, summary);
    }

    if opts.record_fixups {
        let fixups: Vec<String> = manifest.keys().filter(|p| !unmerged.iter().any(|u| &u.path == *p)).cloned().collect();
        match crate::fixups::record(&head, merge_commit, &unmerged, &fixups) {
            Ok(Some(dir)) => eprintln!("Recorded the resolution in {}", dir.display()),
            Ok(None) => {}
            Err(e) => eprintln!("Warning: could not record the resolution: {:#}", e),
        }
    }

    let files: Vec<&str> = unmerged.iter().map(|u| u.path.as_str()).collect();
    let mut facts = vec![format!("Conflicts in {} resolved with llminus.", files.join(", "))];
    facts.extend(next_facts);
    facts.push(build_fact(opts.build.as_ref(), bv.as_ref()));
    let mut res = done(format_notes(&notes, &facts), true, bv);
    res.next = next_summary;
    Ok(res)
}

/// Compare the staged result with linux-next's resolutions: the facts for
/// the notes, the paths whose identical conflict linux-next resolved
/// differently, and the summary for the result header
fn compare_with_next(refs: &[nextref::Reference]) -> (Vec<String>, Vec<String>, Option<&'static str>) {
    let mut facts = Vec::new();
    let mut differ = Vec::new();
    let mut agreements = Vec::new();
    for r in refs {
        let blob = rev_parse(&format!(":{}", r.path));
        let content = crate::gitutil::git_allow_fail(&["show", &format!(":{}", r.path)]).unwrap_or_default();
        let a = nextref::compare(r, blob.as_deref(), &content);
        eprintln!("{}: {}", r.path, a.describe());
        if a == Agreement::Differ {
            differ.push(r.path.clone());
        }
        facts.push(format!("{}: {} (linux-next {}, merge {}).", r.path, a.describe(), &r.next[..12], &r.merge[..12]));
        agreements.push(a);
    }
    (facts, differ, nextref::summary(&agreements))
}

/// Fix the build of the merge commit at HEAD (no merge in progress)
fn fix_build(opts: &Opts) -> Result<Resolution> {
    let Some(head) = rev_parse("HEAD") else { bail!("HEAD does not name a commit") };
    let (Some(first_parent), Some(_), None) = (rev_parse("HEAD^1"), rev_parse("HEAD^2"), rev_parse("HEAD^3")) else {
        return Ok(Resolution::error(
            "no merge in progress, and HEAD is not a two-parent merge: nothing to resolve or fix the build of"));
    };
    if unmerged::any() {
        return Ok(Resolution::error("there are unmerged paths but no merge in progress"));
    }
    if !state::unstaged_tracked()?.is_empty() || crate::gitutil::git_allow_fail(&["diff", "--cached", "--quiet"]).is_none() {
        return Ok(Resolution::error("the working tree or index has changes; fixing the build needs a clean tree"));
    }
    let Some(spec) = &opts.build else {
        return Ok(Resolution::error("fixing the build needs a build command: --build-cmd, $LLMINUS_BUILD_CMD, \
                                     git config llminus.buildCmd, or an executable ~/build.sh (and not --no-build)"));
    };
    let parents = vec![head.clone()];
    let done = |notes: String, staged: bool, build: BuildVerdict| Resolution {
        exit: Exit::Done, notes, reason: "resolved".into(), mode: "build-fix", staged, build: Some(build),
        parents: parents.clone(), next: None,
    };
    let merge_ctx = get_merge_context();
    eprintln!("No merge in progress: checking whether the merge at HEAD added build errors \
               (compared to its first parent {})...", &first_parent[..12]);

    // Errors that the first parent already has are not this merge's to fix
    let bv = build::build_commit(&head, &first_parent, spec)?;
    eprintln!("Build: {}", bv.describe());
    let errors = match bv {
        BuildVerdict::Pass | BuildVerdict::CachedPass => {
            return Ok(done(format_notes("", &[format!(
                "No new build errors compared to the merge's first parent ({}): nothing to fix.", spec.describe())]),
                false, bv));
        }
        BuildVerdict::Infra(why) => return Ok(Resolution::infra(format!("the build could not be judged: {}", why))),
        BuildVerdict::ToolError(why) => return Ok(Resolution::error(format!("the build command failed: {}", why))),
        BuildVerdict::Fail(errors) => errors,
    };

    let p = prompt::build_fix_prompt(&errors, &merge_ctx, opts.backend.has_network());
    let mut manifest = Manifest::new();
    let v = match llm_round(opts, &p, Mode::BuildFix, &[], &manifest)? {
        Ok(v) => v,
        Err(res) => return Ok(res),
    };
    verdict::extend_manifest(&mut manifest, &v);
    if let Err(res) = stage(&manifest)? {
        return Ok(res);
    }
    let mut notes = v.notes.clone();
    match build_and_repair(opts, spec, &parents, &first_parent, Mode::BuildFix, &[], &p, &mut manifest, &mut notes)? {
        Ok(bv) => {
            let fact = format!("Build fix by llminus; {}", build_fact(Some(spec), Some(&bv)).to_lowercase());
            Ok(done(format_notes(&notes, &[fact]), true, bv))
        }
        Err(res) => Ok(res),
    }
}

/// Resolve the merge in progress, or fix the build of the merge at HEAD
pub fn resolve(opts: &Opts, pull_req: Option<&PullRequest>) -> Result<Resolution> {
    crate::gitutil::check_repo()?;
    match unmerged::merge_head_commit() {
        Err(e) => Ok(Resolution::error(format!("{:#}", e))),
        Ok(Some(merge_commit)) => resolve_conflicts(opts, pull_req, &merge_commit),
        Ok(None) => fix_build(opts),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gitutil::git;
    use crate::testutil::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::TempDir;

    /// A fake LLM: does what $FAKE_LLM says, writes the verdict to $LLMINUS_VERDICT
    const FAKE_LLM: &str = r#"#!/bin/bash
prompt=$(cat)
v="$LLMINUS_VERDICT"
FAKE_LLM=$(cat "$(dirname "$0")/scenario")
resolved() { printf 'line 1\nline 2\nline 3\nline 4\nresolved 5\nline 6\nline 7\nline 8\nline 9\nline 10\n' > file.c; }
case "$FAKE_LLM" in
resolve)
    resolved
    echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"}],"notes":"Took both sides."}' > "$v" ;;
moddel)
    resolved
    echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"},{"path":"gone.c","action":"delete"}],"notes":"gone.c was removed upstream."}' > "$v" ;;
abstain)
    echo '{"verdict":"ABSTAIN","notes":"Not sure which side is right."}' > "$v" ;;
none) ;;
undeclared)
    resolved; echo extra >> other.c
    echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"}],"notes":"x"}' > "$v" ;;
commit)
    resolved; /usr/bin/git add file.c; /usr/bin/git commit -q -m sneaky
    echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"}],"notes":"x"}' > "$v" ;;
agreethendiffer)
    if grep -q "Build failure after the resolution" <<< "$prompt"; then
        rm -f BROKEN; sed -i 's/resolved 5/other 5/' file.c
        echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"},{"path":"BROKEN","action":"semantic_fixup","rationale":"oops"}],"notes":"Reworked."}' > "$v"
    else
        resolved; echo x > BROKEN
        echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"},{"path":"BROKEN","action":"semantic_fixup","rationale":"needed"}],"notes":"Took both sides."}' > "$v"
    fi ;;
breakthenfix)
    if grep -q "Build failure after the resolution" <<< "$prompt"; then
        rm -f BROKEN
        echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"},{"path":"BROKEN","action":"semantic_fixup","rationale":"oops"}],"notes":"Took both sides, without BROKEN."}' > "$v"
    else
        resolved; echo x > BROKEN
        echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"},{"path":"BROKEN","action":"semantic_fixup","rationale":"needed"}],"notes":"Took both sides."}' > "$v"
    fi ;;
fixupedit)
    if grep -q "Build failure after the resolution" <<< "$prompt"; then
        echo v2 > fixup.h; rm -f BROKEN
        echo '{"verdict":"RESOLVED","paths":[{"path":"BROKEN","action":"semantic_fixup","rationale":"drop it"}],"notes":"v2 fixup."}' > "$v"
    else
        resolved; echo v1 > fixup.h; echo x > BROKEN
        echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"},{"path":"fixup.h","action":"semantic_fixup","rationale":"api"},{"path":"BROKEN","action":"semantic_fixup","rationale":"x"}],"notes":"v1 fixup."}' > "$v"
    fi ;;
commitreset)
    resolved; /usr/bin/git add file.c; /usr/bin/git commit -q -m wip; /usr/bin/git reset -q --soft HEAD~1
    echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"}],"notes":"x"}' > "$v" ;;
chmod)
    resolved; chmod +x clean.c
    echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"}],"notes":"x"}' > "$v" ;;
failverdict)
    resolved
    echo '{"verdict":"RESOLVED","paths":[{"path":"file.c","action":"resolved"}],"notes":"x"}' > "$v"
    exit 1 ;;
fixbuild)
    rm -f BROKEN
    echo '{"verdict":"RESOLVED","paths":[{"path":"BROKEN","action":"delete"}],"notes":"BROKEN broke the build."}' > "$v" ;;
esac
"#;

    /// A fake `stable build log`: fails if the committed tree has BROKEN
    const FAKE_BUILD: &str = r#"#!/bin/bash
here=$(dirname "$0")
echo "$LLMINUS_BUILD_BASELINE" >> "$here/build.log"
[ -e "$here/infra" ] && exit 75
[ -e "$here/toolerror" ] && { echo "usage: stable build ..."; exit 2; }
git cat-file -e HEAD:BROKEN 2>/dev/null && { echo "x86_64: BROKEN: error: broken"; exit 1; }
exit 0
"#;

    struct Fakes {
        dir: TempDir,
    }

    impl Fakes {
        fn new() -> Self {
            let dir = TempDir::new().unwrap();
            for (name, text) in [("llm", FAKE_LLM), ("build", FAKE_BUILD)] {
                let p = dir.path().join(name);
                fs::write(&p, text).unwrap();
                fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
            }
            Fakes { dir }
        }

        fn opts(&self, build: bool) -> Opts {
            Opts {
                backend: Backend::Command(self.dir.path().join("llm").to_string_lossy().into()),
                llm_timeout: Duration::from_secs(30),
                rate_limit_retries: 0,
                rate_limit_wait: Duration::from_secs(1),
                allow_next_differ: false,
                pr_mail: None,
                prefetch_lore: false,
                side_context: true,
                build: build.then(|| BuildSpec::parse(&self.dir.path().join("build").to_string_lossy(), true, None,
                                                     Duration::from_secs(30)).unwrap()),
                repair_rounds: 2,
                max_tokens: 100_000,
                record_fixups: false,
            }
        }

        fn build_log(&self) -> std::path::PathBuf {
            self.dir.path().join("build.log")
        }
    }

    /// What the fakes do: files next to them (no process environment
    /// changes, other tests spawn processes concurrently)
    fn set_env(fakes: &Fakes, llm: &str, infra: bool) {
        fs::write(fakes.dir.path().join("scenario"), llm).unwrap();
        let infra_file = fakes.dir.path().join("infra");
        if infra { fs::write(infra_file, "").unwrap() } else { let _ = fs::remove_file(infra_file); }
    }

    fn merge_head_exists() -> bool {
        std::path::Path::new(&crate::gitutil::git_path("MERGE_HEAD")).exists()
    }

    fn staged() -> String {
        git(&["diff", "--cached", "--name-only"]).unwrap()
    }

    #[test]
    fn content_conflict_is_resolved_staged_and_built() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let head = git(&["rev-parse", "HEAD"]).unwrap();
        let f = Fakes::new();
        set_env(&f, "resolve", false);

        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Done, "{}", res.reason);
        assert!(res.notes.contains("Took both sides."));
        assert!(res.notes.contains("Build tested by llminus"));
        assert!(!unmerged::any());
        assert!(merge_head_exists(), "the merge must still be in progress");
        assert_eq!(git(&["rev-parse", "HEAD"]).unwrap(), head);
        assert!(git(&["show", ":file.c"]).unwrap().contains("resolved 5"));
        // the build ran against the pre-merge HEAD
        assert_eq!(fs::read_to_string(f.build_log()).unwrap().trim(), head.trim());
        // and left no scratch worktree behind
        assert_eq!(git(&["worktree", "list"]).unwrap().lines().count(), 1);
    }

    #[test]
    fn modify_delete_is_resolved_by_deleting() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        let base = numbered_lines(10);
        create_commit(&dir, "file.c", &base, "base");
        create_commit(&dir, "gone.c", "old\n", "gone.c");
        create_branch(&dir, "feature");
        create_commit(&dir, "file.c", &base.replace("line 5\n", "feature 5\n"), "feature");
        git(&["rm", "-q", "gone.c"]).unwrap();
        git(&["commit", "-q", "-m", "remove gone.c"]).unwrap();
        checkout(&dir, "master");
        create_commit(&dir, "file.c", &base.replace("line 5\n", "master 5\n"), "master");
        create_commit(&dir, "gone.c", "old, modified\n", "touch gone.c");
        merge(&dir, "feature", "Merge feature");
        let um = unmerged::current().unwrap();
        assert!(um.iter().any(|u| u.path == "gone.c" && u.kind() == unmerged::Kind::DeletedByThem));

        let f = Fakes::new();
        set_env(&f, "moddel", false);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Done, "{}", res.reason);
        assert!(!dir.path().join("gone.c").exists());
        assert!(git(&["ls-files", "gone.c"]).unwrap().is_empty());
        assert!(!unmerged::any());
    }

    #[test]
    fn declines_and_bad_verdicts_need_a_human() {
        for (scenario, why) in [("abstain", "gave up"), ("none", "no verdict"), ("undeclared", "other.c")] {
            let dir = init_test_repo();
            let _cwd = CwdGuard::enter(dir.path());
            create_commit(&dir, "other.c", "other\n", "other");
            start_conflicting_merge(&dir);
            let f = Fakes::new();
            set_env(&f, scenario, false);
            let res = resolve(&f.opts(true), None).unwrap();
            assert_eq!(res.exit, Exit::NeedsHuman, "{}: {}", scenario, res.reason);
            assert!(res.reason.contains(why), "{}: {}", scenario, res.reason);
            assert!(merge_head_exists());
        }
    }

    #[test]
    fn an_llm_that_commits_is_an_error() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let f = Fakes::new();
        set_env(&f, "commit", false);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Error, "{}", res.reason);
        assert!(res.reason.contains("git state"), "{}", res.reason);
    }

    /// A "linux-next" that resolved the conflict in progress with the fake
    /// LLM's resolution ("resolved 5"): refs/remotes/linux-next/master
    fn add_linux_next_resolution() {
        let resolved = "line 1\nline 2\nline 3\nline 4\nresolved 5\nline 6\nline 7\nline 8\nline 9\nline 10\n";
        let blob = crate::gitutil::git_with_input(&["hash-object", "-w", "--stdin"], resolved.as_bytes()).unwrap();
        let tmp = TempDir::new().unwrap();
        let index = tmp.path().join("index").to_string_lossy().to_string();
        let env = [("GIT_INDEX_FILE", index.as_str())];
        crate::gitutil::git_bytes_env(&["read-tree", "HEAD"], &env).unwrap();
        crate::gitutil::git_bytes_env(&["update-index", "--cacheinfo", &format!("100644,{},file.c", blob.trim())], &env)
            .unwrap();
        let tree = String::from_utf8(crate::gitutil::git_bytes_env(&["write-tree"], &env).unwrap()).unwrap();
        let theirs = git(&["rev-parse", "MERGE_HEAD^{commit}"]).unwrap();
        let next = git(&["commit-tree", tree.trim(), "-p", "HEAD", "-p", theirs.trim(), "-m", "Merge feature into next"])
            .unwrap();
        git(&["update-ref", "refs/remotes/linux-next/master", next.trim()]).unwrap();
    }

    #[test]
    fn linux_next_is_compared_again_after_a_repair_round() {
        let f = Fakes::new();
        set_env(&f, "agreethendiffer", false);

        // agreed before the build, differs after the repair: a human looks
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        add_linux_next_resolution();
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::NeedsHuman, "{}", res.reason);
        assert!(res.reason.contains("after the repair rounds"), "{}", res.reason);

        // allowed: done, and the result says DIFFER
        let dir = init_test_repo();
        let _cwd2 = { drop(_cwd); CwdGuard::enter(dir.path()) };
        start_conflicting_merge(&dir);
        add_linux_next_resolution();
        let mut opts = f.opts(true);
        opts.allow_next_differ = true;
        let res = resolve(&opts, None).unwrap();
        assert_eq!(res.exit, Exit::Done, "{}", res.reason);
        assert_eq!(res.next, Some("DIFFER"));
        assert!(res.notes.contains("DIFFERENT from linux-next"), "{}", res.notes);

        // a repair that leaves the conflicted file alone still agrees
        set_env(&f, "breakthenfix", false);
        let dir = init_test_repo();
        let _cwd3 = { drop(_cwd2); CwdGuard::enter(dir.path()) };
        start_conflicting_merge(&dir);
        add_linux_next_resolution();
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Done, "{}", res.reason);
        assert_eq!(res.next, Some("AGREE"));
        assert!(!res.notes.contains("DIFFERENT"), "{}", res.notes);
    }

    #[test]
    fn build_errors_get_a_repair_round() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let f = Fakes::new();
        set_env(&f, "breakthenfix", false);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Done, "{}", res.reason);
        assert!(res.notes.contains("without BROKEN"));
        assert_eq!(fs::read_to_string(f.build_log()).unwrap().lines().count(), 2);
        assert!(!staged().contains("BROKEN"));

        // and with no repair rounds left it goes to a human
        let dir = init_test_repo();
        let _cwd2 = { drop(_cwd); CwdGuard::enter(dir.path()) };
        start_conflicting_merge(&dir);
        let mut opts = f.opts(true);
        opts.repair_rounds = 0;
        let res = resolve(&opts, None).unwrap();
        assert_eq!(res.exit, Exit::NeedsHuman);
        assert!(res.reason.contains("x86_64: BROKEN: error: broken"), "{}", res.reason);
    }

    #[test]
    fn a_repair_round_can_change_an_earlier_fixup_and_it_is_staged() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let f = Fakes::new();
        set_env(&f, "fixupedit", false);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Done, "{}", res.reason);
        assert_eq!(git(&["show", ":fixup.h"]).unwrap(), "v2\n", "the repaired fixup must be what's staged");
        assert!(state::unstaged_tracked().unwrap().is_empty());
        assert!(!staged().contains("BROKEN"));
    }

    #[test]
    fn committing_during_the_merge_is_an_error_even_if_undone() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let f = Fakes::new();
        set_env(&f, "commitreset", false);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Error, "{}", res.reason);
        assert!(res.reason.contains("committed during the merge"), "{}", res.reason);
        assert!(merge_head_exists(), "the merge state is put back");
    }

    #[test]
    fn a_mode_change_on_a_staged_path_is_noticed() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        let base = numbered_lines(10);
        create_commit(&dir, "file.c", &base, "base");
        create_branch(&dir, "feature");
        create_commit(&dir, "file.c", &base.replace("line 5\n", "feature 5\n"), "feature");
        create_commit(&dir, "clean.c", "new\n", "feature adds clean.c");
        checkout(&dir, "master");
        create_commit(&dir, "file.c", &base.replace("line 5\n", "master 5\n"), "master");
        merge(&dir, "feature", "Merge feature");
        let f = Fakes::new();
        set_env(&f, "chmod", false);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::NeedsHuman, "{}", res.reason);
        assert!(res.reason.contains("clean.c"), "{}", res.reason);
    }

    #[test]
    fn a_failing_llm_command_is_an_error() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let f = Fakes::new();
        set_env(&f, "failverdict", false);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Error, "{}", res.reason);
    }

    #[test]
    fn a_failing_build_command_is_not_a_build_failure() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let f = Fakes::new();
        set_env(&f, "resolve", false);
        fs::write(f.dir.path().join("toolerror"), "").unwrap();
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Error, "{}", res.reason);
        assert!(res.reason.contains("build command failed"), "{}", res.reason);
    }

    #[test]
    fn the_build_command_gets_a_checkout_and_its_placeholders() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let head = git(&["rev-parse", "HEAD"]).unwrap().trim().to_string();
        let f = Fakes::new();
        set_env(&f, "resolve", false);
        let script = f.dir.path().join("checkout-build");
        fs::write(&script, "#!/bin/bash\n[ -f file.c ] && grep -q 'resolved 5' file.c || exit 2\n\
                            echo \"$1 $2 $LLMINUS_BUILD_BASELINE $LLMINUS_BUILD_COMMIT\" > \"$(dirname \"$0\")/args\"\n").unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        let mut opts = f.opts(true);
        opts.build = Some(BuildSpec::parse(&format!("{} {{baseline}} {{commit}}", script.display()), true, None,
                                           Duration::from_secs(30)).unwrap());
        let res = resolve(&opts, None).unwrap();
        assert_eq!(res.exit, Exit::Done, "{}", res.reason);
        let args = fs::read_to_string(f.dir.path().join("args")).unwrap();
        let w: Vec<&str> = args.split_whitespace().collect();
        assert_eq!(w.len(), 4, "{}", args);
        assert_eq!(w[0], head);
        assert_eq!(w[2], head);
        assert_eq!(w[1], w[3], "{{commit}} and LLMINUS_BUILD_COMMIT agree");
    }

    #[test]
    fn build_hosts_failing_is_infra() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let f = Fakes::new();
        set_env(&f, "resolve", true);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Infra, "{}", res.reason);
    }

    #[test]
    fn no_build_skips_the_build() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let f = Fakes::new();
        set_env(&f, "resolve", true);
        let res = resolve(&f.opts(false), None).unwrap();
        assert_eq!(res.exit, Exit::Done, "{}", res.reason);
        assert!(res.notes.contains("Not build tested"));
        assert!(!f.build_log().exists());
    }

    #[test]
    fn build_fix_on_a_committed_merge() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        create_commit(&dir, "file.c", "x\n", "base");
        create_branch(&dir, "feature");
        create_commit(&dir, "BROKEN", "x\n", "feature breaks the build");
        checkout(&dir, "master");
        create_commit(&dir, "other.c", "y\n", "master");
        merge(&dir, "feature", "Merge feature");
        let first_parent = git(&["rev-parse", "HEAD^1"]).unwrap();

        let f = Fakes::new();
        set_env(&f, "fixbuild", false);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Done, "{}", res.reason);
        assert!(staged().contains("BROKEN"), "the deletion is staged");
        assert!(res.notes.contains("Build fix by llminus"));
        // both builds were judged against the merge's first parent
        for l in fs::read_to_string(f.build_log()).unwrap().lines() {
            assert_eq!(l, first_parent.trim());
        }
    }

    #[test]
    fn build_fix_needs_a_merge_at_head() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        create_commit(&dir, "file.c", "x\n", "base");
        let f = Fakes::new();
        set_env(&f, "fixbuild", false);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Error);
    }

    #[test]
    fn octopus_merges_are_refused() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        create_commit(&dir, "file.c", "x\n", "base");
        let head = git(&["rev-parse", "HEAD"]).unwrap();
        fs::write(crate::gitutil::git_path("MERGE_HEAD"), format!("{}{}", head, head)).unwrap();
        let f = Fakes::new();
        set_env(&f, "resolve", false);
        let res = resolve(&f.opts(true), None).unwrap();
        assert_eq!(res.exit, Exit::Error);
        assert!(res.reason.contains("octopus"), "{}", res.reason);
    }
}
