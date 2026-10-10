//! The prompts.
//!
//! The LLM's job is to understand and edit files. It does not stage, commit
//! or build: llminus stages what the verdict file declares, builds the result
//! as a scratch commit and comes back with the errors if it doesn't build.

use std::fmt::Write as _;

use crate::context::MergeContext;
use crate::gitutil::git_allow_fail;
use crate::history::SimilarResolution;
use crate::nextref::Reference;
use crate::pull::PullRequest;
use crate::unmerged::{Kind, UnmergedPath};
use crate::verdict::VERDICT_FILE;

/// Everything a conflict prompt is built from
pub struct ConflictPrompt<'a> {
    pub unmerged: &'a [UnmergedPath],
    pub similar: &'a [SimilarResolution],
    pub merge_ctx: &'a MergeContext,
    pub pull_req: Option<&'a PullRequest>,
    /// The commits on each side, for describing non-content conflicts
    pub ours: Option<&'a str>,
    pub theirs: Option<&'a str>,
    /// Whether the LLM can run lei itself (codex runs without network)
    pub lore_access: bool,
    /// How llminus will build test the result (None: it won't)
    pub build: Option<String>,
    /// How linux-next resolved these paths
    pub next_refs: &'a [Reference],
    /// linux-next's "manual merge of" reports for these paths (pre-fetched)
    pub lore_reports: &'a [crate::lore::Message],
    /// List each side's commits and the one-sided identifiers per path
    pub side_context: bool,
}

fn lore_reports(out: &mut String, reports: &[crate::lore::Message]) {
    if reports.is_empty() {
        return;
    }
    out.push_str("## linux-next conflict reports for these files\n\n");
    out.push_str("llminus found these \"manual merge of\" reports on lore.kernel.org. They name the commits \
                  that conflicted and usually show how linux-next resolved it - possibly for an earlier \
                  version of the conflict.\n\n");
    for m in reports {
        let _ = writeln!(out, "### {} ({}, {}, <{}>)\n\n```\n{}\n```\n", m.subject, m.from, m.date, m.message_id,
                         m.body.trim_end());
    }
}

fn next_references(out: &mut String, refs: &[Reference]) {
    if refs.is_empty() {
        return;
    }
    out.push_str("## How linux-next resolved these files

");
    out.push_str("linux-next merged these trees before, and resolved these files by hand. Below is \
                  what was done by hand there (`git show --remerge-diff`: the conflict as git left \
                  it, and the resolution). This is a strong hint, not the answer: linux-next has \
                  been wrong before, and its version of the conflict may differ from yours. \
                  Understand it, check it against both sides, and don't copy it blindly. If you \
                  resolve differently, say why in the notes.\n\n");
    for r in refs {
        let _ = writeln!(out, "### {}: linux-next merge {} (\"{}\"){}\n", r.path, &r.merge[..12], r.subject,
                         if r.same_conflict { " - the very same conflict" } else { " - a different version of the conflict" });
        out.push_str("```diff\n");
        out.push_str(&r.remerge_diff);
        out.push_str("```\n\n");
    }
}

fn one_line_commits(range: &str, path: &str, extra: &[&str], max: usize) -> Vec<String> {
    let max = format!("-{}", max);
    let mut args = vec!["log", "--format=%h (\"%s\")", max.as_str()];
    args.extend_from_slice(extra);
    args.extend_from_slice(&[range, "--", path]);
    git_allow_fail(&args)
        .map(|s| s.lines().map(str::to_string).collect())
        .unwrap_or_default()
}

/// The commits on each side that touched a path, with their Link: tags
fn side_commits(u: &UnmergedPath, ours: &str, theirs: &str, out: &mut String) {
    let Some(base) = git_allow_fail(&["merge-base", ours, theirs]).map(|s| s.trim().to_string()) else {
        return;
    };
    let mut any = false;
    for (name, tip) in [("HEAD (ours)", ours), ("the branch being merged (theirs)", theirs)] {
        let range = format!("{}..{}", base, tip);
        let Some(log) = git_allow_fail(&["log", "--no-merges", "-8", "--format=%h%x00%s%x00%b%x1e", &range, "--", &u.path])
        else { continue };
        let commits: Vec<String> = log.split('\x1e').filter(|c| !c.trim().is_empty()).map(|c| {
            let mut f = c.trim_start_matches('\n').splitn(3, '\0');
            let (h, s, b) = (f.next().unwrap_or(""), f.next().unwrap_or(""), f.next().unwrap_or(""));
            let links: Vec<&str> = b.lines().filter(|l| l.starts_with("Link:")).map(|l| l.trim_start_matches("Link:").trim()).take(2).collect();
            if links.is_empty() { format!("{} (\"{}\")", h, s) } else { format!("{} (\"{}\", {})", h, s, links.join(", ")) }
        }).collect();
        if !commits.is_empty() {
            if !any {
                out.push_str("**Commits touching this file since the merge base:**\n");
                any = true;
            }
            let _ = writeln!(out, "- {}: {}", name, commits.join("; "));
        }
    }
    if any {
        out.push('\n');
    }
}

/// Identifiers that only one side's version of a hunk uses, and what the
/// other side's file says about them: a symbol one side renamed or removed
/// is how silent semantic conflicts happen
fn identifier_table(u: &UnmergedPath, ours: &str, theirs: &str, out: &mut String) {
    fn idents(s: &str) -> std::collections::BTreeSet<String> {
        const COMMON: &[&str] = &["use", "pub", "let", "mut", "for", "int", "the", "and", "not", "new", "ret",
                                  "err", "len", "str", "u8", "u16", "u32", "u64", "i32", "i64", "self", "Self",
                                  "void", "char", "long", "const", "static", "struct", "return", "unsigned",
                                  "goto", "else", "while", "true", "false", "NULL", "impl", "unsafe", "match"];
        s.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .filter(|w| w.len() >= 3 && !COMMON.contains(w))
            .filter(|w| w.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_'))
            .filter(|w| !w.chars().all(|c| c.is_ascii_uppercase() || c == '_') || w.contains('_'))
            .map(str::to_string)
            .collect()
    }
    let Some(base) = git_allow_fail(&["merge-base", ours, theirs]).map(|s| s.trim().to_string()) else {
        return;
    };
    let mut rows = Vec::new();
    for c in &u.regions {
        let (o, t) = (idents(&c.ours_content), idents(&c.theirs_content));
        for (ident, only_in, other_tip, this_tip) in
            o.difference(&t).map(|i| (i, "ours", theirs, ours))
             .chain(t.difference(&o).map(|i| (i, "theirs", ours, theirs)))
        {
            if rows.len() >= 12 {
                break;
            }
            let count = git_allow_fail(&["grep", "-c", "-w", "-F", "-e", ident, other_tip, "--", &u.path])
                .and_then(|s| s.trim().rsplit(':').next().and_then(|n| n.parse::<usize>().ok()))
                .unwrap_or(0);
            let changed = git_allow_fail(&["log", "-1", "--format=%h (\"%s\")", &format!("-S{}", ident),
                                           &format!("{}..{}", base, this_tip), "--", &u.path])
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            rows.push(format!("| `{}` | {} | {} | {} |", ident, only_in, count,
                              changed.map(|c| format!("{} side: {}", only_in, c)).unwrap_or_default()));
        }
    }
    if !rows.is_empty() {
        out.push_str("**Identifiers in only one side's version of the conflict** (check whether the other side \
                      renamed, removed or changed the meaning of any of them):\n\n\
                      | identifier | only in | uses in the other side's version of this file | commit that added/removed it |\n\
                      |---|---|---|---|\n");
        out.push_str(&rows.join("\n"));
        out.push_str("\n\n");
    }
}

/// What each side did to a path that has no conflict markers to edit
fn describe_side_history(u: &UnmergedPath, ours: &str, theirs: &str, out: &mut String) {
    let Some(base) = git_allow_fail(&["merge-base", ours, theirs]).map(|s| s.trim().to_string()) else {
        return;
    };
    for (name, tip) in [("HEAD (ours)", ours), ("the branch being merged (theirs)", theirs)] {
        let range = format!("{}..{}", base, tip);
        let deleted = one_line_commits(&range, &u.path, &["--diff-filter=D"], 1);
        let changed = one_line_commits(&range, &u.path, &[], 5);
        if let Some(d) = deleted.first() {
            let _ = writeln!(out, "- Deleted in {} by {}", name, d);
        } else if !changed.is_empty() {
            let _ = writeln!(out, "- Changed in {} by: {}", name, changed.join(", "));
        }
    }
}

fn header(out: &mut String, pull_req: bool) {
    if pull_req {
        out.push_str("# Linux Kernel Pull Request Merge with Conflict Resolution\n\n");
        out.push_str("You are acting as an experienced kernel maintainer resolving conflicts \
                      from a pull request submission on lore.kernel.org.\n\n");
    } else {
        out.push_str("# Linux Kernel Merge Conflict Resolution\n\n");
        out.push_str("You are acting as an experienced kernel maintainer resolving a merge conflict.\n\n");
    }
    out.push_str("**Important:** Incorrect merge resolutions have historically introduced subtle bugs \
                  that affected millions of users and took months to diagnose. A resolution that \
                  compiles but has semantic errors is worse than no resolution at all.\n\n");
    out.push_str("Take the time to fully understand both sides of the conflict before resolving \
                  anything. If after investigation you're not confident, say so (ABSTAIN, below) - \
                  it's better to escalate to a human than to introduce a subtle bug.\n\n");
}

fn merge_context(out: &mut String, ctx: &MergeContext) {
    out.push_str("## Merge Context\n\n");
    if let Some(ref source) = ctx.merge_source {
        let _ = writeln!(out, "**Merging:** `{}`", source);
    }
    if let Some(ref head) = ctx.head_branch {
        let _ = writeln!(out, "**Into:** `{}`", head);
    }
    if let Some(ref msg) = ctx.merge_message {
        let _ = writeln!(out, "**Merge message:** {}", msg.lines().next().unwrap_or(""));
    }
    out.push('\n');
}

fn pull_request(out: &mut String, pr: &PullRequest) {
    out.push_str("## Pull Request Information\n\n");
    let _ = writeln!(out, "- **Subject:** {}", pr.subject);
    let _ = writeln!(out, "- **From:** {}", pr.from);
    let _ = writeln!(out, "- **Date:** {}", pr.date);
    let _ = writeln!(out, "- **Git URL:** {} {}", pr.git_url, pr.git_ref);
    let _ = writeln!(out, "- **Message ID:** {}\n", pr.message_id);
    out.push_str("### Pull Request Email\n\n");
    out.push_str("Read this email carefully. It contains the maintainer's description of the changes \
                  and may include conflict resolution instructions. Use them as guidance, but \
                  evaluate them critically - there may be cleaner or more efficient solutions, and \
                  maintainers' sample resolutions have been wrong before (e.g. done against the \
                  wrong merge base).\n\n```\n");
    out.push_str(&pr.body);
    out.push_str("\n```\n\n");
}

fn similar_resolutions(out: &mut String, similar: &[SimilarResolution]) {
    if similar.is_empty() {
        return;
    }
    out.push_str("## Similar Historical Resolutions\n\n");
    out.push_str("These conflicts were previously resolved in the Linux kernel. Use `git show <hash>` \
                  to examine the full commit message and context - maintainers often explain \
                  their resolution rationale there. `git show --remerge-diff <hash>` shows exactly \
                  what was resolved by hand.\n\n");
    for (i, result) in similar.iter().enumerate() {
        let r = &result.resolution;
        let _ = writeln!(out, "### Historical Resolution {} (similarity: {:.1}%)\n", i + 1, result.similarity * 100.0);
        let _ = writeln!(out, "- **Commit:** `{}`", r.commit_hash);
        let _ = writeln!(out, "- **Summary:** {}", r.commit_summary);
        let _ = writeln!(out, "- **Author:** {}", r.author);
        let _ = writeln!(out, "- **Date:** {}", r.commit_date);
        let _ = writeln!(out, "- **Files:** {}\n",
                         r.files.iter().map(|f| f.file_path.as_str()).collect::<Vec<_>>().join(", "));
        for file in &r.files {
            let _ = writeln!(out, "#### {}\n", file.file_path);
            for (title, hunks) in [("Ours changed", &file.ours_diff), ("Theirs changed", &file.theirs_diff),
                                   ("Final resolution", &file.resolution_diff)] {
                if !hunks.is_empty() {
                    let _ = writeln!(out, "**{}:**\n```diff", title);
                    for h in hunks.iter() {
                        out.push_str(&h.content);
                        out.push('\n');
                    }
                    out.push_str("```\n\n");
                }
            }
        }
    }
}

/// How to search lore with lei
pub fn lore_section(out: &mut String, examples: &[String]) {
    out.push_str("### Searching lore.kernel.org: use `lei`\n\n");
    out.push_str("Fetching lore.kernel.org pages directly gets blocked as a bot. `lei` queries lore's \
                  search API instead (the `https://lore.kernel.org/all/` external), so use it:\n\n");
    out.push_str("```bash\n# matching messages: message-id, subject, from, date\n\
                  lei q -f json --limit=20 '<query>'\n\
                  # one message in full (add -t for its whole thread)\n\
                  lei q -f mboxrd 'mid:<message-id>'\n```\n\n");
    out.push_str("Results go to stdout and searches are never saved (llminus enforces `--no-save -o -`; \
                  don't write results to a folder). Run lei as a plain command; you may pipe its \
                  output into `grep`, `head` or `tail`, nothing else. Query syntax (public-inbox):\n\n");
    out.push_str("- `s:` subject, `b:` body, `f:` from, `t:` to, `c:` cc\n");
    out.push_str("- `dfn:` file name in a diff, `dfa:` / `dfb:` removed / added lines in a diff, \
                  `dfhh:` diff hunk header (the function name)\n");
    out.push_str("- `m:` message-id, `rt:` received time (`rt:3.months.ago..`), `d:` date\n");
    out.push_str("- `\"quoted phrases\"`, `AND`, `OR`, `NOT`, parentheses. Fetch messages one `mid:` at a time \
                  (`m:a OR m:b` matches nothing). A missing message gives empty output.\n\n");
    if !examples.is_empty() {
        out.push_str("Queries for this conflict:\n\n```\n");
        for e in examples {
            out.push_str(e);
            out.push('\n');
        }
        out.push_str("```\n\n");
    }
}

/// lei queries worth starting with for these conflicts
fn lore_examples(unmerged: &[UnmergedPath], ctx: &MergeContext) -> Vec<String> {
    let mut ex = Vec::new();
    for u in unmerged.iter().take(3) {
        ex.push(format!("s:\"manual merge of\" AND dfn:{}", u.path));
        ex.push(format!("dfn:{} AND rt:3.months.ago..", u.path));
    }
    if let Some(ref src) = ctx.merge_source {
        let tag = src.trim_start_matches("tags/");
        ex.push(format!("s:\"GIT PULL\" AND b:\"{}\"", tag));
    }
    ex
}

fn rules(out: &mut String, build: Option<&str>) {
    out.push_str("## Rules\n\n");
    out.push_str("- **Edit the files only.** Don't run `git add`, `git rm`, `git commit`, `git reset`, \
                  `git checkout`, `git stash` or anything else that changes git state (they are blocked). \
                  llminus stages exactly what your verdict file lists.\n");
    if let Some(build) = build {
        let _ = writeln!(out, "- **Don't build.** llminus builds the result ({}) and comes back to you \
                               with any new errors.", build);
    } else {
        out.push_str("- **Don't build.** This run is not build tested, so be especially careful \
                      with includes, renamed symbols and changed function signatures.\n");
    }
    out.push_str("- If a correct merge needs changes in files that merged without conflicts \
                  (a function renamed on one side and newly used on the other, ...), make them, \
                  and list each as a `semantic_fixup` with a rationale.\n");
    out.push_str("- Don't touch anything else.\n\n");
}

fn verdict_section(out: &mut String, unmerged: &[UnmergedPath]) {
    let _ = writeln!(out, "## When you are done: write `{}`\n", VERDICT_FILE);
    let _ = writeln!(out, "Write this file at the top of the working tree. llminus reads it to know what you \
                           decided, checks it against what you actually changed, and stages accordingly:\n");
    out.push_str("```json\n{\n  \"verdict\": \"RESOLVED\",\n  \"paths\": [\n");
    let mut lines: Vec<String> = unmerged.iter().map(|u| {
        let action = if u.kind().is_content() { "resolved" } else { "keep | delete" };
        format!("    {{\"path\": \"{}\", \"action\": \"{}\"}}", u.path, action)
    }).collect();
    lines.push("    {\"path\": \"path/to/other/file.c\", \"action\": \"semantic_fixup\", \"rationale\": \"why\"}".into());
    out.push_str(&lines.join(",\n"));
    out.push_str("\n  ],\n  \"notes\": \"...\"\n}\n```\n\n");
    out.push_str("- **Every** unmerged path listed above must appear exactly once: `resolved` (you \
                  edited it to the merged result, no conflict markers left), `keep` (the file stays, \
                  as it is now in the working tree) or `delete` (the file goes).\n");
    out.push_str("- Any other file you changed must be listed as `semantic_fixup`, with a rationale. \
                  Unlisted changes make the whole resolution go to a human.\n");
    out.push_str("- `notes`: a short, factual explanation for the merge commit message - what \
                  conflicted, why, and how you resolved it (and why any fixup was needed). Plain \
                  text, a few lines per conflict, no headers. It goes into the kernel's history.\n");
    out.push_str("- If you are not confident, write `{\"verdict\": \"ABSTAIN\", \"notes\": \"<what you \
                  found, what is unclear, the options you see>\"}` instead. A human then decides; \
                  that is far better than a plausible-looking wrong resolution.\n\n");
}

/// The prompt for resolving the unmerged paths of a merge
pub fn conflict_prompt(p: &ConflictPrompt) -> String {
    let mut out = String::new();
    header(&mut out, p.pull_req.is_some());
    if let Some(pr) = p.pull_req {
        pull_request(&mut out, pr);
    }
    merge_context(&mut out, p.merge_ctx);

    out.push_str("## Current Conflicts\n\n");
    for u in p.unmerged {
        let _ = writeln!(out, "### File: {}\n", u.path);
        let kind = u.kind();
        if u.binary {
            out.push_str("**Binary file** - decide which version to keep (and say why).\n\n");
        }
        if !kind.is_content() || u.regions.is_empty() {
            let _ = writeln!(out, "**Conflict:** {} - there are no conflict markers to edit.", kind.describe());
            for m in &u.messages {
                let _ = writeln!(out, "- git: `{}`", m);
            }
            if let (Some(ours), Some(theirs)) = (p.ours, p.theirs) {
                describe_side_history(u, ours, theirs, &mut out);
            }
            match kind {
                Kind::DeletedByThem | Kind::DeletedByUs => out.push_str(
                    "\nFind out *why* it was deleted (removed driver? moved/renamed elsewhere? \
                     replaced?) and whether the other side's changes need to be carried over to \
                     where the code went. Then decide `keep` or `delete`. Keeping a file the other \
                     side deleted usually resurrects dead code that no longer builds or isn't built \
                     at all, which no build test will catch.\n\n"),
                _ => out.push_str("\nDecide what the file should be, edit it if needed, and list it.\n\n"),
            }
            continue;
        }
        if let (true, Some(ours), Some(theirs)) = (p.side_context, p.ours, p.theirs) {
            side_commits(u, ours, theirs, &mut out);
        }
        for c in &u.regions {
            out.push_str("**Our version (HEAD):**\n```\n");
            out.push_str(&c.ours_content);
            out.push_str("\n```\n\n**Their version (being merged):**\n```\n");
            out.push_str(&c.theirs_content);
            out.push_str("\n```\n\n");
            if let Some(ref base) = c.base_content {
                out.push_str("**Base version (common ancestor):**\n```\n");
                out.push_str(base);
                out.push_str("\n```\n\n");
            }
        }
        if let (true, Some(ours), Some(theirs)) = (p.side_context, p.ours, p.theirs) {
            identifier_table(u, ours, theirs, &mut out);
        }
    }

    if p.unmerged.iter().any(|u| u.path.ends_with(".rs")) {
        out.push_str("**Note:** the build test may not cover Rust code (it needs CONFIG_RUST and a Rust \
                      toolchain). Check `use` lists, renamed items and changed signatures by hand; \
                      follow the kernel's Rust import style (one item per line in `use` lists).\n\n");
    }

    next_references(&mut out, p.next_refs);
    lore_reports(&mut out, p.lore_reports);
    similar_resolutions(&mut out, p.similar);

    out.push_str("## Investigation Required\n\n");
    out.push_str("Before resolving anything, research both sides. Work through these phases IN ORDER \
                  and keep notes of what you found.\n\n");
    out.push_str("### Phase 1: Maintainer guidance on lore.kernel.org\n\n");
    out.push_str("Maintainers often describe expected conflicts and their resolution (in pull \
                  requests, and in linux-next's \"manual merge of the X tree with the Y tree\" \
                  reports, which include the resolution linux-next used).\n\n");
    if p.lore_access {
        lore_section(&mut out, &lore_examples(p.unmerged, p.merge_ctx));
    } else {
        out.push_str("(You have no network access in this run; what llminus could find is included above.)\n\n");
    }
    out.push_str("### Phase 2: Understand the context\n\n");
    out.push_str("- What subsystem is this, what does the file do, who maintains it \
                  (`git log --oneline -20 -- <file>`)?\n\n");
    out.push_str("### Phase 3: Trace each side's history\n\n");
    out.push_str("- Ours: `git log --oneline MERGE_HEAD..HEAD -- <file>`, then `git show <commit>`\n");
    out.push_str("- Theirs: `git log --oneline HEAD..MERGE_HEAD -- <file>`, then `git show <commit>`\n");
    out.push_str("- `git log --merge -p -- <file>` shows the commits on both sides that touch the conflict.\n");
    out.push_str("- When a symbol changed meaning or was renamed on one side, check every use the \
                  other side added (`git log -S<symbol>`, `git grep <symbol>`): a renamed function \
                  whose old name was reused with different semantics is the classic silent \
                  semantic conflict.\n\n");

    rules(&mut out, p.build.as_deref());
    verdict_section(&mut out, p.unmerged);

    out.push_str("## Tools Available\n\n```bash\n");
    for s in p.similar {
        let _ = writeln!(out, "git show {}", s.resolution.commit_hash);
    }
    out.push_str("git show <hash>^1  # ours\ngit show <hash>^2  # theirs\n");
    out.push_str("git diff MERGE_HEAD...HEAD -- <file>   # what ours did since the merge base\n");
    out.push_str("git diff HEAD...MERGE_HEAD -- <file>   # what theirs did\n```\n");
    out
}

/// The prompt for fixing a build that a (committed) merge broke
pub fn build_fix_prompt(errors: &str, ctx: &MergeContext, lore_access: bool) -> String {
    let mut out = String::new();
    out.push_str("# Linux Kernel Build Issue Resolution\n\n");
    out.push_str("You are acting as an experienced kernel maintainer. A merge completed without \
                  textual conflicts, but the build now has new warnings or errors (compared to the \
                  merge's first parent) that need to be fixed.\n\n");
    out.push_str("**Important:** A fix that silences a warning but has semantic errors is worse than \
                  no fix at all. Take the time to understand the failure; if you're not confident, \
                  ABSTAIN (below).\n\n");
    merge_context(&mut out, ctx);
    out.push_str("## Build Errors\n\n```\n");
    out.push_str(errors);
    out.push_str("\n```\n\n");
    out.push_str("## Investigation\n\n");
    if lore_access {
        out.push_str("Maintainers and linux-next often report post-merge build failures and their \
                      fixes (\"linux-next: build failure after merge of the X tree\").\n\n");
        lore_section(&mut out, &["s:\"build failure after merge\" AND b:\"<symbol or file>\"".to_string()]);
    }
    out.push_str("- Identify the affected files and both merge parents' changes to them: \
                  `git log --oneline HEAD^2 ^HEAD^1 -- <file>`, `git diff HEAD^1 HEAD -- <file>`.\n");
    out.push_str("- Common causes: a function/macro renamed or removed on one side and used on the \
                  other, changed signatures, missing includes, conflicting definitions.\n\n");
    out.push_str("## Rules\n\n");
    out.push_str("- **Edit the files only** - no `git add`/`commit`/`reset`/... (blocked), and don't \
                  build: llminus builds the result and comes back to you with any remaining errors.\n");
    out.push_str("- Fix the cause, not the symptom; keep the fix minimal.\n\n");
    let _ = writeln!(out, "## When you are done: write `{}`\n", VERDICT_FILE);
    out.push_str("```json\n{\n  \"verdict\": \"RESOLVED\",\n  \"paths\": [\n    {\"path\": \"path/to/file.c\", \"action\": \"fix\"}\n  ],\n  \"notes\": \"What broke, why, and how it was fixed.\"\n}\n```\n\n");
    out.push_str("List every file you changed (`fix`, or `delete` for a removed file). `notes` goes into \
                  the merge commit message: short and factual. If you're not confident, write \
                  `{\"verdict\": \"ABSTAIN\", \"notes\": \"...\"}`.\n");
    out
}

/// The prompt for another round after the result failed to build. It is a
/// fresh LLM session, so it gets everything: what was declared so far, the
/// notes, and the original task.
pub fn repair_prompt(errors: &str, round: u32, max_rounds: u32, manifest: &[String], notes: &str,
                     task: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Build failure after the resolution (repair round {} of {})\n", round, max_rounds);
    out.push_str("A resolution of the task below was made, staged by llminus, and built. The build \
                  has new errors (compared to the merge's first parent):\n\n```\n");
    out.push_str(errors);
    out.push_str("\n```\n\n");
    out.push_str("The working tree holds that resolution. What was declared for it:\n\n```\n");
    for m in manifest {
        out.push_str(m);
        out.push('\n');
    }
    out.push_str("```\n\nIts notes:\n\n```\n");
    out.push_str(notes.trim());
    out.push_str("\n```\n\n");
    out.push_str("Fix the build errors in the working tree. The same rules apply: edit files only, no git \
                  commands that change state, no builds.\n\n");
    let _ = writeln!(out, "Then write `{}` again. Earlier declarations stay in force; list the paths \
                           you change now (a new file as `semantic_fixup` or `fix`, with a rationale), \
                           or change an earlier declaration. Put the notes for the **whole** resolution, \
                           including this fix, in `notes`. If you can't fix it with confidence, ABSTAIN.\n", VERDICT_FILE);
    out.push_str("---\n\n# The original task\n\n");
    out.push_str(task);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unmerged::{ConflictFile, Stage};

    fn stage() -> Option<Stage> {
        Some(Stage { mode: "100644".into(), oid: "1".repeat(40) })
    }

    fn content_conflict() -> UnmergedPath {
        UnmergedPath {
            path: "test.c".into(),
            stages: [stage(), stage(), stage()],
            binary: false,
            regions: vec![ConflictFile {
                path: "test.c".into(),
                ours_content: "int ours;".into(),
                theirs_content: "int theirs;".into(),
                base_content: Some("int base;".into()),
            }],
            messages: vec![],
        }
    }

    fn ctx() -> MergeContext {
        MergeContext {
            merge_source: Some("tags/test".into()),
            head_branch: Some("master".into()),
            merge_message: Some("Merge tags/test".into()),
        }
    }

    fn base_prompt<'a>(um: &'a [UnmergedPath], ctx: &'a MergeContext, pr: Option<&'a PullRequest>) -> ConflictPrompt<'a> {
        ConflictPrompt { unmerged: um, similar: &[], merge_ctx: ctx, pull_req: pr, ours: None, theirs: None,
                         lore_access: true, build: Some("x86_64 allmodconfig".into()), next_refs: &[], lore_reports: &[],
                         side_context: true }
    }

    #[test]
    fn conflict_prompt_has_conflicts_rules_and_verdict() {
        let um = [content_conflict()];
        let c = ctx();
        let p = conflict_prompt(&base_prompt(&um, &c, None));
        assert!(p.contains("Linux Kernel Merge Conflict Resolution"));
        assert!(p.contains("int ours;") && p.contains("int theirs;") && p.contains("int base;"));
        assert!(p.contains(VERDICT_FILE));
        assert!(p.contains("\"path\": \"test.c\", \"action\": \"resolved\""));
        assert!(p.contains("lei q -f json"));
        assert!(p.contains("dfn:test.c"));
        // the old commit/build/reset protocol is gone
        assert!(!p.contains("git commit -m"));
        assert!(!p.contains("reset --soft"));
        assert!(!p.contains("stable build log"));
        assert!(!p.contains("Pull Request Information"));
    }

    #[test]
    fn conflict_prompt_with_pull_request() {
        let um = [content_conflict()];
        let c = ctx();
        let pr = PullRequest {
            message_id: "test@kernel.org".into(),
            subject: "Test PR".into(),
            from: "Author <author@kernel.org>".into(),
            date: "2025-12-11".into(),
            git_url: "git://test".into(),
            git_ref: "tags/test".into(),
            body: "Test summary\n\nResolve by keeping both.".into(),
        };
        let p = conflict_prompt(&base_prompt(&um, &c, Some(&pr)));
        assert!(p.contains("Pull Request Information"));
        assert!(p.contains("Resolve by keeping both"));
    }

    #[test]
    fn modify_delete_is_described_not_hidden() {
        let um = [UnmergedPath {
            path: "drivers/android/binder_alloc.c".into(),
            stages: [stage(), stage(), None],
            binary: false,
            regions: vec![],
            messages: vec!["CONFLICT (modify/delete): drivers/android/binder_alloc.c deleted in theirs and modified in HEAD.".into()],
        }];
        let c = ctx();
        let p = conflict_prompt(&base_prompt(&um, &c, None));
        assert!(p.contains("modified in HEAD (ours), deleted in the branch being merged"));
        assert!(p.contains("CONFLICT (modify/delete)"));
        assert!(p.contains("resurrects dead code"));
        assert!(p.contains("\"action\": \"keep | delete\""));
    }

    #[test]
    fn rust_is_flagged_as_not_build_tested() {
        let mut u = content_conflict();
        u.path = "rust/kernel/fs/file.rs".into();
        let um = [u];
        let c = ctx();
        assert!(conflict_prompt(&base_prompt(&um, &c, None)).contains("may not cover Rust code"));
    }

    #[test]
    fn codex_gets_no_lei_instructions() {
        let um = [content_conflict()];
        let c = ctx();
        let mut bp = base_prompt(&um, &c, None);
        bp.lore_access = false;
        assert!(!conflict_prompt(&bp).contains("lei q"));
    }
}
