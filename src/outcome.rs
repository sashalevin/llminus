//! How a run ends: the exit status callers act on, and the notes for the
//! merge commit.
//!
//!   0  resolved (or fixed), staged and - unless --no-build - build tested
//!   1  error: bad usage, a git failure, or the LLM touched git state
//!   2  needs a human: the LLM declined, left conflicts, made undeclared
//!      changes, or the build still fails after the repair rounds
//!   3  the build hosts failed: nothing was judged, blame nothing
//!
//! Status chatter goes to stderr. On 0 only, the result is written to
//! `$(git rev-parse --git-path LLMINUS_RESULT)`:
//!
//! ```text
//! llminus-invocation: <$LLMINUS_INVOCATION_ID, when the caller set it>
//! llminus-mode: conflict | build-fix | pull
//! llminus-staged: yes | no          (no: nothing needed changing)
//! llminus-build: PASS | CACHED_PASS | NOT_BUILT
//! llminus-parents: <sha> [<sha>]
//! llminus-next: AGREE | WEAK_AGREE | WEAK_DIFFER | DIFFER | NONE
//!                                   (vs linux-next's resolutions; DIFFER only
//!                                    with --allow-next-differ)
//! llminus-tree: <the staged tree>
//!
//! <notes for the commit message, ending in "Assisted-by: LLM">
//! ```
//!
//! so a caller can never pick up notes from another run, and can tell a
//! staged fix from "nothing to fix: it builds".

use anyhow::{Context, Result};
use std::io::Write;

use crate::gitutil::git_path;

pub const RESULT_FILE: &str = "LLMINUS_RESULT";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    Done = 0,
    Error = 1,
    NeedsHuman = 2,
    Infra = 3,
}

/// Remove a result file left by an earlier run
pub fn clear_result() {
    let _ = std::fs::remove_file(git_path(RESULT_FILE));
}

/// The resolution notes plus llminus's facts (build result, ...)
pub fn format_notes(notes: &str, facts: &[String]) -> String {
    let mut s = notes.trim().to_string();
    if !facts.is_empty() {
        if !s.is_empty() {
            s.push_str("\n\n");
        }
        s.push_str(&facts.join("\n"));
    }
    s.trim_end().to_string()
}

/// What goes into the merge commit: the notes and the disclosure trailer
/// (Documentation/process/coding-assistants.rst)
pub fn commit_text(body: &str) -> String {
    format!("{}\n\nAssisted-by: LLM\n", body.trim_end())
}

/// Write the result atomically (temporary file + rename): header fields
/// (name without the "llminus-" prefix, value), a blank line, the notes
pub fn write_result(fields: &[(&str, String)], text: &str) -> Result<()> {
    let path = git_path(RESULT_FILE);
    let tmp = format!("{}.tmp.{}", path, std::process::id());
    {
        let mut f = std::fs::File::create(&tmp).with_context(|| format!("Failed to create {}", tmp))?;
        if let Ok(id) = std::env::var("LLMINUS_INVOCATION_ID") {
            writeln!(f, "llminus-invocation: {}", id)?;
        }
        for (k, v) in fields {
            writeln!(f, "llminus-{}: {}", k, v)?;
        }
        writeln!(f)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &path).with_context(|| format!("Failed to write {}", path))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notes_end_with_assisted_by() {
        let n = commit_text(&format_notes("Took both changes.\n", &["Build tested: PASS".into()]));
        assert_eq!(n, "Took both changes.\n\nBuild tested: PASS\n\nAssisted-by: LLM\n");
    }
}
