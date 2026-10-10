//! Snapshots of the repository state around an LLM call.
//!
//! The LLM may only edit files; llminus owns git. A snapshot taken before the
//! call and one taken after tell us which paths the LLM changed (in the index
//! or the working tree) and whether it touched git state it must leave alone:
//! HEAD, the branch, or the merge in progress. Only this worktree's state is
//! compared - other worktrees (parallel merge workers) legitimately change
//! shared refs at the same time.

use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::gitutil::{git_allow_fail, git_bytes, git_path, git_with_input, rev_parse, toplevel};

/// Files in the git directory that describe an in-progress merge (a commit
/// during the merge removes all of them; ORIG_HEAD changes on a reset)
const MERGE_STATE_FILES: [&str; 5] = ["MERGE_HEAD", "MERGE_MSG", "MERGE_MODE", "AUTO_MERGE", "ORIG_HEAD"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    head: Option<String>,
    symbolic_head: Option<String>,
    branch_tip: Option<String>,
    merge_files: Vec<(String, Option<Vec<u8>>)>,
    /// (path, stage) -> "mode oid"
    index: BTreeMap<(String, u8), String>,
    /// Paths whose working tree content differs from the index, or that are
    /// untracked -> "<file type and mode> <content hash, or symlink target>"
    /// ("" if missing)
    worktree: BTreeMap<String, String>,
}

fn index_entries() -> Result<BTreeMap<(String, u8), String>> {
    let raw = git_bytes(&["ls-files", "-s", "-z"])?;
    let mut map = BTreeMap::new();
    for rec in raw.split(|&b| b == 0).filter(|r| !r.is_empty()) {
        let rec = String::from_utf8_lossy(rec);
        let Some((meta, path)) = rec.split_once('\t') else { continue };
        let mut f = meta.split(' ');
        let (Some(mode), Some(oid), Some(stage)) = (f.next(), f.next(), f.next()) else { continue };
        map.insert((path.to_string(), stage.parse().unwrap_or(0)), format!("{} {}", mode, oid));
    }
    Ok(map)
}

/// Paths `git status` lists as changed in the working tree or untracked
fn worktree_changes() -> Result<BTreeMap<String, String>> {
    let raw = git_bytes(&["status", "--porcelain=v2", "-z", "--untracked-files=all", "--no-renames"])?;
    let mut paths = Vec::new();
    for rec in raw.split(|&b| b == 0).filter(|r| !r.is_empty()) {
        let rec = String::from_utf8_lossy(rec);
        // "1 XY sub mH mI mW hH hI path", "u XY sub m1 m2 m3 mW h1 h2 h3 path", "? path"
        let path = match rec.chars().next() {
            Some('1') => rec.splitn(9, ' ').nth(8),
            Some('u') => rec.splitn(11, ' ').nth(10),
            Some('?') => rec.get(2..),
            _ => None,
        };
        if let Some(p) = path {
            paths.push(p.to_string());
        }
    }

    use std::os::unix::fs::PermissionsExt;
    let top = toplevel()?;
    let mut map = BTreeMap::new();
    // regular files are hashed in one go (names with a newline, which
    // --stdin-paths can't take, one by one)
    let mut regular = Vec::new();
    for p in &paths {
        let full = Path::new(&top).join(p);
        let Ok(meta) = full.symlink_metadata() else { continue };
        let mode = meta.permissions().mode();
        if meta.file_type().is_symlink() {
            let target = std::fs::read_link(&full).map(|t| t.to_string_lossy().to_string()).unwrap_or_default();
            map.insert(p.clone(), format!("link {}", target));
        } else if meta.is_file() {
            regular.push((p.clone(), mode));
        } else {
            map.insert(p.clone(), format!("other {:o}", mode));
        }
    }
    let (plain, odd): (Vec<_>, Vec<_>) = regular.into_iter().partition(|(p, _)| !p.contains('\n'));
    if !plain.is_empty() {
        let input: String = plain.iter().map(|(p, _)| format!("{}/{}\n", top, p)).collect();
        let hashes = git_with_input(&["hash-object", "--no-filters", "--stdin-paths"], input.as_bytes())?;
        for ((p, mode), h) in plain.iter().zip(hashes.lines()) {
            map.insert(p.clone(), format!("file {:o} {}", mode & 0o111, h));
        }
    }
    for (p, mode) in odd {
        let full = format!("{}/{}", top, p);
        let h = crate::gitutil::git(&["hash-object", "--no-filters", "--", &full])?;
        map.insert(p, format!("file {:o} {}", mode & 0o111, h.trim()));
    }
    for p in paths {
        map.entry(p).or_default();
    }
    Ok(map)
}

pub fn take() -> Result<Snapshot> {
    let symbolic_head = git_allow_fail(&["symbolic-ref", "-q", "HEAD"])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let branch_tip = symbolic_head.as_deref().and_then(rev_parse);
    Ok(Snapshot {
        head: rev_parse("HEAD"),
        symbolic_head,
        branch_tip,
        merge_files: MERGE_STATE_FILES
            .iter()
            .map(|n| (n.to_string(), std::fs::read(git_path(n)).ok()))
            .collect(),
        index: index_entries()?,
        worktree: worktree_changes()?,
    })
}

impl Snapshot {
    /// Git state that changed between `self` (before) and `after`, which the
    /// LLM must not touch
    pub fn git_violations(&self, after: &Snapshot) -> Vec<String> {
        let mut v = Vec::new();
        if self.head != after.head {
            v.push(format!("HEAD moved from {:?} to {:?}", self.head, after.head));
        }
        if self.symbolic_head != after.symbolic_head {
            v.push(format!("HEAD switched from {:?} to {:?}", self.symbolic_head, after.symbolic_head));
        }
        if self.branch_tip != after.branch_tip {
            v.push(format!("the branch moved from {:?} to {:?}", self.branch_tip, after.branch_tip));
        }
        for ((name, b), (_, a)) in self.merge_files.iter().zip(&after.merge_files) {
            if b != a {
                v.push(format!("{} was {}", name, if a.is_none() { "removed" } else { "changed" }));
            }
        }
        v
    }

    /// Paths whose index entries or working tree content changed
    pub fn changed_paths(&self, after: &Snapshot) -> BTreeSet<String> {
        let mut changed = BTreeSet::new();
        let keys: BTreeSet<&(String, u8)> = self.index.keys().chain(after.index.keys()).collect();
        for k in keys {
            if self.index.get(k) != after.index.get(k) {
                changed.insert(k.0.clone());
            }
        }
        let paths: BTreeSet<&String> = self.worktree.keys().chain(after.worktree.keys()).collect();
        for p in paths {
            if self.worktree.get(p) != after.worktree.get(p) {
                changed.insert(p.clone());
            }
        }
        changed
    }
}

/// Tracked paths with changes in the working tree that are not staged: after
/// llminus has staged a resolution, there must be none
pub fn unstaged_tracked() -> Result<Vec<String>> {
    let raw = git_bytes(&["diff", "--name-only", "-z"])?;
    Ok(raw.split(|&b| b == 0).filter(|r| !r.is_empty()).map(|r| String::from_utf8_lossy(r).to_string()).collect())
}

/// Snapshot of an in-progress merge, taken before handing it to the LLM
pub struct MergeState {
    head: String,
    /// The commit being merged (MERGE_HEAD may name a signed tag)
    merge_commit: String,
    files: Vec<(String, Vec<u8>)>,
}

/// Save the in-progress merge state, if a merge is in progress
pub fn save_merge_state() -> Option<MergeState> {
    let merge_head = std::fs::read_to_string(git_path("MERGE_HEAD")).ok()?;
    let merge_head = merge_head.lines().next().unwrap_or("").trim();
    let merge_commit = rev_parse(&format!("{}^{{commit}}", merge_head))?;
    let head = rev_parse("HEAD")?;
    let files = MERGE_STATE_FILES
        .iter()
        .map(|name| git_path(name))
        .filter_map(|path| std::fs::read(&path).ok().map(|content| (path, content)))
        .collect();

    Some(MergeState { head, merge_commit, files })
}

/// An LLM that commits during a merge concludes it, and undoing the commit
/// with `git reset --soft HEAD~1` doesn't bring MERGE_HEAD back, so the next
/// `git commit` would record a single-parent commit that flattens the whole
/// merge. Restore the merge state when that is what happened. Returns whether
/// it did - which is still a violation the caller reports: the LLM committed.
pub fn restore_merge_state(state: &MergeState) -> bool {
    if Path::new(&git_path("MERGE_HEAD")).exists() {
        return false;
    }

    // Only restore if HEAD is back where it was and the commit it left was a
    // merge of the same two parents, i.e. a temporary commit that was undone.
    // Anything else (e.g. git merge --abort) is left alone.
    let undone_merge = rev_parse("HEAD").as_deref() == Some(state.head.as_str())
        && rev_parse("HEAD@{1}^1").as_deref() == Some(state.head.as_str())
        && rev_parse("HEAD@{1}^2").as_deref() == Some(state.merge_commit.as_str());

    if !undone_merge {
        eprintln!("Warning: the merge is no longer in progress. Check the result before committing.");
        return false;
    }

    for (path, content) in &state.files {
        if let Err(e) = std::fs::write(path, content) {
            eprintln!("Warning: failed to restore {}: {}", path, e);
            return false;
        }
    }
    eprintln!("Restored the merge state cleared by the LLM's temporary commit.");
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gitutil::git;
    use crate::testutil::*;
    use std::fs;

    #[test]
    fn test_restore_merge_state_after_temporary_commit() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);
        let merge_head = fs::read_to_string(git_path("MERGE_HEAD")).unwrap();
        assert_eq!(git(&["cat-file", "-t", merge_head.trim()]).unwrap().trim(), "tag");

        let state = save_merge_state().unwrap();

        // What the prompt asks the LLM to do around the build test
        create_commit(&dir, "file.c", &numbered_lines(10), "WIP: testing merge resolution");
        git(&["reset", "--soft", "HEAD~1"]).unwrap();
        assert!(!Path::new(&git_path("MERGE_HEAD")).exists());

        restore_merge_state(&state);
        assert_eq!(fs::read_to_string(git_path("MERGE_HEAD")).unwrap(), merge_head);
    }

    #[test]
    fn test_restore_merge_state_leaves_aborted_merge() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        start_conflicting_merge(&dir);

        let state = save_merge_state().unwrap();
        git(&["merge", "--abort"]).unwrap();

        restore_merge_state(&state);
        assert!(!Path::new(&git_path("MERGE_HEAD")).exists());
    }

}
