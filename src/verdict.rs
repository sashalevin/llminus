//! The verdict file: how the LLM says what it did.
//!
//! The LLM edits files and then writes a small JSON file naming every path
//! it decided on. llminus checks that against what actually changed, and does
//! the staging itself:
//!
//! ```json
//! {
//!   "verdict": "RESOLVED",
//!   "paths": [
//!     {"path": "drivers/foo.c", "action": "resolved"},
//!     {"path": "drivers/old.c", "action": "delete", "rationale": "removed upstream, ..."},
//!     {"path": "include/foo.h", "action": "semantic_fixup", "rationale": "..."}
//!   ],
//!   "notes": "What conflicted and how it was resolved, for the merge commit."
//! }
//! ```
//!
//! Declarations accumulate over repair rounds in a manifest: a later round
//! may change or add entries, not drop them, and the whole manifest is what
//! gets staged and built.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use crate::gitutil::{git_bytes_env, git_path, toplevel};
use crate::unmerged::{has_conflict_markers, UnmergedPath};

/// File name of the verdict, at the top of the working tree (the LLM can
/// write there with every backend; it is removed again afterwards)
pub const VERDICT_FILE: &str = ".llminus-verdict.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Resolving the unmerged paths of a merge in progress
    Conflict,
    /// Fixing the build on top of a committed merge
    BuildFix,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct PathAction {
    pub path: String,
    pub action: String,
    #[serde(default)]
    pub rationale: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Verdict {
    pub verdict: String,
    #[serde(default)]
    pub paths: Vec<PathAction>,
    #[serde(default)]
    pub notes: String,
}

/// What a verdict amounts to
#[derive(Debug)]
pub enum Decision {
    /// The LLM resolved everything; these are the paths to stage
    Resolved(Verdict),
    /// The LLM declined, or the verdict is unusable: a human has to look
    NeedsHuman(String),
}

/// Everything declared so far, path -> its latest declaration
pub type Manifest = BTreeMap<String, PathAction>;

pub fn verdict_path() -> Result<std::path::PathBuf> {
    Ok(Path::new(&toplevel()?).join(VERDICT_FILE))
}

/// Read the verdict file. A missing or unparsable file is a decline.
pub fn load() -> Result<Decision> {
    let path = verdict_path()?;
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(Decision::NeedsHuman(format!("the LLM wrote no verdict file ({})", VERDICT_FILE)));
    };
    let v: Verdict = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return Ok(Decision::NeedsHuman(format!("the verdict file is not valid: {}", e))),
    };
    match v.verdict.to_ascii_uppercase().as_str() {
        "RESOLVED" => Ok(Decision::Resolved(v)),
        "ABSTAIN" | "ERROR" => Ok(Decision::NeedsHuman(format!("the LLM gave up ({}): {}", v.verdict, v.notes.trim()))),
        other => Ok(Decision::NeedsHuman(format!("unknown verdict '{}'", other))),
    }
}

fn path_is_sane(p: &str) -> bool {
    !p.is_empty()
        && Path::new(p).components().all(|c| matches!(c, Component::Normal(_)))
}

/// Check a RESOLVED verdict against the unmerged paths, the paths that
/// changed during this LLM call, and the manifest of earlier rounds
pub fn validate(v: &Verdict, mode: Mode, unmerged: &[UnmergedPath], changed: &BTreeSet<String>,
                prior: &Manifest) -> std::result::Result<(), String> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for pa in &v.paths {
        if !path_is_sane(&pa.path) {
            return Err(format!("bad path in verdict: '{}'", pa.path));
        }
        if !seen.insert(&pa.path) {
            return Err(format!("{} is listed twice in the verdict", pa.path));
        }
        let is_unmerged = unmerged.iter().any(|u| u.path == pa.path);
        let ok = match (mode, pa.action.as_str()) {
            (Mode::Conflict, "resolved" | "keep" | "delete") => is_unmerged,
            (Mode::Conflict, "semantic_fixup") => !is_unmerged,
            (Mode::BuildFix, "fix" | "semantic_fixup" | "delete") => true,
            _ => false,
        };
        if !ok {
            return Err(format!("{}: action '{}' is not valid here{}", pa.path, pa.action,
                               if is_unmerged { " (it is an unmerged path)" } else { "" }));
        }
        if pa.action == "semantic_fixup" && pa.rationale.trim().is_empty() {
            return Err(format!("{}: a semantic_fixup needs a rationale", pa.path));
        }
        // A file declared for the first time must be one this round changed
        // (else it would stage someone else's change)
        if !is_unmerged && !prior.contains_key(&pa.path) && !changed.contains(&pa.path) {
            return Err(format!("{} is listed as {}, but it was not changed", pa.path, pa.action));
        }
    }

    // The first round decides every unmerged path; later rounds may leave
    // earlier decisions as they were
    for u in unmerged {
        if !seen.contains(u.path.as_str()) && !prior.contains_key(&u.path) {
            return Err(format!("unmerged path {} is missing from the verdict", u.path));
        }
    }

    let undeclared: Vec<&String> = changed.iter()
        .filter(|p| p.as_str() != VERDICT_FILE && !seen.contains(p.as_str()) && !prior.contains_key(*p))
        .collect();
    if !undeclared.is_empty() {
        return Err(format!("changed but not listed in the verdict: {}",
                           undeclared.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
    }
    if mode == Mode::BuildFix && prior.is_empty() && v.paths.is_empty() {
        return Err("RESOLVED, but no files were changed".to_string());
    }
    Ok(())
}

/// Add a validated verdict's declarations to the manifest
pub fn extend_manifest(manifest: &mut Manifest, v: &Verdict) {
    for pa in &v.paths {
        manifest.insert(pa.path.clone(), pa.clone());
    }
}

/// Stage the manifest. All-or-nothing: the result is built in a temporary
/// index first, checked there (nothing unmerged, no conflict markers), and
/// only then becomes the real index; files declared deleted are removed from
/// the working tree last. Ok(Err(problems)) when the result isn't usable.
pub fn apply(manifest: &Manifest) -> Result<std::result::Result<(), String>> {
    let top = toplevel()?;

    // Everything must be what it claims before anything is touched
    for pa in manifest.values() {
        let file = Path::new(&top).join(&pa.path);
        let meta = file.symlink_metadata();
        if meta.as_ref().is_ok_and(|m| m.is_dir()) {
            return Ok(Err(format!("{} is a directory; list the files", pa.path)));
        }
        if matches!(pa.action.as_str(), "keep" | "resolved") && meta.is_err() {
            return Ok(Err(format!("{} is listed as {} but does not exist", pa.path, pa.action)));
        }
    }

    // Take git's own lock, as git does: no other git process can change the
    // index meanwhile, and we never overwrite one that is. The result is
    // staged into the lock file and becomes the index by renaming it.
    let index = git_path("index");
    let lock = format!("{}.lock", index);
    let mut lock_file = match std::fs::OpenOptions::new().write(true).create_new(true).open(&lock) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Ok(Err(format!("{} exists: another git process is using the index", lock)));
        }
        Err(e) => return Err(e).with_context(|| format!("Failed to create {}", lock)),
    };
    let release = |why: std::result::Result<(), String>| -> Result<std::result::Result<(), String>> {
        let _ = std::fs::remove_file(&lock);
        Ok(why)
    };
    if let Err(e) = std::fs::File::open(&index)
        .and_then(|mut src| std::io::copy(&mut src, &mut lock_file))
        .and_then(|_| lock_file.sync_all())
    {
        let _ = std::fs::remove_file(&lock);
        return Err(e).with_context(|| format!("Failed to copy {}", index));
    }
    drop(lock_file);

    let env = [("GIT_INDEX_FILE", lock.as_str())];
    let staged = (|| -> Result<std::result::Result<(), String>> {
        for pa in manifest.values() {
            let exists = Path::new(&top).join(&pa.path).symlink_metadata().is_ok();
            if pa.action == "delete" || !exists {
                git_bytes_env(&["rm", "-q", "--cached", "--ignore-unmatch", "--", &pa.path], &env)?;
            } else {
                git_bytes_env(&["add", "--", &pa.path], &env)?;
            }
        }
        let mut problems = Vec::new();
        let left = String::from_utf8_lossy(&git_bytes_env(&["ls-files", "-u"], &env)?).to_string();
        if !left.trim().is_empty() {
            let paths: BTreeSet<&str> = left.lines().filter_map(|l| l.split('\t').nth(1)).collect();
            problems.push(format!("still unmerged: {}", paths.into_iter().collect::<Vec<_>>().join(", ")));
        }
        for pa in manifest.values().filter(|pa| pa.action != "delete") {
            if let Ok(blob) = git_bytes_env(&["show", &format!(":{}", pa.path)], &env)
                && !blob.iter().take(8000).any(|&b| b == 0)
                && has_conflict_markers(&String::from_utf8_lossy(&blob))
            {
                problems.push(format!("{} still has conflict markers", pa.path));
            }
        }
        Ok(if problems.is_empty() { Ok(()) } else { Err(problems.join("; ")) })
    })();
    match staged {
        Ok(Ok(())) => {}
        Ok(Err(problems)) => return release(Err(problems)),
        Err(e) => {
            let _ = std::fs::remove_file(&lock);
            return Err(e);
        }
    }

    // Files declared deleted go to a trash directory first, so that nothing
    // is lost if publishing the index fails. It is this call's own (made on
    // the first deletion, uniquely named), so no other run can touch it.
    let mut trash: Option<std::path::PathBuf> = None;
    let mut moved: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();
    // Put moved files back. The trash directory only goes once everything in
    // it is back: if a file can't be restored, the trash holds its only copy,
    // so it stays and the error says where.
    let restore = |moved: &[(std::path::PathBuf, std::path::PathBuf)], trash: &Option<std::path::PathBuf>| -> String {
        let stuck: Vec<String> = moved.iter().rev()
            .filter(|(orig, saved)| std::fs::rename(saved, orig).is_err())
            .map(|(orig, saved)| format!("{} (saved as {})", orig.display(), saved.display()))
            .collect();
        if stuck.is_empty() {
            if let Some(t) = trash {
                let _ = std::fs::remove_dir_all(t);
            }
            String::new()
        } else {
            format!("; could not put back {}", stuck.join(", "))
        }
    };
    for (n, pa) in manifest.values().filter(|pa| pa.action == "delete").enumerate() {
        let file = Path::new(&top).join(&pa.path);
        if file.symlink_metadata().is_err() {
            continue;
        }
        let made = match &trash {
            Some(t) => Ok(t.clone()),
            None => tempfile::Builder::new().prefix(".llminus-trash.").tempdir_in(&top).map(|d| d.keep()),
        };
        let res = made.and_then(|t| {
            trash = Some(t.clone());
            let saved = t.join(n.to_string());
            std::fs::rename(&file, &saved).map(|_| saved)
        });
        match res {
            Ok(saved) => moved.push((file, saved)),
            Err(e) => {
                let stuck = restore(&moved, &trash);
                let _ = std::fs::remove_file(&lock);
                return Err(e).with_context(|| format!("Failed to delete {}{}", pa.path, stuck));
            }
        }
    }

    if let Err(e) = std::fs::rename(&lock, &index) {
        let stuck = restore(&moved, &trash);
        let _ = std::fs::remove_file(&lock);
        return Err(e).with_context(|| format!("Failed to update {}{}", index, stuck));
    }
    if let Some(t) = &trash {
        let _ = std::fs::remove_dir_all(t);
    }
    Ok(Ok(()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unmerged::Stage;

    fn unmerged(path: &str) -> UnmergedPath {
        let s = || Some(Stage { mode: "100644".into(), oid: "0".repeat(40) });
        UnmergedPath { path: path.into(), stages: [s(), s(), s()], binary: false, regions: vec![], messages: vec![] }
    }

    fn verdict(paths: &[(&str, &str, &str)]) -> Verdict {
        Verdict {
            verdict: "RESOLVED".into(),
            paths: paths.iter().map(|(p, a, r)| PathAction { path: p.to_string(), action: a.to_string(), rationale: r.to_string() }).collect(),
            notes: "n".into(),
        }
    }

    fn set(p: &[&str]) -> BTreeSet<String> {
        p.iter().map(|s| s.to_string()).collect()
    }

    fn manifest(v: &Verdict) -> Manifest {
        let mut m = Manifest::new();
        extend_manifest(&mut m, v);
        m
    }

    #[test]
    fn every_unmerged_path_must_be_decided() {
        let um = [unmerged("a.c"), unmerged("b.c")];
        let v = verdict(&[("a.c", "resolved", "")]);
        let e = validate(&v, Mode::Conflict, &um, &set(&["a.c"]), &Manifest::new()).unwrap_err();
        assert!(e.contains("b.c"), "{}", e);
        let v = verdict(&[("a.c", "resolved", ""), ("b.c", "delete", "")]);
        validate(&v, Mode::Conflict, &um, &set(&["a.c", "b.c"]), &Manifest::new()).unwrap();
    }

    #[test]
    fn other_edits_must_be_declared_fixups() {
        let um = [unmerged("a.c")];
        let none = Manifest::new();
        let v = verdict(&[("a.c", "resolved", "")]);
        let e = validate(&v, Mode::Conflict, &um, &set(&["a.c", "x.h"]), &none).unwrap_err();
        assert!(e.contains("x.h"), "{}", e);

        let v = verdict(&[("a.c", "resolved", ""), ("x.h", "semantic_fixup", "")]);
        assert!(validate(&v, Mode::Conflict, &um, &set(&["a.c", "x.h"]), &none).unwrap_err().contains("rationale"));

        let v = verdict(&[("a.c", "resolved", ""), ("x.h", "semantic_fixup", "renamed API")]);
        validate(&v, Mode::Conflict, &um, &set(&["a.c", "x.h", VERDICT_FILE]), &none).unwrap();
    }

    #[test]
    fn a_new_fixup_must_have_been_changed() {
        let um = [unmerged("a.c")];
        let v = verdict(&[("a.c", "resolved", ""), ("dirty.c", "semantic_fixup", "x")]);
        let e = validate(&v, Mode::Conflict, &um, &set(&["a.c"]), &Manifest::new()).unwrap_err();
        assert!(e.contains("was not changed"), "{}", e);
    }

    #[test]
    fn repair_rounds_carry_the_manifest() {
        let um = [unmerged("a.c")];
        let first = verdict(&[("a.c", "resolved", ""), ("x.h", "semantic_fixup", "renamed API")]);
        let prior = manifest(&first);
        // round 2 changes x.h again without listing it, and lists nothing else: fine,
        // x.h stays declared and is restaged from the manifest
        let v = verdict(&[]);
        validate(&v, Mode::Conflict, &um, &set(&["x.h"]), &prior).unwrap();
        // but a new file still has to be declared
        let e = validate(&v, Mode::Conflict, &um, &set(&["y.h"]), &prior).unwrap_err();
        assert!(e.contains("y.h"), "{}", e);
    }

    #[test]
    fn bad_verdicts() {
        let um = [unmerged("a.c")];
        for (paths, why) in [
            (vec![("a.c", "resolved", ""), ("a.c", "keep", "")], "twice"),
            (vec![("../etc/passwd", "semantic_fixup", "x")], "bad path"),
            (vec![("/abs", "semantic_fixup", "x")], "bad path"),
            (vec![("a.c", "semantic_fixup", "x")], "unmerged path"),
            (vec![("a.c", "frobnicate", "")], "not valid"),
        ] {
            let v = verdict(&paths);
            let e = validate(&v, Mode::Conflict, &um, &set(&["a.c"]), &Manifest::new()).unwrap_err();
            assert!(e.contains(why), "{:?}: {}", paths, e);
        }
    }

    #[test]
    fn staging_respects_another_gits_index_lock() {
        use crate::testutil::*;
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        create_commit(&dir, "a.c", "a\n", "base");
        std::fs::write(dir.path().join("a.c"), "changed\n").unwrap();
        let lock = crate::gitutil::git_path("index.lock");
        std::fs::write(&lock, "").unwrap();
        let m = manifest(&verdict(&[("a.c", "fix", "")]));
        let r = apply(&m).unwrap();
        assert!(r.unwrap_err().contains("another git process"));
        assert!(std::path::Path::new(&lock).exists(), "someone else's lock is left alone");
        std::fs::remove_file(&lock).unwrap();
        apply(&m).unwrap().unwrap();
        assert_eq!(crate::gitutil::git(&["show", ":a.c"]).unwrap(), "changed\n");
        assert!(!std::path::Path::new(&lock).exists());
    }

    #[test]
    fn build_fix_mode() {
        let v = verdict(&[("x.c", "fix", "")]);
        validate(&v, Mode::BuildFix, &[], &set(&["x.c"]), &Manifest::new()).unwrap();
        let v = verdict(&[]);
        assert!(validate(&v, Mode::BuildFix, &[], &set(&[]), &Manifest::new()).is_err());
    }
}
