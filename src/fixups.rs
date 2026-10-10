//! Recording successful resolutions for exact replay later.
//!
//! rerere only replays conflicted files, so a fixup the LLM had to make in a
//! cleanly merged file is lost when the same merge is redone. Here the whole
//! resolution (the staged result against git's own automatic merge,
//! AUTO_MERGE) is recorded with a manifest of exactly what it applies to:
//! both parent trees, the merge bases, every unmerged path's stages and the
//! automatic-merge blob of every other path it changed. It is only ever to
//! be replayed when all of those match.
//!
//! Recording only, for now: nothing replays these until the evaluation shows
//! replays are safe.

use anyhow::{Context, Result};
use serde_json::json;
use std::path::PathBuf;

use crate::gitutil::{git, git_allow_fail, git_with_input, rev_parse};
use crate::unmerged::UnmergedPath;

fn data_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llminus/fixups")
}

/// Record the staged resolution of the merge of `merge_commit` into `head`.
/// Returns where it went, or None if there is no AUTO_MERGE to record against.
pub fn record(head: &str, merge_commit: &str, unmerged: &[UnmergedPath], other_paths: &[String])
              -> Result<Option<PathBuf>> {
    let Some(auto_merge) = rev_parse("AUTO_MERGE") else {
        return Ok(None);
    };
    let tree = |c: &str| rev_parse(&format!("{}^{{tree}}", c)).unwrap_or_default();
    let bases: Vec<String> = git_allow_fail(&["merge-base", "--all", head, merge_commit])
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_string)
        .collect();
    let blob_at = |rev: &str, path: &str| rev_parse(&format!("{}:{}", rev, path));

    let manifest = json!({
        "version": 1,
        "parent_trees": [tree(head), tree(merge_commit)],
        "merge_bases": bases,
        "unmerged": unmerged.iter().map(|u| json!({
            "path": u.path,
            "stages": u.stages.iter().map(|s| s.as_ref().map(|s| format!("{} {}", s.mode, s.oid))).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "other_paths": other_paths.iter().map(|p| json!({
            "path": p,
            "auto_merge_blob": blob_at(&auto_merge, p),
        })).collect::<Vec<_>>(),
        "result_tree": git(&["write-tree"])?.trim(),
    });
    let manifest_text = serde_json::to_string_pretty(&manifest)?;
    let key = git_with_input(&["hash-object", "--stdin"], manifest_text.as_bytes())?.trim().to_string();

    let diff = git(&["diff", "--cached", "--binary", "--full-index", &auto_merge, "--"])?;
    let dir = data_dir().join(&key);
    std::fs::create_dir_all(&dir).with_context(|| format!("Failed to create {}", dir.display()))?;
    std::fs::write(dir.join("manifest.json"), manifest_text)?;
    std::fs::write(dir.join("resolution.diff"), diff)?;
    Ok(Some(dir))
}
