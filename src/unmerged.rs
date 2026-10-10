//! The unmerged paths of an in-progress merge.
//!
//! Every path with stage entries in the index is reported, whatever the kind
//! of conflict: content conflicts (with marker regions in the working tree),
//! but also modify/delete, add/add, binary and rename conflicts, which leave
//! no markers to find. Whether a path is unmerged is all correctness relies
//! on; the conflict kind is a description for the prompt.

use anyhow::{Context, Result};

use crate::gitutil::{git, git_allow_fail, git_bytes, git_path, rev_parse};

/// One stage entry of an unmerged path
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    pub mode: String,
    pub oid: String,
}

/// One region between conflict markers
#[derive(Debug, Clone)]
pub struct ConflictFile {
    pub path: String,
    pub ours_content: String,
    pub theirs_content: String,
    pub base_content: Option<String>,
}

impl ConflictFile {
    /// Generate embedding text for this conflict
    pub fn to_embedding_text(&self) -> String {
        let mut text = format!("File: {}\n\n", self.path);

        text.push_str("=== OURS ===\n");
        text.push_str(&self.ours_content);
        text.push_str("\n\n");

        text.push_str("=== THEIRS ===\n");
        text.push_str(&self.theirs_content);
        text.push('\n');

        if let Some(ref base) = self.base_content {
            text.push_str("\n=== BASE ===\n");
            text.push_str(base);
            text.push('\n');
        }

        text
    }
}

/// What kind of conflict a path is in, from which stages exist
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Changed on both sides (or renamed to the same path)
    BothModified,
    /// Added on both sides with different content
    BothAdded,
    /// Modified in ours (HEAD), deleted in theirs
    DeletedByThem,
    /// Deleted in ours (HEAD), modified in theirs
    DeletedByUs,
    /// Only ours has it, e.g. a rename/delete or directory/file conflict
    AddedByUs,
    /// Only theirs has it
    AddedByThem,
    /// Deleted on both sides (e.g. renamed differently on each)
    BothDeleted,
}

impl Kind {
    pub fn describe(self) -> &'static str {
        match self {
            Kind::BothModified => "changed on both sides",
            Kind::BothAdded => "added on both sides with different content",
            Kind::DeletedByThem => "modified in HEAD (ours), deleted in the branch being merged (theirs)",
            Kind::DeletedByUs => "deleted in HEAD (ours), modified in the branch being merged (theirs)",
            Kind::AddedByUs => "exists only in HEAD (ours) (e.g. a rename/delete or directory/file conflict)",
            Kind::AddedByThem => "exists only in the branch being merged (theirs) (e.g. a rename/delete or directory/file conflict)",
            Kind::BothDeleted => "deleted on both sides (e.g. renamed differently on each)",
        }
    }

    /// Whether the conflict can be resolved by editing conflict markers
    pub fn is_content(self) -> bool {
        matches!(self, Kind::BothModified | Kind::BothAdded)
    }
}

/// An unmerged path
#[derive(Debug, Clone)]
pub struct UnmergedPath {
    pub path: String,
    /// Stages 1 (base), 2 (ours) and 3 (theirs)
    pub stages: [Option<Stage>; 3],
    /// Any stage's blob looks binary (has a NUL in its first 8000 bytes)
    pub binary: bool,
    /// Conflict marker regions in the working tree file
    pub regions: Vec<ConflictFile>,
    /// git's own CONFLICT messages that mention this path
    pub messages: Vec<String>,
}

impl UnmergedPath {
    pub fn kind(&self) -> Kind {
        match (self.stages[0].is_some(), self.stages[1].is_some(), self.stages[2].is_some()) {
            (_, true, true) => if self.stages[0].is_some() { Kind::BothModified } else { Kind::BothAdded },
            (true, true, false) => Kind::DeletedByThem,
            (true, false, true) => Kind::DeletedByUs,
            (false, true, false) => Kind::AddedByUs,
            (false, false, true) => Kind::AddedByThem,
            _ => Kind::BothDeleted,
        }
    }
}

/// Parse `git ls-files -u -z` output: "<mode> <oid> <stage>\t<path>\0" records
pub fn parse_ls_files_u(raw: &[u8]) -> Vec<UnmergedPath> {
    let mut paths: Vec<UnmergedPath> = Vec::new();
    for rec in raw.split(|&b| b == 0).filter(|r| !r.is_empty()) {
        let rec = String::from_utf8_lossy(rec);
        let Some((meta, path)) = rec.split_once('\t') else { continue };
        let mut fields = meta.split(' ');
        let (Some(mode), Some(oid), Some(stage)) = (fields.next(), fields.next(), fields.next()) else {
            continue;
        };
        let Ok(stage) = stage.parse::<usize>() else { continue };
        if !(1..=3).contains(&stage) {
            continue;
        }
        let entry = Stage { mode: mode.to_string(), oid: oid.to_string() };
        match paths.iter_mut().find(|p| p.path == path) {
            Some(p) => p.stages[stage - 1] = Some(entry),
            None => {
                let mut p = UnmergedPath {
                    path: path.to_string(),
                    stages: [None, None, None],
                    binary: false,
                    regions: Vec::new(),
                    messages: Vec::new(),
                };
                p.stages[stage - 1] = Some(entry);
                paths.push(p);
            }
        }
    }
    paths
}

/// Append a line to a string, adding newline separator if non-empty
fn append_line(s: &mut String, line: &str) {
    if !s.is_empty() {
        s.push('\n');
    }
    s.push_str(line);
}

/// A conflict marker line: at least seven of the marker's character (more
/// with a conflict-marker-size attribute), then nothing or a space
fn is_marker(line: &str, marker: &str) -> bool {
    let Some(c) = marker.chars().next() else { return false };
    let run = line.chars().take_while(|&x| x == c).count();
    let rest = &line[run * c.len_utf8()..];
    run >= 7 && (rest.is_empty() || rest.starts_with(' '))
}

/// Parse conflict markers (standard and diff3/zdiff3 style) out of `content`
pub fn parse_conflict_markers(path: &str, content: &str) -> Vec<ConflictFile> {
    enum State {
        Outside,
        InOurs,
        InBase,
        InTheirs,
    }

    let mut conflicts = Vec::new();
    let mut state = State::Outside;
    let mut ours = String::new();
    let mut theirs = String::new();
    let mut base: Option<String> = None;

    for line in content.lines() {
        if is_marker(line, "<<<<<<<") {
            state = State::InOurs;
            ours.clear();
            theirs.clear();
            base = None;
        } else if matches!(state, State::InOurs) && is_marker(line, "|||||||") {
            state = State::InBase;
            base = Some(String::new());
        } else if matches!(state, State::InOurs | State::InBase) && is_marker(line, "=======") {
            state = State::InTheirs;
        } else if matches!(state, State::InTheirs) && is_marker(line, ">>>>>>>") {
            conflicts.push(ConflictFile {
                path: path.to_string(),
                ours_content: std::mem::take(&mut ours),
                theirs_content: std::mem::take(&mut theirs),
                base_content: base.take(),
            });
            state = State::Outside;
        } else {
            match state {
                State::InOurs => append_line(&mut ours, line),
                State::InBase => {
                    if let Some(ref mut b) = base {
                        append_line(b, line);
                    }
                }
                State::InTheirs => append_line(&mut theirs, line),
                State::Outside => {}
            }
        }
    }

    conflicts
}

/// Whether `content` still has conflict markers
pub fn has_conflict_markers(content: &str) -> bool {
    !parse_conflict_markers("", content).is_empty()
        || content.lines().any(|l| is_marker(l, "<<<<<<<") || is_marker(l, ">>>>>>>"))
}

/// Whether a blob looks binary, the way git decides it
fn blob_is_binary(oid: &str) -> bool {
    match git_bytes(&["cat-file", "blob", oid]) {
        Ok(data) => data.iter().take(8000).any(|&b| b == 0),
        Err(_) => false,
    }
}

/// The commit being merged: MERGE_HEAD peeled (it can name a signed tag).
/// None if no merge is in progress; an error for an octopus merge.
pub fn merge_head_commit() -> Result<Option<String>> {
    let Ok(raw) = std::fs::read_to_string(git_path("MERGE_HEAD")) else {
        return Ok(None);
    };
    let heads: Vec<&str> = raw.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    match heads.as_slice() {
        [] => Ok(None),
        [one] => rev_parse(&format!("{}^{{commit}}", one))
            .map(Some)
            .with_context(|| format!("MERGE_HEAD {} does not name a commit", one)),
        _ => anyhow::bail!("an octopus merge ({} heads) is in progress; llminus handles two-parent merges only", heads.len()),
    }
}

/// git's CONFLICT messages for the merge in progress, by redoing it with
/// `git merge-tree` (which touches neither the index nor the working tree)
fn conflict_messages() -> Vec<String> {
    let Ok(Some(theirs)) = merge_head_commit() else {
        return Vec::new();
    };
    // merge-tree exits 1 when there are conflicts; we only want its messages
    let out = std::process::Command::new("git")
        .args(["merge-tree", "--write-tree", "HEAD", &theirs])
        .output();
    let Ok(out) = out else { return Vec::new() };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.starts_with("CONFLICT ("))
        .map(str::to_string)
        .collect()
}

/// All unmerged paths in the index, with their stages, binary-ness, marker
/// regions and git's conflict messages
pub fn current() -> Result<Vec<UnmergedPath>> {
    let raw = git_bytes(&["ls-files", "-u", "-z"])?;
    let mut paths = parse_ls_files_u(&raw);
    if paths.is_empty() {
        return Ok(paths);
    }

    let top = git(&["rev-parse", "--show-toplevel"])?.trim().to_string();
    let messages = conflict_messages();
    for p in &mut paths {
        p.binary = p.stages.iter().flatten().any(|s| blob_is_binary(&s.oid));
        if !p.binary
            && let Ok(content) = std::fs::read_to_string(std::path::Path::new(&top).join(&p.path))
        {
            p.regions = parse_conflict_markers(&p.path, &content);
        }
        p.messages = messages.iter().filter(|m| m.contains(p.path.as_str())).cloned().collect();
    }
    Ok(paths)
}

/// The marker regions of all current conflicts (for similarity search)
pub fn current_conflicts() -> Result<Vec<ConflictFile>> {
    Ok(current()?.into_iter().flat_map(|p| p.regions).collect())
}

/// Whether any path is still unmerged
pub fn any() -> bool {
    git_allow_fail(&["ls-files", "-u"]).is_some_and(|s| !s.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_parse_conflict_markers() {
        let dir = TempDir::new().unwrap();
        let conflict_file = dir.path().join("conflict.c");
        let content = r#"int main() {
<<<<<<< HEAD
    printf("ours");
=======
    printf("theirs");
>>>>>>> feature
    return 0;
}
"#;
        fs::write(&conflict_file, content).unwrap();

        let conflicts = parse_conflict_markers("conflict.c", &fs::read_to_string(&conflict_file).unwrap());
        assert_eq!(conflicts.len(), 1);
        assert!(conflicts[0].ours_content.contains("ours"));
        assert!(conflicts[0].theirs_content.contains("theirs"));
        assert!(conflicts[0].base_content.is_none());
    }

    #[test]
    fn test_parse_conflict_markers_diff3() {
        let dir = TempDir::new().unwrap();
        let conflict_file = dir.path().join("conflict.c");
        // diff3 style with base content
        let content = r#"int main() {
<<<<<<< HEAD
    printf("ours");
||||||| base
    printf("base");
=======
    printf("theirs");
>>>>>>> feature
    return 0;
}
"#;
        fs::write(&conflict_file, content).unwrap();

        let conflicts = parse_conflict_markers("conflict.c", &fs::read_to_string(&conflict_file).unwrap());
        assert_eq!(conflicts.len(), 1);
        assert!(conflicts[0].ours_content.contains("ours"));
        assert!(conflicts[0].theirs_content.contains("theirs"));
        assert!(conflicts[0].base_content.as_ref().unwrap().contains("base"));
    }

    #[test]
    fn test_parse_multiple_conflicts() {
        let dir = TempDir::new().unwrap();
        let conflict_file = dir.path().join("conflict.c");
        let content = r#"<<<<<<< HEAD
first ours
=======
first theirs
>>>>>>> feature
middle
<<<<<<< HEAD
second ours
=======
second theirs
>>>>>>> feature
"#;
        fs::write(&conflict_file, content).unwrap();

        let conflicts = parse_conflict_markers("conflict.c", &fs::read_to_string(&conflict_file).unwrap());
        assert_eq!(conflicts.len(), 2);
        assert!(conflicts[0].ours_content.contains("first ours"));
        assert!(conflicts[1].ours_content.contains("second ours"));
    }

}
