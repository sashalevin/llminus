//! What is being merged into what.

use crate::gitutil::{git_allow_fail, git_path};

/// Context about the current merge operation
#[derive(Debug, Default)]
pub struct MergeContext {
    /// The branch/tag/ref being merged (from MERGE_HEAD or MERGE_MSG)
    pub merge_source: Option<String>,
    /// The target branch (HEAD)
    pub head_branch: Option<String>,
    /// The merge message (from MERGE_MSG in the git directory)
    pub merge_message: Option<String>,
}

/// Extract context about the current merge operation
pub fn get_merge_context() -> MergeContext {
    let mut ctx = MergeContext {
        // Get current branch name
        head_branch: git_allow_fail(&["rev-parse", "--abbrev-ref", "HEAD"])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s != "HEAD"),
        ..Default::default()
    };

    // Try to read MERGE_MSG for merge context
    if let Ok(merge_msg) = std::fs::read_to_string(git_path("MERGE_MSG")) {
        ctx.merge_message = Some(merge_msg.clone());

        // Parse merge source from MERGE_MSG
        // Common formats:
        // "Merge branch 'feature-branch'"
        // "Merge tag 'v6.1'"
        // "Merge remote-tracking branch 'origin/main'"
        // "Merge commit 'abc123'"
        let first_line = merge_msg.lines().next().unwrap_or("");
        if let Some(source) = parse_merge_source(first_line) {
            ctx.merge_source = Some(source);
        }
    }

    // If no merge source found from MERGE_MSG, try to describe MERGE_HEAD
    if ctx.merge_source.is_none() {
        // Try to get a tag name for MERGE_HEAD
        if let Some(tag) = git_allow_fail(&["describe", "--tags", "--exact-match", "MERGE_HEAD"]) {
            ctx.merge_source = Some(tag.trim().to_string());
        } else if let Some(branch) = git_allow_fail(&["name-rev", "--name-only", "MERGE_HEAD"]) {
            let branch = branch.trim();
            if !branch.is_empty() && branch != "undefined" {
                ctx.merge_source = Some(branch.to_string());
            }
        }
    }

    ctx
}

/// Parse merge source from a merge message first line
pub fn parse_merge_source(line: &str) -> Option<String> {
    // "Merge branch 'feature'" -> "feature"
    // "Merge tag 'v6.1'" -> "v6.1"
    // "Merge remote-tracking branch 'origin/main'" -> "origin/main"
    // "Merge commit 'abc123'" -> "abc123"

    let line = line.trim();

    // Look for quoted source
    if let Some(start) = line.find('\'')
        && let Some(end) = line[start + 1..].find('\'')
    {
        return Some(line[start + 1..start + 1 + end].to_string());
    }

    // Look for "Merge X into Y" pattern without quotes
    if let Some(rest) = line.strip_prefix("Merge ") {
        // Skip "branch ", "tag ", "commit ", "remote-tracking branch "
        let rest = rest
            .strip_prefix("remote-tracking branch ")
            .or_else(|| rest.strip_prefix("branch "))
            .or_else(|| rest.strip_prefix("tag "))
            .or_else(|| rest.strip_prefix("commit "))
            .unwrap_or(rest);

        // Take until " into " or end of line
        if let Some(into_pos) = rest.find(" into ") {
            return Some(rest[..into_pos].trim().to_string());
        }
        let word = rest.split_whitespace().next()?;
        if !word.is_empty() {
            return Some(word.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_merge_source() {
        // Standard branch merge
        assert_eq!(
            parse_merge_source("Merge branch 'feature-branch'"),
            Some("feature-branch".to_string())
        );

        // Tag merge
        assert_eq!(
            parse_merge_source("Merge tag 'v6.1'"),
            Some("v6.1".to_string())
        );

        // Remote tracking branch
        assert_eq!(
            parse_merge_source("Merge remote-tracking branch 'origin/main'"),
            Some("origin/main".to_string())
        );

        // Commit merge
        assert_eq!(
            parse_merge_source("Merge commit 'abc123def'"),
            Some("abc123def".to_string())
        );

        // Branch with "into" target
        assert_eq!(
            parse_merge_source("Merge branch 'feature' into master"),
            Some("feature".to_string())
        );

        // Non-merge line
        assert_eq!(parse_merge_source("Fix bug in foo"), None);
    }

}
