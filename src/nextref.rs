//! How linux-next resolved the same conflicts.
//!
//! Linus checks his resolutions against linux-next's; in the v7.3 merge
//! window every one of his conflict resolutions that could be checked was
//! consistent with what linux-next already had. So for each conflicted path,
//! find the linux-next merge that resolved it by hand and show that
//! resolution to the LLM as a reference - and compare the result with it
//! afterwards.
//!
//! linux-next is a reference, not the truth: it has been wrong (hrtimer,
//! platform-drivers-x86, ...). Provenance matters too: only an upstream
//! linux-next ref is used (refs/remotes/linux-next/master, or what
//! LLMINUS_NEXT_REF names), pinned once per run, never Sasha's own
//! integration trees built with llminus.

use crate::gitutil::{git_allow_fail, rev_parse};

/// A linux-next merge that resolved a path by hand
#[derive(Debug, Clone)]
pub struct Reference {
    pub path: String,
    /// The linux-next tip it was found from
    pub next: String,
    pub merge: String,
    pub subject: String,
    /// `git show --remerge-diff` of the merge, for this path: what was done by hand
    pub remerge_diff: String,
    /// Its parents' versions of the path equal ours and theirs: it resolved
    /// exactly this conflict, so its result can be compared blob for blob
    pub same_conflict: bool,
    /// The path as linux-next's merge left it
    pub result_blob: Option<String>,
}

/// How the resolution compares with linux-next's
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Agreement {
    /// Identical result for the identical conflict
    Agree,
    /// Different result for the identical conflict
    Differ,
    /// linux-next resolved a different version of the conflict; its
    /// hand-written lines are (all) in our result
    WeakAgree,
    WeakDiffer,
}

impl Agreement {
    pub fn describe(&self) -> &'static str {
        match self {
            Agreement::Agree => "same as linux-next's resolution",
            Agreement::Differ => "DIFFERENT from linux-next's resolution of the same conflict",
            Agreement::WeakAgree => "consistent with linux-next's resolution (of a different version of the conflict)",
            Agreement::WeakDiffer => "not consistent with linux-next's resolution (of a different version of the conflict)",
        }
    }
}

/// The result header's summary of a run's agreements: the worst one, or None
/// when linux-next had nothing to compare with
pub fn summary(agreements: &[Agreement]) -> Option<&'static str> {
    let has = |a: Agreement| agreements.contains(&a);
    if has(Agreement::Differ) {
        Some("DIFFER")
    } else if has(Agreement::WeakDiffer) {
        Some("WEAK_DIFFER")
    } else if has(Agreement::WeakAgree) {
        Some("WEAK_AGREE")
    } else if has(Agreement::Agree) {
        Some("AGREE")
    } else {
        None
    }
}

/// The upstream linux-next tip to use, if there is one
pub fn next_tip() -> Option<String> {
    match std::env::var("LLMINUS_NEXT_REF") {
        Ok(r) if r.is_empty() || r == "none" => None,
        Ok(r) => rev_parse(&format!("{}^{{commit}}", r)),
        Err(_) => rev_parse("refs/remotes/linux-next/master^{commit}"),
    }
}

const MAX_DIFF: usize = 12 * 1024;

/// linux-next's hand resolutions of `paths`, given our and their sides
pub fn find(next: &str, ours: &str, theirs: &str, paths: &[String]) -> Vec<Reference> {
    // Only linux-next's own merges: those since it forked from our side
    let Some(base) = git_allow_fail(&["merge-base", next, ours]).map(|s| s.trim().to_string()) else {
        return Vec::new();
    };
    let range = format!("{}..{}", base, next);
    let mut refs = Vec::new();
    for path in paths {
        let Some(log) = git_allow_fail(&["log", "--first-parent", "--merges", "--format=%H%x00%s", "-n", "40",
                                         &range, "--", path]) else { continue };
        for line in log.lines() {
            let Some((merge, subject)) = line.split_once('\0') else { continue };
            let Some(mut diff) = git_allow_fail(&["show", "--remerge-diff", "--format=", merge, "--", path]) else {
                continue;
            };
            if diff.trim().is_empty() {
                continue; // merged cleanly there
            }
            if diff.len() > MAX_DIFF {
                diff.truncate(MAX_DIFF);
                diff.push_str("\n[... truncated]\n");
            }
            let blob = |rev: &str| rev_parse(&format!("{}:{}", rev, path));
            let same_conflict = blob(&format!("{}^1", merge)) == blob(ours) && blob(&format!("{}^2", merge)) == blob(theirs)
                || blob(&format!("{}^1", merge)) == blob(theirs) && blob(&format!("{}^2", merge)) == blob(ours);
            refs.push(Reference {
                path: path.clone(),
                next: next.to_string(),
                merge: merge.to_string(),
                subject: subject.to_string(),
                remerge_diff: diff,
                same_conflict,
                result_blob: blob(merge),
            });
            break; // the newest hand resolution of this path
        }
    }
    refs
}

/// The lines a remerge-diff added by hand (excluding trivial ones)
fn hand_written_lines(remerge_diff: &str) -> Vec<String> {
    remerge_diff.lines()
        .filter(|l| l.starts_with('+') && !l.starts_with("+++"))
        .map(|l| l[1..].trim().to_string())
        .filter(|l| l.len() >= 8)
        .collect()
}

/// Lines the resolution dropped from the conflict (one side's version)
fn dropped_lines(remerge_diff: &str) -> Vec<String> {
    remerge_diff.lines()
        .filter(|l| l.starts_with('-') && !l.starts_with("---"))
        .map(|l| l[1..].trim().to_string())
        .filter(|l| l.len() >= 8 && !["<<<<<<<", "=======", ">>>>>>>", "|||||||"].iter().any(|m| l.starts_with(m)))
        .collect()
}

/// Compare our result for the path (`result_blob`, None if deleted, and its
/// content) with linux-next's
pub fn compare(r: &Reference, result_blob: Option<&str>, result_content: &str) -> Agreement {
    if r.same_conflict {
        return if r.result_blob.as_deref() == result_blob { Agreement::Agree } else { Agreement::Differ };
    }
    let content: Vec<String> = result_content.lines().map(|l| l.trim().to_string()).collect();
    let has = |l: &String| content.iter().any(|c| c == l);
    let added = hand_written_lines(&r.remerge_diff);
    let ok = if !added.is_empty() {
        added.iter().all(has)
    } else {
        // a pick-one-side resolution: what linux-next dropped must be gone
        dropped_lines(&r.remerge_diff).iter().all(|l| !has(l))
    };
    if ok { Agreement::WeakAgree } else { Agreement::WeakDiffer }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gitutil::git;
    use crate::testutil::*;

    #[test]
    fn weak_comparison() {
        let r = Reference {
            path: "f.c".into(), next: "n".into(), merge: "m".into(), subject: "s".into(),
            remerge_diff: "--- a/f.c\n+++ b/f.c\n@@ -1,5 +1,1 @@\n-<<<<<<< 1234\n-\told_call(x);\n-=======\n-\tnew_call(x);\n->>>>>>> 5678\n+\tnew_call(x, flags);\n".into(),
            same_conflict: false, result_blob: None,
        };
        assert_eq!(compare(&r, Some("x"), "int f(void)\n{\n\tnew_call(x, flags);\n}\n"), Agreement::WeakAgree);
        assert_eq!(compare(&r, Some("x"), "int f(void)\n{\n\tnew_call(x);\n}\n"), Agreement::WeakDiffer);
    }

    #[test]
    fn finds_the_linux_next_resolution_of_the_same_conflict() {
        let dir = init_test_repo();
        let _cwd = CwdGuard::enter(dir.path());
        let base = numbered_lines(10);
        create_commit(&dir, "file.c", &base, "base");
        create_branch(&dir, "feature");
        create_commit(&dir, "file.c", &base.replace("line 5\n", "feature 5\n"), "feature");
        checkout(&dir, "master");
        create_branch(&dir, "other");
        create_commit(&dir, "file.c", &base.replace("line 5\n", "other 5\n"), "other tree");
        // "linux-next": master + other + feature, with a hand resolution
        create_branch(&dir, "next");
        merge(&dir, "feature", "Merge feature into next");
        create_commit(&dir, "file.c", &base.replace("line 5\n", "resolved 5\n"), "Merge feature into next");
        let next = git(&["rev-parse", "next"]).unwrap().trim().to_string();
        let ours = git(&["rev-parse", "other"]).unwrap().trim().to_string();
        let theirs = git(&["rev-parse", "feature"]).unwrap().trim().to_string();

        let refs = find(&next, &ours, &theirs, &["file.c".to_string()]);
        assert_eq!(refs.len(), 1, "{:?}", refs);
        assert!(refs[0].same_conflict);
        assert!(refs[0].remerge_diff.contains("+resolved 5"));
        let resolved = git(&["rev-parse", "next:file.c"]).unwrap().trim().to_string();
        assert_eq!(compare(&refs[0], Some(&resolved), ""), Agreement::Agree);
        assert_eq!(compare(&refs[0], Some("0000"), ""), Agreement::Differ);
        // nothing to find for a path linux-next never resolved
        assert!(find(&next, &ours, &theirs, &["other.c".to_string()]).is_empty());
    }

    #[test]
    fn summary_is_the_worst_agreement() {
        use Agreement::*;
        assert_eq!(summary(&[]), None);
        assert_eq!(summary(&[Agree, WeakAgree]), Some("WEAK_AGREE"));
        assert_eq!(summary(&[Agree, Differ, WeakDiffer]), Some("DIFFER"));
        assert_eq!(summary(&[Agree, WeakDiffer]), Some("WEAK_DIFFER"));
        assert_eq!(summary(&[Agree]), Some("AGREE"));
    }
}
