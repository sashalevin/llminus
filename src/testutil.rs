//! Helpers for tests that need a scratch git repository.

#![allow(dead_code)]

use std::path::Path;
use std::process::Command;
use std::fs;
use tempfile::TempDir;

use crate::gitutil::git;
use crate::unmerged;

/// The working directory is process-wide and tests run in parallel, so
/// tests that chdir into a temporary repo must hold this lock.
pub static CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Changes into a directory and restores the original one on drop
pub struct CwdGuard {
    pub original: std::path::PathBuf,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl CwdGuard {
    pub fn enter(dir: &Path) -> Self {
        let lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir).unwrap();
        CwdGuard { original, _lock: lock }
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.original);
    }
}

pub fn init_test_repo() -> TempDir {
    let dir = TempDir::new().unwrap();
    Command::new("git")
        .args(["init", "--initial-branch=master"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    dir
}

pub fn create_commit(dir: &TempDir, filename: &str, content: &str, msg: &str) {
    fs::write(dir.path().join(filename), content).unwrap();
    Command::new("git")
        .args(["add", filename])
        .current_dir(dir.path())
        .output()
        .unwrap();
    Command::new("git")
        .args(["commit", "-m", msg])
        .current_dir(dir.path())
        .output()
        .unwrap();
}

pub fn create_branch(dir: &TempDir, name: &str) {
    Command::new("git")
        .args(["checkout", "-b", name])
        .current_dir(dir.path())
        .output()
        .unwrap();
}

pub fn checkout(dir: &TempDir, name: &str) {
    Command::new("git")
        .args(["checkout", name])
        .current_dir(dir.path())
        .output()
        .unwrap();
}

pub fn merge(dir: &TempDir, branch: &str, msg: &str) {
    Command::new("git")
        .args(["merge", "--no-ff", "-m", msg, branch])
        .current_dir(dir.path())
        .output()
        .unwrap();
}

pub fn numbered_lines(n: usize) -> String {
    (1..=n).map(|i| format!("line {}\n", i)).collect()
}

/// Set up a merge of an annotated tag into master that conflicts in
/// file.c. Like a pulled signed tag, MERGE_HEAD then names the tag.
pub fn start_conflicting_merge(dir: &TempDir) {
    let base = numbered_lines(10);
    create_commit(dir, "file.c", &base, "base");
    create_branch(dir, "feature");
    create_commit(dir, "file.c", &base.replace("line 5\n", "feature 5\n"), "feature");
    git(&["tag", "-a", "-m", "feature tag", "feature-tag"]).unwrap();
    checkout(dir, "master");
    create_commit(dir, "file.c", &base.replace("line 5\n", "master 5\n"), "master");
    merge(dir, "feature-tag", "Merge tag 'feature-tag'");
    assert!(unmerged::any());
}

