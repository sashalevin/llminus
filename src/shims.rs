//! Wrappers put first on the LLM's PATH.
//!
//! `git` is read-only: the LLM edits files, llminus does the staging and owns
//! the merge. `lei` only runs searches, and never saves one. Claude Code's
//! permission rules are enforced as well where they apply, but a wrapper also
//! covers backends that have no rules of their own.

use anyhow::{Context, Result};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// Read-only git: allows the subcommands an investigation needs, refuses
/// options that make a read-only command run a program or write a file, and
/// disarms hooks, fsmonitor, textconv, filter drivers and external diff
/// (after ~/git-ro-shim and stable-tools' ai.py).
const GIT_SHIM: &str = r#"#!/bin/bash
# llminus: read-only git for the LLM. Edit files; llminus stages the result.
G=__REAL__
sub=""; pre=(); rest=()
for a in "$@"; do
  if [ -z "$sub" ]; then
    case "$a" in
      -c|-c*|-C|-C*|--config-env*|--exec-path*|--git-dir*|--work-tree*|--namespace*|--super-prefix*|--attr-source*)
        echo "llminus: git $a is not allowed here" >&2; exit 128 ;;
      -*) pre+=("$a") ;;
      *) sub="$a" ;;
    esac
  else
    rest+=("$a")
  fi
  case "$a" in
    -O*|--open-files-in-pager*|--output|--output=*|--ext-diff|--textconv|--filters|--exec|--exec=*|--upload-pack*|--receive-pack*)
      echo "llminus: git option $a is not allowed here" >&2; exit 128 ;;
  esac
done
positionals() { local n=0 x; for x in "${rest[@]}"; do case "$x" in -*) ;; *) n=$((n+1)) ;; esac; done; echo $n; }
case "$sub" in
  ""|show|log|diff|rev-parse|rev-list|ls-tree|ls-files|cat-file|grep|blame|annotate|merge-base|\
  describe|name-rev|shortlog|whatchanged|for-each-ref|show-ref|status|range-diff|version|help|var) ;;
  branch|tag)
    for x in "${rest[@]}"; do
      case "$x" in -l|--list|-a|--all|-r|--remotes|-v|-vv|--show-current|--contains*|--merged*|--no-merged*|--points-at*|--sort=*|--format=*|--no-color|--color*) ;;
        *) echo "llminus: only listing with git $sub is allowed here" >&2; exit 128 ;;
      esac
    done ;;
  symbolic-ref)
    for x in "${rest[@]}"; do
      case "$x" in -d|--delete|-m) echo "llminus: git symbolic-ref $x is not allowed here" >&2; exit 128 ;; esac
    done
    [ "$(positionals)" -le 1 ] || { echo "llminus: only reading with git symbolic-ref is allowed here" >&2; exit 128; } ;;
  config)
    # reading only: `git config --get/--get-all/--get-regexp/--list ...` or `git config get|list ...`
    read_form=false
    case "${rest[0]:-}" in get|list) read_form=true ;; esac
    for x in "${rest[@]}"; do
      case "$x" in
        --get|--get-all|--get-regexp|--get-urlmatch|-l|--list) read_form=true ;;
        --add|--unset|--unset-all|--replace-all|--rename-section|--remove-section|-e|--edit|--file*|-f|--blob*|set|unset|edit|rename-section|remove-section)
          read_form=false; break ;;
      esac
    done
    $read_form || { echo "llminus: only reading with git config is allowed here" >&2; exit 128; } ;;
  *)
    echo "llminus: git $sub is not allowed here. Edit the files directly and list them in the verdict file; llminus stages, builds and commits." >&2
    exit 128 ;;
esac
safe=(-c core.fsmonitor=false -c core.hooksPath=/dev/null)
while IFS= read -r k; do
  case "$k" in
    alias.*)
      if [ "${k#alias.}" = "${sub,,}" ]; then
        echo "llminus: git alias $sub is not allowed here" >&2; exit 128
      fi ;;
    diff.*.textconv) safe+=(-c "$k=cat" -c "${k%%.textconv}.cachetextconv=false") ;;
    filter.*.process) safe+=(-c "$k=") ;;
    ?*) safe+=(-c "$k=cat") ;;
  esac
done < <("$G" -c core.fsmonitor=false -c core.hooksPath=/dev/null config --name-only --get-regexp \
         '^(diff\..*\.textconv|filter\..*\.(clean|smudge|process)|alias\..*)$' 2>/dev/null)
inject=()
case "$sub" in
  show|diff|log|whatchanged) safe+=(-c diff.external=); inject=(--no-ext-diff --no-textconv) ;;
  blame|annotate) inject=(--no-textconv) ;;
esac
[ -z "$sub" ] && exec "$G" "${safe[@]}" "${pre[@]}"
exec "$G" "${safe[@]}" "${pre[@]}" "$sub" "${inject[@]}" "${rest[@]}"
"#;

/// lei for searches only, never saving a search (a saved search is what
/// `lei q` creates whenever it writes to a folder), never running a program
/// (--alert, --mua), never changing lei's configuration.
const LEI_SHIM: &str = r#"#!/bin/bash
# llminus: lei for lore searches. Results always go to stdout; nothing is saved.
L=__REAL__
[ -n "$L" ] || { echo "llminus: lei is not installed" >&2; exit 127; }
# lei runs git itself (its caches, dates for rt:); give it the real one
export PATH=__ORIGPATH__
if [ "$*" = "ls-external" ]; then exec "$L" ls-external; fi
if [ "$1" != q ]; then
  echo "llminus: only 'lei q' is available here (e.g. lei q -f json --limit=20 's:\"manual merge of\" AND dfn:path/to/file.c')" >&2
  exit 2
fi
shift
fmt=json; args=()
while [ $# -gt 0 ]; do
  case "$1" in
    # forced anyway
    --no-save|--no-import-remote|-o-|--output=-) ;;
    -o|--output)
      [ "$2" = - ] || { echo "llminus: lei output can only go to stdout (-o -)" >&2; exit 2; }
      shift ;;
    -f|--format) fmt="$2"; shift ;;
    --format=*) fmt="${1#--format=}" ;;
    # searching options, with a value
    -I|--include|--exclude|-O|--only|-s|--sort|-d|--dedupe|--offset|-n|--limit)
      [ $# -ge 2 ] || { echo "llminus: lei $1 needs a value" >&2; exit 2; }
      args+=("$1" "$2"); shift ;;
    --include=*|--exclude=*|--only=*|--sort=*|--dedupe=*|--offset=*|--limit=*|-n[0-9]*) args+=("$1") ;;
    # searching options, without a value
    -t|--threads|-tt|-r|--reverse|-q|--quiet|-v|--verbose|--no-local|--no-external|--no-remote|-g|--globoff|--stdin|--pretty)
      args+=("$1") ;;
    -*) echo "llminus: lei option $1 is not allowed here" >&2; exit 2 ;;
    *) args+=("$1") ;;
  esac
  shift
done
case "$fmt" in
  json|mboxrd) ;;
  *) echo "llminus: lei format must be json or mboxrd (not $fmt)" >&2; exit 2 ;;
esac
exec "$L" q --no-save --no-import-remote -o - -f "$fmt" "${args[@]}"
"#;

/// Spawn `cmd`, retrying while the executable is "busy": a script written
/// just before can still be open for writing in a child another thread has
/// forked but not yet exec'd (ETXTBSY)
pub fn spawn_retrying(cmd: &mut std::process::Command) -> std::io::Result<std::process::Child> {
    let mut tries = 0;
    loop {
        match cmd.spawn() {
            Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy && tries < 50 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            r => return r,
        }
    }
}

/// The real binary behind `name`, resolved to its canonical path
pub fn which_real(name: &str) -> Option<PathBuf> {
    which(name).map(|p| p.canonicalize().unwrap_or(p))
}

/// Whether `p` is an executable file
pub fn is_executable(p: &Path) -> bool {
    p.is_file() && p.metadata().is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
}

/// Find `name` on PATH
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(name)).find(|p| is_executable(p))
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

fn write_shim(dir: &Path, name: &str, template: &str, real: Option<PathBuf>) -> Result<()> {
    let real = real
        .map(|p| shell_quote(&p.canonicalize().unwrap_or(p).to_string_lossy()))
        .unwrap_or_else(|| "''".to_string());
    let orig_path = shell_quote(&std::env::var("PATH").unwrap_or_default());
    let path = dir.join(name);
    std::fs::write(&path, template.replace("__REAL__", &real).replace("__ORIGPATH__", &orig_path))
        .with_context(|| format!("Failed to write {}", path.display()))?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

/// A temporary directory holding the git and lei wrappers; removed on drop
pub struct Shims {
    dir: tempfile::TempDir,
}

impl Shims {
    pub fn create() -> Result<Self> {
        let dir = tempfile::Builder::new().prefix("llminus-bin.").tempdir()?;
        write_shim(dir.path(), "git", GIT_SHIM, which("git"))?;
        write_shim(dir.path(), "lei", LEI_SHIM, which("lei"))?;
        Ok(Shims { dir })
    }

    /// PATH with the wrappers first
    pub fn path_env(&self) -> std::ffi::OsString {
        let mut parts = vec![self.dir.path().to_path_buf()];
        if let Some(p) = std::env::var_os("PATH") {
            parts.extend(std::env::split_paths(&p));
        }
        std::env::join_paths(parts).unwrap_or_default()
    }
}

/// The saved searches lei knows about (to warn if an LLM saved one anyway)
pub fn lei_saved_searches() -> Option<Vec<String>> {
    let lei = which("lei")?;
    let out = std::process::Command::new(lei).arg("ls-search").output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).lines().map(str::to_string).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn run(shims: &Shims, prog: &str, args: &[&str]) -> (i32, String, String) {
        let mut cmd = Command::new(shims.dir.path().join(prog));
        cmd.args(args).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
        let out = spawn_retrying(&mut cmd).unwrap().wait_with_output().unwrap();
        (out.status.code().unwrap_or(-1),
         String::from_utf8_lossy(&out.stdout).to_string(),
         String::from_utf8_lossy(&out.stderr).to_string())
    }

    /// Shims whose "real" lei is a stub printing its arguments
    fn shims_with_stub_lei() -> (Shims, tempfile::TempDir) {
        let stub_dir = tempfile::tempdir().unwrap();
        let stub = stub_dir.path().join("lei");
        std::fs::write(&stub, "#!/bin/bash\nprintf '%s\\n' \"$@\"\n").unwrap();
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        let dir = tempfile::Builder::new().prefix("llminus-bin.").tempdir().unwrap();
        write_shim(dir.path(), "git", GIT_SHIM, which("git")).unwrap();
        write_shim(dir.path(), "lei", LEI_SHIM, Some(stub)).unwrap();
        (Shims { dir }, stub_dir)
    }

    #[test]
    fn lei_q_is_forced_to_stdout_without_saving() {
        let (shims, _stub) = shims_with_stub_lei();
        let (rc, out, _) = run(&shims, "lei", &["q", "-f", "mboxrd", "s:foo"]);
        assert_eq!(rc, 0);
        let args: Vec<&str> = out.lines().collect();
        assert_eq!(args, ["q", "--no-save", "--no-import-remote", "-o", "-", "-f", "mboxrd", "s:foo"]);

        // an explicit --no-save is not passed twice, json is the default
        let (_, out, _) = run(&shims, "lei", &["q", "--no-save", "-o", "-", "b:abc"]);
        assert_eq!(out.lines().filter(|l| *l == "--no-save").count(), 1);
        assert!(out.lines().any(|l| l == "json"));
    }

    #[test]
    fn lei_refuses_saving_and_programs() {
        let (shims, _stub) = shims_with_stub_lei();
        for args in [
            vec!["q", "-o", "/tmp/somewhere", "s:foo"],
            vec!["q", "--output=/tmp/x", "s:foo"],
            vec!["q", "-o/tmp/x", "s:foo"],
            vec!["q", "-f", "text", "-o", "/tmp/x", "s:foo"],
            vec!["q", "--alert", "rm -rf /", "s:foo"],
            vec!["q", "--mua=mutt", "s:foo"],
            vec!["q", "-f", "maildir", "s:foo"],
            vec!["q", "--import-remote", "s:foo"],
            vec!["q", "-f", "jsonl", "s:foo"],
            vec!["q", "--lock=none", "s:foo"],
            vec!["up", "/home/x/Mail"],
            vec!["edit-search", "/home/x/Mail"],
            vec!["forget-search", "/home/x/Mail"],
            vec!["add-external", "https://example.com"],
        ] {
            let (rc, out, _) = run(&shims, "lei", &args);
            assert_ne!(rc, 0, "lei {:?} should be refused", args);
            assert!(out.is_empty(), "lei {:?} reached the real lei", args);
        }
    }

    #[test]
    fn lei_runs_with_the_original_path() {
        let stub_dir = tempfile::tempdir().unwrap();
        let stub = stub_dir.path().join("lei");
        std::fs::write(&stub, "#!/bin/bash\necho \"PATH=$PATH\"\n").unwrap();
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        let dir = tempfile::Builder::new().prefix("llminus-bin.").tempdir().unwrap();
        write_shim(dir.path(), "lei", LEI_SHIM, Some(stub)).unwrap();
        let shims = Shims { dir };
        let mut cmd = Command::new(shims.dir.path().join("lei"));
        cmd.args(["q", "x"]).env("PATH", shims.path_env()).stdout(std::process::Stdio::piped());
        let out = spawn_retrying(&mut cmd).unwrap().wait_with_output().unwrap();
        let out = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(out.starts_with("PATH="), "{}", out);
        assert!(!out.contains(&*shims.dir.path().to_string_lossy()), "lei saw the wrappers: {}", out);
    }

    #[test]
    fn git_shim_allows_what_tools_read() {
        let (shims, _stub) = shims_with_stub_lei();
        for args in [
            vec!["config", "--get", "core.bare"], vec!["config", "--list"], vec!["config", "get", "core.bare"],
            vec!["symbolic-ref", "-q", "HEAD"], vec!["branch", "--show-current"], vec!["var", "GIT_PAGER"],
        ] {
            let (rc, _, err) = run(&shims, "git", &args);
            assert!(!err.contains("llminus:"), "git {:?} refused: {}", args, err);
            let _ = rc;
        }
        for args in [
            vec!["config", "core.bare", "false"], vec!["config", "--unset", "x.y"], vec!["config", "set", "x.y", "1"],
            vec!["config", "--file", "/tmp/x", "--get", "a.b"],
            vec!["symbolic-ref", "HEAD", "refs/heads/x"], vec!["symbolic-ref", "-d", "HEAD"],
        ] {
            let (rc, _, _) = run(&shims, "git", &args);
            assert_ne!(rc, 0, "git {:?} should be refused", args);
        }
    }

    #[test]
    fn git_shim_is_read_only() {
        let (shims, _stub) = shims_with_stub_lei();
        for args in [
            vec!["commit", "-m", "x"], vec!["add", "f"], vec!["rm", "f"], vec!["reset", "--hard"],
            vec!["checkout", "HEAD~1"], vec!["stash"], vec!["merge", "--abort"],
            vec!["update-ref", "HEAD", "HEAD~1"], vec!["-c", "alias.x=!sh", "x"], vec!["-C", "/", "log"],
            vec!["branch", "newbranch"], vec!["tag", "v1"], vec!["log", "--output=/tmp/x"],
        ] {
            let (rc, _, _) = run(&shims, "git", &args);
            assert_ne!(rc, 0, "git {:?} should be refused", args);
        }
        let (rc, out, _) = run(&shims, "git", &["version"]);
        assert_eq!(rc, 0);
        assert!(out.starts_with("git version"));
    }
}
