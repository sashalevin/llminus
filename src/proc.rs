//! Running a child (the LLM, the build) so that nothing of it outlives the
//! call:
//!  - in its own process group, killed once the leader has exited but before
//!    it is reaped (so the group ID can't have been reused);
//!  - with LLMINUS_CALL_ID in every descendant's environment: whatever still
//!    carries it afterwards is killed through a pidfd taken before its
//!    environment is read (so a reused PID is never signalled). The ID is
//!    appended to an inherited one ("<outer>:<ours>"), so a call's cleanup
//!    also reaches what a nested llminus started;
//!  - with output read to the end but kept only up to a bound, and pipes not
//!    waited for forever (a leftover holding them open can't hang llminus).

use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// At most this much of a child's stdout/stderr is kept
const MAX_CAPTURE: usize = 64 * 1024 * 1024;

/// What a finished child produced
pub struct Finished {
    /// Exit status (None if killed by a signal)
    pub status: Option<i32>,
    pub timed_out: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

pub struct Io<'a> {
    pub stdin: Option<&'a [u8]>,
    /// Capture stdout (else it goes to our stderr)
    pub capture_stdout: bool,
    /// Capture stderr (else it goes to our stderr)
    pub capture_stderr: bool,
}

/// What to run
pub struct Spec<'a> {
    pub argv: &'a [String],
    pub cwd: &'a Path,
    pub env: Vec<(String, std::ffi::OsString)>,
    pub env_remove: Vec<&'a str>,
}

fn new_call_id() -> String {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{}.{}", std::process::id(), nanos)
}

/// The LLMINUS_CALL_ID a child gets: ours, after any we inherited
fn call_marker(id: &str) -> String {
    match std::env::var("LLMINUS_CALL_ID") {
        Ok(outer) if !outer.is_empty() => format!("{}:{}", outer, id),
        _ => id.to_string(),
    }
}

/// Whether a process environment (NUL-separated) carries call `id`
fn carries(environ: &[u8], id: &[u8]) -> bool {
    environ.split(|&b| b == 0)
        .filter_map(|v| v.strip_prefix(b"LLMINUS_CALL_ID=".as_slice()))
        .any(|chain| chain.split(|&b| b == b':').any(|c| c == id))
}

/// Kill every process whose environment carries this call's id. Each one is
/// pinned with a pidfd before its environment is read, and signalled through
/// it: if the PID was reused in between, the pidfd still names the old
/// process (and the signal fails) or the environment is the new process's.
fn reap_call(id: &str) {
    for _ in 0..10 {
        let mut found = false;
        let Ok(dir) = std::fs::read_dir("/proc") else { return };
        for e in dir.flatten() {
            let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<i32>().ok()) else { continue };
            let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
            if fd < 0 {
                continue;
            }
            let ours = std::fs::read(format!("/proc/{}/environ", pid)).is_ok_and(|env| carries(&env, id.as_bytes()));
            if ours {
                unsafe {
                    libc::syscall(libc::SYS_pidfd_send_signal, fd, libc::SIGKILL, std::ptr::null::<libc::siginfo_t>(), 0);
                }
                found = true;
            }
            unsafe { libc::close(fd); }
        }
        if !found {
            return;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Exited, but not reaped yet (so its process group ID is still ours)
fn exited_unreaped(pid: i32) -> bool {
    unsafe {
        let mut info: libc::siginfo_t = std::mem::zeroed();
        let r = libc::waitid(libc::P_PID, pid as libc::id_t, &mut info,
                             libc::WEXITED | libc::WNOHANG | libc::WNOWAIT);
        r == 0 && info.si_pid() != 0
    }
}

/// Read a pipe to its end, keeping at most MAX_CAPTURE bytes (the rest is
/// drained, so the writer never blocks on a full pipe)
fn read_bounded<R: Read + Send + 'static>(pipe: Option<R>) -> Option<mpsc::Receiver<Vec<u8>>> {
    pipe.map(|mut p| {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut kept = Vec::new();
            let mut chunk = [0u8; 64 * 1024];
            loop {
                match p.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let room = MAX_CAPTURE.saturating_sub(kept.len());
                        kept.extend_from_slice(&chunk[..n.min(room)]);
                    }
                }
            }
            let _ = tx.send(kept);
        });
        rx
    })
}

fn stderr_fd() -> Result<Stdio> {
    use std::os::fd::AsFd;
    Ok(Stdio::from(std::io::stderr().as_fd().try_clone_to_owned()?))
}

/// Run `spec` to completion or `timeout`
pub fn run(spec: &Spec, io: Io, timeout: Duration) -> Result<Finished> {
    let call_id = new_call_id();
    let mut cmd = Command::new(&spec.argv[0]);
    cmd.args(&spec.argv[1..])
        .current_dir(spec.cwd)
        .envs(spec.env.iter().map(|(k, v)| (k, v)))
        .env("LLMINUS_CALL_ID", call_marker(&call_id))
        .stdin(if io.stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(if io.capture_stdout { Stdio::piped() } else { stderr_fd()? })
        .stderr(if io.capture_stderr { Stdio::piped() } else { stderr_fd()? })
        .process_group(0);
    for k in &spec.env_remove {
        cmd.env_remove(k);
    }
    let mut child = crate::shims::spawn_retrying(&mut cmd)
        .with_context(|| format!("Failed to start {}", spec.argv[0]))?;
    let pid = child.id() as i32;

    if let Some(data) = io.stdin {
        let mut stdin = child.stdin.take().expect("piped stdin");
        let data = data.to_vec();
        // The child may exit without reading everything; that's its business
        std::thread::spawn(move || {
            let _ = stdin.write_all(&data);
        });
    }
    let out_rx = read_bounded(child.stdout.take());
    let err_rx = read_bounded(child.stderr.take());

    let start = Instant::now();
    let mut timed_out = false;
    while !exited_unreaped(pid) {
        if start.elapsed() > timeout {
            timed_out = true;
            unsafe { libc::kill(-pid, libc::SIGTERM); }
            std::thread::sleep(Duration::from_secs(5));
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    unsafe { libc::kill(-pid, libc::SIGKILL); }
    let status = child.wait()?;
    reap_call(&call_id);

    let collect = |rx: Option<mpsc::Receiver<Vec<u8>>>| {
        rx.and_then(|rx| rx.recv_timeout(Duration::from_secs(10)).ok()).unwrap_or_default()
    };
    Ok(Finished { status: status.code(), timed_out, stdout: collect(out_rx), stderr: collect(err_rx) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str) -> Vec<String> {
        vec!["sh".into(), "-c".into(), script.into()]
    }

    fn run_sh(script: &str, timeout: Duration) -> Finished {
        let argv = sh(script);
        let spec = Spec { argv: &argv, cwd: Path::new("/"), env: vec![], env_remove: vec![] };
        run(&spec, Io { stdin: None, capture_stdout: true, capture_stderr: true }, timeout).unwrap()
    }

    #[test]
    fn escaped_descendants_are_killed_and_pipes_dont_hang() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("survived");
        // a daemonized helper: new session, keeps stdout open, would write later
        let start = Instant::now();
        let f = run_sh(&format!("setsid sh -c 'sleep 3; touch {}' & echo done", marker.display()),
                       Duration::from_secs(30));
        assert_eq!(f.status, Some(0));
        assert!(String::from_utf8_lossy(&f.stdout).contains("done"));
        assert!(start.elapsed() < Duration::from_secs(20));
        std::thread::sleep(Duration::from_secs(4));
        assert!(!marker.exists(), "an escaped descendant outlived the call");
    }

    #[test]
    fn what_a_nested_call_started_is_killed_too() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("survived");
        // a nested llminus gives its own children "<ours>:<its id>"
        let f = run_sh(&format!("LLMINUS_CALL_ID=\"$LLMINUS_CALL_ID:999.1\" setsid sh -c 'sleep 3; touch {}' \
                                 </dev/null >/dev/null 2>&1 & echo done", marker.display()),
                       Duration::from_secs(30));
        assert_eq!(f.status, Some(0));
        std::thread::sleep(Duration::from_secs(4));
        assert!(!marker.exists(), "a nested call's descendant outlived the outer call");
    }

    #[test]
    fn call_markers_match_whole_components() {
        assert!(carries(b"A=1\0LLMINUS_CALL_ID=1.2:3.4\0", b"1.2"));
        assert!(carries(b"LLMINUS_CALL_ID=1.2:3.4", b"3.4"));
        assert!(!carries(b"LLMINUS_CALL_ID=1.2:3.4", b"3"));
        assert!(!carries(b"LLMINUS_CALL_ID=11.2", b"1.2"));
        assert!(!carries(b"X_LLMINUS_CALL_ID=1.2", b"1.2"));
    }

    #[test]
    fn output_is_bounded() {
        let f = run_sh("head -c 70000000 /dev/zero; echo end >&2", Duration::from_secs(60));
        assert_eq!(f.stdout.len(), MAX_CAPTURE);
        assert!(String::from_utf8_lossy(&f.stderr).contains("end"));
    }
}
