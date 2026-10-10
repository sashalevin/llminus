//! lore.kernel.org access through lei.
//!
//! lei queries the externals configured with `lei add-external` (usually
//! https://lore.kernel.org/all/) through lore's search API, which works where
//! plain HTTP fetches get bot-blocked. Searches are never saved: output always
//! goes to stdout with --no-save (a saved search is what `lei q` creates when
//! it writes to a folder). Quirks, as found by stable-tools' lei.py:
//!  - --no-save doesn't work with -f text, so messages come as mboxrd;
//!  - a message-id that doesn't exist gives exit 0 and no output;
//!  - "m:a OR m:b" silently matches nothing, so fetch one message at a time.

use anyhow::{bail, Context, Result};
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::shims::which;

const LEI_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_OUTPUT: usize = 4 * 1024 * 1024;

/// A search hit
#[derive(Debug, Clone)]
pub struct Hit {
    pub message_id: String,
    pub subject: String,
    pub from: String,
    pub date: String,
}

fn run_lei(args: &[&str]) -> Result<String> {
    let lei = which("lei").context("lei is not installed")?;
    let mut child = Command::new(lei)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to run lei")?;
    let mut out = child.stdout.take().expect("piped");
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = out.by_ref().take(MAX_OUTPUT as u64).read_to_end(&mut buf);
        buf
    });
    let start = Instant::now();
    let status = loop {
        if let Some(st) = child.try_wait()? {
            break st;
        }
        if start.elapsed() > LEI_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            bail!("lei {} timed out", args.join(" "));
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    let stdout = reader.join().unwrap_or_default();
    if !status.success() {
        let mut err = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        bail!("lei {} failed: {}", args.join(" "), err.trim());
    }
    Ok(String::from_utf8_lossy(&stdout).to_string())
}

/// The first message of an mboxrd stream, unescaped
pub fn first_message_from_mboxrd(mbox: &str) -> Option<String> {
    let mut lines = mbox.lines();
    let first = lines.next()?;
    if !first.starts_with("From ") {
        return None;
    }
    let mut msg = String::new();
    for l in lines {
        if l.starts_with("From ") {
            break;
        }
        // mboxrd: one '>' was added in front of every /^>*From / line
        let l = if l.starts_with('>') && l.trim_start_matches('>').starts_with("From ") { &l[1..] } else { l };
        msg.push_str(l);
        msg.push('\n');
    }
    Some(msg)
}

fn norm_msgid(mid: &str) -> &str {
    mid.trim().trim_start_matches('<').trim_end_matches('>')
}

/// The raw message (headers and body) with message-id `mid`, from lei; from
/// lore over HTTP only if lei isn't installed
pub fn fetch_message(mid: &str) -> Result<String> {
    let mid = norm_msgid(mid);
    if which("lei").is_none() {
        return fetch_message_http(mid);
    }
    let query = format!("mid:{}", mid);
    let out = run_lei(&["q", "--no-save", "--no-import-remote", "-o", "-", "-f", "mboxrd", &query])?;
    first_message_from_mboxrd(&out).with_context(|| format!("lei found no message <{}>", mid))
}

fn fetch_message_http(mid: &str) -> Result<String> {
    let url = format!("https://lore.kernel.org/all/{}/raw", mid);
    eprintln!("lei is not installed, fetching {}", url);
    let response = ureq::get(&url).call().with_context(|| format!("Failed to fetch {}", url))?;
    response.into_body().read_to_string().context("Failed to read response body")
}

fn json_str(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        // addresses come as [[name, email], ...]
        serde_json::Value::Array(a) => a.iter().map(|x| match x {
            serde_json::Value::Array(pair) => pair.iter().filter_map(|p| p.as_str()).collect::<Vec<_>>().join(" "),
            other => json_str(other),
        }).collect::<Vec<_>>().join(", "),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Parse `lei q -f json` output
pub fn parse_hits(out: &str) -> Vec<Hit> {
    let values: Vec<serde_json::Value> = match serde_json::from_str::<serde_json::Value>(out.trim()) {
        Ok(serde_json::Value::Array(a)) => a,
        _ => out.lines()
            .map(|l| l.trim().trim_end_matches(','))
            .filter(|l| l.starts_with('{'))
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect(),
    };
    values.iter().filter(|v| v.is_object()).map(|v| Hit {
        message_id: json_str(&v["m"]),
        subject: json_str(&v["s"]),
        from: json_str(&v["f"]),
        date: json_str(&v["dt"]),
    }).filter(|h| !h.message_id.is_empty()).collect()
}

/// A fetched message
#[derive(Debug, Clone)]
pub struct Message {
    pub message_id: String,
    pub subject: String,
    pub from: String,
    pub date: String,
    pub body: String,
}

/// The body of a raw message (after the headers), at most `max` bytes
fn body_of(raw: &str, max: usize) -> String {
    let body = raw.split_once("\n\n").map(|x| x.1).unwrap_or(raw);
    let mut b: String = body.chars().take(max).collect();
    if b.len() < body.len() {
        b.push_str("\n[... truncated]");
    }
    b
}

/// linux-next's "manual merge of the X tree with the Y tree" reports that
/// touch `paths`: at most `per_path` of the newest per path
pub fn manual_merge_reports(paths: &[String], per_path: usize) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::new();
    for path in paths.iter().take(4) {
        let query = format!("s:\"manual merge of\" AND dfn:{} AND rt:6.months.ago..", path);
        let Ok(mut hits) = search(&query, 20) else { continue };
        hits.sort_by(|a, b| b.date.cmp(&a.date));
        for h in hits.into_iter().filter(|h| !h.subject.starts_with("Re:")).take(per_path) {
            if out.iter().any(|m| m.message_id == h.message_id) {
                continue;
            }
            if let Ok(raw) = fetch_message(&h.message_id) {
                out.push(Message { message_id: h.message_id, subject: h.subject, from: h.from, date: h.date,
                                   body: body_of(&raw, 6 * 1024) });
            }
        }
    }
    out
}

/// Search lore; at most `limit` hits
pub fn search(query: &str, limit: usize) -> Result<Vec<Hit>> {
    let limit = format!("--limit={}", limit);
    let out = run_lei(&["q", "--no-save", "--no-import-remote", "-o", "-", "-f", "json", &limit, query])?;
    Ok(parse_hits(&out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mboxrd_unescape() {
        let mbox = "From mboxrd@z Thu Jan  1 00:00:00 1970\nSubject: x\n\n>From the start\n>>From quoted\nbody\nFrom mboxrd@z Thu Jan  1 00:00:00 1970\nSubject: second\n";
        assert_eq!(first_message_from_mboxrd(mbox).unwrap(), "Subject: x\n\nFrom the start\n>From quoted\nbody\n");
        assert!(first_message_from_mboxrd("").is_none());
    }

    #[test]
    fn hits_from_json() {
        let out = r#"[{"m":"abc@kernel.org","s":"[GIT PULL] x","f":[["A B","a@b"]],"dt":"2026-08-24T13:57:51Z"},
null]"#;
        let hits = parse_hits(out);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].message_id, "abc@kernel.org");
        assert_eq!(hits[0].from, "A B a@b");
    }
}
