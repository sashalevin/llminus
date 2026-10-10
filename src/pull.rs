//! `llminus pull`: merge a pull request sent to lore.kernel.org, resolving
//! any conflicts with the LLM.

use anyhow::{bail, Context, Result};
use std::process::Command;

use crate::gitutil::check_repo;
use crate::llm;
use crate::outcome::Exit;
use crate::resolve::{self, Opts};

/// Information parsed from a lore.kernel.org pull request email
#[derive(Debug, Default)]
pub struct PullRequest {
    /// Message ID
    pub message_id: String,
    /// Subject line of the email
    pub subject: String,
    /// Author name and email
    pub from: String,
    /// Date of the email
    pub date: String,
    /// Git repository URL to pull from
    pub git_url: String,
    /// Git ref (tag or branch) to pull
    pub git_ref: String,
    /// The full raw email body (LLM extracts summary and conflict instructions from this)
    pub body: String,
}

/// Parse email headers from raw email text
pub fn parse_email_headers(raw: &str) -> (String, String, String, String, &str) {
    let mut from = String::new();
    let mut subject = String::new();
    let mut date = String::new();
    let mut message_id = String::new();

    // Find the blank line separating headers from body
    let (headers_section, body) = raw.split_once("\n\n")
        .unwrap_or((raw, ""));

    // Parse headers (handle multi-line headers)
    let mut current_header = String::new();
    let mut current_value = String::new();

    for line in headers_section.lines() {
        if line.starts_with(' ') || line.starts_with('\t') {
            // Continuation of previous header
            current_value.push(' ');
            current_value.push_str(line.trim());
        } else if let Some((name, value)) = line.split_once(':') {
            // New header - save previous if any
            if !current_header.is_empty() {
                match current_header.to_lowercase().as_str() {
                    "from" => from = current_value.clone(),
                    "subject" => subject = current_value.clone(),
                    "date" => date = current_value.clone(),
                    "message-id" => message_id = current_value.clone(),
                    _ => {}
                }
            }
            current_header = name.to_string();
            current_value = value.trim().to_string();
        }
    }

    // Don't forget last header
    if !current_header.is_empty() {
        match current_header.to_lowercase().as_str() {
            "from" => from = current_value,
            "subject" => subject = current_value,
            "date" => date = current_value,
            "message-id" => message_id = current_value,
            _ => {}
        }
    }

    (from, subject, date, message_id, body)
}

/// Extract git pull URL and ref from email body
pub fn extract_git_info(body: &str) -> Option<(String, String)> {
    // Look for patterns like:
    // "git://git.kernel.org/pub/scm/linux/kernel/git/riscv/linux tags/riscv-for-linus-6.19-mw2"
    // "https://git.kernel.org/pub/scm/linux/kernel/git/foo/bar.git branch-name"

    for line in body.lines() {
        let line = line.trim();

        // Skip empty lines and common non-URL prefixes
        if line.is_empty() {
            continue;
        }

        // Check for git:// or https:// URLs
        let url_start = if let Some(pos) = line.find("git://") {
            pos
        } else if let Some(pos) = line.find("https://git.") {
            pos
        } else {
            continue;
        };

        let url_part = &line[url_start..];

        // Split into URL and ref
        let parts: Vec<&str> = url_part.split_whitespace().collect();
        if parts.len() >= 2 {
            let url = parts[0].to_string();
            let git_ref = parts[1].to_string();

            // Validate it looks like a kernel git URL
            if url.contains("kernel.org") || url.contains("git.") {
                return Some((url, git_ref));
            }
        }
    }

    None
}

/// Parse a pull request email from lore.kernel.org
pub fn parse_pull_request(message_id: &str, raw: &str) -> Result<PullRequest> {
    let (from, subject, date, parsed_id, body) = parse_email_headers(raw);

    let (git_url, git_ref) = extract_git_info(body)
        .ok_or_else(|| anyhow::anyhow!("Could not find git repository URL in email"))?;

    Ok(PullRequest {
        message_id: if parsed_id.is_empty() { message_id.to_string() } else { parsed_id },
        subject,
        from,
        date,
        git_url,
        git_ref,
        body: body.to_string(),
    })
}

/// A pull request mail for `resolve --pr-mail`: from a file, or fetched by
/// message-id. The git URL may be missing (the merge is already under way).
pub fn load_mail(spec: &str) -> Result<PullRequest> {
    let (raw, id) = if std::path::Path::new(spec).is_file() {
        (std::fs::read_to_string(spec).with_context(|| format!("Failed to read {}", spec))?, String::new())
    } else {
        (crate::lore::fetch_message(spec)?, spec.to_string())
    };
    // a raw mbox's "From " line is not a header
    let raw = raw.strip_prefix("From ").map(|r| r.split_once('\n').map(|x| x.1).unwrap_or("")).unwrap_or(&raw);
    let (from, subject, date, parsed_id, body) = parse_email_headers(raw);
    let (git_url, git_ref) = extract_git_info(body).unwrap_or_default();
    Ok(PullRequest {
        message_id: if parsed_id.is_empty() { id } else { parsed_id },
        subject, from, date, git_url, git_ref,
        body: body.to_string(),
    })
}

/// Execute git pull and return whether there are conflicts
pub fn git_pull(url: &str, git_ref: &str) -> Result<bool> {
    eprintln!("Executing: git pull --no-rebase {} {}", url, git_ref);

    // Always merge: without pull.rebase configured git refuses to pull into
    // a diverged branch, and with pull.rebase=true it would rebase instead.
    // No editor runs here, so strip comments explicitly as an interactive
    // pull would; otherwise a signed tag's signature lands in the message.
    let output = Command::new("git")
        .args(["pull", "--no-rebase", "--cleanup=strip", url, git_ref])
        .output()
        .context("Failed to run git pull")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stdout.is_empty() {
        eprintln!("{}", stdout);
    }
    if !stderr.is_empty() {
        eprintln!("{}", stderr);
    }

    // Check if there are conflicts
    if output.status.success() {
        return Ok(false); // No conflicts
    }

    // Check for merge conflicts specifically
    let conflict_markers = ["CONFLICT", "Automatic merge failed", "fix conflicts"];
    let output_text = format!("{}{}", stdout, stderr);

    for marker in conflict_markers {
        if output_text.contains(marker) {
            return Ok(true); // Has conflicts
        }
    }

    // Some other error
    bail!("git pull failed: {}", stderr);
}

/// Build a merge commit message using the pull request information, summary, and resolution
pub fn build_merge_commit_message(pull_req: &PullRequest, summary: &str, resolution: &str) -> String {
    let mut msg = String::new();

    // Use the subject line as the merge message header
    if !pull_req.subject.is_empty() {
        // Clean up subject - remove [GIT PULL] prefix if present
        let subject = pull_req.subject
            .replace("[GIT PULL]", "")
            .replace("[git pull]", "")
            .trim()
            .to_string();
        msg.push_str(&format!("Merge {} {}\n", pull_req.git_ref, subject));
    } else {
        msg.push_str(&format!("Merge {}\n", pull_req.git_ref));
    }
    msg.push('\n');

    // Add maintainer's summary (extracted by LLM)
    if !summary.is_empty() {
        msg.push_str(summary);
        msg.push_str("\n\n");
    }

    // Add the resolution notes (the LLM's, plus llminus's facts)
    if !resolution.is_empty() {
        msg.push_str("Merge conflict resolution:\n\n");
        msg.push_str(resolution.trim_end());
        msg.push_str("\n\n");
    }

    // Link to lore, and the disclosure trailer
    msg.push_str(&format!("Link: https://lore.kernel.org/all/{}/\n",
        pull_req.message_id.trim_start_matches('<').trim_end_matches('>')));
    if !resolution.is_empty() {
        msg.push_str("Assisted-by: LLM\n");
    }

    msg
}

/// Ask the LLM for the maintainer's summary from the email body. None if
/// that fails (the commit then just has no summary).
fn extract_summary(body: &str, opts: &Opts) -> Option<String> {
    let prompt = format!(r#"Extract ONLY the technical summary from this kernel pull request email.
The summary describes what changes are included (usually as bullet points).
Do NOT include:
- Personal messages to Linus
- Git URLs or repository information
- Merge/conflict resolution instructions
- Diffstat or file change listings
- Sign-offs or signatures

Output ONLY the summary text, nothing else. No preamble, no explanation. Don't
use any tools.

Email body:
{}
"#, body);
    eprintln!("Extracting summary from pull request...");
    let cwd = std::env::current_dir().ok()?;
    let run = llm::run(&opts.backend, &prompt, &llm::RunOpts {
        cwd: &cwd,
        timeout: std::time::Duration::from_secs(600),
        rate_limit_retries: opts.rate_limit_retries,
        rate_limit_wait: opts.rate_limit_wait,
        env: vec![],
        clear_before_attempt: vec![],
    }).ok()?;
    let summary = run.final_message.trim().to_string();
    (run.status == Some(0) && !summary.is_empty()).then_some(summary)
}

/// Pull a kernel pull request from lore.kernel.org
pub fn pull(message_id: &str, opts: &Opts) -> Result<Exit> {
    check_repo()?;

    // Step 1: Fetch and parse the pull request email
    eprintln!("=== Fetching Pull Request ===\n");
    let raw_email = crate::lore::fetch_message(message_id)?;
    let pull_req = parse_pull_request(message_id, &raw_email)?;

    eprintln!("Subject: {}", pull_req.subject);
    eprintln!("From: {}", pull_req.from);
    eprintln!("Date: {}", pull_req.date);
    eprintln!("Git URL: {} {}", pull_req.git_url, pull_req.git_ref);

    // Step 2: Execute git pull
    eprintln!("\n=== Executing Git Pull ===\n");
    if !git_pull(&pull_req.git_url, &pull_req.git_ref)? {
        eprintln!("\n=== Merge Completed Successfully ===");
        eprintln!("No conflicts detected. The merge was completed automatically.");
        return Ok(Exit::Done);
    }

    // Step 3: Resolve the conflicts
    eprintln!("\n=== Merge Conflicts Detected ===\n");
    let res = resolve::resolve(opts, Some(&pull_req))?;
    if res.exit != Exit::Done {
        eprintln!("\nThe merge is left in progress for you to look at: {}", res.reason);
        return Ok(res.exit);
    }

    // Step 4: Commit the merge with pull request information
    eprintln!("\n=== Committing Merge ===\n");
    let summary = extract_summary(&pull_req.body, opts).unwrap_or_else(|| {
        eprintln!("Note: Could not extract summary automatically");
        String::new()
    });
    let commit_msg = build_merge_commit_message(&pull_req, &summary, &res.notes);
    eprintln!("Commit message:\n{}", commit_msg);

    let msg_file = tempfile::NamedTempFile::new()?;
    std::fs::write(msg_file.path(), &commit_msg)?;
    let commit_result = Command::new("git")
        .args(["commit", "--cleanup=strip", "-F"])
        .arg(msg_file.path())
        .output()
        .context("Failed to run git commit")?;
    if !commit_result.status.success() {
        bail!("Failed to commit merge: {}", String::from_utf8_lossy(&commit_result.stderr).trim());
    }
    eprintln!("\n=== Merge Committed Successfully ===");
    Ok(Exit::Done)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_email_headers() {
        let raw = r#"From: Paul Walmsley <paul@kernel.org>
Subject: [GIT PULL] RISC-V updates for v6.19
Date: Thu, 11 Dec 2025 19:36:00 -0700
Message-ID: <test123@kernel.org>

This is the body of the email.
"#;
        let (from, subject, date, msg_id, body) = parse_email_headers(raw);
        assert_eq!(from, "Paul Walmsley <paul@kernel.org>");
        assert_eq!(subject, "[GIT PULL] RISC-V updates for v6.19");
        assert_eq!(date, "Thu, 11 Dec 2025 19:36:00 -0700");
        assert_eq!(msg_id, "<test123@kernel.org>");
        assert!(body.contains("This is the body"));
    }

    #[test]
    fn test_parse_email_headers_multiline() {
        let raw = r#"From: Paul Walmsley <paul@kernel.org>
Subject: [GIT PULL] RISC-V updates
 for v6.19 merge window
Date: Thu, 11 Dec 2025 19:36:00 -0700

Body here.
"#;
        let (_, subject, _, _, _) = parse_email_headers(raw);
        assert!(subject.contains("RISC-V updates"));
        assert!(subject.contains("for v6.19 merge window"));
    }

    #[test]
    fn test_extract_git_info() {
        let body = r#"Please pull this set of changes.

The following changes are available in the Git repository at:

  git://git.kernel.org/pub/scm/linux/kernel/git/riscv/linux tags/riscv-for-linus-6.19

for you to fetch changes up to abc123.
"#;
        let result = extract_git_info(body);
        assert!(result.is_some());
        let (url, git_ref) = result.unwrap();
        assert_eq!(url, "git://git.kernel.org/pub/scm/linux/kernel/git/riscv/linux");
        assert_eq!(git_ref, "tags/riscv-for-linus-6.19");
    }

    #[test]
    fn test_extract_git_info_https() {
        let body = r#"Available at:

  https://git.kernel.org/pub/scm/linux/kernel/git/foo/bar.git feature-branch

Thanks!
"#;
        let result = extract_git_info(body);
        assert!(result.is_some());
        let (url, git_ref) = result.unwrap();
        assert!(url.starts_with("https://git.kernel.org"));
        assert_eq!(git_ref, "feature-branch");
    }

    #[test]
    fn test_extract_git_info_none() {
        let body = "This email has no git URL in it.";
        let result = extract_git_info(body);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_merge_commit_message() {
        let pull_req = PullRequest {
            message_id: "test123@kernel.org".to_string(),
            subject: "[GIT PULL] Important updates for v6.19".to_string(),
            from: "Maintainer <maintainer@kernel.org>".to_string(),
            date: "2025-12-11".to_string(),
            git_url: "git://git.kernel.org/pub/scm/foo".to_string(),
            git_ref: "tags/foo-for-v6.19".to_string(),
            body: String::new(),
        };

        let summary = "This is the maintainer's summary of changes.";
        let resolution = "Resolved by keeping both changes.";
        let msg = build_merge_commit_message(&pull_req, summary, resolution);
        assert!(msg.contains("Merge tags/foo-for-v6.19"));
        assert!(msg.contains("Important updates")); // subject without [GIT PULL]
        assert!(msg.contains("maintainer's summary"));
        assert!(msg.contains("conflict resolution"));
        assert!(msg.contains("keeping both changes"));
        assert!(msg.contains("https://lore.kernel.org/all/test123@kernel.org/"));
    }

}
