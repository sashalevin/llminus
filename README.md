# LLMinus

LLM-assisted merge conflict resolution for the Linux kernel.

Resolving a merge conflict means understanding why the two sides diverged,
what each was trying to do, and how to combine them so the result is at least
as good as both. LLMinus supports a maintainer's decisions about merge
conflicts. It learns from how past conflicts in the kernel's history were
resolved, and has an LLM investigate the branches being merged before
proposing a resolution.

- `learn` walks the kernel's merge commits and records every one that needed
  a manual resolution: what each side changed and how the conflict was
  resolved.
- `vectorize` turns those resolutions into embeddings (Jina's code model,
  `jinaai/jina-embeddings-v2-base-code`, reading up to 4096 tokens; run
  locally with [fastembed](https://github.com/Anush008/fastembed-rs)).
- `find` shows the historical resolutions most similar to the conflicts
  currently in the tree.
- `resolve` resolves the merge in progress with an LLM: every unmerged path
  (content conflicts, but also modify/delete, add/add and binary ones), with
  similar past resolutions and instructions to research both sides and
  lore.kernel.org first. The LLM edits files and writes a verdict; llminus
  checks it, stages the result, and build tests it. With no merge in
  progress it instead fixes the build of the merge commit at HEAD.
- `pull` fetches a `[GIT PULL]` email from lore.kernel.org, pulls the tree
  it names, and on conflicts resolves them as `resolve` does, with the email
  included, so that any resolution instructions from the submitting
  maintainer are taken into account. When everything resolves it commits the
  merge with a summary, the resolution notes and a `Link:` to the email.

## Requirements

- Rust 1.88 or newer
- git 2.38 or newer (`git show --remerge-diff`, `git merge-tree --write-tree`)
- An LLM: Claude Code (`claude`) or Codex (`codex`) for the built-in
  backends, or any command that reads a prompt
- A build command (optional, but without one nothing is build tested):
  `--build-cmd`, `$LLMINUS_BUILD_CMD`, `git config llminus.buildCmd`, or else
  `~/build.sh`. It runs at the top of a scratch worktree whose HEAD is the
  commit to build, with `{commit}` and `{baseline}` replaced and
  `LLMINUS_BUILD_COMMIT` / `LLMINUS_BUILD_BASELINE` set (errors the baseline
  commit already has are not the resolution's), and exits 0 (no new errors),
  1 (new errors) or 75 (couldn't tell, e.g. the build machines are down).
  A command that builds the committed HEAD elsewhere (such as
  [`stable build log`](https://git.kernel.org/pub/scm/linux/kernel/git/sashal/stable-tools.git))
  doesn't need the files: `--build-no-checkout` / `git config
  llminus.buildNoCheckout true` skips checking them out. `--build-desc` tells
  the LLM what the build covers.
- [lei](https://public-inbox.org/lei.html) for lore.kernel.org (recommended:
  plain web fetches of lore get bot-blocked; without lei, `pull` falls back
  to HTTP)
- Network access for `pull`, and for `vectorize`/`find` on first use (they
  download the ~640MB embedding model)

## Building

```sh
cargo build --release
# or install to ~/.cargo/bin
cargo install --path .
```

## Usage

Run llminus from the top of a kernel tree.

```sh
# Optional: build the database of past resolutions, from a range or with no
# range from the entire history. Re-running only processes new merges.
llminus learn v6.18..v7.3
llminus vectorize
llminus find 3            # with conflicts in the tree: the 3 most similar

# Resolve the merge in progress
llminus resolve --llm claude
llminus resolve --llm codex --model gpt-5.4
llminus resolve --llm-cmd "my-llm --prompt-file {prompt}"
llminus resolve --llm claude --no-build
llminus resolve --llm claude --build-cmd "make -j64 allmodconfig all"

# A pull request mail for the prompt (message-id or file), and linux-next's
# "manual merge of" reports for the conflicted files fetched from lore
llminus resolve --llm claude --pr-mail <message-id> --prefetch-lore

# Pull a pull request by message ID and resolve any conflicts
llminus pull 321f40c8-c694-4cc6-a4cb-df0c9b3d3fcd@kernel.org --llm claude

# Replay Linus's conflicted merges of a merge window and compare
llminus eval v7.2..v7.3-rc1 --committer "Linus Torvalds" --llm claude \
    --next-ref tags --i-understand-this-spends-ai-budget
```

### What happens

1. llminus collects the unmerged paths and git's own description of each
   conflict, the commits on each side that touched each file (with their
   `Link:`s), the identifiers only one side's version of a conflict uses,
   how linux-next resolved the same files (below), and similar past
   resolutions (if there is a database), and writes the prompt.
2. The LLM investigates and edits files. It may not change git state: it runs
   with a read-only `git` and a search-only `lei` first on its PATH, and the
   built-in backends add their own restrictions (Claude Code permission
   rules; Codex's sandbox, without network). When done it writes
   `.llminus-verdict.json`: `RESOLVED` with every unmerged path and its
   decision (`resolved`, `keep`, `delete`) plus any other file it had to
   change (`semantic_fixup`, with a rationale) and notes for the commit
   message, or `ABSTAIN` with what it found.
3. llminus checks that HEAD, the branch and the merge in progress are
   untouched and that exactly the declared files changed, then stages them.
4. It builds the result as a scratch commit (`write-tree` + `commit-tree`
   with the merge's parents, in a temporary worktree) with the build command,
   so the merge in progress is never touched. Errors the merge's first parent
   (`{baseline}`) already has don't count. New errors go back to the LLM, for
   at most `--repair-rounds` (2) more rounds.
5. On success the resolution is staged, the merge is still in progress, and
   the notes for the commit message are in
   `$(git rev-parse --git-path LLMINUS_RESULT)`, ending in `Assisted-by: LLM`.
   Committing is up to the caller.

### linux-next as a reference

Linus checks his resolutions against linux-next's. For each conflicted path,
llminus looks for the linux-next merge that resolved it by hand (an upstream
ref only: `refs/remotes/linux-next/master`, or what `LLMINUS_NEXT_REF` names,
`none` for none), shows that resolution to the LLM as a hint, and afterwards
compares. If linux-next resolved the very same conflict (identical parent
blobs) to a different result, that goes to a human unless
`--allow-next-differ`; otherwise the comparison goes into the notes.

### Evaluation

`llminus eval <range>` replays the merges in the range that git can't merge
automatically, each in a scratch repository that borrows the objects but has
refs only for the two parents, and compares llminus's result with the real
merge, path by path (exact, whitespace-only differences, different,
unresolved) and for the whole tree. Results go to `results.jsonl` and
`report.md`, which lists every divergence with its diff. `--next-ref tags`
gives each merge the newest `next-*` tag older than it; the resolution
database is limited to ancestors of the merge's first parent.

Exit status of `resolve` and `pull`:

| | |
|---|---|
| 0 | resolved, staged and build tested (or `--no-build`) |
| 1 | error, or the LLM changed git state it must leave alone |
| 2 | needs a human: the LLM abstained, left conflicts, changed undeclared files, or the build still fails |
| 3 | the build could not be judged (build hosts down): nothing is blamed |

Notes:

- A caller can set `LLMINUS_INVOCATION_ID`; the result file then starts with
  `llminus-invocation: <id>`, so it can't be mistaken for an older run's.
- Each LLM run is limited by `--timeout` (45m); the whole process group is
  killed when it's exceeded. Usage limits are waited out up to
  `--rate-limit-retries` times.
- Successful resolutions are recorded in `~/.local/share/llminus/fixups/`
  with an exact manifest of what they apply to, for replaying later. Nothing
  replays them yet.
- `--max-tokens` (default 100000) limits the prompt size by dropping
  historical examples. The database is `.llminus-resolutions.json` in the
  current directory, or `$LLMINUS_DB`.
- The embedding model is cached in `~/.cache/llminus/fastembed`
  (`$XDG_CACHE_HOME/llminus/fastembed`), or `$FASTEMBED_CACHE_DIR`.

## License

GPL-2.0-only, see [LICENSE](LICENSE).
