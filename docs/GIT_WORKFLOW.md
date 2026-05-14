# Git Workflow

**PR-based workflow** so **imree**, **Eyal**, and **Claude agents** can work in parallel without colliding on `main`. Every change lands on `main` only via a merged Pull Request — never via a local merge or a direct push.

## Why PR-based

- **`main` stays clean** — only GitHub (via PR merge) ever updates it. We never push to `main` from a workstation.
- **Audit trail** — every change has a PR with diff view, commit history, and any review comments. Future-you (or Eyal) can answer "why did this change?" by reading the PR thread.
- **Parallel work** — different owners push different branches and open PRs simultaneously. Conflicts surface in the PR UI, not in a shared local merge.
- **Safer with Claude agents** — pushing `main` requires permission that the auto-classifier conservatively denies. Pushing feature branches doesn't.

## Branches

- **`main`** is the integration branch. **It only moves when a PR is merged on GitHub.** Local `main` is treated read-only: you `pull` it, you never `push` it, you never `merge` into it.
- **Feature branches** follow the naming convention:

  ```
  <owner>/<phase>-<slug>
  ```

  Examples:
  - `imree/phase-2-attack`
  - `eyal/phase-5-gradio-tab1`
  - `agent/phase-1-squeezers`

  Owner = whoever drives the branch (`imree`, `eyal`, `agent` when a Claude agent owns the work).

## Per-phase loop

For each phase (or for a heavy task within a phase):

1. **Sync local main:**
   ```powershell
   git checkout main
   git pull
   ```
2. **Branch:**
   ```powershell
   git checkout -b imree/phase-2-attack
   ```
3. **Work + commit** following the [commit-discipline](.claude/skills/commit-discipline/SKILL.md) skill (Conventional Commits, atomic diffs).
4. **Push the feature branch** to origin:
   ```powershell
   git push -u origin imree/phase-2-attack
   ```
5. **Pre-PR sanity** — make sure the branch is current and tests pass:
   ```powershell
   git fetch
   git rebase origin/main
   pytest
   git push --force-with-lease    # only after a rebase rewrote history
   ```
6. **Open the PR:**
   ```powershell
   gh pr create --base main --title "feat: GlassesAttacker" --body "..."
   ```
   Use a descriptive title (Conventional Commits style) and a short body summarising the change.
7. **Merge via GitHub** — either from the web UI or:
   ```powershell
   gh pr merge --merge --delete-branch
   ```
   `--merge` creates a merge commit (preserves the branch in `main`'s history graph, same effect as `--no-ff` locally). `--delete-branch` removes the remote feature branch.
8. **Sync local main with the new merge commit:**
   ```powershell
   git checkout main
   git pull
   git branch -d imree/phase-2-attack   # local cleanup
   ```

## Conflict policy

- Resolve conflicts **on the feature branch via `git rebase origin/main`**, then `git push --force-with-lease` to update the PR. Never on `main`.
- If a rebase is painful, prefer `git merge origin/main` into the feature branch — same outcome, simpler history within the branch.

## Force-push policy

- ❌ **Never** force-push to `main` (it would only happen if you committed there by accident — don't).
- ✅ `git push --force-with-lease` is allowed on **your own** feature branch (e.g. after a rebase that rewrote history).
- ❌ Never force-push a branch someone else has been pushing to without coordinating first.

## Parallelism rules

Different owners pick different phases. If two owners need the same file (e.g. both editing `app.py` in Phase 5), coordinate via `TODO.md`:

- Each `[ ]` checkbox in `TODO.md` gets an owner tag, e.g. `[ ] Task 9 — App bootstrap @eyal`.
- Don't start work on a tagged item without checking with the owner.
- If parallel work is truly needed on the same file, split by function/section and open small PRs early so the seam is visible.

## Reviewing PRs

For a 2-person + agents project, formal "approve before merge" gating is overkill. But a quick **self-check before merging your own PR** catches a lot:

- Diff looks right (no stray files, no debug prints)
- CI / pytest green
- The PR title + body would tell a teammate in 30 seconds what changed and why

If Eyal opens a PR, imree should at least skim it before merging — and vice versa.

## See also

- [.claude/skills/branch-discipline/SKILL.md](.claude/skills/branch-discipline/SKILL.md) — enforces the technical rules above when Claude is committing, pushing, or merging.
- [.claude/skills/commit-discipline/SKILL.md](.claude/skills/commit-discipline/SKILL.md) — commit-message format.
- [docs/adr/ADR-007-pr-based-workflow.md](docs/adr/ADR-007-pr-based-workflow.md) — why we switched from local-merge to PR-based.
