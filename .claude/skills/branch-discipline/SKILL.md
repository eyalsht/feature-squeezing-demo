---
name: branch-discipline
description: PR-based feature-branch workflow rules for this repo — never commit, merge, or push to local main. Always go through a GitHub PR. Invoke before any commit, push, or merge.
---

# Branch Discipline

This repo uses a **PR-based feature-branch workflow** so imree, Eyal, and Claude agents can work in parallel without colliding. The full runbook is in [`GIT_WORKFLOW.md`](../../../GIT_WORKFLOW.md); this skill enforces the rules technically.

The core rule: **local `main` is read-only.** It updates only via `git pull` after a PR is merged on GitHub. You never `commit`, `merge`, or `push` on `main` from your workstation.

## Before committing

Check the current branch:

```powershell
git branch --show-current
```

- **If on `main`**: STOP. Create a feature branch first:
  ```powershell
  git checkout -b <owner>/<phase>-<slug>
  ```
  Owner is `imree`, `eyal`, or `agent`. Slug is short kebab-case.
- **If on a feature branch**: proceed.

## Before opening a PR

Run, in order:

1. **Rebase on latest `main`:**
   ```powershell
   git fetch
   git rebase origin/main
   ```
   Resolve any conflicts on the feature branch.

2. **Run tests:**
   ```powershell
   pytest
   ```
   All green, no skips that weren't already there.

3. **Push the branch** (first push uses `-u`, later pushes don't need it):
   ```powershell
   git push -u origin <branch>
   ```
   If you rebased and the remote already has the branch, use `git push --force-with-lease`.

4. **Open the PR:**
   ```powershell
   gh pr create --base main --title "<conventional-commit-style title>" --body "<short summary>"
   ```

## Merging the PR

After the PR is open and any checks pass:

```powershell
gh pr merge --merge --delete-branch
```

- `--merge` creates a merge commit on `main` (preserves the branch in the history graph, same effect as a local `--no-ff` merge).
- `--delete-branch` removes the remote feature branch after merge.

Alternative: merge from the GitHub web UI ("Merge pull request" button — pick "Create a merge commit"). Same result.

## After merging

Sync local main and clean up:

```powershell
git checkout main
git pull
git branch -d <branch>     # local branch cleanup; -d is safe (only deletes if merged)
```

## Hard rules

- ❌ **Never commit directly to `main`.** Commits land via PR merge only.
- ❌ **Never merge into local `main`.** Merging happens on GitHub, not on your workstation.
- ❌ **Never `git push origin main`.** Local `main` only moves via `git pull`. (The auto-classifier will block this anyway — that's a feature, not a bug.)
- ❌ **Never force-push `main`.** Not under any circumstance.
- ❌ **Never merge a PR with red tests.** Fix the branch first.
- ✅ `git push --force-with-lease` is allowed on your own feature branch after a clean rebase.
- ✅ `git merge origin/main` into a feature branch is fine if rebase is painful.

## Parallel work

Before starting work on a phase or task, check `TODO.md` for owner tags. If a task is tagged `@eyal` or `@imree`, coordinate before claiming it for an agent. If un-tagged, claim by tagging it on your branch.

## See also

- [`GIT_WORKFLOW.md`](../../../GIT_WORKFLOW.md) — the workflow runbook (the "why" and full runbook).
- [`docs/adr/ADR-007-pr-based-workflow.md`](../../../docs/adr/ADR-007-pr-based-workflow.md) — why we switched from local-merge to PR-based.
- `commit-discipline` — commit message format.
- `tdd-cycle` — the cycle that produces each commit.
