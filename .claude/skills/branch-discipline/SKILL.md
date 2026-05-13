---
name: branch-discipline
description: Feature-branch workflow rules for this repo — never commit directly to main, always rebase + test before merge, --no-ff merges. Invoke before any commit or merge.
---

# Branch Discipline

This repo uses a **feature-branch workflow** so imree, Eyal, and Claude agents can work in parallel without colliding. The full runbook is in [`GIT_WORKFLOW.md`](../../../GIT_WORKFLOW.md); this skill enforces the rules technically.

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

## Before merging to `main`

Run, in order:

1. **Rebase on latest `main`:**
   ```powershell
   git fetch
   git rebase main
   ```
   Resolve any conflicts on the feature branch, never on `main`.

2. **Run tests:**
   ```powershell
   pytest
   ```
   All green, no skips that weren't already there. If any test fails, fix on this branch — do not merge red.

3. **Merge with `--no-ff`** (preserves branch history in the graph):
   ```powershell
   git checkout main
   git merge --no-ff <branch-name>
   ```

4. **Push and clean up:**
   ```powershell
   git push origin main
   git branch -d <branch-name>
   git push origin --delete <branch-name>
   ```

## Hard rules

- ❌ **Never commit directly to `main`.** First commit on `main` should be the `--no-ff` merge commit.
- ❌ **Never force-push to `main`.** Not under any circumstance.
- ❌ **Never merge with red tests.** Even on a side branch — fix first.
- ✅ `git push --force-with-lease` is allowed on your own feature branch after a clean rebase.
- ✅ `git merge main` into a feature branch is fine if rebase is painful.

## Parallel work

Before starting work on a phase or task, check `TODO.md` for owner tags. If a task is tagged `@eyal` or `@imree`, coordinate before claiming it for an agent. If un-tagged, claim by tagging it on your branch.

## See also

- [`GIT_WORKFLOW.md`](../../../GIT_WORKFLOW.md) — the workflow runbook (the "why" and runbook).
- `commit-discipline` — commit message format.
- `tdd-cycle` — the cycle that produces each commit.
