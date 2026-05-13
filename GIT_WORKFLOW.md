# Git Workflow

Feature-branch workflow so **imree**, **Eyal**, and **Claude agents** can work in parallel without colliding on `main`.

## Branches

- **`main`** is the integration branch. **Never commit directly to it.** Only `--no-ff` merges from feature branches land here.
- **Feature branches** follow the naming convention:

  ```
  <owner>/<phase>-<slug>
  ```

  Examples:
  - `imree/phase-0-scaffolding`
  - `eyal/phase-5-gradio-tab1`
  - `agent/phase-2-attack`

  Owner is whoever drives the branch — `imree`, `eyal`, or `agent` (when a Claude agent owns the work).

## Per-phase loop

For each phase (or for a heavy task within a phase):

1. **Sync:**
   ```powershell
   git checkout main
   git pull
   ```
2. **Branch:**
   ```powershell
   git checkout -b imree/phase-2-attack
   ```
3. **Work + commit** following the [commit-discipline](.claude/skills/commit-discipline/SKILL.md) skill (Conventional Commits, atomic diffs).
4. **Pre-merge sanity:**
   ```powershell
   git fetch
   git rebase main
   pytest
   ```
   Both must pass before merging.
5. **Merge with `--no-ff`** (preserves the branch in the history graph):
   ```powershell
   git checkout main
   git merge --no-ff imree/phase-2-attack
   git push origin main
   ```
6. **Delete the feature branch** (local + remote):
   ```powershell
   git branch -d imree/phase-2-attack
   git push origin --delete imree/phase-2-attack
   ```

## Conflict policy

- Resolve conflicts **on the feature branch via `git rebase main`**, never on `main`.
- If a rebase is painful, prefer `git merge main` into the feature branch — same outcome, simpler history within the branch.

## Force-push policy

- **Never** force-push to `main`.
- `git push --force-with-lease` is allowed on **your own** feature branch (e.g. after a rebase that rewrote history).
- Never force-push a branch someone else has been pushing to without coordinating first.

## Parallelism rules

Different owners pick different phases. If two owners need the same file (e.g. both editing `app.py` in Phase 5), coordinate via `TODO.md`:

- Each `[ ]` checkbox in `TODO.md` gets an owner tag, e.g. `[ ] Task 9 — App bootstrap @eyal`.
- Don't start work on a tagged item without checking with the owner.
- If parallel work is truly needed on the same file, split by function/section and commit early to make the seam visible.

## See also

- [.claude/skills/branch-discipline/SKILL.md](.claude/skills/branch-discipline/SKILL.md) — enforces the technical rules above when Claude is committing or merging.
- [.claude/skills/commit-discipline/SKILL.md](.claude/skills/commit-discipline/SKILL.md) — commit-message format.
