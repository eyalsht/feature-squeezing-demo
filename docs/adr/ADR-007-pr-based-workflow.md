# ADR-007: PR-based Git workflow (switching from local-merge)

**Status:** Accepted — 2026-05-13 (supersedes the workflow originally written for [ADR-001](ADR-001-python-3.13.md)-era Phase 0)

## Context

The original `GIT_WORKFLOW.md` defined a **local-merge** flow inspired by `rnn-lstm-sinesurgeon`:

1. Branch from `main`
2. Work + commit on the feature branch
3. `git merge --no-ff <branch>` into local `main`
4. `git push origin main`
5. Delete the feature branch

This worked fine when ran by a human but tripped over a structural mismatch with Claude Code's auto-mode classifier:

- The workflow doc said *"Never commit directly to `main`."*
- The auto-classifier read that and **blocked `git push origin main`** as a related "direct main update" — even though pushing already-merged commits is *not* the thing the rule forbids.
- The block could be reasoned away in conversation but reappeared every push.

Compounding factor: pushing `main` is the *only* action that requires it. Pushing a feature branch never does — it's just a remote branch ref, not a tracked-protected target.

The 2-person + agents team scale also makes a "local-merge then push main" flow lose its main advantage (low ceremony) — every workstation has to be the source of truth at merge time, and there's no shared review surface for an in-progress change.

## Decision

Switch to a **PR-based workflow**:

1. Branch from `main` (same as before).
2. Work + commit on the feature branch.
3. `git push -u origin <branch>` — push the feature branch to origin.
4. `gh pr create --base main --title "..." --body "..."` — open a Pull Request.
5. **Merge on GitHub** (web UI or `gh pr merge --merge --delete-branch`) — the merge commit is created server-side by GitHub.
6. Locally: `git checkout main && git pull` to sync.

Local `main` becomes **read-only** — it only ever advances via `git pull` after a remote merge. Workstations never push to or merge into it.

## Consequences

- ✓ **No auto-classifier collisions.** No one ever needs to `git push origin main` again — GitHub itself moves `main` when a PR is merged.
- ✓ **Better audit trail.** Every change has a PR thread on GitHub with diff view, message, and any review comments. Future imree or Eyal can read the PR to understand "why".
- ✓ **Parallelism scales naturally.** Multiple branches can have open PRs at once. Conflicts surface in the GitHub UI before merge, not as a local stale-main problem.
- ✓ **Plays well with agents.** Pushing feature branches is uncontentious; agents can do it without ceremony.
- ⚠ **One extra step per phase** (push branch + open PR vs. merge locally + push main). Costs ~30 seconds with `gh` CLI.
- ⚠ **`gh` CLI is now a prerequisite** for the smooth path. The settings allowlist permits the relevant `gh` subcommands so it doesn't trigger permission prompts. Falling back to the web UI works fine if `gh` isn't available.
- ⚠ **The pre-existing 17 Phase 0 commits used the old flow** (local-merge into `main`, then a single `git push origin main` after a one-time fixup). Not retroactively rewritten — the history stays as-is. Phase 1 onward uses the new flow.

## See also

- [`GIT_WORKFLOW.md`](../../GIT_WORKFLOW.md) — full runbook for the new flow.
- [`.claude/skills/branch-discipline/SKILL.md`](../../.claude/skills/branch-discipline/SKILL.md) — technical enforcement.
