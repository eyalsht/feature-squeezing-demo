---
name: commit-discipline
description: Conventional Commits format and atomic-diff rules. Invoke before staging or committing changes.
---

# Commit Discipline

## Format

Conventional Commits: `<type>(<scope>)?: <subject>`. Subject is imperative, lowercase, no trailing period, ≤ 72 chars.

| Type | When |
|---|---|
| `feat` | A new user-visible feature or new public API |
| `fix` | A bug fix |
| `chore` | Tooling, deps, configs — nothing the user sees |
| `data` | Adding/updating dataset files |
| `docs` | Markdown docs only (CLAUDE.md, PRD, ADRs, README) |
| `test` | New tests for existing behaviour (mostly the GREEN step uses `feat` instead) |
| `refactor` | Code restructuring with no behaviour change |

Scope is the module name when it narrows usefully: `feat(squeezers): MedianFilterSqueezer`.

## Atomicity

- **One logical change per commit.** A new squeezer + its tests = one commit. A new squeezer + an unrelated CSS tweak = two commits.
- **Target < 300 lines per commit.** Larger commits are usually two changes mashed together.
- Each RED→GREEN→REFACTOR cycle (see `tdd-cycle`) produces exactly one commit.

## Body (optional)

Add a body when the *why* isn't obvious from the subject. Use plain prose, one blank line after the subject:

```
feat(detector): SqueezeDetector with 0.50 threshold

Threshold chosen jointly with the glasses-attack epsilon=0.15;
see ADR-006. Lower values would false-positive on clean inputs
with mild compression artifacts.
```

Do not write a body that just restates the subject.

## What NOT to do

- Don't squash unrelated changes into one commit "to keep history short" — small commits ARE the history.
- Don't write `chore: stuff` or `wip` — if it's truly WIP, leave it uncommitted on the branch.
- Don't amend a commit that's already on `main` (it isn't, because of `branch-discipline`, but as a habit).
- Don't bypass hooks with `--no-verify` unless the user explicitly asks for it.

## Examples from this project

Good:
- `feat(squeezers): BitDepthSqueezer with BaseSqueezer ABC`
- `feat(attack): GlassesAttacker with Face dataclass`
- `data: add 3 LFW identities (6 images) for demo dataset`
- `docs(adr): ADR-001 Python 3.13`
- `fix(app): handle None embedding when no face detected`

Bad:
- `feat: stuff`
- `update`
- `feat: add squeezer and also fix CSS and bump deps` (three changes — three commits)

## See also

- `tdd-cycle` — the cycle that produces each commit.
- `branch-discipline` — where commits land before merging to `main`.
- `GIT_WORKFLOW.md` — branch-level workflow.
