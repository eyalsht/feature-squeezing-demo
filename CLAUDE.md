# CLAUDE.md

Orchestration brief for Claude working in this repo. Keep this file lean; deeper context lives in the linked docs.

## Project

A single-URL Gradio app on Hugging Face Spaces demonstrating Feature Squeezing (Xu et al., NDSS 2018) detecting simulated adversarial-glasses attacks (Sharif et al., CCS 2016) on ArcFace face recognition. Companion artefact for a HUP Seminar presentation by **imree** and **Eyal**.

Canonical references:
- **Product spec** — [`docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md`](docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md) — owned by Eyal. **Do not modify product behaviour without sign-off.**
- **Per-step plan** — [`docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md`](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md) — 12 TDD tasks with pre-written tests; follow it task-by-task.
- **Requirements** — [`PRD.md`](PRD.md). **Decisions** — [`docs/adr/`](docs/adr/). **Phase tracker** — [`TODO.md`](TODO.md).

## Stack

- Python **3.13** (see [ADR-001](docs/adr/ADR-001-python-3.13.md))
- Gradio 4.x (Blocks API)
- `insightface` (ArcFace `buffalo_l`) on `onnxruntime` CPU
- `numpy` + `scipy.ndimage` + `Pillow` for image ops
- `pytest` + `pytest-mock` for tests

## Setup

**Prerequisite (Windows):** Microsoft Visual C++ Build Tools, "Desktop development with C++" workload — needed to build `insightface`'s cython extension. Download from <https://visualstudio.microsoft.com/visual-cpp-build-tools/>. See [ADR-001](docs/adr/ADR-001-python-3.13.md#consequences).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\scripts\dev.ps1 install
.\scripts\dev.ps1 dataset    # one-time: download 6 LFW images
```

## Common commands

```powershell
.\scripts\dev.ps1 test       # pytest -v
.\scripts\dev.ps1 app        # launch Gradio locally on :7860
.\scripts\dev.ps1 smoke      # install + test + import sanity
```

## Discipline (active skills in `.claude/skills/`)

- [`tdd-cycle`](.claude/skills/tdd-cycle/SKILL.md) — RED → GREEN → REFACTOR; one failing test before any implementation. Tests for Eyal's tasks are pre-written in the per-step plan — copy them verbatim.
- [`commit-discipline`](.claude/skills/commit-discipline/SKILL.md) — Conventional Commits, atomic diffs (< 300 lines).
- [`branch-discipline`](.claude/skills/branch-discipline/SKILL.md) — Feature branches per phase, `--no-ff` merge, never commit directly to `main`. See [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md).

## Scope rule

Don't change **product** behaviour (anything described in the design spec or PRD) without explicit sign-off from imree or Eyal. Updating the product = update PRD/ADR first, then implement.

Workflow / tooling / docs are fair game to refine without sign-off, as long as the change is small and reversible.

## Deployment note

First HF Spaces boot downloads ~300 MB of insightface model weights to `/root/.insightface/` (~3–4 min). Cached afterward. The README YAML header configures the Space. See Eyal's plan Task 12.

## Out of scope here (intentionally not enforced)

This project runs the **lean** quality tier — no ruff, no mypy, no coverage gates, no file-size cap. TDD + a green `pytest` is the bar.
