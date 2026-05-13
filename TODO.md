# TODO — Phase Tracker

High-level phase tracker. **Per-step content lives in** [`docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md`](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md) — don't duplicate it here. This file tracks status and ownership only.

**Owner tags:** `@imree` / `@eyal` / `@agent` — claim by tagging before starting work (per [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) parallelism rules).

---

## Phase 0 — Environment & Dep Bump &nbsp; `@imree`

**Goal:** Repo installs and tests pass cleanly on a fresh Python 3.13 venv.

- [x] Project scaffold (requirements, fixtures, dataset helper, spec + plan docs) — committed in `f2885a7` by Eyal
- [x] Workflow scaffold (CLAUDE.md, PRD, TODO, ADRs, GIT_WORKFLOW, ported skills, dev.ps1) — this branch
- [x] Add `.python-version` pinning `3.13`
- [x] Bump `requirements.txt` for Python 3.13 wheel availability (see [ADR-001](docs/adr/ADR-001-python-3.13.md))
- [x] Run smoke install + `pytest` on a fresh 3.13 venv: `.\scripts\dev.ps1 smoke`

---

## Phase 1 — Squeezer Defences

**Goal:** `BitDepthSqueezer` and `MedianFilterSqueezer` implemented and unit-tested.

- [x] Task 2 — `BitDepthSqueezer` + `BaseSqueezer` ABC ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-2--bitdepthsqueezer)) `@imree` — `BitDepthSqueezer` (4 tests), L2 normalisation, shape/dtype preserved
- [x] Task 3 — `MedianFilterSqueezer` ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-3--medianfiltersqueezer)) `@imree` — salt-and-pepper removal, odd-kernel guard (3 tests)

Deliverables: `src/squeezers.py`, `tests/test_squeezers.py`

---

## Phase 2 — Glasses Attack

**Goal:** `GlassesAttacker` + `Face` dataclass implemented and unit-tested.

- [x] Task 4 — `GlassesAttacker` ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-4--glassesattacker)) `@eyal` — eye-region mask, epsilon-scaled noise, pixel-range clamp (4 tests)

Deliverables: `src/attack.py`, `tests/test_attack.py`

---

## Phase 3 — ArcFace Embedder + Detection

**Goal:** `FaceDetector`, `ArcFaceEmbedder`, `SqueezeDetector`, `DetectionResult` implemented and unit-tested (insightface mocked).

- [x] Task 5 — `FaceDetector` + `ArcFaceEmbedder` ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-5--facedetector-and-arcfaceembedder)) `@eyal` — lazy-load insightface, 512-dim L2-normed embedding, None if no face (3 tests)
- [x] Task 6 — `SqueezeDetector` + `DetectionResult` ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-6--detectionresult-and-squeezedetector)) `@eyal` — frozen dataclass, cosine shift detection, threshold=0.50 (3 tests)

Deliverables: `detector.py`, `tests/test_detector.py`

---

## Phase 4 — Dataset & Identity DB

**Goal:** `IdentityDatabase`, `Identity` dataclass implemented; 6 LFW images downloaded.

- [x] Task 7 — `IdentityDatabase` ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-7--identitydatabase)) `@agent` — `register()` for live demo face addition, `load()` from disk, `get()`/`names()` (4 tests)
- [ ] Task 8 — Add 6 dataset images ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-8--dataset-preparation)) — manually copy `alice_1.jpg`, `alice_2.jpg`, `bob_1.jpg`, `bob_2.jpg`, `carol_1.jpg`, `carol_2.jpg` into `dataset/`

Deliverables: `dataset.py`, `tests/test_dataset.py`, `dataset/{alice,bob,carol}_{1,2}.jpg`

---

## Phase 5 — Gradio App

**Goal:** Two-tab Gradio app fully wired and rendering correctly in a browser.

- [ ] Task 9 — App bootstrap (CSS theme + service init) ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-9--gradio-app-css-theme-and-service-bootstrap))
- [ ] Task 10 — Tab 1: Pipeline Demo ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-10--tab-1-pipeline-demo))
- [ ] Task 11 — Tab 2: Verify Identity ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-11--tab-2-verify-identity))

Deliverables: `app.py`

⚠ Phase 5 tasks edit the same file. Coordinate via owner tags — do not start two in parallel.

---

## Phase 6 — Hugging Face Spaces Deployment

**Goal:** Live URL works end-to-end; matches the smoke checklist in the per-step plan.

- [ ] Task 12 — Deploy to HF Spaces ([per-step plan](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md#task-12--hugging-face-spaces-deployment))

Deliverables: `README.md` with HF YAML header + live URL; HF Space connected to GitHub.

---

## Phase 7 — Demo Polish

**Goal:** Demo is presentation-ready and resilient to seminar-day surprises.

- [ ] Append live Space URL to `README.md`
- [ ] Run the manual smoke checklist on the live Space (8 items in Task 12 Step 4)
- [ ] Record a 30-second screen capture as a backup if the seminar Wi-Fi fails
- [ ] Take 2–3 high-DPI screenshots for slides (clean / attacked / squeezed-with-verdict)

---

## Status legend

- `[x]` done — landed on `main`
- `[ ]` not started
- `[~]` in progress (mark the branch name next to it: `[~] Task 4 @eyal eyal/phase-2-attack`)
