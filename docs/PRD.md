# PRD — Feature Squeezing Live Demo

**Status:** Approved — 2026-05-13
**Owners:** imree (workflow), Eyal (product)
**Canonical product spec:** [`docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md`](docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md)
**Implementation plan:** [`docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md`](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md)

---

## 1. Overview

A single-URL Gradio app on Hugging Face Spaces that demonstrates the **Feature Squeezing** defence (Xu et al., NDSS 2018) detecting and neutralising a **simulated adversarial-glasses attack** (Sharif et al., CCS 2016) on ArcFace face recognition. Built as a companion artefact for a HUP Seminar presentation.

The demo tells the story in one screen:

> Original face → glasses attack breaks recognition → squeezing recovers the embedding → verdict banner fires 🚨 ADVERSARIAL DETECTED.

## 2. Users

- **Seminar audience** — opens one URL on their laptop, watches imree + Eyal click through the demo. No setup, no install.
- **Presenters (imree + Eyal)** — drive the demo from the same URL. Need predictable, deterministic UI behaviour during a live talk.

## 3. Functional Requirements

| ID | Requirement | Tab |
|---|---|---|
| FR-1 | Identity dropdown shows ≥ 3 pre-loaded identities; selecting one displays the target photo | 1 |
| FR-2 | Avatar strip / dropdown visually communicates the active identity | 1 |
| FR-3 | Register-face widget accepts 2 photos + a name; adds an identity to the session for the duration of that browser session | 1 |
| FR-4 | "⚔ Launch Attack" button generates a glasses-noise-overlaid image from the selected identity | 1 |
| FR-5 | Bit-depth slider (1–8) controls the `BitDepthSqueezer`; UI shows current value | 1 |
| FR-6 | Median-kernel slider (3 / 5 / 7) controls the `MedianFilterSqueezer`; UI shows current value | 1 |
| FR-7 | Pipeline strip renders 4 panels — Original / Attacked / Bit Squeezed / Median Filtered — each labelled with its cosine similarity to the target embedding | 1 |
| FR-8 | Verdict banner shows 🚨 ADVERSARIAL DETECTED (red) when `max(\|sim_attacked − sim_bit\|, \|sim_attacked − sim_median\|) > 0.50`, otherwise ✅ CLEAN INPUT (green) | 1 |
| FR-9 | Verify Identity tab accepts 2 photos of the same person and renders 3 result rows — clean / attacked / squeezed-attacked — each with similarity bar and verdict chip | 2 |

## 4. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Runs on Hugging Face Spaces **CPU-only free tier** (no GPU dependency) |
| NFR-2 | **Single-session state** — registered faces live in `gr.State`, no persistence across browser refreshes |
| NFR-3 | **First-boot < 4 minutes** including ~300 MB insightface model download |
| NFR-4 | **Python 3.13** (see [ADR-001](docs/adr/ADR-001-python-3.13.md)) |
| NFR-5 | **Deterministic UI** during the live demo — no flaky randomness in the attack output across consecutive clicks on the same input; seed `numpy` RNG inside `GlassesAttacker` if needed |

## 5. Locked Decisions

See `docs/adr/` for the full reasoning behind each.

| Decision | ADR |
|---|---|
| Python 3.13 | [ADR-001](docs/adr/ADR-001-python-3.13.md) |
| ArcFace via `insightface` | [ADR-002](docs/adr/ADR-002-arcface-via-insightface.md) |
| Gradio on Hugging Face Spaces | [ADR-003](docs/adr/ADR-003-gradio-on-hf-spaces.md) |
| Simulated glasses noise (not PGD) | [ADR-004](docs/adr/ADR-004-simulated-glasses-noise.md) |
| 3-identity LFW subset | [ADR-005](docs/adr/ADR-005-lfw-3-identity-dataset.md) |
| Two squeezers (bit-depth + median) | [ADR-006](docs/adr/ADR-006-two-squeezers.md) |

## 6. Success Metrics

- ✅ Verdict banner fires 🚨 on the default attacked input across all 3 pre-loaded identities.
- ✅ Verify Identity tab produces the expected 3-row pattern: ✅ same / ❌ identity-lost / ✅ recovered, for at least one alice/bob/carol pair.
- ✅ Seminar walkthrough runs end-to-end without backend restarts or hard refreshes.
- ✅ All `pytest` tests pass on Python 3.13.

## 7. Out of Scope

- Persistent storage of registered identities across sessions
- A real PGD/FGSM-optimised adversarial patch ([ADR-004](docs/adr/ADR-004-simulated-glasses-noise.md))
- Multi-identity comparison within a single attack run
- The full Xu et al. squeezer menu beyond bit-depth + median ([ADR-006](docs/adr/ADR-006-two-squeezers.md))
- GPU acceleration

## 8. Cross-References

- Design spec — [`docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md`](docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md)
- Implementation plan — [`docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md`](docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md)
- Phase tracker — [`TODO.md`](TODO.md)
- Workflow — [`CLAUDE.md`](CLAUDE.md), [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md)
