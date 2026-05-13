# ADR-003: Gradio on Hugging Face Spaces

**Status:** Accepted — 2026-05-13

## Context

The seminar audience needs to open **one URL** and see the attack-then-defence demo without any setup. Hosting options considered:

- **Hugging Face Spaces (Gradio SDK)** — free public URL, native Gradio support, one-step git push deploy.
- **Streamlit Cloud** — also free, but Streamlit's rerun model is awkward for the multi-stage pipeline UI (state, mid-stage caching, async event wiring).
- **Self-hosted Flask + ngrok** — full control, but adds an ops surface (kept up during seminar, key rotation, etc.).

## Decision

Deploy to **Hugging Face Spaces** using the **Gradio 4.x Blocks API**.

The `README.md` YAML front-matter configures the Space (title, SDK version, app file). HF auto-rebuilds the Space on every push to `main` of the connected repo.

## Consequences

- ✓ Zero ops — the URL stays alive between seminars without imree/Eyal maintaining it.
- ✓ Gradio Blocks supports tabs, custom CSS, and gr.State (perfect for the registered-face session state described in the design spec).
- ✓ HF caches the insightface model weights between Space restarts; only the cold boot is slow.
- ⚠ Free tier CPU-only — see ADR-002 and ADR-004 for the constraints that flow from this.
- ⚠ No persistent storage — registered faces live only in `gr.State` per browser session (acceptable per the spec).

## See also

- `docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md` §2, §8
- ADR-002 (model choice), ADR-004 (attack approach)
