# ADR-005: 3-identity LFW subset as pre-loaded dataset

**Status:** Accepted — 2026-05-13

## Context

The demo needs pre-loaded identities so the dropdown is non-empty and the audience can see a result within seconds of page load. Dataset choices:

- **LFW (Labeled Faces in the Wild)** — public-domain, frontal, well-lit, known to ArcFace.
- **CelebA** — larger, but requires registration/agreement to redistribute.
- **Homemade photos** of imree/Eyal — fun, but burns prep time on lighting/angle quality.

How many identities?
- 1 — boring (no dropdown choice).
- 3 — three distinct demos within one screen, fits the sidebar UI without scrolling.
- 6+ — visual clutter, no extra storytelling value.

How many photos per identity?
- 1 — embedding from a single photo is noisier.
- 2 — averaging two embeddings stabilises the reference (matches the live-registration flow which also uses 2 photos).

## Decision

**3 LFW identities, 2 photos each** (6 JPEGs total), stored at `dataset/{name}_1.jpg` and `dataset/{name}_2.jpg`.

Identities are downloaded by `dataset/prepare_dataset.py` from the public LFW mirror — not committed pre-resolved. Names in code are placeholders (`alice`, `bob`, `carol`); the actual LFW identities are public figures (George W. Bush, Colin Powell, Tony Blair in the current script).

## Consequences

- ✓ Demo has variety (3 distinct identities) without UI clutter.
- ✓ 2-photo averaging gives more stable reference embeddings and mirrors the live-registration flow exactly (`IdentityDatabase.register(name, photo1, photo2)`).
- ✓ LFW is permanent / mirror-stable; no risk of dataset deletion.
- ⚠ The placeholder names (alice/bob/carol) are unrelated to the actual face identity. Acceptable for an academic demo but be explicit about it in the presentation if asked.
- ⚠ The script downloads on first run — needs internet at setup time, not at runtime.

## See also

- `docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md` §9
- `dataset/prepare_dataset.py`
- Eyal's plan Task 8 — Dataset Preparation
