# ADR-006: Two squeezers — bit-depth + median filter only

**Status:** Accepted — 2026-05-13

## Context

Xu et al. (NDSS 2018) describe Feature Squeezing as a **family** of input transformations:

- Bit-depth reduction (colour quantisation)
- Local smoothing — median filter
- Local smoothing — non-local means
- Spatial smoothing — Gaussian blur
- (combination: max embedding shift across all squeezers)

Each squeezer adds another panel to the pipeline strip and another similarity bar. The seminar demo has a fixed-width UI (~1100 px) and a finite attention budget — the audience must grasp the verdict logic in one glance.

## Decision

Implement **two squeezers only**:

1. `BitDepthSqueezer` — reduces 8-bit channels to `bits` levels (slider: 1–8).
2. `MedianFilterSqueezer` — `scipy.ndimage.median_filter` with odd kernel (slider: 3 / 5 / 7).

The pipeline strip has exactly 4 panels: **Original → Adv. Glasses → Bit Squeezed → Median Filtered**. The verdict logic is `max(|sim_attacked − sim_bit|, |sim_attacked − sim_median|) > 0.50 → ADVERSARIAL`.

## Consequences

- ✓ UI fits the 4-panel pipeline strip cleanly; the audience can compare four images in one glance.
- ✓ Two squeezers are enough to demonstrate the *technique* — one colour-domain (bit-depth), one spatial-domain (median filter). Adding more wouldn't change the story.
- ✓ Adding more later is mechanical: subclass `BaseSqueezer`, append to the list passed to `SqueezeDetector`.
- ⚠ The presentation must be honest that we're not running the full Xu et al. menu. Frame it as: "we picked the two most illustrative; the paper composes more."
- ⚠ The detection threshold (0.50) is tuned for these two on this glasses attack — adding squeezers would invalidate the tuning.

## See also

- `docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md` §5, §6
- Eyal's plan Tasks 2, 3 — squeezer implementations
- ADR-004 — attack approach (the threshold is jointly tuned with the attack)
