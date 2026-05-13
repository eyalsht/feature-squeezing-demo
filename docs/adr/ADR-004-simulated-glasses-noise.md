# ADR-004: Simulated glasses noise instead of real PGD/FGSM

**Status:** Accepted — 2026-05-13

## Context

*Accessorize to a Crime* (Sharif et al., CCS 2016) generates **physically realisable** adversarial glasses by optimising a pixel patch over the eye region via projected gradient descent (PGD) against an end-to-end face recognition pipeline. The optimisation:

- requires backprop access to the face model (we have ONNX inference only),
- takes seconds-to-minutes per attack on CPU,
- needs a separate train-time setup that doesn't fit in a single-click demo flow.

## Decision

Generate the attack as **a uniform-noise mask over the eye region**:

```python
mask    = rectangle(left_eye_kp, right_eye_kp, eps_w, eps_h)
noise   = np.random.uniform(-epsilon*255, epsilon*255, mask.shape)
attacked = clip(img + noise * mask, 0, 255)
```

with `epsilon=0.15` by default. The eye region is derived from the insightface landmark keypoints already computed during identity registration.

## Consequences

- ✓ Per-click attack runs in < 100 ms on CPU.
- ✓ The qualitative effect on the embedding (sim drop > 0.50 → identity lost) matches the paper's headline result, which is what the demo needs to show.
- ✓ No gradient access to the model required — works with the ONNX-only insightface stack.
- ⚠ Not a "real" adversarial example in the academic sense — it's a destructive noise overlay, not an optimised perturbation. The presentation should be honest about this:
  - **Frame it as:** "simulated glasses attack — destroys the eye region's contribution to the embedding, analogous to what the paper achieves via optimisation."
  - **Do not claim:** "this is a PGD-optimised adversarial patch."
- ⚠ Feature Squeezing is *expected* to recover this attack easily — the median filter removes the high-frequency noise. This is fine for a demo (the story works) but the audience should be told the optimised version would be harder to defend.

## See also

- `docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md` §6, §7
- Eyal's plan Task 4 — `GlassesAttacker`
