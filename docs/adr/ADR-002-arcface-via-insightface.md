# ADR-002: ArcFace via `insightface`

**Status:** Accepted — 2026-05-13

## Context

The demo needs a face recognition model that produces stable embeddings on which an adversarial-glasses attack visibly fails and Feature Squeezing visibly recovers. Candidates considered:

- **`insightface` (ArcFace, `buffalo_l`)** — pretrained, ONNX runtime, one-liner embedding.
- **`facenet-pytorch`** — pretrained FaceNet/InceptionResnetV1; PyTorch dep is heavy for a demo Space.
- **`dlib`** — old but small; weaker embeddings, less impressive demo.

Constraints:
- Free-tier HF Spaces — CPU only.
- First-boot model download budget ~300 MB.
- Single `embed(img) -> 512-dim L2-normed` interface needed.

## Decision

Use **`insightface==0.7.3`** with the `buffalo_l` model pack (default), running on `CPUExecutionProvider` via `onnxruntime`.

`insightface.app.FaceAnalysis` gives both detection and 512-dim ArcFace embedding in one call (`app.get(img)` returns `Face` objects with `bbox`, `kps`, `normed_embedding`). Wrapped in our `FaceDetector` + `ArcFaceEmbedder` classes (per Eyal's plan, Task 5) to keep the rest of the codebase ONNX/insightface-agnostic.

## Consequences

- ✓ Embeddings are visibly broken by the glasses noise and visibly recovered by median filtering — the story works.
- ✓ Same model also provides the landmark keypoints needed to place the glasses mask (eye coords).
- ✓ CPU latency ~0.5 s per `embed()` call — acceptable for a click-driven demo.
- ⚠ First boot on HF Spaces downloads ~300 MB of model weights to `/root/.insightface/`; cached for subsequent restarts but the first cold-start is ~3–4 min.
- ⚠ `insightface` API is not particularly documented; wrap it behind our own classes (already in plan) so future replacement is local.

## See also

- `docs/superpowers/specs/2026-05-13-feature-squeezing-demo-design.md` §2
- Eyal's plan Task 5 — `FaceDetector` + `ArcFaceEmbedder`
