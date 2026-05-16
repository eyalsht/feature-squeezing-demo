"""Gradient-based impersonation attack for the Attack Lab tab.

Uses facenet-pytorch InceptionResnetV1 (VGGFace2) as a differentiable surrogate;
evaluates the result cross-model via insightface ArcFaceEmbedder.

NOTE: make_glasses_mask is defined locally here because this worktree branches
off origin/main before PR #10 (ui-polish-thresholds) merged the public refactor.
# TODO(merge): replace with `from attack import make_glasses_mask` after PR #10 merges.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local copy of make_glasses_mask — mirrors src/attack.py:28-42
# TODO(merge): replace with import from attack after PR #10 merges
# ---------------------------------------------------------------------------

def make_glasses_mask(img_shape: tuple[int, ...], face) -> np.ndarray:
    """Return float32 mask (H, W, 1) covering the glasses region of *face*."""
    left_eye, right_eye = face.kps[0], face.kps[1]
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    cy = (left_eye[1] + right_eye[1]) / 2

    x1 = max(0, int(left_eye[0] - eye_dist * 0.35))
    x2 = min(img_shape[1], int(right_eye[0] + eye_dist * 0.35))
    y1 = max(0, int(cy - eye_dist * 0.32))
    y2 = min(img_shape[0], int(cy + eye_dist * 0.38))

    mask = np.zeros(img_shape[:2], dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackResult:
    attacked_img: np.ndarray      # full face with optimised glasses, uint8 HxWx3
    glasses_pattern: np.ndarray   # cropped glasses region only, uint8 HxWx3
    sim_before: float             # cosine(attacker_orig, target) via ArcFaceEmbedder
    sim_after: float              # cosine(attacked, target) via ArcFaceEmbedder
    sim_after_torch: float        # same but via the PyTorch surrogate
    converged: bool               # sim_after > sim_before
    iterations_run: int


# ---------------------------------------------------------------------------
# ImpersonationAttacker
# ---------------------------------------------------------------------------

class ImpersonationAttacker:
    """PGD-style impersonation attack via a VGGFace2 InceptionResnetV1 surrogate.

    Optimises a perturbation *delta* confined to the glasses mask so that
    ``cosine(surrogate(x + delta), surrogate(target)) → 1``.

    Cross-model evaluation is performed by the caller-supplied ArcFaceEmbedder
    (insightface buffalo_l) to surface the surrogate-vs-evaluator gap.

    Parameters
    ----------
    lr : float
        Adam learning rate.
    steps : int
        Number of optimisation steps.
    lambda_tv : float
        TV-regularisation weight applied to the glasses region.
    eps_pixel : float
        Maximum absolute perturbation per pixel, in [0, 1] space
        (0.25 means ±63.75 / 255).
    """

    def __init__(
        self,
        lr: float = 0.02,
        steps: int = 200,
        lambda_tv: float = 0.01,
        eps_pixel: float = 0.25,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        self.lr = lr
        self.steps = steps
        self.lambda_tv = lambda_tv
        self.eps_pixel = eps_pixel

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def attack(
        self,
        attacker_img: np.ndarray,
        target_img: np.ndarray,
        mask: np.ndarray,
        arc_embedder=None,
    ) -> AttackResult:
        """Run the impersonation attack.

        Parameters
        ----------
        attacker_img : np.ndarray
            RGB uint8 HxWx3 image of the person to be disguised.
        target_img : np.ndarray
            RGB uint8 HxWx3 image of the person to impersonate.
        mask : np.ndarray
            Float32 HxWx1 binary mask (1 = glasses region, 0 = keep unchanged).
        arc_embedder : ArcFaceEmbedder | None
            insightface-based embedder for honest cross-model evaluation.
            If None, sim_before and sim_after are computed via the surrogate.

        Returns
        -------
        AttackResult
        """
        import torch
        import torch.nn.functional as F
        from facenet_pytorch import InceptionResnetV1

        device = torch.device("cpu")
        model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)

        H, W = attacker_img.shape[:2]
        mask_t = torch.from_numpy(
            mask.transpose(2, 0, 1)  # (1, H, W)
        ).float().to(device)

        # Convert images to float [0, 1] tensors
        x_orig = self._to_tensor(attacker_img, device)   # (1, 3, H, W)
        x_tgt  = self._to_tensor(target_img,  device)

        # Surrogate embeddings for target
        emb_tgt = self._embed_surrogate(model, x_tgt, device)  # (1, 512)

        # Learnable perturbation (initialised to zero)
        delta = torch.zeros_like(x_orig, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for step in range(self.steps):
            optimizer.zero_grad()
            x_adv = x_orig + delta * mask_t
            x_adv_clipped = torch.clamp(x_adv, 0.0, 1.0)
            emb_adv = self._embed_surrogate(model, x_adv_clipped, device)
            cos_loss = 1.0 - F.cosine_similarity(emb_adv, emb_tgt).mean()
            tv_loss  = self._tv_norm(delta * mask_t)
            loss = cos_loss + self.lambda_tv * tv_loss
            loss.backward()
            optimizer.step()
            # PGD clip
            with torch.no_grad():
                delta.clamp_(-self.eps_pixel, self.eps_pixel)

        # Build attacked image (numpy uint8)
        with torch.no_grad():
            x_adv_final = torch.clamp(x_orig + delta * mask_t, 0.0, 1.0)
            attacked_arr = self._to_numpy(x_adv_final)

        # Surrogate similarity
        emb_orig_torch = self._embed_surrogate(model, x_orig, device)
        emb_adv_final  = self._embed_surrogate(model, self._to_tensor(attacked_arr, device), device)
        sim_after_torch = float(
            F.cosine_similarity(emb_adv_final, emb_tgt).item()
        )

        # Cross-model (ArcFace) similarity
        if arc_embedder is not None:
            e_orig   = arc_embedder.embed(attacker_img)
            e_target = arc_embedder.embed(target_img)
            e_adv    = arc_embedder.embed(attacked_arr)
            if e_orig is not None and e_target is not None:
                sim_before = float(np.dot(e_orig, e_target))
            else:
                sim_before = float(
                    F.cosine_similarity(emb_orig_torch, emb_tgt).item()
                )
            if e_adv is not None and e_target is not None:
                sim_after = float(np.dot(e_adv, e_target))
            else:
                sim_after = sim_after_torch
        else:
            sim_before = float(
                F.cosine_similarity(emb_orig_torch, emb_tgt).item()
            )
            sim_after = sim_after_torch

        converged = sim_after > sim_before
        if not converged:
            log.warning(
                "Attack did not converge: sim_before=%.4f sim_after=%.4f",
                sim_before, sim_after,
            )

        # Crop glasses pattern from attacked image
        glasses_pattern = self._crop_glasses(attacked_arr, mask)

        return AttackResult(
            attacked_img=attacked_arr,
            glasses_pattern=glasses_pattern,
            sim_before=sim_before,
            sim_after=sim_after,
            sim_after_torch=sim_after_torch,
            converged=converged,
            iterations_run=self.steps,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(img: np.ndarray, device) -> "torch.Tensor":
        """uint8 HxWx3 → float32 1xCxHxW in [0, 1]."""
        import torch
        t = torch.from_numpy(img.astype(np.float32) / 255.0)  # HxWx3
        return t.permute(2, 0, 1).unsqueeze(0).to(device)      # 1x3xHxW

    @staticmethod
    def _to_numpy(tensor: "torch.Tensor") -> np.ndarray:
        """1x3xHxW float [0,1] → uint8 HxWx3."""
        arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _embed_surrogate(model, x: "torch.Tensor", device) -> "torch.Tensor":
        """Run the surrogate and return a (1, 512) embedding tensor.

        InceptionResnetV1 expects 160×160 input in [-1, 1].
        """
        import torch
        import torch.nn.functional as F
        x_resized = F.interpolate(x, size=(160, 160), mode="bilinear",
                                  align_corners=False)
        x_norm = x_resized * 2.0 - 1.0  # [0,1] → [-1,1]
        with torch.no_grad():
            emb = model(x_norm)  # (1, 512) already L2-normalised by the model
        return emb

    @staticmethod
    def _tv_norm(t: "torch.Tensor") -> "torch.Tensor":
        """Total variation of a (1, C, H, W) tensor."""
        diff_h = t[:, :, 1:, :] - t[:, :, :-1, :]
        diff_w = t[:, :, :, 1:] - t[:, :, :, :-1]
        return diff_h.abs().mean() + diff_w.abs().mean()

    @staticmethod
    def _crop_glasses(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return the bounding-box crop of the glasses region from *img*."""
        m2d = mask[:, :, 0]
        rows = np.any(m2d > 0, axis=1)
        cols = np.any(m2d > 0, axis=0)
        if not rows.any() or not cols.any():
            return img.copy()
        y1, y2 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
        x1, x2 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
        return img[y1:y2, x1:x2].copy()
