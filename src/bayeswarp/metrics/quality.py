from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F


def _ensure_rgb(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    return x.clamp(0, 1)


def _ensure_rgb_299(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_rgb(x)
    return F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)


def _to_clip_input(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_rgb(x)
    return F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)


def compute_fid(real_images: List[torch.Tensor], fake_images: List[torch.Tensor], device: torch.device) -> float:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as e:
        raise ImportError('compute_fid requires torchmetrics and torchvision-compatible Inception weights.') from e
    if len(real_images) == 0 or len(fake_images) == 0:
        return float('nan')
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    real = torch.cat([_ensure_rgb_299(x) for x in real_images], dim=0).to(device)
    fake = torch.cat([_ensure_rgb_299(x) for x in fake_images], dim=0).to(device)
    fid.update(real, real=True)
    fid.update(fake, real=False)
    return float(fid.compute().item())


class SCSComputer:
    def __init__(self, device: torch.device):
        try:
            import open_clip
        except Exception as e:
            raise ImportError('compute_scs requires open-clip-torch.') from e
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def score(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        x1 = _to_clip_input(x1).to(self.device)
        x2 = _to_clip_input(x2).to(self.device)
        z1 = self.model.encode_image(x1)
        z2 = self.model.encode_image(x2)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return float((z1 * z2).sum(dim=-1).mean().item())


def compute_scs(seed_failure_pairs: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
    if len(seed_failure_pairs) == 0:
        return float('nan')
    scs = SCSComputer(device)
    scores = [scs.score(x_seed, x_fail) for x_seed, x_fail in seed_failure_pairs]
    return float(sum(scores) / len(scores))
