from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
from bayeswarp.models.factory import find_last_conv_layer


def normalize_map(attr: torch.Tensor) -> torch.Tensor:
    attr = attr.detach()
    attr = attr - attr.min()
    return attr / (attr.max() - attr.min() + 1e-8)


def _pred_target(model, x: torch.Tensor, target: Optional[int]) -> int:
    if target is not None:
        return int(target)
    with torch.no_grad():
        return int(model(x).argmax(dim=1).item())


def integrated_gradients_saliency(model, x: torch.Tensor, target: int, n_steps: int = 32) -> torch.Tensor:
    model.eval()
    baseline = torch.zeros_like(x)
    grads = []
    for alpha in torch.linspace(0.0, 1.0, steps=n_steps, device=x.device):
        xi = (baseline + alpha * (x - baseline)).clone().detach().requires_grad_(True)
        score = model(xi)[0, target]
        grad = torch.autograd.grad(score, xi, retain_graph=False, create_graph=False)[0]
        grads.append(grad.detach())
    avg_grad = torch.stack(grads, dim=0).mean(dim=0)
    attr = (x - baseline) * avg_grad
    attr = attr.abs().mean(dim=1).squeeze(0)
    return normalize_map(attr)


def smoothgrad_saliency(model, x: torch.Tensor, target: int, nt_samples: int = 16, stdevs: float = 0.1) -> torch.Tensor:
    model.eval()
    x_detached = x.detach()
    attr_sum = torch.zeros_like(x_detached)
    for _ in range(nt_samples):
        noise = torch.randn_like(x_detached) * stdevs
        xn = (x_detached + noise).clone().detach().requires_grad_(True)
        score = model(xn)[0, target]
        grad = torch.autograd.grad(score, xn, retain_graph=False, create_graph=False)[0]
        attr_sum += grad.detach().abs()
    attr = (attr_sum / nt_samples).mean(dim=1).squeeze(0)
    return normalize_map(attr)


def gradcam_saliency(model, x: torch.Tensor, target: int) -> torch.Tensor:
    model.eval()
    layer = find_last_conv_layer(model)
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations['value'] = output.detach()

    def bwd_hook(_, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    logits = model(x)
    score = logits[0, target]
    model.zero_grad(set_to_none=True)
    score.backward()
    h1.remove()
    h2.remove()

    act = activations['value']
    grad = gradients['value']
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze(0).squeeze(0)
    return normalize_map(cam)


def compute_saliency(model, x: torch.Tensor, target: Optional[int], method: str) -> torch.Tensor:
    target = _pred_target(model, x, target)
    method = method.lower()
    if method == 'gradcam':
        return gradcam_saliency(model, x, target)
    if method in {'ig', 'integrated_gradients'}:
        return integrated_gradients_saliency(model, x, target)
    if method in {'smoothgrad', 'sg'}:
        return smoothgrad_saliency(model, x, target)
    raise ValueError(f'Unsupported saliency method: {method}')
