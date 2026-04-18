from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from bayeswarp.models.factory import build_model
from bayeswarp.testing.bayeswarp import BayesWarpTester, BayesWarpConfig
from bayeswarp.metrics.failure import compute_failure_metrics
from bayeswarp.metrics.coverage import neuron_coverage, topk_neuron_coverage, critical_neuron_coverage


def main():
    device = torch.device('cpu')
    model = build_model('lenet5', 10, pretrained=False).to(device).eval()
    x = torch.rand(1, 28, 28)
    cfg = BayesWarpConfig(
        saliency_method='gradcam',
        alpha=0.1,
        area_min=5,
        tau_iou=0.3,
        d_max=3,
        rho=0.6,
        S=2,
        eta=0.1,
        beta=0.02,
        epsilon=1e-4,
        kappa=1.0,
        n=1,
        m=8,
        per_class_budget=1,
        max_target_classes=1,
    )
    tester = BayesWarpTester(model, device, cfg)
    out = tester.run_on_seed(x)
    metrics = compute_failure_metrics([out])
    imgs = [torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28)]
    nc = neuron_coverage(model, imgs)
    tknc = topk_neuron_coverage(model, imgs, 3)
    cnc = critical_neuron_coverage(model, imgs)
    assert 'NoF' in metrics and 'FSR' in metrics and 'TPF' in metrics and 'DoF' in metrics
    assert 0.0 <= nc <= 1.0
    assert 0.0 <= tknc <= 1.0
    assert 0.0 <= cnc <= 1.0
    print('SMOKE_TEST_OK')


if __name__ == '__main__':
    main()
