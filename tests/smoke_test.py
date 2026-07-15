from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
import torch.nn as nn

from bayeswarp.bo.svgp_surrogate import SVGPSurrogate
from bayeswarp.models.factory import build_model
from bayeswarp.mutation.grid_mutator import GridMutator
from bayeswarp.testing.bayeswarp import BayesWarpTester, BayesWarpConfig
from bayeswarp.testing.objective import allocate_budgets
from bayeswarp.metrics.failure import compute_failure_metrics
from bayeswarp.metrics.coverage import neuron_coverage, topk_neuron_coverage, critical_neuron_coverage


def base_config(**overrides) -> BayesWarpConfig:
    cfg = dict(
        saliency_method='smoothgrad',
        alpha=0.4,
        area_min=1,
        tau_iou=0.3,
        d_max=3,
        rho=0.55,
        S=8,
        eta=0.1,
        epsilon=1e-3,
        kappa=1.0,
        r=1.0,
        n=4,
        m=16,
        budget=180,
        max_target_classes=3,
    )
    cfg.update(overrides)
    return BayesWarpConfig(**cfg)


def test_budget_allocation():
    for total, k in [(10000, 9), (10000, 999), (10000, 7)]:
        budgets = allocate_budgets(total, k)
        assert sum(budgets) == total

        assert max(budgets) - min(budgets) <= 1
        assert budgets == sorted(budgets, reverse=True)


def test_grid_mutator_constraints():
    mask = torch.zeros(8, 8)
    mask[2:5, 2:5] = 1.0
    mutator = GridMutator(image_shape=(3, 8, 8), region_mask=mask, n=2, r=0.1, eta=0.1)
    x0 = torch.rand(3, 8, 8)

    u = mutator.clip_u(torch.randn(mutator.dim) * 5.0)
    assert float(u.abs().max()) <= 0.1 + 1e-6

    x = mutator.reconstruct(u, x0)
    delta = (x - x0).abs().sum(dim=0)
    assert float(delta[mask == 0].max()) < 1e-7
    assert torch.allclose(mutator.reconstruct(torch.zeros(mutator.dim), x0), x0)

    pmin, pmax = float(x0.min()), float(x0.max())
    span = pmax - pmin
    assert float(x.min()) >= pmin - 0.1 * span - 1e-6
    assert float(x.max()) <= pmax + 0.1 * span + 1e-6


def test_svgp_surrogate():
    surrogate = SVGPSurrogate(dim=1, m=8, device=torch.device('cpu'), bound=0.1)
    assert surrogate.m_act == 0
    xs = torch.linspace(-0.1, 0.1, 30).unsqueeze(1)
    target = torch.sin(xs.squeeze(1) * 8)
    for xi, yi in zip(xs, target):
        surrogate.add_observation(xi, float(yi))
    assert surrogate.m_act == min(8, xs.size(0))

    first = surrogate.fit_step()
    for _ in range(1000):
        last = surrogate.fit_step()
    assert last > first

    mu, sigma = surrogate.predict(xs)
    assert mu.shape == (30,) and sigma.shape == (30,)
    assert float(sigma.min()) > 0
    assert float((mu - target).abs().mean()) < 0.1


def test_budget_is_exact_and_mutations_stay_in_region():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).eval()
    with torch.no_grad():
        model[1].weight.mul_(6.0)

    calls = {'n': 0}
    forward = model.forward

    def counting_forward(x):
        calls['n'] += x.size(0)
        return forward(x)

    model.forward = counting_forward
    device = torch.device('cpu')
    x0 = torch.rand(1, 28, 28)

    BayesWarpTester(model, device, base_config(budget=0)).run_on_seed(x0)
    overhead = calls['n']

    calls['n'] = 0
    cfg = base_config(budget=180)
    out = BayesWarpTester(model, device, cfg).run_on_seed(x0)
    assert calls['n'] - overhead == cfg.budget

    model.forward = forward
    region = out['region_mask']
    outside = region == 0
    pmin, pmax = float(x0.min()), float(x0.max())
    span = pmax - pmin
    for failure in out['failures']:
        x = failure['x'].squeeze(0)
        delta = (x - x0).abs().sum(dim=0)
        if outside.any():
            assert float(delta[outside].max()) < 1e-6
        assert float(x.min()) >= pmin - cfg.eta * span - 1e-5
        assert float(x.max()) <= pmax + cfg.eta * span + 1e-5
        assert failure['pred'] != failure['og']


def test_pipeline_and_metrics():
    device = torch.device('cpu')
    model = build_model('lenet5', 10, pretrained=False).to(device).eval()
    out = BayesWarpTester(model, device, base_config(budget=12, n=1, r=0.1)).run_on_seed(torch.rand(1, 28, 28))
    metrics = compute_failure_metrics([out])
    assert {'NoF', 'FSR', 'TPF', 'DoF'} <= set(metrics)
    assert metrics['NoF'] == len(out['failures'])
    assert 0.0 <= metrics['FSR'] <= 1.0

    images = [torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28)]
    assert 0.0 <= neuron_coverage(model, images) <= 1.0
    assert 0.0 <= topk_neuron_coverage(model, images, 3) <= 1.0
    assert 0.0 <= critical_neuron_coverage(model, images) <= 1.0


def main():
    test_budget_allocation()
    test_grid_mutator_constraints()
    test_svgp_surrogate()
    test_budget_is_exact_and_mutations_stay_in_region()
    test_pipeline_and_metrics()
    print('SMOKE_TEST_OK')


if __name__ == '__main__':
    main()
