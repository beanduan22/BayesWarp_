from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
import torch.nn as nn

from bayeswarp.bo.acquisition import build_acquisition
from bayeswarp.bo.exact_gp import ExactGPSurrogate
from bayeswarp.bo.svgp_surrogate import SVGPSurrogate
from bayeswarp.models.factory import build_model
from bayeswarp.mutation.grid_mutator import GridMutator, PixelMutator, build_mutator
from bayeswarp.search.cmaes import CMAES
from bayeswarp.testing.bayeswarp import BayesWarpTester, BayesWarpConfig
from bayeswarp.testing.objective import allocate_budgets
from bayeswarp.metrics.failure import compute_failure_metrics
from bayeswarp.metrics.coverage import neuron_coverage, topk_neuron_coverage, critical_neuron_coverage
from bayeswarp.metrics.adequacy import distance_based_surprise_adequacy
from bayeswarp.metrics.redundancy import count_distinct, pairwise_distance_quantile
from bayeswarp.metrics.statistics import cohen_kappa, holm_adjust, wilson_interval


UNIT_MIN = torch.zeros(3, 1, 1)
UNIT_MAX = torch.ones(3, 1, 1)


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


def make_tester(model, device, cfg, channels=1):
    p_min = torch.zeros(channels, 1, 1)
    p_max = torch.ones(channels, 1, 1)
    return BayesWarpTester(model, device, cfg, p_min, p_max)


def test_budget_allocation():
    for total, k in [(10000, 9), (10000, 999), (10000, 7)]:
        budgets = allocate_budgets(total, k)
        assert sum(budgets) == total
        assert max(budgets) - min(budgets) <= 1
        assert budgets == sorted(budgets, reverse=True)


def test_grid_mutator_constraints():
    mask = torch.zeros(8, 8)
    mask[2:5, 2:5] = 1.0
    mutator = GridMutator(
        image_shape=(3, 8, 8), region_mask=mask, p_min=UNIT_MIN, p_max=UNIT_MAX, n=2, r=0.1, eta=0.1
    )
    assert mutator.dim == 4

    x0 = torch.rand(3, 8, 8)
    assert abs(mutator.u_bound - mutator.r) < 1e-6
    u = mutator.clip_u(torch.randn(mutator.dim) * 50.0)
    assert float(u.abs().max()) <= mutator.r + 1e-6
    assert abs(float(mutator.clip_u(torch.full((mutator.dim,), 0.05)).abs().max()) - 0.05) < 1e-6

    deltas = mutator.sample_deltas(64, torch.device('cpu'))
    assert float(deltas.abs().max()) <= mutator.r + 1e-6

    accumulated = mutator.clip_u(torch.full((mutator.dim,), mutator.r) * 10.0)
    assert float(accumulated.abs().max()) <= mutator.r + 1e-6

    x = mutator.reconstruct(u, x0)
    delta = (x - x0).abs().sum(dim=0)
    assert float(delta[mask == 0].max()) < 1e-6
    assert torch.allclose(mutator.reconstruct(torch.zeros(mutator.dim), x0), x0)

    shared = mutator.interpolate_to_region(u)
    assert torch.allclose(shared[0], shared[1], atol=1e-6)
    assert torch.allclose(shared[0], shared[2], atol=1e-6)

    assert float(x.min()) >= -0.1 - 1e-6
    assert float(x.max()) <= 1.1 + 1e-6


def test_pixel_mutator():
    mask = torch.zeros(8, 8)
    mask[2:5, 2:5] = 1.0
    mutator = build_mutator(
        image_shape=(3, 8, 8),
        region_mask=mask,
        p_min=UNIT_MIN,
        p_max=UNIT_MAX,
        n=2,
        r=0.1,
        eta=0.1,
        parameterization='pixel',
    )
    assert isinstance(mutator, PixelMutator)
    assert mutator.dim == 9

    x0 = torch.rand(3, 8, 8)
    x = mutator.reconstruct(mutator.sample_deltas(1, torch.device('cpu'))[0], x0)
    delta = (x - x0).abs().sum(dim=0)
    assert float(delta[mask == 0].max()) < 1e-6


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


def test_exact_gp_surrogate():
    surrogate = ExactGPSurrogate(dim=1, device=torch.device('cpu'), bound=0.1)
    xs = torch.linspace(-0.1, 0.1, 20).unsqueeze(1)
    target = torch.sin(xs.squeeze(1) * 8)
    for xi, yi in zip(xs, target):
        surrogate.add_observation(xi, float(yi))
    for _ in range(200):
        surrogate.fit_step()
    mu, sigma = surrogate.predict(xs)
    assert mu.shape == (20,) and sigma.shape == (20,)
    assert float(sigma.min()) > 0


def test_acquisitions():
    mu = torch.tensor([0.0, 1.0, 2.0])
    sigma = torch.tensor([1.0, 0.5, 0.1])
    ucb = build_acquisition('ucb')(mu, sigma, 1.0, 0.0)
    assert torch.allclose(ucb, mu + sigma)
    ei = build_acquisition('ei')(mu, sigma, 1.0, 1.0)
    assert ei.shape == mu.shape
    assert float(ei.min()) >= 0.0
    assert float(ei[2]) > float(ei[0])


def test_cmaes():
    torch.manual_seed(0)
    optimizer = CMAES(dim=3, device=torch.device('cpu'), bound=0.1, sigma0=0.05, population=8)
    target = torch.tensor([0.05, -0.05, 0.0])
    for _ in range(60):
        offspring = optimizer.ask()
        values = [float(-(candidate - target).pow(2).sum()) for candidate in offspring]
        optimizer.tell(offspring, values)
    assert float((optimizer.mean.float() - target).abs().max()) < 0.05


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

    make_tester(model, device, base_config(budget=0)).run_on_seed(x0)
    overhead = calls['n']

    calls['n'] = 0
    cfg = base_config(budget=180)
    out = make_tester(model, device, cfg).run_on_seed(x0)
    assert calls['n'] - overhead == cfg.budget
    assert out['evaluations_used'] == cfg.budget

    model.forward = forward
    region = out['region_mask']
    outside = region == 0
    for failure in out['failures']:
        x = failure['x'].squeeze(0)
        delta = (x - x0).abs().sum(dim=0)
        if outside.any():
            assert float(delta[outside].max()) < 1e-6
        assert float(x.min()) >= -cfg.eta - 1e-5
        assert float(x.max()) <= 1.0 + cfg.eta + 1e-5
        assert failure['pred'] != failure['og']


def test_search_variants_respect_budget():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).eval()
    device = torch.device('cpu')
    x0 = torch.rand(1, 28, 28)

    for overrides in [
        {'acquisition': 'ei'},
        {'surrogate': 'exact_gp'},
        {'search': 'cmaes', 'cma_population': 5},
    ]:
        cfg = base_config(budget=24, n=2, m=8, **overrides)
        out = make_tester(model, device, cfg).run_on_seed(x0)
        assert out['evaluations_used'] == cfg.budget

    for ablation in ('no_localization', 'no_bayesian', 'no_merging', 'no_grid', 'no_noise'):
        cfg = base_config(budget=18, n=2, m=8, ablation=ablation)
        out = make_tester(model, device, cfg).run_on_seed(x0)
        assert out['evaluations_used'] == cfg.budget


def test_pipeline_and_metrics():
    device = torch.device('cpu')
    model = build_model('lenet5', 10, pretrained=False).to(device).eval()
    cfg = base_config(budget=12, n=1, r=0.1)
    out = make_tester(model, device, cfg).run_on_seed(torch.rand(1, 28, 28))
    metrics = compute_failure_metrics([out], budget=cfg.budget)
    assert {'NoF', 'FSR', 'TPF', 'DoF', 'QFF_median', 'per_seed_dof'} <= set(metrics)
    assert metrics['NoF'] == len(out['failures'])
    assert 0.0 <= metrics['FSR'] <= 1.0
    assert set(out['stage_time_sec']) and all(v >= 0.0 for v in out['stage_time_sec'].values())

    images = [torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28)]
    seeds = [torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28)]
    assert 0.0 <= neuron_coverage(model, images) <= 1.0
    assert 0.0 <= topk_neuron_coverage(model, images, 3) <= 1.0
    assert 0.0 <= critical_neuron_coverage(model, images, seeds) <= 1.0


def test_analysis_metrics():
    reference = torch.tensor([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    reference_preds = [0, 0, 1]
    generated = torch.tensor([[0.1, 0.1]])
    scores = distance_based_surprise_adequacy(generated, [0], reference, reference_preds)
    assert len(scores) == 1 and scores[0] > 0

    vectors = torch.cat([torch.zeros(4, 3, 4, 4), torch.ones(3, 3, 4, 4)], dim=0)
    assert count_distinct(vectors, threshold=1e-6) == 2
    assert pairwise_distance_quantile(vectors, 0.5) >= 0.0

    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert all(a >= b for a, b in zip(adjusted, [0.01, 0.04, 0.03]))
    rate, low, high = wilson_interval(29, 30)
    assert low < rate <= high
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) - 1.0) < 1e-9


def main():
    test_budget_allocation()
    test_grid_mutator_constraints()
    test_pixel_mutator()
    test_svgp_surrogate()
    test_exact_gp_surrogate()
    test_acquisitions()
    test_cmaes()
    test_budget_is_exact_and_mutations_stay_in_region()
    test_search_variants_respect_budget()
    test_pipeline_and_metrics()
    test_analysis_metrics()
    print('SMOKE_TEST_OK')


if __name__ == '__main__':
    main()
