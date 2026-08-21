from __future__ import annotations
import argparse
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
from bayeswarp.metrics.statistics import (
    cluster_bootstrap_interval,
    cohen_kappa,
    hodges_lehmann,
    hodges_lehmann_ci,
    holm_adjust,
    wilson_interval,
)
from bayeswarp.finetune.conditions import (
    continued_condition,
    group_bank_by_source,
    select_generated,
    standard_condition,
)
from bayeswarp.finetune.sources import (
    bank_seed_ids,
    build_manifest,
    class_histogram,
    expand_labels,
    manifest_fingerprint,
    source_indices,
)


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


def _load_script(name):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'_script_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_manifest_is_tamper_evident():
    manifest = build_manifest(
        dataset='cifar10',
        seed_split='train',
        model='resnet18',
        config_path='configs/cifar10_resnet18_smoothgrad.yaml',
        sources=[(11, 3), (7, 1), (22, 3)],
        per_source=10,
        selection_seed=0,
    )
    assert manifest['num_sources'] == 3
    assert manifest['num_samples'] == 30
    assert manifest['class_histogram'] == {'1': 1, '3': 2}
    assert source_indices(manifest) == [11, 7, 22]
    assert manifest['fingerprint'] == manifest_fingerprint(manifest)

    tampered = dict(manifest)
    tampered['sources'] = list(manifest['sources'])[:2]
    assert manifest_fingerprint(tampered) != manifest['fingerprint']

    duplicated = False
    try:
        build_manifest(
            dataset='cifar10', seed_split='train', model='resnet18',
            config_path='c.yaml', sources=[(5, 0), (5, 1)], per_source=2, selection_seed=0,
        )
    except ValueError:
        duplicated = True
    assert duplicated


def test_conditions_preserve_the_source_protocol():
    per_source = 4
    X = torch.rand(5, 3, 32, 32)
    y = torch.tensor([2, 0, 1, 1, 2])

    added, labels, info = continued_condition(X, y, per_source)
    assert added.shape == (20, 3, 32, 32)
    assert torch.equal(labels, expand_labels(y, per_source))
    assert info['distinct_per_source'] == 1.0
    for i in range(5):
        for k in range(per_source):
            assert torch.equal(added[i * per_source + k], X[i])

    augmented, aug_labels, _ = standard_condition(X, y, 'cifar10', per_source, seed=0)
    assert augmented.shape == (20, 3, 32, 32)
    assert torch.equal(aug_labels, labels)
    assert class_histogram(labels.tolist()) == {'0': 4, '1': 8, '2': 8}


def test_generated_selection_is_matched_per_source():
    bank = []
    seed_ids = []
    for seed_id in (101, 202, 303):
        for case in range(6):
            bank.append({
                'x': torch.full((1, 1, 4, 4), float(seed_id + case)),
                'seed_y': seed_id % 10,
                'pred': case % 3,
            })
            seed_ids.append(seed_id)
    scores = [0.9 if index % 6 < 4 else 0.1 for index in range(len(bank))]

    groups = group_bank_by_source(bank, seed_ids)
    X, y, info = select_generated(groups, [101, 202, 303], 4, seed=0)
    assert X.shape[0] == 12
    assert y.tolist() == [1] * 4 + [2] * 4 + [3] * 4
    assert info['target_classes_used'] == 3

    filtered = group_bank_by_source(bank, seed_ids, scs_scores=scores, scs_threshold=0.8)
    assert all(len(v) == 4 for v in filtered.values())
    X2, _, _ = select_generated(filtered, [101, 202, 303], 4, seed=0)
    assert X2.shape[0] == 12

    starved = False
    try:
        select_generated(filtered, [101, 202, 303], 5, seed=0)
    except RuntimeError as e:
        starved = 'fewer than 5 generated cases' in str(e)
    assert starved


def test_bank_seed_ids_fall_back_to_positions():
    pack = {
        'seed_results': [{'seed_idx': 0}, {'seed_idx': 1}],
        'failure_bank': [{'seed_idx': 1}, {'seed_idx': 0}, {'seed_idx': 1}],
    }
    assert bank_seed_ids(pack, [77, 88]) == [88, 77, 88]
    pack['seed_results'] = [{'seed_idx': 0, 'seed_id': 77}, {'seed_idx': 1, 'seed_id': 88}]
    assert bank_seed_ids(pack, None) == [88, 77, 88]


def test_paired_seed_alignment_uses_ids_not_positions():
    import json
    import tempfile
    stats = _load_script('compute_statistics')

    a = {'seed_ids': [10, 20, 30], 'per_seed_cases': [5.0, 7.0, 9.0], 'run': 0}
    # Same seeds in a different order, plus one extra seed on the b side.
    b = {'seed_ids': [30, 10, 20, 40], 'per_seed_cases': [1.0, 2.0, 3.0, 99.0], 'run': 0}

    a_values, _ = stats.average_across_runs([('a.json', a)], 'per_seed_cases', 'seed_ids', False)
    b_values, _ = stats.average_across_runs([('b.json', b)], 'per_seed_cases', 'seed_ids', False)
    result = stats.paired_result(a_values, b_values, [10, 20, 30])
    # Positional pairing gives differences 4, 4, 6 (mean 4.667);
    # pairing by id gives 3, 4, 8 (mean 5.0).
    assert abs(result['mean_difference'] - 5.0) < 1e-9

    with tempfile.TemporaryDirectory() as tmp:
        a_path = Path(tmp) / 'a_run0.json'
        b_path = Path(tmp) / 'b_run0.json'
        a_path.write_text(json.dumps(a), encoding='utf-8')
        b_path.write_text(json.dumps(b), encoding='utf-8')

        strict = argparse.Namespace(
            a=str(a_path), b=str(b_path), metric='nof', allow_seed_intersection=False,
        )
        refused = False
        try:
            stats.paired_seed(strict)
        except ValueError as e:
            refused = 'shared' in str(e)
        assert refused, 'mismatched seed sets must not be silently truncated'

        relaxed = argparse.Namespace(
            a=str(a_path), b=str(b_path), metric='nof', allow_seed_intersection=True,
        )
        out = stats.paired_seed(relaxed)
        assert out['n_pairs'] == 3
        assert out['seed_ids'] == [10, 20, 30]
        assert out['seeds_only_in_b'] == 1
        assert abs(out['mean_difference'] - 5.0) < 1e-9

    missing_ids = False
    try:
        stats.seed_series('x.json', {'per_seed_cases': [1.0]}, 'per_seed_cases', 'seed_ids')
    except KeyError:
        missing_ids = True
    assert missing_ids


def test_paired_estimands_are_reported_together():
    stats = _load_script('compute_statistics')
    ids = list(range(12))
    b = {i: float(i) for i in ids}
    a = {i: b[i] + 1.0 + 0.25 * (i % 4) for i in ids}
    result = stats.paired_result(a, b, ids)
    assert result['primary_test'] == 'wilcoxon_signed_rank'
    assert result['primary_estimate'] == 'hodges_lehmann_shift'
    assert result['mean_difference_ci_low'] <= result['mean_difference'] <= result['mean_difference_ci_high']
    assert result['hodges_lehmann_ci_low'] <= result['hodges_lehmann_shift'] <= result['hodges_lehmann_ci_high']
    assert 'mean_difference_p_value' in result and 'wilcoxon_p_value' in result


def test_holm_family_keeps_labels_and_intervals_together():
    import json
    import tempfile
    stats = _load_script('compute_statistics')

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for index, p_value in enumerate([0.001, 0.02, 0.04, 0.05, 0.2, 0.3]):
            path = Path(tmp) / f'r{index}.json'
            path.write_text(json.dumps({
                'metric': 'scs',
                'n_pairs': 100,
                'wilcoxon_p_value': p_value,
                'hodges_lehmann_shift': 0.1,
                'hodges_lehmann_ci_low': 0.05,
                'hodges_lehmann_ci_high': 0.15,
                'mean_difference_ci_low': -1.0,
                'mean_difference_ci_high': 1.0,
            }), encoding='utf-8')
            paths.append(str(path))

        out = stats.holm_family(argparse.Namespace(
            inputs=paths, labels=None, family_name='scs',
            p_field='wilcoxon_p_value', estimate_field='hodges_lehmann_shift',
        ))
    assert out['family_size'] == 6
    assert out['method'] == 'holm_bonferroni'
    adjusted = [e['p_adjusted'] for e in out['results']]
    assert adjusted == sorted(adjusted)
    assert all(adj >= raw for adj, raw in zip(adjusted, [0.001, 0.02, 0.04, 0.05, 0.2, 0.3]))
    # The interval reported next to the Hodges-Lehmann estimate is its own.
    assert out['results'][0]['ci_low'] == 0.05 and out['results'][0]['ci_high'] == 0.15
    assert not out['all_significant_at_0.05']


def test_human_validation_sample_avoids_seed_clusters():
    import random
    sampler = _load_script('sample_human_validation')

    candidates = []
    for seed_id in range(40):
        cases = 500 if seed_id == 0 else 6
        for run in range(5):
            for k in range(cases):
                candidates.append({
                    'seed_id': seed_id,
                    'source_label': seed_id % 5,
                    'run': run,
                    'item': (seed_id, run, k),
                })

    selected = sampler.balanced_sample(candidates, 30, 1, random.Random(0))
    assert len(selected) == 30
    seeds = [c['seed_id'] for c in selected]
    assert len(set(seeds)) == 30
    labels = {}
    for case in selected:
        labels[case['source_label']] = labels.get(case['source_label'], 0) + 1
    assert max(labels.values()) - min(labels.values()) <= 1
    assert len({c['run'] for c in selected}) > 1

    capped = sampler.balanced_sample(candidates, 60, 2, random.Random(1))
    counts = {}
    for case in capped:
        counts[case['seed_id']] = counts.get(case['seed_id'], 0) + 1
    assert max(counts.values()) <= 2


def test_cluster_bootstrap_widens_when_cases_share_seeds():
    labels = [1] * 24 + [0] * 6
    independent = cluster_bootstrap_interval(labels, list(range(30)), iterations=4000, seed=0)
    clustered = cluster_bootstrap_interval(labels, [i // 6 for i in range(30)], iterations=4000, seed=0)
    assert independent['n_clusters'] == 30 and independent['max_cluster_size'] == 1
    assert clustered['n_clusters'] == 5 and clustered['max_cluster_size'] == 6
    width = lambda r: r['ci_high'] - r['ci_low']
    assert width(clustered) > width(independent)

    _, wilson_low, wilson_high = wilson_interval(24, 30)
    assert width(clustered) > (wilson_high - wilson_low)


def test_hodges_lehmann_matches_the_wilcoxon_estimand():
    a = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]
    b = [2.0, 0.5, 3.0, 0.0, 4.0, 8.0, 1.0, 5.0, 4.0, 2.0]
    estimate = hodges_lehmann(a, b)
    same, low, high = hodges_lehmann_ci(a, b)
    assert abs(estimate - same) < 1e-12
    assert low <= estimate <= high
    assert abs(hodges_lehmann(a, a)) < 1e-12


def test_finetune_protocol_verifier_detects_mismatched_sources():
    import json
    import tempfile
    verifier = _load_script('verify_finetune_protocol')

    def record(condition, fingerprint, num_added=1000):
        return {
            'condition': condition,
            'run': 0,
            'num_added': num_added,
            'protocol': {
                'sources_fingerprint': fingerprint,
                'num_sources': 100,
                'per_source': 10,
                'source_class_histogram': {'0': 100},
                'added_class_histogram': {'0': num_added},
            },
        }

    def run_verifier(directory, expect):
        import contextlib
        import io
        argv = sys.argv
        sys.argv = ['verify', '--results', str(Path(directory) / 'controlled_finetune_*.json')]
        if expect:
            sys.argv += ['--expect-conditions', *expect]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                verifier.main()
            return 0
        except SystemExit as e:
            return int(e.code or 0)
        finally:
            sys.argv = argv

    conditions = ('continued', 'standard', 'random', 'autoattack', 'high_scs', 'bayeswarp')

    with tempfile.TemporaryDirectory() as tmp:
        for condition in conditions:
            (Path(tmp) / f'controlled_finetune_{condition}_run0.json').write_text(
                json.dumps(record(condition, 'sha256:aaa')), encoding='utf-8'
            )
        assert run_verifier(tmp, conditions) == 0

        # One condition drawn from a different source manifest must fail.
        (Path(tmp) / 'controlled_finetune_bayeswarp_run0.json').write_text(
            json.dumps(record('bayeswarp', 'sha256:bbb')), encoding='utf-8'
        )
        assert run_verifier(tmp, conditions) == 1

    with tempfile.TemporaryDirectory() as tmp:
        for condition in conditions:
            count = 500 if condition == 'continued' else 1000
            (Path(tmp) / f'controlled_finetune_{condition}_run0.json').write_text(
                json.dumps(record(condition, 'sha256:aaa', num_added=count)), encoding='utf-8'
            )
        # Unequal added-sample counts must fail too.
        assert run_verifier(tmp, conditions) == 1


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
    test_source_manifest_is_tamper_evident()
    test_conditions_preserve_the_source_protocol()
    test_generated_selection_is_matched_per_source()
    test_bank_seed_ids_fall_back_to_positions()
    test_paired_seed_alignment_uses_ids_not_positions()
    test_paired_estimands_are_reported_together()
    test_holm_family_keeps_labels_and_intervals_together()
    test_human_validation_sample_avoids_seed_clusters()
    test_cluster_bootstrap_widens_when_cases_share_seeds()
    test_hodges_lehmann_matches_the_wilcoxon_estimand()
    test_finetune_protocol_verifier_detects_mismatched_sources()
    print('SMOKE_TEST_OK')


if __name__ == '__main__':
    main()
