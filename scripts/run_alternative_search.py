from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from bayeswarp.pipeline import generate
from bayeswarp.utils.config import load_config
from bayeswarp.utils.device import get_device
from bayeswarp.utils.io import ensure_dir, save_json


CONFIGURATIONS = {
    'svgp_ucb': ({'surrogate': 'svgp', 'acquisition': 'ucb', 'search': 'bayesian'}, 'none'),
    'svgp_ei': ({'surrogate': 'svgp', 'acquisition': 'ei', 'search': 'bayesian'}, 'none'),
    'svgp_kappa0': ({'surrogate': 'svgp', 'acquisition': 'ucb', 'search': 'bayesian', 'kappa': 0.0}, 'none'),
    'svgp_k1': ({'surrogate': 'svgp', 'acquisition': 'ucb', 'search': 'bayesian', 'max_target_classes': 1}, 'none'),
    'no_noise': ({'surrogate': 'svgp', 'acquisition': 'ucb', 'search': 'bayesian'}, 'no_noise'),
    'exactgp_ucb': ({'surrogate': 'exact_gp', 'acquisition': 'ucb', 'search': 'bayesian'}, 'none'),
    'cmaes': ({'search': 'cmaes'}, 'none'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--configuration', required=True, choices=sorted(CONFIGURATIONS))
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--num-seeds', type=int, default=50)
    parser.add_argument('--budget', type=int, default=2000)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    device = get_device()

    overrides, ablation = CONFIGURATIONS[args.configuration]
    cfg = copy.deepcopy(base_cfg)
    cfg['bayeswarp'].update(overrides)
    cfg['bayeswarp']['budget'] = args.budget

    seed_results, _, metrics = generate(
        cfg,
        device,
        ablation=ablation,
        run=args.run,
        num_seeds=args.num_seeds,
        progress=args.configuration,
    )

    metrics['configuration'] = args.configuration
    metrics['budget'] = args.budget
    metrics['num_seeds'] = args.num_seeds
    metrics['time_per_seed_sec'] = metrics['total_time_sec'] / max(1, len(seed_results))

    out_dir = ensure_dir(base_cfg['output_dir'])
    save_json(metrics, out_dir / f'alternative_search_{args.configuration}_run{args.run}.json')


if __name__ == '__main__':
    main()
