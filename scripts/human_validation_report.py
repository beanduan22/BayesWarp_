"""Aggregate the human-validation annotations with clustering-aware intervals.

Wilson intervals assume the annotated cases are independent. They are only
independent if no source seed contributed more than one case, so this report
records the clustering it actually sees and always reports a cluster bootstrap
alongside the Wilson interval, resampling source seeds rather than cases.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from bayeswarp.utils.io import save_json
from bayeswarp.metrics.statistics import cluster_bootstrap_interval, cohen_kappa, wilson_interval


def adjudicate(record):
    a = int(record['annotator_a'])
    b = int(record['annotator_b'])
    if a == b:
        return a
    if 'adjudicator' not in record:
        raise ValueError(f"Task {record['task_id']} needs adjudication but has no adjudicator label")
    return int(record['adjudicator'])


def cluster_of(meta, cluster_key: str):
    """Cluster identity for one annotated case.

    Prefers the dataset-level ``seed_id``; falls back to the positional
    ``seed_idx`` for keys written before seed ids were recorded.
    """
    if cluster_key in meta:
        return meta[cluster_key]
    if cluster_key == 'seed_id' and 'seed_idx' in meta:
        return f"pos:{meta['seed_idx']}"
    raise KeyError(
        f'Annotation key has no "{cluster_key}" field, so cases cannot be grouped into clusters. '
        'Re-generate the sample with scripts/sample_human_validation.py.'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--cluster-key', default='seed_id',
        help='Field in key.json that identifies the cluster a case belongs to.',
    )
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    args = parser.parse_args()

    annotations = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    key = json.loads(Path(args.key).read_text(encoding='utf-8'))

    groups = defaultdict(list)
    primary_a = []
    primary_b = []

    for record in annotations:
        task_id = record['task_id']
        if task_id not in key:
            raise KeyError(f'Unknown task_id in annotations: {task_id}')
        meta = key[task_id]
        primary_a.append(int(record['annotator_a']))
        primary_b.append(int(record['annotator_b']))
        groups[(meta['dataset'], meta['model'], meta['method'])].append((adjudicate(record), meta))

    rows = []
    for (dataset, model, method), entries in sorted(groups.items()):
        total = len(entries)
        labels = [label for label, _ in entries]
        clusters = [cluster_of(meta, args.cluster_key) for _, meta in entries]
        preserved = [meta for label, meta in entries if label == 1]
        rate, low, high = wilson_interval(len(preserved), total)
        bootstrap = cluster_bootstrap_interval(
            labels, clusters,
            iterations=args.bootstrap,
            seed=args.bootstrap_seed,
        )
        rows.append({
            'dataset': dataset,
            'model': model,
            'method': method,
            'n': total,
            'preserved': len(preserved),
            'preserved_rate': rate,
            'ci_low': low,
            'ci_high': high,
            'ci_method': 'wilson (assumes independent cases)',
            'n_clusters': bootstrap['n_clusters'],
            'max_cases_per_cluster': bootstrap['max_cluster_size'],
            'clustered': bootstrap['max_cluster_size'] > 1,
            'cluster_bootstrap_ci_low': bootstrap['ci_low'],
            'cluster_bootstrap_ci_high': bootstrap['ci_high'],
            'cluster_bootstrap_iterations': bootstrap['iterations'],
            'runs': sorted({meta['run'] for _, meta in entries if 'run' in meta}),
            'source_classes': len({meta['source_label'] for _, meta in entries}),
            'verified_dof': len({meta['model_prediction'] for meta in preserved}),
        })

    clustered = [row for row in rows if row['clustered']]
    save_json(
        {
            'cohen_kappa': cohen_kappa(primary_a, primary_b),
            'total_annotated': len(annotations),
            'cluster_key': args.cluster_key,
            'reporting_note': (
                'Report the cluster bootstrap interval whenever any source seed contributed '
                'more than one annotated case; the Wilson interval is only valid when every '
                'cluster holds exactly one case.'
            ),
            'any_clustering': bool(clustered),
            'clustered_rows': [
                f"{row['dataset']}/{row['model']}/{row['method']}" for row in clustered
            ],
            'rows': rows,
        },
        args.out,
    )


if __name__ == '__main__':
    main()
