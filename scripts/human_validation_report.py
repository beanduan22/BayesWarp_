from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from bayeswarp.utils.io import save_json
from bayeswarp.metrics.statistics import cohen_kappa, wilson_interval


def adjudicate(record):
    a = int(record['annotator_a'])
    b = int(record['annotator_b'])
    if a == b:
        return a
    if 'adjudicator' not in record:
        raise ValueError(f"Task {record['task_id']} needs adjudication but has no adjudicator label")
    return int(record['adjudicator'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--out', required=True)
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
        preserved = [meta for label, meta in entries if label == 1]
        rate, low, high = wilson_interval(len(preserved), total)
        rows.append({
            'dataset': dataset,
            'model': model,
            'method': method,
            'n': total,
            'preserved': len(preserved),
            'preserved_rate': rate,
            'ci_low': low,
            'ci_high': high,
            'verified_dof': len({meta['model_prediction'] for meta in preserved}),
        })

    save_json(
        {
            'cohen_kappa': cohen_kappa(primary_a, primary_b),
            'total_annotated': len(annotations),
            'rows': rows,
        },
        args.out,
    )


if __name__ == '__main__':
    main()
