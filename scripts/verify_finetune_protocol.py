"""Check that every controlled fine-tuning condition really used the same sources.

Reads the ``controlled_finetune_*.json`` files a study produced and asserts the
properties the paper claims: one shared source manifest, the same number of
added samples, and the same added-sample class distribution across conditions.
Exits non-zero with a description of the first violation.
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from bayeswarp.utils.io import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Glob for controlled_finetune_*.json')
    parser.add_argument('--expect-conditions', nargs='*', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.results))
    if not paths:
        raise SystemExit(f'No files matched: {args.results}')

    records = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if 'protocol' not in payload:
            raise SystemExit(
                f'{path} predates the shared-source protocol (no "protocol" block). '
                'Re-run scripts/controlled_finetune.py with --sources.'
            )
        records.append((path, payload))

    problems = []
    by_run = {}
    for path, payload in records:
        by_run.setdefault(int(payload['run']), []).append((path, payload))

    for run, entries in sorted(by_run.items()):
        fingerprints = {p['protocol']['sources_fingerprint'] for _, p in entries}
        if len(fingerprints) > 1:
            problems.append(f'run {run}: conditions used {len(fingerprints)} different source manifests')
        counts = {p['num_added'] for _, p in entries}
        if len(counts) > 1:
            problems.append(f'run {run}: added-sample counts differ across conditions: {sorted(counts)}')
        histograms = {json.dumps(p['protocol']['added_class_histogram'], sort_keys=True) for _, p in entries}
        if len(histograms) > 1:
            problems.append(f'run {run}: added-sample class distributions differ across conditions')
        for path, payload in entries:
            protocol = payload['protocol']
            expected = int(protocol['num_sources']) * int(protocol['per_source'])
            if int(payload['num_added']) != expected:
                problems.append(
                    f'{path}: num_added={payload["num_added"]} but the manifest specifies {expected}'
                )
            if protocol['added_class_histogram'] != {
                k: v * int(protocol['per_source']) for k, v in protocol['source_class_histogram'].items()
            }:
                problems.append(
                    f'{path}: added-sample class histogram is not the source histogram scaled by per_source'
                )
        if args.expect_conditions:
            present = {p['condition'] for _, p in entries}
            missing = sorted(set(args.expect_conditions) - present)
            if missing:
                problems.append(f'run {run}: missing conditions {missing}')

    all_fingerprints = {p['protocol']['sources_fingerprint'] for _, p in records}
    if len(all_fingerprints) > 1:
        problems.append(
            f'{len(all_fingerprints)} distinct source manifests across runs; the study is only '
            'source-matched within a run, not across runs'
        )

    summary = {
        'files': len(records),
        'runs': sorted(by_run),
        'conditions': sorted({p['condition'] for _, p in records}),
        'fingerprints': sorted({p['protocol']['sources_fingerprint'] for _, p in records}),
        'num_added': sorted({p['num_added'] for _, p in records}),
        'distinct_per_source': {
            p['condition']: p.get('condition_info', {}).get('distinct_per_source')
            for _, p in records
        },
        'problems': problems,
        'status': 'FAIL' if problems else 'OK',
    }
    if args.out:
        save_json(summary, args.out)
    print(json.dumps(summary, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
