from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import hashlib
import json

import torch


SCHEMA = 'bayeswarp/finetune-sources/1'


def _canonical(manifest: Dict) -> str:
    payload = {
        'schema': SCHEMA,
        'dataset': manifest['dataset'],
        'seed_split': manifest['seed_split'],
        'num_sources': int(manifest['num_sources']),
        'per_source': int(manifest['per_source']),
        'sources': [
            [int(entry['dataset_index']), int(entry['label'])]
            for entry in manifest['sources']
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def manifest_fingerprint(manifest: Dict) -> str:
    return 'sha256:' + hashlib.sha256(_canonical(manifest).encode('utf-8')).hexdigest()


def class_histogram(labels: Sequence[int]) -> Dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {str(label): int(counts[label]) for label in sorted(counts)}


def build_manifest(
    *,
    dataset: str,
    seed_split: str,
    model: str,
    config_path: str,
    sources: Sequence[Tuple[int, int]],
    per_source: int,
    selection_seed: int,
    eligibility: Optional[Dict] = None,
) -> Dict:
    entries = [
        {'position': position, 'dataset_index': int(index), 'label': int(label)}
        for position, (index, label) in enumerate(sources)
    ]
    if len({entry['dataset_index'] for entry in entries}) != len(entries):
        raise ValueError('source dataset indices must be unique')
    manifest = {
        'schema': SCHEMA,
        'config': config_path,
        'dataset': dataset,
        'seed_split': seed_split,
        'model': model,
        'selection_seed': int(selection_seed),
        'num_sources': len(entries),
        'per_source': int(per_source),
        'num_samples': len(entries) * int(per_source),
        'sources': entries,
        'class_histogram': class_histogram([entry['label'] for entry in entries]),
    }
    if eligibility is not None:
        manifest['eligibility'] = eligibility
    manifest['fingerprint'] = manifest_fingerprint(manifest)
    return manifest


def load_manifest(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('schema') != SCHEMA:
        raise ValueError(f'Unexpected source manifest schema: {manifest.get("schema")!r}')
    expected = manifest_fingerprint(manifest)
    if manifest.get('fingerprint') != expected:
        raise ValueError(
            'Source manifest fingerprint does not match its contents; '
            f'stored={manifest.get("fingerprint")!r} computed={expected!r}'
        )
    if manifest['num_sources'] != len(manifest['sources']):
        raise ValueError('Source manifest num_sources disagrees with the source list')
    return manifest


def source_indices(manifest: Dict) -> List[int]:
    return [int(entry['dataset_index']) for entry in manifest['sources']]


def source_labels(manifest: Dict) -> List[int]:
    return [int(entry['label']) for entry in manifest['sources']]


def source_tensors(manifest: Dict, seed_dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    images: List[torch.Tensor] = []
    labels: List[int] = []
    for entry in manifest['sources']:
        index = int(entry['dataset_index'])
        x, y = seed_dataset[index]
        if int(y) != int(entry['label']):
            raise ValueError(
                f'Source label mismatch at dataset index {index}: '
                f'manifest={entry["label"]} dataset={int(y)}. '
                'The manifest was built against a different split or dataset root.'
            )
        images.append(x.unsqueeze(0))
        labels.append(int(y))
    return torch.cat(images, dim=0), torch.tensor(labels, dtype=torch.long)


def expand_labels(labels: torch.Tensor, per_source: int) -> torch.Tensor:
    return labels.repeat_interleave(int(per_source))


def resolve_pack_seed_ids(pack: Dict, fallback_indices: Optional[Sequence[int]] = None) -> List[int]:
    seed_results = pack.get('seed_results') or []
    if seed_results and 'seed_id' in seed_results[0]:
        return [int(r['seed_id']) for r in seed_results]
    if fallback_indices is None:
        raise ValueError(
            'Failure pack has no seed_id fields and no fallback seed indices were provided. '
            'Re-run generation, or pass the deterministic seed subset indices.'
        )
    fallback = [int(v) for v in fallback_indices]
    if seed_results and len(fallback) != len(seed_results):
        raise ValueError(
            f'Fallback seed indices ({len(fallback)}) do not match the pack seed results '
            f'({len(seed_results)}); the pack was produced with a different seed selection.'
        )
    return fallback


def bank_seed_ids(pack: Dict, fallback_indices: Optional[Sequence[int]] = None) -> List[int]:
    bank = pack['failure_bank']
    if bank and 'seed_id' in bank[0]:
        return [int(item['seed_id']) for item in bank]
    positions = resolve_pack_seed_ids(pack, fallback_indices)
    ids = []
    for item in bank:
        position = int(item['seed_idx'])
        if position >= len(positions):
            raise ValueError(f'Case references seed position {position} outside the seed subset')
        ids.append(positions[position])
    return ids
