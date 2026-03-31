from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results')
    parser.add_argument('--pattern', default='metrics_*.json')
    parser.add_argument('--out', default='results/summary.csv')
    args = parser.parse_args()

    rows = []
    root = Path(args.root)
    for path in root.rglob(args.pattern):
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        payload['file'] = str(path)
        rows.append(payload)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df)


if __name__ == '__main__':
    main()
