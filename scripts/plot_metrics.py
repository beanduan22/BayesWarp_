from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--x', default='file')
    parser.add_argument('--y', default='NoF')
    parser.add_argument('--out', default='results/plot.png')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    plt.figure(figsize=(12, 5))
    plt.bar(df[args.x].astype(str), df[args.y])
    plt.xticks(rotation=75, ha='right')
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
