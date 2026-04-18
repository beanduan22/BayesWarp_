from __future__ import annotations
import compileall
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def main():
    ok = compileall.compile_dir(str(REPO), quiet=1)
    if not ok:
        raise SystemExit('compileall failed')
    cmd = [sys.executable, str(REPO / 'tests' / 'smoke_test.py')]
    subprocess.run(cmd, check=True)
    print('REPO_VERIFIED_OK')


if __name__ == '__main__':
    main()
