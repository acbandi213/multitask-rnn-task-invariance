#!/usr/bin/env python
"""Download the stored results (what the figure notebooks read) into results/.

    python scripts/download_results.py              # results.tar.gz   (~130 MB)
    python scripts/download_results.py --features   # + features.tar.gz: simulated unit features of the two main sets of networks

The archives are assets of the GitHub release named by TAG. While the repository is private the download goes through the GitHub
command-line tool (`gh auth login` once); for a public repository a plain download is used. Checksums are in scripts/release.sha256.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

REPO = 'acbandi213/multitask-rnn-task-invariance'
TAG = 'v1.0.0'
ROOT = os.environ.get('MTRNN_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def expected():
    out = {}
    with open(os.path.join(HERE, 'release.sha256')) as f:
        for line in f:
            if line.strip():
                digest, name = line.split()
                out[name] = digest
    return out


def fetch(name, dest):
    if shutil.which('gh'):
        r = subprocess.run(['gh', 'release', 'download', TAG, '--repo', REPO, '--pattern', name, '--dir', dest, '--clobber'])
        if r.returncode == 0:
            return
        print('gh could not download the asset; trying a plain download', file=sys.stderr)
    url = f'https://github.com/{REPO}/releases/download/{TAG}/{name}'
    print('downloading', url)
    urllib.request.urlretrieve(url, os.path.join(dest, name))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features', action='store_true', help='also download the simulated unit features (features.tar.gz)')
    a = p.parse_args()
    tmp = os.path.join(ROOT, 'dist'); os.makedirs(tmp, exist_ok=True)
    want = expected()
    for name in ['results.tar.gz'] + (['features.tar.gz'] if a.features else []):
        path = os.path.join(tmp, name)
        if not (os.path.exists(path) and sha256(path) == want[name]):
            fetch(name, tmp)
        got = sha256(path)
        if got != want[name]:
            sys.exit(f'{name}: checksum {got} does not match scripts/release.sha256 ({want[name]})')
        with tarfile.open(path) as t:
            members = [m for m in t.getmembers() if m.name.startswith('results/') and '..' not in m.name.split('/')]
            t.extractall(ROOT, members=members)
        print(f'{name}: verified and extracted into {os.path.join(ROOT, "results")}')


if __name__ == '__main__':
    main()
