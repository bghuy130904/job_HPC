#!/usr/bin/env python3
"""Apply the reviewed pyCMF patch, only to the exact upstream version or already-patched files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,help='pyCMF git checkout; otherwise patch installed package')
    cfg=p.parse_args()
    patchdir=Path(__file__).resolve().parents[1]/'patches'
    manifest=json.loads((patchdir/'pycmf-sie-initialization.json').read_text())
    if cfg.root:
        root=cfg.root.resolve();strip=1
        mapping={name:root/name for name in manifest}
    else:
        import pycmf.OBDH.main as module
        root=Path(module.__file__).resolve().parents[2];strip=2
        mapping={name:root/Path(name).relative_to('src') for name in manifest}
    states=[]
    for name,path in mapping.items():
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        expected=manifest[name]
        if sha not in expected.values():
            raise SystemExit(f'Unsupported pyCMF version at {path}; use upstream af3614d then apply this patch')
        states.append('after' if sha==expected['after'] else 'before')
    if all(state=='after' for state in states):
        print('pyCMF SIE patch is already applied');return
    if any(state=='after' for state in states):
        raise SystemExit('Partially patched package; restore backups before applying')
    patch=patchdir/'pycmf-sie-initialization.patch'
    args=['patch','--batch','--forward',f'-p{strip}','-d',str(root),'-i',str(patch)]
    subprocess.run(args+['--dry-run'],check=True)
    originals={path:path.read_bytes() for path in mapping.values()}
    for path,content in originals.items():
        backup=path.with_name(path.name+'.pre-sie-fix')
        if backup.exists() and backup.read_bytes()!=content:
            raise SystemExit(f'Conflicting backup at {backup}')
        backup.write_bytes(content)
    try:
        subprocess.run(args,check=True)
        for name,path in mapping.items():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[name]['after']:
                raise RuntimeError(f'Post-patch hash mismatch: {path}')
    except Exception:
        for path,content in originals.items(): path.write_bytes(content)
        raise
    print(f'Applied pyCMF SIE patch in {root}; original .pre-sie-fix files retained')

if __name__=='__main__':main()
