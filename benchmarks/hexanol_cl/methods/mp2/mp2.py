#!/usr/bin/env python3
"""Scan MP2-in-DFT using the unmodified supplied mp2_embed.py (HF base A).

Place mp2_embed.py beside this script; requires installed PySCF and pycmf.
python -u run_mp2_hexanol_hf.py --xyz input.xyz --basis sto-3g \
    --xc-env b3lyp --threads 12 --max-memory 40000 \
    --output mp2_hexanol_cl.json

7 active regions x No CL / n_shells=1,2,3; output schema matches OBDH.
--module-file selects another path to the original mp2_embed.py.
--resume continues successful points with matching settings/code versions.
--dry-run prints settings and the 28 cases without computing.
Results contain only the original HF-based total energy, not comparisons.
A .meta.json and per-point logs support resume and diagnostics.
The original module does not expose subsystem SCF convergence flags:
'completed' means its kernel returned a finite energy, not independent
certification of those internal SCFs. Inspect its SCF logs as usual.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

COUNTS = [2, 5, 8, 11, 14, 17, 21]
LABELS = ['OH', 'CH2OH', 'C2H4OH', 'C3H6OH', 'C4H8OH', 'C5H10OH', 'C6H13OH']
CASES = [('No CL', None)] + [(f'CL (n_shells={n})', n) for n in (1, 2, 3)]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=4, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_xyz(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    if len(lines) < 2 or int(lines[0]) != 21:
        raise ValueError('Expected the supplied 21-atom n-hexanol XYZ.')
    rows = [line.split() for line in lines[2:] if line.strip()]
    expected = ['O', 'H'] + ['C', 'H', 'H'] * 5 + ['C', 'H', 'H', 'H']
    if len(rows) != 21 or [row[0] for row in rows] != expected:
        raise ValueError('XYZ atom order must match the supplied OH-first input.xyz.')
    for row in rows:
        if len(row) != 4 or not all(math.isfinite(float(x)) for x in row[1:]):
            raise ValueError(f'Invalid XYZ row: {row}')
    return '\n'.join(' '.join(row) for row in rows)


def worker(config_path, count, shell, result_path):
    config = json.loads(Path(config_path).read_text())['config']
    started = time.monotonic()
    result = {'active_atom_count': count, 'n_shells': shell or None}
    try:
        from pyscf import gto, scf, lib
        import pyscf
        from pycmf.OBDH import CL_embed, uobdh_embed
        spec = importlib.util.spec_from_file_location('mp2_embed_input', config['module_file'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        lib.num_threads(config['threads'])
        mol = gto.M(atom=config['atom'], unit='Angstrom', basis=config['basis'],
                    charge=0, spin=0, verbose=4, max_memory=config['max_memory'])
        mf = scf.UHF(mol).density_fit(auxbasis=config['auxbasis'])
        mf.conv_tol = config['scf_tol']
        mf.max_cycle = config['scf_cycles']
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('Initial full-system UHF did not converge.')
        calc = module.UMP2_CL(mf, frozen=config['frozen'])
        calc.xc_env = config['xc_env']
        calc.use_embed = True
        calc.active_atoms = list(range(count))
        calc.use_cl = shell != 0
        calc.n_shells = shell or 1
        calc.mu = config['mu']
        calc.max_memory = config['max_memory']
        print(f'CASE active_atoms={calc.active_atoms}, use_cl={calc.use_cl}, '
              f'n_shells={calc.n_shells}', flush=True)
        energy = float(calc.kernel())
        if not math.isfinite(energy):
            raise RuntimeError('Nonfinite total energy.')
        result.update(energy=energy, e_corr=float(calc.e_corr), status='completed',
                      pyscf_version=pyscf.__version__,
                      dependency_sources={str(Path(m.__file__).resolve()):
                          hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                          for m in (CL_embed, uobdh_embed)})
    except Exception as exc:
        result.update(status='failed', error=str(exc))
        traceback.print_exc()
    result['runtime_seconds'] = time.monotonic() - started
    write_json(result_path, result)
    return 0 if result['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--xyz', type=Path, default=Path('input.xyz'))
    parser.add_argument('--output', type=Path, default=Path('mp2_hexanol_cl.json'))
    parser.add_argument('--basis', required=True, help='Use the basis of your benchmark.')
    parser.add_argument('--module-file', type=Path, default=Path(__file__).with_name('mp2_embed.py'))
    parser.add_argument('--frozen', type=int, default=0)
    parser.add_argument('--xc-env', default='b3lyp')
    parser.add_argument('--auxbasis', default=None, help='Default: PySCF automatic auxiliary basis.')
    parser.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    parser.add_argument('--max-memory', type=float, default=4000, help='PySCF memory hint in MB, not a hard cap.')
    parser.add_argument('--mu', type=float, default=1e6)
    parser.add_argument('--scf-tol', type=float, default=1e-10, help='Initial UHF only; internal SCFs use upstream defaults.')
    parser.add_argument('--scf-cycles', type=int, default=200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.threads, args.scf_cycles) < 1:
        parser.error('Thread/iteration counts must be positive.')
    if not all(math.isfinite(x) and x > 0 for x in
               [args.max_memory, args.mu, args.scf_tol]):
        parser.error('Memory, thresholds and mu must be finite and positive.')
    if args.frozen < 0:
        parser.error('--frozen must be nonnegative.')
    atom = read_xyz(args.xyz)
    config = {key: value for key, value in vars(args).items()
              if key not in ('xyz', 'output', 'resume', 'dry_run', 'module_file')}
    config['module_file'] = str(args.module_file.resolve())
    config['module_sha256'] = hashlib.sha256(args.module_file.read_bytes()).hexdigest()
    config['atom'] = atom
    config['active_atom_counts'] = COUNTS
    config['shells'] = [None, 1, 2, 3]
    if args.dry_run:
        print(json.dumps(config, indent=2))
        for label, shell in CASES:
            for count, fragment in zip(COUNTS, LABELS):
                print(f'{label:18s} {fragment:8s} active_atoms={list(range(count))}')
        return 0
    output = args.output.resolve()
    if output in (args.xyz.resolve(), args.module_file.resolve(), Path(__file__).resolve()):
        parser.error('Output cannot overwrite the input or script.')
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_suffix('.meta.json')
    log_dir = output.parent / (output.stem + '_logs')
    data = {'active_atom_counts': COUNTS,
            'results': {label: [None] * len(COUNTS) for label, _ in CASES}}
    meta = {'config': config, 'cases': {}}
    if output.exists() or meta_path.exists():
        if not args.resume:
            parser.error('Output already exists. Use --resume or a new --output.')
        if not (output.exists() and meta_path.exists()):
            parser.error('Resume requires both output JSON and its .meta.json sidecar.')
        previous = json.loads(meta_path.read_text())
        if previous['config'] != config:
            parser.error('Resume settings differ from the saved configuration.')
        import pyscf
        from pycmf.OBDH import CL_embed, uobdh_embed
        current_sources = {str(Path(m.__file__).resolve()):
                           hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                           for m in (CL_embed, uobdh_embed)}
        for point in previous['cases'].values():
            if point.get('status') == 'completed' and (
                    point.get('pyscf_version') != pyscf.__version__ or
                    point.get('dependency_sources') != current_sources):
                parser.error('PySCF/embedding/CL code changed. Use a new --output.')
        meta = previous
        # Recover accepted energies from the sidecar, even if interrupted between writes.
        for label, shell in CASES:
            for i, count in enumerate(COUNTS):
                prior = meta['cases'].get(f'{shell or 0}_{count}', {})
                if prior.get('status') == 'completed':
                    data['results'][label][i] = prior['energy']
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(meta_path, meta)
    write_json(output, data)
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = str(args.threads)
    env['PYTHONUNBUFFERED'] = '1'
    if env.get('TMPDIR'):
        Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    for label, shell in CASES:
        for i, count in enumerate(COUNTS):
            key = f'{shell or 0}_{count}'
            if data['results'][label][i] is not None:
                print(f'SKIP {label}, atoms={count}: already completed', flush=True)
                continue
            point_path = log_dir / f'point_{key}.json'
            log_path = log_dir / f'point_{key}.log'
            point_path.unlink(missing_ok=True)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                       str(meta_path), str(count), str(shell or 0), str(point_path)]
            print(f'RUN {label}, atoms={count}; log={log_path}', flush=True)
            with log_path.open('w') as handle:
                proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            result = json.loads(point_path.read_text()) if point_path.exists() else {
                'status': 'failed', 'error': 'Worker exited without a result; inspect log/job stderr.'}
            result.update(returncode=proc.returncode, log=str(log_path))
            accepted = proc.returncode == 0 and result.get('status') == 'completed'
            if not accepted:
                result['status'] = 'failed'
            meta['cases'][key] = result
            if accepted:
                data['results'][label][i] = result['energy']
            write_json(meta_path, meta)
            write_json(output, data)
            if not accepted:
                print(f'FAILED: {result.get("error", "worker error")}; see {log_path}', file=sys.stderr)
                return 1
            print(f'OK {label}, atoms={count}: E_total={result["energy"]:.12f} Eh', flush=True)
    print(f'Completed all 28 points: {output}', flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        sys.exit(worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]))
    sys.exit(main())
