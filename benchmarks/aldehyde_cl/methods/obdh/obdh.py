#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

COUNTS = [3, 7, 11, 15, 19, 23, 27]
LABELS = ['CHO', 'CHO+(CH=CH)1', 'CHO+(CH=CH)2', 'CHO+(CH=CH)3',
          'CHO+(CH=CH)4', 'CHO+(CH=CH)5', 'C12H14O (all active)']
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
    if len(lines) < 2 or int(lines[0]) != 27:
        raise ValueError('Expected the supplied 27-atom C12 aldehyde XYZ.')
    rows = [line.split() for line in lines[2:] if line.strip()]
    expected = ['O', 'C', 'H'] + ['C', 'H'] * 10 + ['C', 'H', 'H', 'H']
    if len(rows) != 27 or [row[0] for row in rows] != expected:
        raise ValueError('XYZ atom order must match the supplied CHO-first aldehyde.xyz.')
    for row in rows:
        if len(row) != 4 or not all(math.isfinite(float(x)) for x in row[1:]):
            raise ValueError(f'Invalid XYZ row: {row}')
    return '\n'.join(' '.join(row) for row in rows)


def worker(config_path, count, shell, result_path):
    config = json.loads(Path(config_path).read_text())['config']
    started = time.monotonic()
    result = {'active_atom_count': count, 'n_shells': shell or None}
    if count not in config['active_atom_counts'] or (shell or None) not in config['shells']:
        raise ValueError('Worker region/shell is not part of the saved scan configuration.')
    try:
        # Import only after the parent sets thread variables and TMPDIR exists.
        from pyscf import gto, scf, lib
        import pyscf
        from pycmf.OBDH import OBDH_CL
        from pycmf.OBDH import CL_embed, main as obdh_main
        lib.num_threads(config['threads'])
        if not callable(CL_embed.concentric_localization):
            raise RuntimeError('CL implementation is unavailable.')
        mol = gto.M(
            atom=config['atom'],
            unit='Angstrom',
            basis=config['basis'],
            charge=config['charge'],
            spin=config['spin'],
            verbose=4,
            max_memory=config['max_memory'],
        )
        mf = scf.UHF(mol).density_fit(auxbasis=config['auxbasis'])
        mf.conv_tol = config['scf_tol']
        mf.max_cycle = config['scf_cycles']
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('Initial full-system UHF did not converge.')
        calc = OBDH_CL(mf)
        calc.alphaa = tuple(config['alpha'])
        calc.xc_env = config['xc_env']
        calc.use_embed = True
        calc.active_atoms = list(range(count))
        calc.use_cl = shell != 0
        calc.n_shells = shell if shell else 1
        calc.mu = config['mu']
        calc.thresh = config['threshold']
        calc.niter = config['niter']
        calc.second_order = True
        calc.eval_IPEA = False
        calc.mom_select = False
        calc.max_memory = config['max_memory']
        print(f'\nCASE active_atoms={calc.active_atoms}, use_cl={calc.use_cl}, '
              f'n_shells={calc.n_shells}', flush=True)
        energy = float(calc.kernel())  # Embedded TOTAL energy, not e_corr.
        if not math.isfinite(energy):
            raise RuntimeError('Nonfinite total energy.')
        result.update(energy=energy, converged=bool(calc.converged),
                      pyscf_version=pyscf.__version__, source=str(obdh_main.__file__),
                      source_sha256=hashlib.sha256(Path(obdh_main.__file__).read_bytes()).hexdigest(),
                      nocc=[int(x) for x in calc.get_nocc()],
                      nmo=[int(x) for x in calc.get_nmo()])
        if not calc.converged:
            raise RuntimeError('OBDH solver did not converge; energy not accepted.')
        result['status'] = 'converged'
    except Exception as exc:
        result.update(status='failed', error=str(exc))
        traceback.print_exc()
    result['runtime_seconds'] = time.monotonic() - started
    write_json(result_path, result)
    return 0 if result['status'] == 'converged' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--xyz', type=Path, default=Path(__file__).resolve().parents[2] / 'inputs' / 'aldehyde.xyz')
    parser.add_argument('--output', type=Path, default=Path('obdh_aldehyde_cl_new.json'))
    parser.add_argument('--basis', required=True, help='Use the basis of your benchmark.')
    parser.add_argument('--charge', type=int, default=0)
    parser.add_argument('--spin', type=int, default=0,
                        help='N_alpha - N_beta; doublet: 1, triplet: 2')
    parser.add_argument('--alpha', type=float, nargs=2, required=True, metavar=('AX', 'AC'))
    parser.add_argument('--xc-env', default=None, help='Default: AX*HF+(1-AX)*B88,(1-AC)*LYP. '
                        'Current upstream embedding also passes this XC to the subsystem solver.')
    parser.add_argument('--auxbasis', default=None, help='Default: PySCF automatic auxiliary basis.')
    parser.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    parser.add_argument('--max-memory', type=float, default=4000, help='PySCF memory hint in MB, not a hard cap.')
    parser.add_argument('--threshold', type=float, default=1e-8)
    parser.add_argument('--niter', type=int, default=300)
    parser.add_argument('--mu', type=float, default=1e6)
    parser.add_argument('--scf-tol', type=float, default=1e-10, help='Initial UHF only; internal SCFs use upstream defaults.')
    parser.add_argument('--scf-cycles', type=int, default=200)
    parser.add_argument('--shells', type=int, nargs='+', choices=(0, 1, 2, 3),
                        default=[0, 1, 2, 3], help='0 = No CL; 1/2/3 = CL shell count.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if len(set(args.shells)) != len(args.shells):
        parser.error('--shells must not contain duplicates.')
    cases = [(label, shell) for label, shell in CASES if (shell or 0) in args.shells]
    if min(args.threads, args.niter, args.scf_cycles) < 1:
        parser.error('Thread/iteration counts must be positive.')
    if not all(math.isfinite(x) and x > 0 for x in
               [args.max_memory, args.threshold, args.mu, args.scf_tol]):
        parser.error('Memory, thresholds and mu must be finite and positive.')
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in args.alpha):
        parser.error('AX and AC must be between 0 and 1.')
    atom = read_xyz(args.xyz)
    config = {key: value for key, value in vars(args).items()
              if key not in ('xyz', 'output', 'resume', 'dry_run')}
    config['atom'] = atom
    config['active_atom_counts'] = COUNTS
    config['shells'] = [shell for _, shell in cases]
    if args.dry_run:
        print(json.dumps(config, indent=2))
        for label, shell in cases:
            for count, fragment in zip(COUNTS, LABELS):
                print(f'{label:18s} {fragment:8s} active_atoms={list(range(count))}')
        return 0
    output = args.output.resolve()
    if output in (args.xyz.resolve(), Path(__file__).resolve()):
        parser.error('Output cannot overwrite the input or script.')
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_suffix('.meta.json')
    log_dir = output.parent / (output.stem + '_logs')
    data = {'active_atom_counts': COUNTS,
            'results': {label: [None] * len(COUNTS) for label, _ in cases}}
    meta = {'config': config, 'cases': {}}
    if output.exists() or meta_path.exists():
        if not args.resume:
            parser.error('Output already exists. Use --resume or a new --output.')
        if not (output.exists() and meta_path.exists()):
            parser.error('Resume requires both output JSON and its .meta.json sidecar.')
        previous = json.loads(meta_path.read_text())
        if previous['config'] != config:
            parser.error('Resume settings differ from the saved configuration.')
        meta = previous
        # Recover accepted energies from the sidecar, even if interrupted between writes.
        for label, shell in cases:
            for i, count in enumerate(COUNTS):
                prior = meta['cases'].get(f'{shell or 0}_{count}', {})
                if prior.get('status') == 'converged':
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
    for label, shell in cases:
        for i, count in enumerate(COUNTS):
            key = f'{shell or 0}_{count}'
            if data['results'][label][i] is not None:
                print(f'SKIP {label}, atoms={count}: already converged', flush=True)
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
            accepted = proc.returncode == 0 and result.get('status') == 'converged'
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
    print(f'Completed all {len(COUNTS) * len(cases)} points: {output}', flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        sys.exit(worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]))
    sys.exit(main())

