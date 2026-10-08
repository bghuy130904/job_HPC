#!/usr/bin/env python3
"""SIE: read geometries, run standard methods once per point, write results."""
import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
import traceback

from pyscf import dft, gto, lib, mp, scf

METHODS = ('uhf', 'ump2', 'pbe', 'pbe0', 'b3lyp', 'b2plyp',
           'dh_matched', 'obdh', 'obmp2')
FIELDS = ('system', 'point', 'method', 'reference', 'energy_Ha',
          'converged', 'error')
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / 'inputs' / 'input.json'


def run_point(mol, cfg):
    """One standard reference for WFT; one UKS calculation per DFT/DH method."""
    hf = None
    hf_error = None
    if any(name in cfg.methods for name in ('uhf', 'ump2', 'obdh', 'obmp2')):
        try:
            hf = scf.UHF(mol)
            hf.max_cycle = cfg.scf_cycles
            hf.kernel()  # PySCF's default initial guess; no supplied density.
        except Exception as exc:
            hf_error = str(exc)
            traceback.print_exc()
    rows = []
    for name in cfg.methods:
        print(f'\n=== {name} ===', flush=True)
        row = dict(method=name, reference='uhf' if name in
                   ('uhf', 'ump2', 'obdh', 'obmp2') else name,
                   energy_Ha=None, converged=False, error='')
        try:
            if name in ('uhf', 'ump2', 'obdh', 'obmp2'):
                if hf_error is not None:
                    raise RuntimeError(f'UHF reference failed: {hf_error}')
                if name == 'uhf':
                    energy, converged = hf.e_tot, hf.converged
                else:
                    if not hf.converged:
                        raise RuntimeError('UHF reference did not converge')
                    if name == 'ump2':
                        post = mp.UMP2(hf).density_fit()
                        post.kernel()
                        energy, converged = hf.e_tot + post.e_corr, hf.converged
                    else:
                        from pycmf.OBDH import OBDH_CL, OBMP2_CL
                        post = (OBDH_CL if name == 'obdh' else OBMP2_CL)(hf)
                        post.use_embed = False
                        post.use_cl = False
                        post.alphaa = tuple(cfg.alpha)
                        post.niter = cfg.ob_cycles
                        post.thresh = cfg.ob_tol
                        # Set the grid for the UKS constructed inside standard_kernel.
                        with lib.temporary_env(dft.gen_grid.Grids, level=cfg.grid):
                            energy = post.kernel()
                        converged = post.converged
            else:
                ks = dft.UKS(mol)
                ks.grids.level = cfg.grid
                ks.max_cycle = cfg.scf_cycles
                if name in ('b2plyp', 'dh_matched'):
                    ax, ac = (.53, .27) if name == 'b2plyp' else cfg.alpha
                    ks.xc = f'{ax}*HF + {1-ax}*B88, {1-ac}*LYP'
                    row['reference'] = ks.xc
                else:
                    ks.xc = name
                ks.kernel()
                energy, converged = ks.e_tot, ks.converged
                if name in ('b2plyp', 'dh_matched'):
                    if not ks.converged:
                        raise RuntimeError('UKS reference did not converge')
                    # Keep canonical GKS orbitals/epsilon; do not run HF SCF.
                    carrier = ks.to_hf()
                    carrier.converged = True
                    post = mp.UMP2(carrier).density_fit()
                    post.kernel()
                    energy += ac * post.e_corr
            energy = float(energy)
            if not math.isfinite(energy):
                raise RuntimeError('Non-finite energy')
            row.update(energy_Ha=energy, converged=bool(converged))
        except Exception as exc:
            row['error'] = str(exc)
            traceback.print_exc()
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS,
                        default=list(METHODS))
    parser.add_argument('--systems', nargs='+')
    parser.add_argument('--points', nargs='+')
    parser.add_argument('--basis', default='aug-cc-pVDZ')
    parser.add_argument('--charge', type=int, default=1)
    parser.add_argument('--spin', type=int, default=1)
    parser.add_argument('--alpha', nargs=2, type=float, default=[.5, .4],
                        metavar=('HF', 'PT2'))
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-memory', type=float, default=4000,
                        help='PySCF memory budget in MB.')
    parser.add_argument('--grid', type=int, default=4)
    parser.add_argument('--scf-cycles', type=int, default=200)
    parser.add_argument('--ob-cycles', type=int, default=300)
    parser.add_argument('--ob-tol', type=float, default=1e-6)
    cfg = parser.parse_args()
    if any(not math.isfinite(a) or not 0 <= a <= 1 for a in cfg.alpha):
        parser.error('--alpha coefficients must be between 0 and 1')
    data = json.loads(cfg.input.read_text())
    systems = cfg.systems or list(data)
    for system in systems:
        if system not in data:
            parser.error(f'Unknown system: {system}')
        for point in cfg.points or list(data[system]):
            if point not in data[system]:
                parser.error(f'Unknown point for {system}: {point}')
    lib.num_threads(cfg.threads)
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with (cfg.outdir / 'results.csv').open('w', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=FIELDS)
        writer.writeheader()
        for system in systems:
            for point in cfg.points or list(data[system]):
                stem = cfg.outdir / f'{system}__{point}'
                print(f'Running {system} / {point}', flush=True)
                with Path(str(stem) + '.log').open('w') as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        try:
                            mol = gto.Mole()
                            mol.stdout = log
                            mol.build(
                                atom=[(a['element'], a['coordinates'])
                                      for a in data[system][point]],
                                unit='Angstrom', basis=cfg.basis,
                                charge=cfg.charge, spin=cfg.spin,
                                verbose=4, output=None, max_memory=cfg.max_memory,
                            )
                            mol.stdout = log
                            rows = run_point(mol, cfg)
                        except Exception as exc:
                            traceback.print_exc()
                            rows = [dict(method=m, reference='', energy_Ha=None,
                                         converged=False, error=str(exc))
                                    for m in cfg.methods]
                for row in rows:
                    row.update(system=system, point=point)
                    writer.writerow(row)
                    failures += bool(row['error'] or not row['converged'])
                table.flush()
                Path(str(stem) + '.json').write_text(json.dumps(rows, indent=2))
    print(f'Saved {cfg.outdir / "results.csv"}; failed/unconverged: {failures}',
          flush=True)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
