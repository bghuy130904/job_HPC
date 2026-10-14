#!/usr/bin/env python3
"""Rigid C2-C1-O-H torsion scan, retaining all atoms and the full AO basis."""
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

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
KCAL = 627.5094740631


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def read_xyz(path):
    rows = Path(path).read_text().splitlines()
    atoms = [line.split('#', 1)[0].split() for line in rows[2:] if line.strip()]
    if len(atoms) != int(rows[0]) or any(len(row) != 4 for row in atoms):
        raise ValueError('XYZ atom count or row format is invalid.')
    symbols = [row[0] for row in atoms]
    coords = [[float(x) for x in row[1:]] for row in atoms]
    if not all(math.isfinite(x) for row in coords for x in row):
        raise ValueError('Nonfinite coordinates.')
    return symbols, coords


def dihedral(coords, indices):
    import numpy as np
    p0, p1, p2, p3 = np.asarray(coords)[indices]
    axis = p2 - p1
    axis /= np.linalg.norm(axis)
    v = p0 - p1
    w = p3 - p2
    v -= np.dot(v, axis) * axis
    w -= np.dot(w, axis) * axis
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w))))


def rotate_oh(coords, indices, angle):
    import numpy as np
    coords = np.array(coords, dtype=float, copy=True)
    _, c, o, h = indices
    axis = coords[o] - coords[c]
    axis /= np.linalg.norm(axis)
    # Set the actual dihedral, rather than rotating by an offset from the XYZ.
    theta = np.radians(angle - dihedral(coords, indices))
    v = coords[h] - coords[o]
    coords[h] = coords[o] + (v*np.cos(theta) + np.cross(axis, v)*np.sin(theta)
                            + axis*np.dot(axis, v)*(1-np.cos(theta)))
    actual = dihedral(coords, indices)
    if abs((actual-angle+180) % 360-180) > 1e-8:
        raise RuntimeError('Torsion construction failed.')
    return coords


def worker(config_path, phi, ns, point_path):
    config = json.loads(Path(config_path).read_text())['config']
    result = {'phi': phi, 'n_shells': ns, 'shell_label': ns-1}
    start = time.monotonic()
    try:
        import numpy as np
        import pyscf
        from pyscf import gto, scf, lib
        from pycmf.OBDH import OBDH_CL, main as ob_main, uobdh_embed, uobdh_solver, CL_embed
        lib.num_threads(config['threads'])
        scf_records = []
        original_kernel = scf.hf.SCF.kernel

        def checked_kernel(mf, *args, **kwargs):
            # Apply user SCF controls to upstream full-system and embedded SCFs.
            mf.max_cycle = config['scf_cycles']
            mf.conv_tol = config['scf_tol']
            energy = original_kernel(mf, *args, **kwargs)
            scf_records.append({'class': type(mf).__name__, 'converged': bool(mf.converged),
                                'energy_hartree': float(mf.e_tot)})
            if not mf.converged:
                raise RuntimeError(f'{type(mf).__name__} SCF did not converge.')
            return energy

        scf.hf.SCF.kernel = checked_kernel
        coords = rotate_oh(config['coords'], config['dihedral'], phi)
        mol = gto.M(atom=list(zip(config['symbols'], coords)), unit='Angstrom',
                    basis=config['basis'], charge=0, spin=0, verbose=4,
                    max_memory=config['max_memory'])
        mf = scf.UHF(mol).density_fit()
        mf.kernel()
        if config['method'] == 'obdh':
            calc = OBDH_CL(mf)
            calc.alphaa = tuple(config['alpha'])
            calc.niter = config['ob_cycles']
            calc.thresh = config['ob_tol']
            calc.second_order = True
            calc.eval_IPEA = False
            calc.mom_select = False
        else:
            sys.path.insert(0, str(REPO / 'shared'))
            from mp2_embed import UMP2_CL
            calc = UMP2_CL(mf, frozen=0)
        calc.xc_env = config['xc_env']
        calc.use_embed = True
        calc.use_cl = True
        calc.active_atoms = config['active_atoms']
        calc.n_shells = ns
        calc.mu = config['mu']
        calc.max_memory = config['max_memory']
        print(f'CASE phi={phi}, n_shells={ns}, active={calc.active_atoms}', flush=True)
        energy = float(calc.kernel())
        if not math.isfinite(energy):
            raise RuntimeError('Nonfinite energy.')
        if config['method'] == 'obdh' and not calc.converged:
            raise RuntimeError('OBDH did not converge; energy not accepted.')
        sources = [ob_main, uobdh_embed, uobdh_solver, CL_embed]
        if config['method'] == 'mp2':
            import mp2_embed
            sources.append(mp2_embed)
        result.update(status='converged', energy_hartree=energy,
                      dihedral_degrees=dihedral(coords, config['dihedral']),
                      pyscf_version=pyscf.__version__, scfs=scf_records,
                      source_sha256={m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                     for m in sources})
        if config['method'] == 'obdh':
            result['nocc'] = [int(x) for x in calc.get_nocc()]
            result['nmo'] = [int(x) for x in calc.get_nmo()]
    except Exception as exc:
        result.update(status='failed', error=str(exc))
        traceback.print_exc()
    result['runtime_seconds'] = time.monotonic()-start
    write_json(point_path, result)
    return 0 if result['status'] == 'converged' else 1


def export_pes(config, meta, out, ns):
    # Legacy JSONs use the minimum on each scan as their energy zero.
    accepted = [pt for pt in meta['points'].values()
                if pt.get('n_shells') == ns and pt.get('status') == 'converged']
    ref = min(accepted, key=lambda pt: pt['energy_hartree']) if accepted else {}
    e_ref = ref.get('energy_hartree')
    scan = []
    for angle in config['angles']:
        pt = meta['points'].get(f'ns{ns}_phi{angle:g}',{})
        energy = pt.get('energy_hartree') if pt.get('status') == 'converged' else None
        scan.append(dict(phi=angle,energy_hartree=energy,
                         e_rel_kcal=(energy-e_ref)*KCAL if energy is not None and e_ref is not None else None,
                         status=pt.get('status','pending')))
    suffix = '_mp2' if config['method'] == 'mp2' else ''
    write_json(out/f'{config["molecule"]}{suffix}_pes_ns{ns}.json',dict(
        molecule=config['label'],method=config['method'],basis=config['basis'],n_shells=ns,
        e_ref_hartree=e_ref,reference_angle=ref.get('phi'),config=config,scan=scan))

def main(method=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'inputs/scan.json')
    p.add_argument('--method', choices=['obdh','mp2'], default=method, required=method is None)
    p.add_argument('--molecule', choices=['ethanol','1-butanol'], required=True)
    p.add_argument('--basis')
    p.add_argument('--alpha', type=float, nargs=2)
    p.add_argument('--xc-env', help='Environment and upstream OBDH subsystem XC. Default matched DH-BLYP.')
    p.add_argument('--angles', type=float, nargs='+')
    p.add_argument('--shells', type=int, nargs='+', choices=[1,2,3])
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--max-memory', type=float, default=4000)
    p.add_argument('--scf-cycles', type=int, default=200)
    p.add_argument('--scf-tol', type=float, default=1e-10)
    p.add_argument('--ob-cycles', type=int, default=300)
    p.add_argument('--ob-tol', type=float, default=1e-8)
    p.add_argument('--mu', type=float, default=1e6)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    spec = json.loads(a.config.read_text())
    molecule = spec['molecules'][a.molecule]
    symbols, coords = read_xyz(a.config.parent/molecule['xyz'])
    alpha = a.alpha or spec['alpha']
    xc = a.xc_env or (f'{alpha[0]}*HF+{1-alpha[0]}*B88,{1-alpha[1]}*LYP' if a.alpha else spec['xc_env'])
    angles = sorted(set(a.angles or spec['angles']) | {180.0})
    shells = sorted(set(a.shells or spec['n_shells']))
    if any(not math.isfinite(x) or not 0 <= x <= 180 for x in angles):
        p.error('Angles must be finite and in [0,180].')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in alpha):
        p.error('Alpha must lie in [0,1].')
    if min(a.threads,a.scf_cycles,a.ob_cycles) < 1 or any(not math.isfinite(x) or x<=0 for x in
            [a.scf_tol,a.ob_tol,a.max_memory,a.mu]):
        p.error('Counts, tolerances, memory and mu must be positive.')
    config = dict(method=a.method, molecule=a.molecule, label=molecule['label'],
                  basis=a.basis or spec['basis'], alpha=alpha, xc_env=xc,
                  symbols=symbols, coords=coords, dihedral=molecule['dihedral'],
                  active_atoms=molecule['active_atoms'], angles=angles, shells=shells,
                  threads=a.threads,max_memory=a.max_memory,scf_cycles=a.scf_cycles,
                  scf_tol=a.scf_tol,ob_cycles=a.ob_cycles,ob_tol=a.ob_tol,mu=a.mu)
    if a.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    out = a.output_dir.resolve()
    if out == (ROOT/'references').resolve():
        p.error('Output directory cannot overwrite legacy references.')
    out.mkdir(parents=True,exist_ok=True)
    meta_path = out/f'{a.molecule}_{a.method}.meta.json'
    meta = {'config': config, 'points': {}}
    if meta_path.exists():
        if not a.resume:
            p.error('Run exists; use --resume or another output directory.')
        meta = json.loads(meta_path.read_text())
        if meta['config'] != config:
            p.error('Resume configuration differs from the saved run.')
    if a.resume:
        import importlib
        for point in meta['points'].values():
            if point.get('status') != 'converged':
                continue
            import pyscf
            if point.get('pyscf_version') != pyscf.__version__:
                p.error('PySCF version changed; use a new output directory.')
            sys.path.insert(0,str(REPO/'shared'))
            for name, digest in point.get('source_sha256',{}).items():
                module = importlib.import_module(name)
                if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
                    p.error(f'Source changed: {name}; use a new output directory.')
    write_json(meta_path,meta)
    logs = out/'logs'
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        env[name] = str(a.threads)
    failures = 0
    for ns in shells:
        for phi in angles:
            key = f'ns{ns}_phi{phi:g}'
            if meta['points'].get(key,{}).get('status') == 'converged':
                continue
            log_path = logs/f'{a.molecule}_{a.method}_{key}.log'
            point_path = log_path.with_suffix('.json')
            point_path.unlink(missing_ok=True)
            print(f'RUN {a.molecule} {a.method} {key}: {log_path}',flush=True)
            with log_path.open('w') as log:
                proc = subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),
                                       '--worker',str(meta_path),str(phi),str(ns),str(point_path)],
                                      stdout=log,stderr=subprocess.STDOUT,env=env)
            result = json.loads(point_path.read_text()) if point_path.exists() else dict(
                status='failed',error=f'Worker exited {proc.returncode} without a result.')
            if proc.returncode != 0:
                result['status'] = 'failed'
            result['returncode'] = proc.returncode
            meta['points'][key] = result
            failures += result['status'] != 'converged'
            write_json(meta_path,meta)
            export_pes(config,meta,out,ns)
            print(f'{result["status"]}: {result.get("energy_hartree",result.get("error"))}',flush=True)
    for ns in shells:
        export_pes(config,meta,out,ns)
    return 1 if failures else 0


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        sys.exit(worker(sys.argv[2],float(sys.argv[3]),int(sys.argv[4]),sys.argv[5]))
    sys.exit(main())
