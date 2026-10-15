"""RHF/UHF canonical CCSD(T) CP scan for the shared acetic-acid dimer input."""
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import tempfile
from datetime import datetime

HARTREE_TO_KCAL = 627.509474
REPO_ROOT = Path(__file__).resolve().parents[4]


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def parse_r(value):
    values = json.loads(value) if isinstance(value, str) else value
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        values = [values]
    if not isinstance(values, list) or not values:
        raise ValueError('R must be a number or nonempty JSON list')
    values = [float(r) for r in values]
    if any(not math.isfinite(r) or r <= 0 for r in values):
        raise ValueError('R values must be finite and positive')
    if len(set(values)) != len(values):
        raise ValueError('Duplicate R values would overwrite fragment logs')
    return values


def geometries(config, scale):
    a, b = config['MONOMER_A_RAW'], config['MONOMER_B_RAW']
    ca = [sum(row[i+1] for row in a)/len(a) for i in range(3)]
    cb = [sum(row[i+1] for row in b)/len(b) for i in range(3)]
    shifted = [[row[0], *(row[i+1]+(scale-1)*(cb[i]-ca[i]) for i in range(3))] for row in b]
    def block(rows, ghost=False):
        return '\n'.join(f"{'Ghost-' if ghost else ''}{atom} {x:.9f} {y:.9f} {z:.9f}" for atom,x,y,z in rows)
    return {'AB': block(a)+'\n'+block(shifted),
            'A_CP': block(a)+'\n'+block(shifted, True),
            'B_CP': block(a, True)+'\n'+block(shifted)}


def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f'Non-finite {name}')
    return value


def run_fragment(atom, name, r, settings, run_dir):
    from pyscf import gto, scf, cc
    label = f'{name}_R{r:.17g}'
    log_path = run_dir/'logs'/f'{label}.log'
    record = dict(Method='CCSD(T)', R_scale=r, Fragment=name, Basis=settings['basis'],
                  Charge=0, Spin=0, Reference=settings['reference'].upper(), Frozen=settings['frozen'], Density_fitting=False,
                  Status='FAILED', Stage='BUILD', HF_converged='', HF_internal_stable='',
                  CCSD_converged='', Triples_completed=False, Log=str(log_path))
    start = time.monotonic()
    mol = mf = mycc = None
    # Geometry is retained independently of a successful electronic calculation.
    (run_dir/'geometries'/f'{label}.xyz').write_text(f"{len(atom.splitlines())}\n{name}; R/R0={r}; ghost labels preserved\n{atom}\n")
    try:
        with log_path.open('w', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                mol = gto.Mole()
                mol.atom = atom
                mol.basis = settings['basis']
                mol.unit = 'Angstrom'
                mol.charge = mol.spin = 0
                mol.max_memory = settings['memory']
                mol.verbose = 4
                mol.stdout = log
                mol.incore_anyway = False
                mol.build()
                record.update(Nelectron=mol.nelectron, NAO=mol.nao_nr())
                record['Stage'] = 'HF'
                mf = scf.UHF(mol) if settings['reference']=='uhf' else scf.RHF(mol)
                mf.conv_tol = settings['hf_tol']
                mf.max_cycle = settings['hf_cycles']
                mf.max_memory = settings['memory']
                # Keep HDF5 checkpoint locks off the NFS result filesystem.
                checkpoint_root = Path(os.environ.get('PYSCF_SCRATCH') or
                                       os.environ.get('TMPDIR') or tempfile.gettempdir())
                checkpoint_dir = checkpoint_root/'dimmer_cl_ccsdt'/run_dir.name/'checkpoints'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                mf.chkfile = str(checkpoint_dir/f'{label}.chk')
                record['HF_checkpoint'] = mf.chkfile
                print(f'HF checkpoint: {mf.chkfile}', flush=True)
                mf.kernel()
                record['HF_converged'] = bool(mf.converged)
                record['E_HF_Eh'] = finite(mf.e_tot, 'HF energy')
                record['HF_S2'] = float(mf.spin_square()[0])
                if not mf.converged:
                    raise RuntimeError('HF not converged')
                record['Stage'] = 'HF_STABILITY'
                record['E_HF_initial_Eh'] = record['E_HF_Eh']
                record['HF_initial_S2'] = record['HF_S2']
                stable = False
                for macro in range(settings['stability_cycles'] + 1):
                    mo, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
                    record['HF_stability_cycles'] = macro
                    record['HF_internal_stable'] = bool(stable)
                    if stable or macro == settings['stability_cycles']:
                        break
                    mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))
                    record['HF_converged'] = bool(mf.converged)
                    if not mf.converged:
                        raise RuntimeError('HF not converged during stability optimization')
                record['E_HF_Eh'] = finite(mf.e_tot, 'stable HF energy')
                record['HF_S2'] = float(mf.spin_square()[0])
                if not stable:
                    raise RuntimeError('HF internal unstable after stability optimization')
                if settings['frozen'] >= mol.nelectron//2:
                    raise ValueError('FROZEN must leave at least one correlated occupied orbital')
                record['Stage'] = 'CCSD'
                mycc = cc.CCSD(mf, frozen=settings['frozen'])
                mycc.conv_tol = settings['cc_tol']
                mycc.conv_tol_normt = settings['cc_normt']
                mycc.max_cycle = settings['cc_cycles']
                mycc.max_memory = settings['memory']
                mycc.incore_complete = False
                mycc.kernel()
                record['CCSD_converged'] = bool(mycc.converged)
                record['E_corr_CCSD_Eh'] = finite(mycc.e_corr, 'CCSD correlation energy')
                record['E_CCSD_Eh'] = finite(mycc.e_tot, 'CCSD total energy')
                if not mycc.converged:
                    raise RuntimeError('CCSD not converged; triples skipped')
                if abs(mycc.e_tot - (mf.e_tot + mycc.e_corr)) > 1e-9:
                    raise RuntimeError('CCSD total energy does not equal HF + correlation')
                record['Stage'] = 'TRIPLES'
                triples = finite(mycc.ccsd_t(), '(T) correction')
                total = finite(mycc.e_tot + triples, 'CCSD(T) total energy')
                record.update(E_T_Eh=triples, E_total_Eh=total, Triples_completed=True,
                              Status='OK', Stage='DONE')
                print(f'E_HF={mf.e_tot:.12f}; E_corr_CCSD={mycc.e_corr:.12f}; E_T={triples:.12f}')
                print(f'E_CCSD(T)={total:.12f} Eh')
    except Exception as exc:
        record['Status'] = 'Not convergence' if any(t in str(exc).lower() for t in ('not converg', 'unstable')) else 'FAILED'
        record['Error'] = f'{type(exc).__name__}: {exc}'
    finally:
        record['Runtime_s'] = time.monotonic()-start
        if mol is not None:
            mol.stdout = None
        del mycc, mf, mol
        gc.collect()
    return record


def interaction_row(r, records, settings):
    row = dict(Method='CCSD(T)', Reference=settings['reference'].upper(), Basis=settings['basis'], Frozen=settings['frozen'],
               Density_fitting=False, R_scale=r)
    for rec in records:
        f = rec['Fragment']
        row[f'Status_{f}'] = rec['Status']
        # Keep driver-compatible names; these are full CCSD(T), not embedded energies.
        row[f'E_{f}_Embed_Eh'] = rec.get('E_total_Eh', '') if rec['Status']=='OK' else ''
    row['Status'] = 'OK' if all(rec['Status']=='OK' for rec in records) else (
        'Not convergence' if any(rec['Status']=='Not convergence' for rec in records) else 'FAILED')
    row['E_int_Eh'] = row['E_int_kcal_mol'] = ''
    if row['Status']=='OK':
        for component, output in [('E_HF_Eh','E_int_HF_Eh'), ('E_corr_CCSD_Eh','E_int_corr_CCSD_Eh'), ('E_T_Eh','E_int_T_Eh')]:
            row[output] = records[0][component]-records[1][component]-records[2][component]
        total = row['E_AB_Embed_Eh']-row['E_A_CP_Embed_Eh']-row['E_B_CP_Embed_Eh']
        if abs(total-sum(row[k] for k in ('E_int_HF_Eh','E_int_corr_CCSD_Eh','E_int_T_Eh'))) > 1e-9:
            raise RuntimeError('Interaction energy component sum is inconsistent')
        row['E_int_Eh'] = total
        row['E_int_kcal_mol'] = total*HARTREE_TO_KCAL
    return row


def main():
    import pyscf
    from pyscf import lib
    input_path = Path(os.environ.get('GEOMETRY_FILE', str(Path(__file__).resolve().parents[2]/'inputs/aad.json')))
    config = json.loads(input_path.read_text())
    values = parse_r(os.environ.get('R', config['R_VALUES']))
    settings = dict(reference=os.environ.get('REFERENCE','uhf').lower(), basis=os.environ.get('BASIS','def2-QZVPPD'), memory=int(os.environ.get('MEMORY','90000')),
                    frozen=int(os.environ.get('FROZEN','0')), hf_tol=float(os.environ.get('HF_TOL','1e-10')),
                    hf_cycles=int(os.environ.get('HF_MAX_CYCLE','200')), stability_cycles=int(os.environ.get('HF_STABILITY_CYCLES','10')), cc_tol=float(os.environ.get('CC_TOL','1e-8')),
                    cc_normt=float(os.environ.get('CC_NORMT','1e-6')), cc_cycles=int(os.environ.get('CC_MAX_CYCLE','100')))
    if settings['reference'] not in ('uhf','rhf'):
        raise ValueError('REFERENCE must be uhf or rhf')
    if settings['stability_cycles'] < 0 or settings['hf_cycles'] <= 0 or settings['cc_cycles'] <= 0:
        raise ValueError('Iteration limits must be positive (stability cycles may be 0)')
    if any(not math.isfinite(settings[k]) or settings[k] <= 0 for k in ('hf_tol','cc_tol','cc_normt')):
        raise ValueError('Convergence tolerances must be finite and positive')
    if settings['frozen'] < 0 or settings['memory'] <= 0:
        raise ValueError('FROZEN must be nonnegative and MEMORY positive')
    threads=int(os.environ.get('OMP_NUM_THREADS','1'))
    lib.num_threads(threads)
    scratch=os.environ.get('PYSCF_SCRATCH') or os.environ.get('TMPDIR')
    if scratch:
        Path(scratch).mkdir(parents=True, exist_ok=True)
        lib.param.TMPDIR=scratch
    run_id=os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir=Path(os.environ.get('RUN_DIR',str(REPO_ROOT/'results/dimmer_cl/ccsdt'/run_id)))
    # Refuse to silently overwrite an earlier calculation.
    if (run_dir/'metadata.json').exists():
        raise FileExistsError(f'Existing run metadata: {run_dir}; choose a new RUN_DIR')
    for name in ('logs','geometries'):
        (run_dir/name).mkdir(parents=True,exist_ok=True)
    metadata=dict(Method='CCSD(T)',Reference=settings['reference'].upper(),Charge=0,Spin=0,Density_fitting=False,
                  PySCF_version=pyscf.__version__,Input_file=str(input_path.resolve()),
                  Input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  Geometry=config,R_VALUES=values,Settings=settings,Threads=threads,
                  Counterpoise=True,R_definition='dimensionless scale R/R0; geometric-center translation',
                  Total_energy='E_HF + E_corr_CCSD + E_T',Started=datetime.now().isoformat())
    (run_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    all_records,rows=[],[]
    for i,r in enumerate(values,1):
        start=time.monotonic()
        records=[]
        for name,atom in geometries(config,r).items():
            print(f'Running {name}, R/R0={r}',flush=True)
            rec=run_fragment(atom,name,r,settings,run_dir)
            records.append(rec);all_records.append(rec)
            write_csv(run_dir/'convergence.csv',all_records)
            print(f"{name}: {rec['Status']}; {rec.get('Error','')}",flush=True)
        row=interaction_row(r,records,settings)
        row.update(Point=i,Runtime_Point_s=time.monotonic()-start)
        rows.append(row)
        write_csv(run_dir/'interaction_energies.csv',rows)
        print(f"R/R0={r}: E_int={row['E_int_kcal_mol']} kcal/mol; {row['Status']}",flush=True)
    # Finishes the scan but reports incomplete reference data to the submit script.
    return 0 if all(row['Status']=='OK' for row in rows) else 1


if __name__=='__main__':
    raise SystemExit(main())
