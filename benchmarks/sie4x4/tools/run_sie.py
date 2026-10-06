#!/usr/bin/env python3
"""SIE4x4: stable unrestricted SCF seeds -> all distinct OB basins -> diagnostics.
No spin-contamination rejection. Dissociation energies use supermolecules.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import pyscf
from pyscf import df, dft, gto, lib, mp, scf

HERE = Path(__file__).resolve().parent.parent
POINTS = ['R_1.0', 'R_1.25', 'R_1.5', 'R_1.75']
DL = 'dissociation_limit'
REF = dict(zip(['H2_plus_He', 'He2_plus', 'NH3_2_plus', 'H2O_2_plus'],
               [[64.4,58.9,48.7,38.3], [56.9,46.9,31.3,19.1],
                [35.9,25.9,13.4,4.9], [39.7,29.1,16.9,9.3]]))
KCAL = 627.5094740631


def fragments(system, atoms):
    # Chemical identity, independent of separation and orientation.
    if system == 'H2_plus_He':
        return [[0], [1], [2]]
    if system == 'He2_plus':
        return [[0], [1]]
    heavy = 'N' if system == 'NH3_2_plus' else 'O'
    centers = [i for i,a in enumerate(atoms) if a['element'] == heavy]
    if len(centers) != 2:
        raise ValueError('Expected two molecular fragments')
    xyz = np.array([a['coordinates'] for a in atoms])
    groups = [[centers[0]], [centers[1]]]
    for i in range(len(atoms)):
        if i not in centers:
            groups[int(np.argmin(np.linalg.norm(xyz[centers]-xyz[i], axis=1)))].append(i)
    return [sorted(g) for g in groups]


def build(atoms, basis, charge=1, spin=1):
    return gto.M(atom=[(a['element'], a['coordinates']) for a in atoms],
                 unit='Angstrom', basis=basis, charge=charge, spin=spin,
                 symmetry=False, verbose=0)


def factory(mol, source, cfg):
    if source == 'uhf':
        mf = scf.UHF(mol).density_fit(auxbasis=cfg.scf_aux)
    else:
        mf = dft.UKS(mol).density_fit(auxbasis=cfg.scf_aux)
        mf.xc = source
        mf.grids.level = cfg.grid
    mf.conv_tol = cfg.scf_tol
    mf.conv_tol_grad = cfg.scf_grad_tol
    mf.max_cycle = cfg.scf_cycles
    # PySCF extra-cycle defaults permit an energy OR gradient test. Require both.
    mf.check_convergence = lambda env: (abs(env['e_tot']-env['last_hf_e']) <= cfg.scf_tol
                                         and env['norm_gorb'] <= cfg.scf_grad_tol)
    mf.verbose = 4
    return mf


def stabilize(mf, dm, cfg):
    mf.kernel(dm0=dm)
    if not mf.converged:
        # Temporary level shift regularizes near-degenerate orbital updates.
        # Remove it and reconverge before stability/selection.
        mf.level_shift = .2
        mf.kernel(dm0=mf.make_rdm1())
        mf.level_shift = 0
        mf.kernel(dm0=mf.make_rdm1())
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(dm0=mf.make_rdm1())
    stable = False
    for attempt in range(cfg.stability_cycles + 1):
        if not mf.converged:
            break
        orbitals, _, stable, _ = mf.stability(internal=True, external=False,
                                             return_status=True)
        if stable or attempt == cfg.stability_cycles:
            break
        mf.kernel(dm0=mf.make_rdm1(orbitals, mf.mo_occ))
        if not mf.converged:
            mf = mf.newton()
            mf.kernel(dm0=mf.make_rdm1())
    return mf, bool(mf.converged and stable), bool(stable)


def localized_density(mol, atoms, groups, side, source, cfg):
    """Preserve ALL AO blocks within each fragment, including different atoms."""
    dm = np.zeros((2, mol.nao_nr(), mol.nao_nr()))
    slices = mol.aoslice_by_atom()
    for n, group in enumerate(groups):
        charge = int(n == side)
        nuclear = sum(int(gto.charge(atoms[i]['element'])) for i in group)
        spin = (nuclear-charge) % 2
        frag = build([atoms[i] for i in group], cfg.basis, charge, spin)
        fm = factory(frag, source, cfg)
        fm.verbose = 0
        fm.kernel()
        if not fm.converged:
            fm = fm.newton().run()
        if not fm.converged:
            raise RuntimeError('Fragment SCF did not converge')
        indices = np.concatenate([np.arange(slices[i,2], slices[i,3]) for i in group])
        block = np.asarray(fm.make_rdm1())
        for sigma in range(2):
            dm[sigma][np.ix_(indices,indices)] = block[sigma]
    return dm


def diagnostics(mol, dm, groups, coeff=None, occ=None):
    dm = np.asarray(dm)
    overlap = mol.intor_symmetric('int1e_ovlp')
    populations = np.einsum('sij,ji->si', dm, overlap).real
    slices = mol.aoslice_by_atom()
    atom_pop = np.array([populations[:,p:q].sum(axis=1) for _,_,p,q in slices])
    spin = [float((atom_pop[g,0]-atom_pop[g,1]).sum()) for g in groups]
    charge = [float(sum(mol.atom_charge(i) for i in g)-atom_pop[g].sum()) for g in groups]
    # Descriptive Mulliken thresholds only. Never use these labels for validity.
    frac = abs(spin[0]-spin[1]) / max(abs(spin[0])+abs(spin[1]), 1e-12)
    label = 'loc' if frac >= .8 else ('deloc' if frac <= .2 else 'mixed')
    s2 = None
    if coeff is not None:
        occupied = [np.asarray(coeff[s])[:,np.asarray(occ[s])>0] for s in range(2)]
        s2 = float(scf.uhf.spin_square(occupied, overlap)[0])
    ne = populations.sum(axis=1)
    sane = bool(np.isfinite(dm).all() and
                np.max(np.abs(dm-dm.conj().transpose(0,2,1))) < 1e-7 and
                np.max(np.abs(ne-np.array(mol.nelec))) < 1e-5)
    return dict(s2=s2, fragment_spin=spin, fragment_charge=charge, label=label,
                localization_fraction=float(frac), electrons=ne.tolist(), density_valid=sane)


def density_distance(a, b, overlap):
    # Frobenius distance in the orthonormal AO representation.
    eig, v = np.linalg.eigh(overlap)
    half = (v*np.sqrt(np.maximum(eig,0)))@v.T
    return float(np.linalg.norm(np.asarray([half@x@half for x in (a-b)])))


def seed_pool(mol, atoms, groups, source, cfg, rows):
    pool = []
    names = ['minao', 'atom', 'huckel', 'localized_A', 'localized_B']
    for name in names:
        row = dict(stage='scf', source=source, guess=name, valid=False)
        try:
            mf = factory(mol, source, cfg)
            dm = (localized_density(mol, atoms, groups, int(name.endswith('B')), source, cfg)
                  if name.startswith('localized') else mf.get_init_guess(key=name))
            mf, valid, stable = stabilize(mf, dm, cfg)
            final_dm = np.asarray(mf.make_rdm1())
            row.update(energy=float(mf.e_tot), converged=bool(mf.converged),
                       orbital_gradient=float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ,mf.get_fock(dm=final_dm,level_shift_factor=0)))),
                       stable=stable, **diagnostics(mol, final_dm, groups, mf.mo_coeff, mf.mo_occ))
            row['valid'] = bool(valid and row['density_valid'] and np.isfinite(mf.e_tot) and row['orbital_gradient'] <= cfg.scf_grad_tol)
            duplicate = next((old for old in pool if density_distance(final_dm,old[2],mf.get_ovlp())
                               < cfg.density_tol), None)
            row['duplicate_of'] = duplicate[0] if duplicate else None
            if row['valid'] and duplicate is None:
                pool.append((name,mf,final_dm))
        except Exception as exc:
            row['exception'] = str(exc)
            traceback.print_exc()
        rows.append(row)
    return pool


def ob_candidate(mol, groups, seed, source, method, cfg):
    from pycmf.OBDH import OBDH_CL, OBMP2_CL
    name, mf, _ = seed
    row = dict(stage=method, source=source, guess=name, valid=False)
    try:
        # Keep UKS orbital basin, replace operator with UHF WITHOUT SCF relaxation.
        carrier = (mf.to_hf() if hasattr(mf, "xc") else mf).copy()
        ob = (OBDH_CL if method == 'obdh' else OBMP2_CL)(carrier)
        if not hasattr(ob, 'dft_grid_level'):
            raise RuntimeError('Install pyCMF fix/sie-initialization before running OB')
        ob.with_df = df.DF(mol, auxbasis=df.make_auxbasis(mol, mp2fit=True))
        ob.alphaa = tuple(cfg.alpha)
        ob.dft_grid_level = cfg.grid
        ob.niter = cfg.ob_cycles
        ob.thresh = cfg.ob_tol
        ob.use_embed = False
        ob.use_cl = False
        ob.verbose = 0
        energy = float(ob.kernel())
        row.update(energy=energy, converged=bool(ob.converged), stability='not_tested_for_OB',
                   **diagnostics(mol,ob.gamma,groups,ob.mo_coeff,ob.mo_occ))
        row['valid'] = bool(ob.converged and row['density_valid'] and np.isfinite(energy))
        row['iteration'] = ob.last_iteration
        row['seed_label'] = diagnostics(mol,seed[2],groups)['label']
    except Exception as exc:
        row['exception'] = str(exc)
        traceback.print_exc()
    return row


def select(rows):
    good = [r for r in rows if r.get('valid') and np.isfinite(r.get('energy',np.nan))]
    return min(good, key=lambda r:r['energy']) if good else None


def summarize(cases, outdir, cfg):
    candidates, selected = [], []
    for case in cases:
        candidates.extend(case['candidates'])
        for method in cfg.methods:
            subset = [r for r in case['candidates'] if r['stage'] == method]
            for source in ['all']+sorted({r['source'] for r in subset}):
                chosen = select(subset if source=='all' else [r for r in subset if r['source']==source])
                selected.append(dict(system=case['system'], point=case['point'], method=method,
                                     init_source=source, energy=chosen['energy'] if chosen else None,
                                     valid=chosen is not None, label=chosen.get('label') if chosen else None,
                                     s2=chosen.get('s2') if chosen else None,
                                     guess=chosen.get('guess') if chosen else None))
    lookup = {(r['system'],r['point'],r['method'],r['init_source']):r for r in selected}
    errors = []
    for system in cfg.systems:
        for point in cfg.points:
            if point == DL:
                continue
            for method, source in sorted({(r['method'],r['init_source']) for r in selected}):
                finite = lookup.get((system,point,method,source))
                limit = lookup.get((system,DL,method,source))
                de = (limit['energy']-finite['energy'])*KCAL if finite and limit and finite['valid'] and limit['valid'] else None
                ref = REF[system][POINTS.index(point)]
                errors.append(dict(system=system,point=point,method=method,init_source=source,
                                   De_kcal=de,reference=ref,error_kcal=de-ref if de is not None else None))
    stats = []
    for method,source in sorted({(r['method'],r['init_source']) for r in selected}):
        sample = [r['error_kcal'] for r in errors if r['method']==method and r['init_source']==source and r['error_kcal'] is not None]
        requested = len(cfg.systems)*len([p for p in cfg.points if p!=DL])
        stats.append(dict(method=method,init_source=source,n_valid=len(sample),n_requested=requested,
                          complete_SIE4x4=len(sample)==16, MUE_16=np.mean(np.abs(sample)) if len(sample)==16 else None,
                          MUE_available=np.mean(np.abs(sample)) if sample else None))
    comparisons = []
    xc = f'{cfg.alpha[0]}*HF + {1-cfg.alpha[0]}*B88, {1-cfg.alpha[1]}*LYP' if hasattr(cfg,'alpha') else None
    for system in cfg.systems:
        for point in cfg.points:
            for method in ['obdh','obmp2']:
                hf = lookup.get((system,point,method,'uhf'))
                ks = lookup.get((system,point,method,xc))
                if hf is None and ks is None: continue
                paired = bool(hf and ks and hf['valid'] and ks['valid'])
                comparisons.append(dict(system=system,point=point,method=method,
                                        paired_valid=paired,
                                        HF_energy=hf['energy'] if hf else None,
                                        UKS_energy=ks['energy'] if ks else None,
                                        HF_label=hf['label'] if hf else None,
                                        UKS_label=ks['label'] if ks else None,
                                        delta_E_Eh=ks['energy']-hf['energy'] if paired else None))
    common_stats = []
    all_methods = sorted({r['method'] for r in errors})
    for method in all_methods:
        for source in sorted({r['init_source'] for r in errors if r['method']==method}):
            common = [(r['system'],r['point']) for r in errors if r['method']==method and r['init_source']==source and r['error_kcal'] is not None]
            shared = [key for key in common if all(any(r['method']==m and r['init_source']=='all' and (r['system'],r['point'])==key and r['error_kcal'] is not None for r in errors) for m in all_methods)]
            sample = [r['error_kcal'] for r in errors if r['method']==method and r['init_source']==source and (r['system'],r['point']) in shared]
            common_stats.append(dict(method=method,init_source=source,n_common=len(sample),
                                     MUE_common=np.mean(np.abs(sample)) if sample else None))
    frames = dict(candidates=pd.DataFrame(candidates),selected=pd.DataFrame(selected),
                  init_comparison=pd.DataFrame(comparisons),errors=pd.DataFrame(errors),
                  statistics=pd.DataFrame(stats),common_statistics=pd.DataFrame(common_stats))
    for name,frame in frames.items():
        frame.to_csv(outdir/f'{name}.csv',index=False)
    with pd.ExcelWriter(outdir/'SIE4x4.xlsx',engine='openpyxl') as writer:
        for name,frame in frames.items():
            frame.to_excel(writer,sheet_name=name,index=False)
        pd.DataFrame([{'config':k,'value':str(v)} for k,v in vars(cfg).items()]).to_excel(writer,sheet_name='config',index=False)
    return selected


def clean_json(value):
    if isinstance(value, dict): return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.generic): return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def main(default_methods=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=HERE/'inputs/input.json')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--basis',default='aug-cc-pVDZ')
    p.add_argument('--grid',type=int,default=4)
    p.add_argument('--scf-aux',default='def2-universal-jkfit')
    p.add_argument('--methods',nargs='+',choices=['uhf','ump2','pbe','pbe0','dh_matched','obdh','obmp2'],
                   default=default_methods or ['uhf','ump2','pbe','pbe0','dh_matched','obdh','obmp2'])
    p.add_argument('--systems',nargs='+',choices=list(REF),default=list(REF))
    p.add_argument('--points',nargs='+',choices=POINTS+[DL],default=POINTS+[DL])
    p.add_argument('--alpha',nargs=2,type=float,default=[.53,.39],metavar=('HF','PT2'))
    p.add_argument('--scf-tol',type=float,default=1e-9)
    p.add_argument('--scf-grad-tol',type=float,default=1e-6)
    p.add_argument('--scf-cycles',type=int,default=200)
    p.add_argument('--stability-cycles',type=int,default=6)
    p.add_argument('--ob-cycles',type=int,default=300)
    p.add_argument('--ob-tol',type=float,default=1e-6)
    p.add_argument('--density-tol',type=float,default=1e-5)
    p.add_argument('--threads',type=int,default=1)
    cfg = p.parse_args()
    if any(a<0 or a>1 for a in cfg.alpha):
        p.error('alpha coefficients must be between 0 and 1')
    lib.num_threads(cfg.threads)
    cfg.outdir.mkdir(parents=True,exist_ok=True)
    data = json.loads(cfg.input.read_text())
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}
    config['pyscf_version'] = pyscf.__version__
    config['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['input_sha256'] = hashlib.sha256(cfg.input.read_bytes()).hexdigest()
    if any(m in cfg.methods for m in ['obdh','obmp2']):
        import pycmf.OBDH.main as core
        import pycmf.OBDH.uobdh_solver as solver
        config['pycmf_core_sha256'] = hashlib.sha256(Path(core.__file__).read_bytes()+Path(solver.__file__).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    (cfg.outdir/'config.json').write_text(json.dumps(config,indent=2))
    cases = []
    xc = f'{cfg.alpha[0]}*HF + {1-cfg.alpha[0]}*B88, {1-cfg.alpha[1]}*LYP'
    for system in cfg.systems:
        for point in cfg.points:
            stem = cfg.outdir/f'{system}__{point}'
            if Path(str(stem)+'.json').exists():
                cached = json.loads(Path(str(stem)+'.json').read_text())
                if cached.get('fingerprint') != fingerprint:
                    raise RuntimeError('Output directory contains different config/code; choose a new --outdir')
                cases.append(cached)
                continue
            print(f'Running {system} {point}',flush=True)
            atoms = data[system][point]
            groups = fragments(system,atoms)
            mol = build(atoms,cfg.basis)
            rows = []
            sources = set()
            if set(cfg.methods)&{'uhf','ump2','obdh','obmp2'}: sources.add('uhf')
            if set(cfg.methods)&{'dh_matched','obdh','obmp2'}: sources.add(xc)
            for method in ['pbe','pbe0']:
                if method in cfg.methods: sources.add(method)
            with Path(str(stem)+'.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                mol.stdout = log
                for source in sorted(sources):
                    pool = seed_pool(mol,atoms,groups,source,cfg,rows)
                    scf_method = 'uhf' if source=='uhf' else source
                    if scf_method in cfg.methods:
                        for name,mf,dm in pool:
                            rows.append(dict(stage=scf_method,source=source,guess=name,energy=float(mf.e_tot),
                                             converged=True,stable=True,valid=True,**diagnostics(mol,dm,groups,mf.mo_coeff,mf.mo_occ)))
                    # UMP2/DH are evaluated on the LOWEST stable SCF reference per source.
                    # They are not orbital variational methods: don't rank SCF basins by PT2 energy.
                    if pool and ((source=='uhf' and 'ump2' in cfg.methods) or (source==xc and 'dh_matched' in cfg.methods)):
                        seed = min(pool,key=lambda item:item[1].e_tot)
                        name,mf,dm = seed
                        method = 'ump2' if source=='uhf' else 'dh_matched'
                        row = dict(stage=method,source=source,guess=name,valid=False)
                        try:
                            reference = (mf.to_hf() if hasattr(mf, "xc") else mf).copy()
                            # One-shot PT2 on canonical KS orbitals; to_hf resets converged.
                            # Do not trigger noncanonical HF-MP2 orbital iterations here.
                            reference.converged = True
                            post = mp.UMP2(reference).density_fit()
                            post.kernel()
                            row.update(energy=float(mf.e_tot+(1 if method=='ump2' else cfg.alpha[1])*post.e_corr),
                                       converged=True,stable=True,**diagnostics(mol,dm,groups,mf.mo_coeff,mf.mo_occ))
                            row['valid'] = row['density_valid'] and np.isfinite(row['energy'])
                        except Exception as exc:
                            row['exception'] = str(exc)
                            traceback.print_exc()
                        rows.append(row)
                    if source in ['uhf',xc]:
                        for method in ['obdh','obmp2']:
                            if method in cfg.methods:
                                rows.extend(ob_candidate(mol,groups,seed,source,method,cfg) for seed in pool)
            for row in rows:
                row.update(system=system,point=point)
            case = dict(system=system,point=point,fingerprint=fingerprint,candidates=rows)
            temporary = Path(str(stem)+'.tmp')
            temporary.write_text(json.dumps(clean_json(case),indent=2,allow_nan=False))
            temporary.replace(Path(str(stem)+'.json'))
            cases.append(case)
            summarize(cases,cfg.outdir,cfg)
    selected = summarize(cases,cfg.outdir,cfg)
    missing = [r for r in selected if r['init_source']=='all' and not r['valid']]
    print(f'{cfg.outdir}/SIE4x4.xlsx; missing selected energies: {len(missing)}',flush=True)
    return int(bool(missing))


if __name__ == '__main__':
    sys.exit(main())
