#!/usr/bin/env python3
"""SIE scaling_v2: converged DFT -> one OBMP2 evaluation; paired full OBDH.

All energies use the supermolecule at R_DL. This driver uses the existing pyCMF
operators without editing the library. obdh_1st is an evaluated one-shot method,
not an unconverged full OBDH result. See README.md for the sheet definitions.
"""
import argparse
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyscf
from pyscf import df, dft, lib

import run_sie as base

PROTOCOL = 'scaling_v2'
LAYOUT = 'independent_baseline_and_diagnostics_v1'
METHODS = ('obdh', 'obdh_1st', 'dh_matched')


def digest(value):
    return hashlib.sha256(json.dumps(base.clean_json(value), sort_keys=True).encode()).hexdigest()


def xc_string(alpha):
    ax, ac = alpha
    return f'{ax}*HF + {1-ax}*B88, {1-ac}*LYP'


def flat_diagnostics(mol, dm, groups, coeff, occ):
    row = base.diagnostics(mol, dm, groups, coeff, occ)
    for field in ('fragment_charge', 'fragment_spin'):
        for i, value in enumerate(row[field], 1):
            row[f'{field}_{i}'] = value
        del row[field]
    row['N_alpha'], row['N_beta'] = row.pop('electrons')
    return row


def factory(mol, source, cfg, ob_aux):
    # The DFT reference uses the same RI operator as the DFT term inside OBDH.
    # pyCMF currently assigns ob.with_df to its internal UKS object.
    mf = base.factory(mol, 'uhf' if source == 'uhf' else xc_string(cfg.alpha), cfg)
    if source == 'dft':
        mf.with_df = ob_aux
    mf.max_memory = cfg.max_memory
    return mf


def save_orbitals(directory, system, point, source, guess, state, mf, geometry, alpha,
                  export_role='candidate'):
    directory.mkdir(exist_ok=True)
    token = hashlib.sha256(guess.encode()).hexdigest()[:12]
    path = directory / f'{system}__{point}__{source}__{token}__{state}.npz'
    metadata = dict(system=system, point=point, source=source, guess=guess, state=state,
                    geometry=geometry, alpha=list(alpha), export_role=export_role)
    np.savez_compressed(path, coeff=np.asarray(mf.mo_coeff), occ=np.asarray(mf.mo_occ),
                        epsilon=np.asarray(mf.mo_energy), metadata=json.dumps(metadata))
    return str(path)


def load_neighbors(directories, system, point, geometry):
    result = []
    for directory in directories:
        for path in sorted((directory / 'orbitals').glob(f'{system}__{point}__*.npz')):
            with np.load(path, allow_pickle=False) as saved:
                meta = json.loads(str(saved['metadata']))
                if meta['geometry'] != geometry:
                    raise ValueError(f'Neighbor geometry/basis/charge/spin mismatch: {path}')
                # Only the independently selected baseline is a transferable
                # reference. Never import inherited neighbor/restart candidates.
                if meta.get('export_role') != 'selected_baseline_obdh':
                    continue
                result.append(dict(path=str(path.resolve()), meta=meta,
                                   coeff=saved['coeff'].copy(), occ=saved['occ'].copy(),
                                   epsilon=saved['epsilon'].copy()))
    return result


def neighbor_ready(directory, cfg, expected_alpha=None):
    """Require finished case checkpoints and saved orbitals for the requested scope."""
    try:
        config = json.loads((directory/'config.json').read_text())
        if config.get('protocol') != PROTOCOL or config.get('pipeline_layout') != LAYOUT or config.get('basis') != cfg.basis or config.get('grid') != cfg.grid:
            return False
        if config.get('input_sha256') != hashlib.sha256(cfg.input.read_bytes()).hexdigest():
            return False
        for field in ('scf_aux','ob_aux','scf_policy','scf_tol','scf_grad_tol','scf_cycles',
                      'stability_cycles','ob_tol','ob_cycles','density_tol','pyscf_version','pycmf_core_sha256'):
            if hasattr(cfg,field) and config.get(field)!=getattr(cfg,field):
                return False
        expected_driver=hashlib.sha256(Path(__file__).read_bytes()+Path(base.__file__).read_bytes()).hexdigest()
        if config.get('driver_sha256')!=expected_driver:
            return False
        if expected_alpha is not None and not np.allclose(config['alpha'], expected_alpha, atol=1e-10, rtol=0):
            return False
        fingerprint = digest(config)
        for system in cfg.systems:
            for point in cfg.points:
                case = json.loads((directory/f'{system}__{point}.json').read_text())
                if case.get('fingerprint') != fingerprint:
                    return False
                # A finished baseline failure is still a finished calculation.
                # C2 will explicitly report a missing OB reference for that case.
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def nearby_alphas(alpha, step):
    ax, ac = alpha
    lower, upper = (.1,.9) if all(.1 <= v <= .9 for v in alpha) else (0.,1.)
    seen = set()
    for pair in [(ax, ac-step), (ax-step, ac), (ax, ac+step), (ax+step, ac)]:
        pair = tuple(round(v,10) for v in pair)
        if all(lower <= v <= upper for v in pair) and pair not in seen:
            seen.add(pair)
            yield pair


def configure_neighbors(cfg):
    cfg.neighbor_root = cfg.neighbor_root or cfg.outdir.parent
    if cfg.neighbor_dir:
        cfg.neighbor_selection = 'explicit'
        return
    if cfg.neighbor_mode == 'off':
        cfg.neighbor_selection = 'disabled'
        return
    # Freeze the source chosen on the first invocation. Newly completed adjacent
    # runs must not silently change the reference during resume.
    saved_config = cfg.outdir/'config.json'
    if saved_config.exists():
        saved = json.loads(saved_config.read_text())
        if saved.get('neighbor_mode') == 'auto' and 'neighbor_selection' in saved:
            cfg.neighbor_dir = [Path(p) for p in saved.get('neighbor_dir', [])]
            cfg.neighbor_selection = saved['neighbor_selection']
            if 'neighbor_preparation_error' in saved:
                cfg.neighbor_preparation_error=saved['neighbor_preparation_error']
            return
    # Fix the adjacent pair before checking availability. Array scheduling must
    # not decide which physical reference is used.
    alpha = next(nearby_alphas(cfg.alpha, cfg.neighbor_step), None)
    if alpha is None:
        raise ValueError('No adjacent Alpha exists for the requested neighbor step')
    stem = f'recalc_obdh_({alpha[0]:g},{alpha[1]:g})'
    directory = cfg.neighbor_root/stem
    if directory.resolve() != cfg.outdir.resolve() and neighbor_ready(directory, cfg, alpha):
        cfg.neighbor_dir = [directory.resolve()]
        cfg.neighbor_selection = 'auto_completed_adjacent'
        return
    if not cfg.prepare_neighbor:
        cfg.neighbor_selection = 'none_available_C2_not_performed'
        return
    # A supporting run has C2 disabled to avoid recursive preparation. It is
    # stored inside this run and never presented as another primary grid result.
    directory = (cfg.outdir/'neighbor_reference').resolve()
    command = [sys.executable, str(Path(__file__).resolve()), '--input',str(cfg.input.resolve()),
        '--outdir',str(directory),'--alpha',*[str(v) for v in alpha],
        '--methods','obdh','--ob-init',*cfg.ob_init,'--neighbor-mode','off',
        '--run-role','neighbor_reference','--ob-restarts','0','--ob-probe-directions','0',
        '--basis',cfg.basis,'--grid',str(cfg.grid),'--scf-aux',cfg.scf_aux,'--ob-aux',cfg.ob_aux,
        '--scf-policy',cfg.scf_policy,'--scf-tol',str(cfg.scf_tol),'--scf-grad-tol',str(cfg.scf_grad_tol),
        '--scf-cycles',str(cfg.scf_cycles),'--stability-cycles',str(cfg.stability_cycles),
        '--ob-cycles',str(cfg.ob_cycles),'--ob-tol',str(cfg.ob_tol),'--density-tol',str(cfg.density_tol),
        '--threads',str(cfg.threads),'--max-memory',str(cfg.max_memory),
        '--systems',*cfg.systems,'--points',*cfg.points]
    print(f'C2 DIAGNOSTIC: preparing the predetermined adjacent Alpha {alpha} in {directory}. '
          'This supporting calculation never supplies the primary selected energies.',flush=True)
    process = subprocess.run(command, check=False)
    # A nonzero exit may mean some independent baselines failed. Use available
    # baseline OB states; report the absent ones, without recursively rescuing them.
    if not neighbor_ready(directory, cfg, alpha):
        cfg.neighbor_selection='supporting_neighbor_incomplete'
        cfg.neighbor_preparation_error=f'C2 source incomplete (exit {process.returncode}): {directory}'
        print(cfg.neighbor_preparation_error+'; independent baselines will still run.',flush=True)
        return
    cfg.neighbor_dir = [directory]
    cfg.neighbor_selection = 'prepared_supporting_adjacent'


def seed_pool(mol, atoms, groups, source, cfg, ob_aux, neighbors, rows):
    if neighbors:
        raise ValueError('Baseline SCF must be independent of neighbor calculations')
    names = ['fragment_average', 'minao', 'atom', 'huckel', 'localized_A', 'localized_B']
    if cfg.scf_policy == 'standard':
        names = ['minao']
    local = {}
    pool = []
    for name in names:
        row = dict(stage='scf', source=source, guess=name, valid=False,
                   scf_converged=False, scf_stable=False, neighbor_path=None,
                   result_scope='baseline', initialization_path=f'current_{source}_scf')
        print(f'\n=== SCF source={source} guess={name} alpha={cfg.alpha} ===', flush=True)
        print('BASELINE SCF: starting independently at the current Alpha. '
              'The guess name identifies the starting density, not the final loc/deloc label. '
              + (f'Target DFT reference: {xc_string(cfg.alpha)}.' if source=='dft' else
                 'UHF reference: no DFT functional and no Alpha-dependent HF operator.'),flush=True)
        try:
            mf = factory(mol, source, cfg, ob_aux)
            orbital_guess = None
            if name.startswith('localized') or name == 'fragment_average':
                for side in (0, 1):
                    if side not in local:
                        # Fragment densities are guesses only, never the DL energy reference.
                        local[side] = base.localized_density(mol, atoms, groups, side,
                            'uhf' if source == 'uhf' else xc_string(cfg.alpha), cfg)
                dm = .5 * (local[0] + local[1]) if name == 'fragment_average' else local[int(name.endswith('B'))]
            else:
                dm = mf.get_init_guess(key=name)
            mf, valid, stable = base.stabilize(mf, dm, cfg, orbital_guess)
            dm = np.asarray(mf.make_rdm1())
            grad = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ,
                        mf.get_fock(dm=dm, level_shift_factor=0))))
            row.update(energy=float(mf.e_tot), scf_converged=bool(mf.converged),
                       scf_stable=stable, orbital_gradient=grad,
                       **flat_diagnostics(mol, dm, groups, mf.mo_coeff, mf.mo_occ))
            row['valid'] = bool(valid and row['density_valid'] and np.isfinite(mf.e_tot)
                                and grad <= cfg.scf_grad_tol)
            duplicate = next((seed['guess'] for seed in pool
                if base.density_distance(dm, seed['dm'], mf.get_ovlp()) < cfg.density_tol), None)
            row['duplicate_of'] = duplicate
            if row['valid'] and duplicate is None:
                pool.append(dict(guess=name, mf=mf, dm=dm, provenance=row['neighbor_path']))
        except Exception as exc:
            row['exception'] = str(exc)
            traceback.print_exc()
        rows.append(row)
        print('SCF_RESULT ' + json.dumps(base.clean_json(row)), flush=True)
        print(f'SCF DECISION: converged={row.get("scf_converged")}; stable={row.get("scf_stable")}; '
              f'valid={row["valid"]}; final type={row.get("label")}; '
              f'duplicate of={row.get("duplicate_of")}; energy={row.get("energy")} Ha. '
              'Only distinct, valid current-Alpha SCF solutions continue to the baseline methods.',flush=True)
    return pool


def orbital_identity(mf):
    value=hashlib.sha256()
    for name in ('mo_coeff','mo_occ','mo_energy'):
        array=np.ascontiguousarray(np.asarray(getattr(mf,name)))
        value.update(str(array.shape).encode())
        value.update(array.tobytes())
    return value.hexdigest()[:20]


class Trace:
    """Match each printed energy to the state passed to make_amp in that cycle."""
    pattern = re.compile(r'Iter (\d+): (?:E_tot=([^,]+), )?E_corr=([^,]+), dE=([^,]+), df_ia=([^,]+), dRMS=([^\s]+)')

    def __init__(self, stream, mol, groups):
        self.stream, self.mol, self.groups = stream, mol, groups
        self.rows, self.last_state, self.buffer = [], None, ''

    def amplitudes(self, original, post):
        answer = original(post)
        coeff, occ, eps = [np.asarray(getattr(post, key)).copy()
                           for key in ('mo_coeff', 'mo_occ', 'mo_energy')]
        dm = np.asarray(post._scf.make_rdm1(coeff, occ))
        self.last_state = dict(coeff=coeff, occ=occ, epsilon=eps, dm=dm)
        row = flat_diagnostics(self.mol, dm, self.groups, coeff, occ)
        da = eps[0][occ[0] > 0, None] - eps[0][None, occ[0] == 0]
        db = eps[1][occ[1] > 0, None] - eps[1][None, occ[1] == 0]
        row['gap_alpha_Ha'] = float(-np.max(da)) if da.size else None
        row['gap_beta_Ha'] = float(-np.max(db)) if db.size else None
        # Restrict denominator diagnostics to nonzero antisymmetrized amplitudes.
        # Threshold is diagnostic only; it never changes amplitudes or energies.
        for label, left, right, amplitude in zip(('aa', 'bb', 'ab'), (da, db, da),
                                                (da, db, db), answer[1][:3]):
            denominator = left[:, :, None, None] + right[None, None, :, :] - post.shift
            mask = np.abs(amplitude) > 1e-12
            row[f't2_max_{label}'] = float(np.max(np.abs(amplitude))) if amplitude.size else 0.
            row[f'denom_min_active_{label}_Ha'] = float(np.min(np.abs(denominator[mask]))) if mask.any() else None
        self.rows.append(row)
        return answer

    def write(self, value):
        self.stream.write(value)
        self.buffer += value
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            match = self.pattern.search(line)
            if match and self.rows:
                cycle, energy, corr, de, fia, residual = match.groups()
                self.rows[-1].update(cycle=int(cycle) + 1,
                    energy=float(energy) if energy is not None else None,
                    correlation_printed_Ha=float(corr), delta_energy_Ha=float(de),
                    effective_fia=float(fia), diis_residual_rms=float(residual))
        return len(value)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def energy_components(mf, dm, alpha):
    """Decompose the DFT term on a supplied density with this MF's RI/grid."""
    mol = mf.mol
    dm = np.asarray(dm)
    j, k = mf.get_jk(mol, dm)
    total_dm = dm.sum(axis=0)
    e1 = float(np.einsum('ij,ji', total_dm, mf.get_hcore()).real)
    ej = float(.5 * np.einsum('ij,ji', total_dm, np.asarray(j).sum(axis=0)).real)
    ex = float(-.5 * np.einsum('sij,sji', dm, k).real)
    if mf.grids.coords is None:
        mf.get_veff(mol, dm)  # initialize/prune grids as UKS does
    xb88 = float(mf._numint.nr_uks(mol, mf.grids, 'B88,', dm, max_memory=mf.max_memory)[1])
    clyp = float(mf._numint.nr_uks(mol, mf.grids, ',LYP', dm, max_memory=mf.max_memory)[1])
    ax, ac = alpha
    edft = e1 + ej + mol.energy_nuc() + ax * ex + (1-ax) * xb88 + (1-ac) * clyp
    return dict(E_one_electron_Ha=e1, E_Hartree_Ha=ej, E_nuclear_Ha=float(mol.energy_nuc()),
                E_x_HF_Ha=ex, E_x_B88_Ha=xb88, E_c_LYP_Ha=clyp, E_DFT_Ha=float(edft))


def ob_run(mol, groups, seed, source, cfg, ob_aux, one_shot=False):
    from pycmf.OBDH import OBDH_CL, OBMP2_CL
    from pycmf.OBDH import uobdh_solver as solver
    method = 'obdh_1st' if one_shot else 'obdh'
    scope=seed.get('result_scope','diagnostic' if seed.get('provenance') else 'baseline')
    path=seed.get('initialization_path',f'current_{source}_scf')
    reference_id=seed.setdefault('reference_id',orbital_identity(seed['mf']))
    row = dict(stage=method, source=source, guess=seed['guess'], valid=False,
               evaluation_complete=False, ob_converged=False,
               scf_converged=True if seed.get('is_scf', True) else None,
               scf_stable=True if seed.get('is_scf', True) else None, ob_stability='not_tested',
               evaluation_mode='dft_converged_then_obmp2_one_shot' if one_shot else 'self_consistent_obdh',
               residual_operator='pure_OBMP2' if one_shot else 'OBDH',
               seed_provenance=seed.get('provenance'), result_scope=scope,
               initialization_path=path, initial_orbital_id=reference_id,
               neighbor_alpha=seed.get('neighbor_alpha'),
               initial_label=base.diagnostics(mol,seed['dm'],groups)['label'],
               current_scf_reference=bool(seed.get('is_scf',True)))
    print(f'\n=== {method} source={source} guess={seed["guess"]} ===', flush=True)
    print(f'CALCULATION ROLE: {scope.upper()}. Initialization path: {path}. '
          f'Initial orbital identity: {reference_id}. Initial type: {row["initial_label"]}.',flush=True)
    print(f'CURRENT TARGET: Alpha={tuple(cfg.alpha)}; DFT term={xc_string(cfg.alpha)}; '
          f'scaled correlation coefficient={cfg.alpha[1]}.',flush=True)
    if scope=='diagnostic':
        print(f'DIAGNOSTIC ONLY: input orbitals come from {seed.get("provenance")}. '
              f'Neighbor Alpha={seed.get("neighbor_alpha")}; current target Alpha={tuple(cfg.alpha)}. '
              'No new current-Alpha SCF is performed here. The source column records ancestry. '
              'This result cannot replace a baseline selected energy.',flush=True)
    elif source=='dft':
        print('DFT INITIALIZATION: use the current-Alpha converged DFT orbitals, occupations and '
              'orbital energies. Switch to HF operator evaluation without running HF SCF.',flush=True)
    if one_shot:
        print('ONE-SHOT DEFINITION: keep the converged DFT state unchanged; evaluate pure OBMP2 '
              'correlation once. Any pure-OBMP2 total printed inside the solver is NOT the final '
              'reported one-shot energy. Report E_DFT_SCF + alpha_c * E_corr_OBMP2 below.',flush=True)
    else:
        print(f'FULL OBDH: allow orbitals and orbital energies to change; maximum {cfg.ob_cycles} '
              f'iterations, solver energy threshold {cfg.ob_tol}. Effective F_ia and DIIS residual '
              'are solver diagnostics, not a formal OB Hessian or an exact energy gradient.',flush=True)
    trace = Trace(sys.stdout, mol, groups)
    try:
        carrier = base.hf_orbital_carrier(seed['mf'])
        post = (OBMP2_CL if one_shot else OBDH_CL)(carrier)
        post.with_df = ob_aux
        post.alphaa = tuple(cfg.alpha)
        post.use_embed = post.use_cl = False
        post.eval_IPEA = False
        post.niter = 1 if one_shot else cfg.ob_cycles
        post.thresh = cfg.ob_tol
        post.verbose = 0
        original = solver.make_amp
        with patch.object(solver, 'make_amp', side_effect=lambda obj: trace.amplitudes(original, obj)), \
                lib.temporary_env(dft.gen_grid.Grids, level=cfg.grid), contextlib.redirect_stdout(trace):
            total = float(post.kernel())
        state = trace.last_state
        if state is None or not trace.rows:
            raise RuntimeError('No evaluated orbital state was captured from pyCMF')
        distance = base.density_distance(np.asarray(post.gamma), state['dm'], mol.intor_symmetric('int1e_ovlp'))
        if distance > 1e-7:
            raise RuntimeError('pyCMF returns orbitals/density from a different iteration than energy; update the solver before scaling_v2')
        for item in trace.rows:
            item['correlation_convention'] = 'unscaled_OBMP2_correction' if one_shot else 'alpha_c_times_OBMP2_correction'
            item['residual_operator'] = row['residual_operator']
        evaluator = seed['mf'] if one_shot else factory(mol, 'dft', cfg, ob_aux)
        if not one_shot:
            # Internal pyCMF UKS builds its grid on the initial seed density.
            evaluator.get_veff(mol, seed['dm'])
        components = energy_components(evaluator, state['dm'], cfg.alpha)
        row.update(components)
        row.update(**flat_diagnostics(mol, state['dm'], groups, state['coeff'], state['occ']))
        row['initial_to_final_density_distance']=base.density_distance(state['dm'],seed['dm'],mol.intor_symmetric('int1e_ovlp'))
        row['initial_to_final_label_changed']=row['initial_label']!=row['label']
        row['same_final_branch_certified']=False
        row.update(evaluation_complete=True, ob_converged=bool(post.converged),
                   energy_state_density_distance=distance, cycles=len(trace.rows),
                   effective_fia=trace.rows[-1].get('effective_fia'),
                   diis_residual_rms=trace.rows[-1].get('diis_residual_rms'))
        if one_shot:
            if source != 'dft':
                raise ValueError('obdh_1st requires a converged DFT reference')
            if base.density_distance(state['dm'], seed['dm'], mol.intor_symmetric('int1e_ovlp')) > 1e-7:
                raise RuntimeError('One-shot evaluation changed the converged DFT reference density')
            row['E_corr_OBMP2_Ha'] = float(post.e_corr)
            row['E_corr_scaled_Ha'] = cfg.alpha[1] * row['E_corr_OBMP2_Ha']
            row['E_DFT_SCF_Ha'] = float(seed['mf'].e_tot)
            row['DFT_reconstruction_error_Ha'] = components['E_DFT_Ha'] - seed['mf'].e_tot
            if abs(row['DFT_reconstruction_error_Ha']) > 1e-7:
                raise RuntimeError('DFT decomposition does not reproduce the converged DFT energy')
            row['energy'] = float(seed['mf'].e_tot + row['E_corr_scaled_Ha'])
            row['status'] = 'evaluated_one_shot'
            row['valid'] = bool(row['density_valid'] and np.isfinite(row['energy']))
        else:
            row['energy'] = total
            row['E_corr_scaled_Ha'] = trace.rows[-1]['correlation_printed_Ha']
            row['E_corr_OBMP2_Ha'] = row['E_corr_scaled_Ha'] / cfg.alpha[1] if cfg.alpha[1] else None
            row['energy_reconstruction_error_Ha'] = total - components['E_DFT_Ha'] - row['E_corr_scaled_Ha']
            if abs(row['energy_reconstruction_error_Ha']) > 1e-7:
                raise RuntimeError('OBDH energy does not match the decomposed DFT/correlation terms at the returned state')
            row['status'] = 'converged' if post.converged else 'not_converged'
            row['valid'] = bool(post.converged and row['density_valid'] and np.isfinite(total))
        row['frozen_Dx_Ha'] = row['E_x_HF_Ha'] - row['E_x_B88_Ha']
        row['frozen_Dc_Ha'] = (row['E_corr_OBMP2_Ha'] - row['E_c_LYP_Ha']) if row['E_corr_OBMP2_Ha'] is not None else None
        carrier.mo_coeff, carrier.mo_occ, carrier.mo_energy = state['coeff'], state['occ'], state['epsilon']
        print(f'REPORTED RESULT: method={method}; scope={scope}; path={path}; '
              f'energy={row["energy"]:.12f} Ha; valid={row["valid"]}; '
              f'status={row["status"]}; evaluated states={row["cycles"]}.',flush=True)
        print(f'ENERGY COMPONENTS: DFT={row["E_DFT_Ha"]:.12f} Ha; '
              f'scaled OBMP2 correction={row["E_corr_scaled_Ha"]:.12f} Ha. '
              f'Initial/final type={row["initial_label"]}/{row["label"]}; '
              f'density distance={row["initial_to_final_density_distance"]:.6e}; '
              f'determinant <S^2>={row.get("s2")}. '
              'Equal type labels do not certify that two solutions belong to the same branch.',flush=True)
        return row, trace.rows, carrier
    except Exception as exc:
        row.update(exception=str(exc), status='evaluation_failed')
        print(f'CALCULATION FAILED: method={method}; scope={scope}; path={path}; '
              f'reason={exc}. No zero energy and no diagnostic replacement will be used.',flush=True)
        traceback.print_exc()
        return row, trace.rows, None


def choose(rows, method):
    good = [r for r in rows if r.get('valid') and np.isfinite(r.get('energy', np.nan))]
    if not good:
        return None
    # One-shot methods use the lowest DFT reference, not the lowest post-correction energy.
    key = 'E_DFT_SCF_Ha' if method in ('obdh_1st', 'dh_matched') else 'energy'
    return min(good, key=lambda row: row[key])


def baseline_choice(case,method,source):
    subset=[r for r in case['candidates'] if r['stage']==method and r['source']==source
            and r.get('result_scope','baseline')=='baseline']
    if method in ('obdh_1st','dh_matched'):
        reference=choose([r for r in case['candidates'] if r['stage']=='dft_base' and r['source']==source
                          and r.get('result_scope','baseline')=='baseline'],'dft_base')
        subset=[r for r in subset if reference and r['guess']==reference['guess']]
    return choose(subset,method)


def orbital_probes(mol, groups, carrier, source, guess, cfg, ob_aux):
    """Directional energy checks, explicitly not a full OB Hessian/stability proof.

    Occupations and orbital energies/denominators stay fixed; amplitudes and DFT
    density terms are recalculated. Only the selected directions are tested.
    """
    from scipy.linalg import expm
    from pycmf.OBDH import OBMP2_CL
    rng = np.random.default_rng(1729)
    original_coeff = np.asarray(carrier.mo_coeff).copy()
    occ = np.asarray(carrier.mo_occ)
    ks = factory(mol, 'dft', cfg, ob_aux)
    ks.get_veff(mol, np.asarray(carrier.make_rdm1()))
    def evaluate(coeff):
        state = carrier.copy()
        state.mo_coeff = tuple(coeff)
        post = OBMP2_CL(state)
        post.with_df, post.alphaa = ob_aux, tuple(cfg.alpha)
        post.use_embed = post.use_cl = post.eval_IPEA = False
        post.niter = 1
        captured=io.StringIO()
        try:
            with lib.temporary_env(dft.gen_grid.Grids, level=cfg.grid), contextlib.redirect_stdout(captured):
                post.kernel()
        except Exception:
            print(captured.getvalue(),flush=True)
            raise
        dm = np.asarray(state.make_rdm1(coeff, occ))
        if base.density_distance(dm, np.asarray(post.gamma), mol.intor_symmetric('int1e_ovlp')) > 1e-7:
            raise RuntimeError('Orbital probe returned a different orbital state')
        return float(ks.energy_tot(dm=dm) + cfg.alpha[1]*post.e_corr)
    center = evaluate(original_coeff)
    rows = []
    for direction in range(cfg.ob_probe_directions):
        rotations = []
        for spin in range(2):
            occupied = np.flatnonzero(occ[spin] > 0)
            virtual = np.flatnonzero(occ[spin] == 0)
            rotation = np.zeros((occ.shape[1], occ.shape[1]))
            block = rng.normal(size=(len(occupied), len(virtual)))
            norm = np.linalg.norm(block)
            if norm:
                block /= norm * np.sqrt(2.)
            rotation[np.ix_(occupied, virtual)] = block
            rotation[np.ix_(virtual, occupied)] = -block.T
            rotations.append(rotation)
        step = cfg.ob_probe_step
        minus = evaluate([original_coeff[s] @ expm(-step*rotations[s]) for s in range(2)])
        plus = evaluate([original_coeff[s] @ expm(step*rotations[s]) for s in range(2)])
        rows.append(dict(source=source, guess=guess, direction=direction+1,
            probe='energy_direction_fixed_orbital_energies', angle_rad=step,
            E_center_Ha=center, E_minus_Ha=minus, E_plus_Ha=plus,
            gradient_Ha_per_rad=(plus-minus)/(2*step),
            curvature_Ha_per_rad2=(plus+minus-2*center)/(step*step),
            downhill_direction_found=min(minus,plus) < center-1e-8,
            ob_stability='not_certified_by_directional_probe'))
        print(f'FROZEN-ORBITAL ENERGY PROBE: source={source}; guess={guess}; direction={direction+1}; '
              f'angle={step} rad; E_minus={minus:.12f}; E_center={center:.12f}; '
              f'E_plus={plus:.12f} Ha. No orbital reconvergence. Fixed orbital energies; '
              'this is not the full energy Hessian and does not affect selected.',flush=True)
    return rows


def restart_checks(mol, groups, carrier, anchor, cfg, ob_aux, system, point, geometry):
    """Perturb the final OB orbitals and reconverge OB at the SAME Alpha."""
    from scipy.linalg import expm
    coeff = np.asarray(carrier.mo_coeff).copy()
    occ = np.asarray(carrier.mo_occ)
    eps = np.asarray(carrier.mo_energy)
    overlap = mol.intor_symmetric('int1e_ovlp')
    slices = mol.aoslice_by_atom()
    weights = np.zeros(mol.nao_nr())
    for sign, group in zip((1.,-1.),groups[:2]):
        for atom in group:
            weights[slices[atom,2]:slices[atom,3]] = sign
    operator = .5*(weights[:,None]*overlap + overlap*weights[None,:])
    best = None
    for spin in range(2):
        occupied, virtual = np.flatnonzero(occ[spin]>0), np.flatnonzero(occ[spin]==0)
        populations = np.einsum('pi,pq,qi->i',coeff[spin],operator,coeff[spin])
        for i in occupied:
            for a in virtual:
                score = abs(populations[i]-populations[a])/max(abs(eps[spin,a]-eps[spin,i]),.02)
                if best is None or score > best[0]:
                    best = score,spin,i,a
    if best is None:
        return [],[],[]
    trials, candidates, iterations = [],[],[]
    rng = np.random.default_rng(821)
    for trial in range(cfg.ob_restarts):
        rotations = np.zeros((2,occ.shape[1],occ.shape[1]))
        if trial < 2:
            _,spin,i,a = best
            rotations[spin,i,a] = 1. if trial == 0 else -1.
            rotations[spin,a,i] = -rotations[spin,i,a]
            kind = 'fragment_transfer_plus' if trial == 0 else 'fragment_transfer_minus'
        else:
            for spin in range(2):
                occupied,virtual = np.flatnonzero(occ[spin]>0),np.flatnonzero(occ[spin]==0)
                block = rng.normal(size=(len(occupied),len(virtual)))
                norm = np.linalg.norm(block)
                if norm: block /= norm*np.sqrt(2.)
                rotations[spin][np.ix_(occupied,virtual)] = block
                rotations[spin][np.ix_(virtual,occupied)] = -block.T
            kind = 'random_occupied_virtual'
        mf = carrier.copy()
        mf.mo_coeff = tuple(coeff[s] @ expm(cfg.ob_restart_angle*rotations[s]) for s in range(2))
        dm = np.asarray(mf.make_rdm1())
        guess = f'restart_{trial+1}_from_{anchor["guess"]}'
        seed = dict(guess=guess,mf=mf,dm=dm,is_scf=False,provenance=f'OB_restart:{anchor["guess"]}',
                    result_scope='diagnostic',initialization_path='perturbed_baseline_obdh')
        row,trace,final = ob_run(mol,groups,seed,anchor['source'],cfg,ob_aux)
        row.update(candidate_origin='ob_restart',restart_of=anchor['guess'])
        candidates.append(row)
        iterations += [dict(method='obdh',source=anchor['source'],guess=guess,**r) for r in trace]
        valid = bool(row['valid'])
        distance = base.density_distance(np.asarray(final.make_rdm1()),np.asarray(carrier.make_rdm1()),overlap) if valid else None
        delta = row['energy']-anchor['energy'] if valid else None
        trials.append(dict(source=anchor['source'],anchor_guess=anchor['guess'],restart_guess=guess,
            perturbation=kind,angle_rad=cfg.ob_restart_angle,restart_converged=valid,
            E_anchor_Ha=anchor['energy'],E_restart_Ha=row.get('energy'),delta_energy_Ha=delta,
            density_distance=distance,returned_same_density=distance < cfg.density_tol if valid else None,
            lower_solution_found=delta < -cfg.restart_energy_tol if valid else None,
            anchor_label=anchor.get('label'),restart_label=row.get('label'),
            status='restart_converged' if valid else 'restart_not_converged',
            formal_ob_stability='not_certified_by_restarts'))
    return trials,candidates,iterations


def report(cases, cfg):
    candidates = [dict(system=c['system'], point=c['point'], **r) for c in cases for r in c['candidates']]
    iterations = [dict(system=c['system'], point=c['point'], **r) for c in cases for r in c['iterations']]
    probes = [dict(system=c['system'], point=c['point'], **r) for c in cases for r in c.get('orbital_probes', [])]
    restarts = [dict(system=c['system'], point=c['point'], **r) for c in cases for r in c.get('ob_restarts', [])]
    selected, convergence, paired, neighbor_comparison, diagnostic_best = [], [], [], [], []
    methods = ['dft_base'] + cfg.methods
    for case in cases:
        for method in methods:
            sources = cfg.ob_init if method == 'obdh' else ['dft']
            for source in sources:
                all_subset = [r for r in case['candidates'] if r['stage'] == method and r['source'] == source]
                subset = [r for r in all_subset if r.get('result_scope','baseline')=='baseline']
                picked = baseline_choice(case,method,source)
                row = dict(system=case['system'], point=case['point'], method=method, init_source=source,
                           alpha=str(tuple(cfg.alpha)), valid=bool(picked), energy=None,
                           status='missing_independent_baseline_result', guess=None,
                           result_scope='baseline',initialization_path=f'current_{source}_scf')
                if picked:
                    row.update(picked)
                selected.append(row)
                if method=='obdh':
                    best=choose(all_subset,method)
                    diagnostic_best.append(dict(system=case['system'],point=case['point'],
                        method=method,ancestry_source=source,valid=bool(best),
                        energy=best.get('energy') if best else None,guess=best.get('guess') if best else None,
                        initialization_path=best.get('initialization_path') if best else None,
                        neighbor_alpha=best.get('neighbor_alpha') if best else None,
                        result_scope=best.get('result_scope','baseline') if best else None,
                        label=best.get('label') if best else None,
                        replaces_selected=False,baseline_valid=bool(picked),
                        delta_from_baseline_Ha=best['energy']-picked['energy'] if best and picked else None,
                        meaning='Lowest found among baseline and diagnostics; NOT a fresh-SCF baseline result'))
                failed_scf = [r for r in case['candidates'] if r['stage'] == 'scf' and r['source'] == source]
                convergence.append(dict(system=case['system'], point=case['point'], method=method,
                    init_source=source, valid=bool(picked), status=row['status'],
                    n_scf_attempts=len(failed_scf), n_scf_valid=sum(r['valid'] for r in failed_scf),
                    n_method_attempts=len(subset), n_method_valid=sum(r['valid'] for r in subset),
                    failure_stage=None if picked else 'scf' if not any(r['valid'] for r in failed_scf) else 'method',
                    exception=case.get('exception') if not picked else None,
                    candidate_exceptions=' | '.join(sorted({str(r['exception']) for r in failed_scf+subset if r.get('exception')})) if not picked else None,
                    ob_converged=row.get('ob_converged'), evaluation_complete=row.get('evaluation_complete')))
        first = [r for r in case['candidates'] if r['stage'] == 'obdh_1st']
        for one in first:
            full = next((r for r in case['candidates'] if r['stage'] == 'obdh' and r['source'] == 'dft'
                         and r.get('result_scope','baseline')=='baseline'
                         and r['guess'] == one['guess']), None)
            same_start=bool(full and one.get('initial_orbital_id') and
                            one.get('initial_orbital_id')==full.get('initial_orbital_id'))
            valid = bool(one['valid'] and full and full['valid'] and same_start)
            paired.append(dict(system=case['system'], point=case['point'], guess=one['guess'],
                paired_valid=valid, E_first_Ha=one.get('energy'), E_full_Ha=full.get('energy') if full else None,
                delta_full_minus_first_Ha=full['energy']-one['energy'] if valid else None,
                delta_DFT_Ha=full['E_DFT_Ha']-one['E_DFT_Ha'] if valid else None,
                delta_corr_scaled_Ha=full['E_corr_scaled_Ha']-one['E_corr_scaled_Ha'] if valid else None,
                first_label=one.get('label'), full_label=full.get('label') if full else None,
                initial_orbital_id=one.get('initial_orbital_id'),same_initial_orbitals=same_start,
                result_scope=one.get('result_scope','baseline'),
                label_changed=one.get('label')!=full.get('label') if valid else None,
                same_final_branch_certified=False,
                interpretation='self_consistency_with_type_change' if valid and one.get('label')!=full.get('label')
                    else 'same_type_label_branch_not_certified' if valid else 'missing_valid_identical_start_pair'))
        for source in cfg.ob_init:
            obs = [r for r in case['candidates'] if r['stage']=='obdh' and r['source']==source
                   and r.get('candidate_origin')!='ob_restart']
            ordinary = choose([r for r in obs if not r.get('seed_provenance')],'obdh')
            neighbor = choose([r for r in obs if r.get('seed_provenance')],'obdh')
            valid = bool(ordinary and neighbor)
            neighbor_comparison.append(dict(system=case['system'],point=case['point'],source=source,
                paired_valid=valid,ordinary_guess=ordinary.get('guess') if ordinary else None,
                neighbor_guess=neighbor.get('guess') if neighbor else None,
                E_ordinary_Ha=ordinary.get('energy') if ordinary else None,
                E_neighbor_Ha=neighbor.get('energy') if neighbor else None,
                delta_neighbor_minus_ordinary_Ha=neighbor['energy']-ordinary['energy'] if valid else None,
                ordinary_label=ordinary.get('label') if ordinary else None,
                neighbor_label=neighbor.get('label') if neighbor else None,
                status='compared' if valid else 'no_valid_OB_pair_see_SCF_candidates'))
    lookup = {(r['system'], r['point'], r['method'], r['init_source']): r for r in selected}
    errors, derivatives = [], []
    for row in selected:
        if row['point'] == base.DL:
            continue
        limit = lookup.get((row['system'], base.DL, row['method'], row['init_source']))
        valid = bool(row['valid'] and limit and limit['valid'])
        reference = base.REF[row['system']][base.POINTS.index(row['point'])]
        de = (limit['energy']-row['energy'])*base.KCAL if valid else None
        errors.append(dict(system=row['system'], point=row['point'], method=row['method'],
            init_source=row['init_source'], valid=valid, De_kcal=de, reference_kcal=reference,
            error_kcal=de-reference if valid else None, abs_error_kcal=abs(de-reference) if valid else None,
            label_R=row.get('label'), label_DL=limit.get('label') if limit else None,
            guess_R=row.get('guess'), guess_DL=limit.get('guess') if limit else None,
            status='valid' if valid else 'Missing valid R or R_DL'))
        derivative = dict(system=row['system'], point=row['point'], method=row['method'],
                          init_source=row['init_source'], valid=valid)
        for component in ('E_DFT_Ha', 'E_corr_scaled_Ha', 'frozen_Dx_Ha', 'frozen_Dc_Ha'):
            a, b = row.get(component), limit.get(component) if limit else None
            derivative[component.replace('_Ha', '_reaction_kcal')] = (b-a)*base.KCAL if valid and a is not None and b is not None else None
        derivatives.append(derivative)
    statistics = []
    for method, source in sorted({(r['method'], r['init_source']) for r in selected}):
        sample = [r['error_kcal'] for r in errors if r['method'] == method and r['init_source'] == source and r['valid']]
        complete = len(sample) == 16 and set(cfg.systems) == set(base.REF) and set(base.POINTS).issubset(cfg.points)
        statistics.append(dict(method=method, init_source=source, n_valid=len(sample), n_requested=len(cfg.systems)*len([p for p in cfg.points if p != base.DL]),
            MUE_16=float(np.mean(np.abs(sample))) if complete else None,
            MSE_16=float(np.mean(sample)) if complete else None,
            RMSE_16=float(np.sqrt(np.mean(np.square(sample)))) if complete else None,
            MaxAE_16=float(np.max(np.abs(sample))) if complete else None,
            status='complete_16' if complete else 'Incomplete primary data: see convergence for SCF/OB/resource failure'))
    checks = []
    for case in cases:
        identity = dict(system=case['system'],point=case['point'])
        trials = case.get('ob_restarts', [])
        active = [r for r in case['candidates'] if r['stage']=='obdh' and r.get('valid')]
        c1_complete = bool(active and all(sum(t['source']==s and t['restart_converged'] for t in trials)>=getattr(cfg,'ob_restarts',0)
                                          for s in cfg.ob_init) and trials)
        checks.append(dict(**identity,check='C1_OB_perturb_reconverge',
            status='partial_no_formal_OB_stability' if c1_complete else 'missing_baseline_or_restart',
            execution_complete=c1_complete,pdf_complete=False,attempts=len(trials),
            lower_solution_found=any(t.get('lower_solution_found') for t in trials),
            different_density_found=any(t['restart_converged'] and not t['returned_same_density'] for t in trials),
            detail='Empirical OB restart test; not a full Hessian/stability certificate'))
        attempts = [r for r in case['candidates'] if (r['stage']=='scf' and r.get('neighbor_path'))
                    or (r['stage']=='obdh' and r['guess'].startswith('direct_neighbor_'))]
        direct = [r for r in attempts if r['stage']=='obdh']
        c2_complete = bool(direct and all(any(r['source']==source and r.get('valid') for r in direct)
                                         for source in cfg.ob_init))
        checks.append(dict(**identity,check='C2_neighbor_initialization',
            status='partial_no_formal_OB_stability' if c2_complete else 'missing_valid_direct_neighbor',
            execution_complete=c2_complete,pdf_complete=False,
            attempts=len(attempts),detail=getattr(cfg,'neighbor_selection','not_configured')))
        relevant = [r for r in errors if r['system']==case['system'] and r['point']==case['point']]
        if case['point']==base.DL:
            relevant = [r for r in selected if r['system']==case['system'] and r['point']==case['point']]
        c3_complete=bool(relevant and all(r['valid'] for r in relevant))
        checks.append(dict(**identity,check='C3_signed_errors',
            status='data_ready_for_scan_summary' if c3_complete else 'blocked_missing_baseline_energy',
            execution_complete=c3_complete,pdf_complete=False,
            attempts=len(relevant),detail='DL is the energy reference; signed errors apply to finite R'))
        relevant = [r for r in paired if r['system']==case['system'] and r['point']==case['point']]
        first_selected=next((r for r in selected if r['system']==case['system'] and r['point']==case['point'] and r['method']=='obdh_1st'),None)
        c4_complete=bool(first_selected and first_selected['valid'] and any(r['paired_valid'] and r['guess']==first_selected['guess'] for r in relevant))
        checks.append(dict(**identity,check='C4_same_DFT_seed_decomposition',
            status='paired_data_ready_branch_and_slope_review_required' if c4_complete else 'blocked_missing_selected_first_pair',
            execution_complete=c4_complete,pdf_complete=False,
            attempts=len(relevant),detail='Converged DFT -> one OBMP2 correction versus full OBDH from the same seed'))
    readme = [
        ('protocol', PROTOCOL),
        ('units', 'selected.energy and energy_components: Hartree; errors/statistics/reaction component differences: kcal/mol.'),
        ('obdh_1st', 'Converged matched DFT -> one pure OBMP2 correlation evaluation on its unchanged orbitals and orbital energies; E = E_DFT_SCF + alpha_c * E_corr_OBMP2.'),
        ('status', 'evaluated_one_shot is a completed one-shot calculation; ob_converged=False is expected and is not accepted-unconverged OBDH.'),
        ('selected', 'One row per method/source/geometry. Missing energy stays blank. All attempts remain in candidates.'),
        ('selection', 'PRIMARY BASELINE ONLY: current-Alpha SCF references. Neighbor/restart cannot replace selected. obdh_1st uses lowest DFT reference BEFORE correction. Full baseline OBDH uses lowest valid independent result per source.'),
        ('diagnostic_best', 'Lowest found including independent baseline, direct-neighbor and restarts; source means ancestry for direct neighbor. This sheet never supplies baseline errors/statistics.'),
        ('initialization_paths', 'current_uhf_scf and current_dft_scf are independent starts. neighbor_obdh is a direct OB start WITHOUT current-Alpha SCF. perturbed_baseline_obdh is a restart of the baseline OB solution.'),
        ('paired_relaxation', 'Same DFT seed: full OBDH minus obdh_1st. This is the effect of self-consistency including orbital/eigenvalue/amplitude changes, not a proof of an energy minimum.'),
        ('iterations', 'Energy, populations, gap, nonzero-amplitude denominators and t2 for the same evaluated state. dRMS is DIIS residual, not density change.'),
        ('frozen_derivatives', 'Component differences at the supplied state; not total derivatives of the self-consistent energy with respect to Alpha.'),
        ('OB_stability', 'not_tested: stable SCF seeds do not certify stability of the OB solution. Convergence and solution labels are reported separately.'),
        ('orbital_probes', 'Directional orbital-energy checks on converged OB states, fixed occupations/eigenvalues, recomputed amplitudes/density energy. A downhill direction is evidence against an energy minimum; absence of one is NOT a stability certificate or full Hessian.'),
        ('ob_restarts', 'C1: perturb the selected converged OB state and reconverge OB at the same Alpha; report final density/energy differences and lower solutions. This is distinct from the fixed-state energy probes.'),
        ('C1_C4', 'execution_complete describes available computations only. pdf_complete is FALSE: formal OB stability and scientific branch/slope interpretation are not certified by this driver. See scan_summary for cross-Alpha signed errors and directional slopes.'),
        ('reference', 'D_e=(E_supermolecule(R_DL)-E_supermolecule(R))*627.5094740631. No isolated-fragment replacement.'),
        ('populations', 'Mulliken fragment charge/spin; determinant S2. Labels never reject a result.'),
        ('fit_basis', 'DFT SCF and OB use the same OB auxiliary basis so the converged DFT term is the one evaluated by OBDH. UHF SCF keeps scf_aux.'),
    ]
    component_keys = {key for row in selected for key in row if key.startswith(('E_', 'frozen_D')) or 'reconstruction_error' in key}
    components = [{k: v for k,v in row.items() if k in component_keys or k in ('system','point','method','init_source','guess','valid')}
                  for row in selected]
    selected_view = [{k:v for k,v in row.items() if k not in component_keys} for row in selected]
    branch_map=[{k:r.get(k) for k in ('system','point','method','init_source','valid','guess','label','localization_fraction','fragment_charge_1','fragment_charge_2','fragment_spin_1','fragment_spin_2','s2','initialization_path','initial_orbital_id','initial_label','initial_to_final_label_changed')} for r in selected]
    frames = dict(statistics=statistics, selected=selected_view, diagnostic_best=diagnostic_best,branch_map=branch_map,
                  convergence=convergence, errors=errors,
                  paired_relaxation=paired, energy_components=components, frozen_derivatives=derivatives,
                  candidates=candidates, iterations=iterations, orbital_probes=probes, ob_restarts=restarts,
                  C1_C4=checks, neighbor_comparison=neighbor_comparison,
                  readme=[dict(item=k, meaning=v) for k, v in readme])
    frames['config'] = [dict(key=k, value=str(v)) for k,v in vars(cfg).items()]
    frames['config'] += [dict(key=f'resolved_ob_auxbasis.{system}',value=str(next(c['resolved_ob_auxbasis'] for c in cases if c['system']==system and 'resolved_ob_auxbasis' in c)))
                         for system in sorted({c['system'] for c in cases if 'resolved_ob_auxbasis' in c})]
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(cfg.outdir/f'{name}.csv', index=False)
    from openpyxl.styles import Border, Side, PatternFill, Font, Alignment
    border = Border(*(Side(style='thin', color='000000') for _ in range(4)))
    white = PatternFill(fill_type='solid', fgColor='FFFFFF')
    header_font = Font(name='Calibri', size=11, bold=True, color='000000')
    body_font = Font(name='Calibri', size=11, color='000000')
    header_align = Alignment(vertical='top', wrap_text=True)
    body_align = Alignment(vertical='top')
    with pd.ExcelWriter(cfg.outdir/'SIE4x4.xlsx', engine='openpyxl') as writer:
        for name, rows in frames.items():
            frame = pd.DataFrame(rows)
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            # Plain white cells and black borders. No frozen panes or colored guides.
            for cells in sheet:
                for cell in cells:
                    cell.border = border
                    cell.fill = white
                    cell.font = header_font if cell.row == 1 else body_font
                    cell.alignment = header_align if cell.row == 1 else body_align
            sheet.row_dimensions[1].height = 32
            for cells in sheet.columns:
                letter = cells[0].column_letter
                maxlen = max((len(str(c.value)) for c in cells if c.value is not None), default=12)
                sheet.column_dimensions[letter].width = min(38, max(14, maxlen + 2))
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=base.HERE/'inputs/input.json')
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--alpha', nargs=2, type=float, default=[.5, .4], metavar=('HF','PT2'))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=['obdh','obdh_1st'])
    p.add_argument('--ob-init', nargs='+', choices=['uhf','dft'], default=['uhf','dft'])
    p.add_argument('--neighbor-dir', nargs='*', type=Path, default=[])
    p.add_argument('--neighbor-mode',choices=['auto','off'],default='auto')
    p.add_argument('--neighbor-root',type=Path)
    p.add_argument('--neighbor-step',type=float,default=.1)
    p.add_argument('--prepare-neighbor',action='store_true',help='Prepare one supporting adjacent run when none is ready.')
    p.add_argument('--run-role',choices=['benchmark','neighbor_reference'],default='benchmark')
    p.add_argument('--confirm-c1-c4',action='store_true',help='Require available baseline/diagnostic executions; this is NOT certification of full PDF compliance or formal OB stability.')
    p.add_argument('--retry-incomplete',action='store_true',help='Recompute checkpoints missing primary results with the SAME config/code; complete primary cases are reused.')
    p.add_argument('--scf-policy', choices=['standard','multiguess'], default='multiguess')
    p.add_argument('--basis', default='aug-cc-pVDZ')
    p.add_argument('--grid', type=int, default=4)
    p.add_argument('--scf-aux', default='def2-universal-jkfit')
    p.add_argument('--ob-aux', default='auto', help='auto: PySCF MP2-fit auxiliary basis; also used for matched DFT SCF.')
    p.add_argument('--scf-tol', type=float, default=1e-9)
    p.add_argument('--scf-grad-tol', type=float, default=1e-6)
    p.add_argument('--scf-cycles', type=int, default=200)
    p.add_argument('--stability-cycles', type=int, default=6)
    p.add_argument('--ob-cycles', type=int, default=300)
    p.add_argument('--ob-tol', type=float, default=1e-6)
    p.add_argument('--density-tol', type=float, default=1e-5)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--max-memory', type=float, default=4000)
    p.add_argument('--ob-probe-directions', type=int, default=2,
                   help='Directional energy probes per converged OB candidate; 0 disables. Not a full Hessian.')
    p.add_argument('--ob-probe-step', type=float, default=1e-4)
    p.add_argument('--ob-restarts',type=int,default=2)
    p.add_argument('--ob-restart-angle',type=float,default=.05)
    p.add_argument('--restart-energy-tol',type=float,default=1e-7)
    p.add_argument('--systems', nargs='+', choices=list(base.REF), default=list(base.REF))
    p.add_argument('--points', nargs='+', choices=base.POINTS+[base.DL], default=base.POINTS+[base.DL])
    cfg = p.parse_args()
    if any(not np.isfinite(a) or not 0 <= a <= 1 for a in cfg.alpha):
        p.error('Alpha must be finite and between zero and one')
    if cfg.ob_cycles < 1 or cfg.scf_cycles < 1 or cfg.stability_cycles < 0 or cfg.max_memory <= 0 or cfg.ob_probe_directions < 0 or cfg.ob_probe_step <= 0 or cfg.threads < 1:
        p.error('Invalid cycle limit or memory budget')
    if any(not np.isfinite(v) or v <= 0 for v in [cfg.ob_tol,cfg.scf_tol,cfg.scf_grad_tol,cfg.max_memory,cfg.ob_probe_step,cfg.density_tol]):
        p.error('Tolerances must be positive')
    if cfg.ob_restarts < 0 or any(not np.isfinite(v) or v <= 0 for v in [cfg.neighbor_step,cfg.ob_restart_angle,cfg.restart_energy_tol]):
        p.error('Invalid neighbor/restart settings')
    if cfg.neighbor_step > 1:
        p.error('neighbor-step must not exceed one')
    if cfg.confirm_c1_c4 and (not {'obdh','obdh_1st'}.issubset(cfg.methods) or 'dft' not in cfg.ob_init or cfg.ob_restarts < 2 or cfg.neighbor_mode=='off'):
        p.error('C1-C4 confirmation requires obdh and obdh_1st, DFT init, two OB restarts, and neighbor mode auto')
    lib.num_threads(cfg.threads)
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    (cfg.outdir/'orbitals').mkdir(exist_ok=True)
    print('\nSIE SCALING V2: INDEPENDENT BASELINES AND SEPARATE DIAGNOSTICS',flush=True)
    print(f'Alpha={tuple(cfg.alpha)}; basis={cfg.basis}; DFT grid={cfg.grid}; threads={cfg.threads}; '
          f'PySCF memory budget={cfg.max_memory} MB. Output={cfg.outdir.resolve()}',flush=True)
    print('PRIMARY RESULT: fresh current-Alpha UHF/DFT SCF -> baseline OBDH. '
          'ONE-SHOT: converged matched DFT -> ONE pure OBMP2 evaluation; no orbital relaxation. '
          'C4 compares this with full OBDH from the identical DFT orbital state. '
          'NEIGHBOR/RESTART: additional diagnostics, NEVER replacements for primary energies. '
          'Formal OB Hessian stability is not implemented; no status means PDF C1-C4 is fully certified.',flush=True)
    from pycmf.OBDH import main as core, uobdh_solver as solver
    cfg.pyscf_version=pyscf.__version__
    cfg.pycmf_core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()+Path(solver.__file__).read_bytes()).hexdigest()
    configure_neighbors(cfg)
    for directory in cfg.neighbor_dir:
        if directory.resolve() == cfg.outdir.resolve():
            p.error('Neighbor directory must be a different run')
        if not (directory/'orbitals').is_dir():
            p.error(f'Neighbor directory has no saved orbitals: {directory}')
        if not neighbor_ready(directory,cfg):
            p.error(f'Neighbor has old layout, incompatible settings/code or unfinished checkpoints: {directory}')
    print(f'C2 neighbor selection: {cfg.neighbor_selection}; sources: {[str(p) for p in cfg.neighbor_dir]}',flush=True)
    if cfg.confirm_c1_c4 and not cfg.neighbor_dir:
        print('C2 has no completed source. Baselines will run independently; C2 coverage will be marked missing.',flush=True)
    data = json.loads(cfg.input.read_text())
    from pycmf.OBDH import main as core, uobdh_solver as solver
    config = base.clean_json({k: str(v) if isinstance(v, Path) else [str(x) for x in v] if k=='neighbor_dir' else v for k,v in vars(cfg).items() if k!='retry_incomplete'})
    config.update(protocol=PROTOCOL,pipeline_layout=LAYOUT, pyscf_version=pyscf.__version__,
        input_sha256=hashlib.sha256(cfg.input.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()+Path(base.__file__).read_bytes()).hexdigest(),
        pycmf_core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()+Path(solver.__file__).read_bytes()).hexdigest())
    config['neighbors'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in cfg.neighbor_dir for path in sorted((directory/'orbitals').glob('*.npz'))}
    fingerprint = digest(config)
    target = cfg.outdir/'config.json'
    if target.exists() and digest(json.loads(target.read_text())) != fingerprint:
        raise RuntimeError('Output contains a different config/code/neighbor snapshot; choose a new --outdir')
    target.write_text(json.dumps(config, indent=2))
    cases = []
    for system in cfg.systems:
        for point in cfg.points:
            stem = cfg.outdir/f'{system}__{point}'
            checkpoint = Path(str(stem)+'.json')
            if checkpoint.exists():
                case = json.loads(checkpoint.read_text())
                if case['fingerprint'] != fingerprint:
                    raise RuntimeError(f'Checkpoint mismatch: {checkpoint}')
                if not (cfg.retry_incomplete and not case.get('primary_complete',False)):
                    print(f'RESUME: reuse {system} {point}; primary_complete={case.get("primary_complete")}',flush=True)
                    cases.append(case)
                    report(cases, cfg)
                    continue
                print(f'RETRY: recompute incomplete case {system} {point}; existing successful cases stay cached.',flush=True)
                # Remove only this case's exports so a failed retry cannot expose
                # stale orbitals as a newly successful independent baseline.
                for old in (cfg.outdir/'orbitals').glob(f'{system}__{point}__*.npz'):
                    old.unlink()
            print(f'Running {system} {point} alpha={cfg.alpha}', flush=True)
            print(f'CASE FILES: full calculation log={stem}.log; scientific checkpoint={stem}.json',flush=True)
            case = dict(system=system, point=point, fingerprint=fingerprint, candidates=[], iterations=[], orbital_probes=[],ob_restarts=[])
            with Path(str(stem)+'.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    print(f'CASE: system={system}; geometry={point}; charge=+1; spin=1 (doublet); '
                          f'Alpha={tuple(cfg.alpha)}. All energies use this supermolecule. '
                          'The dissociation reference is the same supermolecule at R_DL, not isolated fragments.',flush=True)
                    atoms = data[system][point]
                    mol = base.build(atoms, cfg.basis)
                    mol.stdout, mol.max_memory = log, cfg.max_memory
                    groups = base.fragments(system, atoms)
                    geometry = digest(dict(atoms=atoms, basis=cfg.basis, charge=1, spin=1))
                    neighbors = load_neighbors(cfg.neighbor_dir, system, point, geometry)
                    case['neighbor_files'] = [n['path'] for n in neighbors]
                    converged_states = []
                    ob_aux = df.DF(mol, auxbasis=df.make_auxbasis(mol, mp2fit=True) if cfg.ob_aux=='auto' else cfg.ob_aux)
                    ob_aux.max_memory = cfg.max_memory
                    case['resolved_ob_auxbasis'] = ob_aux.auxbasis
                    case['xc_base'] = xc_string(cfg.alpha)
                    for source in ['uhf','dft']:
                        if source=='uhf' and ('obdh' not in cfg.methods or source not in cfg.ob_init):
                            continue
                        pool = seed_pool(mol, atoms, groups, source, cfg, ob_aux, [], case['candidates'])
                        for seed in pool:
                            mf = seed['mf']
                            seed.update(result_scope='baseline',initialization_path=f'current_{source}_scf',
                                        reference_id=orbital_identity(mf))
                            save_orbitals(cfg.outdir/'orbitals', system, point, source, seed['guess'], 'scf', mf, geometry, cfg.alpha)
                            if source=='dft':
                                component = energy_components(mf, seed['dm'], cfg.alpha)
                                case['candidates'].append(dict(stage='dft_base',source=source,guess=seed['guess'],
                                    energy=float(mf.e_tot), E_DFT_SCF_Ha=float(mf.e_tot), valid=True,
                                    status='converged_dft',scf_converged=True,scf_stable=True,
                                    result_scope='baseline',initialization_path='current_dft_scf',
                                    initial_orbital_id=seed['reference_id'],**component,
                                    **flat_diagnostics(mol,seed['dm'],groups,mf.mo_coeff,mf.mo_occ)))
                            if source=='dft' and 'obdh_1st' in cfg.methods:
                                row, trace, _ = ob_run(mol, groups, seed, source, cfg, ob_aux, one_shot=True)
                                case['candidates'].append(row)
                                case['iterations'] += [dict(method='obdh_1st', source=source, guess=seed['guess'], **r) for r in trace]
                            if 'obdh' in cfg.methods and source in cfg.ob_init:
                                row, trace, carrier = ob_run(mol, groups, seed, source, cfg, ob_aux)
                                case['candidates'].append(row)
                                case['iterations'] += [dict(method='obdh', source=source, guess=seed['guess'], **r) for r in trace]
                                if row['valid']:
                                    converged_states.append((row,carrier))
                                    save_orbitals(cfg.outdir/'orbitals', system, point, source, seed['guess'], 'obdh', carrier, geometry, cfg.alpha)
                                    if cfg.ob_probe_directions:
                                        try:
                                            case['orbital_probes'] += orbital_probes(mol, groups, carrier, source, seed['guess'], cfg, ob_aux)
                                        except Exception as exc:
                                            case['orbital_probes'].append(dict(source=source,guess=seed['guess'],exception=str(exc),ob_stability='probe_failed'))
                                            traceback.print_exc()
                            if source=='dft' and 'dh_matched' in cfg.methods:
                                from pyscf import mp
                                carrier = base.hf_orbital_carrier(mf)
                                carrier.converged = True
                                post = mp.UMP2(carrier).density_fit(auxbasis=ob_aux.auxbasis)
                                post.kernel()
                                case['candidates'].append(dict(stage='dh_matched',source=source,guess=seed['guess'],
                                    energy=float(mf.e_tot+cfg.alpha[1]*post.e_corr), E_DFT_SCF_Ha=float(mf.e_tot),
                                    valid=bool(np.isfinite(post.e_corr)),status='evaluated_one_shot',scf_converged=True,
                                    E_corr_MP2_Ha=float(post.e_corr), **flat_diagnostics(mol,seed['dm'],groups,mf.mo_coeff,mf.mo_occ)))
                        states=[(r,mf) for r,mf in converged_states if r['source']==source and r.get('result_scope')=='baseline']
                        if states:
                            anchor,export_mf=min(states,key=lambda pair:pair[0]['energy'])
                            save_orbitals(cfg.outdir/'orbitals',system,point,source,
                                'selected_baseline','obdh',export_mf,geometry,cfg.alpha,
                                export_role='selected_baseline_obdh')
                        # C2 ONLY: direct neighboring OB orbitals. Source records
                        # ancestry, NOT a fresh current-Alpha SCF calculation.
                        if 'obdh' in cfg.methods and source in cfg.ob_init:
                            for i, neighbor in enumerate(neighbors):
                                if neighbor['meta']['source'] != source or neighbor['meta']['state'] != 'obdh':
                                    continue
                                mf = factory(mol, source, cfg, ob_aux)
                                mf.mo_coeff, mf.mo_occ, mf.mo_energy = neighbor['coeff'], neighbor['occ'], neighbor['epsilon']
                                seed = dict(guess=f'direct_neighbor_{i}', mf=mf,
                                    dm=np.asarray(mf.make_rdm1()), is_scf=False, provenance=neighbor['path'],
                                    result_scope='diagnostic',initialization_path='neighbor_obdh')
                                seed['neighbor_alpha']=neighbor['meta']['alpha']
                                row, trace, carrier = ob_run(mol, groups, seed, source, cfg, ob_aux)
                                case['candidates'].append(row)
                                case['iterations'] += [dict(method='obdh', source=source, guess=seed['guess'], **r) for r in trace]
                                if row['valid']:
                                    if cfg.ob_probe_directions:
                                        try:
                                            case['orbital_probes'] += orbital_probes(mol, groups, carrier, source, seed['guess'], cfg, ob_aux)
                                        except Exception as exc:
                                            case['orbital_probes'].append(dict(source=source,guess=seed['guess'],exception=str(exc),ob_stability='probe_failed'))
                                            traceback.print_exc()
                    # C1 tests the baseline solution, never a rescued neighbor.
                    if cfg.ob_restarts and 'obdh' in cfg.methods:
                        for source in cfg.ob_init:
                            states = [(r,mf) for r,mf in converged_states if r['source']==source and r.get('result_scope')=='baseline']
                            if not states:
                                continue
                            anchor,carrier = min(states,key=lambda pair:pair[0]['energy'])
                            trials,rows,trace = restart_checks(mol,groups,carrier,anchor,cfg,ob_aux,system,point,geometry)
                            case['ob_restarts'] += trials
                            case['candidates'] += rows
                            case['iterations'] += trace
                    for source in cfg.ob_init:
                        baseline=choose([r for r in case['candidates'] if r['stage']=='obdh' and r['source']==source and r.get('result_scope')=='baseline'],'obdh')
                        diagnostic=choose([r for r in case['candidates'] if r['stage']=='obdh' and r['source']==source],'obdh')
                        print(f'CASE SELECTION: source={source}; baseline='+
                              (f'{baseline["energy"]:.12f} Ha from {baseline["guess"]}, type={baseline.get("label")}' if baseline else 'MISSING (no independent OBDH solution converged)')+
                              '; lowest found including diagnostics='+
                              (f'{diagnostic["energy"]:.12f} Ha via {diagnostic.get("initialization_path")}' if diagnostic else 'MISSING')+
                              '. Only the independent baseline enters selected/errors/statistics.',flush=True)
                except Exception as exc:
                    case['exception'] = str(exc)
                    traceback.print_exc()
            expected=[(method,source) for method in cfg.methods for source in (cfg.ob_init if method=='obdh' else ['dft'])]
            case['primary_complete']=not case.get('exception') and all(baseline_choice(case,method,source) is not None for method,source in expected)
            temporary = Path(str(stem)+'.tmp')
            temporary.write_text(json.dumps(base.clean_json(case), indent=2, allow_nan=False))
            temporary.replace(checkpoint)
            cases.append(case)
            report(cases, cfg)
    selected = report(cases, cfg)
    failed = sum(not row['valid'] for row in selected)
    incomplete = 0
    if cfg.confirm_c1_c4:
        coverage = pd.read_csv(cfg.outdir/'C1_C4.csv')
        incomplete = int((coverage.execution_complete!=True).sum())
        print(f'Available diagnostic executions incomplete: {incomplete}; see C1_C4. '
              'PDF C1-C4 scientific compliance is NOT certified: formal OB stability and '
              'branch/slope interpretation remain separate requirements.',flush=True)
    print(f'Saved {cfg.outdir}/SIE4x4.xlsx; missing selected energies: {failed}', flush=True)
    return int(bool(failed or incomplete))


if __name__ == '__main__':
    sys.exit(main())

