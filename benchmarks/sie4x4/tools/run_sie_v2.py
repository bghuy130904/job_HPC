#!/usr/bin/env python3
"""SIE scaling_v2: converged DFT -> one OBMP2 evaluation; paired full OBDH.

All energies use the supermolecule at R_DL. This driver uses the existing pyCMF
operators without editing the library. obdh_1st is an evaluated one-shot method,
not an unconverged full OBDH result. See README.md for the sheet definitions.
"""
import argparse
import contextlib
import hashlib
import json
import re
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


def save_orbitals(directory, system, point, source, guess, state, mf, geometry, alpha):
    directory.mkdir(exist_ok=True)
    token = hashlib.sha256(guess.encode()).hexdigest()[:12]
    path = directory / f'{system}__{point}__{source}__{token}__{state}.npz'
    metadata = dict(system=system, point=point, source=source, guess=guess, state=state,
                    geometry=geometry, alpha=list(alpha))
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
                result.append(dict(path=str(path.resolve()), meta=meta,
                                   coeff=saved['coeff'].copy(), occ=saved['occ'].copy(),
                                   epsilon=saved['epsilon'].copy()))
    return result


def seed_pool(mol, atoms, groups, source, cfg, ob_aux, neighbors, rows):
    names = ['fragment_average', 'minao', 'atom', 'huckel', 'localized_A', 'localized_B']
    if cfg.scf_policy == 'standard':
        names = ['minao']
    warm = {f'neighbor_{i}_{n["meta"]["state"]}': n for i, n in enumerate(neighbors)
            if n['meta']['source'] == source}
    names += list(warm)
    local = {}
    pool = []
    for name in names:
        row = dict(stage='scf', source=source, guess=name, valid=False,
                   scf_converged=False, scf_stable=False, neighbor_path=None)
        print(f'\n=== SCF source={source} guess={name} alpha={cfg.alpha} ===', flush=True)
        try:
            mf = factory(mol, source, cfg, ob_aux)
            orbital_guess = None
            if name in warm:
                n = warm[name]
                dm = mf.make_rdm1(n['coeff'], n['occ'])
                orbital_guess = n['coeff'], n['occ']
                row['neighbor_path'] = n['path']
            elif name.startswith('localized') or name == 'fragment_average':
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
    return pool


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
    row = dict(stage=method, source=source, guess=seed['guess'], valid=False,
               evaluation_complete=False, ob_converged=False,
               scf_converged=True if seed.get('is_scf', True) else None,
               scf_stable=True if seed.get('is_scf', True) else None, ob_stability='not_tested',
               evaluation_mode='dft_converged_then_obmp2_one_shot' if one_shot else 'self_consistent_obdh',
               residual_operator='pure_OBMP2' if one_shot else 'OBDH',
               seed_provenance=seed.get('provenance'))
    print(f'\n=== {method} source={source} guess={seed["guess"]} ===', flush=True)
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
        return row, trace.rows, carrier
    except Exception as exc:
        row.update(exception=str(exc), status='evaluation_failed')
        traceback.print_exc()
        return row, trace.rows, None


def choose(rows, method):
    good = [r for r in rows if r.get('valid') and np.isfinite(r.get('energy', np.nan))]
    if not good:
        return None
    # One-shot methods use the lowest DFT reference, not the lowest post-correction energy.
    key = 'E_DFT_SCF_Ha' if method in ('obdh_1st', 'dh_matched') else 'energy'
    return min(good, key=lambda row: row[key])


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
        with lib.temporary_env(dft.gen_grid.Grids, level=cfg.grid):
            post.kernel()
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
    return rows


def report(cases, cfg):
    candidates = [dict(system=c['system'], point=c['point'], **r) for c in cases for r in c['candidates']]
    iterations = [dict(system=c['system'], point=c['point'], **r) for c in cases for r in c['iterations']]
    probes = [dict(system=c['system'], point=c['point'], **r) for c in cases for r in c.get('orbital_probes', [])]
    selected, convergence, paired = [], [], []
    methods = ['dft_base'] + cfg.methods
    for case in cases:
        for method in methods:
            sources = cfg.ob_init if method == 'obdh' else ['dft']
            for source in sources:
                subset = [r for r in case['candidates'] if r['stage'] == method and r['source'] == source]
                if method in ('obdh_1st', 'dh_matched'):
                    reference = choose([r for r in case['candidates'] if r['stage']=='dft_base' and r['source']==source], 'dft_base')
                    eligible = [r for r in subset if reference and r['guess']==reference['guess']]
                else:
                    eligible = subset
                picked = choose(eligible, method)
                row = dict(system=case['system'], point=case['point'], method=method, init_source=source,
                           alpha=str(tuple(cfg.alpha)), valid=bool(picked), energy=None,
                           status='missing_valid_result', guess=None)
                if picked:
                    row.update(picked)
                selected.append(row)
                failed_scf = [r for r in case['candidates'] if r['stage'] == 'scf' and r['source'] == source]
                convergence.append(dict(system=case['system'], point=case['point'], method=method,
                    init_source=source, valid=bool(picked), status=row['status'],
                    n_scf_attempts=len(failed_scf), n_scf_valid=sum(r['valid'] for r in failed_scf),
                    n_method_attempts=len(subset), n_method_valid=sum(r['valid'] for r in subset),
                    failure_stage=None if picked else 'scf' if not any(r['valid'] for r in failed_scf) else 'method',
                    ob_converged=row.get('ob_converged'), evaluation_complete=row.get('evaluation_complete')))
        first = [r for r in case['candidates'] if r['stage'] == 'obdh_1st']
        for one in first:
            full = next((r for r in case['candidates'] if r['stage'] == 'obdh' and r['source'] == 'dft'
                         and r['guess'] == one['guess']), None)
            valid = bool(one['valid'] and full and full['valid'])
            paired.append(dict(system=case['system'], point=case['point'], guess=one['guess'],
                paired_valid=valid, E_first_Ha=one.get('energy'), E_full_Ha=full.get('energy') if full else None,
                delta_full_minus_first_Ha=full['energy']-one['energy'] if valid else None,
                delta_DFT_Ha=full['E_DFT_Ha']-one['E_DFT_Ha'] if valid else None,
                delta_corr_scaled_Ha=full['E_corr_scaled_Ha']-one['E_corr_scaled_Ha'] if valid else None,
                first_label=one.get('label'), full_label=full.get('label') if full else None))
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
            status='complete_16' if complete else 'Not convergence / incomplete data'))
    readme = [
        ('protocol', PROTOCOL),
        ('units', 'selected.energy and energy_components: Hartree; errors/statistics/reaction component differences: kcal/mol.'),
        ('obdh_1st', 'Converged matched DFT -> one pure OBMP2 correlation evaluation on its unchanged orbitals and orbital energies; E = E_DFT_SCF + alpha_c * E_corr_OBMP2.'),
        ('status', 'evaluated_one_shot is a completed one-shot calculation; ob_converged=False is expected and is not accepted-unconverged OBDH.'),
        ('selected', 'One row per method/source/geometry. Missing energy stays blank. All attempts remain in candidates.'),
        ('selection', 'obdh_1st/dh_matched use lowest converged stable DFT reference. Full OBDH uses lowest valid converged result per source. No combined all source.'),
        ('paired_relaxation', 'Same DFT seed: full OBDH minus obdh_1st. This is the effect of self-consistency including orbital/eigenvalue/amplitude changes, not a proof of an energy minimum.'),
        ('iterations', 'Energy, populations, gap, nonzero-amplitude denominators and t2 for the same evaluated state. dRMS is DIIS residual, not density change.'),
        ('frozen_derivatives', 'Component differences at the supplied state; not total derivatives of the self-consistent energy with respect to Alpha.'),
        ('OB_stability', 'not_tested: stable SCF seeds do not certify stability of the OB solution. Convergence and solution labels are reported separately.'),
        ('orbital_probes', 'Directional orbital-energy checks on converged OB states, fixed occupations/eigenvalues, recomputed amplitudes/density energy. A downhill direction is evidence against an energy minimum; absence of one is NOT a stability certificate or full Hessian.'),
        ('reference', 'D_e=(E_supermolecule(R_DL)-E_supermolecule(R))*627.5094740631. No isolated-fragment replacement.'),
        ('populations', 'Mulliken fragment charge/spin; determinant S2. Labels never reject a result.'),
        ('fit_basis', 'DFT SCF and OB use the same OB auxiliary basis so the converged DFT term is the one evaluated by OBDH. UHF SCF keeps scf_aux.'),
    ]
    component_keys = {key for row in selected for key in row if key.startswith(('E_', 'frozen_D')) or 'reconstruction_error' in key}
    components = [{k: v for k,v in row.items() if k in component_keys or k in ('system','point','method','init_source','guess','valid')}
                  for row in selected]
    selected_view = [{k:v for k,v in row.items() if k not in component_keys} for row in selected]
    frames = dict(statistics=statistics, selected=selected_view, convergence=convergence, errors=errors,
                  paired_relaxation=paired, energy_components=components, frozen_derivatives=derivatives,
                  candidates=candidates, iterations=iterations, orbital_probes=probes,
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
    p.add_argument('--systems', nargs='+', choices=list(base.REF), default=list(base.REF))
    p.add_argument('--points', nargs='+', choices=base.POINTS+[base.DL], default=base.POINTS+[base.DL])
    cfg = p.parse_args()
    for directory in cfg.neighbor_dir:
        if directory.resolve() == cfg.outdir.resolve():
            p.error('Neighbor directory must be a different, completed run')
        if not (directory/'orbitals').is_dir():
            p.error(f'Neighbor directory has no saved orbitals: {directory}')
    if any(not np.isfinite(a) or not 0 <= a <= 1 for a in cfg.alpha):
        p.error('Alpha must be finite and between zero and one')
    if cfg.ob_cycles < 1 or cfg.scf_cycles < 1 or cfg.stability_cycles < 0 or cfg.max_memory <= 0 or cfg.ob_probe_directions < 0 or cfg.ob_probe_step <= 0 or cfg.threads < 1:
        p.error('Invalid cycle limit or memory budget')
    if any(not np.isfinite(v) or v <= 0 for v in [cfg.ob_tol,cfg.scf_tol,cfg.scf_grad_tol,cfg.max_memory,cfg.ob_probe_step,cfg.density_tol]):
        p.error('Tolerances must be positive')
    lib.num_threads(cfg.threads)
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    data = json.loads(cfg.input.read_text())
    from pycmf.OBDH import main as core, uobdh_solver as solver
    config = base.clean_json({k: str(v) if isinstance(v, Path) else [str(x) for x in v] if k=='neighbor_dir' else v for k,v in vars(cfg).items()})
    config.update(protocol=PROTOCOL, pyscf_version=pyscf.__version__,
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
                cases.append(case)
                report(cases, cfg)
                continue
            print(f'Running {system} {point} alpha={cfg.alpha}', flush=True)
            case = dict(system=system, point=point, fingerprint=fingerprint, candidates=[], iterations=[], orbital_probes=[])
            with Path(str(stem)+'.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    atoms = data[system][point]
                    mol = base.build(atoms, cfg.basis)
                    mol.stdout, mol.max_memory = log, cfg.max_memory
                    groups = base.fragments(system, atoms)
                    geometry = digest(dict(atoms=atoms, basis=cfg.basis, charge=1, spin=1))
                    neighbors = load_neighbors(cfg.neighbor_dir, system, point, geometry)
                    ob_aux = df.DF(mol, auxbasis=df.make_auxbasis(mol, mp2fit=True) if cfg.ob_aux=='auto' else cfg.ob_aux)
                    ob_aux.max_memory = cfg.max_memory
                    case['resolved_ob_auxbasis'] = ob_aux.auxbasis
                    case['xc_base'] = xc_string(cfg.alpha)
                    for source in ['uhf','dft']:
                        if source=='uhf' and ('obdh' not in cfg.methods or source not in cfg.ob_init):
                            continue
                        pool = seed_pool(mol, atoms, groups, source, cfg, ob_aux, neighbors, case['candidates'])
                        for seed in pool:
                            mf = seed['mf']
                            save_orbitals(cfg.outdir/'orbitals', system, point, source, seed['guess'], 'scf', mf, geometry, cfg.alpha)
                            if source=='dft':
                                component = energy_components(mf, seed['dm'], cfg.alpha)
                                case['candidates'].append(dict(stage='dft_base',source=source,guess=seed['guess'],
                                    energy=float(mf.e_tot), E_DFT_SCF_Ha=float(mf.e_tot), valid=True,
                                    status='converged_dft',scf_converged=True,scf_stable=True,**component,
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
                        # Direct OB continuation: retain the neighboring OB orbitals;
                        # do not collapse them back to a UHF/UKS SCF basin first.
                        if 'obdh' in cfg.methods and source in cfg.ob_init:
                            for i, neighbor in enumerate(neighbors):
                                if neighbor['meta']['source'] != source or neighbor['meta']['state'] != 'obdh':
                                    continue
                                mf = factory(mol, source, cfg, ob_aux)
                                mf.mo_coeff, mf.mo_occ, mf.mo_energy = neighbor['coeff'], neighbor['occ'], neighbor['epsilon']
                                seed = dict(guess=f'direct_neighbor_{i}', mf=mf,
                                    dm=np.asarray(mf.make_rdm1()), is_scf=False, provenance=neighbor['path'])
                                row, trace, carrier = ob_run(mol, groups, seed, source, cfg, ob_aux)
                                case['candidates'].append(row)
                                case['iterations'] += [dict(method='obdh', source=source, guess=seed['guess'], **r) for r in trace]
                                if row['valid']:
                                    save_orbitals(cfg.outdir/'orbitals', system, point, source, seed['guess'], 'obdh', carrier, geometry, cfg.alpha)
                                    if cfg.ob_probe_directions:
                                        try:
                                            case['orbital_probes'] += orbital_probes(mol, groups, carrier, source, seed['guess'], cfg, ob_aux)
                                        except Exception as exc:
                                            case['orbital_probes'].append(dict(source=source,guess=seed['guess'],exception=str(exc),ob_stability='probe_failed'))
                                            traceback.print_exc()
                except Exception as exc:
                    case['exception'] = str(exc)
                    traceback.print_exc()
            temporary = Path(str(stem)+'.tmp')
            temporary.write_text(json.dumps(base.clean_json(case), indent=2, allow_nan=False))
            temporary.replace(checkpoint)
            cases.append(case)
            report(cases, cfg)
    selected = report(cases, cfg)
    failed = sum(not row['valid'] for row in selected)
    print(f'Saved {cfg.outdir}/SIE4x4.xlsx; missing selected energies: {failed}', flush=True)
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())
