#!/usr/bin/env python3
"""Optional real SCF regression for the He-spectator metastable basin (cc-pVDZ).
Run separately from the fast unit tests. Prints the actual energies/populations.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from pyscf import lib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_sie as sie


def main():
    lib.num_threads(1)
    cfg=argparse.Namespace(basis='cc-pVDZ',grid=3,scf_aux='def2-universal-jkfit',
                           scf_tol=1e-9,scf_grad_tol=1e-6,scf_cycles=80,stability_cycles=6)
    atoms=json.loads((sie.HERE/'inputs/input.json').read_text())['H2_plus_He'][sie.DL]
    groups=sie.fragments('H2_plus_He',atoms);mol=sie.build(atoms,cfg.basis)
    seed=sie.factory(mol,'.53*HF+.47*B88,.61*LYP',cfg)
    seed,valid,_=sie.stabilize(seed,seed.get_init_guess(),cfg)
    if not valid: raise RuntimeError('Reference UKS seed did not converge stably')
    results=[]
    for source in ['pbe','pbe0']:
        mf,valid,stable=sie.stabilize(sie.factory(mol,source,cfg),seed.make_rdm1(),cfg,
                                   (np.array(seed.mo_coeff),np.array(seed.mo_occ)))
        diag=sie.diagnostics(mol,mf.make_rdm1(),groups,mf.mo_coeff,mf.mo_occ)
        gradient=float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ,mf.get_fock())))
        results.append(dict(source=source,energy=float(mf.e_tot),valid=valid,stable=stable,
                            orbital_gradient=gradient,**diag))
        # This PARTICULAR regression has an accessible lower basin with neutral He.
        # This is a test expectation, never a general S2/charge acceptance filter.
        if not (valid and diag['density_valid'] and gradient<=cfg.scf_grad_tol
                and mf.e_tot < -3.4 and abs(diag['fragment_charge'][2])<1e-3):
            raise AssertionError(f'Missed neutral-spectator basin: {results[-1]}')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
