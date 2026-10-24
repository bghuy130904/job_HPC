"""Scientific protocol checks for scaling_v2; requires PySCF and pyCMF."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from pyscf import df, lib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_sie_scaling as sie


class ReportingTests(unittest.TestCase):
    def test_missing_lowest_dft_evaluation_is_not_replaced_by_higher_reference(self):
        rows = [dict(stage='dft_base',source='dft',guess=g,energy=e,valid=True)
                for g,e in [('low',-5.),('high',-4.9)]]
        rows += [dict(stage='obdh_1st',source='dft',guess='low',valid=False),
                 dict(stage='obdh_1st',source='dft',guess='high',valid=True,
                      energy=-5.1,E_DFT_SCF_Ha=-4.9)]
        cfg = argparse.Namespace(methods=['obdh_1st'],ob_init=['dft'],
                                 systems=['He2_plus'],points=['R_1.0'],alpha=[.5,.4])
        with tempfile.TemporaryDirectory() as directory:
            cfg.outdir = Path(directory)
            result = sie.report([dict(system='He2_plus',point='R_1.0',candidates=rows,iterations=[])],cfg)
            first = next(r for r in result if r['method']=='obdh_1st')
            self.assertFalse(first['valid'])
            self.assertIsNone(first['energy'])
            stats = pd.read_csv(cfg.outdir/'statistics.csv')
            self.assertTrue(stats.MUE_16.isna().all())
            self.assertTrue((stats.n_valid==0).all())

    def test_neighbor_geometry_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'orbitals').mkdir()
            np.savez(root/'orbitals/He2_plus__R_1.0__test.npz',
                     metadata=json.dumps(dict(geometry='wrong')))
            with self.assertRaisesRegex(ValueError,'geometry/basis'):
                sie.load_neighbors([root],'He2_plus','R_1.0','correct')


@unittest.skipUnless(importlib.util.find_spec('pycmf'), 'pyCMF is required')
class FirstIterationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)
        cls.cfg = argparse.Namespace(alpha=[.5,.4],basis='6-31g',grid=1,
            scf_aux='def2-universal-jkfit',scf_tol=1e-10,scf_grad_tol=1e-6,
            scf_cycles=100,stability_cycles=2,ob_cycles=1,ob_tol=1e-6,
            max_memory=2000,ob_probe_directions=1,ob_probe_step=1e-4)
        data = json.loads((sie.base.HERE/'inputs/input.json').read_text())
        atoms = data['He2_plus']['R_1.0']
        cls.mol = sie.base.build(atoms,cls.cfg.basis)
        cls.mol.stdout = io.StringIO()
        cls.groups = sie.base.fragments('He2_plus',atoms)
        cls.aux = df.DF(cls.mol,auxbasis=df.make_auxbasis(cls.mol,mp2fit=True))
        cls.ks = sie.factory(cls.mol,'dft',cls.cfg,cls.aux)
        cls.ks.kernel()
        if not cls.ks.converged:
            raise AssertionError('DFT fixture failed to converge')

    def test_first_is_dft_scf_plus_one_obmp2_correction_on_unchanged_orbitals(self):
        from pycmf.OBDH import OBMP2_CL
        coeff = np.asarray(self.ks.mo_coeff).copy()
        eps = np.asarray(self.ks.mo_energy).copy()
        seed = dict(guess='minao',mf=self.ks,dm=np.asarray(self.ks.make_rdm1()))
        with contextlib.redirect_stdout(io.StringIO()):
            row,trace,carrier = sie.ob_run(self.mol,self.groups,seed,'dft',self.cfg,self.aux,True)
            independent = OBMP2_CL(sie.base.hf_orbital_carrier(self.ks))
            independent.with_df = self.aux
            independent.niter = 1
            independent.use_embed = independent.use_cl = independent.eval_IPEA = False
            independent.kernel()
        self.assertTrue(row['valid'],row.get('exception'))
        self.assertEqual(row['status'],'evaluated_one_shot')
        self.assertFalse(row['ob_converged'])
        self.assertEqual(len(trace),1)
        self.assertAlmostEqual(row['energy'],self.ks.e_tot+self.cfg.alpha[1]*independent.e_corr,places=10)
        self.assertAlmostEqual(row['E_DFT_Ha'],self.ks.e_tot,places=9)
        # pyCMF groups occupied/virtual columns and may permute columns within
        # either group. Test orbital identity and epsilon, allowing this ordering.
        overlap = self.mol.intor_symmetric('int1e_ovlp')
        for spin in range(2):
            correspondence = np.abs(coeff[spin].T @ overlap @ carrier.mo_coeff[spin])
            order = correspondence.argmax(axis=0)
            np.testing.assert_allclose(correspondence.max(axis=0),1.,atol=1e-10)
            np.testing.assert_allclose(carrier.mo_energy[spin],eps[spin][order],atol=1e-12)

    def test_hf_initialized_obdh_one_iteration_is_a_different_protocol(self):
        seed = dict(guess='minao',mf=self.ks,dm=np.asarray(self.ks.make_rdm1()))
        with contextlib.redirect_stdout(io.StringIO()):
            first,_,_ = sie.ob_run(self.mol,self.groups,seed,'dft',self.cfg,self.aux,True)
            hf = sie.factory(self.mol,'uhf',self.cfg,self.aux)
            hf.kernel()
            old,_,_ = sie.ob_run(self.mol,self.groups,
                dict(guess='minao',mf=hf,dm=np.asarray(hf.make_rdm1())),
                'uhf',self.cfg,self.aux,False)
        self.assertTrue(first['valid'])
        self.assertFalse(old['valid'])
        self.assertGreater(abs(first['energy']-old['energy']),1e-6)


if __name__ == '__main__':
    unittest.main()
