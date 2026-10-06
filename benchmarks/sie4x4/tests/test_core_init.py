"""Benchmark adapter passes a HF carrier to unmodified pyCMF."""
import argparse
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from pyscf import dft,gto
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_sie as sie


class BenchmarkAdapterTests(unittest.TestCase):
    def test_uks_orbitals_are_copied_to_hf_without_scf(self):
        mol=gto.M(atom='He 0 0 0; He 0 0 1.1',basis='cc-pvdz',charge=1,spin=1,verbose=0)
        seed=dft.UKS(mol)
        n=mol.nao
        w,v=np.linalg.eigh(mol.intor_symmetric('int1e_ovlp'))
        coeff=(v*(1/np.sqrt(w)))@v.T
        seed.mo_coeff=np.array([coeff,coeff]);seed.mo_energy=np.array([np.arange(n),np.arange(n)])
        seed.mo_occ=np.zeros((2,n));seed.mo_occ[0,:2]=1;seed.mo_occ[1,:1]=1
        original=seed.mo_coeff.copy()
        with patch.object(seed,'kernel',side_effect=AssertionError('Do not rerun seed SCF')):
            carrier=sie.hf_orbital_carrier(seed)
        self.assertFalse(hasattr(carrier,'xc'))
        np.testing.assert_array_equal(carrier.mo_coeff,original)
        carrier.mo_coeff[0][0,0]=2
        np.testing.assert_array_equal(seed.mo_coeff,original)

    def test_unmodified_solver_needs_no_new_attributes_and_grid_is_restored(self):
        mol=gto.M(atom='He 0 0 0; He 0 0 1.1',basis='cc-pvdz',charge=1,spin=1,verbose=0)
        seed=dft.UKS(mol)
        n=mol.nao
        w,v=np.linalg.eigh(mol.intor_symmetric('int1e_ovlp'));coeff=(v*(1/np.sqrt(w)))@v.T
        seed.mo_coeff=np.array([coeff,coeff]);seed.mo_energy=np.array([np.arange(n),np.arange(n)])
        seed.mo_occ=np.zeros((2,n));seed.mo_occ[0,:2]=1;seed.mo_occ[1,:1]=1
        test=self
        class OriginalAPI:
            def __init__(self,mf):
                test.assertFalse(hasattr(mf,'xc'))
                self.mo_coeff=mf.mo_coeff;self.mo_occ=mf.mo_occ
                self.gamma=mf.make_rdm1();self.converged=True
            def kernel(self):
                test.assertEqual(dft.UKS(mol).grids.level,4)
                print('Iter 2: E_tot=-5.000, E_corr=-0.1, dE=1.00e-08, df_ia=2.00e-07, dRMS=3.00e-06')
                return -5.
        cfg=argparse.Namespace(alpha=[.53,.39],grid=4,ob_cycles=100,ob_tol=1e-6)
        old_grid=dft.gen_grid.Grids.level
        with patch('pycmf.OBDH.OBDH_CL',OriginalAPI),contextlib.redirect_stdout(io.StringIO()):
            row=sie.ob_candidate(mol,[[0],[1]],('test',seed,seed.make_rdm1()),'uks','obdh',cfg)
        self.assertTrue(row['valid'],row)
        self.assertEqual(row['iteration']['cycle'],3)
        self.assertEqual(row['iteration']['effective_fia'],2e-7)
        self.assertEqual(dft.gen_grid.Grids.level,old_grid)

if __name__=='__main__':unittest.main()
