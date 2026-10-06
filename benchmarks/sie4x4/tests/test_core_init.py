"""Regression: KS orbitals must never make the HF BCH operator a KS operator."""
import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
from pyscf import dft,df,gto,scf
from pycmf.OBDH import OBDH_CL
import pycmf.OBDH.main as core


class SeedConversionTests(unittest.TestCase):
    def test_kernel_passes_hf_operator_and_preserves_ks_seed(self):
        mol=gto.M(atom='He 0 0 0; He 0 0 1.1',basis='cc-pvdz',charge=1,spin=1,verbose=0)
        seed=dft.UKS(mol)
        n=mol.nao
        seed.mo_coeff=np.array([np.eye(n),np.eye(n)])
        seed.mo_occ=np.zeros((2,n));seed.mo_occ[0,:2]=1;seed.mo_occ[1,:1]=1
        seed.mo_energy=np.array([np.arange(n),np.arange(n)])
        original=seed.mo_coeff.copy()
        ob=OBDH_CL(seed)
        def fake_solver(mp,mol,mf,xc_code,**kwargs):
            self.assertIsInstance(mf,scf.uhf.UHF)
            self.assertFalse(hasattr(mf,'xc'))
            np.testing.assert_array_equal(mf.mo_coeff,original)
            mf.mo_coeff[0][0,0]=2
            return -5.,-4.,np.zeros((2,n,n))
        with patch.object(core,'obmp2_iter',side_effect=fake_solver),contextlib.redirect_stdout(io.StringIO()):
            ob.kernel()
        np.testing.assert_array_equal(seed.mo_coeff,original)
        self.assertTrue(hasattr(seed,'xc'))

if __name__=='__main__': unittest.main()
