import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_sie as sie
from collect_SIE4x4 import parse


class ProtocolTests(unittest.TestCase):
    def test_fragments_do_not_merge_at_short_distance_or_rotation(self):
        data=json.loads((sie.HERE/'inputs/input.json').read_text())
        expected={'H2_plus_He':[[0],[1],[2]],'He2_plus':[[0],[1]],
                  'NH3_2_plus':[[0,2,3,4],[1,5,6,7]],'H2O_2_plus':[[0,2,3],[1,4,5]]}
        for system,points in data.items():
            for atoms in points.values():
                self.assertEqual(sie.fragments(system,atoms),expected[system])

    def test_no_spin_gate_and_no_nonconverged_fallback(self):
        valid=dict(energy=-3.,valid=True,s2=1.2)
        failed=dict(energy=-4.,valid=False,s2=.75)
        self.assertIs(sie.select([valid,failed]),valid)
        self.assertIsNone(sie.select([failed]))

    def test_stability_does_not_replace_convergence(self):
        class FailedSCF:
            converged=False
            def kernel(self,dm0): pass
            def make_rdm1(self): return np.zeros((2,1,1))
            def newton(self): return self
            def stability(self,**kw): raise AssertionError('Nonconverged SCF cannot be accepted')
        _,valid,stable=sie.stabilize(FailedSCF(),None,argparse.Namespace(stability_cycles=2))
        self.assertFalse(valid);self.assertFalse(stable)

    def test_fragment_density_keeps_interatomic_blocks(self):
        atoms=[dict(element='H',coordinates=[0,0,0]),dict(element='H',coordinates=[0,0,.8]),
               dict(element='H',coordinates=[0,0,5]),dict(element='H',coordinates=[0,0,5.8])]
        mol=sie.build(atoms,'sto-3g',charge=0,spin=0)
        block=np.array([[[.6,.2],[.2,.4]],[[.5,.1],[.1,.5]]])
        class Fragment:
            converged=True;verbose=0
            def kernel(self): pass
            def make_rdm1(self): return block
        with patch.object(sie,'factory',return_value=Fragment()):
            dm=sie.localized_density(mol,atoms,[[0,1],[2,3]],0,'uhf',argparse.Namespace(basis='sto-3g'))
        self.assertEqual(dm[0,0,1],.2)
        self.assertEqual(dm[1,2,3],.1)
        self.assertEqual(dm[0,0,2],0.)

    def test_orca_final_verdict_and_normal_termination_required(self):
        text='''SCF CONVERGED AFTER 12 CYCLES
Stability Analysis indicates an UNSTABLE HF/KS wave function
SCF CONVERGED AFTER 18 CYCLES
Stability Analysis indicates a STABLE HF/KS wave function
Total Energy : -5.000000 Eh
Expectation value of <S**2> : 1.100000
FINAL SINGLE POINT ENERGY -5.100000
ORCA TERMINATED NORMALLY
'''
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.out';path.write_text(text)
            self.assertTrue(parse(path)['valid'])
            path.write_text(text.replace('ORCA TERMINATED NORMALLY',''))
            self.assertFalse(parse(path)['valid'])
            path.write_text(text.replace('Stability Analysis indicates a STABLE HF/KS wave function',''))
            self.assertFalse(parse(path)['valid'])

    def test_partial_statistics_are_not_full_benchmark(self):
        cfg=argparse.Namespace(methods=['uhf'],systems=['He2_plus'],points=['R_1.0',sie.DL])
        cases=[dict(system='He2_plus',point=point,candidates=[dict(stage='uhf',source='uhf',energy=e,valid=True)])
               for point,e in [('R_1.0',-5.),(sie.DL,-4.9)]]
        with tempfile.TemporaryDirectory() as d:
            sie.summarize(cases,Path(d),cfg)
            frame=sie.pd.read_csv(Path(d)/'statistics.csv')
            self.assertTrue(frame.MUE_16.isna().all())
            self.assertTrue((frame.n_valid==1).all())
            self.assertFalse(frame.complete_SIE4x4.any())

if __name__=='__main__': unittest.main()
