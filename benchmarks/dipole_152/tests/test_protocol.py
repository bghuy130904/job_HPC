"""Scientific validity and reporting regressions for the shared dipole protocol."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_dipole as run
import report_dipole as report


class ProtocolTests(unittest.TestCase):
    def test_benchmark_defaults_to_nested_multiguess(self):
        args=run.parser().parse_args(['--outdir','unused'])
        self.assertEqual(args.scf_policy,'multiguess')
        for job in (Path(__file__).resolve().parents[1]/'jobs').glob('*/dipole_*.sh'):
            self.assertIn('${SCF_POLICY:-multiguess}',job.read_text())

    def calculator(self,invalid=False,nonlinear=False):
        calc=run.Calculation.__new__(run.Calculation)
        calc.nuclear=np.array([4.,5.,6.])
        calc.cfg=dict(fields=[1e-4,5e-5],ff_tol=1e-5,curvature_warn=1e-5)
        def evaluate(method,axis=None,field=0.):
            slope=1.+(field*field*1e8 if nonlinear else 0.)
            return dict(valid=not(invalid and axis=='x' and field>0),energy_Eh=-10.+field*slope),None
        calc.evaluate=evaluate
        return calc

    def test_finite_field_sign_and_nuclear_term_once(self):
        from pyscf.data import nist
        rows=self.calculator().finite_fields('uhf')
        self.assertTrue(all(r['valid'] for r in rows))
        np.testing.assert_allclose(rows[0]['vector_D'],np.array([3.,4.,5.])*nist.AU2DEBYE,rtol=1e-10)
        self.assertEqual(rows[0]['step_check'],'passed')

    def test_failed_field_cannot_produce_accepted_dipole(self):
        rows=self.calculator(invalid=True).finite_fields('obmp2')
        self.assertTrue(all(not r['valid'] for r in rows))
        self.assertIsNotNone(rows[0]['magnitude_D']) # Preserve last iterate separately.

    def test_field_step_disagreement_not_hidden_by_small_curvature(self):
        rows=self.calculator(nonlinear=True).finite_fields('obmp2')
        self.assertTrue(all(not r['valid'] and r['step_check']=='failed' for r in rows))
        self.assertTrue(all(abs(a['curvature_Eh'])<1e-10 for r in rows for a in r['axes'].values()))

    def test_natural_occupations_use_overlap_metric(self):
        class Mol:
            nelec=(1,0)
            def nao_nr(self):return 2
            def intor_symmetric(self,name):return np.diag([2.,.5])
        dm=np.array([[[.5,0],[0,0]],np.zeros((2,2))])
        d=run.density_diagnostics(Mol(),dm)
        self.assertTrue(d['density_valid']);self.assertAlmostEqual(d['occupation_max'],1.)
        self.assertEqual(d['occupations_outside_0_1'],0)

    def test_scf_selects_energy_not_dipole_or_spin(self):
        class MF:
            def kernel(self):pass
        calc=run.Calculation.__new__(run.Calculation)
        calc.cfg=dict(scf_policy='multiguess',guesses=['a','b'],init_guess='a',stability_cycles=0)
        calc.refs={};calc.candidates=[]
        class Stable(MF):
            converged=True;mo_coeff=None;mo_occ=None
            def stability(self,**kw):return None,None,True,None
        pool=[Stable(),Stable()]
        def factory(source):return pool.pop(0)
        calc.factory=factory
        def info(mf):
            return dict(valid=True,energy_Eh=-2 if mf.init_guess=='a' else -1,s2=5 if mf.init_guess=='a' else .75)
        calc.scf_info=info
        _,selected=calc.reference('uhf')
        self.assertEqual(selected['energy_Eh'],-2)
        self.assertEqual(sum(c['selected'] for c in calc.candidates),1)

    def test_failed_reference_has_no_lowest_energy_fallback(self):
        calc=run.Calculation.__new__(run.Calculation)
        calc.cfg=dict(scf_policy='standard',init_guess='minao')
        calc.refs={};calc.candidates=[]
        class MF:
            def kernel(self):pass
        calc.factory=lambda source:MF()
        calc.scf_info=lambda mf:dict(valid=False,energy_Eh=-99,scf_converged=False)
        state,record=calc.reference('uhf')
        self.assertIsNone(state);self.assertFalse(record['valid'])

    def test_multiguess_checks_and_follows_stability_inside_each_seed(self):
        calc=run.Calculation.__new__(run.Calculation)
        calc.cfg=dict(scf_policy='multiguess',guesses=['a','b'],stability_cycles=2)
        calc.refs={};calc.candidates=[]
        events=[]
        class MF:
            converged=True;mo_coeff='initial';mo_occ='occupied'
            def __init__(self):self.checks=0
            def kernel(self,**kw):events.append((self.init_guess,'scf',kw.get('mo_coeff')))
            def newton(self):return self
            def stability(self,**kw):
                events.append((self.init_guess,'stability',None));self.checks+=1
                return 'unstable_orbitals',None,self.checks>1,None
        calc.factory=lambda source:MF()
        calc.scf_info=lambda mf:dict(valid=True,energy_Eh=-1.)
        calc.reference('uhf')
        for guess in ['a','b']:
            self.assertEqual([e[1:] for e in events if e[0]==guess],
                             [('scf',None),('stability',None),('scf','unstable_orbitals'),('stability',None)])
        self.assertTrue(all(c['valid'] and c['stable'] for c in calc.candidates))

    def test_report_blanks_failed_values_and_keeps_analytic_separate(self):
        meta=dict(config=dict(methods=['uhf','obmp2'],dipole='both',fields=[1e-4]),
                  requested_indices=[0,1],inputs=[dict(name='A'),dict(name='B')])
        case=dict(index=0,molecule='A',zero=[dict(method='uhf',valid=True,energy_Eh=-1,
            densities=dict(analytic=dict(valid=True,magnitude_D=0.,vector_D=[0,0,0])))],
            finite_field=dict(obmp2=[dict(field_au=1e-4,valid=False,magnitude_D=123.,vector_D=[123.,0,0])]))
        tables=report.rows_for(meta,[case],{'A':1.,'B':1.})
        self.assertEqual(len(tables['Results']),10)
        row=next(r for r in tables['Results'] if r['molecule']=='A' and r['method']=='uhf' and r['estimator']=='analytic')
        self.assertEqual(row['dipole_D'],0.) # Real zero is not missing.
        ff=next(r for r in tables['Results'] if r['molecule']=='A' and r['method']=='obmp2' and r['estimator']=='finite_field')
        self.assertIsNone(ff['dipole_D'])
        self.assertEqual(next(r for r in tables['Dipole_vectors'] if r['method']=='obmp2' and r['molecule']=='A' and r['estimator']=='finite_field')['last_magnitude_D'],123.)
        self.assertTrue(all(s['n_requested']==2 for s in tables['Statistics']))
        self.assertTrue(all(s['RMSE_complete_pct'] is None for s in tables['Statistics']))
        uhf_ff=next(r for r in tables['Results'] if r['method']=='uhf' and r['estimator']=='finite_field')
        self.assertIsNone(uhf_ff['dipole_D'])

    def test_empty_merge_reference_keeps_statistics_blank(self):
        meta=dict(config=dict(methods=['uhf'],dipole='dm',fields=[1e-4]),requested_indices=[0],inputs=[dict(name='A')])
        tables=report.rows_for(meta,[],{})
        self.assertEqual(tables['Statistics'][0]['n_paired'],0)
        self.assertIsNone(tables['Statistics'][0]['RMSE_available_pct'])

    def test_spin_reversal_at_ms_zero_preserves_branch(self):
        from types import SimpleNamespace
        left=SimpleNamespace(mo_coeff=(np.eye(2),np.eye(2)),mo_occ=(np.array([1,0]),np.array([0,1])))
        right=SimpleNamespace(mo_coeff=(np.eye(2),np.eye(2)),mo_occ=left.mo_occ[::-1])
        self.assertAlmostEqual(run.occupied_overlap(left,right,np.eye(2)),1.)

    def test_density_failure_does_not_erase_converged_energy(self):
        calc=run.Calculation.__new__(run.Calculation)
        calc.cfg=dict(dipole='dm',overlap_min=.95,alpha=[.53,.27]);calc.posts={}
        class Reference:
            e_tot=-1.
            def make_rdm1(self):return np.zeros((2,1,1))
        class PT:
            e_corr=-.05
            def make_rdm1(self,**kw):raise ValueError('density-only failure')
        calc.reference=lambda *args:(Reference(),dict(valid=True))
        calc.pt2=lambda mf:PT()
        calc.add_density=lambda r,name,dm:r['densities'].update({name:dict(valid=True,magnitude_D=0.)})
        with patch('sys.stderr'),patch('sys.stdout'):
            record,_=calc.evaluate('ump2')
        self.assertTrue(record['valid'])
        self.assertAlmostEqual(record['energy_Eh'],-1.05)
        self.assertIn('density_error',record)
        self.assertNotIn('t2',record['densities'])
        self.assertIn('determinant',record['densities'])

    def test_occupation_violation_is_diagnostic_not_hidden_filter(self):
        class Mol:
            nelec=(1,0)
            def nao_nr(self):return 2
            def intor_symmetric(self,name):return np.eye(2)
        dm=np.array([np.diag([-.05,1.05]),np.zeros((2,2))])
        result=run.density_diagnostics(Mol(),dm)
        self.assertTrue(result['density_valid'])
        self.assertEqual(result['occupations_outside_0_1'],2)

    def test_field_cache_releases_tensors_but_keeps_energy_and_zero_seed(self):
        calc=run.Calculation.__new__(run.Calculation)
        seed=object();field_state=object()
        calc.posts={('obmp2',None,0.):({'energy_Eh':-1},seed),
                    ('obmp2','x',1e-4):({'energy_Eh':-1.1},field_state),
                    ('cc_amplitudes','x',1e-4):(object(),object())}
        calc.refs={('uhf',None,0.):(object(),{}),('uhf','x',1e-4):(object(),{})}
        calc.release_field('x',1e-4)
        self.assertIs(calc.posts[('obmp2',None,0.)][1],seed)
        self.assertEqual(calc.posts[('obmp2','x',1e-4)],({'energy_Eh':-1.1},None))
        self.assertNotIn(('cc_amplitudes','x',1e-4),calc.posts)
        self.assertNotIn(('uhf','x',1e-4),calc.refs)
        self.assertIn(('uhf',None,0.),calc.refs)

    def test_dh_scaling_and_named_functional(self):
        cfg={'alpha':[.6,.4]}
        self.assertIn('0.6*HF',run.xc_for('dh_matched',cfg))
        self.assertIn('0.53*HF',run.xc_for('b2plyp',cfg))
        self.assertNotEqual(run.xc_for('dh_matched',cfg),run.xc_for('b2plyp',cfg))


if __name__=='__main__':unittest.main()
