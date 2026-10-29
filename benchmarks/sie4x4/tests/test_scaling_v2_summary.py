"""Cross-Alpha diagnostics must not silently use a different orbital branch."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import summarize_sie_v2 as summary


class SummaryTests(unittest.TestCase):
    def test_scan_rejects_mixed_basis_and_duplicate_alpha(self):
        for duplicate in [False,True]:
            with self.subTest(duplicate=duplicate),tempfile.TemporaryDirectory() as d:
                root=Path(d)
                for index in [0,1]:
                    folder=root/f'recalc_obdh_{index}';folder.mkdir()
                    cfg=dict(pipeline_layout=summary.LAYOUT,alpha=[.5,.4 if duplicate else .4+.1*index],
                             basis='6-31g' if index==0 or duplicate else 'cc-pvdz')
                    (folder/'config.json').write_text(json.dumps(cfg))
                with self.assertRaisesRegex(ValueError,'Duplicate Alpha' if duplicate else 'Incompatible basis'):
                    summary.read_scan(root)

    def test_paired_error_uses_only_selected_first_reference_at_each_geometry(self):
        selected=pd.DataFrame([dict(alpha='(0.5, 0.4)',system='He2_plus',point=p,
            method='obdh_1st',valid=True,guess='a') for p in ['R_1.0','dissociation_limit']])
        pairs=[]
        for guess,shift in [('a',0.),('wrong_b',-.5)]:
            for point,ef,eo in [('R_1.0',-5.,-4.99),('dissociation_limit',-4.9,-4.89)]:
                pairs.append(dict(alpha='(0.5, 0.4)',system='He2_plus',point=point,
                    guess=guess,paired_valid=True,E_full_Ha=ef+(shift if point=='R_1.0' else 0),
                    E_first_Ha=eo,label_changed=point=='R_1.0'))
        result=summary.paired_errors(pd.DataFrame(pairs),selected)
        self.assertEqual(len(result),1)
        self.assertEqual(result.iloc[0].guess_R,'a')
        self.assertAlmostEqual(result.iloc[0].full_same_start_error_kcal,.1*627.5094740631-56.9)
        self.assertFalse(result.iloc[0].same_final_branch_certified)

    def test_slope_stencil_type_change_is_not_certified_as_orbital_response(self):
        errors=pd.DataFrame([dict(alpha_x=x,alpha_c=.4,system='He2_plus',point='R_1.0',
            method='obdh',init_source='uhf',valid=True,error_kcal=2*x,
            label_R='loc' if x<.5 else 'deloc',label_DL='loc') for x in [.4,.5,.6]])
        derivative=pd.DataFrame([dict(alpha_x=.5,alpha_c=.4,system='He2_plus',point='R_1.0',
            method='obdh',init_source='uhf',valid=True,frozen_Dx_reaction_kcal=1.,frozen_Dc_reaction_kcal=3.)])
        result=summary.slope_comparison(errors,derivative)
        x=result[result.axis=='alpha_x'].iloc[0]
        self.assertAlmostEqual(x.observed_central_slope_kcal_per_alpha,2.)
        self.assertTrue(x.labels_change_in_stencil)
        self.assertFalse(x.same_branch_certified)
        self.assertIn('type_change',x.interpretation)
        c=result[result.axis=='alpha_c'].iloc[0]
        self.assertEqual(c.interpretation,'missing_baseline_points')


if __name__=='__main__':unittest.main()
