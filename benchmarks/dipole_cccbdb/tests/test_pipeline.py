"""Dataset/launcher integration checks; no HPC submission or PySCF required."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['HF', 'HCl', 'CO', 'HCN', 'H2O', 'NH3', 'H2CO', 'H2S', 'SO2', 'OCS']


class DatasetTests(unittest.TestCase):
    def test_input_and_reference_pairing(self):
        data = json.loads((ROOT / 'inputs/cccbdb_inputs.json').read_text())
        ref = json.loads((ROOT / 'references/ref_cccbdb.json').read_text())
        sources = json.loads((ROOT / 'references/cccbdb_sources.json').read_text())
        self.assertEqual(list(data), NAMES)  # Changes affect array indices.
        self.assertEqual(set(data), set(ref))
        self.assertEqual(set(data), set(sources['molecules']))
        nuclear_charge = dict(H=1, C=6, N=7, O=8, F=9, S=16, Cl=17)
        for name, props in data.items():
            self.assertEqual((props['charge'], props['spin'], props['multiplicity']), (0, 0, 1))
            self.assertEqual(props['unit'], 'Angstrom')
            ne = sum(nuclear_charge[a] for a, xyz in props['geometry'])
            self.assertEqual(ne % 2, 0)
            for atom, xyz in props['geometry']:
                self.assertEqual(len(xyz), 3)
                self.assertTrue(all(math.isfinite(v) for v in xyz))
            self.assertTrue(isinstance(ref[name], (int, float)) and math.isfinite(ref[name]) and ref[name] > 0)
            self.assertEqual(props['geometry_source'], sources['molecules'][name]['geometry_url'])
            self.assertEqual(props['cas'], sources['molecules'][name]['cas'])

    def test_reference_character_is_preserved(self):
        sources = json.loads((ROOT / 'references/cccbdb_sources.json').read_text())['molecules']
        self.assertEqual(sources['HCl']['dipole_kind'], 'mu_e')
        self.assertEqual(sources['H2O']['dipole_kind'], 'mu_e')
        self.assertEqual(sources['SO2']['dipole_kind'], 'mu_0')
        self.assertEqual(sources['CO']['dipole_kind'], 'vibrational_character_unspecified')


class EntryPointTests(unittest.TestCase):
    def test_dataset_defaults_and_cli_overrides(self):
        spec = importlib.util.spec_from_file_location('cccbdb_test_driver', ROOT / 'tools/cccbdb_driver.py')
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        import run_dipole

        def capture(methods):
            return run_dipole.parser(methods).parse_args()

        for extra, policy in [([], 'standard'), (['--scf-policy', 'multiguess'], 'multiguess')]:
            argv = ['entry.py', '--outdir', '/tmp/cccbdb-test', *extra]
            with patch.object(sys, 'argv', argv), patch.object(driver, 'run_main', side_effect=capture):
                args = driver.main(['uhf', 'obmp2'])
                self.assertIs(sys.argv, argv)
            self.assertEqual(args.input, ROOT / 'inputs/cccbdb_inputs.json')
            self.assertEqual(args.ref_json, ROOT / 'references/ref_cccbdb.json')
            self.assertEqual(args.methods, ['uhf', 'obmp2'])
            self.assertEqual(args.scf_policy, policy)
        with patch.object(sys, 'argv', ['entry.py', '--outdir', '/tmp/cccbdb-test',
                                       '--input', '/tmp/alternate.json', '--ref-json', '/tmp/ref.json']), \
                patch.object(driver, 'run_main', side_effect=capture):
            args = driver.main(['uhf'])
        self.assertEqual(args.input, Path('/tmp/alternate.json'))
        self.assertEqual(args.ref_json, Path('/tmp/ref.json'))


class SubmissionTests(unittest.TestCase):
    def test_submission_freezes_parameters_and_creates_scheduler_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mock = root / 'sbatch'
            mock.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                            'print(json.dumps(dict(args=sys.argv[1:],alpha=[os.environ["ALPHA_X"],'
                            'os.environ["ALPHA_C"]],output=os.environ["OUTPUT_DIR"],'
                            'code_root=os.environ["CODE_ROOT"])))\n')
            mock.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       RESULT_ROOT=str(root / 'results'), CODE_ROOT='/home/giahuy/Code/job',
                       ALPHA_X='0.125', ALPHA_C='0.40')
            env.pop('OUTPUT_DIR', None)
            for family in ('obdh', 'dft', 'ccsd'):
                job = ROOT / 'jobs' / family / 'job.sh'
                subprocess.run(['bash', '-n', str(job)], check=True)
                self.assertIn('#SBATCH --array=0-9%3', job.read_text())
                run = subprocess.run(['bash', str(job), '--submit', '--dipole', 'dm'],
                                     env=env, text=True, capture_output=True, check=True)
                data = json.loads(run.stdout)
                suffix = 'scaling/ax_0.125_ac_0.40' if family == 'obdh' else 'v1'
                self.assertEqual(data['output'], str(root / 'results' / suffix))
                self.assertEqual(data['alpha'], ['0.125', '0.40'])
                self.assertIn('--job-name=' + family + '_cccbdb_ax_0.125_ac_0.40', data['args'])
                self.assertEqual(data['args'][-3:], [str(job), '--dipole', 'dm'])
                self.assertTrue((root / 'results/_output').is_dir())
                self.assertTrue((root / 'results/_error').is_dir())

    def test_inconsistent_alpha_outdir_and_missing_merge_are_rejected(self):
        job = ROOT / 'jobs/obdh/job.sh'
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, RESULT_ROOT=folder, ALPHA_X='0.60', ALPHA_C='0.30')
            env.pop('OUTPUT_DIR', None)
            for extra in [['--alpha', '0.2', '0.3'], ['--outdir', '/tmp/other']]:
                run = subprocess.run(['bash', str(job), '--submit', *extra],
                                     env=env, text=True, capture_output=True)
                self.assertEqual(run.returncode, 2)
                self.assertIn('parameters agree', run.stderr)
            run = subprocess.run(['bash', str(ROOT / 'jobs/obdh/merge.sh')],
                                 env=env, text=True, capture_output=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn('/scaling/ax_0.60_ac_0.30/dipole.meta.json', run.stderr)


if __name__ == '__main__':
    unittest.main()
