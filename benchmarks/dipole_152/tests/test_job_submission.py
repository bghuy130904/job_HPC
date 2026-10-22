"""Check alpha/output/job-name agreement without submitting to a real scheduler."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

JOB=Path(__file__).resolve().parents[1]/'jobs/obdh/dipole_sp.sh'


class SubmissionTests(unittest.TestCase):
    def test_alpha_pair_controls_submission_and_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);mock=root/'sbatch'
            mock.write_text('#!'+sys.executable+'\nimport json,os,sys\nprint(json.dumps(dict(args=sys.argv[1:],alpha=[os.environ["ALPHA_X"],os.environ["ALPHA_C"]],output=os.environ["OUTPUT_DIR"])))\n')
            mock.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],RESULT_ROOT=str(root/'results'))
            for key in ['ALPHA_X','ALPHA_C','OUTPUT_DIR']:env.pop(key,None)
            for pair in [None,('0.60','0.50'),('0.125','0.40')]:
                current=dict(env)
                if pair:current.update(ALPHA_X=pair[0],ALPHA_C=pair[1])
                result=subprocess.run(['bash',str(JOB),'--submit','--dipole','dm'],env=current,text=True,capture_output=True,check=True)
                data=json.loads(result.stdout);ax,ac=pair or ('0.50','0.40')
                tag=f'ax_{ax}_ac_{ac}'
                self.assertEqual(data['alpha'],[ax,ac])
                self.assertEqual(data['output'],str(root/'results/sp_scaling'/tag))
                self.assertIn('--job-name=wftdipolesp_'+tag,data['args'])
                self.assertEqual(data['args'][-3:],[str(JOB),'--dipole','dm'])
                self.assertTrue((root/'results/_output').is_dir())
                self.assertTrue((root/'results/_error').is_dir())
            bad=subprocess.run(['bash',str(JOB),'--submit'],env=dict(env,ALPHA_X='1.2'),capture_output=True,text=True)
            self.assertEqual(bad.returncode,2)
            self.assertIn('Alpha must',bad.stderr)

    def test_merge_uses_the_same_alpha_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job=root/'dipole_sp.sh';job.write_text(JOB.read_text())
            (root/'merge_sp.sh').write_text('printf "%s\\n" "$OUTPUT_DIR"\nprintf "%s\\n" "$@"\n')
            env=dict(os.environ,RESULT_ROOT=str(root/'results'),ALPHA_X='0.60',ALPHA_C='0.50')
            env.pop('OUTPUT_DIR',None)
            result=subprocess.run(['bash',str(job),'--merge','--no-excel'],env=env,text=True,capture_output=True,check=True)
            self.assertEqual(result.stdout.splitlines(),[str(root/'results/sp_scaling/ax_0.60_ac_0.50'),'--no-excel'])


if __name__=='__main__':unittest.main()
