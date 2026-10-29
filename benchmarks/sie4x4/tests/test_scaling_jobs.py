"""Exercise Slurm argument construction, array mapping and task-owned temp cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'jobs/obdh/job.sh'


class ArrayJobTests(unittest.TestCase):
    def setup_environment(self,root):
        bin_dir=root/'bin';bin_dir.mkdir()
        capture=root/'capture.jsonl'
        for name,body in {
            'sbatch':"with open(os.environ['CAPTURE'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nprint('6001' if any(a.startswith('--dependency=') for a in sys.argv) else '6000')\n",
            'python':"with open(os.environ['CAPTURE'],'a') as f: f.write(json.dumps(dict(args=sys.argv[1:],tmp=os.environ['TMPDIR'],pyscf_tmp=os.environ['PYSCF_TMPDIR']))+'\\n')\nraise SystemExit(int(os.environ.get('FAKE_EXIT','0')))\n"
        }.items():
            path=bin_dir/name
            path.write_text(f'#!{sys.executable}\nimport json,os,sys\n'+body)
            path.chmod(0o755)
        env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
        env.update(PATH=str(bin_dir)+os.pathsep+env['PATH'],CAPTURE=str(capture),
            RESULT_ROOT=str(root/'results'),SCAN_ROOT=str(root/'scan'),TMP_ROOT=str(root/'temp'),
            CODE_ROOT=str(root/'code'),SLURM_CPUS_PER_TASK='1',
            **{'BASH_FUNC_module%%':'() { return 0; }','BASH_FUNC_source%%':'() { return 0; }'})
        return env,capture

    def test_submit_creates_one_array_and_afterany_summary(self):
        with tempfile.TemporaryDirectory() as d:
            env,capture=self.setup_environment(Path(d))
            run=subprocess.run(['bash',str(SCRIPT),'--submit'],env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            calls=[json.loads(x) for x in capture.read_text().splitlines()]
            self.assertEqual(len(calls),2)
            self.assertIn('--array=0-80%3',calls[0])
            self.assertIn('--dependency=afterany:6000',calls[1])
            self.assertTrue(calls[1][-1].endswith('/merge.sh'))

    def test_mapping_and_cleanup_even_when_python_fails(self):
        for index,alpha in [(0,('0.1','0.1')),(8,('0.1','0.9')),(9,('0.2','0.1')),(80,('0.9','0.9'))]:
            with self.subTest(index=index),tempfile.TemporaryDirectory() as d:
                root=Path(d);env,capture=self.setup_environment(root)
                env.update(SLURM_JOB_ID='6100',SLURM_ARRAY_JOB_ID='6000',SLURM_ARRAY_TASK_ID=str(index),FAKE_EXIT='7')
                keep=root/'temp/6000/other_task';keep.mkdir(parents=True);(keep/'do_not_delete').write_text('keep')
                run=subprocess.run(['bash',str(SCRIPT)],env=env,capture_output=True,text=True)
                self.assertEqual(run.returncode,7,run.stderr)
                item=json.loads(capture.read_text().splitlines()[0]);args=item['args']
                i=args.index('--alpha');self.assertEqual(tuple(args[i+1:i+3]),alpha)
                self.assertEqual(item['tmp'],item['pyscf_tmp'])
                self.assertTrue(item['tmp'].startswith(str(root/'temp')))
                self.assertFalse(Path(item['tmp']).exists())
                self.assertTrue((keep/'do_not_delete').exists())


if __name__=='__main__':unittest.main()
