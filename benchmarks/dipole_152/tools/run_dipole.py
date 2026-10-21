#!/usr/bin/env python3
"""Shared dipole benchmark: explicit SCF policy, density definitions and checkpoints."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('uhf','ump2','obmp2','obdh','pbe','pbe0','b3lyp','b2plyp','dh_matched','ccsd','ccsd_t')
SCHEMA = 2


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True).encode()).hexdigest()


@contextlib.contextmanager
def directory_lock(path,timeout=60.,label='Metadata'):
    """Atomic mkdir exclusion, without network-filesystem advisory flock RPCs.

    Never steal a lock by age: a long calculation may still own it. On a hard
    kill, owner.json identifies the job to check before manual cleanup.
    """
    path=Path(path)
    print(f'{label}: acquiring lock {path}',flush=True)
    start=time.monotonic();next_notice=10.
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            elapsed=time.monotonic()-start
            if elapsed>=timeout:
                raise TimeoutError(f'{label} lock busy after {elapsed:.1f}s: {path}. Check {path / "owner.json"} and its job before removing a stale lock.')
            if elapsed>=next_notice:
                print(f'{label}: still waiting for lock ({elapsed:.0f}s): {path}',flush=True)
                next_notice+=10.
            time.sleep(.2)
    owner=path/'owner.json'
    try:
        owner.write_text(json.dumps(dict(pid=os.getpid(),host=os.uname().nodename,
            job_id=os.environ.get('SLURM_JOB_ID'),array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),
            created=time.time()))+'\n')
        print(f'{label}: lock acquired',flush=True)
        yield
    finally:
        owner.unlink(missing_ok=True)
        path.rmdir()


def metadata_lock(path,timeout=60.):
    return directory_lock(str(path)+'.d',timeout=timeout)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stem(index, name):
    return f'{index:03d}_'+re.sub(r'[^A-Za-z0-9_.-]','_',name)


def xc_for(method, cfg):
    if method in ('b2plyp','dh_matched'):
        # Named B2PLYP has fixed coefficients by definition. dh_matched uses CLI.
        ax,ac = (.53,.27) if method=='b2plyp' else cfg['alpha']
        return f'{ax}*HF + {1-ax}*B88, {1-ac}*LYP'
    return method


def ref_key(method):
    return method if method in ('pbe','pbe0','b3lyp','b2plyp','dh_matched') else 'uhf'


def gradient(mf):
    dm=mf.make_rdm1()
    f=mf.get_fock(dm=dm,level_shift_factor=0)
    return float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ,f)))


def occupied_overlap(left, right, overlap):
    def score(coeff,occ):
        vals=[]
        for c0,o0,c,o in zip(left.mo_coeff,left.mo_occ,coeff,occ):
            block=c0[:,np.asarray(o0)>0].conj().T @ overlap @ c[:,np.asarray(o)>0]
            vals.append(float(np.linalg.svd(block,compute_uv=False).min()) if block.size else 1.)
        return min(vals)
    value=score(right.mo_coeff,right.mo_occ)
    # A global alpha/beta interchange at M_s=0 is the same spin-free state.
    counts=lambda obj:[int(np.count_nonzero(o)) for o in obj.mo_occ]
    if counts(left)[0]==counts(left)[1] and counts(right)[0]==counts(right)[1]:
        value=max(value,score(right.mo_coeff[::-1],right.mo_occ[::-1]))
    return value


def density_diagnostics(mol, dm):
    """AO representation; natural occupations use S^(1/2) D S^(1/2)."""
    dm=np.asarray(dm)
    if dm.shape != (2,mol.nao_nr(),mol.nao_nr()):
        raise ValueError('Expected two AO spin density matrices')
    overlap=mol.intor_symmetric('int1e_ovlp')
    e,v=np.linalg.eigh(overlap)
    half=(v*np.sqrt(np.maximum(e,0)))@v.T
    spectra=np.concatenate([np.linalg.eigvalsh(half@((d+d.conj().T)*.5)@half) for d in dm])
    ne=np.einsum('sij,ji->s',dm,overlap).real
    herm=float(np.max(abs(dm-dm.conj().transpose(0,2,1))))
    valid=bool(np.isfinite(dm).all() and herm<1e-7 and np.max(abs(ne-mol.nelec))<1e-5)
    return dict(electrons_alpha=float(ne[0]),electrons_beta=float(ne[1]),
                expected_alpha=mol.nelec[0],expected_beta=mol.nelec[1],hermiticity_max=herm,
                occupation_min=float(spectra.min()),occupation_max=float(spectra.max()),
                occupations_outside_0_1=int(((spectra < -1e-6)|(spectra>1+1e-6)).sum()),density_valid=valid)


class IterationLog:
    pattern=re.compile(r'Iter (\d+):.*?dE=([^, ]+), df_ia=([^, ]+), dRMS=([^, ]+)')
    def __init__(self,stream): self.stream,self.last=stream,None
    def write(self,value):
        self.stream.write(value)
        for m in self.pattern.finditer(value):
            self.last=dict(cycles=int(m[1])+1,delta_energy=float(m[2]),effective_fia=float(m[3]),density_rms=float(m[4]))
        return len(value)
    def flush(self): self.stream.flush()
    def __getattr__(self,name): return getattr(self.stream,name)


class Calculation:
    def __init__(self,mol,cfg):
        self.mol,self.cfg=mol,cfg
        self.refs,self.posts={},{}
        self.candidates,self.points=[],[]
        self.h0=__import__('pyscf').scf.UHF(mol).get_hcore()
        with mol.with_common_orig((0,0,0)):
            self.r=mol.intor('int1e_r',comp=3)
        self.nuclear=np.einsum('i,ix->x',mol.atom_charges(),mol.atom_coords())

    def factory(self,source,h=None):
        from pyscf import scf,dft,df
        mf=scf.UHF(self.mol) if source=='uhf' else dft.UKS(self.mol)
        if source!='uhf':
            mf.xc=xc_for(source,self.cfg);mf.grids.level=self.cfg['grid']
        aux=self.cfg['scf_aux']
        if aux!='none': mf=mf.density_fit(auxbasis=None if aux=='auto' else aux)
        mf.conv_tol=self.cfg['scf_tol'];mf.conv_tol_grad=self.cfg['scf_grad_tol']
        mf.max_cycle=self.cfg['scf_cycles'];mf.verbose=4
        mf.check_convergence=lambda env: (abs(env['e_tot']-env['last_hf_e'])<=self.cfg['scf_tol']
                                            and env['norm_gorb']<=self.cfg['scf_grad_tol'])
        if h is not None: mf.get_hcore=lambda *a,**k:h
        return mf

    def scf_info(self,mf):
        info=dict(scf_converged=bool(mf.converged),energy_Eh=float(mf.e_tot),gradient=gradient(mf),
                  s2=float(mf.spin_square()[0]),**density_diagnostics(self.mol,mf.make_rdm1()))
        info['valid']=bool(info['scf_converged'] and info['density_valid'] and
                           np.isfinite(info['energy_Eh']) and info['gradient']<=self.cfg['scf_grad_tol'])
        return info

    def reference(self,source,axis=None,field=0.):
        key=(source,axis,field)
        if key in self.refs:return self.refs[key]
        if axis is not None:
            zero,_=self.reference(source)
            if zero is None:self.refs[key]=(None,dict(valid=False,error='No valid zero-field reference'));return self.refs[key]
            mf=self.factory(source,self.h0+field*self.r['xyz'.index(axis)])
            mf.kernel(dm0=zero.make_rdm1())
            # Continue the selected branch. Never run a fresh seed search in a field.
            if not mf.converged and self.cfg['scf_policy']!='standard':
                mf=mf.newton();mf.kernel(mo_coeff=mf.mo_coeff,mo_occ=mf.mo_occ)
            info=self.scf_info(mf)
            info['reference_overlap']=occupied_overlap(zero,mf,mf.get_ovlp())
            info['valid']=bool(info['valid'] and info['reference_overlap']>=self.cfg['overlap_min'])
            self.refs[key]=(mf if info['valid'] else None,info)
            return self.refs[key]
        guesses=self.cfg['guesses'] if self.cfg['scf_policy']=='multiguess' else [self.cfg['init_guess']]
        pool=[]
        for guess in guesses:
            row=dict(source=source,guess=guess,valid=False,stable=None)
            try:
                print(f'SCF START source={source} guess={guess}',flush=True)
                mf=self.factory(source);mf.init_guess=guess;mf.kernel()
                if self.cfg['scf_policy']!='standard':
                    if not mf.converged:
                        mf=mf.newton();mf.kernel(mo_coeff=mf.mo_coeff,mo_occ=mf.mo_occ)
                    if mf.converged:
                        stable=False
                        for attempt in range(self.cfg['stability_cycles']+1):
                            print(f'STABILITY source={source} guess={guess} attempt={attempt}',flush=True)
                            orb,_,stable,_=mf.stability(internal=True,external=False,return_status=True)
                            if stable or attempt==self.cfg['stability_cycles']:break
                            mf=mf.newton();mf.kernel(mo_coeff=orb,mo_occ=mf.mo_occ)
                            if not mf.converged:break
                        row['stable']=bool(stable)
                row.update(self.scf_info(mf))
                if self.cfg['scf_policy']!='standard':row['valid']=bool(row['valid'] and row['stable'])
                if row['valid']:pool.append((mf,row))
            except Exception as exc:
                row.update(error=f'{type(exc).__name__}: {exc}');traceback.print_exc()
            self.candidates.append(row)
            print(f'SCF END source={source} guess={guess} valid={row["valid"]} stable={row["stable"]}',flush=True)
        # SCF energy only; never choose a reference by MP2 energy, dipole or S2.
        selected=min(pool,key=lambda x:x[1]['energy_Eh']) if pool else (None,dict(valid=False,error='No converged admissible SCF reference'))
        for row in self.candidates:
            if row['source']==source:row['selected']=row is selected[1]
        self.refs[key]=selected
        return selected

    def add_density(self,record,name,dm,extra_valid=True):
        from pyscf import scf
        diag=density_diagnostics(self.mol,dm)
        vector=np.asarray(scf.hf.dip_moment(self.mol,dm,unit='Debye',verbose=0),float)
        record['densities'][name]=dict(vector_D=vector.tolist(),magnitude_D=float(np.linalg.norm(vector)),
            valid=bool(record['valid'] and extra_valid and diag['density_valid']),**diag)

    def pt2(self,mf):
        from pyscf import mp,df
        carrier=mf.to_hf() if hasattr(mf,'xc') else mf.copy()
        # Preserve canonical KS orbital eigenvalues for one-shot DH, no HF kernel.
        carrier.converged=True
        pt=mp.UMP2(carrier,frozen=self.cfg['frozen'])
        if self.cfg['pt2_df']:
            aux=self.cfg['mp2_aux'];aux=df.make_auxbasis(self.mol,mp2fit=True) if aux=='auto' else aux
            pt=pt.density_fit(auxbasis=aux)
        pt.kernel()
        return pt

    def evaluate(self,method,axis=None,field=0.):
        key=(method,axis,field)
        if key in self.posts:return self.posts[key]
        source=ref_key(method)
        start=time.time()
        record=dict(method=method,source=source,axis=axis,field_au=field,valid=False,
                    status='error',energy_Eh=None,densities={},post_converged=None,
                    reference_formalism='UHF' if source=='uhf' else 'UKS',
                    xc=xc_for(source,self.cfg) if source!='uhf' else None)
        state=None
        density_builder=None
        try:
            print(f'\n=== {method} axis={axis} field={field:+.8e} ===',flush=True)
            mf,info=self.reference(source,axis,field)
            record['scf']=dict(info)
            if mf is None:raise RuntimeError(info.get('error','SCF convergence or field branch check failed'))
            want_dm=axis is None and self.cfg['dipole'] in ('dm','both')
            if method in ('uhf','pbe','pbe0','b3lyp'):
                record.update(energy_Eh=float(mf.e_tot),post_converged=True)
            elif method in ('ump2','b2plyp','dh_matched'):
                pt=self.pt2(mf)
                scale=1. if method=='ump2' else .27 if method=='b2plyp' else self.cfg['alpha'][1]
                record.update(energy_Eh=float(mf.e_tot+scale*pt.e_corr),post_converged=True,pt2_coefficient=scale)
                if want_dm:
                    def density_builder():
                        dm0=np.asarray(mf.make_rdm1())
                        return dm0+scale*(np.asarray(pt.make_rdm1(ao_repr=True))-dm0)
            elif method in ('obmp2','obdh'):
                from pyscf import df,dft,lib
                from pycmf.OBDH import OBMP2_CL,OBDH_CL
                carrier=mf.copy()
                if axis is not None and self.cfg['field_seed']=='ob':
                    z,zs=self.evaluate(method)
                    if not z['valid']:raise RuntimeError('No valid zero-field OB state')
                    for attr in ('mo_coeff','mo_occ','mo_energy'):
                        setattr(carrier,attr,tuple(np.array(v,copy=True) for v in getattr(zs,attr)))
                ob=(OBMP2_CL if method=='obmp2' else OBDH_CL)(carrier)
                aux=self.cfg['mp2_aux'];aux=df.make_auxbasis(self.mol,mp2fit=True) if aux=='auto' else aux
                ob.with_df=df.DF(self.mol,auxbasis=aux)
                ob.alphaa=tuple(self.cfg['alpha']);ob.niter=self.cfg['ob_cycles'];ob.thresh=self.cfg['ob_tol']
                ob.use_embed=False;ob.use_cl=False;ob.eval_IPEA=False
                ob.shift=self.cfg['shift'];ob.second_order=self.cfg['second_order']
                output=IterationLog(sys.stdout)
                with lib.temporary_env(dft.gen_grid.Grids,level=self.cfg['grid']),contextlib.redirect_stdout(output):
                    energy=ob.kernel()
                record.update(energy_Eh=float(energy),post_converged=bool(ob.converged),iteration=output.last,
                              ob_stability='not_tested',alpha=self.cfg['alpha'] if method=='obdh' else None)
                state=ob
                if axis is not None:
                    _,zs=self.evaluate(method)
                    record['ob_overlap']=occupied_overlap(zs,ob,mf.get_ovlp())
                if want_dm:
                    from pycmf.OBDH.obdh_rdm1 import make_rdm1_unrelaxed
                    density_builder=lambda:make_rdm1_unrelaxed(ob,alpha_c=self.cfg['alpha'][1] if method=='obdh' else 1.)
            else:
                from pyscf import cc
                # CCSD and CCSD(T) share the SAME SCF/CC amplitudes at each field.
                cc_key=('cc_amplitudes',axis,field)
                if cc_key not in self.posts:
                    c=cc.UCCSD(mf,frozen=self.cfg['frozen'])
                    c.conv_tol=self.cfg['cc_tol'];c.conv_tol_normt=self.cfg['cc_grad_tol'];c.max_cycle=self.cfg['cc_cycles']
                    eris=c.ao2mo();c.kernel(eris=eris)
                    self.posts[cc_key]=(c,eris)
                c,eris=self.posts[cc_key]
                energy=float(c.e_tot)
                if method=='ccsd_t' and c.converged:energy+=float(c.ccsd_t(eris=eris))
                record.update(energy_Eh=energy,post_converged=bool(c.converged))
                if want_dm and c.converged:
                    def density_builder():
                        if method=='ccsd':
                            c.solve_lambda(eris=eris);record['lambda_converged']=bool(c.converged_lambda)
                            return c.make_rdm1(ao_repr=True)
                        from pyscf.cc import uccsd_t_lambda,uccsd_t_rdm
                        ok,l1,l2=uccsd_t_lambda.kernel(c,eris,c.t1,c.t2,
                            max_cycle=self.cfg['cc_cycles'],tol=self.cfg['cc_tol'],verbose=4)
                        record['lambda_converged']=bool(ok)
                        return uccsd_t_rdm.make_rdm1(c,c.t1,c.t2,l1,l2,eris=eris,ao_repr=True)
            if state is not None:
                record['orbital_density_valid']=density_diagnostics(self.mol,state._gamma)['density_valid']
            record['valid']=bool(info['valid'] and record['post_converged'] and np.isfinite(record['energy_Eh'])
                                 and record.get('ob_overlap',1.)>=self.cfg['overlap_min'] and record.get('orbital_density_valid',True))
            record['status']='converged' if record['valid'] else 'not_converged'
            if axis is None:
                det=state._gamma if state is not None else mf.make_rdm1()
                analytic=method in ('uhf','pbe','pbe0','b3lyp')
                self.add_density(record,'analytic' if analytic else 'determinant',det)
                if want_dm and not analytic:
                    label='cc_lambda' if method in ('ccsd','ccsd_t') else 'scaled_t2' if method in ('obdh','b2plyp','dh_matched') else 't2'
                    if density_builder is not None:
                        try:
                            dm=density_builder()
                            self.add_density(record,label,dm,record.get('lambda_converged',True))
                        except Exception as exc:
                            # Density construction is independent of converged energy
                            # derivatives. Preserve valid energy/FF if only this fails.
                            record['density_error']=f'{type(exc).__name__}: {exc}'
                            traceback.print_exc()
        except Exception as exc:
            record.update(error=f'{type(exc).__name__}: {exc}');traceback.print_exc()
            record['valid']=False;record['status']='error'
        record['seconds']=time.time()-start
        print(f'=== DONE {method} axis={axis} field={field:+.8e} status={record["status"]} seconds={record["seconds"]:.1f} ===',flush=True)
        self.posts[key]=(record,state)
        if axis is not None:self.points.append(record)
        return record,state

    def release_field(self,axis,field):
        """Keep scalar checkpoints, not all field amplitudes/integral objects."""
        for key in list(self.posts):
            if key[1:]==(axis,field):
                if key[0]=='cc_amplitudes':
                    del self.posts[key]
                else:
                    record,_=self.posts[key]
                    self.posts[key]=(record,None)
        for key in list(self.refs):
            if key[1:]==(axis,field):del self.refs[key]
        import gc
        gc.collect()

    def finite_fields(self,method):
        from pyscf.data import nist
        zero,_=self.evaluate(method)
        results=[]
        for field in self.cfg['fields']:
            axes={};vector=[]
            for axis in 'xyz':
                plus,_=self.evaluate(method,axis,field);minus,_=self.evaluate(method,axis,-field)
                ok=bool(zero['valid'] and plus['valid'] and minus['valid'])
                mu=None;curv=None
                if plus['energy_Eh'] is not None and minus['energy_Eh'] is not None:
                    mu=float((self.nuclear['xyz'.index(axis)]-(plus['energy_Eh']-minus['energy_Eh'])/(2*field))*nist.AU2DEBYE)
                    if zero['energy_Eh'] is not None:curv=plus['energy_Eh']+minus['energy_Eh']-2*zero['energy_Eh']
                axes[axis]=dict(valid=ok,mu_D=mu,curvature_Eh=curv,
                    curvature_warning=curv is not None and abs(curv)>self.cfg['curvature_warn'])
                vector.append(mu)
            finite=all(x is not None and np.isfinite(x) for x in vector)
            results.append(dict(field_au=field,axes=axes,vector_D=vector,
                magnitude_D=float(np.linalg.norm(vector)) if finite else None,
                valid=bool(finite and all(v['valid'] for v in axes.values())),
                step_check='not_tested_single_step' if len(self.cfg['fields'])==1 else 'pending'))
        valid=[r for r in results if r['valid']]
        if len(results)>1:
            spread=max((np.linalg.norm(np.array(a['vector_D'])-b['vector_D']) for a in valid for b in valid),default=None)
            step_ok=len(valid)==len(results) and spread is not None and spread<=self.cfg['ff_tol']
            for r in results:
                r.update(step_check='passed' if step_ok else 'failed' if len(valid)==len(results) else 'not_tested_invalid_field',step_spread_D=spread,
                         valid=bool(r['valid'] and step_ok))
        return results


def run_case(config,index,name,props):
    from pyscf import gto,lib
    lib.num_threads(config['threads'])
    # Some PID-isolated containers mount a host /proc. Only the memory monitor
    # needs a fallback; do not change any SCF/OB equations or thresholds.
    memory_monitor='pyscf_current_memory'
    try:
        lib.current_memory()
    except FileNotFoundError:
        import resource
        def peak_memory():
            value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return (value/(1024**2 if sys.platform=='darwin' else 1024),0.)
        lib.current_memory=lib.misc.current_memory=peak_memory
        memory_monitor='resource_peak_rss'
    start=time.time()
    mol=gto.Mole();mol.stdout=sys.stdout
    basis=config['basis']
    if 'pcv' in basis.lower():basis={'default':basis,'H':basis.lower().replace('pcv','pv')}
    mol.build(atom=props['geometry'],basis=basis,unit='Angstrom',charge=props['charge'],spin=props['spin'],
              verbose=4,output=None,max_memory=config['max_memory'],parse_arg=False,dump_input=False)
    mol.stdout=sys.stdout
    calc=Calculation(mol,config)
    result=dict(schema_version=SCHEMA,index=index,molecule=name,charge=mol.charge,spin=mol.spin,
                nao=mol.nao_nr(),memory_monitor=memory_monitor,zero=[],finite_field={},field_points=[],scf_candidates=[])
    for method in config['methods']:
        record,_=calc.evaluate(method);result['zero'].append(record)
    calc.posts.pop(('cc_amplitudes',None,0.),None)
    if config['dipole'] in ('ff','both'):
        # Work point-by-point. CCSD and CCSD(T) share a single CC solve,
        # then release its tensors before moving to the next field point.
        for field in config['fields']:
            for axis in 'xyz':
                for signed_field in (field,-field):
                    for method in config['methods']:calc.evaluate(method,axis,signed_field)
                    calc.release_field(axis,signed_field)
        for method in config['methods']:
            result['finite_field'][method]=calc.finite_fields(method)
    result.update(field_points=calc.points,scf_candidates=calc.candidates,seconds=time.time()-start)
    from report_dipole import definitions
    result['complete']=all(r['valid'] and all(r['densities'].get(label,{}).get('valid',False)
        for label,field in definitions(r['method'],config['dipole'],[]) if field is None)
        for r in result['zero']) and all(r['valid'] for rs in result['finite_field'].values() for r in rs)
    return result


def worker(meta_path,index,result_path,threads=None,max_memory=None):
    print(f'Worker START pid={os.getpid()} index={index} metadata={meta_path}',flush=True)
    meta=json.loads(Path(meta_path).read_text());item=meta['inputs'][index]
    config=dict(meta['config'])
    if threads is not None:config['threads']=threads
    if max_memory is not None:config['max_memory']=max_memory
    try:
        case=run_case(config,index,item['name'],item['props'])
        case['execution']=dict(threads=config['threads'],max_memory=config['max_memory'])
    except Exception as exc:
        traceback.print_exc();case=dict(index=index,molecule=item['name'],zero=[],finite_field={},field_points=[],
            scf_candidates=[],complete=False,error=f'{type(exc).__name__}: {exc}')
    case['fingerprint']=meta['fingerprint'];write_json(result_path,case)
    return int(not case['complete'])


def parser(default_methods=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'inputs/sp_inputs.json')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=default_methods or list(METHODS))
    p.add_argument('--funcs',nargs='+',choices=['pbe','pbe0','b3lyp','b2plyp','dh_matched'])
    p.add_argument('--levels',nargs='+',choices=['ccsd','ccsd_t'])
    p.add_argument('--basis',default='aug-cc-pCVQZ')
    p.add_argument('--alpha',nargs=2,type=float,default=[.5,.4])
    p.add_argument('--dipole',choices=['det','dm','unrel','ff','both'],default='dm')
    p.add_argument('--fields',nargs='+',type=float,default=[1e-4])
    p.add_argument('--field',type=float,help='Alias for a single --fields value')
    p.add_argument('--ff-tol',type=float,default=1e-3,help='Max signed-vector spread across field steps, Debye')
    p.add_argument('--curvature-warn',type=float,default=1e-5)
    p.add_argument('--overlap-min',type=float,default=.95)
    p.add_argument('--field-seed',choices=['hf','ob'],default='ob')
    p.add_argument('--scf-policy',choices=['standard','multiguess'],default='multiguess')
    p.add_argument('--init-guess',default='minao')
    p.add_argument('--guesses',nargs='+',default=['minao','atom','1e'])
    p.add_argument('--stability-cycles',type=int,default=6)
    p.add_argument('--scf-tol',type=float,default=1e-10)
    p.add_argument('--scf-grad-tol',type=float,default=1e-6)
    p.add_argument('--scf-cycles',type=int,default=200)
    p.add_argument('--ob-tol',type=float,default=1e-8)
    p.add_argument('--ob-cycles',type=int,default=300)
    p.add_argument('--cc-tol',type=float,default=1e-9)
    p.add_argument('--cc-grad-tol',type=float,default=1e-7)
    p.add_argument('--cc-cycles',type=int,default=100)
    p.add_argument('--grid',type=int,default=4)
    p.add_argument('--scf-aux',default='none',help='none: exact SCF; auto or named JK fitting basis')
    p.add_argument('--mp2-aux',default='auto',help='auto: make_auxbasis(mp2fit=True), independent of SCF fitting')
    p.add_argument('--no-pt2-df',dest='pt2_df',action='store_false',default=True)
    p.add_argument('--frozen',type=int,default=0)
    p.add_argument('--shift',type=float,default=0.)
    p.add_argument('--first-order',dest='second_order',action='store_false',default=True)
    p.add_argument('--threads',type=int,default=int(os.environ.get('OMP_NUM_THREADS','1')))
    p.add_argument('--max-memory',type=int,default=int(os.environ.get('PYSCF_MAX_MEMORY','4000')))
    p.add_argument('--only',nargs='+')
    p.add_argument('--index',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--merge',action='store_true')
    p.add_argument('--no-excel',action='store_true')
    p.add_argument('--ref-json',type=Path)
    p.add_argument('--output-xlsx',type=Path)
    p.add_argument('--worker',nargs=3,metavar=('META','INDEX','RESULT'),help=argparse.SUPPRESS)
    return p


def main(default_methods=None):
    p=parser(default_methods);a=p.parse_args()
    print(f'Dipole START pid={os.getpid()} outdir={a.outdir.resolve()}',flush=True)
    if a.worker:return worker(a.worker[0],int(a.worker[1]),a.worker[2],a.threads,a.max_memory)
    from report_dipole import collect
    if a.merge:
        collect(a.outdir,a.ref_json,a.output_xlsx,excel=not a.no_excel);return 0
    if a.funcs:a.methods=a.funcs
    if a.levels:a.methods=a.levels
    if a.dipole=='unrel':a.dipole='dm'
    if a.field is not None:a.fields=[a.field]
    a.fields=sorted(set(a.fields),reverse=True)
    if any(not math.isfinite(v) or not 0<=v<=1 for v in a.alpha):p.error('alpha must be in [0,1]')
    if not a.fields or any(not math.isfinite(f) or f<=0 for f in a.fields):p.error('fields must be finite and positive')
    if any(x<=0 for x in (a.scf_cycles,a.ob_cycles,a.cc_cycles,a.threads,a.max_memory)):p.error('cycles/resources must be positive')
    if any(not math.isfinite(x) or x<=0 for x in (a.scf_tol,a.scf_grad_tol,a.ob_tol,a.cc_tol,a.cc_grad_tol,a.ff_tol)):p.error('tolerances must be finite and positive')
    if not 0<=a.overlap_min<=1:p.error('overlap-min must be in [0,1]')
    if a.grid<0 or a.grid>9 or a.stability_cycles<0:p.error('grid must be 0..9; stability-cycles must be nonnegative')
    if a.frozen<0 or (a.frozen and set(a.methods)&{'obmp2','obdh'}):p.error('OBMP2/OBDH currently require frozen=0')
    data=json.loads(a.input.read_text());inputs=[dict(name=n,props=props) for n,props in data.items()]
    index=a.index
    if index is None and not a.only and 'SLURM_ARRAY_TASK_ID' in os.environ:index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    if index is not None and a.only:p.error('choose --index or --only')
    if index is not None and not 0<=index<len(inputs):p.error('index outside input')
    indices=[index] if index is not None else [i for i,x in enumerate(inputs) if not a.only or x['name'] in a.only]
    if a.only and set(a.only)-data.keys():p.error('unknown molecules: '+str(sorted(set(a.only)-data.keys())))
    selected_methods=list(dict.fromkeys(a.methods));a.methods=selected_methods
    execution={'input','outdir','only','index','resume','merge','no_excel','ref_json','output_xlsx','worker','funcs','levels','field'}
    config={k:v for k,v in vars(a).items() if k not in execution}
    config['formalism']='unrestricted UHF/UKS; no restricted-DH reproduction implied'
    config['origin_bohr']=[0.,0.,0.]
    config['named_b2plyp_alpha']=[.53,.27]
    import pyscf
    hashes={Path(__file__).name:file_hash(__file__)}
    if set(a.methods)&{'obdh','obmp2'}:
        from pycmf.OBDH import main as core,uobdh_solver,obdh_rdm1
        hashes.update({Path(m.__file__).name:file_hash(m.__file__) for m in [core,uobdh_solver,obdh_rdm1]})
    physical={k:v for k,v in config.items() if k not in ('threads','max_memory')}
    fingerprint=digest(dict(schema=SCHEMA,config=physical,input_sha256=file_hash(a.input),code=hashes,pyscf=pyscf.__version__))
    a.outdir.mkdir(parents=True,exist_ok=True);(a.outdir/'points').mkdir(exist_ok=True);(a.outdir/'logs').mkdir(exist_ok=True)
    meta_path=a.outdir/'dipole.meta.json'
    def update_meta(status=None):
        with metadata_lock(a.outdir/'.metadata.lock'):
            if meta_path.exists():
                meta=json.loads(meta_path.read_text())
                if meta['fingerprint']!=fingerprint:raise RuntimeError('Output contains different physics/input/code; choose a new --outdir')
            else:
                meta=dict(schema_version=SCHEMA,fingerprint=fingerprint,config=config,inputs=inputs,
                    provenance=dict(code_sha256=hashes,pyscf=pyscf.__version__,input_sha256=file_hash(a.input),input_path=str(a.input.resolve())),
                    requested_indices=[],cases={},invocations=[],ref_json=str(a.ref_json.resolve()) if a.ref_json else None)
            meta['requested_indices']=sorted(set(meta['requested_indices'])|set(indices))
            if status is None:
                meta.setdefault('invocations',[]).append(dict(time=time.time(),threads=a.threads,max_memory=a.max_memory,job_id=os.environ.get('SLURM_JOB_ID'),indices=indices))
            if status is not None:meta['cases'].update(status)
            print(f'Metadata: writing {meta_path}',flush=True)
            write_json(meta_path,meta)
            write_json(a.outdir/'config.json',meta['config'])
            print('Metadata: saved',flush=True)
    update_meta()
    failures=0
    for i in indices:
        item=inputs[i];name=item['name'];token=stem(i,name);result_path=a.outdir/'points'/f'{token}.json'
        with directory_lock(a.outdir/'points'/f'.{token}.lock.d',timeout=0.,label=f'Case {name}'):
            if result_path.exists():
                cached=json.loads(result_path.read_text())
                if cached.get('fingerprint')!=fingerprint:raise RuntimeError('Checkpoint fingerprint differs')
                if not a.resume:p.error(f'{result_path} exists; use --resume or a new output directory')
                if cached.get('complete'):
                    update_meta({str(i):dict(molecule=name,complete=True,path=str(result_path.relative_to(a.outdir)),exit_code=0)})
                    print(f'SKIP {name}: complete',flush=True);continue
            print(f'Running {i}: {name}',flush=True)
            print(f'Live calculation log: {(a.outdir/"logs"/f"{token}.log").resolve()}',flush=True)
            result_path.unlink(missing_ok=True)
            with (a.outdir/'logs'/f'{token}.log').open('a') as log:
                log.write(f'\n=== ATTEMPT {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n');log.flush()
                code=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--outdir',str(a.outdir),
                    '--worker',str(meta_path),str(i),str(result_path),'--threads',str(a.threads),'--max-memory',str(a.max_memory)],stdout=log,stderr=subprocess.STDOUT).returncode
            case=json.loads(result_path.read_text()) if result_path.exists() else dict(index=i,molecule=name,
                complete=False,error=f'Worker exited {code} without checkpoint',fingerprint=fingerprint,zero=[],finite_field={},field_points=[],scf_candidates=[])
            if not result_path.exists():write_json(result_path,case)
            failures+=not case['complete']
            update_meta({str(i):dict(molecule=name,complete=case['complete'],path=str(result_path.relative_to(a.outdir)),exit_code=code)})
            print(f'{name}: '+('complete' if case['complete'] else 'failed or incomplete')+f' -> {result_path}',flush=True)
    # Array tasks only write their own checkpoint. One explicit merge avoids races.
    if index is None:collect(a.outdir,a.ref_json,a.output_xlsx,excel=not a.no_excel)
    return int(bool(failures))


if __name__=='__main__':sys.exit(main())
