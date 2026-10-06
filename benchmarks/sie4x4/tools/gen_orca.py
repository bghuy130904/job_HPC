#!/usr/bin/env python3
"""Generate fresh ORCA inputs; default plus both charge-localized guesses for ALL methods."""
import argparse
import json
from pathlib import Path
from run_sie import HERE, POINTS, DL, fragments

FUNCTIONALS = ['PBE','B2PLYP','B2GP-PLYP','DSD-PBEP86','PWPB95']
STABILITY = '''%scf
 STABPerform true
 STABRestartUHFifUnstable true
 MaxIter 500
end
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=HERE/'inputs/input.json')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--basis',default='aug-cc-pVDZ')
    p.add_argument('--nproc',type=int,default=16)
    p.add_argument('--maxcore',type=int,default=3500)
    cfg=p.parse_args()
    cfg.outdir.mkdir(parents=True,exist_ok=True)
    data=json.loads(cfg.input.read_text())
    recipes=[]
    def write(tag,func,atoms,indices,charge,spin,moread=False,stable=True):
        # AutoAux avoids nonexistent manually guessed /J names for Dunning basis sets.
        keywords=f'! UKS {func} {cfg.basis} AutoAux RIJCOSX TightSCF DEFGRID3 NoFrozenCore'
        if moread: keywords+=' MOREAD'
        text=f'{keywords}\n%pal nprocs {cfg.nproc} end\n%maxcore {cfg.maxcore}\n'
        if moread: text+='%moinp "merged.gbw"\n'
        if stable: text+=STABILITY
        text+=f'* xyz {charge} {spin+1}\n'
        text+='\n'.join(a['element']+' '+' '.join(f'{x:.8f}' for x in a['coordinates']) for i in indices for a in [atoms[i]])
        text+='\n*\n'
        (cfg.outdir/f'{tag}.inp').write_text(text)
    for func in FUNCTIONALS:
        for system,points in data.items():
            for point in POINTS+[DL]:
                atoms=points[point]
                groups=fragments(system,atoms)
                base=f'{func.replace("-","")}_{system}_{point}'
                write(base+'_def',func,atoms,list(range(len(atoms))),1,1)
                recipe=dict(tag=base,func=func,system=system,point=point,variants=['def'],atom_groups={'def':groups})
                for side,suffix in [(0,'loc'),(1,'locB')]:
                    if system=='H2_plus_He':
                        # neutral H doublet; H+ with spectator He singlet, keep He 6 A away.
                        neutral=1-side
                        ga,gb=[neutral],sorted([side,2])
                        qa,qb,sa,sb=0,1,1,0
                    else:
                        ga,gb=groups[side],groups[1-side]
                        qa,qb,sa,sb=1,0,1,0
                    write(base+'_'+suffix+'_fragA',func,atoms,ga,qa,sa,stable=False)
                    write(base+'_'+suffix+'_fragB',func,atoms,gb,qb,sb,stable=False)
                    write(base+'_'+suffix,func,atoms,ga+gb,1,1,moread=True)
                    recipe['variants'].append(suffix)
                    order=ga+gb
                    recipe['atom_groups'][suffix]=[[order.index(i) for i in g] for g in groups]
                recipes.append(recipe)
    (cfg.outdir/'recipes.json').write_text(json.dumps(recipes,indent=2))
    (cfg.outdir/'config.json').write_text(json.dumps({k:str(v) for k,v in vars(cfg).items()},indent=2))
    print(f'Generated {len(recipes)} cases / {len(recipes)*7} inputs in {cfg.outdir}')

if __name__=='__main__': main()
