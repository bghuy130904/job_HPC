#!/usr/bin/env python3
"""Strict ORCA collector: raw candidates retained; no S2 gate; unknown stability invalid.
For DH, select the lowest stable SCF-reference energy, THEN report its DH energy.
"""
import argparse
import json
import re
from pathlib import Path
import pandas as pd
from run_sie import REF, POINTS, DL, KCAL

NUMBER=r'([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)'


def parse(path):
    if not path.exists(): return dict(valid=False,reason='missing_output')
    text=path.read_text(errors='replace')
    final=re.findall(r'FINAL SINGLE POINT ENERGY\s+'+NUMBER,text)
    refs=re.findall(r'^\s*Total Energy\s*:\s*'+NUMBER+r'\s*Eh',text,re.M)
    s2=re.findall(r'Expectation value of <S\*\*2>\s*:\s*'+NUMBER,text)
    # Last verdict belongs to the final followed wavefunction, not the first unstable one.
    verdicts=re.findall(r'Stability Analysis indicates an?\s+(UNSTABLE|STABLE)\s+(?:HF/KS\s+)?wave\s*function',text,re.I)
    normal='ORCA TERMINATED NORMALLY' in text
    events=re.findall(r'SCF CONVERGED AFTER|SCF NOT CONVERGED',text)
    conv=bool(events and events[-1]=='SCF CONVERGED AFTER')
    stable=verdicts[-1].upper()=='STABLE' if verdicts else None
    atom_data={}
    sections=re.findall(r'MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS(.*?)(?:Sum of atomic|$)',text,re.S|re.I)
    if sections:
        for i,q,spin in re.findall(r'^\s*(\d+)\s+[A-Za-z]+\s*:\s*'+NUMBER+r'\s+'+NUMBER,sections[-1],re.M):
            atom_data[int(i)]=dict(charge=float(q),spin=float(spin))
    return dict(atom_populations=atom_data,energy=float(final[-1]) if final else None,
                reference_energy=float(refs[-1]) if refs else None,
                s2=float(s2[-1]) if s2 else None,converged=conv,normal_termination=normal,
                stable=stable,valid=bool(final and conv and normal and stable is True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--recipes',type=Path,required=True)
    cfg=p.parse_args()
    recipes=json.loads(cfg.recipes.read_text())
    raw,selected=[],[]
    for recipe in recipes:
        rows=[]
        for variant in recipe['variants']:
            r=dict(func=recipe['func'],system=recipe['system'],point=recipe['point'],guess=variant,
                   **parse(cfg.outdir/f'{recipe["tag"]}_{variant}.out'))
            groups=recipe.get('atom_groups',{}).get(variant)
            pops=r.get('atom_populations',{})
            if groups and all(i in pops for group in groups for i in group):
                r['fragment_spin']=[sum(pops[i]['spin'] for i in g) for g in groups]
                r['fragment_charge']=[sum(pops[i]['charge'] for i in g) for g in groups]
                sa,sb=r['fragment_spin'][:2]
                frac=abs(sa-sb)/max(abs(sa)+abs(sb),1e-12)
                r['label']='loc' if frac>=.8 else ('deloc' if frac<=.2 else 'mixed')
            raw.append(r);rows.append(r)
        dh=recipe['func']!='PBE'
        good=[r for r in rows if r['valid'] and (not dh or r['reference_energy'] is not None)]
        chosen=min(good,key=lambda r:r['reference_energy'] if dh else r['energy']) if good else None
        selected.append(dict(func=recipe['func'],system=recipe['system'],point=recipe['point'],
                             energy=chosen['energy'] if chosen else None,guess=chosen['guess'] if chosen else None,
                             valid=chosen is not None))
    lookup={(r['func'],r['system'],r['point']):r for r in selected}
    errors=[];stats=[]
    for func in sorted({r['func'] for r in selected}):
        for system in REF:
            for i,point in enumerate(POINTS):
                a=lookup.get((func,system,point)); b=lookup.get((func,system,DL))
                de=(b['energy']-a['energy'])*KCAL if a and b and a['valid'] and b['valid'] else None
                errors.append(dict(func=func,system=system,point=point,De_kcal=de,
                                   error_kcal=de-REF[system][i] if de is not None else None))
        sample=[r['error_kcal'] for r in errors if r['func']==func and r['error_kcal'] is not None]
        stats.append(dict(func=func,n_valid=len(sample),MUE_16=sum(map(abs,sample))/16 if len(sample)==16 else None))
    with pd.ExcelWriter(cfg.outdir/'SIE4x4_ORCA.xlsx') as w:
        for name,rows in [('candidates',raw),('selected',selected),('errors',errors),('statistics',stats)]:
            pd.DataFrame(rows).to_excel(w,sheet_name=name,index=False)
            pd.DataFrame(rows).to_csv(cfg.outdir/f'{name}.csv',index=False)
    missing=sum(not r['valid'] for r in selected)
    print(f'Missing/unvalidated cases: {missing}; stability unknown is NOT accepted')
    return int(bool(missing))

if __name__=='__main__':
    raise SystemExit(main())
