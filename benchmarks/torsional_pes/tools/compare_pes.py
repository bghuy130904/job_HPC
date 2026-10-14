#!/usr/bin/env python3
"""Compare computed PES to legacy values at matching angles; no interpolation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--references',type=Path,default=ROOT/'references')
    p.add_argument('--output-dir',type=Path,required=True)
    a = p.parse_args()
    files = [f for d in a.runs for f in sorted(d.glob('*_pes_ns*.json'))]
    if not files:
        p.error('No computed PES JSONs found.')
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    rows=[]
    metrics=[]
    for f in files:
        data=json.loads(f.read_text())
        ref_path=a.references/f.name
        if not ref_path.exists():
            continue
        ref=json.loads(ref_path.read_text())
        old={float(pt['phi']):pt['e_rel_kcal'] for pt in ref['scan']}
        anchor = old[180.0]
        old = {phi: energy-anchor for phi,energy in old.items()}
        scan=[dict(pt) for pt in data['scan'] if pt.get('e_rel_kcal') is not None]
        anchor_new = next((pt['e_rel_kcal'] for pt in scan if pt['phi']==180),None)
        if anchor_new is None:
            continue
        for pt in scan:
            pt['e_rel_kcal'] -= anchor_new
        if not scan:
            continue
        method=data['method']
        mol=data['config']['molecule']
        ns=data['n_shells']
        ax=axes[int(method=='mp2'),int(mol=='1-butanol')]
        color=f'C{ns-1}'
        ax.plot(list(old),list(old.values()),color=color,linestyle='--',alpha=.65,
                label=f'Legacy shell {ns-1}')
        ax.plot([pt['phi'] for pt in scan],[pt['e_rel_kcal'] for pt in scan],
                color=color,marker='o',label=f'New shell {ns-1}')
        diffs=[]
        energies=[]
        refs=[]
        for pt in scan:
            if float(pt['phi']) not in old:
                continue
            delta=pt['e_rel_kcal']-old[float(pt['phi'])]
            diffs.append(delta)
            energies.append(pt['e_rel_kcal'])
            refs.append(old[float(pt['phi'])])
            rows.append(dict(molecule=mol,method=method,n_shells=ns,phi=pt['phi'],
                             legacy_kcal=old[float(pt['phi'])],new_kcal=pt['e_rel_kcal'],delta_kcal=delta))
        if diffs:
            corr=float(np.corrcoef(energies,refs)[0,1]) if len(diffs)>2 and np.std(energies)>0 else None
            metrics.append(dict(molecule=mol,method=method,n_shells=ns,n_points=len(diffs),
                                rms_delta_kcal=float(np.sqrt(np.mean(np.square(diffs)))),
                                max_abs_delta_kcal=float(np.max(np.abs(diffs))),pearson=corr,
                                absolute_180_delta_hartree=(data['e_ref_hartree']+anchor_new/627.5094740631)
                                    -(ref['e_ref_hartree']+anchor/627.5094740631)))
    for row,method in enumerate(['OBDH','MP2-in-DFT']):
        for col,mol in enumerate(['Ethanol','1-Butanol']):
            ax=axes[row,col]
            ax.set_title(f'{mol}: {method}')
            ax.set_ylabel('E(phi) - E(180) / kcal mol$^{-1}$')
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=8)
            else:
                ax.text(.5,.5,'No accepted calculation',ha='center',transform=ax.transAxes)
            ax.set_xlabel('C2-C1-O-H dihedral / degrees')
    fig.tight_layout()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output_dir/'pes_comparison.png',dpi=160)
    with (a.output_dir/'comparison.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ['molecule'])
        writer.writeheader()
        writer.writerows(rows)
    (a.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    main()
