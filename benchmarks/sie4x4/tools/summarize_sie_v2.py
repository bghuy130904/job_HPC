#!/usr/bin/env python3
"""Aggregate independent scaling_v2 baselines without rerunning quantum chemistry."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAYOUT='independent_baseline_and_diagnostics_v1'
PANELS=[('dft_base','dft','DFT reference'),('obdh','uhf','Current UHF -> OBDH'),
        ('obdh','dft','Current DFT -> OBDH'),('obdh_1st','dft','Current DFT + one OBMP2 evaluation')]


def read_scan(root):
    tables={name:[] for name in ('statistics','selected','errors','paired_relaxation',
        'diagnostic_best','branch_map','frozen_derivatives','C1_C4')}
    configs=[]
    seen=set();signature=None
    for directory in sorted(root.glob('recalc_obdh_*')):
        config=directory/'config.json'
        if not config.is_file():
            continue
        cfg=json.loads(config.read_text())
        if cfg.get('pipeline_layout')!=LAYOUT:
            raise ValueError(f'Old/mixed pipeline at {directory}; choose the fixed_pipeline scan root')
        alpha=tuple(cfg['alpha'])
        if alpha in seen:
            raise ValueError(f'Duplicate Alpha {alpha} at {directory}; do not combine repeated runs')
        seen.add(alpha)
        fields=('basis','grid','scf_aux','ob_aux','methods','ob_init','systems','points','scf_policy',
                'scf_tol','scf_grad_tol','scf_cycles','stability_cycles','ob_tol','ob_cycles',
                'density_tol','input_sha256','driver_sha256','pycmf_core_sha256','pyscf_version')
        actual={k:cfg.get(k) for k in fields}
        if signature is None:signature=actual
        elif actual!=signature:
            raise ValueError(f'Incompatible basis/method/input/code/settings at {directory}; use a separate SCAN_ROOT')
        configs.append(dict(alpha=str(tuple(cfg['alpha'])),directory=str(directory),
                            neighbor_selection=cfg.get('neighbor_selection')))
        for name in tables:
            path=directory/f'{name}.csv'
            if not path.exists() or path.stat().st_size<2:
                continue
            frame=pd.read_csv(path)
            if 'alpha' in frame:
                frame=frame.drop(columns='alpha')
            frame.insert(0,'alpha',str(tuple(cfg['alpha'])))
            frame.insert(1,'alpha_x',cfg['alpha'][0])
            frame.insert(2,'alpha_c',cfg['alpha'][1])
            tables[name].append(frame)
    return {k:pd.concat(v,ignore_index=True) if v else pd.DataFrame() for k,v in tables.items()},configs


def paired_errors(paired, selected):
    """Use ONLY the full OBDH paired to the selected one-shot reference at each R."""
    if paired.empty or selected.empty:
        return pd.DataFrame()
    first=selected[(selected.method=='obdh_1st') & (selected.valid==True)]
    pairs=paired[paired.paired_valid==True].merge(
        first[['alpha','system','point','guess']],on=['alpha','system','point','guess'],how='inner')
    # No best-over-corrected-energies selection is made here.
    lookup={(r.alpha,r.system,r.point):r for r in pairs.itertuples()}
    reference={'H2_plus_He':[64.4,58.9,48.7,38.3],'He2_plus':[56.9,46.9,31.3,19.1],
               'NH3_2_plus':[35.9,25.9,13.4,4.9],'H2O_2_plus':[39.7,29.1,16.9,9.3]}
    rows=[]
    for (alpha,system,point),r in lookup.items():
        if point=='dissociation_limit':
            continue
        dl=lookup.get((alpha,system,'dissociation_limit'))
        if dl is None:
            continue
        index=['R_1.0','R_1.25','R_1.5','R_1.75'].index(point)
        ef=(dl.E_full_Ha-r.E_full_Ha)*627.5094740631-reference[system][index]
        e1=(dl.E_first_Ha-r.E_first_Ha)*627.5094740631-reference[system][index]
        rows.append(dict(alpha=alpha,system=system,point=point,first_error_kcal=e1,
            full_same_start_error_kcal=ef,guess_R=r.guess,guess_DL=dl.guess,
            type_changed_R=r.label_changed,type_changed_DL=dl.label_changed,
            same_final_branch_certified=False))
    return pd.DataFrame(rows)


def slope_comparison(errors, derivatives, anchor=(.5,.4),step=.1):
    """Compare fixed-state components with observed slopes; do not assert equality."""
    if errors.empty or derivatives.empty:
        return pd.DataFrame()
    lookup={(round(r.alpha_x,10),round(r.alpha_c,10),r.system,r.point,r.method,r.init_source):r
            for r in errors.itertuples() if r.valid}
    rows=[]
    for r in derivatives.itertuples():
        if r.method not in ('obdh','obdh_1st') or not r.valid or not np.allclose([r.alpha_x,r.alpha_c],anchor,atol=1e-10,rtol=0):
            continue
        for axis,component in [(0,'frozen_Dx_reaction_kcal'),(1,'frozen_Dc_reaction_kcal')]:
            left=list(anchor);right=list(anchor);left[axis]-=step;right[axis]+=step
            key=(r.system,r.point,r.method,r.init_source)
            a=lookup.get((round(left[0],10),round(left[1],10),*key))
            b=lookup.get((round(right[0],10),round(right[1],10),*key))
            c=lookup.get((*anchor,*key))
            valid=a is not None and b is not None and c is not None
            observed=(b.error_kcal-a.error_kcal)/(2*step) if valid else None
            fixed=getattr(r,component,None)
            changed=valid and len({(t.label_R,t.label_DL) for t in (a,b,c)})>1
            rows.append(dict(system=r.system,point=r.point,method=r.method,init_source=r.init_source,
                alpha_anchor=str(anchor),axis='alpha_x' if axis==0 else 'alpha_c',
                frozen_component_kcal_per_alpha=fixed,observed_central_slope_kcal_per_alpha=observed,
                difference_kcal_per_alpha=observed-fixed if observed is not None and pd.notna(fixed) else None,
                labels_change_in_stencil=changed if valid else None,
                same_branch_certified=False,
                interpretation='type_change_do_not_interpret_as_same_branch_response' if changed else
                    'same_labels_only_orbital_response_and_branch_identity_not_certified' if valid else 'missing_baseline_points'))
    return pd.DataFrame(rows)


def plots(tables,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    values=np.arange(1,10)/10
    stats=tables['statistics']
    if not stats.empty:
        fig,axes=plt.subplots(2,2,figsize=(13,12),layout='constrained')
        cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#dddddd')
        maximum=max(1.,float(stats.MUE_16.max()))
        for ax,(method,source,title) in zip(axes.flat,PANELS):
            data=stats[(stats.method==method)&(stats.init_source==source)]
            z=np.full((9,9),np.nan);n=np.zeros((9,9),int)
            for r in data.itertuples():
                i=int(round(r.alpha_x*10))-1;j=int(round(r.alpha_c*10))-1
                if not (0<=i<9 and 0<=j<9):continue
                n[i,j]=r.n_valid
                if r.n_valid==16 and pd.notna(r.MUE_16):z[i,j]=r.MUE_16
            im=ax.imshow(np.ma.masked_invalid(z),origin='lower',vmin=0,vmax=maximum,cmap=cmap)
            for i in range(9):
                for j in range(9):
                    ax.text(j,i,f'{z[i,j]:.2f}' if np.isfinite(z[i,j]) else f'{n[i,j]}/16',
                        ha='center',va='center',fontsize=8,color='white' if z[i,j]>.65*maximum else 'black')
            ax.set_xticks(range(9),[f'{v:.1f}' for v in values]);ax.set_yticks(range(9),[f'{v:.1f}' for v in values])
            ax.set_xlabel('alpha_c');ax.set_ylabel('alpha_x');ax.set_title(title)
            if np.isfinite(z).any():
                i,j=np.unravel_index(np.nanargmin(z),z.shape)
                ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,color='black',lw=2))
        fig.colorbar(im,ax=axes,label='Baseline MUE over 16 errors (kcal/mol)',shrink=.75)
        fig.suptitle('Independent baselines ONLY. Gray cells: incomplete, never zero.')
        fig.savefig(out/'baseline_MUE_heatmap.png',dpi=220);plt.close(fig)
    errors=tables['errors']
    if errors.empty:return
    systems=['H2_plus_He','He2_plus','NH3_2_plus','H2O_2_plus']
    points=['R_1.0','R_1.25','R_1.5','R_1.75']
    for axvalue in [.7,.9]:
        fig,axes=plt.subplots(4,4,figsize=(16,12),layout='constrained',sharex=True)
        for i,system in enumerate(systems):
            for j,point in enumerate(points):
                ax=axes[i,j];ax.axhline(0,color='black',lw=.7)
                for (method,source,title),color in zip(PANELS[1:],['#1f77b4','#d62728','#228833']):
                    data=errors[(np.isclose(errors.alpha_x,axvalue))&(errors.system==system)&
                        (errors.point==point)&(errors.method==method)&(errors.init_source==source)].sort_values('alpha_c')
                    previous=None
                    for r in data.itertuples():
                        if not r.valid or pd.isna(r.error_kcal):previous=None;continue
                        ax.plot(r.alpha_c,r.error_kcal,'o',color=color,ms=3)
                        if previous is not None and abs(r.alpha_c-previous.alpha_c-.1)<1e-8 and (r.label_R,r.label_DL)==(previous.label_R,previous.label_DL):
                            ax.plot([previous.alpha_c,r.alpha_c],[previous.error_kcal,r.error_kcal],color=color,lw=1)
                        previous=r
                    ax.plot([],[],color=color,marker='o',label=title)
                ax.set_title(f'{system}: {point}',fontsize=10);ax.set_xticks(values[::2]);ax.grid(alpha=.2)
                if i==3:ax.set_xlabel('alpha_c')
                if j==0:ax.set_ylabel('Calculated - reference (kcal/mol)')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'Signed BASELINE errors: alpha_x={axvalue}. Lines break at missing points or type-label changes.')
        fig.savefig(out/f'signed_errors_ax_{axvalue:g}.png',dpi=200);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--no-plots',action='store_true')
    cfg=p.parse_args();tables,configs=read_scan(cfg.root)
    if not configs:raise SystemExit('No fixed-pipeline runs found; nothing has been replaced or invented.')
    out=cfg.root/'scan_summary';out.mkdir(parents=True,exist_ok=True)
    tables['paired_errors']=paired_errors(tables['paired_relaxation'],tables['selected'])
    paired_stats=[]
    for run in configs:
        frame=tables['paired_errors']
        sample=frame[frame.alpha==run['alpha']] if not frame.empty else frame
        complete=len(sample)==16
        paired_stats.append(dict(alpha=run['alpha'],n_valid=len(sample),
            MUE_first_16=float(sample.first_error_kcal.abs().mean()) if complete else None,
            MUE_full_same_start_16=float(sample.full_same_start_error_kcal.abs().mean()) if complete else None,
            type_changes_at_R=int(sample.type_changed_R.sum()) if len(sample) else 0,
            same_final_branch_certified=False,
            status='complete_16_same_initial_DFT_references' if complete else 'incomplete_no_partial_MUE'))
    tables['paired_statistics']=pd.DataFrame(paired_stats)
    tables['slope_comparison']=slope_comparison(tables['errors'],tables['frozen_derivatives'])
    tables['runs']=pd.DataFrame(configs)
    definitions=[('selected','Independent current-Alpha SCF -> method only; diagnostic starts never replace it.'),
        ('diagnostic_best','Lowest found including neighbor/restart. Source is ancestry, not fresh SCF.'),
        ('paired_errors','One-shot and full OBDH start from the identical selected DFT reference at each geometry.'),
        ('label_changed','An observed classification change; same label does not certify same branch.'),
        ('slope_comparison','Fixed-state Dx/Dc versus observed central slopes at (0.5,0.4). Not an exact response identity.'),
        ('C1_C4','Execution coverage is not full PDF compliance. Formal OB stability remains unimplemented.'),
        ('missing','Missing energies/errors/MUE remain empty. MUE_16 requires all 16 errors.'),
        ('reference','All dissociation energies use E_supermolecule(R_DL)-E_supermolecule(R).')]
    tables['readme']=pd.DataFrame(definitions,columns=['item','meaning'])
    from openpyxl.styles import Border,Side,Font
    border=Border(*(Side(style='thin',color='000000') for _ in range(4)))
    target=out/'SIE_scaling_summary.xlsx';temporary=out/'SIE_scaling_summary.tmp.xlsx'
    with pd.ExcelWriter(temporary,engine='openpyxl') as writer:
        for name,frame in tables.items():
            frame.to_csv(out/f'{name}.csv',index=False)
            frame.to_excel(writer,sheet_name=name,index=False)
            ws=writer.sheets[name]
            for row in ws:
                for cell in row:cell.border=border;cell.font=Font(name='Calibri',size=11,bold=cell.row==1)
            for column in ws.columns:
                ws.column_dimensions[column[0].column_letter].width=min(45,max(15,len(str(column[0].value))+2))
    temporary.replace(target)
    if not cfg.no_plots:plots(tables,out)
    print(f'SCAN SUMMARY: {len(configs)}/81 Alpha folders found. Missing runs are not zero. Saved {target}',flush=True)
    print('This is a computational report, not a certificate of PDF C1-C4 compliance or OB stability.',flush=True)


if __name__=='__main__':main()
