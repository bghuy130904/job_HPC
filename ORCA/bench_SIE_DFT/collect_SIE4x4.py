#!/usr/bin/env python3
"""
Trich xuat nang luong tong (Hartree) tinh bang ORCA cho bo SIE4x4
-> 1 bang Excel duy nhat.

Cot PBE la KIEM CHUNG PIPELINE (khong phai DH): He2+ phai cho sai so
-68.3 / -58.9 / -49.5 / -41.2 kcal/mol nhu Table 1. Neu PBE lech thi
setup ORCA sai, va cac cot DH cung khong dang tin.

Voi moi diem, script doc CA HAI basin (_def = delocalized, _loc = fragment
guess -> localized), loc nghiem hop le (SCF converged VA |<S^2>-0.75| <= 0.05),
roi lay nghiem HOP LE THAP NHAT (nguyen ly bien phan, theo SI Bao-Truhlar).
Chi gia tri duoc chon moi ghi vao bang.

Chay:  OUT_DIR=<thu muc chua .out>  python3 collect_SIE4x4.py
"""
import os
import re

import pandas as pd

OUT_DIR = os.environ.get('OUT_DIR', '.')
XLSX = os.path.join(OUT_DIR, 'SIE4x4_DH_total_energy.xlsx')
S2_TOL = 0.05

# PBE = cot kiem chung pipeline (phai khop Table 1 cua bai bao), khong phai DH.
FUNCS = ['PBE', 'B2PLYP', 'B2GPPLYP', 'DSDPBEP86', 'PWPB95']
LABEL = {'PBE': 'PBE', 'B2PLYP': 'B2PLYP', 'B2GPPLYP': 'B2GP-PLYP',
         'DSDPBEP86': 'DSD-PBEP86', 'PWPB95': 'PWPB95'}

SYSTEMS = ['H2_plus_He', 'He2_plus', 'NH3_2_plus', 'H2O_2_plus']
SYSLBL = {'H2_plus_He': 'H2+...He', 'He2_plus': 'He2+',
          'NH3_2_plus': '(NH3)2+', 'H2O_2_plus': '(H2O)2+'}
PTS = ['R_1.0', 'R_1.25', 'R_1.5', 'R_1.75', 'dissociation_limit']
PTLBL = {'R_1.0': '1', 'R_1.25': '1.25', 'R_1.5': '1.5', 'R_1.75': '1.75',
         'dissociation_limit': 'dissociation limit'}


def parse(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors='ignore').read()
    m = re.findall(r'FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)', txt)
    if not m:
        return None
    s2 = re.findall(r'Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)', txt)
    conv = ('SUCCESS' in txt) and ('SCF NOT CONVERGED' not in txt)
    return dict(E=float(m[-1]),
                s2=float(s2[-1]) if s2 else None,
                conv=conv)


def valid(r):
    if r is None or not r['conv']:
        return False
    return r['s2'] is None or abs(r['s2'] - 0.75) <= S2_TOL


def pick(func, s, pt):
    """Nghiem hop le THAP NHAT giua _def (deloc) va _loc (localized)."""
    cands = [r for r in (parse(f'{OUT_DIR}/{func}_{s}_{pt}_{k}.out')
                         for k in ('def', 'loc')) if valid(r)]
    return min(cands, key=lambda r: r['E'])['E'] if cands else None


rows = []
for s in SYSTEMS:
    for i, pt in enumerate(PTS):
        row = {'': SYSLBL[s] if i == 0 else '', 'R/Re': PTLBL[pt]}
        for f in FUNCS:
            row[LABEL[f]] = pick(f, s, pt)
        rows.append(row)

df = pd.DataFrame(rows)

with pd.ExcelWriter(XLSX, engine='openpyxl') as w:
    df.to_excel(w, sheet_name='E_total', index=False)
    sh = w.sheets['E_total']
    for col in sh.columns:
        width = max(len(str(c.value or '')) for c in col) + 2
        sh.column_dimensions[col[0].column_letter].width = min(width, 24)
    for r in sh.iter_rows(min_row=2, min_col=3):
        for c in r:
            c.number_format = '0.000000000'

print(f'\n[Xong] {XLSX}   ({len(df)} hang x {len(FUNCS)} method, Hartree)\n')
print(df.to_string(index=False))

miss = [(LABEL[f], SYSLBL[s], pt) for f in FUNCS for s in SYSTEMS for pt in PTS
        if pick(f, s, pt) is None]
if miss:
    print(f'\n*** {len(miss)} diem KHONG co nghiem hop le (o trong trong bang):')
    for f, s, pt in miss:
        print(f'    {f:12s} {s:10s} {pt}')

# =====================================================================
# KIEM TRA PIPELINE bang PBE  (chi in ra console, KHONG vao Excel)
#
# PBE la functional local -> chi co 1 basin, va bai bao da bao cao san
# sai so cua no (Table 1). Neu ORCA/PBE khong tai lap duoc cac so nay thi
# setup (basis, grid, supermolecule, frozen-core...) dang SAI, va moi con
# so DH trong bang tren cung KHONG dang tin.
# =====================================================================
EH2KCAL = 627.509608
RPTS = PTS[:4]
DL = 'dissociation_limit'
BENCH = {                       # Table 1, benchmark De (kcal/mol)
    'H2_plus_He': [64.4, 58.9, 48.7, 38.3],
    'He2_plus':   [56.9, 46.9, 31.3, 19.1],
    'NH3_2_plus': [35.9, 25.9, 13.4, 4.9],
    'H2O_2_plus': [39.7, 29.1, 16.9, 9.3],
}
PBE_PAPER = {                   # sai so PBE trong Table 1 (de doi chieu)
    'H2_plus_He': [-54.8, -50.9, -46.7, -42.3],
    'He2_plus':   [-68.3, -58.9, -49.5, -41.2],
    'NH3_2_plus': [34.0, 40.7, 47.1, 52.5],
    'H2O_2_plus': [22.0, 30.7, 38.1, 43.7],
}

print('\n' + '=' * 72)
print('  KIEM TRA PIPELINE: PBE (ORCA) vs Table 1 cua bai bao')
print('=' * 72)
print(f"{'he':10s} {'R/Re':>6s} {'sai so PBE':>11s} {'bai bao':>9s} {'lech':>7s}")

dev = []
for s in SYSTEMS:
    e_dl = pick('PBE', s, DL)
    if e_dl is None:
        print(f'{SYSLBL[s]:10s}  !! thieu diem dl -> khong kiem tra duoc')
        continue
    for i, pt in enumerate(RPTS):
        e_r = pick('PBE', s, pt)
        if e_r is None:
            print(f'{SYSLBL[s]:10s} {PTLBL[pt]:>6s}  !! thieu nghiem')
            continue
        err = (e_dl - e_r) * EH2KCAL - BENCH[s][i]
        ref = PBE_PAPER[s][i]
        d = err - ref
        dev.append(abs(d))
        flag = '' if abs(d) < 1.0 else '   <== LECH'
        print(f'{SYSLBL[s]:10s} {PTLBL[pt]:>6s} {err:11.1f} {ref:9.1f} '
              f'{d:+7.1f}{flag}')

if dev:
    mx = max(dev)
    print('-' * 72)
    print(f'  Lech lon nhat: {mx:.1f} kcal/mol')
    if mx < 1.0:
        print('  => PIPELINE OK. Cac so DH trong bang tren dang tin cay.')
    else:
        print('  => PIPELINE SAI. Kiem tra: aug-cc-pVTZ? DEFGRID3? NoFrozenCore?')
        print('     supermolecule o dl? nghiem delocalized (spin 0.50/0.50)?')
        print('     KHONG dung so DH cho den khi PBE tai lap duoc Table 1.')