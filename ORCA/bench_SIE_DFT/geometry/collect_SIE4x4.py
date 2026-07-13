#!/usr/bin/env python3
"""
Trich xuat ket qua ORCA cho SIE4x4 va ap dung DUNG tieu chi cua bai bao:

  1. Voi moi diem, thu ca 2 basin: _def (delocalized) va _loc (localized).
  2. Nghiem HOP LE  <=>  |<S^2> - 0.75| <= 0.05  VA  SCF converged.
  3. Lay nghiem HOP LE THAP NHAT (nguyen ly bien phan) -> dung tinh De.
  4. De = E(dissociation_limit) - E(R)   [supermolecule ca hai ve]

Xuat: bang De + sai so vs benchmark (Table 1), MUE moi functional,
      va bang audit (basin nao thang, <S^2>, spin population).
"""
import os
import re
import glob

OUT_DIR = os.environ.get('OUT_DIR', '.')
EH2KCAL = 627.509608
S2_TOL = 0.05

FUNCS = ['PBE', 'B2PLYP', 'B2GPPLYP', 'DSDPBEP86', 'PWPB95']
SYSTEMS = ['H2_plus_He', 'He2_plus', 'NH3_2_plus', 'H2O_2_plus']
RPTS = ['R_1.0', 'R_1.25', 'R_1.5', 'R_1.75']
DL = 'dissociation_limit'

BENCH = {
    'H2_plus_He': [64.4, 58.9, 48.7, 38.3],
    'He2_plus':   [56.9, 46.9, 31.3, 19.1],
    'NH3_2_plus': [35.9, 25.9, 13.4, 4.9],
    'H2O_2_plus': [39.7, 29.1, 16.9, 9.3],
}


def parse(path):
    """-> dict(E, s2, converged, spin=[...]) hoac None."""
    if not os.path.exists(path):
        return None
    txt = open(path, errors='ignore').read()

    m = re.findall(r'FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)', txt)
    if not m:
        return None
    E = float(m[-1])

    s2 = None
    m = re.findall(r'Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)', txt)
    if m:
        s2 = float(m[-1])

    conv = ('SUCCESS' in txt and 'SCF NOT CONVERGED' not in txt)

    # Mulliken spin population theo nguyen tu (block cuoi cung)
    spin = []
    blocks = re.findall(
        r'MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS.*?\n(.*?)\n\s*Sum of atomic charges',
        txt, re.S)
    if blocks:
        for line in blocks[-1].strip().split('\n'):
            f = line.split()
            if len(f) >= 5:
                try:
                    spin.append(float(f[-1]))
                except ValueError:
                    pass
    return dict(E=E, s2=s2, conv=conv, spin=spin)


def valid(r):
    if r is None or not r['conv']:
        return False
    if r['s2'] is None:
        return True                       # PBE/RKS co the khong in <S^2>
    return abs(r['s2'] - 0.75) <= S2_TOL


def pick(func, sysname, pt):
    """Chon nghiem hop le thap nhat giua _def va _loc."""
    cands = []
    for kind in ('def', 'loc'):
        r = parse(f'{OUT_DIR}/{func}_{sysname}_{pt}_{kind}.out')
        if r is None:
            continue
        r['kind'] = kind
        (cands if valid(r) else []).append(r)
    if not cands:
        return None
    return min(cands, key=lambda r: r['E'])


audit = []
print(f"\n{'='*86}")
print(f"{'func':11s} {'he':12s} {'R/Re':>6s} {'De':>9s} {'bench':>8s} "
      f"{'sai so':>8s}  {'basin':>5s} {'<S2>':>6s}")
print('=' * 86)

mues = {}
for func in FUNCS:
    errs = []
    for s in SYSTEMS:
        rdl = pick(func, s, DL)
        if rdl is None:
            print(f"{func:11s} {s:12s}  !! khong co nghiem hop le tai dl -> BO QUA")
            continue
        for i, pt in enumerate(RPTS):
            r = pick(func, s, pt)
            if r is None:
                print(f"{func:11s} {s:12s} {pt:>6s}  !! khong co nghiem hop le")
                continue
            de = (rdl['E'] - r['E']) * EH2KCAL
            b = BENCH[s][i]
            err = de - b
            errs.append(abs(err))
            s2 = r['s2'] if r['s2'] is not None else float('nan')
            print(f"{func:11s} {s:12s} {pt.replace('R_',''):>6s} {de:9.1f} "
                  f"{b:8.1f} {err:8.1f}  {r['kind']:>5s} {s2:6.3f}")
            audit.append((func, s, pt, r['kind'], r['E'], r['s2'], r['spin']))
        # audit diem dl
        audit.append((func, s, DL, rdl['kind'], rdl['E'], rdl['s2'], rdl['spin']))
    if errs:
        mues[func] = sum(errs) / len(errs)
    print('-' * 86)

print(f"\n{'='*40}\n  MUE (cationic, kcal/mol)\n{'='*40}")
print(f"  {'PBE (tham chieu Table 1)':30s} 45.1   <- bai bao")
for f, m in mues.items():
    print(f"  {f:30s} {m:5.1f}")

print(f"\n{'='*86}\n  AUDIT: basin thang tai tung diem (quan trong cho phan bien luan)\n{'='*86}")
print(f"{'func':11s} {'he':12s} {'diem':>20s} {'basin':>6s} {'E':>15s} {'<S2>':>7s}  spin")
for func, s, pt, kind, E, s2, spin in audit:
    sp = ' '.join(f'{v:+.2f}' for v in spin[:4])
    s2s = f'{s2:.3f}' if s2 is not None else '  -  '
    print(f"{func:11s} {s:12s} {pt:>20s} {kind:>6s} {E:15.8f} {s2s:>7s}  {sp}")
