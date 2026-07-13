#!/usr/bin/env python3
"""
Sinh toan bo file .inp ORCA cho SIE4x4 (Bao-Gagliardi-Truhlar, JPCL 2018)
voi cac functional double-hybrid + PBE (validation).

Protocol theo bai bao / SI:
  - Gioi han phan ly = SUPERMOLECULE (2 manh cach xa), KHONG tinh rieng roi cong.
  - Cho phep pha doi xung spin & khong gian -> phai THU CA HAI basin:
      * run "def" : guess mac dinh cua ORCA  -> thuong ra nghiem delocalized
      * run "loc" : fragment guess (cation + neutral) -> ra nghiem localized
    roi lay nghiem HOP LE THAP NHAT (bien phan).
  - Kiem <S^2> ~ 0.75 va spin population tren tung manh.

Moi diem (functional, he, R) sinh 4 input:
    <tag>_fragA.inp   manh mang lo trong (cation, doublet)
    <tag>_fragB.inp   manh trung hoa (singlet)
    <tag>_loc.inp     supermolecule, MOREAD tu merged.gbw  -> basin localized
    <tag>_def.inp     supermolecule, guess mac dinh        -> basin delocalized
PBE chi sinh _def (chi co 1 basin; dung de validate voi Table 1).
"""
import json
import os
import numpy as np

OUTDIR = 'orca_inp'
BASIS = 'aug-cc-pVTZ'
AUXC = 'aug-cc-pVTZ/C'      # aux cho RI-MP2 (bat buoc cho DH)
AUXJ = 'aug-cc-pVTZ/J'      # aux cho RIJCOSX
NPROC = 16                  # khop --ntasks trong SLURM
MAXCORE = 6000              # MB / core

# --- functional: ten ORCA -> (co phan PT2?, %HF de ghi chu) ---
FUNCTIONALS = {
    'PBE':         (False, 0),      # validation vs Table 1
    'B2PLYP':      (True, 53),      # cung %HF voi OBDH  <-- so sanh chinh
    'B2GP-PLYP':   (True, 65),
    'DSD-PBEP86':  (True, 69),
    'PWPB95':      (True, 50),
}

SYSTEMS = ['H2_plus_He', 'He2_plus', 'NH3_2_plus', 'H2O_2_plus']
POINTS = ['R_1.0', 'R_1.25', 'R_1.5', 'R_1.75', 'dissociation_limit']
HEAVY = {'H2_plus_He': 'H', 'He2_plus': 'He', 'NH3_2_plus': 'N', 'H2O_2_plus': 'O'}


# ---------------------------------------------------------------- fragments
def split_fragments(sysname, coords):
    """Chia thanh 2 manh. fragA = manh chua lo trong (cation, doublet),
    fragB = manh trung hoa (singlet).

    Quy tac: gan moi H vao nguyen tu nang GAN NHAT (robust o moi khoang cach,
    khong dung cutoff tuyet doi nhu ban goc - o R_1.0 hai N chi cach 2.17 A).
    """
    el = [a['element'] for a in coords]
    xyz = np.array([a['coordinates'] for a in coords])

    if sysname == 'He2_plus':
        # He+ (lo trong) + He
        return ([0], 1, 2), ([1], 0, 1)

    if sysname == 'H2_plus_He':
        # H* trung hoa (giu electron, doublet) | {H+, He} : +1, 2e, singlet
        # atom0 = H(-z), atom1 = H(+z), atom2 = He(+z xa)
        return ([0], 0, 2), ([1, 2], 1, 1)

    # NH3_2+ / H2O_2+ : nhom H theo nguyen tu nang gan nhat
    hv = [i for i, e in enumerate(el) if e == HEAVY[sysname]]
    assert len(hv) == 2, f'{sysname}: expected 2 heavy atoms'
    g = {hv[0]: [hv[0]], hv[1]: [hv[1]]}
    for i, e in enumerate(el):
        if i in hv:
            continue
        d = [np.linalg.norm(xyz[i] - xyz[h]) for h in hv]
        g[hv[int(np.argmin(d))]].append(i)
    a = sorted(g[hv[0]])   # manh chua nguyen tu nang dau tien -> mang lo trong
    b = sorted(g[hv[1]])
    return (a, 1, 2), (b, 0, 1)


# ---------------------------------------------------------------- inp writer
def kwline(func, extra=''):
    has_pt2, _ = FUNCTIONALS[func]
    ks = ['UKS', func, BASIS]
    if has_pt2:
        ks.append(AUXC)                 # RI-MP2 aux : BAT BUOC cho DH
    ks += ['RIJCOSX', AUXJ, 'TightSCF', 'DEFGRID3', 'NoFrozenCore', 'UNO']
    if extra:
        ks.append(extra)
    return '! ' + ' '.join(ks)


def geom_block(coords, idx, charge, mult):
    lines = [f'* xyz {charge} {mult}']
    for i in idx:
        e = coords[i]['element']
        x, y, z = coords[i]['coordinates']
        lines.append(f'{e:2s} {x:>15.8f} {y:>15.8f} {z:>15.8f}')
    lines.append('*')
    return '\n'.join(lines)


HEADER = """# {title}
# {note}
%pal nprocs {np} end
%maxcore {mc}
"""

SCF_STAB = """%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 500
end
"""


def write(path, text):
    with open(path, 'w') as f:
        f.write(text.rstrip() + '\n')


def main():
    data = json.load(open('input.json'))
    os.makedirs(OUTDIR, exist_ok=True)
    recipes = []   # (func, sys, point, has_loc)

    for func, (has_pt2, xhf) in FUNCTIONALS.items():
        fsafe = func.replace('-', '')
        for s in SYSTEMS:
            for p in POINTS:
                coords = data[s][p]
                (ia, qa, ma), (ib, qb, mb) = split_fragments(s, coords)
                order = list(ia) + list(ib)          # fragA truoc, fragB sau
                tag = f'{fsafe}_{s}_{p}'
                note = f'{func} ({xhf}% HF), {BASIS}, SIE4x4 {s} {p}'

                # --- run mac dinh (basin delocalized) ---
                txt = (HEADER.format(title=f'{tag}_def', note=note, np=NPROC, mc=MAXCORE)
                       + kwline(func) + '\n' + SCF_STAB + '\n'
                       + geom_block(coords, order, 1, 2) + '\n')
                write(f'{OUTDIR}/{tag}_def.inp', txt)

                if not has_pt2:      # PBE: 1 basin, khong can loc guess
                    recipes.append((func, s, p, False))
                    continue

                # --- fragment A (cation, mang lo trong) ---
                txt = (HEADER.format(title=f'{tag}_fragA', note=note + ' | fragA (cation)',
                                     np=NPROC, mc=MAXCORE)
                       + kwline(func) + '\n\n'
                       + geom_block(coords, ia, qa, ma) + '\n')
                write(f'{OUTDIR}/{tag}_fragA.inp', txt)

                # --- fragment B (trung hoa) ---
                txt = (HEADER.format(title=f'{tag}_fragB', note=note + ' | fragB (neutral)',
                                     np=NPROC, mc=MAXCORE)
                       + kwline(func) + '\n\n'
                       + geom_block(coords, ib, qb, mb) + '\n')
                write(f'{OUTDIR}/{tag}_fragB.inp', txt)

                # --- run localized (MOREAD tu merged.gbw) ---
                txt = (HEADER.format(title=f'{tag}_loc', note=note + ' | fragment (localized) guess',
                                     np=NPROC, mc=MAXCORE)
                       + kwline(func, extra='MOREAD') + '\n'
                       + '%moinp "merged.gbw"\n'
                       + SCF_STAB + '\n'
                       + geom_block(coords, order, 1, 2) + '\n')
                write(f'{OUTDIR}/{tag}_loc.inp', txt)

                recipes.append((func, s, p, True))

    with open(f'{OUTDIR}/recipes.json', 'w') as f:
        json.dump(recipes, f, indent=1)

    n = len(os.listdir(OUTDIR)) - 1
    print(f'Da sinh {n} file .inp trong {OUTDIR}/')
    for func, (hp, xhf) in FUNCTIONALS.items():
        kind = 'DH (def+loc)' if hp else 'GGA (def only)'
        print(f'  {func:12s} {xhf:3d}% HF   {kind}')


if __name__ == '__main__':
    main()
