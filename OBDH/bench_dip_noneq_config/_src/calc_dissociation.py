#!/usr/bin/env python3
"""
Dipole moment doc duong phan ly FH va FCl — UHF / UMP2 / OBMP2 / OBDH.

BOI CANH
    Hait & Head-Gordon (JCTC 2018) ghi nhan: quanh diem Coulson-Fischer
    (r ~ 1.35 A o FH), MP2 vot len 5.6083 D trong khi tham chieu CCSD(2) chi
    2.4125 D — dinh nhon PHI VAT LY, sai cho (cuc dai that o r ~ 1.60 A).
    Double hybrid ke thua benh nay tu unrestricted MP2. Day la bai toan ma
    toi uu orbital sinh ra de chua -> phep thu tot nhat cho OBDH.

    Bai bao dung orbital UNRESTRICTED cho double hybrid o phan nay (khac phan
    can bang, vi tham chieu restricted cho mu PHAN KY khi r -> vo cung).
    Script nay cung dung unrestricted, khop voi ho.

VI SAO CAN CONTINUATION (bai bao KHONG mo ta buoc nay)
    Quanh diem Coulson-Fischer, nghiem doi xung va nghiem pha doi xung gan
    nhu suy bien. Quet da guess chon nghiem THAP NHAT o tung diem — nhung neu
    thu tu hai nghiem dao qua lai giua cac diem lien nhau, duong mu(r) se GAY
    va ban khong phan biet duoc gay do vat ly hay do nhay nghiem.

    Nen moi diem duoc giai theo BA cach:
      forward : dm0 = nghiem cua diem r NHO hon lien truoc
      backward: dm0 = nghiem cua diem r LON hon lien truoc
      scan    : quet da guess doc lap (minao/atom/1e), lay E thap nhat
    Cot branch_switch = 1 khi ba cach KHONG cung mot nghiem. No phai bat len
    o DUNG MOT vung (diem Coulson-Fischer). Neu rai rac -> co nhay nghiem gia.
    Chenh lech forward/backward la hysteresis, do truc tiep duoc.

DIPOLE
    UHF        : giai tich (bien phan -> Hellmann-Feynman ap dung)
    UMP2/OB*   : finite field, sai phan trung tam F = 1e-4 a.u. (nhu bai bao)
    Cot mu_dm_* van ghi de doi chieu, KHONG dung lam ket qua:
    _gamma cua pyCMF chi la mat do DINH THUC tu orbital da toi uu
    (uobdh_solver.py:650), so chiem tu nhien ra dung 1/0 — khong co dong gop
    amplitude. Do tren tap SP: trung vi lech 12.7 % (OBMP2), 2.4 % (OBDH).

Vi du:
    python calc_dissociation.py --input dissociation_inputs.json \
           --outdir /data/.../diss --system FH
    python calc_dissociation.py --outdir /data/.../diss --merge \
           --ref-json dissociation_reference.json
"""

import os
import json
import time
import glob
import argparse

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
GRAD_TOL     = 1e-4
INIT_GUESSES = ('minao', 'atom', '1e')
BASIS        = {"default": "aug-cc-pcvqz", "H": "aug-cc-pvqz"}
FIELD        = 1e-4
AU2DEBYE     = 2.541746
CURV_WARN    = 1e-6      # siet lai: OBMP2 tai r=1.65 co curv 2.2e-6 ma khong bat co
DEGEN_TOL    = 1e-6          # |E_a - E_b| nho hon -> coi la cung nghiem
ALPHAA       = (0.53, 0.39)
OBDH_THRESH  = 1e-8
OBDH_NITER   = 300


def _lazy_imports():
    global gto, scf, mp, stabilize_scf, OBDH_CL, OBMP2_CL
    from pyscf import gto, scf, mp                          # noqa: F401
    from pycmf.OBDH.stability import stabilize_scf          # noqa: F401
    from pycmf.OBDH import OBDH_CL, OBMP2_CL                # noqa: F401
    if not hasattr(np.linalg, 'linalg'):
        np.linalg.linalg = np.linalg


# ----------------------------------------------------------------------
def build_mol(atoms, r, max_memory):
    mol = gto.Mole()
    mol.atom = [(atoms[0], (0., 0., 0.)), (atoms[1], (0., 0., float(r)))]
    mol.charge = 0
    mol.spin = 0
    mol.basis = BASIS
    mol.max_memory = max_memory
    mol.verbose = 0
    mol.build()
    return mol


def _mk_uhf(mol, hcore=None):
    mf = scf.UHF(mol).density_fit(auxbasis="def2-universal-jkfit")
    mf.verbose = 0
    mf.max_cycle = 200
    if hcore is not None:
        mf.get_hcore = lambda *a, **k: hcore
    return mf


def grad_norm(mf):
    dm = mf.make_rdm1()
    return float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock(dm=dm))))


def _mixed_guess(mol, mix=0.2):
    """Tron HOMO/LUMO cho alpha de pha doi xung spin. Bat buoc voi spin=0:
    guess doi xung khong bao gio tu roi khoi nghiem RHF, ma qua diem
    Coulson-Fischer thi nghiem dung LA nghiem pha doi xung."""
    mf = scf.UHF(mol); mf.verbose = 0
    mf.max_cycle = 1
    mf.kernel()
    mo = np.array(mf.mo_coeff); occ = np.array(mf.mo_occ)
    nocc = int(occ[0].sum())
    if nocc < 1 or mo[0].shape[1] <= nocc:
        return mf.make_rdm1()
    c, s = np.cos(mix), np.sin(mix)
    a = mo[0].copy()
    a[:, nocc-1], a[:, nocc] = (c*mo[0][:, nocc-1] + s*mo[0][:, nocc],
                                -s*mo[0][:, nocc-1] + c*mo[0][:, nocc])
    return scf.uhf.make_rdm1((a, mo[1]), occ)


def _solve(mol, guess=None, dm0=None, hcore=None, stabilize=True):
    """DIIS -> Newton neu ket -> stabilize_scf. stability() chi co nghia khi
    da o diem dung, nen Newton chay TRUOC."""
    mf = _mk_uhf(mol, hcore)
    if guess is not None:
        mf.init_guess = guess
    if dm0 is None and guess is None:
        # spin=0: guess doi xung khong bao gio tu pha doi xung -> tron alpha/beta
        dm0 = _mixed_guess(mol)
    mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    if stabilize:
        # stabilize_scf chay o MOI diem, khong chi diem dau. Quet day 1.55-1.75
        # truoc day bam nhanh doi xung (S2 = 0.0000) suot 21 diem vi continuation
        # ke thua nghiem doi xung tu diem dau, trong khi qua diem Coulson-Fischer
        # thi nghiem PHA DOI XUNG moi la nghiem thap hon.
        mf = stabilize_scf(mf, max_macro_cycles=10, verbose=False)
        try:
            if abs(mf.spin_square()[0]) < 1e-6:      # van doi xung -> ep thu
                alt = _mk_uhf(mol, hcore)
                alt.kernel(dm0=_mixed_guess(mol))
                if not alt.converged:
                    alt = alt.newton(); alt.kernel(alt.mo_coeff, alt.mo_occ)
                alt = stabilize_scf(alt, max_macro_cycles=10, verbose=False)
                if alt.e_tot < mf.e_tot - 1e-9 and grad_norm(alt) <= GRAD_TOL:
                    mf = alt
        except Exception:
            pass
    return mf


def scan_solution(mol):
    """Quet da guess, tra ve nghiem hoi tu co E thap nhat."""
    best = None
    for g in INIT_GUESSES:
        try:
            mf = _solve(mol, guess=g)
            if grad_norm(mf) <= GRAD_TOL and (best is None or mf.e_tot < best.e_tot - 1e-9):
                best = mf
        except Exception:
            continue
    if best is None:
        best = _solve(mol)
    return best


# ----------------------------------------------------------------------
def _mk_solver(mf, hybrid, mo_start=None):
    s = OBDH_CL(mf) if hybrid else OBMP2_CL(mf)
    s.verbose = 0
    s.alphaa = ALPHAA
    s.thresh = OBDH_THRESH
    s.niter = OBDH_NITER
    s.second_order = True
    s.mom_select = False
    s.use_embed = False
    s.use_cl = False
    if mo_start is not None:          # continuation cho vong lap OO
        try:
            s.mo_coeff = mo_start
        except Exception:
            pass
    return s


def _energy(mol, key, hcore, dm0, mo_start=None):
    mf = _solve(mol, dm0=dm0, hcore=hcore, stabilize=False)
    if key == "uhf":
        return float(mf.e_tot)
    if key == "ump2":
        pt = mp.UMP2(mf); pt.verbose = 0; pt.kernel()
        return float(pt.e_tot)
    s = _mk_solver(mf, hybrid=(key == "obdh"), mo_start=mo_start)
    s.run()
    return float(s.ene_tot)


def _zero_field(mol, key, mf):
    """Tra ve (E, dipole tu mat do)."""
    if key == "uhf":
        return float(mf.e_tot), np.asarray(mf.dip_moment(unit='Debye', verbose=0), float), None
    if key == "ump2":
        pt = mp.UMP2(mf); pt.verbose = 0; pt.kernel()
        v = scf.hf.dip_moment(mol, pt.make_rdm1(ao_repr=True), unit='Debye', verbose=0)
        return float(pt.e_tot), np.asarray(v, float), None
    s = _mk_solver(mf, hybrid=(key == "obdh"))
    s.run()
    g = s._gamma
    v = scf.hf.dip_moment(mol, (g[0], g[1]), unit='Debye', verbose=0)
    return float(s.ene_tot), np.asarray(v, float), getattr(s, 'mo_coeff', None)


def dipole_ff(mol, key, mf0, e0, mo_start=None):
    """mu = mu_nuc - dE/dF. Diem truong bam theo nghiem truong 0."""
    h0 = _mk_uhf(mol).get_hcore()
    with mol.with_common_orig((0, 0, 0)):
        rr = mol.intor('int1e_r', comp=3)
    dm0 = mf0.make_rdm1()
    d = np.zeros(3); curv = 0.0
    for x in range(3):
        f = np.zeros(3); f[x] = FIELD
        pert = np.einsum('i,iuv->uv', f, rr)
        ep = _energy(mol, key, h0 + pert, dm0, mo_start)
        em = _energy(mol, key, h0 - pert, dm0, mo_start)
        d[x] = (ep - em) / (2 * FIELD)
        curv = max(curv, abs(ep + em - 2 * e0))
    nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    return (nuc - d) * AU2DEBYE, curv


METHODS = {"uhf": "UHF", "ump2": "UMP2", "obmp2": "OBMP2", "obdh": "OBDH"}
ANALYTIC = {"uhf"}


# ----------------------------------------------------------------------
def run_curve(system, atoms, rs, methods, max_memory, outdir):
    """Quet ca hai chieu roi tinh property tren nghiem thap nhat."""
    rs = sorted(rs)
    mols = {r: build_mol(atoms, r, max_memory) for r in rs}

    # --- pha 1: chi SCF, ba cach, ca hai chieu ---
    print(f"[{system}] pha 1: SCF forward/backward/scan tren {len(rs)} diem", flush=True)
    fwd, bwd, scn = {}, {}, {}
    dm = None
    for r in rs:                                   # r tang dan
        mf = _solve(mols[r], dm0=dm); fwd[r] = mf; dm = mf.make_rdm1()
    dm = None
    for r in reversed(rs):                         # r giam dan
        mf = _solve(mols[r], dm0=dm); bwd[r] = mf; dm = mf.make_rdm1()
    for r in rs:
        scn[r] = scan_solution(mols[r])

    rows = []
    for r in rs:
        mol = mols[r]
        E = {'fwd': fwd[r].e_tot, 'bwd': bwd[r].e_tot, 'scan': scn[r].e_tot}
        best_tag = min(E, key=E.get)
        mf = {'fwd': fwd[r], 'bwd': bwd[r], 'scan': scn[r]}[best_tag]
        row = {"system": system, "r": r, "nao": int(mol.nao),
               "E_fwd": E['fwd'], "E_bwd": E['bwd'], "E_scan": E['scan'],
               "branch_used": best_tag,
               "hysteresis_mH": (E['fwd'] - E['bwd']) * 1000.0,
               "branch_switch": int(max(E.values()) - min(E.values()) > DEGEN_TOL),
               "grad_norm": grad_norm(mf), "E_UHF": float(mf.e_tot),
               "S2": float(mf.spin_square()[0]), "status": "ok"}
        flags = []
        if row["grad_norm"] > GRAD_TOL:
            flags.append("SCF_not_converged")
        if row["branch_switch"]:
            flags.append(f"branch(dE={max(E.values())-min(E.values()):.2e})")

        # --- pha 2: property ---
        for k in methods:
            F = METHODS[k]; t1 = time.time()
            try:
                e0, dip_dm, mo_ob = _zero_field(mol, k, mf)
                row[f"E_{F}"] = e0
                row[f"mu_dm_{F}"] = float(np.linalg.norm(dip_dm))
                if k in ANALYTIC:
                    dip, curv = dip_dm, 0.0
                else:
                    dip, curv = dipole_ff(mol, k, mf, e0, mo_ob)
                    row[f"curv_{F}"] = float(curv)
                    if curv > CURV_WARN:
                        flags.append(f"{F}:FF_jump({curv:.1e})")
                # duong cong 1 chieu: giu DAU theo truc z (cuc tinh co y nghia)
                row[f"mu_{F}"] = float(dip[2])
                row[f"absmu_{F}"] = float(np.linalg.norm(dip))
            except Exception as e:
                flags.append(f"{F}:ERR({type(e).__name__})")
            row[f"t_{F}"] = round(time.time() - t1, 1)

        if flags:
            row["status"] = "WARN " + " ".join(flags)
        rows.append(row)
        os.makedirs(os.path.join(outdir, "rows"), exist_ok=True)
        pd.DataFrame([row]).to_csv(
            os.path.join(outdir, "rows", f"{system}_{r:07.4f}.csv"), index=False)
        mus = "  ".join(f"{METHODS[k]}={row.get('mu_'+METHODS[k])}" for k in methods)
        print(f"  r={r:6.3f}  {row['status']:20} branch={best_tag:5} "
              f"S2={row['S2']:.4f}  {mus}", flush=True)
    return rows


# ----------------------------------------------------------------------
def do_merge(outdir, out_xlsx, ref_json=None):
    files = sorted(glob.glob(os.path.join(outdir, "rows", "*.csv")))
    if not files:
        print(f"[MERGE] khong co file row nao trong {outdir}/rows")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(["system", "r"])
    if ref_json and os.path.exists(ref_json):
        ref = json.load(open(ref_json))
        bm = {(s, round(p['r'], 4)): p['benchmark'] for s, d in ref.items() for p in d['points']}
        df['benchmark'] = [bm.get((s, round(r, 4)), np.nan) for s, r in zip(df.system, df.r)]
    os.makedirs(os.path.dirname(out_xlsx) or ".", exist_ok=True)
    df.to_csv(os.path.splitext(out_xlsx)[0] + ".csv", index=False)
    try:
        df.to_excel(out_xlsx, index=False)
    except ImportError:
        print("[MERGE] thieu openpyxl -> chi ghi CSV")

    print(f"[MERGE] {len(df)} diem -> {out_xlsx}")
    for s, g in df.groupby("system"):
        sw = g[g.branch_switch == 1]
        print(f"\n  === {s} ({len(g)} diem) ===")
        print(f"    branch_switch tai r = {list(np.round(sw.r,3))}"
              f"   {'(1 vung lien tuc = tot)' if len(sw)<=4 else '(RAI RAC -> nghi nhay nghiem gia)'}")
        print(f"    hysteresis |fwd-bwd| max = {g.hysteresis_mH.abs().max():.3f} mH"
              f" tai r = {g.loc[g.hysteresis_mH.abs().idxmax(),'r']:.3f}")
        bad = g[g.status != 'ok']
        if len(bad):
            for _, x in bad.iterrows():
                print(f"    r={x.r:6.3f}  {x.status}")
        if 'benchmark' in g and g.benchmark.notna().any():
            k = g[g.benchmark.notna()]
            print(f"    {'method':8}{'mu_max':>9}{'r(max)':>9}{'RMSE':>9}{'MAX|d|':>9}")
            rb = k.loc[k.benchmark.abs().idxmax(), 'r']
            print(f"    {'benchmark':8}{k.benchmark.abs().max():9.4f}{rb:9.3f}"
                  f"{'-':>9}{'-':>9}")
            for F in ["UHF", "UMP2", "OBMP2", "OBDH"]:
                c = f"mu_{F}"
                if c not in k or k[c].isna().all():
                    continue
                v = k[c].abs(); e = (k[c].abs() - k.benchmark.abs())
                print(f"    {F:8}{v.max():9.4f}{k.loc[v.idxmax(),'r']:9.3f}"
                      f"{np.sqrt((e**2).mean()):9.4f}{e.abs().max():9.4f}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--system", default=None, help="FH hoac FCl (mac dinh: ca hai)")
    ap.add_argument("--methods", nargs="+", default=["uhf", "ump2", "obmp2", "obdh"],
                    choices=list(METHODS))
    ap.add_argument("--max-memory", type=int,
                    default=int(os.environ.get("PYSCF_MAX_MEMORY", "24000")))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--output-xlsx", default=None)
    ap.add_argument("--ref-json", default=None)
    args = ap.parse_args()

    out_xlsx = args.output_xlsx or os.path.join(args.outdir, "dissociation.xlsx")
    if args.merge:
        do_merge(args.outdir, out_xlsx, args.ref_json)
        return
    if not args.input:
        ap.error("--input la bat buoc khi khong dung --merge")

    _lazy_imports()
    geo = json.load(open(args.input))
    systems = {}
    for k, v in geo.items():
        systems.setdefault(v["system"], []).append((v["r"], [a[0] for a in v["geometry"]]))
    todo = [args.system] if args.system else sorted(systems)

    for s in todo:
        if s not in systems:
            print(f"[CANH BAO] khong co he {s} trong input"); continue
        pts = sorted(systems[s]); atoms = pts[0][1]
        run_curve(s, atoms, [r for r, _ in pts], args.methods, args.max_memory, args.outdir)

    do_merge(args.outdir, out_xlsx, args.ref_json)


if __name__ == "__main__":
    main()