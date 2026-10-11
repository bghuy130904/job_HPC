"""
AAD interaction energy: MP2 or OBMP2 in B3LYP, with SPADE, CL and counterpoise.

This script executes the selected MP2 or OBMP2 in DFT (B3LYP) embedding
R-scan calculation with Counterpoise (CP) correction.

Workflow:
    UHF reference + stability -> full-system B3LYP -> SPADE -> embedded UHF
    -> CL -> OBMP2 -> recombined embedding total energy
for the dimer and for each ghost-atom-corrected monomer (Counterpoise).

Interaction energy (CP corrected):
    E_int = E_AB(Embed) - E_A_ghost(Embed) - E_B_ghost(Embed)
"""

import gc
import os
import sys
import time
import json
from datetime import datetime
import contextlib
from pathlib import Path
import numpy as np
import pandas as pd
from pyscf import lib, scf, gto, ao2mo

# ==============================================================================
# 1. CONFIGURATION & GEOMETRY
# ==============================================================================

REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_DIR = REPO_ROOT / 'shared'
sys.path.insert(0, str(SHARED_DIR))
INPUT_FILE = Path(os.environ.get('INPUT_FILE', str(Path(__file__).resolve().parents[2] / 'inputs/aad.json')))
CONFIG = json.loads(INPUT_FILE.read_text())
METHOD = os.environ.get('METHOD', 'obmp2').lower()
if METHOD not in ('mp2', 'obmp2'):
    raise ValueError('METHOD must be mp2 or obmp2')
RUN_ID = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
RUN_DIR = Path(os.environ.get('RUN_DIR', str(REPO_ROOT / 'results/dimmer_cl' / METHOD / RUN_ID)))
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Nhận tham số từ môi trường Slurm/Bash
ENV_R = os.environ.get("R")
# Mặc định chuỗi scan R nếu không truyền từ môi trường
R_VALUES = json.loads(ENV_R) if ENV_R else CONFIG['R_VALUES']
if isinstance(R_VALUES, (int, float)):
    R_VALUES = [R_VALUES]
if not isinstance(R_VALUES, list) or not R_VALUES:
    raise ValueError("R must be a number or nonempty JSON list")
R_VALUES = [float(r) for r in R_VALUES]
if any(not np.isfinite(r) or r <= 0 for r in R_VALUES):
    raise ValueError("R values must be finite and positive")

BASIS = os.environ.get("BASIS", "def2-QZVPPD")
MEMORY = int(os.environ.get("MEMORY", 90000)) # MB
CSV_FILE = os.environ.get("CSV_FILE", str(RUN_DIR / "interaction_energies.csv"))
XLSX_FILE = os.environ.get("XLSX_FILE", str(RUN_DIR / "interaction_energies.xlsx"))
CONVERGENCE_FILE = os.environ.get("CONVERGENCE_FILE", str(RUN_DIR / "convergence.csv"))
LOG_DIR = Path(os.environ.get("LOG_DIR", str(RUN_DIR / "logs")))
CONVERGENCE_RECORDS = []

# Hằng số vật lý
HARTREE_TO_KCAL = 627.509474

MONOMER_A_RAW = CONFIG['MONOMER_A_RAW']
MONOMER_B_RAW = CONFIG['MONOMER_B_RAW']

# Thêm đường dẫn pyCMF
PYCMF_PATH = os.environ.get("PYCMF_PATH", "/share/installs/pyCMF")
if PYCMF_PATH not in sys.path:
    sys.path.insert(0, PYCMF_PATH)

from pycmf.OBDH.stability import stabilize_scf
if METHOD == 'mp2':
    from mp2_embed import UMP2_CL as EmbeddingMethod
else:
    from pycmf.OBDH import OBMP2_CL as EmbeddingMethod

# ==============================================================================
# 2. UTILITIES & HELPER FUNCTIONS
# ==============================================================================
def configure_scratch_directory():
    """Use PYSCF_SCRATCH for large temporary files when it is defined."""
    scratch_dir = os.environ.get("PYSCF_SCRATCH")
    if scratch_dir:
        os.makedirs(scratch_dir, exist_ok=True)
        lib.param.TMPDIR = scratch_dir
        print(f"PySCF scratch directory: {scratch_dir}")
    return scratch_dir

def format_coords(atom_list, is_ghost=False):
    lines = []
    prefix = "Ghost-" if is_ghost else ""
    for atom, x, y, z in atom_list:
        lines.append(f"{prefix}{atom}   {x:.9f}   {y:.9f}   {z:.9f}")
    return "\n".join(lines)

def build_molecule(atom_string):
    """Build PySCF Mole object with standard parameters."""
    mol = gto.Mole()
    mol.atom = atom_string
    mol.basis = BASIS
    mol.charge = 0
    mol.spin = 0
    mol.max_memory = MEMORY
    mol.verbose = 4
    mol.build()
    return mol

# ==============================================================================
# 3. GEOMETRY GENERATION
# ==============================================================================
def get_shifted_geometries(scale_factor):
    """
    Tạo hình học Dimer, MonA(CP), MonB(CP) dựa trên hệ số nhân R/R0.
    Dịch chuyển Monomer B dựa trên khoảng cách tâm hình học.
    """
    coords_A = np.array([[x, y, z] for _, x, y, z in MONOMER_A_RAW])
    coords_B = np.array([[x, y, z] for _, x, y, z in MONOMER_B_RAW])

    center_A = np.mean(coords_A, axis=0)
    center_B = np.mean(coords_B, axis=0)

    R_vec = center_B - center_A
    shift_vec = (scale_factor - 1.0) * R_vec

    scaled_B_raw = []
    for i, (atom, x, y, z) in enumerate(MONOMER_B_RAW):
        new_xyz = coords_B[i] + shift_vec
        scaled_B_raw.append([atom, new_xyz[0], new_xyz[1], new_xyz[2]])

    str_A_real = format_coords(MONOMER_A_RAW, is_ghost=False)
    str_A_ghost = format_coords(MONOMER_A_RAW, is_ghost=True)
    str_B_real = format_coords(scaled_B_raw, is_ghost=False)
    str_B_ghost = format_coords(scaled_B_raw, is_ghost=True)

    dimer_geo = str_A_real + "\n" + str_B_real
    monA_cp_geo = str_A_real + "\n" + str_B_ghost
    monB_cp_geo = str_A_ghost + "\n" + str_B_real

    return dimer_geo, monA_cp_geo, monB_cp_geo


# ==============================================================================
# 4. CORE CALCULATION DRIVER (OBMP2-IN-B3LYP)
# ==============================================================================
def _run_embedding_calculation(mol, active_atoms, xc_env='b3lyp', label="", record=None):
    """
    UHF reference/stability, then backend full B3LYP -> SPADE -> embedded UHF -> CL -> OBMP2.
    """
    print(f"\n--- UHF Reference for {label} ---")
    mol.incore_anyway = False

    mf = scf.UHF(mol).density_fit()
    mf.max_memory = MEMORY
    mf.verbose = mol.verbose
    mf.conv_tol = 1e-10
    mf.max_cycle = 200

    start_scf = time.time()
    mf.kernel()
    record['UHF_converged'] = bool(mf.converged)
    if not mf.converged:
        raise RuntimeError(f"{label}: UHF Not convergence")
    print(f"UHF runtime = {time.time() - start_scf:.3f} s")
    print(f"UHF Energy = {mf.e_tot:.12f} Eh")

    # Kiểm tra và ổn định nghiệm SCF
    mf = stabilize_scf(mf, max_macro_cycles=10, verbose=True)
    if not mf.converged:
        raise RuntimeError(f"{label}: stabilized UHF Not convergence")
    _, _, stable, _ = mf.stability(return_status=True)
    record['UHF_stable'] = bool(stable)
    if not stable:
        raise RuntimeError(f"{label}: unstable UHF after stabilization")

    # Nén ERI nếu có
    if hasattr(mf, "_eri") and mf._eri is not None:
        if mf._eri.ndim == 4:
            mf._eri = ao2mo.restore(8, mf._eri, mol.nao)

    gc.collect()

    print(f"\n--- {METHOD.upper()}-in-B3LYP Embedding for {label} ---")

    # Khởi tạo đối tượng OBMP2_CL
    mp_emb = EmbeddingMethod(mf)
    mp_emb.thresh = 1e-6
    mp_emb.second_order = True
    mp_emb.use_embed = True
    mp_emb.xc_env = xc_env
    mp_emb.active_atoms = active_atoms
    mp_emb.mu = 1e6
    mp_emb.use_cl = True
    mp_emb.n_shells = 2
    mp_emb.frozen = None
    mp_emb.is_hybrid = False
    mp_emb.with_df = mf.with_df
    mp_emb.max_memory = MEMORY
    mp_emb.niter = int(os.environ.get("OBMP2_NITER", "1000"))
    mp_emb.diis_space = int(os.environ.get("DIIS_SPACE", "8"))

    start_embed = time.time()
    e_tot = float(mp_emb.kernel())
    if METHOD == 'obmp2':
        record['OBMP2_converged'] = bool(getattr(mp_emb, 'converged', False))
        record['active_nocc'] = str(tuple(int(n) for n in mp_emb.get_nocc()))
        record['active_nmo'] = str(tuple(int(n) for n in mp_emb.get_nmo()))
    else:
        record['Solver'] = 'one-shot UMP2; embedded UHF convergence checked by shared module'
    if not np.isfinite(e_tot):
        raise RuntimeError(f"{label}: non-finite embedding total energy")
    if METHOD == "obmp2" and not bool(getattr(mp_emb, "converged", False)):
        raise RuntimeError(f"{label}: OBMP2 Not convergence")
    print(f"Embedding runtime = {time.time() - start_embed:.3f} s")

    if hasattr(mf, "_eri"): del mf._eri
    del mp_emb, mf
    gc.collect()

    return e_tot


def run_embedding_calculation(mol, active_atoms, xc_env='b3lyp', label=""):
    """Call the unchanged backend once; retain fragment diagnostics on failure."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label.replace(' ', '_')}.log"
    record = dict(Method=METHOD, Label=label, Basis=BASIS, Status="FAILED", Log=str(log_path))
    original_stdout = mol.stdout
    print(f"Running {label}; log: {log_path}")
    try:
        with log_path.open('w', buffering=1) as log:
            mol.stdout = log
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                energy = _run_embedding_calculation(mol, active_atoms, xc_env, label, record)
        # The backend does not expose the DFT and embedded-UHF SCF objects.
        # Reject an explicit SCF failure reported in its calculation log.
        backend_log = log_path.read_text().split(f'--- {METHOD.upper()}-in-B3LYP Embedding', 1)[-1]
        backend_scf_failed = 'SCF not converged' in backend_log
        record['Backend_SCF_no_failure_reported'] = not backend_scf_failed
        if backend_scf_failed:
            raise RuntimeError(f"{label}: backend SCF Not convergence; see {log_path}")
        record.update(Status="OK", E_total_Eh=energy)
        return energy
    except Exception as exc:
        record['Status'] = "Not convergence" if any(token in str(exc).lower() for token in ('not converg', 'unstable')) else "FAILED"
        record['Error'] = str(exc)
        raise
    finally:
        mol.stdout = original_stdout
        CONVERGENCE_RECORDS.append(record)
        pd.DataFrame(CONVERGENCE_RECORDS).to_csv(CONVERGENCE_FILE, index=False)


# ==============================================================================
# 5. INDIVIDUAL ENERGY ROUTINES
# ==============================================================================
def calculate_dimer_energy(r_value):
    """Calculate the selected method in B3LYP total energy of the AAD dimer at one R."""
    gc.collect()
    print("\n" + "#" * 80)
    print(f"#       AAD {METHOD.upper()}-in-B3LYP EMBED: R_scale = {r_value:.2f}")
    print("#" * 80)

    dimer_geo, _, _ = get_shifted_geometries(r_value)
    mol = build_molecule(dimer_geo)

    print("\nDimer information")
    print("------------------")
    print(f"R_scale             : {r_value:.2f}")
    print(f"Basis               : {BASIS}")

    # Active region Dimer: nhóm -COOH của MonA (0-3) và MonB (8-11)
    active_dimer = CONFIG['active_atoms']

    start_time = time.time()
    e_ab = run_embedding_calculation(mol, active_dimer, xc_env='b3lyp', label=f"DIMER_R{r_value:.8g}")
    runtime = time.time() - start_time

    print("\nDIMER RESULT")
    print("------------")
    print(f"E_AB ({METHOD.upper()}-in-B3LYP) = {e_ab:.12f} Eh")
    print(f"Total Runtime = {runtime:.3f} s")

    del mol
    gc.collect()
    return e_ab


def calculate_monomer_cp_energy(r_value, label):
    """
    Calculate the {METHOD.upper()}-in-B3LYP energy of one monomer with Ghost atoms (Counterpoise).
    Giữ nguyên Active Space CONFIG['active_atoms'] trùng khớp hoàn toàn với Dimer.
    """
    gc.collect()
    print("\n" + "#" * 80)
    print(f"#       {label} (Counterpoise): {METHOD.upper()}-in-B3LYP EMBED")
    print(f"#       R_scale = {r_value:.2f}")
    print("#" * 80)

    _, monA_cp_geo, monB_cp_geo = get_shifted_geometries(r_value)

    if label == "MONOMER A":
        geo = monA_cp_geo
    else:
        geo = monB_cp_geo

    mol = build_molecule(geo)

    print(f"\n{label} information")
    print("-------------------")
    print(f"Basis               : {BASIS}")

    active_atoms_cp = CONFIG['active_atoms']

    start_time = time.time()
    energy = run_embedding_calculation(mol, active_atoms_cp, xc_env='b3lyp', label=f"{label}_CP_R{r_value:.8g}")
    runtime = time.time() - start_time

    print(f"\n{label} CP Energy = {energy:.12f} Eh")
    print(f"Total Runtime = {runtime:.3f} s")

    del mol
    gc.collect()
    return energy


# ==============================================================================
# 6. INTERACTION ENERGY WORKFLOW
# ==============================================================================
def calculate_interaction_energy_at_r(r_value):
    """Perform the full CP workflow for one R value."""
    energies, statuses = {}, {}
    tasks = {
        "AB": lambda: calculate_dimer_energy(r_value),
        "A_CP": lambda: calculate_monomer_cp_energy(r_value, "MONOMER A"),
        "B_CP": lambda: calculate_monomer_cp_energy(r_value, "MONOMER B"),
    }
    for fragment, task in tasks.items():
        try:
            energies[fragment] = task()
            statuses[fragment] = "OK"
        except Exception as exc:
            energies[fragment] = np.nan
            statuses[fragment] = "Not convergence" if any(token in str(exc).lower() for token in ('not converg', 'unstable')) else f"FAILED: {exc}"
            print(f"{fragment}, R/R0={r_value}: {statuses[fragment]}")
    if not all(status == "OK" for status in statuses.values()):
        return {
            "R_scale": r_value,
            "E_AB_Embed_Eh": energies["AB"],
            "E_A_CP_Embed_Eh": energies["A_CP"],
            "E_B_CP_Embed_Eh": energies["B_CP"],
            "E_int_Eh": np.nan,
            "E_int_kcal_mol": np.nan,
            "Status": "Not convergence" if "Not convergence" in statuses.values() else "FAILED",
            **{f"Status_{fragment}": status for fragment, status in statuses.items()},
        }
    e_ab, e_a_cp, e_b_cp = (energies[k] for k in ("AB", "A_CP", "B_CP"))

    e_int = e_ab - e_a_cp - e_b_cp
    e_int_kcal = e_int * HARTREE_TO_KCAL

    print("\n" + "=" * 95)
    print(f"INTERACTION ENERGY (CP Corrected): R_scale = {r_value:.2f}")
    print("=" * 95)

    print(f"E_AB (Embed)   = {e_ab:.12f} Eh")
    print(f"E_A_CP (Embed) = {e_a_cp:.12f} Eh")
    print(f"E_B_CP (Embed) = {e_b_cp:.12f} Eh")
    print(f"E_int          = {e_int:.12f} Eh")
    print(f"E_int          = {e_int_kcal:.8f} kcal/mol")

    return {
        "R_scale": r_value,
        "E_AB_Embed_Eh": e_ab,
        "E_A_CP_Embed_Eh": e_a_cp,
        "E_B_CP_Embed_Eh": e_b_cp,
        "E_int_Eh": e_int,
        "E_int_kcal_mol": e_int_kcal,
        "Status": "OK",
        **{f"Status_{fragment}": status for fragment, status in statuses.items()},
    }


# ==============================================================================
# 7. MAIN EXECUTION LOOP
# ==============================================================================
if __name__ == "__main__":
    configure_scratch_directory()

    print("\n" + "="*95)
    print(f" >>> STARTING AAD {METHOD.upper()}-in-B3LYP EMBEDDING R-SCAN (Hệ số nhân) <<<")
    print("="*95)

    results_list = []
    total_start = time.time()

    for idx, r in enumerate(R_VALUES, start=1):
        try:
            point_start = time.time()
            print(f"\n[SCAN POINT {idx}/{len(R_VALUES)}] Chạy cho R_scale = {r:.2f}")

            result_point = calculate_interaction_energy_at_r(r)
            result_point["Point"] = idx
            result_point["Runtime_Point_s"] = time.time() - point_start

            results_list.append(result_point)

        except Exception as e:
            print(f"\n[ERROR] Lỗi tại điểm R = {r}: {e}")
            results_list.append({
                "Point": idx, "R_scale": r, "Status": f"FAILED: {str(e)}"
            })

        # Checkpoint completed points even if a later point is interrupted.
        pd.DataFrame(results_list).to_csv(CSV_FILE, index=False, encoding='utf-8')

    print("\n" + "="*95)
    print(f" >>> HOÀN THÀNH R-SCAN. Tổng thời gian: {time.time() - total_start:.2f} s <<<")
    print("="*95)

    df = pd.DataFrame(results_list)

    cols_order = ["Point", "R_scale", "E_int_kcal_mol", "E_int_Eh",
                  "E_AB_Embed_Eh", "E_A_CP_Embed_Eh", "E_B_CP_Embed_Eh",
                  "Runtime_Point_s", "Status", "Status_AB", "Status_A_CP", "Status_B_CP"]
    existing_cols = [c for c in cols_order if c in df.columns]
    df = df[existing_cols]

    # 1. Xuất Excel
    try:
        df.to_excel(XLSX_FILE, index=False, engine='openpyxl')
        print(f"\n[+] Đã xuất thành công file Excel: {XLSX_FILE}")
    except ImportError:
        print("\n[WARNING] Không thể xuất Excel vì thiếu thư viện 'openpyxl'.")
        print("Hãy cài đặt bằng: pip install openpyxl")

    # 2. Xuất CSV (backup)
    df.to_csv(CSV_FILE, index=False, encoding='utf-8')
    print(f"[+] Đã xuất thành công file CSV:   {CSV_FILE}")
