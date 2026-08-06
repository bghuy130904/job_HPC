"""
(Double hybrid hoac DFT)-in-DFT embedding, projection-based (SPADE + CL truncation).

Dung lai toan bo ha tang cua uobdh_embed.py / ump2_embed.py. Vung A duoc mo ta bang
mot functional `base_xc` cong (tuy chon) mot so hang PT2 one-shot he so `a_c`:

    E_WF[A] = E_KS[g~A ; base_xc] + a_c * E_corr^MP2[A]

Tong nang luong (Lee et al., Acc. Chem. Res. 2019, 52, 1359, Eq. 9 / Eq. 1):

    E = E_WF[A] + ( E_DFT[g~A + gB ; xc_env] - E_DFT[g~A ; xc_env] ) + mu * tr[g~A PB]

Cau hinh:
    B2PLYP-in-DFT : base_xc = '0.53*HF + 0.47*B88, 0.73*LYP', a_c = 0.27
    PBE0-in-DFT   : base_xc = 'pbe0',                          a_c = 0.0
    DSD/other DH  : dat base_xc + a_c tuong ung

Khi a_c = 0 va base_xc == xc_env, day la DFT-in-DFT dong nhat: tong nang luong
phai trung khit KS-DFT toan he (test dong nhat manh nhat cua framework).
"""

import numpy as np
import scipy.linalg as la
from pyscf import lib, scf, dft, mp as pyscf_mp

from pycmf.OBDH.uobdh_embed import (run_full_dft, spade_partition, build_density_matrix,
                          build_embedding_potential)
from pycmf.OBDH.CL_embed import concentric_localization

def _cl_truncate(mf_emb, mol, atom_indices_A, n_shells, mu_threshold=1e5, verbose=True):
    """Cắt không gian virtual bằng Concentric Localization, ghi đè MO của mf_emb."""
    active_aos = []
    aoslice = mol.aoslice_by_atom()
    for atom_id in atom_indices_A:
        p0, p1 = aoslice[atom_id][2], aoslice[atom_id][3]
        active_aos.extend(range(p0, p1))

    S_mat = mf_emb.get_ovlp()
    F_mat = mf_emb.get_fock()

    new_mo_coeff, new_mo_energy, new_mo_occ = [], [], []
    for s in [0, 1]:
        C_s, occ_s, eps_s, F_s = mf_emb.mo_coeff[s], mf_emb.mo_occ[s], mf_emb.mo_energy[s], F_mat[s]

        idx_occ = occ_s > 0
        C_occ_A, eps_occ_A = C_s[:, idx_occ], eps_s[idx_occ]

        idx_vir_eff = (occ_s == 0) & (eps_s < mu_threshold)
        C_vir_eff = C_s[:, idx_vir_eff]

        C_vir_CL = concentric_localization(C_vir_eff, S_mat, F_s, active_aos,
                                           n_shells=n_shells, verbose=verbose)
        F_vir = C_vir_CL.T.conj() @ F_s @ C_vir_CL
        evals_vir, evecs_vir = la.eigh(F_vir)
        C_vir_CL = C_vir_CL @ evecs_vir

        new_mo_coeff.append(np.hstack([C_occ_A, C_vir_CL]))
        new_mo_energy.append(np.concatenate([eps_occ_A, evals_vir]))
        new_mo_occ.append(np.concatenate([np.ones(C_occ_A.shape[1]),
                                          np.zeros(C_vir_CL.shape[1])]))

    mf_emb.mo_coeff = (new_mo_coeff[0], new_mo_coeff[1])
    mf_emb.mo_energy = (new_mo_energy[0], new_mo_energy[1])
    mf_emb.mo_occ = (new_mo_occ[0], new_mo_occ[1])
    return mf_emb

def _patch_veff(mf, v_emb):
    """Cong v_emb vao get_veff, giu nguyen tag ecoul/exc de energy_elec cua UKS con dung."""
    original = mf.get_veff

    def get_veff_emb(mol_=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        veff = original(mol_, dm, dm_last, vhf_last, hermi)
        v = np.asarray(veff) + np.asarray(v_emb)
        tags = {}
        for key in ('ecoul', 'exc', 'vj', 'vk'):
            if hasattr(veff, key):
                tags[key] = getattr(veff, key)
        return lib.tag_array(v, **tags) if tags else v

    mf.get_veff = get_veff_emb
    return mf


def run_embed_dh(mpobj, mol, h_core_full, v_emb, gamma_init, num_active_orbs,
                 atom_indices_A, use_cl=False, cl_n_shells=1, cl_mu_threshold=1e5):
    """KS(base_xc) duoi the embedding, (tuy chon) cat virtual bang CL, roi PT2 one-shot."""
    print(f"   [Embedded DH] Initializing UKS({mpobj.base_xc}) with Embedding Potential...")
    mol_emb = mol.copy()
    na, nb = num_active_orbs
    mol_emb.nelectron = na + nb
    mol_emb.spin = na - nb

    ks_emb = dft.UKS(mol_emb).density_fit()
    ks_emb.xc = mpobj.base_xc
    ks_emb.verbose = mol.verbose
    if mpobj.with_df is not None:
        ks_emb.with_df = mpobj.with_df
        ks_emb.with_df.mol = mol_emb
    ks_emb.max_memory = mpobj.max_memory
    ks_emb.get_hcore = lambda *args: h_core_full
    _patch_veff(ks_emb, v_emb)

    try:
        ks_emb.kernel(dm0=gamma_init)
    except Exception as e:
        print(f"   [Warning] UKS kernel failed: {e}. Trying without dm0...")
        ks_emb.kernel()
    print(f"   [Embedded DH] KS-in-DFT converged (SCF flag = {ks_emb.converged})")

    if use_cl and mpobj.a_c != 0.0:
        print(f"   [Embedded DH] Performing Concentric Localization (n_shells={cl_n_shells})...")
        ks_emb = _cl_truncate(ks_emb, mol, atom_indices_A, cl_n_shells, cl_mu_threshold)
        print(f"   [Embedded DH] CL truncation done. NMO alpha={ks_emb.mo_coeff[0].shape[1]}, "
              f"beta={ks_emb.mo_coeff[1].shape[1]}")

    e_corr = 0.0
    if mpobj.a_c != 0.0:
        print(f"   [Embedded DH] Running one-shot PT2 on the KS reference (a_c={mpobj.a_c})...")
        pt = pyscf_mp.UMP2(ks_emb, frozen=mpobj.frozen)
        pt.verbose = mol.verbose
        pt.max_memory = mpobj.max_memory
        pt.kernel(mo_energy=ks_emb.mo_energy, mo_coeff=ks_emb.mo_coeff)
        e_corr = pt.e_corr
        print(f"   [Embedded DH] E_corr(PT2) = {e_corr:.8f},  a_c*E_corr = {mpobj.a_c*e_corr:.8f}")

    dm_A = ks_emb.make_rdm1(ks_emb.mo_coeff, ks_emb.mo_occ)
    return e_corr, (dm_A[0], dm_A[1])


def dh_embed_kernel(mpobj):
    mol = mpobj.mol
    xc_env = mpobj.xc_env

    print('\n' + '=' * 70)
    print(f'{mpobj.method_name}-IN-DFT EMBEDDING WITH SPADE PARTITIONING')
    print('=' * 70)
    print(f'  base_xc (region A) : {mpobj.base_xc}   a_c = {mpobj.a_c}')
    print(f'  xc_env  (region B) : {xc_env}')
    print('\n--- STEP 1: Running Full System DFT ---')
    ks_full = run_full_dft(mol, xc_env, df_obj=mpobj.with_df)
    print(f"Full DFT Energy ({xc_env}): {ks_full.e_tot:.8f} Eh")
    h_core_full = ks_full.get_hcore()
    S = ks_full.get_ovlp()

    C_occ_a = ks_full.mo_coeff[0][:, ks_full.mo_occ[0] > 0]
    C_occ_b = ks_full.mo_coeff[1][:, ks_full.mo_occ[1] > 0]
    atom_indices_A = mpobj.active_atoms

    print("\n --- Partitioning ---")
    C_A_a, C_B_a = spade_partition(mol, S, C_occ_a, atom_indices_A, True, "Alpha")
    C_A_b, C_B_b = spade_partition(mol, S, C_occ_b, atom_indices_A, False, "Beta")

    na_act, nb_act = C_A_a.shape[1], C_A_b.shape[1]
    gamma_A = (build_density_matrix(C_A_a), build_density_matrix(C_A_b))
    gamma_B = (build_density_matrix(C_B_a), build_density_matrix(C_B_b))

    print("\n--- Constructing Potentials ---")
    v_emb, P_B = build_embedding_potential(mol, xc_env, S, mpobj.mu, ks_full, gamma_B, gamma_A)

    print(f"\n--- Running {mpobj.method_name} in DFT Environment ---")
    e_corr_A, gamma_wf = run_embed_dh(
        mpobj, mol, h_core_full, v_emb, gamma_A, (na_act, nb_act), atom_indices_A,
        use_cl=mpobj.use_cl, cl_n_shells=mpobj.n_shells, cl_mu_threshold=1e5)

    gam_A = [gamma_wf[0], gamma_wf[1]]
    gamma_relax = (gamma_wf[0] + gamma_B[0], gamma_wf[1] + gamma_B[1])
    e_nuc = mol.energy_nuc()

    # E_KS[g~A ; base_xc] -- h_core TRAN, khong kem v_emb / mu*P
    ks_A_base = dft.UKS(mol).density_fit()
    ks_A_base.xc = mpobj.base_xc
    ks_A_base.verbose = mol.verbose
    if mpobj.with_df is not None:
        ks_A_base.with_df = mpobj.with_df
    e_base_A = ks_A_base.energy_elec(gam_A, h1e=h_core_full)[0] + e_nuc

    # E_DFT[g~A ; xc_env] -- cung h_core tran
    ks_A_env = dft.UKS(mol).density_fit()
    ks_A_env.xc = xc_env
    ks_A_env.verbose = mol.verbose
    if mpobj.with_df is not None:
        ks_A_env.with_df = mpobj.with_df
    e_env_A = ks_A_env.energy_elec(gam_A, h1e=h_core_full)[0] + e_nuc

    e_dft_full_relax = ks_full.energy_tot(dm=gamma_relax)

    e_wf_A = e_base_A + mpobj.a_c * e_corr_A
    e_baseline = e_dft_full_relax - e_env_A
    e_ortho = mpobj.mu * (np.einsum('ij,ji', gamma_wf[0], P_B[0]) +
                          np.einsum('ij,ji', gamma_wf[1], P_B[1]))
    e_final = e_wf_A + e_baseline + e_ortho

    print("-" * 60)
    print(f"E_base[A] ({mpobj.base_xc})".ljust(32)[:32] + f": {e_base_A:.8f}")
    print(f"a_c * E_corr (PT2, A)           : {mpobj.a_c * e_corr_A:.8f}")
    print(f"E_WF[A]                         : {e_wf_A:.8f}")
    print(f"Baseline (Full - A)             : {e_baseline:.8f}")
    print(f"Orthogonality Correction        : {e_ortho:.8f}")
    print("-" * 60)
    print(f"Total {mpobj.method_name}-in-DFT Energy : {e_final:.8f} Eh")
    print(f"Ref DFT Energy                  : {ks_full.e_tot:.8f} Eh")
    print(f"Difference vs DFT               : {(e_final - ks_full.e_tot)*1e6:.2f} uEh")
    print("=" * 60)

    mpobj._gamma = gamma_relax
    mpobj.e_corr = e_corr_A
    return e_final, ks_full.e_tot


class DH_CL:
    """(Double hybrid | DFT)-in-DFT embedding voi SPADE partitioning va CL truncation.

        m = DH_CL(mf, base_xc='0.53*HF + 0.47*B88, 0.73*LYP', a_c=0.27)  # B2PLYP-in-DFT
        m = DH_CL(mf, base_xc='pbe0', a_c=0.0)                            # PBE0-in-DFT
    """

    PRESETS = {
        'b2plyp':  ('0.53*HF + 0.47*B88, 0.73*LYP', 0.27),
        'pbe0':    ('pbe0', 0.0),
        'b3lyp':   ('b3lyp', 0.0),
        'dsd-blyp': ('0.71*HF + 0.29*B88, 0.54*LYP', 0.46),
    }

    def __init__(self, mf, base_xc=None, a_c=None, preset=None, frozen=None):
        self._scf = mf
        self.mol = mf.mol
        self.with_df = getattr(mf, 'with_df', None)
        self.verbose = mf.verbose
        self.max_memory = getattr(mf, 'max_memory', 4000)

        if preset is not None:
            base_xc, a_c = self.PRESETS[preset.lower()]
        self.base_xc = base_xc if base_xc is not None else 'pbe0'
        self.a_c = 0.0 if a_c is None else a_c
        self.method_name = (preset.upper() if preset else
                            ('DH' if self.a_c else self.base_xc.upper()))

        self.use_embed = False
        self.use_cl = False
        self.active_atoms = []
        self.n_shells = 1
        self.mu = 1e6
        self.xc_env = 'b3lyp'
        self.frozen = frozen

        self.e_corr = None
        self.e_tot = None
        self.e_ref = None
        self._gamma = None

    def kernel(self):
        if self.use_embed:
            self.e_tot, self.e_ref = dh_embed_kernel(self)
        else:
            print('\n' + '=' * 70)
            print(f'RUNNING STANDARD {self.method_name} (NO EMBEDDING)')
            print('=' * 70)
            ks = dft.UKS(self.mol).density_fit()
            ks.xc = self.base_xc
            ks.verbose = self.mol.verbose
            ks.max_memory = self.max_memory
            if self.with_df is not None:
                ks.with_df = self.with_df
            ks.kernel()
            e_corr = 0.0
            if self.a_c != 0.0:
                pt = pyscf_mp.UMP2(ks, frozen=self.frozen)
                pt.verbose = self.mol.verbose
                pt.max_memory = self.max_memory
                pt.kernel()
                e_corr = pt.e_corr
            self.e_corr = e_corr
            self.e_tot = ks.e_tot + self.a_c * e_corr
            self.e_ref = ks.e_tot
            print("-" * 60)
            print(f"E_base ({self.base_xc})".ljust(32)[:32] + f": {ks.e_tot:.8f}")
            print(f"a_c * E_corr (PT2)              : {self.a_c * e_corr:.8f}")
            print(f"Total {self.method_name} Energy       : {self.e_tot:.8f} Eh")
            print("=" * 60)
        return self.e_tot

    def run(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.kernel()
        return self