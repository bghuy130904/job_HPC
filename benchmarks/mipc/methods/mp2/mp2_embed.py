"""
MP2-in-DFT embedding (projection-based, SPADE + CL truncation).

Dùng lại toàn bộ hạ tầng của uobdh_embed.py; phần WFT được thay bằng UMP2
tiêu chuẩn của PySCF thay cho OBMP2.

Công thức năng lượng (Lee et al., Acc. Chem. Res. 2019, 52, 1359, Eq. 9;
dạng không xấp xỉ, đánh giá trên mật độ tham chiếu đã embed):

    E = E_WF[A] + ( E_DFT[g~A + gB] - E_DFT[g~A] ) + mu * tr[g~A PB]
    E_WF[A] = E_HF[g~A] + E_corr^MP2[A]

Khác OBMP2 một điểm bản chất: MP2 không tối ưu orbital, nên g~A chính là mật
độ của tham chiếu HF-in-DFT (không có mật độ relax), và test self-embedding
tái tạo MP2 thường đúng tới ~1e-9 Eh.
"""

import numpy as np
import scipy.linalg as la
from pyscf import scf, dft, mp as pyscf_mp

from pycmf.OBDH.uobdh_embed import (run_full_dft, spade_partition, build_density_matrix,
                          get_subsystem_hcore, build_embedding_potential)
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


def run_embed_ump2(mpobj, mol, h_core_full, v_emb, gamma_init, num_active_orbs,
                   atom_indices_A, use_cl=False, cl_n_shells=1, cl_mu_threshold=1e5):
    """UHF dưới thế embedding, (tuỳ chọn) cắt virtual bằng CL, rồi chạy UMP2."""
    print("   [Embedded UMP2] Initializing UHF with Embedding Potential...")
    mol_emb = mol.copy()
    na, nb = num_active_orbs
    mol_emb.nelectron = na + nb
    mol_emb.spin = na - nb

    mf_emb = scf.UHF(mol_emb).density_fit()
    mf_emb.verbose = mol.verbose
    if mpobj.with_df is not None:
        mf_emb.with_df = mpobj.with_df
        mf_emb.with_df.mol = mol_emb        # tích phân DF chỉ phụ thuộc hình học + basis
    mf_emb.max_memory = mpobj.max_memory
    original_get_veff = mf_emb.get_veff

    def get_veff_emb(mol_, dm, dm_last=0, vhf_last=0):
        veff = original_get_veff(mol_, dm, dm_last, vhf_last)
        return np.array([veff[0] + v_emb[0], veff[1] + v_emb[1]])

    mf_emb.get_veff = get_veff_emb
    mf_emb.get_hcore = lambda *args: h_core_full

    try:
        mf_emb.kernel(dm0=gamma_init)
    except Exception as e:
        print(f"   [Warning] UHF kernel failed: {e}. Trying without dm0...")
        mf_emb.kernel()
    print(f"   [Embedded UMP2] UHF-in-DFT Reference Energy: {mf_emb.e_tot:.8f}")

    if use_cl:
        print(f"   [Embedded UMP2] Performing Concentric Localization (n_shells={cl_n_shells})...")
        mf_emb = _cl_truncate(mf_emb, mol, atom_indices_A, cl_n_shells, cl_mu_threshold)
        print(f"   [Embedded UMP2] CL truncation done. NMO alpha={mf_emb.mo_coeff[0].shape[1]}, "
              f"beta={mf_emb.mo_coeff[1].shape[1]}")

    print("   [Embedded UMP2] Running UMP2 on the embedded reference...")
    pt = pyscf_mp.UMP2(mf_emb, frozen=mpobj.frozen)
    pt.verbose = mol.verbose
    pt.max_memory = mpobj.max_memory
    pt.kernel(mo_energy=mf_emb.mo_energy, mo_coeff=mf_emb.mo_coeff)
    print(f"   [Embedded UMP2] E_corr = {pt.e_corr:.8f}")

    # g~A: mat do cua tham chieu HF-in-DFT (MP2 khong toi uu orbital)
    dm_A = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)
    return pt.e_corr, (dm_A[0], dm_A[1])


def mp2_embed_kernel(mpobj):
    mol = mpobj.mol
    xc_code = mpobj.xc_env
    S = mpobj._scf.get_ovlp()

    print('\n' + '=' * 70)
    print('MP2-IN-DFT EMBEDDING WITH SPADE PARTITIONING')
    print('=' * 70)
    print('\n--- STEP 1: Running Full System DFT ---')
    ks_full = run_full_dft(mol, xc_code, df_obj=mpobj.with_df)
    print(f"Full DFT Energy ({xc_code}): {ks_full.e_tot:.8f} Eh")
    h_core_full = ks_full.get_hcore()

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
    v_emb, P_B = build_embedding_potential(mol, xc_code, S, mpobj.mu, ks_full, gamma_B, gamma_A)

    print("\n--- Running MP2 in DFT Environment ---")
    e_corr_A, gamma_wf = run_embed_ump2(
        mpobj, mol, h_core_full, v_emb, gamma_A, (na_act, nb_act), atom_indices_A,
        use_cl=mpobj.use_cl, cl_n_shells=mpobj.n_shells, cl_mu_threshold=1e5)

    gamma_wf_a, gamma_wf_b = gamma_wf
    gam_A = [gamma_wf_a, gamma_wf_b]
    gamma_relax = (gamma_wf_a + gamma_B[0], gamma_wf_b + gamma_B[1])
    e_nuc = mol.energy_nuc()

    # E_HF[g~A]: ham HF thuan, h_core TRAN (khong kem v_emb / mu*P)
    hf_A = scf.UHF(mol).density_fit()
    hf_A.verbose = mol.verbose
    if mpobj.with_df is not None:
        hf_A.with_df = mpobj.with_df
    e_hf_A = hf_A.energy_elec(gam_A, h1e=h_core_full)[0] + e_nuc

    # E_DFT[g~A]: cung functional moi truong, cung h_core tran
    ks_A = dft.UKS(mol).density_fit()
    ks_A.xc = xc_code
    ks_A.verbose = mol.verbose
    if mpobj.with_df is not None:
        ks_A.with_df = mpobj.with_df
    e_dft_A = ks_A.energy_elec(gam_A, h1e=h_core_full)[0] + e_nuc

    e_dft_full_relax = ks_full.energy_tot(dm=gamma_relax)

    e_wf_A = e_hf_A + e_corr_A
    e_baseline = e_dft_full_relax - e_dft_A
    e_ortho = mpobj.mu * (np.einsum('ij,ji', gamma_wf_a, P_B[0]) +
                          np.einsum('ij,ji', gamma_wf_b, P_B[1]))
    e_final = e_wf_A + e_baseline + e_ortho

    print("-" * 60)
    print(f"E_HF[A] (embedded reference)    : {e_hf_A:.8f}")
    print(f"E_corr (MP2, A)                 : {e_corr_A:.8f}")
    print(f"E_WF[A] = E_HF[A] + E_corr      : {e_wf_A:.8f}")
    print(f"Baseline (Full - A)             : {e_baseline:.8f}")
    print(f"Orthogonality Correction        : {e_ortho:.8f}")
    print("-" * 60)
    print(f"Total MP2-in-DFT Energy         : {e_final:.8f} Eh")
    print(f"Ref DFT Energy                  : {ks_full.e_tot:.8f} Eh")
    print(f"Difference vs DFT               : {(e_final - ks_full.e_tot)*1e6:.2f} uEh")
    print("=" * 60)

    mpobj._gamma = gamma_relax
    mpobj.e_corr = e_corr_A
    return e_final, ks_full.e_tot


class UMP2_CL:
    """MP2-in-DFT embedding với SPADE partitioning và CL virtual truncation.

    Thuộc tính điều khiển giống OBDH_CL:
        use_embed, active_atoms, mu, use_cl, n_shells
    Riêng:
        xc_env  -- functional mô tả môi trường B (mặc định 'b3lyp').
                   Không có alphaa: MP2 không phải double hybrid.
        frozen  -- truyền thẳng cho pyscf UMP2 (None = tương quan toàn bộ electron).
    """

    def __init__(self, mf, frozen=None):
        self._scf = mf
        self.mol = mf.mol
        self.with_df = getattr(mf, 'with_df', None)
        self.verbose = mf.verbose
        self.max_memory = getattr(mf, 'max_memory', 4000)

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
            self.e_tot, self.e_ref = mp2_embed_kernel(self)
        else:
            print('\n' + '=' * 70)
            print('RUNNING STANDARD UMP2 (NO EMBEDDING)')
            print('=' * 70)
            pt = pyscf_mp.UMP2(self._scf, frozen=self.frozen)
            pt.verbose = self.mol.verbose
            pt.max_memory = self.max_memory
            pt.kernel()
            self.e_corr = pt.e_corr
            self.e_tot = self._scf.e_tot + pt.e_corr
            self.e_ref = self._scf.e_tot
            print("-" * 60)
            print(f"E_HF                            : {self._scf.e_tot:.8f}")
            print(f"E_corr (MP2)                    : {pt.e_corr:.8f}")
            print(f"Total UMP2 Energy               : {self.e_tot:.8f} Eh")
            print("=" * 60)
        return self.e_tot

    def run(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.kernel()
        return self