"""
Orbital-Optimized Opposite-Spin Scaled Second-Order Correlation (O2)
Tham khảo: Rohini C. Lochan and Martin Head-Gordon, JCP 126, 164101 (2007)
"""
import numpy as np
from scipy.linalg import expm
import pyscf
from pyscf import gto, scf, ao2mo

class O2_HeadGordon_Explicit:
    def __init__(self, mol, c_os=1.2, max_cycle=50, tol_grad=1e-5):
        self.mol = mol
        self.c_os = c_os               # Hệ số cos = 1.2 tối ưu từ bài báo
        self.max_cycle = max_cycle
        self.tol_grad = tol_grad
        
        # Bước 1: Khởi tạo với RHF orbitals
        print("[-] Khởi tạo: Tính RHF làm Initial Guess Orbitals...")
        self.mf = scf.RHF(mol).run(verbose=0)
        self.nocc = mol.nelectron // 2
        self.nmo = self.mf.mo_coeff.shape[1]
        self.nvir = self.nmo - self.nocc
        
        self.h1e = self.mf.get_hcore()
        self.mo_coeff = self.mf.mo_coeff.copy()

    def get_fock_and_energy_ref(self, C):
        """Tính E_ref và Fock matrix trong AO và MO basis"""
        C_occ = C[:, :self.nocc]
        D = 2.0 * np.dot(C_occ, C_occ.T)
        V_eff = self.mf.get_veff(self.mol, D)
        
        # Năng lượng mean-field E_ref
        E_ref = np.einsum('ij,ji->', self.h1e + 0.5 * V_eff, D) + self.mol.energy_nuc()
        
        # Ma trận Fock F
        F_ao = self.h1e + V_eff
        F_mo = np.dot(C.T, np.dot(F_ao, C))
        return E_ref, F_ao, F_mo

    def compute_os_correlation(self, C, F_mo):
        """
        Tính năng lượng E_OS và amplitudes t_iajb
        E_OS = - sum_{iajb} (ia|jb)^2 / Delta_ij^ab
        """
        nocc, nvir = self.nocc, self.nvir
        C_occ = C[:, :nocc]
        C_vir = C[:, nocc:]
        
        # Orbital energies epsilon_p từ đường chéo Fock matrix
        eps = np.diag(F_mo)
        eps_i = eps[:nocc]
        eps_a = eps[nocc:]
        
        # MO 2-electron integrals: (ia|jb) -> shape: (nocc, nvir, nocc, nvir)
        eri_iajb = ao2mo.general(self.mol, (C_occ, C_vir, C_occ, C_vir), compact=False)
        ia_jb = eri_iajb.reshape(nocc, nvir, nocc, nvir)
        
        # Delta_ij^ab = eps_a + eps_b - eps_i - eps_j
        Delta = (eps_a[None, :, None, None] + eps_a[None, None, None, :] 
               - eps_i[:, None, None, None] - eps_i[None, None, :, None])
        
        Delta[np.abs(Delta) < 1e-12] = 1e-12
        
        # Biểu diễn t_{iajb} với shape chuẩn (nocc, nvir, nocc, nvir)
        t_iajb = ia_jb / Delta
        
        # Năng lượng tương quan Opposite-Spin (E_OS)
        E_os = - np.sum(t_iajb * ia_jb)
        return E_os, t_iajb, ia_jb, Delta

    def compute_l4_and_lagrangian(self, C, F_mo, t_iajb):
        """
        Tính P_oo, P_vv, số hạng L4 (Eq. 8) và toàn bộ Lagrangian L_vo (Eq. 7)
        """
        nocc, nvir = self.nocc, self.nvir
        C_occ = C[:, :nocc]
        C_vir = C[:, nocc:]
        
        # 1. Unrelaxed 1-particle Density Matrix (t_iajb có trục: 0:i, 1:a, 2:j, 3:b)
        # P_oo: P_ij shape (nocc, nocc)
        P_oo = - np.einsum('iakb,jakb->ij', t_iajb, t_iajb)
        
        # P_vv: P_ab shape (nvir, nvir)
        P_vv =   np.einsum('iajc,ibjc->ab', t_iajb, t_iajb)
        
        # Khối Fock matrix
        # F_vo có shape (nvir, nocc)
        F_vo = F_mo[nocc:, :nocc]
        
        # 2. SỐ HẠNG L4 (Equation 8): (L4)_vo = F_vo * P_oo + P_vv * F_vo
        # (L4)_ai = sum_j F_aj P_ji + sum_b P_ab F_bi
        L4 = np.dot(F_vo, P_oo) + np.dot(P_vv, F_vo)
        
        # 3. Các số hạng tích phân 2-electron:
        # eri_kjib: (C_occ, C_occ, C_occ, C_vir) -> shape (nocc, nocc, nocc, nvir) -> trục (k, j, i, b)
        eri_kjib = ao2mo.general(self.mol, (C_occ, C_occ, C_occ, C_vir), compact=False)
        eri_kjib = eri_kjib.reshape(nocc, nocc, nocc, nvir)
        
        # eri_acjb: (C_vir, C_vir, C_occ, C_vir) -> shape (nvir, nvir, nocc, nvir) -> trục (a, c, j, b)
        eri_acjb = ao2mo.general(self.mol, (C_vir, C_vir, C_occ, C_vir), compact=False)
        eri_acjb = eri_acjb.reshape(nvir, nvir, nocc, nvir)
        
        # Khớp chính xác chiều với t_iajb (k: occ, a: vir, j: occ, b: vir)
        # Thu được tensor kết quả shape (nvir, nocc) tương ứng chỉ số (a, i):
        L_2e_occ = - 2.0 * np.einsum('kajb,kjib->ai', t_iajb, eri_kjib)
        L_2e_vir = + 2.0 * np.einsum('icjb,acjb->ai', t_iajb, eri_acjb)
        
        # Đóng góp orbital energies (L3)
        L3 = np.dot(F_vo, P_oo) + np.dot(P_vv, F_vo)
        
        # Tổng Lagrangian (Eq. 7): L_vo = L1 + L2 + L3 + L4
        L_vo = L_2e_occ + L_2e_vir + L3 + L4
        
        # 4. Orbital Gradient dE/dtheta_vo (Equation 6): 2 * F_vo + 2 * c_os * L_vo
        grad_vo = 2.0 * F_vo + 2.0 * self.c_os * L_vo
        
        return L4, L_vo, grad_vo

    def kernel(self):
        """Vòng lặp Geometric Direct Minimization (GDM)"""
        print("\n" + "="*80)
        print(f"{'Iter':>4} | {'E_O2 (Hartree)':>18} | {'E_ref':>14} | {'E_OS':>12} | {'Max |Grad|':>12}")
        print("="*80)
        
        C = self.mo_coeff.copy()
        nocc, nvir = self.nocc, self.nvir
        
        for cycle in range(1, self.max_cycle + 1):
            # Tính Fock và E_ref
            E_ref, F_ao, F_mo = self.get_fock_and_energy_ref(C)
            
            # Tính E_OS và amplitudes
            E_os, t_iajb, ia_jb, Delta = self.compute_os_correlation(C, F_mo)
            
            # Cập nhật tổng năng lượng E_O2 (Equation 1)
            E_O2 = E_ref + self.c_os * E_os
            
            # Tính L4, Lagrangian và Orbital Gradient (Equation 6, 7, 8)
            L4, L_vo, grad_vo = self.compute_l4_and_lagrangian(C, F_mo, t_iajb)
            
            max_g = np.max(np.abs(grad_vo))
            print(f"{cycle:4d} | {E_O2:18.10f} | {E_ref:14.8f} | {E_os:12.8f} | {max_g:12.3e}")
            
            # Kiểm tra tiêu chí hội tụ gradient
            if max_g < self.tol_grad:
                print("="*80)
                print(f"[+] HỘI TỤ THÀNH CÔNG tại chu trình {cycle}!")
                print(f"[+] ||L4|| Frobenius tại điểm dừng: {np.linalg.norm(L4):.6e}")
                break
                
            # Tạo bước quay GDM với Preconditioner xấp xỉ Hessian
            eps = np.diag(F_mo)
            # H_diag có shape (nvir, nocc): H_{ai, ai} ≈ 4 * (eps_a - eps_i)
            H_diag = 4.0 * (eps[nocc:, None] - eps[None, :nocc])
            H_diag[np.abs(H_diag) < 0.2] = 0.2  # Damping để tránh phân kỳ
            
            theta_vo = - grad_vo / H_diag
            
            # Đóng gói ma trận phản đối xứng kappa (skew-symmetric)
            kappa = np.zeros((self.nmo, self.nmo))
            kappa[nocc:, :nocc] = theta_vo
            kappa[:nocc, nocc:] = -theta_vo.T
            
            # Biến đổi unita C_new = C_old * exp(kappa)
            U = expm(kappa)
            C = np.dot(C, U)
        else:
            print("[!] Đạt số chu trình tối đa trước khi hội tụ hoàn toàn.")
            
        self.mo_coeff = C
        self.e_tot = E_O2
        return E_O2


if __name__ == "__main__":
    mol = gto.Mole()
    mol.atom = '''
    N   0.0000   0.0000   0.0000
    N   0.0000   0.0000   1.0977
    '''
    mol.basis = 'sto-3g'
    mol.build()

    solver = O2_HeadGordon_Explicit(mol, c_os=1.2, max_cycle=30, tol_grad=1e-8)
    final_energy = solver.kernel()
    print(f"\n>> Năng lượng O2 cuối cùng: {final_energy:.10f} Hartree")