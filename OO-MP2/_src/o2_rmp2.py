"""
Orbital-Optimized Opposite-Spin Scaled Second-Order Correlation (O2)
Reference: Rohini C. Lochan and Martin Head-Gordon, JCP 126, 164101 (2007)
"""
import numpy as np
from scipy.linalg import expm
import pyscf
from pyscf import gto, scf, ao2mo

class O2_RMP2:
    def __init__(self, mol, c_os=1.2, max_cycle=50, tol_grad=1e-5, verbose=0):
        self.mol = mol
        self.c_os= c_os
        self.max_cycle = max_cycle 
        self.tol_grad  = tol_grad

        print("--- RHF calculating for initial guess orbitals ---")
        self.mf = scf.RHF(mol).run(verbose=verbose)
        self.nocc = mol.nelectron // 2
        self.nmo  = self.mf.mo_coeff.shape[1]
        self.nvir = self.nmo - self.nocc

        self.h1e = self.mf.get_hcore()
        self.mo_coeff = self.mf.mo_coeff.copy()

    def get_fock_and_energy_ref(self,C):
        """Calculating E_ref and Fock matrix on AO and MO basis"""
        C_occ = C[:, :self.nocc]
        D     = 2.0 * np.dot(C_occ, C_occ.T)
        V_eff = self.mf.get_veff(self.mol, D)

        E_ref = np.einsum('ij,ji->', self.h1e + 0.5 * V_eff, D) \
                + self.mol.energy_nuc() 

        F_ao  = self.h1e + V_eff
        F_mo  = np.dot(C.T, np.dot(F_ao, C))
        return E_ref, F_ao, F_mo

    def compute_os_correlation_energy_and_amplitude(self, C, F_mo):
        """
        Calculating opposite-spin energy E_os and amplitude t_iajb
        E_os = - sum_{iajb} (ia|jb)^2 / Delta_ij^ab
        """

        nocc, nvir = self.nocc, self.nvir
        C_occ = C[:, :nocc]
        C_vir = C[:, nocc:]

        eps = np.diag(F_mo)
        eps_i = eps[:nocc]
        eps_a = eps[nocc:]

        eri_iajb = ao2mo.general(self.mol, (C_occ, C_vir, C_occ, C_vir),\
                                  compact=False)
        ia_jb    = eri_iajb.reshape(nocc, nvir, nocc, nvir)

        Delta = (eps_a[None, :, None, None] + eps_a[None, None, None, :]\
                 - eps_i[:, None, None, None] - eps_i[None, None, :, None]) 
        Delta[np.abs(Delta) < 1e-12] = 1e-12

        t_iajb = ia_jb / Delta

        E_os = -np.sum(t_iajb * ia_jb)
        return E_os, t_iajb, ia_jb, Delta

    def compute_l4_and_lagrangian(self, C, F_mo, t_iajb):
        """
        Calculating the oo and vv elements of the OS one-particle density matrix P_oo, P_vv
        and the fourth OS Lagrangian L4 term (Eq.8) which arises because the orbitals do not
        satisfy the Brillouin condition F_vo=0, and is given by :
                L4_vo = F_vo * P_oo + F_ov * P_vv
        """
        nocc, nvir = self.nocc, self.nvir
        C_occ = C[:, :nocc]
        C_vir = C[:, nocc:]
        
        P_oo = - np.einsum('iakb,jakb->ij', t_iajb, t_iajb)  
        P_vv =   np.einsum('iajc,ibjc->ab', t_iajb, t_iajb)  
        
        F_vo = F_mo[nocc:, :nocc]   
        L4 = np.dot(F_vo, P_oo) + np.dot(P_vv, F_vo)
        

        eri_abjc = ao2mo.general(self.mol, (C_vir, C_vir, C_occ, C_vir), 
                                 compact=False).reshape(nvir, nvir, nocc, nvir)
        L1 = np.einsum('ibjc,abjc->ai', t_iajb, eri_abjc)
        
        eri_jkib = ao2mo.general(self.mol, (C_occ, C_occ, C_occ, C_vir), 
                                 compact=False).reshape(nocc, nocc, nocc, nvir)
        L2 = - np.einsum('jakb,jkib->ai', t_iajb, eri_jkib)
        
        eri_jkai = ao2mo.general(self.mol, (C_occ, C_occ, C_vir, C_occ), 
                                 compact=False).reshape(nocc, nocc, nvir, nocc)
        eri_bcai = ao2mo.general(self.mol, (C_vir, C_vir, C_vir, C_occ), 
                                 compact=False).reshape(nvir, nvir, nvir, nocc)
        
        L3 = 2.0 * np.einsum('jk,jkai->ai', P_oo, eri_jkai) + \
             2.0 * np.einsum('bc,bcai->ai', P_vv, eri_bcai)
        
        L_vo = L1 + L2 + L3 + L4

        # Orbital Gradient dE/dtheta_vo (Equation 6): 2 * F_vo + 2 * c_os * L_vo
        grad_vo = 2.0 * F_vo + 2.0 * self.c_os * L_vo
        
        return L4, L_vo, grad_vo
    
    def kernel(self):
        """
        Geometric Direct Minimization (GDM)
        """
        print("\n" + "="*80)
        print(f"{'Iter':>4} | {'E_O2 (Hartree)':>18} | {'E_ref':>14} | {'E_OS':>12} | {'Max |Grad|':>12}")
        print("="*80)
        C = self.mo_coeff.copy()
        nocc, nvir = self.nocc, self.nvir

        for cycle in range(1, self.max_cycle + 1):

            E_ref, _, F_mo = self.get_fock_and_energy_ref(C)

            E_os, t_iajb, _, _ = self.compute_os_correlation_energy_and_amplitude(C, F_mo)

            E_O2 = E_ref + self.c_os * E_os

            L4, _, grad_vo = self.compute_l4_and_lagrangian(C, F_mo, t_iajb)

            max_g = np.max(np.abs(grad_vo))
            print(f"{cycle:4d} | {E_O2:18.10f} | {E_ref:14.8f} | {E_os:12.8f} | {max_g:12.3e}")

            if max_g < self.tol_grad:
                print("="*80)
                print(f"[+] Convergence at {cycle}!")
                print(f"[+] ||L4|| Frobenius tại điểm dừng: {np.linalg.norm(L4):.6e}")
                break

            # GDM with preconditioner approach Hessian
            eps = np.diag(F_mo)

            # H_diag : (nvir, nocc) : H_{ai,ai} ~ 4 * (eps_a - eps_i)
            H_diag  = 4.0 * (eps[nocc:, None] - eps[None, :nocc])
            H_diag[np.abs(H_diag) < 0.2] = 0.2 # Damping parameter

            theta_vo = - grad_vo / H_diag

            # skew-symmetric matrix 
            kappa = np.zeros((self.nmo, self.nmo))
            kappa[nocc:, :nocc] = theta_vo
            kappa[:nocc, nocc:] = -theta_vo.T

            U = expm(kappa)
            C = np.dot(C, U)
        else:
            print("--- [!] The cycle ended before convergence ---")

        self.mo_coeff = C
        self.e_tot = E_O2
        return E_O2

if __name__ == "__main__":
    mol = gto.Mole()
    mol.atom = '''
    N   0.0000   0.0000   0.0000
    N   0.0000   0.0000   1.0977
    '''
    mol.basis = 'cc-pvdz'
    mol.build()

    solver = O2_RMP2(mol, c_os=1.2, max_cycle=30, tol_grad=1e-8)
    final_energy = solver.kernel()
    print(f"\n>> Final O2 energy: {final_energy:.10f} Hartree")
            