import json
import time
import numpy as np       # Import numpy để tính Norm
import pandas as pd
from pyscf import gto, scf, cc

# Import hàm stabilize_scf từ module pycmf của bạn
from pycmf.OBDH.stability import stabilize_scf
from functools import reduce

def compute_ccsd_dipole(mycc, mf, mol, unit='Debye'):
    """
    PySCF không có sẵn dip_moment() cho CCSD/UCCSD object,
    nên phải tự dựng rdm1 -> AO basis -> co với tích phân lưỡng cực.
    """
    dm1 = mycc.make_rdm1()

    if isinstance(mf.mo_coeff, (tuple, list)):
        # UHF/UCCSD: dm1 = (dm1a, dm1b), mo_coeff = (mo_a, mo_b)
        mo_a, mo_b = mf.mo_coeff
        dm1a, dm1b = dm1
        dm1a_ao = reduce(np.dot, (mo_a, dm1a, mo_a.T))
        dm1b_ao = reduce(np.dot, (mo_b, dm1b, mo_b.T))
        dm1_ao = dm1a_ao + dm1b_ao
    else:
        # RHF/CCSD
        dm1_ao = reduce(np.dot, (mf.mo_coeff, dm1, mf.mo_coeff.T))

    with mol.with_common_orig((0, 0, 0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)

    el_dip = np.einsum('xij,ji->x', ao_dip, dm1_ao).real

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nucl_dip = np.einsum('i,ix->x', charges, coords)

    mol_dip = nucl_dip - el_dip
    if unit.upper() == 'DEBYE':
        mol_dip = mol_dip * 2.541746   # 1 a.u. = 2.541746 Debye

    return mol_dip

def calculate_uccsd_df_stable_dipole(json_filepath, output_excel="dipole_results.xlsx"):
    # Đọc dữ liệu từ file JSON
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # Khởi tạo list để lưu trữ kết quả cho Excel
    results = []

    # Lặp qua từng phân tử trong file
    for mol_name, properties in data.items():
        print(f"\n{'='*60}")
        print(f"Đang xử lý phân tử: {mol_name}")
        
        # 1. Chuyển đổi định dạng hình học (geometry) cho PySCF
        pyscf_geom = []
        for atom in properties['geometry']:
            atom_symbol = atom[0]
            coordinates = tuple(atom[1])
            pyscf_geom.append((atom_symbol, coordinates))
            
        try:
            start_time = time.time()
            
            # 2. Khởi tạo đối tượng phân tử (Molecule)
            mol = gto.Mole()
            mol.atom = pyscf_geom
            mol.charge = properties['charge']
            mol.spin = properties['spin'] 
            
            # Khai báo basis set hỗn hợp
            mol.basis = {
                'default': 'aug-cc-pcvqz',
                'H': 'aug-cc-pvqz'
            }
            mol.build()
            
            # 3. Chạy Unrestricted Hartree-Fock (UHF) với Density Fitting
            print("  -> Đang chạy UHF (Density Fitting)...")
            mf = scf.UHF(mol).density_fit(auxbasis='def2-universal-jkfit') 
            mf.verbose = 0
            mf.kernel()
            
            # 4. KIỂM TRA VÀ ỔN ĐỊNH HÓA SCF (Tránh điểm yên ngựa)
            print("  -> Đang kiểm tra tính ổn định (Stability Check)...")
            mf = stabilize_scf(mf, max_macro_cycles=10, verbose=True)
            
            # Lấy năng lượng UHF sau khi đã ổn định
            uhf_energy = mf.e_tot
            
            # 5. Chạy UCCSD dựa trên hàm sóng đã được ổn định
            print("  -> Đang chạy UCCSD...")
            mycc = cc.CCSD(mf)
            mycc.verbose = 0
            uccsd_energy = mycc.kernel()[0]
            
            # 6. Tính toán Dipole Moment
            print("  -> Đang tính Dipole Moment...")
            dipole = compute_ccsd_dipole(mycc, mf, mol) 
            
            # ---> TÍNH NORM CỦA DIPOLE ĐỂ LƯU THÀNH 1 SỐ DUY NHẤT <---
            if dipole is not None:
                # np.linalg.norm tính độ dài của vector [x, y, z]
                dipole_norm = np.linalg.norm(dipole)
            else:
                dipole_norm = None
                
            # Thêm kết quả vào danh sách (chỉ lưu Tên chất và số Dipole Norm)
            results.append({
                "Tên chất": mol_name,
                "Dipole (Debye)": dipole_norm
            })
            
            print(f"\n[KẾT QUẢ {mol_name}]")
            print(f" - UHF Energy     : {uhf_energy:.6f} Hartree")
            print(f" - UCCSD Energy   : {uccsd_energy:.6f} Hartree")
            if dipole is not None:
                print(f" - Dipole Vector  : [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
                print(f" - Dipole Norm    : {dipole_norm:.4f} Debye")
            print(f" - Thời gian chạy : {time.time() - start_time:.2f} giây")
            
        except Exception as e:
            print(f"\n[LỖI] Không thể tính toán cho {mol_name}. Chi tiết: {e}")
            results.append({
                "Tên chất": mol_name,
                "Dipole (Debye)": "Lỗi/Không tính được"
            })
            
        
    # 7. XUẤT DỮ LIỆU RA FILE EXCEL
    if len(results) > 0:
        df = pd.DataFrame(results)
        # Lưu Excel, giữ giá trị float để dễ xử lý trong Excel
        df.to_excel(output_excel, index=False)
        print(f"\n{'='*60}")
        print(f" ĐÃ LƯU THÀNH CÔNG KẾT QUẢ VÀO FILE: {output_excel}")
        print(f"{'='*60}")

if __name__ == "__main__":
    _input = "/home/giahuy/Code/job/geometry/dipole_152/sp_inputs.json"
    _output = "/data/giahuy/Result/CCSD/dipole_moments/dipole152_sp.xlsx"
    calculate_uccsd_df_stable_dipole(_input, _output)