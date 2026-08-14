import time
from pyscf import scf, gto
from mp2_embed import UMP2_CL

mol = gto.Mole()
mol.atom = '''
O    0.000000   0.000000   0.000000
H    0.000000   0.000000   0.962000
C    1.430000   0.000000   0.000000
H    1.790000   1.026000   0.000000
H    1.790000  -0.513000   0.889000
C    1.943000  -0.513000  -1.334000
H    1.583000   0.051000  -2.200000
H    1.583000  -1.540000  -1.334000
C    3.463000  -0.513000  -1.334000
H    3.823000   0.051000  -0.468000
H    3.823000  -1.540000  -1.334000
C    3.976000  -0.000000  -2.668000
H    3.616000   1.027000  -2.668000
H    3.616000  -0.513000  -3.534000
C    5.496000  -0.000000  -2.668000
H    5.856000  -1.027000  -2.668000
H    5.856000   0.513000  -1.802000
C    6.009000   0.513000  -4.002000
H    5.649000   1.540000  -4.002000
H    5.649000   0.000000  -4.868000
H    7.099000   0.513000  -4.002000                         
'''
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.basis = 'cc-pvdz'
mol.max_memory = 4000
mol.build()

print("\n\n>>>>>>>> CHAY CHE DO: MP2-in-DFT (EMBEDDING & TRUNCATION) <<<<<<<<")
mf_emb = scf.UHF(mol).density_fit()
mf_emb.max_memory = 4000
mf_emb.with_df.max_memory = 4000
mf_emb.run()

mppp_emb = UMP2_CL(mf_emb)
mppp_emb.xc_env = '0.53*HF + 0.47*B88, 0.61*LYP'       # functional mo ta moi truong B (khong co alphaa: MP2 khong phai double hybrid)
mppp_emb.use_embed = True      # Bat Embedding
mppp_emb.active_atoms = list(range(8))
mppp_emb.mu = 1e6
mppp_emb.use_cl = True         # Bat CL Truncation (chi co y nghia khi use_embed=True)
mppp_emb.n_shells = 2
mppp_emb.frozen = None         # None = tuong quan toan bo electron

start2 = time.time()
mppp_emb.run()
# print('=> Thoi gian chay (Embed + CL): ', time.time() - start2)