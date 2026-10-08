# C12 conjugated aldehyde embedding scan

Geometry: `inputs/aldehyde.xyz` (user-supplied C12H14O, Angstrom).
Both drivers read this file by default, independently of the working directory.
Required order: O, C, H (CHO); ten C,H pairs; terminal C,H,H,H.
The element order is validated; preserve atom identity when optimizing geometry.

Active prefixes contain 3, 7, 11, 15, 19, 23, 27 atoms:
CHO, then one through five CH=CH units, then terminal CH3.
All calculations retain the complete molecular geometry and basis.
No caps are added. The final point is all-active embedding, not a separately
computed non-embedded reference.

## Run from the repository root on the cluster

```bash
sbatch benchmarks/aldehyde_cl/jobs/mp2/input.sh
sbatch benchmarks/aldehyde_cl/jobs/obdh/input.sh
```

Job defaults: cc-pVDZ, No CL (`--shells 0`), seven regions.
Jobs use the existing cluster installation at `/home/giahuy/Code/job`.
MP2 retains HF-based MP2-in-B3LYP; OBDH retains alpha=(0.5,0.4)
and the upstream default environment. These are inherited method settings,
not a matched DH/OBDH comparison. The computational kernels are unchanged.

To run all four virtual-space settings (28 points), use a separate output:

```bash
sbatch benchmarks/aldehyde_cl/jobs/mp2/input.sh --shells 0 1 2 3 --output /data/giahuy/Result/MP2_in_DFT/bench_conv_C12_aldehyde/mp2_aldehyde_all_shells.json
sbatch benchmarks/aldehyde_cl/jobs/obdh/input.sh --shells 0 1 2 3 --output /data/giahuy/Result/OBDH_in_DFT/bench_conv_C12_aldehyde/obdh_aldehyde_all_shells.json
```

Use `--resume` only with the same configuration and output path.
Python defaults to all four settings when `--shells` is omitted.
Values 1,2,3 correspond to the paper's 0th,1st,2nd CL shells.
The JSON output, metadata sidecar and point logs use aldehyde-specific names.
Slurm stdout/stderr are created in the submission directory.

## Validate inputs without PySCF calculations

```bash
python benchmarks/aldehyde_cl/methods/mp2/mp2.py --basis cc-pvdz --shells 0 --dry-run
python benchmarks/aldehyde_cl/methods/obdh/obdh.py --basis cc-pvdz --alpha 0.5 0.4 --shells 0 --dry-run
```

Actual calculations require PySCF and pycmf. MP2's inherited `completed`
status means a finite kernel result; check internal SCF convergence in logs.
