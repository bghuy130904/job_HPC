# Acetic acid dimer — CL embedding and CP interaction energy

`inputs/aad.json` contains both monomer geometries (Angstrom), R/R0 scan points
and active atom indices. The driver translates monomer B along the original
geometric-center separation. R is a dimensionless scale, not a distance in Å.

One driver, `methods/obmp2/intE.py`, supports `METHOD=mp2` and `METHOD=obmp2`.
MP2 imports `shared/mp2_embed.py` using `sys.path.insert`; OBMP2 uses the installed
`pycmf.OBDH.OBMP2_CL`. No embedding implementation is copied into this benchmark.
Both use the same geometry, ghost basis, active atom indices, B3LYP, mu=1e6,
and CL with two shells. CP subtraction uses the kernel's total embedding energy.
Each kernel is called once. Failed systems are recorded individually and leave
the interaction energy blank rather than zero. All three systems are attempted.

## Run from the repository root

```bash
METHOD=obmp2 BASIS=sto-3g R='[0.95,1.0,1.05,1.25,2.0]' python benchmarks/dimmer_cl/methods/obmp2/intE.py
METHOD=mp2 BASIS=sto-3g R='1.0' python benchmarks/dimmer_cl/methods/obmp2/intE.py
mkdir -p /data/giahuy/Result/OBMP2_in_DFT/dimmer_cl/{_output,_error}
sbatch --export=ALL,METHOD=obmp2 benchmarks/dimmer_cl/jobs/obmp2/run.sh
sbatch --export=ALL,METHOD=mp2 benchmarks/dimmer_cl/jobs/obmp2/run.sh
```

Installed dependencies: PySCF, pyCMF, NumPy, pandas; openpyxl for Excel output.
Set `PYCMF_PATH` to the directory containing package pycmf if it is not installed.
Run paths default to `results/dimmer_cl/<method>/<timestamp-or-Slurm-job-id>/`.
Override `RUN_DIR`, `BASIS`, `MEMORY`, `R`, `INPUT_FILE`, `OBMP2_NITER` or `DIIS_SPACE`
through environment variables. Default basis is def2-QZVPPD; OBMP2_NITER=1000.
Do not reuse a run directory for a different calculation.

The submission script follows `hexanol_cl/jobs/obdh/input.sh`: code under
`/home/giahuy/Code/job`, Python 3.9 and `/home/giahuy/.venv`, scratch under
`/scratch/$SLURM_JOB_ID`, and results directly under `/data/giahuy/Result`.
Slurm stdout/stderr go to `dimmer_cl/_output` and `dimmer_cl/_error`; create these
directories before submitting, as shown above. Calculation tables and logs go to
`/data/giahuy/Result/OBMP2_in_DFT/dimmer_cl/<method>/<Slurm-job-id>/`, with the driver output
in `intE_dimmer_cl.txt`. Slurm stdout/stderr filenames are `intE.out` and `intE.err`,
as in the latest user-edited job file. Direct Python runs retain the repository results default.
No HPC job is submitted by adding these files to the repository.

Collect run tables with:

```bash
python benchmarks/dimmer_cl/tools/collect.py results/dimmer_cl/mp2/RUN_ID results/dimmer_cl/obmp2/RUN_ID --output results/dimmer_cl/comparison.csv
```

SPADE partitioning and CL are recomputed for each system. The same active atom
indices include ghost AO centers in CP monomers and do not imply identical MO
dimensions. Internal backend SCF failures are checked from their logs when flags
are unavailable. This organization does not change the pyCMF kernels or their
convergence criteria and does not establish convergence in the large basis.

Validation: both complete CP calculations at R/R0=1.0 with STO-3G converged
(AB and both ghost monomers). E_int: MP2-in-B3LYP = -3.253033 kcal/mol;
OBMP2-in-B3LYP = -4.374281 kcal/mol, matching the prior driver tests within
1e-8 Hartree. Shared imports, Python compilation and Bash syntax were checked.

## Reference and change record

`references/intE_ccsdt.csv` normalizes the supplied CCSD(T) table to the driver
column names without changing values; `references/raw/intE_ccsdt.csv` is the
original. The reference is marked UNVERIFIED: basis/convergence provenance is
missing and the large interaction-energy offset needs checking. See
`references/README.md` before using it for error statistics.

`methods/obmp2/intE_changes_explained.md` sits beside `intE.py` and documents
the original-to-current driver changes with code snippets and a complete diff.
