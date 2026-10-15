# CCSD(T) counterpoise interaction-energy scan

`intE.py` reads the same `../../inputs/aad.json` as the OBMP2 benchmark. It
uses the same R/R0 translation and ghost centers, with neutral spin=0
references for AB, A_CP and B_CP. The default `REFERENCE=uhf` selects UHF and
UCCSD(T), matching the embedding spin formalism. `REFERENCE=rhf` selects RHF
and RCCSD(T); PySCF dispatches CCSD from the supplied HF object. This is canonical CCSD(T), not CCSDT,
embedding, or a CBS extrapolation. Neither HF nor CC uses density fitting.
All electrons are correlated by default (`FROZEN=0`), matching the all-electron
embedding setup. No source pyCMF changes or pyCMF dependency are required.

For **each** system the total is explicitly:

```
E_CCSD(T) = E_HF + E_corr_CCSD + E_T
E_int = E_AB_CCSD(T) - E_A_CP_CCSD(T) - E_B_CP_CCSD(T)
```

HF ⟨S²⟩ and the reference type are saved for each system; a UHF singlet
can be spin contaminated even when SCF converges.

`mycc.kernel()` computes CCSD amplitudes. Only after `mycc.converged` is true
does the driver call `mycc.ccsd_t()` and add that correction to `mycc.e_tot`.
See the official energy convention: https://pyscf.org/user/cc.html.
The log and convergence table record HF, CCSD correlation, CCSD total,
(T) correction and final total separately. The interaction table also records
the HF, CCSD correlation and (T) contributions to E_int and checks their sum.

## Running

From the repository root, with PySCF installed:

```bash
BASIS=sto-3g R='[1.0,2.0]' MEMORY=2000 python benchmarks/dimmer_cl/methods/ccsdt/intE.py
```

Use a matching basis for comparison with the embedding calculations. The job
script defaults to def2-QZVPPD, 12 cores and 100G, as the existing dimer job.
This does not guarantee the full canonical CCSD(T) calculation fits in those
resources; large-basis triples require substantially more time and disk than
the small-basis smoke test.

```bash
mkdir -p /data/giahuy/Result/CCSDT/dimmer_cl/{_output,_error}
sbatch benchmarks/dimmer_cl/jobs/ccsdt/run.sh
# To select one scale and a different basis:
sbatch --export=ALL,BASIS=cc-pvdz,R=1.0 benchmarks/dimmer_cl/jobs/ccsdt/run.sh
```

Launcher code path: `/home/giahuy/Code/job/benchmarks/dimmer_cl/methods/ccsdt/intE.py`.
Output: `/data/giahuy/Result/CCSDT/dimmer_cl/<Slurm-job-id>/`.
Scratch: `/scratch/$SLURM_JOB_ID`. Python 3.9 and `/home/giahuy/.venv` follow
the existing submit format. Slurm logs remain `intE.out`/`intE.err` under
`_output`/`_error`; concurrent submissions share these names, as in the template.

Direct Python runs default to `results/dimmer_cl/ccsdt/<timestamp>/`.
Environment variables: `GEOMETRY_FILE` (JSON geometry; avoids the launcher INPUT_FILE code-path variable), `RUN_DIR`, `R`, `BASIS`,
`MEMORY` (MB), `OMP_NUM_THREADS`, `PYSCF_SCRATCH`, `REFERENCE`, `FROZEN` (integer occupied
orbitals frozen in each system), `HF_TOL` (1e-10), `HF_MAX_CYCLE` (200), `HF_STABILITY_CYCLES` (10),
`CC_TOL` (1e-8), `CC_NORMT` (1e-6), `CC_MAX_CYCLE` (100).
`REFERENCE` defaults to `uhf`; specify `REFERENCE=rhf` for a restricted comparison.
Set frozen core consistently across AB and CP monomers; the default 0 avoids
freezing the same number in systems with different electron counts.

## Outputs and failed calculations

- `interaction_energies.csv`: E_int and individual totals, with driver-compatible
  `E_AB_Embed_Eh`, `E_A_CP_Embed_Eh`, `E_B_CP_Embed_Eh` column names. These contain
  full CCSD(T) energies despite the inherited `Embed` name; `Method=CCSD(T)`
  distinguishes them from embedding results.
- `convergence.csv`: stage-specific status, HF/internal stability/CCSD flags,
  triples completion and separate energy components per fragment.
- `metadata.json`: method/version, settings, exact input geometry and SHA256,
  charge/spin, R definition and CP convention.
- `logs/`, `geometries/`: per-fragment PySCF log and XYZ including ghost labels.
- HF checkpoints: `<PYSCF_SCRATCH>/dimmer_cl_ccsdt/<run-dir-name>/checkpoints/`.
  Fallbacks are `TMPDIR`, then the system temporary directory. The exact file
  path is saved as `HF_checkpoint` in convergence.csv and printed in the log.
  On HPC the existing launcher chooses `/scratch/$SLURM_JOB_ID`, avoiding HDF5
  checkpoint locks on the NFS result directory. Checkpoints are temporary and
  may be purged by cluster scratch cleanup; they are not copied back to NFS.

Only converged HF, internally stable references and converged CCSD are accepted.
An internally unstable HF solution is restarted from the orbitals returned by
PySCF stability analysis, up to 10 macro cycles, like the reference stabilization
in the embedding workflow. Both initial/final HF energies and ⟨S²⟩ are retained.
External stability is not checked; the driver does not automatically change the
requested spin formalism. It does not force a broken-symmetry UHF initial guess;
UHF may converge to the same closed-shell solution as RHF. (T) is noniterative: completion and
finite energy are checked, not a fabricated triples convergence flag.
These checks do not establish accuracy of a single-reference approximation.

All three fragments are attempted even if one fails. Failed total/E_int fields
remain blank and have explicit status; never 0. CSV tables are written after
each fragment/scan point using atomic file replacement. A failed fragment makes
the scan exit nonzero after completing remaining points. A hard process kill
cannot be caught; already written checkpoints remain available. Existing
metadata prevents silently overwriting a run; this is not a resume implementation.

The supplied `references/intE_ccsdt.csv` remains UNVERIFIED. New calculations are
saved separately and do not silently replace that reference or subtract its offset.

## Local validation (PySCF 2.14.0, STO-3G, all electrons)

| R/R0 | UHF-based CCSD(T), kcal/mol | RHF-based CCSD(T), kcal/mol |
| --- | ---: | ---: |
| 1.0 | -1.467962762 | -1.698306590 |
| 2.0 | -0.084573431 | -0.084367017 |

All six fragments in each two-point scan converged and completed triples.
The UHF reference of AB at R/R0=1 required one stability macro cycle; its
final HF S² was 0.0790097. Singlet UHF is therefore not assumed equivalent to
RHF. HF stability optimization follows the same reference class, not a method
change. These are small-basis pipeline tests, not substitutes for matched
large-basis CCSD(T) benchmark energies. Large-basis resources are untested.

Additional checks passed: scalar/list R parsing, geometry equality with the
embedding driver at four R values, energy-component CP sums, blank failed
energies and skipping triples for unconverged CCSD. The final log routing was
verified with a complete restricted CP rerun at R/R0=1. The local test-only
compatibility shim handled missing /proc memory reporting and one-thread
execution; it did not change scientific kernels.

## NFS checkpoint lock fix (15 October 2026)

A live diagnosis on c3 showed PID 5304 in syscall 73 (`flock` on x86-64),
arguments fd=5 and operation=6 (`LOCK_EX | LOCK_NB`). File descriptor 5 pointed
to the HF .chk file on the NFS results filesystem. The process waited in
`rpc_wait_bit_killable` after `init E`, before the first printed SCF cycle.

The driver now writes HF checkpoints to the configured scratch directory,
keeping the normal HDF5 locking behavior. Results paths, energies, SCF/CCSD
parameters and scientific kernels are unchanged. This does not alter a running
process: use the updated driver in a new job. It does not fix NFS server/client
configuration and does not guarantee sufficient scratch for CCSD integrals.
In the observed allocation `/scratch` shared the root filesystem, with only
52 GB free; canonical TZVPD out-of-core requires a larger scratch allocation.
