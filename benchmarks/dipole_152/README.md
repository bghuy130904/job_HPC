# Dipole 152 — shared production protocol

`methods/{obdh,dft,ccsd}/calc_dipmomt_*.py` are thin entry points, like
`benchmarks/sie4x4/methods/*`. All calculation logic lives in
`tools/run_dipole.py`; reporting lives in `tools/report_dipole.py`.
Existing `inputs/`, `references/` and historical results are preserved.
Use a fresh output directory for the v2 protocol. Old CSV-only runs are not
silently treated as v2 checkpoints with verified convergence.

## What was checked against SIE and the recent benchmarks

- SIE production `tools/run_sie.py`: explicit CLI parameters, SCF candidate
  records, finite/converged validity, source/input hashes, per-case JSON and
  full logs, machine-readable CSVs, a separate Excel report, and thin method
  entry points. Its production protocol samples several SCF/OB basins.
- SIE `run_sie_standard.py`: one standard reference per WFT method group,
  independent UKS calculations for DFT/DH, reject an unconverged reference
  before correlation, and pass basis/alpha/cycle limits through CLI.
- SIE jobs: `/home` code, `/data` output, `JOB_SCRATCH_PATH`, `TMPDIR`, module
  and venv, thread environment, explicit CLI and a driver `.txt` log.
- SIE production currently uses `config.json`, per-case `.json`/`.log`, CSV
  and XLSX; it does **not** use `.meta.json`. The versioned metadata/resume
  convention comes from the recent aldehyde/torsional benchmarks. This
  benchmark now provides both `config.json` and `dipole.meta.json`.
- Confirmed the remote workspace/benchmark-layout head before editing:
  `319b9fff7ec47227e80de4b602c7e3c262a010a0`.

The dipole protocol intentionally does not copy SIE's fragment-localized
seeds or ground-state ranking of multiple OB solutions. These are different
chemical problems. Here all WFT/OB methods start from the same selected UHF
reference. A multiguess search is finite, not proof of the global minimum.

## Reference selection and validity

Choose explicitly with `--scf-policy`:

| Policy | Zero-field calculation | Selection |
|---|---|---|
| `standard` (explicit diagnostic option) | One PySCF initial guess, configured limits; no fallback/stability | Accept only a converged, finite reference with valid electron count and gradient |
| `multiguess` (benchmark default) | For **each** `--guesses` seed: SCF, Newton fallback if needed, internal stability check, follow unstable orbitals and reconverge/recheck | Lowest admissible **SCF energy** for that reference Hamiltonian |

No selection by dipole, reference error, S2, or a localization/symmetry label.
No lowest-energy fallback when every seed is invalid. Full candidates and the
selected flag are saved. The benchmark default is
`--scf-policy multiguess`, including stability for every guess. Use `standard`
explicitly when testing the plain pipeline. A one-guess stability test uses
`--scf-policy multiguess --guesses minao`; there is no separate public
`stable` switch.
The job scripts expose this as `SCF_POLICY` or a trailing CLI option.

A valid SCF requires both energy and gradient convergence. SCF stability is
checked only after convergence. Stability of an HF/KS reference is not a
stability test of the final OB solution. Core pyCMF equations are unchanged;
its convergence flag and final printed effective F_ia are recorded.

The default SCF gradient tolerance is `--scf-grad-tol 1e-4`, matching the
acceptance threshold in the earlier dipole pipeline. Newton retains PySCF's
default AH settings. Override the gradient tolerance explicitly if needed.
If no admissible reference exists, the reference is `reference_failed` and
post-SCF methods are `not_run_reference_failed`; they were not attempted.
The merge also recognizes this cause in older generic-error checkpoints.

After SCF, MP2, CC or OB runs only on the selected reference. Unconverged OB,
CC, Lambda or field calculations remain in JSON/logs but cannot populate an
accepted dipole. Natural occupation violations are recorded, not removed from
benchmark statistics by a hidden gate. AO electron count, Hermiticity and
finite values are separate checks.

## Dipole definitions

| Estimator | Meaning |
|---|---|
| `analytic` | UHF/KS determinant expectation value |
| `determinant` | Reference determinant, or optimized OB determinant |
| `t2` | Determinant + standard oo/vv T2 density correction |
| `scaled_t2` | Determinant + alpha_c times the T2 correction; an explicit OBDH/DH prescription |
| `cc_lambda` | CC Lambda density without the SCF orbital-response Z-vector |
| `finite_field` | Actual central differences of energies at ±F; nuclear contribution added once |

`--dipole det`: cheap determinant/analytic only.
`--dipole dm` (alias `unrel`): also T2/scaled-T2/CC-Lambda densities.
`--dipole ff`: determinants plus actual field calculations, no costly CC Lambda.
`--dipole both`: both density and field estimates, including CCSD(T) Lambda.
For CCSD(T), `both` can be substantially more expensive than `ff`.

No column labelled finite_field is filled from an analytic density. Full signed
vectors are retained, in addition to magnitudes. All energies are Hartree,
fields atomic units, dipoles Debye and coordinates Angstrom. Field integrals
and nuclear dipoles use origin (0,0,0) Bohr. For charged species this origin is
part of the definition.

Field calculations continue from the selected zero-field SCF density; OB can
use zero-field OB orbitals (`--field-seed ob`) or the UHF field orbitals (`hf`).
There is no independent multiguess/stability search in a field. Field SCF and
post-SCF convergence are required. An occupied-subspace singular-value check
(`--overlap-min`) detects branch changes; at M_s=0 it permits a global alpha /
beta exchange. This is a numerical continuity check, not a proof of electronic
state identity. No overlap check is replaced by an energy-curvature threshold.

Use several `--fields` to check signed-vector step sensitivity. If all field
points converge but vectors differ by more than `--ff-tol`, accepted FF values
stay blank, while raw vectors/energies remain inspectable. With one step,
`step_check=not_tested_single_step` is explicit. Curvature is a warning only:
physical polarizability also contributes to it, and cancellation may hide a
branch switch. There is no automatic choice of the best field or best density
based on reference error.

## Parameters and scientific comparability

All basis, alpha, SCF/OB/CC limits and tolerances, grid, fitting choices and
field strengths are CLI parameters. Defaults live in the shared parser, not
in separate calculation implementations.

- `--alpha ax ac` controls **OBDH** and **dh_matched**. Named B2PLYP remains
  (0.53,0.27) by definition. For another pair use `dh_matched`, not a renamed
  B2PLYP. Actual coefficients/formalism are recorded in the run configuration.
- All present methods use **UHF/UKS**. This is not a restricted-DH reproduction
  of the equilibrium DH numbers in Hait/Head-Gordon 2018.
- `--basis aug-cc-pCVQZ` maps H to aug-cc-pVQZ, as in the old inputs. Other
  core-valence cardinalities follow the same mapping. Some PySCF installs
  need basis-set-exchange for augmented core-valence basis data.
- `--scf-aux none` means exact SCF. `auto` or a named JK basis enables SCF DF.
  `--mp2-aux auto` uses `make_auxbasis(mp2fit=True)` independently of SCF DF;
  a JK fitting basis is never silently inherited for OB correlation.
- One-shot UMP2/DH use DF by default; `--no-pt2-df` disables that fitting.
  OBMP2_CL/OBDH_CL are intrinsically DF in the current core. CCSD uses the
  ordinary UCCSD integral path. These algorithm choices are recorded.
- Frozen core is configurable for MP2/CC. The present OB core requires
  `--frozen 0`; a nonzero value with OB is rejected rather than silently ignored.
- DH PT2 keeps canonical KS eigenvalues and orbitals; converting the carrier
  to HF does not run HF SCF or trigger noncanonical MP2 orbital iterations.
- DM T2 and finite-field OB results are different prescriptions in current
  source. This pipeline does not assert their mathematical equivalence or
  alter the core to make the dipoles agree.

## Output contract

| Path below output directory | Purpose |
|---|---|
| `config.json` | Initial run configuration, readable like SIE |
| `dipole.meta.json` | Authoritative fingerprint, input geometries, requested indices, case status, source hashes and invocation resources |
| `points/000_AlH2.json` | One complete molecule checkpoint: zero-field methods, field points, candidates and all diagnostics |
| `logs/000_AlH2.log` | Full native SCF/CC/OB log, with explicit method/axis/field markers; retries append an attempt marker |
| `dipole_<array-index>.txt` | Job launcher/driver output; no grep-filtered log loss |

| `summary.txt` | Merge summary, completeness and statistics |
| `Results.csv`, `Statistics.csv`, etc. | Separate narrow tables, same data as workbook sheets |
| `dipole.workbook.json` | Neutral workbook layout/data/formulas for reproducible review |
| `dipole.xlsx` | Formatted workbook produced during serial completion or explicit merge |

Live monitoring: the driver `.txt` prints startup, metadata lock/write stages
and the exact per-molecule log path. Workers run with Python `-u`; the native
SCF/OB/CC iterations go to `logs/<index>_<name>.log` while calculation is in
progress. Follow that file with `tail -f`. Locks use atomic directory creation,
not advisory `flock`, which can hang in filesystem lock RPCs on some HPC mounts
even with `LOCK_NB`. Metadata contention is reported every 10 seconds and times
out after 60 seconds. `.metadata.lock.d/owner.json` records host, PID and SLURM
job/task. Per-case `.lock.d` directories reject duplicate workers. Locks are
removed on normal/error exit; hard-killed jobs can leave a directory behind.
Never remove it until its owner job has stopped; locks are not stolen by age.
Stop all old array tasks before switching from the older flock implementation:
the two locking protocols do not coordinate. The old empty `.metadata.lock`
file is unused by the new protocol. A source hash change requires a fresh
output directory if the old directory already has metadata/checkpoints.
`points/*.json` is written
after the whole molecule finishes, not after every SCF/OB iteration.

Resume accepts only matching physical parameters, input and core/driver hashes.
`--only`, array index, output path, report destination, threads and memory are
not physical settings. Changing resources does not invalidate completed
energies; each invocation and actual case resources are recorded. Changing
alpha/basis/grid/tolerances/methods does invalidate the checkpoint. Resume
skips only complete cases and retries failed cases, keeping the previous log.
It currently retries the entire failed molecule, not individual field points.

SLURM array tasks lock metadata updates and write distinct case files. They
never regenerate a shared Excel file concurrently. Run a single merge after
the array. Workers run in separate Python processes so one molecule's large
amplitudes/integrals do not accumulate across the dataset. A killed worker
produces an explicit failed checkpoint without invented energies. Field tensors
are released point-by-point; CCSD and CCSD(T) share amplitudes only at the
current point, rather than retaining every field ERI/amplitude object. A
T2/Lambda density construction error is recorded separately and does not
erase independently converged energy/finite-field results.

## Excel layout

One workbook per output directory / method job. No hundreds-of-columns wide
summary and no merged method labels inside sortable data tables.

| Sheet | Content |
|---|---|
| Results | Molecule, method, estimator, field, accepted magnitude/reference/error, validity/status |
| Statistics | Requested/converged/reference-paired counts, regularized RMSE and MAE; complete RMSE stays blank for partial data |
| Energies | Accepted energy separately from the last iterate energy |
| Dipole_vectors | Signed components and raw magnitude, including invalid estimates |
| Convergence | Zero/field SCF and post-SCF convergence, gradients, OB iteration diagnostics and errors |
| SCF_candidates | Guess, energy, convergence/stability, S2 and selected flag |
| Field_points | Raw E(+F)/E(-F), validity and subspace overlaps |
| Density_checks | Trace/electron count, Hermiticity and natural occupation range |
| Molecules | Charge, spin, atom/AO counts and source checkpoint |
| Config | Parameters, source/reference hashes, estimator definitions and units |

Numeric values remain numeric. Real zero dipoles remain zero; missing or
invalid results stay blank. Error and aggregate formulas are inspectable.
Headers/identifier columns are frozen, filtering is enabled, number precision
is explicit, gridlines are hidden, and the layout is neutral/monochrome.

`n_converged` counts accepted dipoles. `n_paired` also requires a reference.
Regularized error is 100*(mu-mu_ref)/max(mu_ref,1 D), matching 2018. It is not
the purely relative RMAD used in Table 2 of the 2022 paper. Common-cohort
statistics use the intersection across all requested method/estimator/field
groups and are a merge-time snapshot; rerun merge after changing source data.

The cluster exporter uses the existing SIE dependency, openpyxl, and requests
Excel recalculation on open. Optional `export_artifact_xlsx.mjs` consumes the
same plan for Work Mode review; it is not required on HPC.

## Commands

Run from the root of job_HPC; pycmf must be installed for OB methods.

```bash
python -u benchmarks/dipole_152/methods/obdh/calc_dipmomt_obdh.py \
  --input benchmarks/dipole_152/inputs/sp_inputs.json \
  --ref-json benchmarks/dipole_152/references/ref_sp.json \
  --outdir /data/giahuy/Result/OBDH/dipole_moments/sp_v2 \
  --methods uhf ump2 obmp2 obdh --basis aug-cc-pCVTZ --alpha 0.53 0.27 \
  --scf-policy multiguess --scf-cycles 200 --ob-cycles 300 \
  --scf-tol 1e-12 --scf-grad-tol 1e-4 --ob-tol 1e-10 \
  --dipole both --fields 0.0002 0.0001 0.00005 --grid 4 \
  --threads 12 --max-memory 40000
```

For a plain route use `--scf-policy standard`. For a quick diagnosis add
`--only CN BO NF NO`. Resubmitting the same physical run adds `--resume`.
Every old method path is an entry point, so `--funcs` and `--levels` remain
aliases for DFT and CC method lists. Custom coefficients belong in
`--methods dh_matched`.

The production job templates pass explicit parameters, create scratch, preserve
exit codes, and put trailing `"$@"` last so CLI overrides are not overwritten.
Environment defaults: BASIS, ALPHA_X, ALPHA_C, SCF_POLICY, DIPOLE_MODE,
OUTPUT_DIR and CODE_ROOT. Multiguess **with stability on every guess** is the template default.
Set SCF_POLICY=standard only for the no-wrapper comparison.

```bash
bash benchmarks/dipole_152/jobs/obdh/dipole_sp.sh --submit
bash benchmarks/dipole_152/jobs/obdh/dipole_sp.sh --merge
```

For OB SP scaling, edit only `ALPHA=(0.50 0.40)` near the top of
`jobs/obdh/dipole_sp.sh`. The pair sets Python `--alpha`, the output directory
`sp_scaling/ax_0.50_ac_0.40`, and the submitted job name
`wftdipolesp_ax_0.50_ac_0.40`. `--submit` calls sbatch with the computed name
and creates scheduler log directories first; shell variables in `#SBATCH`
directives cannot do this. `--merge` uses the same pair/output directory.
Use these Bash commands from the login node. Direct `sbatch dipole_sp.sh`
also calculates the output path, but uses the generic SBATCH job name.
`ALPHA_X`/`ALPHA_C` environment overrides remain available. Decimal strings
are preserved without rounding (including values such as 0.125); use a
consistent spelling for folder names. An additional Python `--alpha` argument
is rejected so that folder/job names cannot silently describe a different pair.
The submitted environment freezes the pair and output directory. Changing the
file for the next submission does not alter an already submitted job.
For merging an older pair after editing ALPHA, set OUTPUT_DIR explicitly or
use the existing `merge_sp.sh`. These changes apply only to the OB SP job.

Merge only after the array completes, or submit it with an `afterany` dependency
to inspect incomplete jobs. Merge script names are `merge_sp.sh` and
`merge_nsp.sh` for all families; CC's old `merge_dipole.sh` is the SP alias.
Scheduler `_output` and `_error` directories must exist before submission;
creating them inside a job is too late for SLURM's output-file opening.
No job was submitted on the user's HPC during local verification.

## Validation and scope

`tests/test_protocol.py` checks field sign/nuclear contribution, failed-field
rejection, step sensitivity, overlap-metric occupations, reference selection,
missing/real-zero data and statistics. Small numerical tests cover ordinary
and DF MP2, OBMP2/OBDH, DFT/DH, CCSD/CCSD(T) energies/densities/fields; array
metadata and resume are tested separately. These are implementation tests,
not a new 152-molecule accuracy benchmark.

`tools/plain_uhf_obmp2.py` retains the plain-default reproduction experiment.
`tools/audit_density.py` is an earlier diagnostic runner; its historical helper
is explicitly under `archive/dipole_152/legacy_pipeline`, not imported by the
new production engine. The core pyCMF source and the SIE benchmark are not
modified by this refactor.
