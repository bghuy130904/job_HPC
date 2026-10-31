# Dipole CCCBDB — twenty experimental geometries

Dipoles of twenty small, neutral, ground-state singlets at the Cartesian
experimental geometries published by [NIST CCCBDB](https://cccbdb.nist.gov/expgeom1x.asp).
Coordinates are copied at the displayed precision, without optimization.
All distances are Angstrom; reference dipole magnitudes are Debye.

## Dataset and references

The JSON insertion order is the SLURM array order. Every species has charge 0,
PySCF spin 0 (= N_alpha - N_beta) and multiplicity 1.

| Array index | Molecule | CAS | Reference / D | Character in main table |
|---|---|---|---:|---|
| 0 | HF | 7664-39-3 | 1.827 | mu_0 |
| 1 | HCl | 7647-01-0 | 1.093 | mu_e |
| 2 | CO | 630-08-0 | 0.112 | MW; vibrational character unspecified |
| 3 | HCN | 74-90-8 | 2.980 | mu_0 |
| 4 | H2O | 7732-18-5 | 1.857 | mu_e |
| 5 | NH3 | 7664-41-7 | 1.476 | mu_0 |
| 6 | H2CO | 50-00-0 | 2.332 | mu_0 |
| 7 | H2S | 7783-06-4 | 0.977 | mu_0 |
| 8 | SO2 | 7446-09-5 | 1.633 | mu_0 |
| 9 | OCS | 463-58-1 | 0.715 | mu_0 |
| 10 | CH3F | 593-53-3 | 1.847 | mu_0 |
| 11 | CH3Cl | 74-87-3 | 1.870 | vibrational character unspecified |
| 12 | CH2F2 | 75-10-5 | 1.970 | vibrational character unspecified |
| 13 | CH3OH | 67-56-1 | 1.672 | mu_0 |
| 14 | CH2Cl2 | 75-09-2 | 1.620 | mu_0 |
| 15 | H2O2 | 7722-84-1 | 1.770 | mu_e |
| 16 | N2O | 10024-97-2 | 0.161 | mu_0 |
| 17 | HNO | 14332-28-6 | 1.670 | mu_0 |
| 18 | H2CS | 865-36-1 | 1.647 | vibrational character unspecified |
| 19 | PH3 | 7803-51-2 | 0.580 | vibrational character unspecified |

Selection is fixed to the **total magnitude in the main Electric dipole
moment table, state=1/config=1**, on each neutral species page. The values
are experimental references, not the theoretical references of dipole_152.
For example, the CO excited-triplet value 1.375 D is excluded. HCl's separate
v=0 row is 1.109 D; HCN's vibration-specific table lists 2.985 D; SO2's
comment gives mu_e=1.62673 D. These do not replace the selected main rows.
`references/cccbdb_sources.json` records each geometry URL, dipole URL,
bibliographic identifier, measurement comments, and the access date.

The selection mixes mu_e and mu_0 because the source main table does.
A fixed-geometry electronic calculation does not include vibrational
averaging. Its difference from these experimental numbers therefore includes
more than electronic-method error. No vibrational correction is applied.
The published Cartesian coordinates can also differ slightly from the
rounded internal-coordinate table; the Cartesian table defines this input.
Signed calculated vectors are retained, but reference errors compare magnitudes.

The ten additions (indices 10–19) were read on 2026-10-31 from the
CCCBDB **One molecule all properties** pages, which contain both the
Cartesian table and the main dipole table. The original ten entries and
their indices are unchanged. Internal-coordinate citations for the additions
are stored separately; the Cartesian tables do not identify a separate citation.

CH3CN was considered but excluded because its displayed H coordinates do
not reproduce the stated C3v symmetry; CH2Cl2 was selected instead.
CH3F's Cartesian C–F distance is 1.3646 Å, whereas its internal table lists
1.383 Å. H2O2's geometry has a 119.8° dihedral, whereas the dipole comment
uses an equilibrium angle of 111.5°. These source discrepancies are
recorded, not silently corrected. For H2O2 only the C2 minimum
(state=1/config=1) is selected; the non-minimum C2h row is excluded.

## Repository pipeline checked before adding this benchmark

The current `workspace/benchmark-layout` layout separates shared geometry,
reference data, calculation entry points, launchers and analysis tools.

| Location | Role and existing examples |
|---|---|
| `inputs/` | Shared JSON/XYZ: dipole_152, SIE4x4, torsional_pes, dimmer_cl |
| `references/` | Shared reference values: dipole_152 JSON; dimmer_cl CSV |
| `methods/<method>/` | Calculation entry points; current dipole and SIE scripts delegate to a common engine |
| `jobs/<method>/` | SLURM submission and merge scripts; production paths inferred from dipole_152 and SIE4x4 |
| `tools/` | Common orchestration/reporting and benchmark utilities |
| `results/<benchmark>/` | Repo-side historical/validation results; HPC production results go to the `/data/giahuy/Result` tree |

Not every older benchmark has every directory, and some legacy launchers
still contain older absolute paths. This benchmark follows the current
dipole_152 job convention: `/home/giahuy/Code/job` for source,
`/data/giahuy/Result/<family>/...` for output, `/scratch/<job>/<task>` for
temporary files, `python3.9`, `/home/giahuy/.venv`, explicit thread settings,
Python `-u`, a driver `.txt`, native molecule logs and separate array merge.
SIE4x4 uses `config.json` and point files; the versioned `.meta.json` contract
here comes from the dipole engine, not from all benchmarks universally.

## Files in this benchmark

| Path | Purpose |
|---|---|
| `inputs/cccbdb_inputs.json` | Twenty geometries, charge, spin, multiplicity, units, CAS and geometry source |
| `references/ref_cccbdb.json` | Flat `{molecule: dipole_D}` map consumed by reporting, matching dipole_152 |
| `references/cccbdb_sources.json` | Source and measurement details, separate from the numeric reference map |
| `methods/obdh/calc_dipmomt_obdh.py` | UHF, UMP2, OBMP2, OBDH entry point |
| `methods/dft/calc_dipmomt_dft.py` | PBE0, B3LYP, B2PLYP entry point; other engine methods available through CLI |
| `methods/ccsd/calc_dipmomt_ccsd.py` | CCSD, CCSD(T) entry point |
| `tools/cccbdb_driver.py` | Supplies local input/reference defaults and the plain `standard` SCF policy |
| `jobs/{obdh,dft,ccsd}/job.sh` | Array submission (`0-19%3`) and merge, with the established SLURM format |
| `jobs/{obdh,dft,ccsd}/merge.sh` | Alias for the same job script's `--merge`, so alpha/output defaults agree |
| `tests/test_pipeline.py` | Dataset consistency, dataset CLI defaults and mock submission checks |

Calculation and report implementations are reused directly from
`../dipole_152/tools/run_dipole.py` and `report_dipole.py`. There is no copied
SCF, T2-density, finite-field or Excel implementation. Keep both benchmarks
when copying the repository to HPC. No existing benchmark or pyCMF equations
are changed.

The default here is the plain UHF/UKS route (`--scf-policy standard`), with
one initial guess and no multiguess/stability wrapper. Choose `multiguess`
explicitly to request the engine's other policy. All formalisms remain
unrestricted, including these singlets, as in the existing dipole engine.

## Submission and paths

From `/home/giahuy/Code/job`:

```bash
bash benchmarks/dipole_cccbdb/jobs/obdh/job.sh --submit
bash benchmarks/dipole_cccbdb/jobs/dft/job.sh --submit
bash benchmarks/dipole_cccbdb/jobs/ccsd/job.sh --submit
```

All three jobs use Bigmem, 12 CPUs/task, 50G, 48 hours and up to three
simultaneous molecules. The templates retain the dipole_152 basis default
`aug-cc-pCVQZ`; H maps to `aug-cc-pVQZ` in the existing engine. `BASIS` and
trailing Python options can override settings. `MAX_MEMORY` defaults to
47000 MB. This is a configured limit, not a guarantee about peak memory.

| Family | Methods | Default dipole mode | Default output directory |
|---|---|---|---|
| OB | uhf, ump2, obmp2, obdh | both | `/data/giahuy/Result/OBDH/dipole_cccbdb/scaling_v2/ax_0.50_ac_0.40` |
| DFT | pbe0, b3lyp, b2plyp | dm | `/data/giahuy/Result/DFT/dipole_cccbdb/v2` |
| CC | ccsd, ccsd_t | ff | `/data/giahuy/Result/CCSD/dipole_cccbdb/v2` |

For OB scaling, edit only `ALPHA=(0.50 0.40)` near the top of the OB `job.sh`.
The same pair determines `--alpha`, the output folder and submitted job name.
Environment overrides `ALPHA_X`/`ALPHA_C` are also supported. The submission
freezes the pair/output in the job environment. Use the same decimal spelling
when returning to a previous folder. B2PLYP remains fixed at (0.53,0.27);
choose `--methods dh_matched` for a DFT double hybrid with custom coefficients.

```bash
ALPHA_X=0.60 ALPHA_C=0.30 BASIS=aug-cc-pVTZ \
  bash benchmarks/dipole_cccbdb/jobs/obdh/job.sh --submit --dipole dm
```

`CODE_ROOT`, `RESULT_ROOT`, `OUTPUT_DIR`, `BASIS`, `SCF_POLICY`, `DIPOLE_MODE`
and `MAX_MEMORY` follow the other jobs' environment convention. Use a new
`OUTPUT_DIR` when changing basis, methods, tolerances or estimator settings;
the engine rejects incompatible checkpoints. Set `OUTPUT_DIR` explicitly
when merging an older run after changing the alpha pair in the script.
Use `--submit`: scheduler `_output` and `_error` directories must exist
before SLURM opens its log files, and the wrapper creates them first.

### Expanded input and existing results

Appending species changes the hash of the **whole input file**. Existing
ten-species checkpoints cannot be resumed with this twenty-species input,
even though indices 0–9 and their geometries are unchanged. New defaults
are `v2` for CC/DFT and `scaling_v2/<alpha-tag>` for OB, preserving the old
directories. Do not copy old point files into the new directory.

To calculate only the ten additions (repeat for `dft` or `obdh` as needed):

```bash
ARRAY_SPEC='10-19%3' bash benchmarks/dipole_cccbdb/jobs/ccsd/job.sh --submit
```

The default `--submit` runs all 20. A new-only run reports only its requested
indices; it does not import old results. `ARRAY_SPEC` applies to the submission
wrapper; with raw `sbatch`, pass `--array` explicitly.
To merge an old run, set `OUTPUT_DIR` to its old path; report-only merge
uses the geometries stored in that run's metadata.

After the corresponding array finishes:

```bash
bash benchmarks/dipole_cccbdb/jobs/obdh/job.sh --merge
bash benchmarks/dipole_cccbdb/jobs/dft/merge.sh
bash benchmarks/dipole_cccbdb/jobs/ccsd/merge.sh
```

`--merge` only reads existing checkpoints and creates the report; it does
not rerun quantum calculations. A missing metadata file is an error.

## Direct Python use

The local input and reference are defaults, so the entry point can be run
from any working directory. For an initial small-basis check:

```bash
python -u benchmarks/dipole_cccbdb/methods/obdh/calc_dipmomt_obdh.py \
  --outdir /data/giahuy/Result/OBDH/dipole_cccbdb/check_sto3g \
  --only HF H2O --methods uhf ump2 obmp2 obdh --basis sto-3g \
  --alpha 0.50 0.40 --dipole dm --scf-policy standard \
  --threads 4 --max-memory 4000
```

Omit `--only` to run all twenty serially. `--index 0` runs only HF; SLURM array
indices are read automatically. `--resume` skips complete matching cases and
retries failed molecules. To generate a report after indexed/array execution:

```bash
python -u benchmarks/dipole_cccbdb/methods/obdh/calc_dipmomt_obdh.py \
  --outdir /data/giahuy/Result/OBDH/dipole_cccbdb/check_sto3g --merge
```

`--dipole det` requests determinant dipoles; `dm` adds the engine's T2,
scaled-T2 or CC-Lambda density as appropriate; `ff` requests energy finite
differences; `both` requests both density and field estimates. `unrel` is
the existing engine's alias for `dm`. For OBMP2 the added density is T2;
for OBDH it is alpha_c-scaled T2. These are estimator labels, not a claim
that the density equals an energy derivative. CC Lambda omits the SCF
orbital-response Z-vector. The default field step is 1e-4 a.u.; multiple
`--fields` enable the existing step-consistency check.

## Files produced when running

| Path under output directory | When / contents |
|---|---|
| `dipole_<index>.txt` | Created by `.sh` at startup; paths, parameters and driver progress |
| `config.json` | Initial run parameters |
| `dipole.meta.json` | Input geometries, physical fingerprint, source hashes, indices and case status |
| `logs/000_HF.log` … `logs/019_PH3.log` | Live native SCF/MP2/OB/CC and field logs, one per molecule; attempts append |
| `points/000_HF.json` … `points/019_PH3.json` | Molecule checkpoints written after each molecule completes or fails |
| `.metadata.lock.d/`, `points/.*.lock.d/` | Temporary metadata/case locks, normally removed on exit |
| `merge.txt` | Merge launcher output |
| `summary.txt` | Completion and statistics after serial completion or merge |
| `Results.csv`, `Statistics.csv`, `Energies.csv`, etc. | Narrow report tables, including vectors and convergence |
| `dipole.workbook.json` | Reproducible workbook data/formulas |
| `dipole.xlsx` | Workbook after serial completion or one explicit array merge |

SLURM `.out` and `.err` files go to `RESULT_ROOT/_output` and `_error`.
They are separate from the Python driver `.txt` and native molecule logs.
Array workers do not write the shared workbook concurrently. Unconverged
dipoles stay blank in accepted results; raw diagnostics remain available.
This benchmark inherits the engine's statistics and regularized errors
`100*(mu-mu_ref)/max(mu_ref,1 D)`, rather than defining a new score.

For live monitoring of array task 0, follow `dipole_0.txt` and then the
exact `Live calculation log:` path it prints. The log directory is created
before molecule calculations. No HPC jobs are submitted by creating these files.

## Local validation

### Original ten-species validation

Validated with Python 3.12, PySCF 2.14.0 and the available pyCMF checkout
at commit `1bd2c79`:

- Five dataset, CLI-default and mock-SLURM tests passed:
  `python -m unittest discover -s benchmarks/dipole_cccbdb/tests -v`.
- All ten geometries completed UHF/UMP2 density calculations with STO-3G,
  standard SCF, one thread and no PT2 DF: 30 accepted report rows, each
  paired with its local experimental reference.
- HF completed UHF, UMP2, OBMP2, OBDH, PBE0, B3LYP, B2PLYP, CCSD and
  CCSD(T), using STO-3G, alpha=(0.5,0.4), density plus finite field at
  1e-4 a.u.: 24 accepted estimator rows. This checks execution paths,
  not accuracy with a production basis.
- A separate indexed PBE0 run produced only its checkpoint/log/metadata;
  explicit merge then created CSV/XLSX with the CCCBDB HF reference.
- Python syntax, Bash syntax and whitespace checks passed. Generated
  workbooks were opened and their expected sheets/reference pairs checked.

Validation outputs remain outside the repository. The configured HPC
aug-cc-pCVQZ calculation has not been run or submitted.

### Twenty-species extension (2026-10-31)

- Five dataset/CLI/mock-SLURM tests passed, including submission of the default
  `0-19%3` array and an explicit `10-19%3` subset for all three families.
- All 20 inputs built successfully with PySCF 2.14.0 and `aug-cc-pVDZ`;
  formula, electron count, coordinates, non-overlap and reference pairing checked.
- Expected rotations/reflections for each of the ten new geometries matched
  same-element atoms within 0.0002 Å (four-decimal published coordinates).
- New coordinates, dipole magnitudes, dipole citation identifiers and source
  URLs were checked against the tables read directly in the CCCBDB browser pages.
  Original ten input records and reference values are unchanged.
- No new SCF/dipole calculations or HPC submissions were performed for this
  extension. The sandbox lacked a resolver for `aug-cc-pCVQZ`; the production
  basis setting is unchanged and was not validated by this local build check.
