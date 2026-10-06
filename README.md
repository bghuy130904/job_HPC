# job_HPC — workspace by benchmark

Organized from commit 822fd36915ff16abd392b9af3e16c99b1986dda9. This change organizes files only; it does not assess or change the numerical implementations.

## Find a benchmark

| Benchmark | Method folders | Inputs and reference data |
| --- | --- | --- |
| dipole_152 | ccsd, dft, obdh | inputs/sp_inputs.json, inputs/nsp_inputs.json; references/ref_sp.json and ref_nsp.json |
| dipole_dissociation | ccsd, dft, obdh | inputs/dissociation_inputs.json; references/dissociation_reference.json |
| sie4x4 | dft, obdh; ORCA scripts under tools and jobs/orca | inputs/input.json; inputs/orca_generated/ |
| mipc | mp2, obmp2, obdh, b2plyp, pbe0 | Geometries remain in the existing method scripts; extracting them is a separate code change |
| hexanol_cl | mp2, obdh | inputs/input.xyz |
| retinal | obmp2 | Existing scripts only |

The obdh dipole driver already supports UHF, UMP2, OBMP2 and OBDH through method selection. Keep one driver instead of copying it into four folders. Method folder names describe existing drivers, not separate copies for every supported method.

## Folder rules

- benchmarks/<benchmark>/inputs/: one shared copy of identical geometry data.
- benchmarks/<benchmark>/references/: shared reference values.
- benchmarks/<benchmark>/methods/<method>/: calculation scripts; local helper imports stay beside their callers.
- benchmarks/<benchmark>/jobs/<method>/: submission and merge scripts.
- benchmarks/<benchmark>/tools/: input generators and collectors.
- shared/: one implementation of identical helper modules; method folders contain relative symbolic links where needed for existing import names.
- results/<benchmark>/<method>/legacy/: existing result workbooks. Keep their names to preserve their history.
- experiments/: exploratory scripts without an established benchmark.
- archive/: earlier launchers and input variants; excluded from the main benchmark navigation.

Use results/<benchmark>/<method>/<run_id>/ for future outputs. Use one results folder per run instead of overwriting a folder called final.

Do not create a new calculation script when only basis, coefficients, method selection or active region changes. Store those choices in a run configuration when the driver supports it. If it currently hard-codes a choice, changing that interface is a separate task.

## What moved

FILE_MAP.csv maps every original tracked file to its current location. Identical copies were consolidated; differing input variants and result versions were preserved. ORCA generated inputs remain under their benchmark, with one generator and the two distinct collector versions given distinct names.

Earlier dipole geometries remain under archive/dipole_152/earlier_inputs rather than being silently substituted for the benchmark inputs. No scientific choice between different geometries was made.

## Running jobs after reorganization

File contents are unchanged. Existing launcher and Python absolute paths still use the old HPC layout. This branch is an organized workspace, not a tested replacement for current HPC submissions. Update those paths in a separate launcher migration before submitting from this layout. No numerical benchmark was run for this organizational change.
