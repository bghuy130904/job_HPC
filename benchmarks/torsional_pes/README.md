# Torsional PES: ethanol and 1-butanol

Restored calculation drivers for the supplied `bench_torison_potential.zip`.
The archive retained geometries, active indices, 12 PES JSONs and a plotting
script, but not calculation sources or full run settings.

## Layout

- `inputs/`: original XYZs and one shared `scan.json` configuration.
- `references/`: all 12 original JSONs, unchanged.
- `methods/obdh/obdh.py`, `methods/mp2/mp2.py`: method entry points.
- `methods/shared/scan.py`: geometry generation, isolated workers, diagnostics,
  checkpointing and the legacy-compatible output schema.
- `jobs/{obdh,mp2}/input.sh`: Slurm submissions for both molecules.
- `tools/compare_pes.py`: compare matching angles and generate PNG/CSV/metrics.
- `tests/`: geometric invariants.

The MP2 driver imports the repository's existing `shared/mp2_embed.py`.
OBDH imports installed `pycmf.OBDH.OBDH_CL`. No computational kernel is copied
or changed. All calculations use the complete molecular AO basis, SPADE occupied
partitioning and CL virtual truncation; there are no caps or geometry relaxation.
Active atom indices (zero based): ethanol `[7,8]`; butanol `[10,11,12,13,14]`.
`--shells 1 2 3` means paper shells 0,1,2, respectively. No-CL is not shell 0.

## Scientific settings

Default basis `6-311++G(2d,p)` is recorded in the original JSONs.
Default alpha `(0.5,0.4)` comes from the supplied paper; both methods use the
matched environment `0.5*HF+0.5*B88,0.6*LYP`. The old JSONs do not identify
alpha, XC, DF auxiliary basis, grids, projector shift, solver tolerances or source
versions. These defaults are reconstruction assumptions, not recovered metadata.
Upstream OBDH uses `xc_env` in its active solver as well as the environment;
changing `--xc-env` changes both in that implementation.

Use PySCF and pycmf with `OBDH_CL`, `CL_embed` and the unrestricted solver.
Full and embedded SCFs use the requested `--scf-cycles` / `--scf-tol` controls.
OBDH additionally requires its upstream convergence flag, which currently tests
energy change and occupied-virtual Fock elements. Failed points have null energies,
not zeros. Outputs record source hashes, PySCF version and SCF convergence flags.
Resume requires matching settings and dependency hashes.

## Run

From the repository root:

```bash
python benchmarks/torsional_pes/methods/obdh/obdh.py \
  --molecule ethanol --threads 12 --max-memory 40000 \
  --output-dir results/torsional_pes/obdh/run01
python benchmarks/torsional_pes/methods/mp2/mp2.py \
  --molecule 1-butanol --threads 12 --max-memory 40000 \
  --output-dir results/torsional_pes/mp2/run01
```

Defaults run all 19 angles and three CL settings (57 points per molecule/method).
For a preliminary five-angle run, add `--angles 0 60 90 120 180 --shells 1`.
`--resume` retries failures and skips accepted points; `--dry-run` prints inputs.
The 180-degree anchor is included automatically for comparison.

Cluster submissions:

```bash
sbatch benchmarks/torsional_pes/jobs/obdh/input.sh
sbatch benchmarks/torsional_pes/jobs/mp2/input.sh
```

Jobs default to `/home/giahuy/Code/job` and write into
`/data/giahuy/Result/torsional_pes/<method>/<job_id>`.
Override `REPO_DIR` / `OUTPUT_DIR` with `sbatch --export=ALL,...` as needed.
Extra arguments are passed to both molecule runs, e.g. `--shells 1 --angles 0 60 90 120 180`.
For resubmission, retain the original OUTPUT_DIR and pass `--resume`.
Slurm stdout/stderr are created in the submission directory, before the job starts.

## Output and comparison

Every method/molecule has a metadata sidecar and per-point JSON/logs.
PES JSONs retain `molecule`, `basis`, `n_shells`, `e_ref_hartree` and
`scan[{phi,e_rel_kcal}]`, and add absolute energies, status and configuration.
Their zero is the lowest accepted sampled energy (a partial scan is not a proven
minimum). Compare both curves at a common 180-degree anchor to avoid differences
in the minimum definition.

```bash
python benchmarks/torsional_pes/tools/compare_pes.py \
  --runs results/torsional_pes/obdh/smoke_matched results/torsional_pes/mp2/smoke_matched \
  --output-dir results/torsional_pes/validation/smoke_matched
python -m unittest discover -s benchmarks/torsional_pes/tests -v
```

See `results/torsional_pes/validation/VALIDATION.md` for actual preliminary
calculations, discrepancies and the limits of reproducibility.
