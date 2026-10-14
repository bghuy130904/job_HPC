# Preliminary reconstruction validation

The original archive contains 12 relative PES profiles, no calculation sources
and no full run metadata. The restored benchmark reuses the current OBDH and
shared MP2 embedding kernels rather than guessing or fitting the old energies.

## Calculations

- Two molecules: ethanol (active OH), 1-butanol (active CH2OH).
- Absolute C2-C1-O-H angles: 0,60,90,120,180 degrees.
- CL n_shells=1 (paper 0th shell), basis 6-311++G(2d,p).
- Alpha=(0.5,0.4), matched DH-BLYP environment for both methods.
- Independent UHF initial guess per geometry; no multi-guess selection.
- SCF tolerance 1e-10, maximum 200 cycles; OBDH energy tolerance 1e-8,
  maximum 300 iterations plus upstream F_ai convergence criterion.
- SCF and solver convergence are checked; absolute energies, source hashes and
  per-point logs are preserved. No geometry optimization was performed.

Two five-point ethanol controls additionally test OBDH alpha=(0.5,0.5)
and MP2-in-B3LYP. Those do not improve agreement with the corresponding old
profiles: OBDH control maximum relative error 0.16935 kcal/mol, MP2-in-B3LYP
2.88751 kcal/mol, using a common 180-degree zero.

## Angle definition

The supplied ethanol XYZ starts at 0 degrees, but butanol starts at 180 degrees.
The restored default sets the actual dihedral, measuring and correcting the
starting angle. A rotation-offset mode is exposed explicitly for alternate
protocols, not silently substituted. New output records the signed physical
angle as well as the scan parameter. The old profile matches the absolute-angle
interpretation best for OBDH; the earlier concern about an offset is not an
established error in the old source.

## Interpretation and limits

The metrics compare relative energies at identical sampled angles after moving
both datasets to the same 180-degree zero. Original files instead use each
profile's minimum; coarse sampling may miss that minimum. The numerical comparison never
interpolates or invents calculated energies. Solid plot lines connect the five
computed markers as guides; dashed lines show all 19 archived samples. See metrics.json and comparison.csv
under smoke_matched for the final numerical results.

A close relative curve does not establish absolute-energy reproduction.
The original JSONs do not record the numerical or solver settings required to
identify the source of absolute shifts. Current MP2 embedding does not reproduce
the archived MP2 profiles; neither changing the ethanol environment to B3LYP nor
changing OBDH alpha to (0.5,0.5) solves that discrepancy. This validation does not
claim the legacy MP2 data or the current MP2 kernel is erroneous. A definitive
explanation requires the original computational kernel/version or richer old
logs. The current shared MP2 implementation evaluates active HF plus MP2
correlation with the relaxed DFT embedding baseline; its source is left intact.

Only shell 0 was recalculated here. The original shell 1 and shell 2 curves are
preserved in references; the restored driver supports their full recalculation
on HPC. Five sampled angles cannot exclude additional extrema between points.

The exact runtime versions and commits are in environment.json. This runtime
needed a nonnumerical PySCF memory-probe adaptation to /proc/self/statm because
its PID namespace differs from the mounted /proc. No electronic-structure
kernel was changed.

Geometry invariants and starting dihedrals pass unit tests; both Slurm launchers
pass bash syntax validation. Resume preserves accepted points and rejects
changes in physical settings or dependency hashes.

## Final primary-run results

20 of 20 primary points and all 10 ethanol control points converged.

| Molecule | Method | Points | RMS difference (kcal/mol) | Maximum difference (kcal/mol) | Absolute E(180) shift (Eh) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1-butanol | obdh | 5 | 0.144931 | 0.199776 | -0.056076109 |
| ethanol | obdh | 5 | 0.013501 | 0.019279 | +0.005864404 |
| 1-butanol | mp2 | 5 | 2.789876 | 3.815813 | +0.045147355 |
| ethanol | mp2 | 5 | 1.902660 | 2.825618 | +0.050506892 |

The OBDH sampled shapes agree closely, particularly ethanol, but absolute total energies are not reproduced. MP2 is materially different for both molecules.

Interpreting butanol legacy phi as a rotation offset worsens OBDH RMS to 0.830474 kcal/mol (versus 0.144931 with the labeled absolute dihedral), so it does not explain the discrepancy.
