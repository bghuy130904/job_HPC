# CCSD(T) reference supplied for dimmer_cl

`raw/intE_ccsdt.csv` preserves the uploaded file byte for byte.
`intE_ccsdt.csv` contains the same eight points and the same numeric values,
with headers aligned to the interaction-energy driver:

| Original column | Repository column | Meaning |
| --- | --- | --- |
| R | R_scale | Dimensionless R/R0, subject to geometry confirmation |
| E_dimer_AB | E_AB_Embed_Eh | Supplied full-system CCSD(T) AB energy, Eh |
| E_frag_A_basisAB | E_A_CP_Embed_Eh | Supplied A energy with AB basis, Eh |
| E_frag_B_basisAB | E_B_CP_Embed_Eh | Supplied B energy with AB basis, Eh |
| E_int_Eh | E_int_Eh | AB − A_CP − B_CP, Eh |
| E_int_kcal_mol | E_int_kcal_mol | Interaction energy, kcal/mol |
| time_s | Runtime_Point_s | Supplied elapsed time, seconds |

The `Embed` energy column names are retained for schema compatibility only;
they do not mean that the CCSD(T) reference used embedding. `Method=CCSD(T)`
is the user-provided method label, not independently verified from calculation logs.
`Point` enumerates rows. `Status=UNVERIFIED`, blank `Basis` and
`Convergence_status=UNKNOWN` explicitly preserve missing provenance.

## Checks performed

All eight R values are unique, positive and match the default driver scan.
All supplied numbers are finite. CP subtraction and conversion using
627.509474 kcal/mol/Eh agree with the supplied interaction energies within
1e-10 Eh and 1e-6 kcal/mol respectively. Numeric consistency does not establish
that the three total energies were calculated with the same method and setup.

The interaction energies range from approximately −481 to −466 kcal/mol;
at R/R0=2 they are still −466.054701 kcal/mol. This large offset requires
checking the original calculations before using these values as a validated
benchmark. Possible causes to investigate include inconsistent energy levels
between AB and the monomers, geometry, charge/spin, basis, or CP setup.
No cause is established by this CSV alone, and no offset has been subtracted.

Confirm basis/auxiliary basis and correlation convention (CCSD(T) versus
CCSDT, frozen core, DF approximations), geometry and R definition, charges/spins,
ghost centers, and convergence for AB/A_CP/B_CP from the source driver/logs.
Do not compare against the STO-3G tests until the basis and other settings match.

The current intE driver calculates energies; it does not read reference files.
The normalized table can be joined to matching calculation tables on `R_scale`
after provenance is verified. `tools/collect.py` only combines run tables;
it does not compute reference errors or validate CCSD(T). `UNVERIFIED` rows
must not be silently treated as validated reference points in error statistics.
