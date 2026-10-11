Direct Python outputs go to `<run_id>/` with interaction_energies.csv,
convergence.csv, metadata.json, logs/, geometries/ and checkpoints/.
HPC jobs write under /data/giahuy/Result/CCSDT/dimmer_cl/<Slurm-job-id>/.
Use matched basis/settings and all three fragment statuses before promoting a
run to references. The old uploaded reference is retained separately.
