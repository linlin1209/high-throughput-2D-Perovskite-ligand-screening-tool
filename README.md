# high-throughput-2D-Perovskite-ligand-screening-tool
# Software requirement: 
- ORCA version 4.0
- LAMMPS
- AmberTools
- SBATCH submission system
# 1. MD
- step0.prepare_ligand_batch: parametrized xyz files in the executed folder in batch
- step1.build_bulk_sims: writting LAMMPS MD simulation files and submitting MD jobs
- step2.evaluate_octahedra: evaluated bond and angle deviation for each ligands' simulations
- step3.collect_stabilities: collect all ligands' stability
# 2. ML
- ML_all_model.py: hyper-parameter tuning and training for all the models
- ML_single_model.py: training RF model using the tuned hyper-parameters
# 3. MD_test: 
example run for one ligand, see README file in that folder
