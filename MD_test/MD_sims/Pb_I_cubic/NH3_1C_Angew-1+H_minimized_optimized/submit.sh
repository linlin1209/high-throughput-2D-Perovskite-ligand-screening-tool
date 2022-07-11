#!/bin/sh
#SBATCH --job-name=ML_NH3_1C_Angew-1+H_minimized_optimized
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 4:00:00
#SBATCH -A standby
#SBATCH -o ML_NH3_1C_Angew-1+H_minimized_optimized.slurm.out
#SBATCH -e ML_NH3_1C_Angew-1+H_minimized_optimized.err

# Load default LAMMPS, compiler, and openmpi packages
module load gcc/9.3.0 
module load openmpi/3.1.4 
module load ffmpeg/4.2.2  
module load  openblas/0.3.8  
module load  gsl/2.4  


# cd into the submission directory
cd /depot/bsavoie/data/Lin/perov_ML_lin/mono/MD_sims/Pb_I_cubic/NH3_1C_Angew-1+H_minimized_optimized

for((i=1;i<=5;i++)) do
   cd run$i

   # Submiting LAMMPS job for eval.in.init
   mpirun -np 128 /depot/bsavoie/apps/lammps_bell/bin/lmp -in eval.in.init >> LAMMPS_run.out &
   wait
   cd ..
   wait
done

