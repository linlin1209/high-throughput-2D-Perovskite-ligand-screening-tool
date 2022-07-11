#!/bin/bash
#
#SBATCH --job-name NH3_2C_Angew-1+H_minimized_Q
#SBATCH -o orca.out
#SBATCH -e orca.err
#SBATCH -A bsavoie
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 4:00:00

#load necessary module

module load openmpi/3.1.4

/depot/bsavoie/apps/orca_4_1_2/orca charge.in > charge.out
