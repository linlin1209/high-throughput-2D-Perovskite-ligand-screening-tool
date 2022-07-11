#!/bin/bash
#
#SBATCH --job-name NH3_2C_Angew-1+H_minimized_ext_Q
#SBATCH -o orca.out
#SBATCH -e orca.err
#SBATCH -A bsavoie
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4:00:00

python /home/lin1209/perov_ml_github/util/extract_charges.py charges.vpot -xyz ../geo_xtb/xtbopt.xyz -out charge.out -q 1 --two_step
