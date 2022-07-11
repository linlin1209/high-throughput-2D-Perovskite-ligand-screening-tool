#!/bin/bash
#
#SBATCH --job-name NH3_2C_Angew-1+H_minimized_geo
#SBATCH -o xtb.out
#SBATCH -e xtb.err
#SBATCH -A bsavoie
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4:00:00
/depot/bsavoie/apps/xTB/bin/xtb /home/lin1209/test_ligand/NH3_2C_Angew-1+H_minimized.xyz --opt --chrg 1
