#!/bin/bash
python /home/lin1209/perov_ml_github/util/make_db.py /home/lin1209/test_ligand/NH3_2C_Angew-1+H_minimized/geo_xtb/xtbopt.xyz /home/lin1209/perov_ml_github/gaff2_perov.dat -charge_file ../charge/charge_parse/fit_charges.db -m 1 -q 1 -amber /depot/bsavoie/apps/amber20/bin/ --perovskite -o NH3_2C_Angew-1+H_minimized > makedb.out
