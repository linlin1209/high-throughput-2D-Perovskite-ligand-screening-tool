#!/bin/env python
"""
@author: Stephen Shiring

This script will collect, process (i.e., determine if it's stable or not), and insert into the database all the 
formabilities outcomes it discovers inside each directory. It expects a directory structure to look like:

/[master]/
/[master]/Ligands/
/[master]/MD_sim/
/[master]/MD_sim/[metal]_[halide]_[phase]  (eg: Pb_I_cubic)

/[master]/Ligands/: contains files pertaining to geometry optimization of a ligand; expects, at the minimum, *.png (chemical drawing) and *.smi (smiles string)
files, which it will copy over and read, respectively (these are generated when the ligand optimization and parameters file job creation task is run). Expects
ligand filenames to conform to [headgroup]_[linker]_ligand; e.g.: NH3_2C_Anthracene-center

/[master]/MD_sim/: this directory contains the MD simulations to determine stability from. The MD job builder will generate 5 independent trajectories, stored:
/[master]/MD_sim/run[N], where N = 1-5. These contain the nve+nvt (100ps) runs. ML.parse_form_bulk.sh will evaluate the formabilities for each run, storing them in
stability.out inside /[master]/MD_sim/. Stability contains the computed bond length quadratic elongation and bond angle variance for each run.

This script will read the stability file, average each bond and angle parameter, and then determine if the stability criteria are satisfied. It processes every
ligand directory inside /[master]/, skipping those who already have entries inside the database (unless --update is specified)
         
"""

import argparse
import sys
import os
import shutil

# Add root directory to system path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import functions_db
import functions

def main(argv):
    
    parser = argparse.ArgumentParser(description='')
    
    # Optional arguments    
    parser.add_argument('-name', dest='db_name', type=str, default='ML_perovskites.db',
                        help='Name of database to create. Stored in /data/ directory. Specify "None" to skip db-related steps. Default: ML_perovskites.db')
    
    parser.add_argument('-exclude_dirs', dest='exclude_dirs', type=str, default='Stability Formability',
                        help='Space-delimited string specifying directories to avoid processing. Default: Stability')
    
    parser.add_argument('-f', dest='stability_file', type=str, default='stability.out',
                        help='File containing stability evaluations. Default: stability.out')
    
    parser.add_argument('-b', dest='bond_lqe', type=float, default=0.1,
                        help='Cut off for difference between bond lqe and ideal perovskite. Default: 0.1')
    
    parser.add_argument('-a', dest='angle_var', type=float, default=30.0,
                        help='Cut off for difference in angle variance from ideal perovskite. Default: 30.0')
    
    parser.add_argument('-d', dest='output_dir', type=str, default='Stability',
                        help='Output directory. Default: Stability')
    
    parser.add_argument('-o', dest='output_name', type=str, default='stabilities.out',
                        help='Output filename, written to output_dir. Default: stabilities.out')
    
    parser.add_argument('--update', dest='force_update', action='store_const', const=True, default=False,
                        help = 'When invoked, update any previous-processed ligand entries.')
    
    parser.add_argument('--plot', dest='make_plots', action='store_const', const=True, default=False,
                        help = 'When invoked, generate violin plots of the bond and angle data.')
    
    args = parser.parse_args()
    
    args.exclude_dirs = args.exclude_dirs.split()
    
    if args.db_name.lower() == 'none':
        args.db_name = None
    
    # Check that we are in 'MD_sims' directory, ensure it's in current path
    if 'MD_sims' not in os.getcwd():
        print('Error: Not in the MD_sims directory. Aborting...')
        exit()

    # Make some directories to store output and chem drawing images, depending on their stability classification
    if not os.path.isdir( os.path.join( args.output_dir ) ): 
        os.mkdir( os.path.join( args.output_dir ) )
    if not os.path.isdir( os.path.join( args.output_dir, 'Passed' ) ): 
        os.mkdir( os.path.join( args.output_dir, 'Passed' ) )
    if not os.path.isdir( os.path.join( args.output_dir, 'Failed') ): 
        os.mkdir( os.path.join( args.output_dir, 'Failed' ) )
    if not os.path.isdir( os.path.join( args.output_dir, 'Strained' ) ):
        os.mkdir( os.path.join( args.output_dir, 'Strained' ) )
        
    # Start logger
    sys.stdout = functions.Logger(os.path.join(args.output_dir, 'collected_stability') )

    print("{}\n\nPROGRAM CALL: python {}\n".format('-'*150, ' '.join([ i for i in sys.argv])))
    
    # Obtain the parent directory (series_name), which is two levels about (series_name/MD_sims/metal_halide_crystal)
    parent_dir = os.path.join( '/', *os.getcwd().split('/')[:-2] )
    
    # Directionary to hold results, Stability filename, and acceptable cut off values for bond_lqe and angle_var 
    Results = {}
    stability_file   = args.stability_file
    bond_lqe_cutoff  = args.bond_lqe
    angle_var_cutoff = args.angle_var
    
    # Find all directories in current directory
    path = './'
    dirs = sorted([os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o)) and o not in args.exclude_dirs])
    
    # Holds general info about this cluster of simulations
    # Expects master directory name to be [metal]_[halide]_[perov_geom] (Pb_I_cubic)
    Info = {}
    info = os.getcwd().split('/')[-1]
    info = info.split('_')
    Info['metal']      = info[0]
    Info['halide']     = info[1]
    Info['perov_geom'] = info[2]
    
    # Data dictionary to hold all final (average) bond_lqes and angle vars, and a stability score for each ligand
    Data = {}
    Data['bond_lqe']    = []
    Data['angle_var']   = []
    Data['stability']   = [] # [bond_lqe, angle_var, stability] where stability is 0 = did not form, 1 = formed but strained, 2 = fully formed
    
    # Create a database connection
    # all databases live in /data/ directory
    if args.db_name != None:
        if not os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', args.db_name)):
            print('ERROR: Unable to find database "{}". Aborting...'.format(args.db_name))
            exit()
    
        connection = functions_db.create_connection(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', args.db_name))
    
    if args.db_name != None:
        Metals     = functions_db.get_indices(connection, 'metals')
        Halides    = functions_db.get_indices(connection, 'halides')
        Ligands    = functions_db.get_indices(connection, 'ligands')
        
#        ligand_rows = functions_db.retrieve_rows(connection, "SELECT name FROM ligands")
    
    # Loop over each directory and process the stability file, if present
    for d in dirs:
        print('Processing {}...'.format(d[2:]))
        
        name               = d[2:].split('_')
        Info['headgroup']  = name[0]
        Info['linker']     = name[1]
        Info['body_name']  = name[2]  # the base body name, such as 4T, BT, anthracene-1, etc. This gets inserted into the ligands table, since various simulations of different parameters may be run on the same ligand body. the name of the perov sim is what distinguishes between the sims, not the ligand body which may be idential
        Info['name']       = d[2:]    # total name, as in the name of the simulation directory, which may contain further clarifying information beyond the ligand body name such as 'minimized' and 'optimized'. This gets inserted into the perovskites table
        
        Results[d] = {}
        Results[d]['name']             = Info['name']
        Results[d]['smiles']           = ''
        Results[d]['description']      = ''
        Results[d]['metal']            = Info['metal']
        Results[d]['halide']           = Info['halide']
        Results[d]['headgroup']        = Info['headgroup']
        Results[d]['linker']           = Info['linker']
        Results[d]['perov_geom']       = Info['perov_geom']
        Results[d]['synthesized']      = False
        Results[d]['path']             = ''
        Results[d]['fit']              = True    # Did it pass the ligand fitter? Obviously if we are running an MD sim, then it fit.
        Results[d]['formed']           = False
        Results[d]['disordered']       = False   # Formed, but octahedron may be disordered (bond_lqe_check == True but angle_var_check == False)
        Results[d]['comments']         = ''
        Results[d]['out_missing']      = False
        Results[d]['bond_lqe']         = []
        Results[d]['bond_lqe_nan']     = 0
        Results[d]['bond_lqe_avg']     = 1000
        Results[d]['bond_lqe_diff']    = 1000
        Results[d]['bond_lqe_check']   = False
        Results[d]['angle_var']        = []
        Results[d]['angle_var_nan']    = 0
        Results[d]['angle_var_avg']    = 1000
        Results[d]['angle_var_diff']   = 1000
        Results[d]['angle_var_check']  = False
        Results[d]['problematic']      = False  # If there's an issue with this, such as it's missing bond_lqe or angle_lqe data
        
        if os.path.isfile(os.path.join(d, stability_file)):
            with open(os.path.join(d, stability_file), 'r') as f:
                for lc,line in enumerate(f):
                    if lc > 0:
                        fields = line.split()
                        if len(fields) == 3:
                            if fields[1] == 'nan':
                                Results[d]['bond_lqe_nan'] += 1
                            else:
                                Results[d]['bond_lqe'].append(float(fields[1]))
                            
                            if fields[2] == 'nan':
                                Results[d]['angle_var_nan'] += 1
                            else:
                                Results[d]['angle_var'].append(float(fields[2]))
            
            # Compute avergae bond_lqe, then compute difference from ideal bond_lqe (=1.0) for use evaluating
            if len(Results[d]['bond_lqe']) > 0:
                Results[d]['bond_lqe_avg']    = sum(Results[d]['bond_lqe'])/len(Results[d]['bond_lqe'])
                Results[d]['bond_lqe_diff']   = abs(1.0 - Results[d]['bond_lqe_avg'])
                Results[d]['bond_lqe_check']  = True if Results[d]['bond_lqe_diff'] <= bond_lqe_cutoff else False
                Data['bond_lqe'].append(Results[d]['bond_lqe_diff'])
                print('   {:15} {:>6.4f} with {} NAN - diff: {} --> {}'.format('Bond_lqe avg:', Results[d]['bond_lqe_avg'], Results[d]['bond_lqe_nan'], Results[d]['bond_lqe_diff'], 'Passed' if Results[d]['bond_lqe_check'] else 'Failed'))
            else:
                print('    Bond_lqe avg not computed because no bond_lqe data found in stability file.')
                Results[d]['problematic']    = True
                
            # Compute average angle_var, then use this for evaluating (ideal angle_var = 0.0, although in practice it seems that the angle_var of the experimental ligands is ~20.0 deg)
            if len(Results[d]['angle_var']) > 0:
                Results[d]['angle_var_avg']    = sum(Results[d]['angle_var'])/len(Results[d]['angle_var'])
                Results[d]['angle_var_diff']   = abs(Results[d]['angle_var_avg'])
                Results[d]['angle_var_check']  = True if Results[d]['angle_var_diff'] <= angle_var_cutoff else False
                Data['angle_var'].append(Results[d]['angle_var_diff'])
                print('   {:15} {:>7.4f} with {} NAN - diff: {} -> {}'.format('Angle_var avg:', Results[d]['angle_var_avg'], Results[d]['angle_var_nan'], Results[d]['angle_var_diff'], 'Passed' if Results[d]['angle_var_check'] else 'Failed'))
            else:
                print('    Angle_var avg not computed because no angle_var data found in stability file.')
                Results[d]['problematic']    = True
    
            # Evaluate stability of ligand
            # formed (2): both bond_lqe and angle_var passed their checks
            # formed but disordered (1): bond_lqe passed but angle_var didn't pass
            # did not form (0): if both bond_lqe and angle_var didn't pass or bond_lqe didn't pass
            if not Results[d]['problematic']:
                if Results[d]['bond_lqe_check'] and Results[d]['angle_var_check']:
                    Results[d]['comments'] = 'True'
                    Results[d]['formed']   = True
                    Data['stability'].append([d[2:], Results[d]['bond_lqe_diff'], Results[d]['angle_var_diff'], 2])
                elif Results[d]['bond_lqe_check'] and not Results[d]['angle_var_check']:
                    Results[d]['comments']   = 'Most likely true: octahedron may be a bit distorted as angles are enlarged'
                    Results[d]['formed']     = True
                    Results[d]['disordered'] = True
                    Data['stability'].append([d[2:], Results[d]['bond_lqe_diff'], Results[d]['angle_var_diff'], 1])
                elif not Results[d]['bond_lqe_check'] and Results[d]['angle_var_check']:
                    Results[d]['comments'] = 'False'
                    Results[d]['formed']   = False
                    Data['stability'].append([d[2:], Results[d]['bond_lqe_diff'], Results[d]['angle_var_diff'], 0])
                else:
                    Results[d]['comments'] = 'False'
                    Results[d]['formed']   = False
                    Data['stability'].append([d[2:], Results[d]['bond_lqe_diff'], Results[d]['angle_var_diff'], 0])
            else:
                Results[d]['comments'] = 'Issue with this ligand, missing bond_lqe and/or angle_var data.'
                
            print('    Comments: {}'.format(Results[d]['comments']))
        
        else:
            Results[d]['out_missing'] = True
            continue
            
    # Summarize and update db
    Summary = {}
    Summary['Passed']       = []
    Summary['Failed']       = []
    Summary['Missing']      = []
    Summary['Problematic']  = []
    
    for job in sorted(Results.keys()):
        if Results[job]['out_missing']:
            Summary['Missing'].append(job[2:])
            continue
        elif Results[job]['problematic']:
            Summary['Problematic'].append(job[2:])
            continue
        else:
            # Copy over chemical structure image into appropriate directory
            img_path = os.path.join(parent_dir, 'Ligands', job[2:-10], job[2:]+'.png')
            if Results[job]['formed'] and not Results[job]['disordered']:
                try: shutil.copy(img_path, os.path.join(args.output_dir, 'Passed'))
                except: pass
                Summary['Passed'].append(job[2:])
            elif Results[job]['formed'] and Results[job]['disordered']:
                try: shutil.copy(img_path, os.path.join(args.output_dir, 'Strained'))
                except: pass
                Summary['Passed'].append(job[2:])
            else:
                try: shutil.copy(img_path, os.path.join(args.output_dir, 'Failed'))
                except: pass
                Summary['Failed'].append(job[2:])
            
            # Save smiles string
            smi_path = os.path.join(parent_dir, 'Ligands', job[2:-10], job[2:]+'.smi')
            try:
                with open(smi_path, 'r') as f:
                    smiles = f.readline()
                Results[job]['smiles'] = smiles.split()[0]
            except:
                Results[job]['smiles'] = 'None'
            
            
            if args.db_name != None:
                # Check
                if Results[job]['name'] not in list(Ligands.keys()):
                    print('WARNING: {} not found in ligands table. Skipping inserting record into perovskites_A table...'.format(Results[job]['name']))
                    continue
            
                # Update existing record or insert new record?
                if functions_db.check_perovskite_A(connection, record = { 'metal_id': Metals[Info['metal']], 'halide_id': Halides[Info['halide']], \
                                                                          'ligand_id': Ligands[Results[job]['name']], 'perov_geom': Info['perov_geom'] }):
                    # Update
                    if args.force_update:
                        print('Updating...')
                    else:
                        print('Not updating existing record...')
                    
                else:
                    # Insert new record into perovskites_A
                    print('inserting new row...')
                    Perovskite = {}
                    Perovskite['metal_id']         = Metals[Info['metal']] 
                    Perovskite['halide_id']        = Halides[Info['halide']]
                    Perovskite['ligand_id']        = Ligands[Results[job]['name']] 
                    Perovskite['perov_geom']       = Info['perov_geom'] 
                    Perovskite['name']             = Results[job]['name']
                    Perovskite['description']      = Results[job]['description'] 
                    Perovskite['synthesized']      = 1 if Results[job]['synthesized'] else 0 
                    Perovskite['path']             = Results[job]['path'] 
                    Perovskite['fit']              = 1 if Results[job]['fit'] else 0 
                    Perovskite['bond_lqe']         = Results[job]['bond_lqe_avg'] 
                    Perovskite['bond_lqe_diff']    = Results[job]['bond_lqe_diff'] 
                    Perovskite['bond_lqe_check']   = 1 if Results[job]['bond_lqe_check'] else 0
                    Perovskite['angle_var']        = Results[job]['angle_var_avg'] 
                    Perovskite['angle_var_diff']   = Results[job]['angle_var_diff']  
                    Perovskite['angle_var_check']  = 1 if Results[job]['angle_var_check'] else 0
                    Perovskite['formed']           = 1 if Results[job]['formed'] else 0
                    Perovskite['disordered']       = 1 if Results[job]['disordered'] else 0
                    Perovskite['volume_frac']      = -1 
                    Perovskite['comments']         = Results[job]['comments'] 
    
                    row_id = functions_db.add_perovskite_A(connection, Perovskite, commit=True)
                    if row_id != -1: 
                        print('-> New row inserted: ID {}'.format(row_id)) 
                    else:
                        print('failed to insert new row')

    # close db connection if necessary
    if args.db_name != None:
        connection.close()
    
    # Write out data file
    with open(os.path.join(args.output_dir, args.output_name), 'w') as o:
    #    o.write('{}\n\n'.format(' '.join([str(_) for _ in Data['bond_lqe']])))
     #   o.write('{}\n\n'.format(' '.join([str(_) for _ in Data['angle_var']])))
     
        o.write('{:^100} {:^7} {:^8} {}\n'.format('name', 'bond', 'angle', 'label (0: unstable, 1: strained, 2: stable)'))
        
        for v in Data['stability']:
            o.write('{:100} {:7.4f} {:8.4f} {:2}\n'.format(v[0], v[1], v[2], v[3]))

    # Print summary
    print('\nSummary:')
    print('\nPassed ({}):'.format(len(Summary['Passed'])))
    for i in Summary['Passed']:
        print('  {}'.format(i))
    print('\nFailed ({}):'.format(len(Summary['Failed'])))
    for i in Summary['Failed']:
        print('  {}'.format(i))
    print('\nMissing ({}):'.format(len(Summary['Missing'])))
    for i in Summary['Missing']:
        print('  {}'.format(i))
    print('\nProblematic ({}):'.format(len(Summary['Problematic'])))
    for i in Summary['Problematic']:
        print('  {}'.format(i))
    print()
        
    print('Total:       {}'.format(len(Data['stability'])))
    print('Passed:      {}'.format(len(Summary['Passed'])))
    print('Failed:      {}'.format(len(Summary['Failed'])))
    print('Missing:     {}'.format(len(Summary['Missing'])))
    print('Problematic: {}'.format(len(Summary['Problematic'])))
    print('\nFinished!\n')
    return

if  __name__ == '__main__': 
    main(sys.argv[1:])
