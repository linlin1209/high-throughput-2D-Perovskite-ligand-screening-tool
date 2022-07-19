#!/bin/env python
import sys, os, shutil, subprocess ,argparse

# Append root directory to system path and import common functions
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import functions
import functions_writers

def main(argv):
    global repo_path
    repo_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Builds and submits MD bulk stability sims. Operates on all discovered paired xyz/db files sharing a filename.')
    
    # positional arguments
    parser.add_argument('cation', type=str,
                        help='Identity of element A / cation. Specify a single atom or an *.xyz file for multiple atoms / small molecule.')
    
    parser.add_argument('metal', type=str,
                        help='Identity of element B / metal cation.')
    
    parser.add_argument('anion', type=str,
                        help='Identity of element X / anion. Also accepts a space-delimited string of 2 atoms to generate a heterojunction.')
    
    parser.add_argument('bond_length', type=float,
                        help='"Bond length" between metal-halide. Metal-metal distance is twice this (so unit cell parameter is inputed as half).')
    
    # Optional arguments    
    parser.add_argument('-exclude_xyzs', dest='exclude_xyzs', type=str, default='',
                        help='Starting number for run directories, inclusive.')
    
    parser.add_argument('-start', dest='start', type=int, default=1,
                        help='Starting number for run directorys, inclusive.')

    parser.add_argument('-end', dest='end', type=int, default=5,
                        help='Ending number for run directorys, inclusive.')
    
    parser.add_argument('-config', dest='config_name', type=str, default='config.txt',
                        help='Configuration filename. Default: config.txt (in current working directory')
    
    args   = parser.parse_args()
    config = functions.read_config(args.config_name)
    
    sys.stdout = functions.Logger('step1.build_bulk_sims')

    print("{}\n\nPROGRAM CALL: python {}\n".format('-'*150, ' '.join([ i for i in sys.argv])))
    
    args.exclude_xyzs = args.exclude_xyzs.split()
    
    submitted_jobs = []
    
    # Find all minimized xyz files
    d = './'
    files = sorted( [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f.endswith('.xyz') and f not in args.exclude_xyzs] )
    
    for f in files:
        dir_prefix = f.strip('.xyz')
        print('Working {}...'.format(dir_prefix))
        
        # Check to make sure there isn't a stray .xyz file
        if not os.path.isfile(dir_prefix+'.db'):
                print('   Missing forcefield database file, skipping...')
                continue

        # Build the perovskite and simulation directory if it doesn't exist
        if not os.path.isdir(dir_prefix): 
            subprocess.call('python {}/util/perovskite_builder.py {} {} {} {} -dims "3 0 3" -surface {} -FF {}.db --monolayer -o {} -mixing_rule lb -y_pad 30.0'.format(repo_path, args.cation, args.metal, args.anion, args.bond_length, f, f.strip('.xyz'), f.strip('.xyz') ), shell=True)
        
            if not os.path.isdir(dir_prefix): 
                print('perovskite_builder.py failed to build "{}" perovskite, skipping...'.format(f))
                continue
        
            # move the xyz file to simulation directory
            shutil.move(f, '{}/{}_single.xyz'.format(dir_prefix,dir_prefix))
            # move the db file to simulation directory
            shutil.move(dir_prefix+'.db',dir_prefix)
        
        # Move into the directory and build and submit the individual runs, if these directories already exist, skip that particular run
        with functions.cd( dir_prefix ):
            if not os.path.isfile(dir_prefix+'.data'):
                print('   Missing simulation data file, skipping...')
                continue
            if not os.path.isfile(dir_prefix+'.in.settings'):
                print('   Missing simulation settings file, skipping...')
                continue
                
            # create jobs for each run
            for i in range(args.start, args.end+1):
                if os.path.isdir('run{}'.format(i)):
                    print('Run directory run{} already exists, skipping to avoid overwritting files.'.format(i))
                    continue
                    
                os.mkdir('run{}'.format(i))
                shutil.copy(dir_prefix+'.data', 'run{}'.format(i))
                shutil.copy(dir_prefix+'.in.settings', 'run{}'.format(i))
                
                with functions.cd ( 'run{}'.format(i) ):
                    c = functions_writers.write_LAMMPS_init(dir_prefix, data_name=f.strip('.xyz')+'.data', timesteps_npt=100000, timesteps_nvt=100000, nve_cycles=1, pressure_axis='x z', output='eval.in.init')
                    if not c:
                        print('Error writing input script for NVT job. Aborting...')
                        exit()
                    
                    c = functions_writers.make_LAMMPS_submission('eval.in.init', 'ML_{}_r{}'.format(f.strip('.xyz'), i,), config['JOB_MD_NODES'], config['JOB_MD_PPN'], config['JOB_MD_QUEUE'], config['JOB_MD_WALLTIME'], lammps_exe=config['LAMMPS_PATH'])
                    if not c:
                        print('Error writing submission script. Aborting...')
                        exit()
                    
                    submission_ID = subprocess.check_output('sbatch submit.sh', shell=True)
                    submitted_jobs.append('Created and submitted ML_{}_r{}. JobID: {}'.format(f.strip('xyz'), i, submission_ID))
    
    print('Submitted:')
    for s in submitted_jobs:
        print(s)
    print('total jobs submitted: {}'.format(len(submitted_jobs)))
    
    print('\nFinished!\n')
    return

if  __name__ == '__main__': 
    main(sys.argv[1:])
