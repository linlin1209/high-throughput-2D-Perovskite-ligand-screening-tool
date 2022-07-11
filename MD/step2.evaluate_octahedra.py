#!/bin/env python
#author: Stephen Shiring

import argparse,math,os,sys
import numpy
import numpy as np
from copy import deepcopy

# Append root directory to system path and import common functions
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import functions

def main(argv):
    parser = argparse.ArgumentParser(description='Parse the BX6 octahedrons\' geometry and quantify PbX 6 distortion by calculating bond length '\
                                     'quadratic elongation (<lambda>) and their bond angle variance (sigma^2).')

    # positional arguments
    parser.add_argument('-d', type=str, dest='dirs', default='',
                        help = 'Space-delimited string of directories to operate on. If empty, operate on all discovered directories. Default: "" (empty)')
    
    parser.add_argument('-traj', type=str, dest='traj_file', default='0.nvt.lammpstrj',
                        help = 'Name of trajectory to parse. Default: 0.nvt.lammpstrj')
    
    parser.add_argument('-s', type=str, dest='success_file', default='0.success',
                        help = 'Name of corresponding success file for trajectory to parse. Default: 0.success')
    
    parser.add_argument('-f_start', type=int, dest='f_start', default=0,
                        help = 'Index of the frame to start on (0-indexing, inclusive, with the first frame being 0 irrespective of timestamp) Default: 10')

    parser.add_argument('-f_end',   type=int, dest='f_end', default=100,
                        help = 'Index of the frame to end on (0-indexing, inclusive, with the first frame being 0 irrespective of timestamp) Default: 100')

    parser.add_argument('-f_every', type=int, dest='f_every', default=1,         
                        help = 'Frequency of frames to parse Default: 1, every frame is parsed')

    parser.add_argument('-o',       type=str, dest='output', default='stability.out',
                        help='Name of output file containing stabilities. Default: stability.out')
    
    parser.add_argument('--disordered',  dest='flag_disordered',        action='store_const', const=True, default=False,
                        help = 'Invoke for systems that are disordered, i.e., not an ideal perovskite framework.')
    
    parser.add_argument('--debug',  dest='flag_debug',        action='store_const', const=True, default=False,
                        help = 'When invoked, print out diagnostic info to log file.')
    
    parser.add_argument('-exclude_dirs', dest='exclude_dirs', type=str, default='Stability',
                        help='Space-delimited string specifying directories to avoid processing. Default: Stability')
    
    
    args = parser.parse_args()
    sys.stdout = functions.Logger('step2.eval_octa')
    print("{}\n\nPROGRAM CALL: python evaluate_octahedra.py {}\n".format('-'*150, ' '.join([ i for i in argv])))
    
    # Infer metal and halide(s) of the simulation from the master directory name
    # Expects master directory name to be [metal]_[halide]_[perov_geom] (Pb_I_cubic)
    Info = {}
    info = os.getcwd().split('/')[-1]
    info = info.split('_')
    Info['metal']      = [ info[0] ]
    Info['halide']     = info[1].split('-')
    Info['perov_geom'] = info[2]
    Info['elements']   = Info['metal'] + Info['halide']

    
    traj_file = args.traj_file
    success_file = args.success_file
    
    args.dirs = [ './'+_ for _ in args.dirs.split() ]
    if len(args.dirs) == 0:
        # Look over all directories in current directory
        path = './'
        dirs = sorted([os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o)) and o not in args.exclude_dirs])
    else:
        dirs = args.dirs
    
    for d in dirs:
        print('Operating on {}'.format(d[2:]))
        
        # Move into operating directory
        with functions.cd(d):
            
            # If there are no run dirs, skip
            path = './'
            run_dirs = sorted([os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o)) and 'run' in o])
            if len(run_dirs) == 0:
                print('   ERROR: No run directories found, skipping...')
            
            # Check if map file exists, if not, skip
            #if not os.path.isfile(d[2:]+'_map.map'):
            mapfile = [ _ for _ in os.listdir() if len(_.split('_correspondence.map'))>1]
            #mapfile = d[2:].strip('/')+'_correspondence.map'
            if len(mapfile) == 0:
                print('   ERROR: Did not find TAFFI-style map file, skipping...')
                continue
            mapfile = mapfile[0] 
            prefix = mapfile.split('_correspondence.map')[0]
    
            atomtypes, elements, masses, charges, adj_list, masses_dict, at_indices, at_counts = parse_map(mapfile, Info['elements'])
            #atom_info,at_indices,bond_info,angle_info,torsion_info,box_lo,box_hi = read_lmpdata(d[2:].strip('/')+'.data',Info['elements'])
            lmpatom = read_lmpdata(prefix+'.data')
            # at_indices: key:element, value: elements' indicies
            
            
            # Data dictionary, holds run/bond_lqe/angle_var for the runs
            Data = {}
            Data['runs']       = []
            Data['bond_lqe']   = []    # bond length quadratic elongation
            Data['angle_var']  = []    # angle variance

            if os.path.isfile(args.output):
               continue
            
            # Loop over run directories
            for run in run_dirs:
                with functions.cd(run):
                    
                    # Check that both the LAMMPS trajectory and success files are present
                    if not os.path.isfile(traj_file):
                        print('   ERROR: Missing trajectory file in {}, skipping...'.format(run[2:]))
                        continue
                    if not os.path.isfile(success_file):
                        print('   ERROR: LAMMPS simulation for {} is still running or failed, skipping...'.format(run[2:]))
                        continue
                    
                    coord,bel,box_lo,box_hi,at_indices = read_lmptrj(traj_file,Info['elements'],lmpatom)
                    bond_lqes, angle_vars = eval_octahedra(Info['metal'], Info['halide'], at_indices, traj_file, args.f_start, args.f_end, args.f_every)
                    
                    Data['runs'].append(run[2:])
                    Data['bond_lqe'].append(numpy.mean(numpy.array(bond_lqes)))
                    Data['angle_var'].append(numpy.mean(numpy.array(angle_vars)))
                    
            write_data(Data, fname=args.output)

    print('Finished!')
    return

# Write out data file
def write_data(Data, fname='stability.out'):
    with open(fname, 'w') as o:
        o.write('{:15} {:^15} {:^15}\n'.format('run', 'bond_lqe', 'angle_var'))
        for i,r in enumerate(Data['runs']):
                o.write('{:15} {:< 15.12f} {:< 15.12f}\n'.format(r, Data['bond_lqe'][i], Data['angle_var'][i]))

def eval_octahedra(metals, halides, at_indices, traj_file, f_start, f_end, f_every):
    # Define unit vectors to check halide positions. Defined here to avoid repetition and just passed to each function.
    # order: Top axial position [0], Bottom axial position [1], +x (x1) equitorial position [2],
    #        -x (x2) equitorial position [3], +z (z1) equitorial position [4], -z (z2) equitorial position [5]
    site_vectors = [ (0,1,0), (0,-1,0), (1,0,0), (-1,0,0), (0,0,1), (0,0,-1) ]
    
    bond_lqes    = []
    angle_vars   = []
    
    for geo,ids,types,timestep,box in frame_generator(traj_file, f_start, f_end, f_every, unwrap=False):
        # print 'On timestep: {}'.format(timestep)
        # Data['timesteps'].append(timestep)
                        
        for metal in metals:
            # Collect all target indices
            # all indices correspond to LAMMPS ids (i.e., counting starts at 1. to visualize in VMD, subtract 1)
            for halide in halides:
                if len(at_indices[halide]) == 0:
                    continue
    
                # Take subset of geometries                            
                metal_geos      = deepcopy(numpy.take(geo, at_indices[metal], axis=0))
                halide_geos     = deepcopy(numpy.take(geo, at_indices[halide], axis=0))
                
                # Loop over each metal site: 
                for idx, metal_coord in enumerate(metal_geos):
                    
                    ##
                    # Define perovskite plane by the current metal and its 2 nearest neighbors
                    ##
                    
                    # Get 3 nearest metal atoms
                    dist_metal_metal    = numpy.zeros([len(metal_geos), 2])     # Holds distance, metal index
                    geo_unwraped_metals = numpy.zeros([len(metal_geos), 3])     # Holds distance, metal index
                    for j in range(len(metal_geos)):
                        unwraped_metal_coord = unwrap_pbc(metal_coord, metal_geos[j], box) 
                        geo_unwraped_metals[j] = unwraped_metal_coord
                        dist = calc_dist(metal_coord, unwraped_metal_coord)
                        dist_metal_metal[j] = dist,j
                        
                    # Sort by distance (column 0)            
                    # .argsort() returns an numpy.array of indices that sort the given numpy.array. 
                    # Now call .argsort() on the column to sort, and it will give an array of row indices that sort that particular column to pass as an index to the original array.
                    dist_metal_metal = dist_metal_metal[numpy.argsort(dist_metal_metal[:, 0])]
                    
                    # Define the perovskite plane by the current metal site and its 2 closest metal atoms.
                    # The finding nearest neighbors algorithm above will return the current metal site as the first element in the list (since it's not excluded and its distance is 0),
                    # so just its coordinates as is from the list
                    # Find the vector normal to this plane.            
                    v_1 = geo_unwraped_metals[int(dist_metal_metal[1,1])] - geo_unwraped_metals[int(dist_metal_metal[0,1])]
                    v_2 = geo_unwraped_metals[int(dist_metal_metal[2,1])] - geo_unwraped_metals[int(dist_metal_metal[0,1])]
                    v_n = numpy.cross(v_1, v_2)
                    v_n = v_n / numpy.linalg.norm(v_n)
                    
                    ##
                    # Determine halide sites
                    # if, for whatever reason, a metal site doesn't have all of its halides / they can't be
                    # identified, then skip that site.
                    ##
                    axial_sites      = [0, 0]   # top, bottom
                    equitorial_sites = [0, 0, 0, 0]   # +x, -x, +z, -z
                    
                    found_axial_sites      = [False, False]   # top, bottom
                    found_equitorial_sites = [False, False, False, False]   # +x, -x, +z, -z
                    
                    working_coords = numpy.zeros([11,3])
                    working_coords[0] = deepcopy(metal_coord)
                    
                    # Get 10 nearest halide atoms. Fully occupied it has 6 neighbors, but in some frames there is a transient bond elongation along the apical direction
                    # that the parser may miss. Including additional neighbors ensures that we catch that, since the dot product will weed out any halides on 
                    # adjacent sites.
                    halide_neighbors = numpy.zeros([len(halide_geos), 3])     # Holds distance, index, atom index in sim
                    geo_unwrapped_halides = numpy.zeros([len(halide_geos), 3])     # Holds unwrapped coordinates
                    for i in range(len(halide_geos)):
                        unwrapped_halide_coord = unwrap_pbc(metal_coord, halide_geos[i], box) 
                        geo_unwrapped_halides[i] = unwrapped_halide_coord
                        dist = calc_dist(metal_coord, unwrapped_halide_coord)
                        halide_neighbors[i] = dist,i,at_indices[halide][i]
                    halide_neighbors = halide_neighbors[numpy.argsort(halide_neighbors[:, 0])]
                    
                    for i in range(10):
                        working_coords[i+1] = geo_unwrapped_halides[int(halide_neighbors[i][1])]
                        
                    working_coords -= working_coords[0]             # Center about the metal atom
                    
                    # Loop over the halide positions, compute dot product between its vector and the top axial position vector
                    # use a cutoff to determine position; if dot product value is (+), points towards top axial position, while (-) will point towards bottom axial position.
                    # if dot product is 0 (or near 0), they are orthogonal, i.e. in an equitorial position.
                    # reordered to check all primary axes first, then the off diagonal axial positions
                    # only rounded (originally) the axial positions
                    # not tracking an individual occupancy in the equitorial position, but rather just total occupancy.
                    
                    for wc_i, wc in enumerate(working_coords[1:]):
                        ref_wc = deepcopy(wc)
                        wc     = wc / numpy.linalg.norm(wc)
                
                        for vec_i, vec in enumerate(site_vectors):
                            if vec_i == 0 or vec_i == 1:
                                if round(numpy.dot(wc, vec),4) >= 0.8900:
                                    if vec_i == 0:
                                        axial_sites[0]       = deepcopy(ref_wc)
                                        found_axial_sites[0] = True
                                        break
                                    elif vec_i == 1:
                                        axial_sites[1]       = deepcopy(ref_wc)
                                        found_axial_sites[1] = True
                                        break
                            else:
                                if round(numpy.dot(wc, vec),4) >= 0.8400:
                                    if vec_i == 2:
                                        if calc_dist(working_coords[0], ref_wc) <= 4.5:
                                            equitorial_sites[0]       = deepcopy(ref_wc)
                                            found_equitorial_sites[0] = True
                                            break
                                    elif vec_i == 3:
                                        if calc_dist(working_coords[0], ref_wc) <= 4.5:
                                            equitorial_sites[1]       = deepcopy(ref_wc)
                                            found_equitorial_sites[1] = True
                                            break
                                    elif vec_i == 4:
                                        if calc_dist(working_coords[0], ref_wc) <= 4.5:
                                            equitorial_sites[2]       = deepcopy(ref_wc)
                                            found_equitorial_sites[2] = True
                                            break
                                    elif vec_i == 5:
                                        if calc_dist(working_coords[0], ref_wc) <= 4.5:
                                            equitorial_sites[3]       = deepcopy(ref_wc)
                                            found_equitorial_sites[3] = True
                                            break
    
                    # There are 12 unique 90-degree angles within the perovskite octahedron that we need to check
                    # these are:
                    # apical1/top - axial1, apical1 - axial2, apical1 - axial3, apical1 - axial4
                    # apical2/bottom - axial1, apical2 - axial2, apical2 - axial3, apical2 - axial4
                    # axial1 - axial2, axial1 - axial4, axial3 - axial2, axial3 - axial4
                    # (+x,+z), (+x,-z), (-x,+z), (-x,-z)
                    # store the indices to loop over
                    axial_check_sites = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]  # apical, equitorial
                    equitorial_check_sites = [(0,2),(0,3),(1,2),(1,3)] #equitorial, equitorial
                    
                    Bonds = []
                    Angles = []
                    
                    counter = 0                        
                    for site in axial_check_sites:
                        if found_axial_sites[site[0]] is False or found_equitorial_sites[site[1]] is False:
                            # skip if one of the sites wasn't found
                            continue
                        else:
                            Angles.append( calc_angle( axial_sites[site[0]], working_coords[0], equitorial_sites[site[1]] )*180.0/numpy.pi )
                            
                        counter += 1
                    
                    for site in equitorial_check_sites:
                        if found_equitorial_sites[site[0]] is False or found_equitorial_sites[site[1]] is False:
                            continue
                        else:
                            Angles.append( calc_angle( equitorial_sites[site[0]], working_coords[0], equitorial_sites[site[1]] )*180.0/numpy.pi )
                        counter += 1
                    
                    for i,site in enumerate(found_axial_sites):
                        if site is not False:
                            Bonds.append( calc_bond( working_coords[0], axial_sites[i] ) )
                    
                    for i,site in enumerate(found_equitorial_sites):
                        if site is not False:
                            Bonds.append( calc_bond( working_coords[0], equitorial_sites[i] ) )
                    
                    Bonds = numpy.array(Bonds)
                    Angles = numpy.array(Angles)
                    
                    if len(Bonds) > 0:
                        bond_lqe = 0
                        bond_avg = numpy.mean(Bonds)
                        for b in Bonds:
                            bond_lqe += (b/bond_avg)**2
                        bond_lqe = bond_lqe / 6
                        bond_lqes.append(bond_lqe)
                        
                    if len(Angles) > 0:
                        angle_var = 0
                        for a in Angles:
                            angle_var += (a - 90.0)**2
                        angle_var = angle_var / 11
                        angle_vars.append(angle_var)

    
    return bond_lqes, angle_vars

# Loop for parsing the mapfile information
def parse_map(map_file, atoms_list):

    at_indices = {}
    at_counts  = {}
    for atom in atoms_list:
        at_indices[atom] = []
        at_counts[atom]  = 0
    
    atomtypes  = []
    elements   = []
    masses     = []
    charges    = []
    adj_list   = []
    masses_dict = {}
    
    with open(map_file,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc > 1 and len(fields) > 4:
                atomtypes += [fields[0]]
                elements  += [fields[1]]
                masses    += [float(fields[3])]
                charges   += [float(fields[4])]
                adj_list  += [ [int(_) for _ in fields[5:] ] ]
                
                if fields[1] in atoms_list:
                    at_indices[fields[1]].append(lc-2)
                    at_counts[fields[1]] += 1
                
                if str(fields[3]) not in masses_dict:
                    masses_dict[str(fields[3])] =  fields[1]
                
    return atomtypes, elements, masses, charges, adj_list, masses_dict, at_indices, at_counts


def read_lmpdata(dataname):
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}
    lmpatom = {} # key:lammps atom type(just a number) value:Element
    flag = 0
    with open(dataname,'r') as f:
      for lc,lines in enumerate(f):
         fields = lines.split()
         if len(fields)>0  and fields[0] == 'Masses':
            flag = 1 
            tag = 0
            continue
         if flag == 1:
            if len(fields) == 0: 
               tag += 1
               if tag == 2: 
                  flag = 0
               continue
            mass = float(fields[1]) 
            for key in mass_dict:
               if (abs(mass-mass_dict[key])) < 0.01:
                  ele = key
                  break
            lmpatom[int(fields[0])] = ele
            continue

    return lmpatom 

def read_lmptrj(data_name,atoms_list,lmpatom):
   at_indices = {}
   for atom in atoms_list:
        at_indices[atom] = []

   coord = {}
   vel = {}
   box_lo = {}
   box_hi = {}
   print('reading {}...'.format(data_name),end='\r')
   with open(data_name,'r') as f:
      flag = 0
      for lc,lines in enumerate(f):
         fields = lines.split()
         if flag == 0 and len(fields) < 2: continue
         if fields[0] == 'ITEM:' and fields[1] == 'TIMESTEP':
            flag = 1
            continue

         if fields[0] == 'ITEM:' and fields[1] == 'ATOMS' and fields[2] == 'id':
            flag  = 2
            continue

         if fields[0] == 'ITEM:' and fields[1] == 'BOX' and fields[2] == 'BOUNDS':
            flag  = 3
            box_lo[time]  = []
            box_hi[time] = []
            continue

         if flag == 1:
            time = float(fields[0])
            if len(list(coord.keys())) == 1: 
               break
            coord[time] = {}
            vel[time] = {}
            flag = 0
            continue 
            
         if flag == 2:
            index = int(fields[0])
            atype = int(fields[1])
            for _ in atoms_list:    
               if lmpatom[atype] == _:
                  at_indices[_].append(index-1) 
            coord[time][index] = [ float(i) for i in fields[2:5]]
            vel[time][index] = [ float(i) for i in fields[5:8]]
            continue

         if flag == 3:
            if len(box_lo[time]) == 3: 
               flag = 0
               continue
            box_lo[time].append(fields[0])
            box_hi[time].append(fields[1])
            continue

   return coord,vel,box_lo,box_hi,at_indices 
         

# Generator function that yields the geometry, atomids, and atomtypes of each frame
# with a user specified frequency
def frame_generator(name,start,end,every,unwrap=True,adj_list=None):

    if unwrap is True and adj_list is None:
        print("ERROR in frame_generator: unwrap option is True but no adjacency_list is supplied. Exiting...")
        quit()

    # Parse data for the monitored molecules from the trajectories
    # NOTE: the structure of the molecule based parse is almost identical to the type based parse
    #       save that the molecule centroids and charges are used for the parse
    # Initialize subdictionary and "boxes" sub-sub dictionary (holds the box dimensions for each parsed frame)

    # Parse Trajectories
    frame       = -1                                                  # Frame counter (total number of frames in the trajectory)
    frame_count = -1                                                  # Frame counter (number of parsed frames in the trajectory)
    frame_flag  =  0                                                  # Flag for marking the start of a parsed frame
    atom_flag   =  0                                                  # Flag for marking the start of a parsed Atom data block
    N_atom_flag =  0                                                  # Flag for marking the place the number of atoms should be updated
    atom_count  =  0                                                  # Atom counter for each frame
    box_flag    =  0                                                  # Flag for marking the start of the box parse
    box_count   = -1                                                  # Line counter for keeping track of the box dimensions.

    # Open the trajectory file for reading
    with open(name,'r') as f:

        # Iterate over the lines of the original trajectory file
        for lines in f:

            fields = lines.split()

            # Find the start of each frame and check if it is included in the user-requested range
            if len(fields) == 2 and fields[1] == "TIMESTEP":
                frame += 1
                if frame >= start and frame <= end and (frame-start) % every == 0:
                    frame_flag = 1
                    frame_count += 1
                elif frame > end:
                    break
            # Parse commands for when a user-requested frame is being parsed
            if frame_flag == 1:

                # Header parse commands
                if atom_flag == 0 and N_atom_flag == 0 and box_flag == 0:
                    if len(fields) > 2 and fields[1] == "ATOMS":
                        atom_flag = 1
                        id_ind   = fields.index('id')   - 2
                        type_ind = fields.index('type') - 2
                        x_ind    = fields.index('x')    - 2
                        y_ind    = fields.index('y')    - 2
                        z_ind    = fields.index('z')    - 2
                        continue
                    if len(fields) > 2 and fields[1] == "NUMBER":                        
                        N_atom_flag = 1
                        continue

                    if len(fields) > 2 and fields[1] == "BOX":
                        box      = numpy.zeros([3,2])
                        box_flag = 1
                        continue
                    
                    if len(fields) == 1:
                        timestep = fields[0]
                        continue

                # Update the number of atoms in each frame
                if N_atom_flag == 1:

                    # Intialize total geometry of the molecules being parsed in this frame
                    # Note: from here forward the N_current acts as a counter of the number of atoms that have been parsed from the trajectory.
                    N_atoms     = int(fields[0])
                    geo         = numpy.zeros([N_atoms,3])                    
                    ids         = [ -1 for _ in range(N_atoms) ]
                    types       = [ -1 for _ in range(N_atoms) ]
                    N_atom_flag = 0
                    continue

                # Read in box dimensions
                if box_flag == 1:
                    box_count += 1
                    box[box_count] = [float(fields[0]),float(fields[1])]

                    # After all box data has been parsed, save the box_lengths/2 to temporary variables for unwrapping coordinates and reset flags/counters
                    if box_count == 2:
                        box_count = -1
                        box_flag = 0
                    continue

                # Parse relevant atoms
                if atom_flag == 1:
                    geo[atom_count]   = numpy.array([ float(fields[x_ind]),float(fields[y_ind]),float(fields[z_ind]) ])
                    ids[atom_count]   = int(fields[id_ind])
                    types[atom_count] = int(fields[type_ind])                    
                    atom_count += 1

                    # Reset flags once all atoms have been parsed
                    if atom_count == N_atoms:

                        frame_flag = 0
                        atom_flag  = 0
                        atom_count = 0       

                        # Sort based on ids
                        ids,sort_ind =  list(zip(*sorted([ (k,count_k) for count_k,k in enumerate(ids) ])))
                        geo = geo[list(sort_ind)]
                        types = [ types[_] for _ in sort_ind ]
                        
                        # Upwrap the geometry
                        if unwrap is True:
                            geo = unwrap_geo(geo,adj_list,box)

                        yield geo,ids,types,timestep,box

# Unwrap the PBC between a given ref (3x1 array) and target (3x1) geom. box is actual box dimensions
def unwrap_pbc(ref_coord,target_coord,box):
    bx_2 = ( box[0,1] - box[0,0] ) / 2.0
    by_2 = ( box[1,1] - box[1,0] ) / 2.0
    bz_2 = ( box[2,1] - box[2,0] ) / 2.0
    
    if (target_coord[0] - ref_coord[0])   >  bx_2: target_coord[0] -= (bx_2*2.0) 
    elif (target_coord[0] - ref_coord[0]) < -bx_2: target_coord[0] += (bx_2*2.0) 
    if (target_coord[1] - ref_coord[1])   >  by_2: target_coord[1] -= (by_2*2.0) 
    elif (target_coord[1] - ref_coord[1]) < -by_2: target_coord[1] += (by_2*2.0) 
    if (target_coord[2] - ref_coord[2])   >  bz_2: target_coord[2] -= (bz_2*2.0) 
    elif (target_coord[2] - ref_coord[2]) < -bz_2: target_coord[2] += (bz_2*2.0) 
    
    return target_coord

# Description: Performed the periodic boundary unwrap of the geometry
def unwrap_geo(geo,adj_list,box):

    bx_2 = ( box[0,1] - box[0,0] ) / 2.0
    by_2 = ( box[1,1] - box[1,0] ) / 2.0
    bz_2 = ( box[2,1] - box[2,0] ) / 2.0

    # Unwrap the molecules using the adjacency matrix
    # Loops over the individual atoms and if they haven't been unwrapped yet, performs a walk
    # of the molecular graphs unwrapping based on the bonds. 
    unwrapped = []
    for count_i,i in enumerate(geo):

        # Skip if this atom has already been unwrapped
        if count_i in unwrapped:
            continue

        # Proceed with a walk of the molecular graph
        # The molecular graph is cumulatively built up in the "unwrap" list and is initially seeded with the current atom
        else:
            unwrap     = [count_i]    # list of indices to unwrap (next loop)
            unwrapped += [count_i]    # list of indices that have already been unwrapped (first index is left in place)
            for j in unwrap:

                # new holds the index in geo of bonded atoms to j that need to be unwrapped
                new = [ k for k in adj_list[j] if k not in unwrapped ] 

                # unwrap the new atoms
                for k in new:
                    unwrapped += [k]
                    if (geo[k][0] - geo[j][0])   >  bx_2: geo[k,0] -= (bx_2*2.0) 
                    elif (geo[k][0] - geo[j][0]) < -bx_2: geo[k,0] += (bx_2*2.0) 
                    if (geo[k][1] - geo[j][1])   >  by_2: geo[k,1] -= (by_2*2.0) 
                    elif (geo[k][1] - geo[j][1]) < -by_2: geo[k,1] += (by_2*2.0) 
                    if (geo[k][2] - geo[j][2])   >  bz_2: geo[k,2] -= (bz_2*2.0) 
                    elif (geo[k][2] - geo[j][2]) < -bz_2: geo[k,2] += (bz_2*2.0) 

                # append the just unwrapped atoms to the molecular graph so that their connections can be looped over and unwrapped. 
                unwrap += new

    return geo

def calc_dist(ref,target):
    return math.sqrt((target[0]-ref[0])**2+(target[1]-ref[1])**2+(target[2]-ref[2])**2)

def calc_bond(atom_1,atom_2):
    return numpy.linalg.norm(atom_2 - atom_1)

def calc_angle(atom_1,atom_2,atom_3):
        if numpy.arccos(numpy.dot(atom_1-atom_2,atom_3-atom_2)/(numpy.linalg.norm(atom_1-atom_2)*numpy.linalg.norm(atom_3-atom_2)))<20.0*numpy.pi/180.0:
                print(atom_1,atom_2, atom_3)
                print()
        return numpy.arccos(numpy.dot(atom_1-atom_2,atom_3-atom_2)/(numpy.linalg.norm(atom_1-atom_2)*numpy.linalg.norm(atom_3-atom_2)))
    
if  __name__ == '__main__': 
    main(sys.argv[1:])
