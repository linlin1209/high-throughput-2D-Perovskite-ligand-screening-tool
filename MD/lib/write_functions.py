#!/bin/env python
# Author: Brett Savoie (brettsavoie@gmail.com)

from numpy import *
from id_types import *


# Description: Simple wrapper function for writing xyz file
#
# Inputs      name:     string holding the filename of the output
#             elements: list of element types (list of strings)
#             geo:      Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
# Returns     None
#
def xyz_write(name,elements,geo,append_opt=False):

    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'
        
    with open(name,open_cond) as f:
        f.write('{}\n\n'.format(len(elements)))
        for count_i,i in enumerate(elements):
            f.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i,geo[count_i][0],geo[count_i][1],geo[count_i][2]))
    return 

# Description: Simple wrapper function for writing a mol (V2000) file
#
# Inputs      name:     string holding the filename of the output
#             elements: list of element types (list of strings)
#             geo:      Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#             adj_mat:  NxN array holding the molecular graph
#
# Returns     None
#
def mol_write(name,elements,geo,adj_mat,append_opt=False):

    # Consistency check
    if len(elements) >= 1000:
        print("ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return 

    # Check for append vs overwrite condition
    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'

    # Parse the hybridizations so that the bond order can be written
    hybridizations = Hybridization_finder(elements,adj_mat)

    # Parse the basename for the mol header
    base_name = name.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    # Get the bond orders
    bond_mat = get_bonds(elements,adj_mat)    

    # Write the file
    with open(name,open_cond) as f:

        # Write the header
        f.write('{}\nGenerated by mol_write.py\n\n'.format(base_name))

        # Write the number of atoms and bonds
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(len(elements),int(sum(adj_mat/2.0))))

        # Write the geometry
        for count_i,i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n".format(geo[count_i][0],geo[count_i][1],geo[count_i][2],i))

        # Write the bonds
        bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 and count_j > count_i ] 
        for i in bonds:

            # Calculate bond order from the bond_mat
            bond_order = int(bond_mat[i[0],i[1]])

#             if hybridizations[i[0]] == "sp2" and hybridizations[i[1]] == "sp2":
#                 bond_order = 2
#             elif hybridizations[i[0]] == "sp" and hybridizations[i[1]] == "sp":
#                 bond_order = 3
#             else:
#                 bond_order = 1
                
            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i[0]+1,i[1]+1,bond_order))
        f.write("M  END\n$$$$\n")

    return 

# Description: Simple wrapper function for writing a pdb file
#
# Inputs      name:     string holding the filename of the output
#             elements: list of element types (list of strings)
#             geo:      Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#             adj_mat:  NxN array holding the molecular graph
#
# Returns     None
def pdb_write(name,elements,geo,adj_mat,append_opt=False):

    # Check for append vs overwrite condition
    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'

    # Parse the basename for the mol header
    base_name = name.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    # Write the file
    with open(name,open_cond) as f:

        # Write the header
        f.write('COMPND    {}\n'.format(base_name))
        f.write('AUTHOR    Generated by pdb_write.py\n')

        # Write the geometry
        for count_i,i in enumerate(elements):
            f.write("HETATM {:>4d} {:>2s} {:>5s} {:>5d}      {:> 6.3f}  {:> 6.3f}  {:> 6.3f}  1.00  0.00 {:>11s}\n".format(count_i+1,i.upper(),"LIG",1,geo[count_i][0],geo[count_i][1],geo[count_i][2],i))

        # Write the bonds
        con_count = 0
        for count_i,i in enumerate(adj_mat):
            cons = sorted([ count_j+1 for count_j,j in enumerate(i) if j == 1 ])
            if cons > 0:
                for j in range(len(cons)):                
                    if j % 4 == 0:
                        if j == 0:
                            f.write("CONECT {:>4d}".format(count_i+1))
                        else:
                            f.write("\nCONECT {:>4d}".format(count_i+1))
                    f.write(" {:>4d}".format(cons.pop(0)))
                    con_count += 1
                f.write("\n")


        f.write("MASTER        0    0    0    0    0    0    0    0 {:>4d} {:>4d} {:>4d} {:>4d}\n".format(len(elements),0,con_count,0))
        f.write("END\n")

    return 
