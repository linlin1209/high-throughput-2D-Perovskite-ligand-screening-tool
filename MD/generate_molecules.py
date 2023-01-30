import sys,argparse,subprocess,os,time,math,shutil
from openbabel import pybel
from openbabel import openbabel as ob
from rdkit import Chem
from copy import deepcopy
import itertools


def main(argv):

   all_xyz_smiles = []
   ### For Monomers
   ATOM_PROP_ATOM_LABEL = 'atomLabel'

   bodies_smi = ["[*]C1=CC=CC=C1 |$L1$|",
   "[*]C1=CC=CO1 |$L1$|",
   "[*]C1=CC=CS1 |$L1$|",
   "[*]C1=NC=CS1 |$L1$|",
   "[*]C1=NC=CN1 |$L1$|",
   "[*]C1=CN=CC=C1 |$L1$|",
   "[*]C1=NC=CO1 |$L1$|",
   "[*]C1=CC(SC=C2)=C2S1 |$L1$|",
   "[*]C1=CC(C=CS2)=C2S1 |$L1$|",
   "O=C(N1)C2=C([*])SC=C2C1=O |$;;;;;L1$|",
   "O=C1C(C2=CN1)=C([*])NC2=O |$;;;;;;;L1$|",
   "[*]C1=CC=CC2=NSN=C21 |$L1$|",
   "[*]C(C=C1)=CC2=C1C3=C(C2)C=CC=C3 |$L1$|",
   "[*]C1=CC(C=C(OC=C2)C2=C3)=C3O1 |$L1$|",
   "[*]C1=CC(C=C(SC=C2)C2=C3)=C3S1 |$L1$|",
   "[*]C(S1)=CC2=C1C3=CC(C4)=C(C=C3C2)C5=C4C=CS5 |$L1$|"]

   anchor_smiles = {
       "L1" : ["[NH3+]C[*]", "[NH3+]CC[*]","[NH3+]CCC[*]","[NH3+]C/C=C/[*]","[NH3+]CCO[*]","[NH3+]CCCO[*]"]
   }

   all_products_smi = []
   for i in bodies_smi:
       tmp_mol, tmp_smi = rgroup_enumerate(i,anchor_smiles)
       all_products_smi += tmp_smi

   all_xyz_smiles += all_products_smi

   with open('monomers.smi','w') as f:
       for i in all_products_smi:
           f.write("{}\n".format(i))

   ### For Dimers
   bodies_smi = [
   "[*]C1=CC=C([*])C=C1 |$L1;;;;;L2$|",
   "[*]C1=CC=C([*])O1 |$L1;;;;;L2$|",
   "[*]C1=CC=C([*])S1 |$L1;;;;;L2$|",
   "[*]C1=NC=C([*])S1 |$L1;;;;;L2$|",
   "[*]C1=NC=C([*])N1 |$L1;;;;;L2$|",
   "[*]C1=NC=C([*])C=C1 |$L1;;;;;L2$|",
   "[*]C1=CN=C([*])O1 |$L1;;;;;L2$|",
   "[*]C1=CC(SC([*])=C2)=C2S1 |$L1;;;;;;L2$|",
   "[*]C1=CC(C=C([*])S2)=C2S1 |$L1;;;;;;L2$|",
   "O=C(N1)C2=C([*])SC([*])=C2C1=O |$;;;;;L1;;;L2$|",
   "O=C1C(C2=C([*])N1)=C([*])NC2=O |$;;;;;L1;;;L2$|",
   "[*]C1=CC=C([*])C2=NSN=C21 |$L1;;;;;L2$|",
   "[*]C1=CC(CC2=C3C=CC([*])=C2)=C3C=C1 |$L1;;;;;;;;;;L2$|",
   "[*]C1=CC(C=C(OC([*])=C2)C2=C3)=C3O1 |$L1;;;;;;;;L2$|",
   "[*]C1=CC(C=C(SC([*])=C2)C2=C3)=C3S1 |$L1;;;;;;;;L2$|",
   "[*]C1=CC2=C(C(C=C(CC3=C4SC([*])=C3)C4=C5)=C5C2)S1 |$L1;;;;;;;;;;;;;L2$|"
   ]

   second_bodies_smi = ["[*]C1=CC=CC=C1",
   "[*]C1=CC=CO1",
   "[*]C1=CC=CS1",
   "[*]C1=NC=CS1",
   "[*]C1=NC=CN1",
   "[*]C1=CN=CC=C1",
   "[*]C1=NC=CO1",
   "[*]C1=CC(SC=C2)=C2S1",
   "[*]C1=CC(C=CS2)=C2S1",
   "O=C(N1)C2=C([*])SC=C2C1=O",
   "O=C1C(C2=CN1)=C([*])NC2=O",
   "[*]C1=CC=CC2=NSN=C21",
   "[*]C(C=C1)=CC2=C1C3=C(C2)C=CC=C3",
   "[*]C1=CC(C=C(OC=C2)C2=C3)=C3O1",
   "[*]C1=CC(C=C(SC=C2)C2=C3)=C3S1",
   "[*]C(S1)=CC2=C1C3=CC(C4)=C(C=C3C2)C5=C4C=CS5"]

   anchor_smiles = {
       "L1" : ["[NH3+]C[*]", "[NH3+]CC[*]","[NH3+]CCC[*]","[NH3+]C/C=C/[*]","[NH3+]CCO[*]","[NH3+]CCCO[*]"],
       "L2":second_bodies_smi
   }

   all_products_smi = []
   for i in bodies_smi:
       tmp_mol, tmp_smi = rgroup_enumerate(i,anchor_smiles)
       all_products_smi += tmp_smi

   all_xyz_smiles += all_products_smi

   with open('dimer.smi','w') as f:
       for i in all_products_smi:
           f.write("{}\n".format(i))

   ### For 4mers
   ligand_candidates = [
       "[Na]C1=CC=C([Cs])C=C1[K]",
       "[Na]C1=C([K])C=C([Cs])O1",
       "[Na]C1=C([K])C=C([Cs])S1",
       "[Na]C1=NC([K])=C([Cs])S1",
       "[Na]C1=NC([K])=C([Cs])N1",
       "[Cs]C1=NC=C([Na])C([K])=C1",
       "[Cs]C1=C([K])N=C([Na])O1",
       "[Na]C1=C([K])C(SC([Cs])=C2)=C2S1",
       "[Na]C1=C([K])C(C=C([Cs])S2)=C2S1",
       "[Na]C1=C([K])C=C([Cs])C2=NSN=C21",
       "[Cs]C1=CC(C([K])C2=C3C=CC([Na])=C2)=C3C=C1",
       "[Na]C1=CC(C=C(OC([Cs])=C2)C2=C3[K])=C3O1",
       "[Na]C1=CC(C=C(SC([Cs])=C2)C2=C3[K])=C3S1",
       "[Cs]C1=CC2=C(C(C=C(C([K])C3=C4SC([Na])=C3)C4=C5)=C5C2)S1",
       "O=C(NC([Na])=C12)C1=C([Cs])NC2=O",
       "O=C(N1)C2=C([Na])SC([Cs])=C2C1=O"
   ]


   anchor_candidates = ["[NH3+]C[*]", "[NH3+]CC[*]","[NH3+]CCC[*]","[NH3+]C/C=C/[*]","[NH3+]CCO[*]","[NH3+]CCCO[*]"]
   side_chain_candidates = ["*[H]","*C","*CC","*F","*OC","C#N"]

   input_mols = []
   for i in side_chain_candidates:
       for j in anchor_candidates:
           for k in ligand_candidates:
               for l in ligand_candidates:
                   tmp_input = (Chem.MolFromSmiles(j),Chem.MolFromSmiles(k),Chem.MolFromSmiles(l),Chem.MolFromSmiles(i))
                   input_mols.append(tmp_input)
                   
   all_products_smi = []
   for count_i,i in enumerate(input_mols):
       product, product_smi = _combine_mol(i[0],i[1],i[2],i[3])
       all_products_smi.append(product_smi)

   all_xyz_smiles += all_products_smi
       
   with open('4mer.smi','w') as f:
       for i in all_products_smi:
           f.write("{}\n".format(i))

   for count_i,i in enumerate(all_xyz_smiles):
      xyzname = '{}.xyz'.format(count_i)
      smi2xyz(i,xyzname,opt_option=True)

def _combine_core_and_rgroups(core, rgroups):
    ATOM_PROP_ATOM_LABEL = 'atomLabel'
    """
    Helper function for rgroup enumeration
    """
    product_mols = []
    rgroup_names = rgroups.keys()
    product = Chem.RWMol(core)
    remove_count = 0
    for at in core.GetAtoms():
        if at.HasProp(ATOM_PROP_ATOM_LABEL):
            label = at.GetProp(ATOM_PROP_ATOM_LABEL)
            if label not in rgroup_names:
                continue

            # find where the rgroup will attach to the existing
            # product
            attach_idx = None
            remove_idx = None
            rgroup = Chem.RWMol(rgroups[label])
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 0:
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        # attach_idx will go down by 1
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            if attach_idx is None:
                raise ValueError("Invalid rgroup provided")

            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)

    # clean labeled atoms that has been replaced
    product_clean = Chem.RWMol(product)
    remove_count = 0
    for at in product.GetAtoms():
        if at.HasProp(ATOM_PROP_ATOM_LABEL):
            label = at.GetProp(ATOM_PROP_ATOM_LABEL)
            if label not in rgroup_names:
                continue
            product_clean.RemoveAtom(at.GetIdx()-remove_count)
            remove_count += 1
    product_clean_smi = Chem.MolToSmiles(product_clean)
    product_clean_smi = product_clean_smi.replace('~','-')
    return product_clean, product_clean_smi

def _combine_mol(anchor,ligand_A,ligand_B,side_chain):
    ATOM_PROP_ATOM_LABEL = 'atomLabel'
    """
    Helper function for rgroup enumeration
    """
    
    # first remove side chain tag atom on ligand B
    clean_ligand_B = Chem.RWMol(ligand_B)
    for at in ligand_B.GetAtoms():
        if at.GetAtomicNum() == 19:
            clean_ligand_B.RemoveAtom(at.GetIdx())
            break
    
    
    product = Chem.RWMol(ligand_A)
    # attach anchor
    for at in ligand_A.GetAtoms():
        if at.GetAtomicNum() == 55:

            
            # find where the anchor will attach to the existing
            # product
            attach_idx = None
            remove_idx = None
            rgroup = Chem.RWMol(anchor)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 0:
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        # attach_idx will go down by 1
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            if attach_idx is None:
                raise ValueError("Invalid rgroup provided")

            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())

    # attach side chains
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 19:

            
            # find where the anchor will attach to the existing
            # product
            attach_idx = None
            remove_idx = None
            rgroup = Chem.RWMol(side_chain)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 0:
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        # attach_idx will go down by 1
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            if attach_idx is None:
                raise ValueError("Invalid rgroup provided")

            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())
    # attach ligand B
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 11:

            
            # find where the anchor will attach to the existing
            # product
            attach_idx = None
            remove_idx = None
            rgroup = Chem.RWMol(clean_ligand_B)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 11:
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        # attach_idx will go down by 1
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            if attach_idx is None:
                raise ValueError("Invalid rgroup provided")

            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())
    # attach ligand A again
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 55:

            
            # find where the anchor will attach to the existing
            # product
            attach_idx = None
            remove_idx = None
            rgroup = Chem.RWMol(ligand_A)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 11:
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        # attach_idx will go down by 1
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            if attach_idx is None:
                raise ValueError("Invalid rgroup provided")

            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())
    # attach side chains
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 19:

            
            # find where the anchor will attach to the existing
            # product
            attach_idx = None
            remove_idx = None
            rgroup = Chem.RWMol(side_chain)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 0:
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        # attach_idx will go down by 1
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            if attach_idx is None:
                raise ValueError("Invalid rgroup provided")

            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())
    # attach ligand B again
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 55:

            
            # find where the anchor will attach to the existing
            # product
            attach_idx = None
            remove_idx = None
            rgroup = Chem.RWMol(clean_ligand_B)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 55:
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        # attach_idx will go down by 1
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            if attach_idx is None:
                raise ValueError("Invalid rgroup provided")

            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())            
    # attach Get rid of Na at the very end
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 11:
            product.RemoveAtom(at.GetIdx())
            break
            
    product_smi = Chem.MolToSmiles(product)
    product_smi = product_smi.replace('~','-')
    return product, product_smi


def rgroup_enumerate(core_smi,rgroups):
    #print(rgroups_smiles)
    keys, values = zip(*rgroups.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    all_products = []
    all_products_smi = []
    for i in permutations_dicts:
        rgroups = {key:Chem.MolFromSmiles(i[key]) for key in i}
        product_mol, product_smi = _combine_core_and_rgroups(Chem.MolFromSmiles(core_smi),rgroups)
        all_products.append(product_mol)
        all_products_smi.append(product_smi)
    return all_products, all_products_smi # in mol format

def smi2xyz(smiles,xyzname,opt_option=True):
      if opt_option:
         # read in smiles, add H, and minimize the structure
         mol = pybel.readstring("smi", smiles)
         pybel._builder.Build(mol.OBMol)
         mol.addh()
         globalopt(mol)
          
         # Write out xyz file
         xyz = pybel.Outputfile("xyz", xyzname, overwrite=True)
         xyz.write(mol)
         xyz.close()

      else:
         subprocess.call("obabel -:\"{}\" -oxyz -O {} --gen3d".format(smiles,xyzname),shell=True)

      return
   

# Use openbabel to do a FF-based geometry minimization and return the most likely conformer
def globalopt(mol, debug=False, fast=False):

    # Initialize forcefield
    ff = pybel._forcefields["mmff94"]
    success = ff.Setup(mol.OBMol)
    if not success:
        ff = pybel._forcefields["uff"]
        success = ff.Setup(mol.OBMol)
        if not success:
            sys.exit("Cannot set up forcefield")

    if debug:
        ff.GetCoordinates(mol.OBMol)
        mol.write("sdf", "1.sdf", overwrite=True)

    # Initial structure minimization
    if fast:
        ff.SteepestDescent(50, 1.0e-3)
    else:
        ff.SteepestDescent(500, 1.0e-4)
    if debug:
        ff.GetCoordinates(mol.OBMol)
        mol.write("sdf", "2.sdf", overwrite=True)
    
    # Find lowest-energy conformer
    if fast:
        ff.WeightedRotorSearch(20, 5)
    else:
        ff.WeightedRotorSearch(100, 20)
    if debug:
        ff.GetCoordinates(mol.OBMol)
        mol.write("sdf", "3.sdf", overwrite=True)

    # Final minimization
    if fast:
        ff.ConjugateGradients(50, 1.0e-4)
    else:
        ff.ConjugateGradients(500, 1.0e-6)

    if debug:
        ff.GetCoordinates(mol.OBMol)
        mol.write("sdf", "4.sdf", overwrite=True)

    return



if __name__ == "__main__":
   main(sys.argv[1:])
