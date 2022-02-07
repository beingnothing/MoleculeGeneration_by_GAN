import os
from rdkit import Chem
import numpy as np

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    paths = []
    root = './dude'
    for subdir in os.listdir(root):
        # site_path = os.path.join(root, subdir, 'receptor.pdb')
        lig_path = os.path.join(root, subdir, 'actives_final.sdf')
        smi_path = os.path.join(root, subdir, 'actives_final.ism')
        mols_dir = os.path.join("./dude", subdir, 'actives_final_clusterd')
        mkdirs(mols_dir)
        with open(smi_path, 'r') as f:
            lines = f.read().splitlines()
            smi_ids = [line.split()[-1] for line in lines]
        mols = Chem.SDMolSupplier(lig_path)
        mols_dic = {}
        for mol in mols:
            mols_dic[mol.GetProp("_Name")] = mol
        clusterd_mols = [mols_dic[smi_id] for smi_id in smi_ids]
        for mol in clusterd_mols:
            mol_path = os.path.join(mols_dir, mol.GetProp("_Name") + '.mol')
            Chem.MolToMolFile(mol, mol_path)
