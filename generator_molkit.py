from moleculekit.molecule import Molecule, calculateUniqueBonds
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures, getChannels, _getOccupancyC, _getGridCenters
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping, metal_atypes
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.home import home
from moleculekit.util import uniformRandomRotation

from rdkit import Chem
from rdkit.Chem import AllChem
import math
import os
import numpy as np
import multiprocessing
import torch

vocab_list = ["pad", "start", "end",
              "C", "c", "N", "n", "S", "s", "P", "O", "o",
              "B", "F", "I",
              "Cl", "[nH]", "Br", # "X", "Y", "Z",
              "1", "2", "3", "4", "5", "6",
              "#", "=", "-", "(", ")"  # Misc
]

vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}

vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}


resolution = 1.
lig_size = 24
lig_N = [lig_size, lig_size, lig_size]
lig_bbm = (np.zeros(3) - float(lig_size * 1. / 2))
lig_nvoxels = np.ceil(lig_N).astype(int)
lig_global_centers = _getGridCenters(*list(lig_nvoxels), resolution) + lig_bbm

protein_size = 24
protein_N = [protein_size, protein_size, protein_size]
protein_bbm = (np.zeros(3) - float(protein_size * 1. / 2))
protein_nvoxels = np.ceil(protein_N).astype(int)
protein_global_centers = _getGridCenters(*list(protein_nvoxels), resolution) + protein_bbm


def molecule_process(in_smile):
    try:
        m = Chem.MolFromSmiles(in_smile)
        mh = Chem.AddHs(m)
        AllChem.EmbedMolecule(mh)
        Chem.AllChem.MMFFOptimizeMolecule(mh)
        m = Chem.RemoveHs(mh)
        mol = SmallMol(m)
        #mol = SmallMol(in_smile)
        return mol
    except:
        return None


def rotate(coords, rotMat, center=(0,0,0)):
    """
    Rotate a selection of atoms by a given rotation around a center
    """
    newcoords = coords - center
    return np.dot(newcoords, np.transpose(rotMat)) + center


def voxelize(x, lig=None, format='smi', displacement=2., rotation=True):
    assert format == 'smi' or format == 'protein'
    if format == 'smi':
        # mol = molecule_process(x)
        # if not mol:
        #     return None
        # mol = SmallMol()
        mol = x
        channels = getChannels(mol)[0][:, [0, 1, 2, 3, 7]]
        coords = mol.get('coords')
        center = mol.getCenter()
        # Do the rotation
        if rotation:
            rrot = uniformRandomRotation()  # Rotation
            coords = rotate(coords, rrot, center=center)

        # Do the translation
        center = center + (np.random.rand(3) - 0.5) * 2 * displacement

        centers2D = lig_global_centers + center
        occupancy = _getOccupancyC(coords.astype(np.float32),
                                   centers2D.reshape(-1, 3),
                                   channels).reshape(lig_size, lig_size, lig_size, 5)
        # vox, centers, N = getVoxelDescriptors(mol, userchannels=channels, buffer=1)
    else:
        # protein = Molecule(x)
        protein = x

        protein.filter("protein or ions and (not resname CL) and (not resname NA)")
        # protein.filter("protein or water or element {}".format(" ".join(metal_atypes)))
        # protein.filter('protein')

        # protein = prepareProteinForAtomtyping(protein, segment=False)
        protein = prepareProteinForAtomtyping(protein)

        #protein.bonds = protein._getBonds()
        #protein.bonds, protein.bondtype = calculateUniqueBonds(protein.bonds, protein.bondtype)

        coords = lig.get('coords')
        center = lig.getCenter()
        features, center, N = getVoxelDescriptors(protein, boxsize=[protein_size, protein_size, protein_size], center=center, buffer=1)
        
        # channels = getChannels(protein)[0][:, [0, 1, 2, 3, 4, 5, 7]]
        # coords = protein.get('coords')
        # center = np.mean(coords, axis=0)

        # Do the rotation
        # if rotation:
        #     rrot = uniformRandomRotation()  # Rotation
        #     coords = rotate(coords, rrot, center=center)

        # Do the translation
        # center = center + (np.random.rand(3) - 0.5) * 2 * displacement
        # centers2D = protein_global_centers + center

        # occupancy = _getOccupancyC(coords.astype(np.float32),
        #                            centers2D.reshape(-1, 3),
        #                            channels).reshape(protein_size, protein_size, protein_size, 7)
        # vox, centers, N = getVoxelDescriptors(protein, userchannels=channels, buffer=1)
        occupancy = features[:, [0, 1, 2, 3, 4, 5, 7]].reshape(protein_size, protein_size, protein_size, 7)
    return occupancy.astype(np.float32).transpose(3, 0, 1, 2,)


def shape_representation(file_pair):
    pro_file, smi_file, smi_str = file_pair
    try:
        protein = Molecule(pro_file)
    except:
        return None
    mol = molecule_process(smi_str)
    try:
        lig = SmallMol(smi_file)
    except:
        try:
            lig = SmallMol(smi_file.split('.')[1] + '.mol2')
        except:
            return None
    try:
        protein_vox = voxelize(x=protein, lig=lig, format='protein')
        smi_vox = voxelize(x=mol, format='smi')
    except:
        return None
    return protein_vox, smi_vox


def caption_representation(smile_index):
    smile_str = list(smile_index)
    end_token = smile_str.index(2)
    smile_str = "".join([vocab_i2c_v1[i] for i in smile_str[1:end_token]])

    mol = molecule_process(smile_str)
    if not mol:
        return None
    smi_vox = voxelize(mol, format='smi')
    return torch.Tensor(smi_vox), torch.Tensor(smile_index), end_token + 1


def shape_gather_fn(inputs):
    # batch_size = len(inputs)
    # z = np.random.normal(0, 1, [batch_size, 1])
    pro = [item[0] for item in inputs]
    smi = [item[1] for item in inputs]
    return torch.Tensor(pro), torch.Tensor(smi)


def caption_gather_fn(inputs):
    inputs.sort(key=lambda x: x[2], reverse=True)
    images, smiles, lengths = zip(*inputs)
    images = torch.stack(images, 0)  # Stack images

    # Merge smiles (from tuple of 1D tensor to 2D tensor).
    # lengths = [len(smile) for smile in smiles]
    targets = torch.zeros(len(smiles), max(lengths)).long()
    for i, smile in enumerate(smiles):
        end = lengths[i]
        targets[i, :end] = smile[:end]
    return images, targets, lengths


class Batch_prep:
    def __init__(self, n_proc=6, mp_pool=None):
        # pass
        if mp_pool:
            self.mp = mp_pool
        elif n_proc > 1:
            self.mp = multiprocessing.Pool(n_proc)
        else:
            raise NotImplementedError("Use multiprocessing for now!")

    def transform_data(self, data, mode='shape'):
        assert mode == 'shape' or mode == 'caption'
        if mode == 'caption':
            inputs = self.mp.map(caption_representation, data)
            # inputs = list(map(caption_representation, data))

            inputs = list(filter(lambda x: x is not None, inputs))
            return caption_gather_fn(inputs)
        elif mode == 'shape':
            inputs = self.mp.map(shape_representation, data)
            # inputs = list(map(shape_representation, data))
            
            inputs = list(filter(lambda x: x is not None, inputs))
            return shape_gather_fn(inputs)
        else:
            raise ValueError


def queue_datagen(inputs, batch_size=128, n_proc=12, mp_pool=None, mode='shape'):
    n_batches = int(math.ceil(len(inputs) / batch_size))
    sh_indencies = np.arange(len(inputs))

    my_batch_prep = Batch_prep(n_proc=n_proc, mp_pool=mp_pool)

    while True:
        np.random.shuffle(sh_indencies)
        for i in range(n_batches):
            batch_idx = sh_indencies[i * batch_size:(i + 1) * batch_size]
            yield my_batch_prep.transform_data(inputs[batch_idx], mode)


if __name__ == '__main__':
    paths = []
    root = './bicycle_gan/data'
    for subdir in os.listdir(root):
        if subdir == 'refined-set': continue
        site_path = os.path.join(root, subdir, 'site.mol2')
        lig_path = os.path.join(root, subdir, 'lig.pdb')
        paths.append([site_path, lig_path])
    multiproc = multiprocessing.Pool(1)
    for i, item in enumerate(queue_datagen(np.array(paths), batch_size=2, n_proc=2, mp_pool=multiproc, mode='shape')):
        print(item)




