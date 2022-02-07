# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

from htmd.molecule.util import uniformRandomRotation
from htmd.smallmol.smallmol import SmallMol
from htmd.molecule.voxeldescriptors import _getOccupancyC, _getGridCenters, _getAtomtypePropertiesPDBQT

from htmd.builder.preparation import proteinPrepare
from htmd.molecule.molecule import Molecule

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import math
import random
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
size = 24
N = [size, size, size]
bbm = (np.zeros(3) - float(size * 1. / 2))
global_centers = _getGridCenters(bbm, N, resolution)

protein_size = 24
protein_N = [protein_size, protein_size, protein_size]
protein_bbm = (np.zeros(3) - float(protein_size * 1. / 2))
protein_global_centers = _getGridCenters(protein_bbm, protein_N, resolution)


def string_gen_V1(in_string):
    out = in_string.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")
    return out


def tokenize_v1(in_string, return_torch=True):
    caption = []
    caption.append(0)
    caption.extend([vocab_c2i_v1[x] for x in in_string])
    caption.append(1)
    if return_torch:
        return torch.Tensor(caption)
    return caption


def get_aromatic_groups(in_mol):
    """
    Obtain groups of aromatic rings
    """
    groups = []
    ring_atoms = in_mol.GetRingInfo().AtomRings()
    for ring_group in ring_atoms:
        if all([in_mol.GetAtomWithIdx(x).GetIsAromatic() for x in ring_group]):
            groups.append(ring_group)
    return groups


def generate_representation(in_smile):
    """
    Makes embeddings of Molecule.
    """
    try:
        m = Chem.MolFromSmiles(in_smile)
        mh = Chem.AddHs(m)
        AllChem.EmbedMolecule(mh)
        Chem.AllChem.MMFFOptimizeMolecule(mh)
        m = Chem.RemoveHs(mh)
        mol = SmallMol(m)
        return mol
    except:  # Rarely the conformer generation fails
        return None


def generate_sigmas(mol):
    """
    Calculates sigmas for elements as well as pharmacophores.
    Returns sigmas, coordinates and center of ligand.
    """
    coords = mol.getCoords()
    n_atoms = len(coords)
    lig_center = mol.getCenter()

    # Calculate all the channels
    multisigmas = mol._getChannelRadii()[:, [0, 1, 2, 3, 7]]

    aromatic_groups = get_aromatic_groups(mol._mol)
    aromatics = [coords[np.array(a_group)].mean(axis=0) for a_group in aromatic_groups]
    aromatics = np.array(aromatics)
    if len(aromatics) == 0:  # Make sure the shape is correct
        aromatics = aromatics.reshape(aromatics.shape[0], 3)

    # Generate the pharmacophores
    aromatic_loc = aromatics + (np.random.rand(*aromatics.shape) - 0.5)

    acceptor_ph = (multisigmas[:, 2] > 0.01)
    donor_ph = (multisigmas[:, 3] > 0.01)

    # Generate locations
    acc_loc = coords[acceptor_ph]
    acc_loc = acc_loc + (np.random.rand(*acc_loc.shape) - 0.5)
    donor_loc = coords[donor_ph]

    donor_loc = donor_loc + (np.random.rand(*donor_loc.shape) - 0.5)
    coords = np.vstack([coords, aromatic_loc, acc_loc, donor_loc])

    final_sigmas = np.zeros((coords.shape[0], 8))
    final_sigmas[:n_atoms, :5] = multisigmas
    pos1 = n_atoms + len(aromatic_loc)  # aromatics end

    final_sigmas[n_atoms:(pos1), 5] = 2.
    pos2 = pos1 + len(acc_loc)
    final_sigmas[pos1:pos2, 6] = 2.
    final_sigmas[pos2:, 7] = 2.

    return final_sigmas, coords, lig_center


def rotate(coords, rotMat, center=(0,0,0)):
    """
    Rotate a selection of atoms by a given rotation around a center
    """

    newcoords = coords - center
    return np.dot(newcoords, np.transpose(rotMat)) + center


def voxelize(x, format='smi', displacement=2., rotation=True):
    """
    Generates molecule or protein representation.
    """
    assert format == 'smi' or format == 'protein'
    if format == 'smi':
        mol = x
        coords = mol.getCoords()
        #n_atoms = len(coords)
        center = mol.getCenter()
        channels = mol._getChannelRadii()[:, [0, 1, 2, 3, 7]]
        # Do the rotation
        if rotation:
            rrot = uniformRandomRotation()  # Rotation
            coords = rotate(coords, rrot, center=center)

        # Do the translation
        center = center + (np.random.rand(3) - 0.5) * 2 * displacement

        centers2D = global_centers + center
        occupancy = _getOccupancyC(coords.astype(np.float32),
                                centers2D.reshape(-1, 3),
                                channels).reshape(size, size, size, 5)
    else:
        protein = x

        protein.filter("protein or ions and (not resname CL) and (not resname NA)")
        #protein = prepareProteinForAtomtyping(protein, segment=False)
        protein = proteinPrepare(protein)

        #channels = getChannels(protein)[0][:, [0, 1, 2, 3, 4, 5, 7]]
        channels = _getAtomtypePropertiesPDBQT(protein)[:, [0, 1, 2, 3, 4, 5, 7]]
        coords = protein.get('coords')
        center = np.mean(coords, axis=0)

        # Do the rotation
        if rotation:
            rrot = uniformRandomRotation()  # Rotation
            coords = rotate(coords, rrot, center=center)

        # Do the translation
        center = center + (np.random.rand(3) - 0.5) * 2 * displacement

        centers2D = protein_global_centers + center
        occupancy = _getOccupancyC(coords.astype(np.float32),
                                   centers2D.reshape(-1, 3),
                                   channels).reshape(protein_size, protein_size, protein_size, 7)
    return occupancy.astype(np.float32).transpose(3, 0, 1, 2,)


def shape_representation(file_pair):
    pro_file, smi_str = file_pair
    protein = Molecule(pro_file)
    # smi = SmallMol(smi_file)
    mol = generate_representation(smi_str)
    try:
        protein_vox = voxelize(protein, format='protein')
        smi_vox = voxelize(mol, format='smi')
    except:
        return None
    return protein_vox, smi_vox


def caption_representation(smile_index):
    smile_str = list(smile_index)
    end_token = smile_str.index(2)
    smile_str = "".join([vocab_i2c_v1[i] for i in smile_str[1:end_token]])

    mol = generate_representation(smile_str)
    if not mol:
        return None
    smi_vox = voxelize(mol, format='smi')
    return torch.Tensor(smi_vox), torch.Tensor(smile_index), end_token + 1


def generate_representation_v1(smile):
    """
    Generate voxelized and string representation of a molecule
    """
    # Convert smile to 3D structure

    smile_str = list(smile)
    end_token = smile_str.index(2)
    smile_str = "".join([vocab_i2c_v1[i] for i in smile_str[1:end_token]])

    mol = generate_representation(smile_str)
    if mol is None:
        return None

    # Generate sigmas
    #sigmas, coords, lig_center = generate_sigmas(mol)
    #vox = voxelize(sigmas, coords, lig_center)
    vox = voxelize(mol)

    return torch.Tensor(vox), torch.Tensor(smile), end_token + 1


def shape_gather_fn(inputs):
    # batch_size = len(inputs)
    # z = np.random.normal(0, 1, [batch_size, 1])
    pro = [item[0] for item in inputs]
    smi = [item[1] for item in inputs]
    return torch.Tensor(pro), torch.Tensor(smi)

def caption_gather_fn(inputs):
    """
    Collects and creates a batch.
    """
    # Sort a data list by smiles length (descending order)
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
        if mp_pool:
            self.mp = mp_pool
        elif n_proc > 1:
            self.mp = multiprocessing.Pool(n_proc)
        else:
            pass
            #raise NotImplementedError("Use multiprocessing for now!")

    def transform_data(self, data, mode='shape'):
        assert mode == 'shape' or mode == 'caption'
        if mode == 'caption':
            inputs = self.mp.map(caption_representation, data)
            #inputs = list(map(caption_representation, data))
            inputs = list(filter(lambda x: x is not None, inputs))
            return caption_gather_fn(inputs)
        elif mode == 'shape':
            # inputs = self.mp.map(shape_representation, data)
            inputs = list(map(shape_representation, data))
            inputs = list(filter(lambda x: x is not None, inputs))
            return shape_gather_fn(inputs)
        else:
            raise ValueError


def queue_datagen(inputs, batch_size=128, n_proc=12, mp_pool=None, mode='shape'):
    """
    Continuously produce representations.
    """
    n_batches = math.ceil(len(inputs) / batch_size)
    sh_indencies = np.arange(len(inputs))

    my_batch_prep = Batch_prep(n_proc=n_proc, mp_pool=mp_pool)

    #while True:
        #np.random.shuffle(sh_indencies)
    for i in range(n_batches):
        batch_idx = sh_indencies[i * batch_size:(i + 1) * batch_size]
        yield my_batch_prep.transform_data(inputs[batch_idx])




# def prepareProteinForAtomtyping(mol, guessBonds=True, protonate=True, pH=7, verbose=True):
#     """Prepares a Molecule object for atom typing.

#     Parameters
#     ----------
#     mol : Molecule object
#         The protein to prepare
#     guessBonds : bool
#         Drops the bonds in the molecule and guesses them from scratch
#     protonate : bool
#         Protonates the protein for the given pH and optimizes hydrogen networks
#     pH : float
#         The pH for protonation
#     verbose : bool
#         Set to False to turn of the printing

#     Returns
#     -------
#     mol : Molecule object
#         The prepared Molecule
#     """
#     from moleculekit.tools.autosegment import autoSegment2

#     mol = mol.copy()
#     if (
#         guessBonds
#     ):  # Need to guess bonds at the start for atom selection and for autoSegment
#         mol.bondtype = np.array([], dtype=object)
#         mol.bonds = mol._guessBonds()

#     protsel = mol.atomselect("protein")
#     metalsel = mol.atomselect("element {}".format(" ".join(metal_atypes)))
#     watersel = mol.atomselect("water")
#     notallowed = ~(protsel | metalsel | watersel)

#     if not np.any(protsel):
#         raise RuntimeError("No protein atoms found in Molecule")

#     if np.any(notallowed):
#         resnames = np.unique(mol.resname[notallowed])
#         raise RuntimeError(
#             "Found atoms with resnames {} in the Molecule which can cause issues with the voxelization. Please make sure to only pass protein atoms and metals.".format(
#                 resnames
#             )
#         )

#     protmol = mol.copy()
#     protmol.filter(protsel, _logger=False)
#     metalmol = mol.copy()
#     metalmol.filter(metalsel, _logger=False)
#     watermol = mol.copy()
#     watermol.filter(watersel, _logger=False)

#     if protonate:
#         from moleculekit.tools.preparation import proteinPrepare

#         if np.all(protmol.segid == "") and np.all(protmol.chain == ""):
#             protmol = autoSegment2(
#                 protmol, fields=("segid", "chain"), basename="K", _logger=verbose
#             )  # We need segments to prepare the protein
#         protmol = proteinPrepare(
#             protmol, pH=pH, verbose=verbose, _loggerLevel="INFO" if verbose else "ERROR"
#         )

#     if guessBonds:
#         protmol.bonds = protmol._guessBonds()
#         # TODO: Should we remove bonds between metals and protein?

#     mol = protmol.copy()
#     mol.append(metalmol)
#     mol.append(watermol)
#     return mol