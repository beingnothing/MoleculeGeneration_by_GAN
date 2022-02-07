from moleculekit.molecule import Molecule, calculateUniqueBonds
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures, getChannels, _getOccupancyC, _getGridCenters
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping, metal_atypes
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.home import home
from moleculekit.util import uniformRandomRotation

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
import math
import os
import numpy as np
import torch
from torch.autograd import Variable

from caption.models.networks import EncoderCNN_v3, DecoderRNN
from options.test_options import TestOptions
from models.bicycle_gan_model import BiCycleGANModel

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

class CompoundGenerator:
    def __init__(self, use_cuda=True):
        self.use_cuda = False
        self.encoder = EncoderCNN_v3(5)
        self.decoder = DecoderRNN(512, 1024, 29, 1)
        # self.vae_model = LigandVAE(use_cuda=use_cuda)

        # self.vae_model.eval()
        self.encoder.eval()
        self.decoder.eval()

        if use_cuda:
            assert torch.cuda.is_available()
            self.encoder.cuda()
            self.decoder.cuda()
            # self.vae_model.cuda()
            self.use_cuda = True

    def load_weight(self, encoder_weights, decoder_weights):
        """
        Load the weights of the models.
        :param vae_weights: str - VAE model weights path
        :param encoder_weights: str - captioning model encoder weights path
        :param decoder_weights: str - captioning model decoder model weights path
        :return: None
        """
        # self.vae_model.load_state_dict(torch.load(vae_weights, map_location='cpu'))
        self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))


    def caption_shape(self, in_shapes, probab=False):
        """
        Generates SMILES representation from in_shapes
        """
        embedding = self.encoder(in_shapes)
        if probab:
            captions = self.decoder.sample_prob(embedding)
        else:
            captions = self.decoder.sample(embedding)

        captions = torch.stack(captions, 1)
        if self.use_cuda:
            captions = captions.cpu().data.numpy()
        else:
            captions = captions.data.numpy()
        return decode_smiles(captions)

    def generate_molecules(self, shape_input, n_attemps=300, lam_fact=1., probab=False, filter_unique_valid=True):
        """
        Generate novel compounds from a seed compound.
        :param smile_str: string - SMILES representation of a molecule
        :param n_attemps: int - number of decoding attempts
        :param lam_fact: float - latent space pertrubation factor
        :param probab: boolean - use probabilistic decoding
        :param filter_unique_canonical: boolean - filter for valid and unique molecules
        :return: list of RDKit molecules.
        """
        if self.use_cuda:
            shape_input = shape_input.cuda()

        # shape_input = shape_input.unsqueeze(0).repeat(n_attemps, 1, 1, 1, 1)
        shape_input = shape_input.repeat(n_attemps, 1, 1, 1, 1)
        with torch.no_grad():
            shape_input = Variable(shape_input)

        # recoded_shapes, _, _ = self.vae_model(shape_input, cond_input, lam_fact)
        smiles = self.caption_shape(shape_input, probab=probab)
        if filter_unique_valid:
            return filter_unique_canonical(smiles)
        return [Chem.MolFromSmiles(x) for x in smiles], smiles

def pros_representation(pro_file, user_center=None):
    try:
        protein = Molecule(pro_file)
    except:
        return None
    try:
        protein.filter("protein or ions and (not resname CL) and (not resname NA)")
        protein = prepareProteinForAtomtyping(protein)
        if user_center:
            center = np.array(user_center)
        else:
            coords = protein.get('coords')
            center = np.mean(coords, axis=0)
        features, center, N = getVoxelDescriptors(protein, boxsize=[protein_size, protein_size, protein_size], center=center, buffer=1)
    except:
        return None
    occupancy = features[:, [0, 1, 2, 3, 4, 5, 7]].reshape(protein_size, protein_size, protein_size, 7)
    protein_vox = occupancy.astype(np.float32).transpose(3, 0, 1, 2,)
    return torch.Tensor([protein_vox])

def filter_unique_canonical(in_mols):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids
    return [Chem.MolFromSmiles(x) for x in set(xresults)]  # Check for duplicates and filter out invalids

def decode_smiles(in_tensor):
    """
    Decodes input tensor to a list of strings.
    :param in_tensor:
    :return:
    """
    gen_smiles = []
    for sample in in_tensor:
        csmile = ""
        for xchar in sample[1:]:
            if xchar == 2:
                break
            csmile += vocab_i2c_v1[xchar]
        gen_smiles.append(csmile)
    return gen_smiles


# Load the weights of the LiGANN models
opt = TestOptions().parse()
model = BiCycleGANModel(opt)
# model = torch.nn.DataParallel(model).cuda()
# model.netG.load_state_dict(torch.load("./checkpoints/DUDE/latest_net_G.pth"))
model.setup(opt)

# Load the weights of the caption models
my_gen = CompoundGenerator(use_cuda=True)
# vae_weights =  os.path.join(home(), "modelweights/vae-210000.pkl")
encoder_weights = "./caption/saved_models/modelweights/encoder-210000.pkl"
decoder_weights = "./caption/saved_models/modelweights/decoder-210000.pkl"
my_gen.load_weight(encoder_weights, decoder_weights)

pro_path = './data/test/5p9k.pdb'
user_center = [19.7, 7.6, 1.0]
pro_vox = pros_representation(pro_path, user_center)
model.set_input(pro_vox)
pred_lig = model.test()
gen_mols, gen_smiles = my_gen.generate_molecules(pred_lig,
                                     n_attemps=20,  # How many attemps of generations will be carried out
                                     lam_fact=1.,  # Variability factor
                                     probab=False,  # Probabilistic RNN decoding
                                     filter_unique_valid=False)  # Filter out invalids and replicates
print(gen_smiles)
IPythonConsole.ShowMols(gen_mols)