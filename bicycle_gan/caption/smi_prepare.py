# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from tqdm import tqdm
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=False, type=str, default="./data/traindataset/zinc15_druglike_clean_canonical_max60.smi", help="Path to input smi file.")
parser.add_argument("-o", "--output", required=False, type=str, default="./data/zinc15_max60", help="Path to output numpy arrays.")
args = vars(parser.parse_args())

smiles_path = args["input"]
np_save_path = args["output"]

truncate_len = 60
# smiles_ = open(smiles_path, "r").read().splitlines()[1:]
# smiles = []
# for smi in smiles_:
#     # if len(smi) > truncate_len:
#     #     continue
#     smiles.append(smi.split()[0])
# # strings = np.zeros((len(smiles), truncate_len+2), dtype='uint8')


# vocab_list = ["pad", "start", "end",
#     "C", "c", "N", "n", "S", "s", "P", "O", "o",
#     "B", "F", "I",
#     "X", "Y", "Z",
#     "1", "2", "3", "4", "5", "6",
#     "#", "=", "-", "(", ")", "[", "]", "+", 'H'
# ]
vocab_list = ["pad", "start", "end",
    "C", "c", "N", "n", "S", "s", "P", "O", "o",
    "B", "F", "I",
    "X", "Y", "Z",
    "1", "2", "3", "4", "5", "6",
    "#", "=", "-", "(", ")"
]
vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}

strings = []
# smiles_ = open(smiles_path, "r").read().splitlines()
smiles = open(smiles_path, "r").read().strip().split()
for i, sstring in enumerate(tqdm(smiles)):
    mol = Chem.MolFromSmiles(sstring)
    if not mol:
        print("Failed to parse molecule: " + sstring)
        continue

    _sstring = Chem.MolToSmiles(mol)  # Make the SMILES canonical.
    _sstring = _sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")
    if len(_sstring) > 60:
        print("smi length > 60")
        continue
    try:
        vals = [1] + [vocab_c2i_v1[xchar] for xchar in _sstring] + [2]
    except KeyError:
        print("key error: " + _sstring)
        continue
        # raise ValueError(("Unkown SMILES tokens: {} in string '{}'."
        #                   .format(", ".join([x for x in sstring if x not in vocab_c2i_v1]),
        #                                                               sstring)))
    #strings[i, :len(vals)] = vals
    tmp = np.zeros(truncate_len + 2, dtype='uint8')
    tmp[:len(vals)] = vals
    strings.append(tmp)

strings = np.array(strings)
print(strings.shape)
np.save(np_save_path, strings)
