import os
import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np

from models.networks import EncoderCNN_v3, DecoderRNN, VAE
#from keras.utils.data_utils import GeneratorEnqueuer
from tqdm import tqdm
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from generator_htmd import queue_datagen
from torch.nn import DataParallel
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=False, type=str, default="./data/zinc15_max60.npy", help="input file")
parser.add_argument("--output_dir", required=False, type=str, default="./saved_models/zinc15_max60/", help="model save folder")
parser.add_argument("-p", "--pre_model", required=False, type=str, help="pretrained model name", default="5000")
args = vars(parser.parse_args())

cap_loss = 0.
caption_start = 4000
batch_size = 128

savedir = args["output_dir"]
os.makedirs(savedir, exist_ok=True)
smiles = np.load(args["input"])
# smiles = smiles_src[26880000:, ]
import multiprocessing
multiproc = multiprocessing.Pool(6)


# Define the networks
encoder = EncoderCNN_v3(5)
decoder = DecoderRNN(512, 1024, 29, 1)
vae_model = VAE(use_cuda=True)

# load data using multiple gpus
encoder = torch.nn.DataParallel(encoder, device_ids=[0, 1, 2])
decoder = torch.nn.DataParallel(decoder, device_ids=[0, 1, 2])
vae_model = torch.nn.DataParallel(vae_model, device_ids=[0, 1, 2])

# load pretrained model
# if args["pre_model"]:
#     encoder.load_state_dict(torch.load(os.path.join('saved_models', 'encoder-%s.pkl' % args["pre_model"])))
#     decoder.load_state_dict(torch.load(os.path.join('saved_models', 'decoder-%s.pkl' % args["pre_model"])))
#     vae_model.load_state_dict(torch.load(os.path.join('saved_models', 'vae-%s.pkl' % args["pre_model"])))
#     caption_start = 1000

encoder.cuda()
decoder.cuda()
vae_model.cuda()

# Caption optimizer
criterion = nn.CrossEntropyLoss()
caption_params = list(decoder.parameters()) + list(encoder.parameters())
caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

encoder.train()
decoder.train()

# VAE optimizer and loss
reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False


def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # return BCE + KLD
    return BCE + KLD, (BCE, KLD)

vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-4)
vae_model.train()

# save training loss info
log_file = open(os.path.join(savedir, "log.txt"), "a")

# data generator
my_gen = queue_datagen(smiles, batch_size=batch_size, mp_pool=multiproc)
#mg = GeneratorEnqueuer(my_gen, seed=0)
#mg.start()
#mt_gen = mg.get()


def loss_plot(vae_loss_list, cap_loss_list, bce_loss_list, kld_loss_list, savedir):
    # lines = open('./saved_models/log.txt', "r").read().splitlines()
    # init_vae, init_cap = [], []
    # for i, line in enumerate(lines):
    #     if i >= 10:
    #         init_cap.append(float(line.split()[3][:-1]))
    #     init_vae.append(float(line.split()[5]))
    # init_vae += vae_loss_list
    # vae_loss_list = init_vae
    # init_cap += cap_loss_list
    # cap_loss_list = init_cap

    x1 = range(0, len(vae_loss_list))
    plt.subplot(4, 1, 1)
    plt.plot(x1, vae_loss_list, '.-')
    plt.title('Loss Curve during training (BATCH_SIZE=128)')
    plt.ylabel('VAE Loss')

    x1_1 = range(0, len(bce_loss_list))
    plt.subplot(4, 1, 2)
    plt.plot(x1_1, bce_loss_list, '.-')
    plt.ylabel('BCE Loss')

    x1_2 = range(0, len(kld_loss_list))
    plt.subplot(4, 1, 3)
    plt.plot(x1_2, kld_loss_list, '.-')
    plt.ylabel('KLD Loss')

    x2 = range(0, len(cap_loss_list))
    plt.subplot(4, 1, 4)
    plt.plot(x2, cap_loss_list, '.-')
    plt.ylabel('CAP Loss')
    plt.xlabel('Training iter (10 times)')
    plt.savefig(os.path.join(savedir, 'loss.jpg'))
    #plt.show()

vae_loss_list, cap_loss_list = [], []
bce_loss_list, kld_loss_list = [], []
# training process
with torch.autograd.set_detect_anomaly(True):
    #while True:
    my_gen = queue_datagen(smiles, batch_size=batch_size, n_proc=12, mp_pool=multiproc)
    for i, (mol_batch, caption, lengths) in tqdm(enumerate(my_gen)):
        in_data = Variable(mol_batch.cuda())
        #in_data = mol_batch.cuda()
        vae_optimizer.zero_grad()

        recon_batch, mu, logvar = vae_model(in_data)
        vae_loss, bce_kld = loss_function(recon_batch, in_data, mu, logvar)

        vae_loss.backward(retain_graph=True if i >= caption_start else False)
        # p_loss = vae_loss.data[0]
        p_loss = vae_loss.data.item()
        vae_optimizer.step()

        if i >= caption_start:  # Start by autoencoder optimization
            #import pdb
            #pdb.set_trace()
            # captions = Variable(caption.cuda())
            captions = caption.cuda()
            lengths = torch.Tensor(lengths)
            lengths.requires_grad = False
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(recon_batch.detach())
            #features = encoder(recon_batch)
            outputs = decoder(features, captions, lengths)
            cap_loss = criterion(outputs, targets)
            cap_loss.backward()
            caption_optimizer.step()

        if (i + 1) % 5000 == 0:
            torch.save(decoder.cpu().state_dict(),
                        os.path.join(savedir,
                                    'decoder-%d.pkl' % (i + 1)))
            torch.save(encoder.cpu().state_dict(),
                        os.path.join(savedir,
                                    'encoder-%d.pkl' % (i + 1)))
            torch.save(vae_model.cpu().state_dict(),
                        os.path.join(savedir,
                                    'vae-%d.pkl' % (i + 1)))
            decoder.cuda()
            encoder.cuda()
            vae_model.cuda()

        if (i + 1) % 100 == 0:
            vae_loss_list.append(p_loss)
            bce_loss_list.append(bce_kld[0].data.item())
            kld_loss_list.append(bce_kld[1].data.item())
            if type(cap_loss) != float:
                cap_loss_list.append(cap_loss.data.item())
            result = "Step: {}, caption_loss: {:.5f}, " \
                        "VAE_loss: {:.5f}, BCE_loss: {:.5f}, KLD_loss: {:.5f}".format(i + 1,
                                                float(cap_loss.data.cpu().numpy()) if type(cap_loss) != float else 0.,
                                                p_loss, bce_kld[0].data.item(), bce_kld[1].data.item())
            log_file.write(result + "\n")
            log_file.flush()
            print(result)

        # Reduce the LR
        if (i + 1) % 60000 == 0:
            # Command = "Reducing learning rate".format(i+1, float(loss.data.cpu().numpy()))
            log_file.write("Reducing LR\n")
            for param_group in caption_optimizer.param_groups:
                lr = param_group["lr"] / 2.
                param_group["lr"] = lr

        if i == 210000:
            # We are Done!
            log_file.close()
            loss_plot(vae_loss_list, cap_loss_list, bce_loss_list, kld_loss_list, savedir)
            # Cleanup
            del my_gen
            multiproc.close()
            sys.exit()

