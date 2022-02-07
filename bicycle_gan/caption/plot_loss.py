import os
import matplotlib.pyplot as plt



def loss_plot(savedir):
    lines = open('./saved_models/log.txt', "r").read().splitlines()
    vae_loss_list, cap_loss_list = [], []
    for i, line in enumerate(lines):
        if len(line.split()) != 6:
            continue
        if i >= 10:
            cap_loss_list.append(float(line.split()[3][:-1]))
        vae_loss_list.append(float(line.split()[5]))

    x1 = range(0, len(vae_loss_list))
    plt.subplot(2, 1, 1)
    plt.plot(x1, vae_loss_list, '.-')
    plt.title('Loss Curve during training (BATCH_SIZE=128)')
    plt.ylabel('VAE Loss')

    # x1_1 = range(0, len(bce_loss_list))
    # plt.subplot(4, 1, 2)
    # plt.plot(x1_1, bce_loss_list, '.-')
    # plt.ylabel('BCE Loss')

    # x1_2 = range(0, len(kld_loss_list))
    # plt.subplot(4, 1, 3)
    # plt.plot(x1_2, kld_loss_list, '.-')
    # plt.ylabel('KLD Loss')

    x2 = range(0, len(cap_loss_list))
    plt.subplot(2, 1, 2)
    plt.plot(x2, cap_loss_list, '.-')
    plt.ylabel('CAP Loss')
    plt.xlabel('Training iter (100 times)')
    plt.savefig(os.path.join(savedir, 'loss50000.jpg'))
    plt.show()

loss_plot('./saved_models')