import time
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from models.bicycle_gan_model import BiCycleGANModel
from options.train_options import TrainOptions
import os
import sys
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from generator_dude import queue_datagen

import matplotlib.pyplot as plt

if __name__ == '__main__':
    import numpy as np
    import pickle
    import multiprocessing
    paths = []
    root = './data/dude'
    for subdir in os.listdir(root):
        site_path = os.path.join(root, subdir, 'pro_vox.pkl')
        if not os.path.exists(site_path):
            continue
        if not pickle.load(open(site_path, 'rb')):
            continue
        # lig_path = os.path.join(root, subdir, 'actives_final.sdf')
        smi_path = os.path.join(root, subdir, 'actives_final.ism')
        with open(smi_path, 'r') as f:
            lig_smis = f.read().splitlines()
        # clustered_path = os.path.join(root, subdir, 'actives_final_clusterd')
        # if not os.path.exists(smi_path):
        #     continue
        # with open(smi_path, 'r') as f:
        #     lines = f.read().splitlines()
        #     smi_ids = [line[-1] for line in lines]

        for j, smi in enumerate(lig_smis):
            paths.append([site_path, smi.split()[0]])
    opt = TrainOptions().parse()   # get training options
    f.close()


    # mg = GeneratorEnqueuer(my_gen, seed=0)
    # mg.start()
    # mt_gen = mg.get()
    # dataset_size = len(dataset)    # get the number of images in the dataset.
    # print('The number of training images = %d' % dataset_size)

    model = BiCycleGANModel(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    if os.path.exists(os.path.join(model.save_dir, "log.txt")):
        os.remove(os.path.join(model.save_dir, "log.txt"))
    if os.path.exists(os.path.join(model.save_dir, "log_loss.txt")):
        os.remove(os.path.join(model.save_dir, "log_loss.txt"))
    with open(os.path.join(model.save_dir, "log_loss.txt"), "a") as loss_file:
        loss_file.write('G_GAN\tD\tG_GAN2\tD2\tG_L1\tz_L1\tkl\n')
    loss_file.close()

    loss_list = []
    total_iters = 0                # the total number of training iterations
    multiproc = multiprocessing.Pool(8)
    epoch = opt.epoch_count
    end_epoch = 20
    # for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    #1-201 outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    # iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

    my_gen = queue_datagen(np.array(paths), batch_size=opt.batch_size, n_proc=8, mp_pool=multiproc)
    # my_gen = queue_datagen(np.array(paths), batch_size=opt.batch_size, n_proc=1)
    for i, data in enumerate(my_gen):  # inner loop within one epoch

        iter_start_time = time.time()  # timer for computation per iteration
        # if total_iters % opt.print_freq == 0:
        #     t_data = iter_start_time - iter_data_time

        # total_iters += opt.batch_size
        # epoch_iter += opt.batch_size
        total_iters += 1
        epoch_iter += 1

        model.set_input(data)         # unpack data from dataset and apply preprocessing

        # if not model.is_train():      # if this batch of input data is enough for training.
        #     print('skip this batch')
        #     continue
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
        #     save_result = total_iters % opt.update_html_freq == 0

        # if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
        losses = model.get_current_losses()
        print('epoch: %s, iters: %s, losses: %s' % (epoch, total_iters, losses))
        # t_comp = (time.time() - iter_start_time) / opt.batch_size

        loss_items = [v for k,v in losses.items()]
        loss_list.append(sum(loss_items))

        with open(os.path.join(model.save_dir, "log.txt"), "a") as log_file:
            log_file.write('epoch: %s, iters: %s, losses: %s\n' % (epoch, total_iters, str(losses)))
        with open(os.path.join(model.save_dir, "log_loss.txt"), "a") as loss_file:
            loss_file.write("\t".join([str(v) for v in loss_items]) + "\n")


        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            model.save_networks('latest')

        # iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            if epoch > end_epoch:
                break
        
        if total_iters % int(math.ceil(len(paths) / opt.batch_size)) == 0:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()                     # update learning rates at the end of every epoch.
            epoch += 1
            epoch_iter = 0
            epoch_start_time = time.time()


    x = range(0,len(loss_list))
    plt.plot(x, loss_list, '.-')
    plt_title = 'LiGANN Loss Curve during training (BATCH_SIZE=128)'
    plt.title(plt_title)
    plt.xlabel('training iter')
    plt.ylabel('LiGANN Loss')
    plt.savefig(os.path.join(model.save_dir, 'loss_ligann.jpg'))
    #plt.show()