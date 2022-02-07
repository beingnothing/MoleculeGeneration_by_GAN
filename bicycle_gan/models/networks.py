# # The netwokrs are based on BycycleGAN (https://github.com/junyanz/BicycleGAN) and
# # Ligdream (https://github.com/compsciencelab/ligdream)
# # Networks are written in pytorch==0.3.1

import torch
import torch.nn as nn
import numpy as np
import os
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import lr_scheduler
import functools
from torch.nn import init



class ListModule(object):
    # should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(
                self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class D_N3Dv1LayersMulti(nn.Module):
    """
    Disriminator network of 3D-BicycleGAN
    """
    def __init__(self, input_nc, ndf=64, norm_layer=None, use_sigmoid=False, gpu_ids=[], num_D=1):
        super(D_N3Dv1LayersMulti, self).__init__()

        self.gpu_ids = gpu_ids
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(
                input_nc, ndf, norm_layer, use_sigmoid)
            self.model = nn.Sequential(*layers)
        else:
            self.model = ListModule(self, 'model')
            layers = self.get_layers(
                input_nc, ndf, norm_layer, use_sigmoid)
            self.model.append(nn.Sequential(*layers))
            self.down = nn.AvgPool3d(3, stride=2, padding=[
                1, 1, 1], count_include_pad=False)
            for i in range(num_D - 1):
                ndf = int(round(ndf / (2 ** (i + 1))))
                layers = self.get_layers(
                    input_nc, ndf, norm_layer, use_sigmoid)
                self.model.append(nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d,
                   use_sigmoid=False):
        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw,
                              stride=1, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 5):
            use_stride = 1
            if n in [1, 3]:
                use_stride = 2

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=use_stride, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = 8
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    def parallel_forward(self, model, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            return model(input)

    def forward(self, input):
        if self.num_D == 1:
            return self.parallel_forward(self.model, input)
        result = []
        down = input
        for i in range(self.num_D):
            result.append(self.parallel_forward(self.model[i], down))
            if i != self.num_D - 1:
                down = self.parallel_forward(self.down, down)
        return result


class G_Unet_add_all3D(nn.Module):
    """
    Generator network of 3D-BicycleGAN
    """
    def __init__(self, input_nc=7, output_nc=5, nz=7, num_downs=8, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, gpu_ids=[], upsample='basic'):

        super(G_Unet_add_all3D, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz

        # construct unet blocks
        unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                      stride=1)
        unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                      upsample=upsample, stride=1)
        for i in range(num_downs - 6):  # 2 iterations
            if i == 0:
                stride = 2
            elif i == 1:
                stride = 1
            else:
                raise NotImplementedError("Too big, cannot handle!")

            unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                            norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                            upsample=upsample, stride=stride)
        unet_block = UnetBlock_with_z3D(ngf * 4, ngf * 4, ngf * 8, nz, unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=2)
        unet_block = UnetBlock_with_z3D(ngf * 2, ngf * 2, ngf * 4, nz, unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=1)
        unet_block = UnetBlock_with_z3D(
            ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
            stride=2)
        unet_block = UnetBlock_with_z3D(input_nc, output_nc, ngf, nz, unet_block,
                                        outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=1)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z3D(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero',
                 stride=2):
        super(UnetBlock_with_z3D, self).__init__()
        
        downconv = []
        if padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv3d(input_nc, inner_nc,
                               kernel_size=3, stride=stride, padding=p)]

        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = downconv
            up = [uprelu] + upconv + [nn.Sigmoid()]
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3), x.size(4))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero', stride=2):
    if upsample == 'basic':
        upconv = [nn.ConvTranspose3d(
            inplanes, outplanes, kernel_size=3, stride=stride, padding=1,
            output_padding=1 if stride == 2 else 0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


def fetch_simple_block3d(in_lay, out_lay, nl, norm_layer, stride=1, kw=3, padw=1):
    return [nn.Conv3d(in_lay, out_lay, kernel_size=kw,
                      stride=stride, padding=padw),
            nl(),
            norm_layer(out_lay)]


class E_3DNLayers(nn.Module):
    """
    E network of 3D-BicycleGAN
    """
    def __init__(self, input_nc=7, output_nc=5, ndf=64,
                 norm_layer='instance', nl_layer=None, gpu_ids=[], vaeLike=False):
        super(E_3DNLayers, self).__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike

        nf_mult = 1
        kw, padw = 3, 1

        # Network
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw,
                              stride=1, padding=padw),
                    nl_layer()]
        # Repeats
        sequence.extend(fetch_simple_block3d(ndf, ndf * 2, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 2, ndf * 2, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 2, ndf * 4, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 4, ndf * 4, nl=nl_layer, norm_layer=norm_layer))

        sequence += [nn.AvgPool3d(3)]
        self.conv = nn.Sequential(*sequence)
        # self.conv = nn.Conv3d(input_nc, ndf, 3, 1, 1)
        self.fc = nn.Sequential(*[nn.Linear(ndf * 4, output_nc)])
        # self.fc = nn.Sequential(*[nn.Linear(ndf, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * 4, output_nc)])
            # self.fcVar = nn.Sequential(*[nn.Linear(ndf, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)

        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output





class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor list) - - tpyically the prediction output from a discriminator; supports multi Ds.
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        all_losses = []
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            all_losses.append(loss)
        total_loss = sum(all_losses)
        return total_loss, all_losses

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net





def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def define_G(input_nc, output_nc, nz, ngf, netG='unet_128', norm='batch', nl='relu',
             use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], where_add='input', upsample='bilinear'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)
    net = G_Unet_add_all3D(input_nc=input_nc, output_nc=output_nc, nz=nz, num_downs=7, ngf=ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                           use_dropout=use_dropout, upsample=upsample, gpu_ids=gpu_ids)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, norm='batch', nl='lrelu', init_type='xavier', init_gain=0.02, num_Ds=1, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    net = D_N3Dv1LayersMulti(input_nc=input_nc, ndf=ndf, norm_layer=norm_layer, gpu_ids=gpu_ids)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_E(input_nc, output_nc, ndf, netE,
             norm='batch', nl='lrelu',
             init_type='xavier', init_gain=0.02, gpu_ids=[], vaeLike=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)

    net = E_3DNLayers(input_nc, output_nc, ndf, norm_layer=norm_layer,
                   nl_layer=nl_layer, vaeLike=vaeLike, gpu_ids=gpu_ids)
    return init_net(net, init_type, init_gain, gpu_ids)




