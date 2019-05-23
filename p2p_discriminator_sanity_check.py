# this could be merged into p2p_train.py

from __future__ import print_function
import argparse
import os
from dataset_torch_3 import DenoisingDataset
import time
import datetime
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
from lib import pytorch_ssim

from networks.p2p_networks import define_G, define_D, get_scheduler, update_learning_rate

# TODO default values should go here
default_train_data = ['datasets/train/NIND_128_112']

# TODO check parameters
# Hul112Disc should go with Hul128Net and 128cs dataset
# Hul144Disc should go with Hul160Net and 128cs dataset

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation-in-mthesis-denoise')
parser.add_argument('--batch_size', type=int, default=19, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='plateau', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.75, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--weight_ssim_0', type=float, default=0.4, help='weight on SSIM term in objective')
parser.add_argument('--weight_L1_0', type=float, default=0.1, help='weight on L1 term in objective')
parser.add_argument('--weight_ssim_1', type=float, help='weight on SSIM term in objective')
parser.add_argument('--weight_L1_1', type=float, help='weight on L1 term in objective')
parser.add_argument('--train_data', nargs='*', help="(space-separated) Path(s) to the pre-cropped training data (default: %s)"%(" ".join(default_train_data)))
parser.add_argument('--time_limit', default=172800, type=int, help='Time limit in seconds')
parser.add_argument('--find_noise', action='store_true', help='(DnCNN) Model noise if set, otherwise generate clean image')
parser.add_argument('--compressionmin', type=str, default=100, help='Minimum compression level ([1,100], default=100)')
parser.add_argument('--compressionmax', type=int, default=100, help='Maximum compression level ([1,100], default=100)')
parser.add_argument('--sigmamin', type=int, default=0, help='Minimum sigma (noise) value ([0,100], default=0)')
parser.add_argument('--sigmamax', type=int, default=0, help='Maximum sigma (noise) value ([0,100], default=0)')
parser.add_argument('--yval', type=str, help='Use a specified noise value for y. Default is to use all that is available, possible values are "x" (use the ground-truth, useful with artificial noise or compression) or any ISO value s.a. ISO64000')
parser.add_argument('--test_reserve', nargs='*', help='Space separated list of image sets to be reserved for testing')
parser.add_argument('--exact_reserve', action='store_true', help='If this is set, the test reserve string must match exactly, otherwise any set that contains a test reserve string will be ignored')
parser.add_argument('--do_sizecheck', action='store_true', help='Skip crop size check for faster initial loading (rely on filename only)')
parser.add_argument('--cuda_device', default=0, type=int, help='Device number (default: 0, typically 0-3, -1 for CPU)')
parser.add_argument('--expname', type=str, help='Experiment name used to save and/or load results and models (default autogenerated from time+CLI)')
parser.add_argument('--resume', action='store_true', help='Look for an experiment with the same parameters and continue (to force continuing an experiment with different parameters use --expname instead)')
parser.add_argument('--result_dir', default='results/train', type=str, help='Directory where results are stored (default: results/train)')
parser.add_argument('--models_dir', default='models', type=str, help='Directory where models are saved/loaded (default: models)')
parser.add_argument('--lr_gamma', default=.75, type=float, help='Learning rate decrease rate for plateau, StepLR (default: 0.75)')
parser.add_argument('--lr_step_size', default=5, type=int, help='Step size for StepLR, patience for plateau scheduler')
parser.add_argument('--model', default='Hul128Net', type=str, help='Model type (UNet, Resnet, Hul160Net, Hul128Net)')
parser.add_argument('--D_ratio_0', default=0.9, type=float, help='How often D learns compared to G initially ( (0,1])')
parser.add_argument('--D_ratio_1', default=0.33, type=float, help='How often D learns compared to G when D is in use ( (0,1])')
parser.add_argument('--D_ratio_2', default=0.1, type=float, help='How often D learns compared to G when D is too good ( (0,1])')
parser.add_argument('--lr_min', default=0.00000005, type=float, help='Minimum learning rate (training stops when both lr are below threshold, default: 0.00000005)')
parser.add_argument('--min_ssim_l', default=0.12, type=float, help='Minimum SSIM score before using GAN loss')
parser.add_argument('--post_fail_ssim_num', default=25, type=int, help='How many times SSIM is used exclusively when min_ssim_l threshold is not met')
parser.add_argument('--lr_update_min_D_ratio', default=0.2, type=float, help='Minimum use of the discriminator (vs SSIM) for LR reduction')
parser.add_argument('--not_conditional', action='store_true', help='Discriminator does not see noisy image')
parser.add_argument('--debug_D', action='store_true', help='Discriminator does not see noisy image')
parser.add_argument('--netD', default='Hul112Disc', type=str, help='Discriminator network type (basic, Hul144Disc, Hul112Disc)')
parser.add_argument('--load_g', type=str, help='Generator model to load')
parser.add_argument('--load_d', type=str, help='Discriminator model to load')
parser.add_argument('--D_loss_f', default='MSE', type=str, help='GAN loss (BCEWithLogits, MSE)')
#parser.add_argument('--min_loss_D', default=0.1, type=float) # TODO
parser.add_argument('--load_g_state_dict_path', help='Load state dictionary into model')
parser.add_argument('--load_d_state_dict_path', help='Load state dictionary into model')
parser.add_argument('--loss_d_max_threshold', type=float, default=0.)
parser.add_argument('--finalpool', action='store_true', help='Final pooling on the discriminator (instead of convolution)')
parser.add_argument('--out_activation', help='Specific discriminator output activation')

args = parser.parse_args()
print(args)

cudnn.benchmark = True

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# process some args

if args.train_data == None or args.train_data == []:
    train_data = default_train_data
else:
    train_data = args.train_data

if args.cuda_device >= 0 and torch.cuda.is_available():
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda:"+str(args.cuda_device))
else:
    device = torch.device('cpu')

D_n_layers = args.input_nc if args.not_conditional else args.input_nc + args.output_nc


# fun

def gen_target_probabilities(target_real=True):
    if target_real:
        res = 19/20+torch.rand(args.batch_size,1,1,1)/20
    else:
        res = torch.rand(args.batch_size,1,1,1)/20
    return res.to(device)

def set_requires_grad(net, requires_grad = False):
    for param in net.parameters():
        param.requires_grad = requires_grad

def find_experiment():
    exp = None
    bname = ('_'.join(sys.argv).replace('/','-')).replace('_--resume','')
    for adir in os.listdir(args.models_dir):
        if adir[17:]==bname:
            exp = adir
    return exp


if args.expname:
    expname = args.expname
else:
    if args.resume:
        expname = find_experiment()
        if expname == None:
            sys.exit('Error: cannot resume experiment (404)')
        print("Error: resume not implemented")
    else:
        expname = datetime.datetime.now().isoformat()[:-10]+'_'+'_'.join(sys.argv).replace('/','-')
print(expname)

save_dir = os.path.join('models', expname)
res_dir = os.path.join(args.result_dir, expname)



print('===> Loading datasets')
DDataset = DenoisingDataset(train_data, compressionmin=args.compressionmin, compressionmax=args.compressionmax, sigmamin=args.sigmamin, sigmamax=args.sigmamax, test_reserve=args.test_reserve, yval=args.yval, do_sizecheck=args.do_sizecheck, exact_reserve=args.exact_reserve)
training_data_loader = DataLoader(dataset=DDataset, num_workers=args.threads, drop_last=True, batch_size=args.batch_size, shuffle=True)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.test_batch_size, shuffle=False)

loss_crop_lb = int((DDataset.cs-DDataset.ucs)/4)
loss_crop_up = int(DDataset.cs)-loss_crop_lb

if args.D_loss_f == 'MSE':
    criterionGAN = nn.MSELoss().to(device)
    dout_activation = 'PReLU'
else:
    criterionGAN = nn.BCEWithLogitsLoss().to(device)
    dout_activation = None
if args.out_activation is not None:
    dout_activation = args.out_activation
# BCEWithLogitsLoss output:
# target >  0       0.5     1       -1
# res v
#   0       .6931   .6931   .6931   .6931
#   0.5     .9741   .7241   .4741   1.4741
#   1       1.3133  .8133   .3133   2.3133
#   -1      0.3133  .8133   1.3133  -.6867

print('===> Building models')
if args.load_d:
    net_d = torch.load(args.load_d, map_location=device)
else:
    net_d = define_D(D_n_layers, args.ndf, args.netD, gpu_id=device, out_activation=dout_activation, finalpool=args.finalpool)

# load state dic for compatibility
if args.load_d_state_dict_path:
    net_d.load_state_dict(torch.load(args.load_d_state_dict_path))

# setup optimizer

optimizer_d = optim.Adam(net_d.parameters(), lr=args.lr)#, betas=(args.beta1, 0.999))
net_d_scheduler = get_scheduler(optimizer_d, args, generator=False)

use_D = False
useful_discriminator = False


start_time = time.time()
iterations_before_d = args.post_fail_ssim_num
for epoch in range(args.epoch_count, args.niter + args.niter_decay + 1):
    ### train ###
    # reset counters at the beginning of batch
    total_loss_d = 0
    total_loss_g_D = 0
    total_loss_g_std = 0
    total_loss_g_ssim = 0
    num_train_d = 0
    num_train_g_D = 0
    num_train_g_std = 0
    loss_d_item = 1
    loss_d_item_str = 'nan'
    for iteration, batch in enumerate(training_data_loader, 1):
        cleanimg, noisyimg = batch[0].to(device), batch[1].to(device)
        # generate clean image ("fake")
        gnoisyimg = noisyimg
        # determine whether we use and/or update discriminator
        if loss_d_item < 0.2 or args.D_loss_f != 'MSE':   #MSE
            useful_discriminator = True
        if (loss_d_item < 0.001 and args.D_loss_f == 'MSE'):# or (loss_d_item > 5.0 and args.D_loss_f == 'BCEWithLogits'):  # critical
            D_ratio = args.D_ratio_2
        elif use_D:
            if loss_d_item > 0.221 and args.D_loss_f == 'MSE':  # needs improvement
                D_ratio = args.D_ratio_0
            else:   # std
                D_ratio = args.D_ratio_1
        else:   # init
            D_ratio = args.D_ratio_0

        discriminator_learns = random.random() < D_ratio

        if args.not_conditional:
            fake_ab = gnoisyimg[:,:,loss_crop_lb:loss_crop_up, loss_crop_lb:loss_crop_up]
            real_ab = cleanimg[:,:,loss_crop_lb:loss_crop_up, loss_crop_lb:loss_crop_up]
        else:
            fake_ab = torch.cat([noisyimg[:,:,loss_crop_lb:loss_crop_up, loss_crop_lb:loss_crop_up], gnoisyimg[:,:,loss_crop_lb:loss_crop_up, loss_crop_lb:loss_crop_up]], 1)
            real_ab = torch.cat([noisyimg[:,:,loss_crop_lb:loss_crop_up, loss_crop_lb:loss_crop_up], cleanimg[:,:,loss_crop_lb:loss_crop_up, loss_crop_lb:loss_crop_up]], 1)

        ## train discriminator ##
        if discriminator_learns or iteration == 1:
            d_in_chan = 3 if args.not_conditional else 6

            target_false_probabilities = gen_target_probabilities(False)
            target_true_probabilities = gen_target_probabilities(True)
            set_requires_grad(net_d, True)
            optimizer_d.zero_grad()
            pred_fake = net_d(fake_ab.detach())
            loss_D_fake = criterionGAN(pred_fake, target_false_probabilities)
            pred_real = net_d(real_ab)
            loss_D_real = criterionGAN(pred_real, target_true_probabilities)
            loss_d = (loss_D_fake + loss_D_real)/2  # not cat?
            if args.debug_D:
                print("pred_fake, pred_real")
                print(pred_fake)
                print(pred_real)
            loss_d.backward()
            optimizer_d.step()
            loss_d_item = loss_d.item()
            total_loss_d += loss_d_item
            loss_d_item_str = "{:.4f}".format(loss_d_item)
            num_train_d += 1
        else:
            loss_d_item_str = 'nan'

        ## train generator ##
        set_requires_grad(net_d, False)
        target_true_probabilities = gen_target_probabilities(True)
        if use_D:
            pred_fake = net_d(fake_ab)
        if use_D:
            num_train_g_D += 1
        else:
            iterations_before_d -= 1
            num_train_g_std += 1
        print("===> Epoch[{}]({}/{}): Loss_D: {}".format(
            epoch, iteration, len(training_data_loader), loss_d_item_str))
    if num_train_d > 5:
        update_learning_rate(net_d_scheduler, optimizer_d, loss_avg=total_loss_d/num_train_d)
    print('Discriminator average loss: '+str(total_loss_d/num_train_d))

    #checkpoint
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    except OSError as err:
        save_dir = save_dir[0:255]
        os.makedirs(save_dir)
    #if not os.path.exists(res_dir):
    #    os.makedirs(res_dir)
    net_d_model_out_path = os.path.join(save_dir, "netD_model_epoch_%d.pth" % epoch)
    torch.save(net_d, net_d_model_out_path)
    print("Checkpoint saved to {} at {}".format(save_dir, datetime.datetime.now().isoformat()))
    if args.time_limit is not None and args.time_limit < time.time() - start_time:
        print('Time is up.')
        break
    # TODO check this
    elif optimizer_d.param_groups[0]['lr'] < args.lr_min:
        print('Minimum learning rate reached.')
        break
