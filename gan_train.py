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
from train_utils import get_crop_boundaries, gen_target_probabilities
from torch.optim import lr_scheduler
import statistics
import math

from networks.Hul import Hulb128Net, Hul112Disc

default_train_data = ['datasets/train/NIND_128_112']
default_beta1 = 0.5
default_lr = 0.0003
default_weight_SSIM = 0.2
default_weight_L1 = 0.05
default_test_reserve = ['ursulines-red stefantiek', 'ursulines-building', 'MuseeL-Bobo', 'CourtineDeVillersDebris', 'C500D', 'Pen-pile']
default_d_network = 'Hul112Disc'
default_g_network = 'Hulb128Net'

# Training settings

parser = argparse.ArgumentParser(description='(c)GAN trainer for mthesis-denoise')
parser.add_argument('--batch_size', type=int, default=19, help='Training batch size')
parser.add_argument('--time_limit', type=int, default=172800, help='Time limit (ends training)')
parser.add_argument('--g_activation', type=str, default='PReLU', help='Final activation function for generator')
parser.add_argument('--g_funit', type=int, default=32, help='Filter unit size for generator')
parser.add_argument('--d_activation', type=str, default='PReLU', help='Final activation function for discriminator')
parser.add_argument('--d_funit', type=int, default=24, help='Filter unit size for discriminator')
parser.add_argument('--d_weights_dict_path', help='Discriminator weights dictionary')
parser.add_argument('--g_weights_dict_path', help='Generator weights dictionary')
parser.add_argument('--d_model_path', help='Discriminator pretrained model path')
parser.add_argument('--g_model_path', help='Generator pretrained model path')
parser.add_argument('--beta1', type=float, default=default_beta1, help='beta1 for adam. default=0.5')
parser.add_argument('--d_loss_function', type=str, default='MSE', help='Discriminator loss function')
parser.add_argument('--d_lr', type=float, default=default_lr, help='Initial learning rate for adam (discriminator)')
parser.add_argument('--g_lr', type=float, default=default_lr, help='Initial learning rate for adam (generator)')
parser.add_argument('--weight_SSIM', type=float, default=default_weight_SSIM, help='Weight on SSIM term in objective')
parser.add_argument('--weight_L1', type=float, default=default_weight_L1, help='Weight on L1 term in objective')
parser.add_argument('--test_reserve', nargs='*', help='Space separated list of image sets to be reserved for testing')
parser.add_argument('--train_data', nargs='*', help="(space-separated) Path(s) to the pre-cropped training data (default: %s)"%(" ".join(default_train_data)))
parser.add_argument('--cuda_device', default=0, type=int, help='Device number (default: 0, typically 0-3, -1 for CPU)')
parser.add_argument('--d_network', type=str, default=default_d_network, help='Discriminator network (default: %s)'%default_d_network)
parser.add_argument('--g_network', type=str, default=default_g_network, help='Generator network (default: %s)'%default_g_network)
parser.add_argument('--threads', type=int, default=8, help='Number of threads for data loader to use')
parser.add_argument('--min_lr', type=float, default=0.00000005, help='Minimum learning rate (ends training)')
parser.add_argument('--not_conditional', action='store_true', help='Regular GAN instead of cGAN')
parser.add_argument('--epochs', type=int, default=9001, help='Number of epochs (ends training)')
parser.add_argument('--compute_SSIM_anyway', action='store_true', help='Compute and display SSIM loss even if not used')
parser.add_argument('--freeze_generator', action='store_true', help='Freeze generator until discriminator is useful')

args = parser.parse_args()

# process some arguments

if args.test_reserve is None or args.test_reserve == []:
    test_reserve = default_test_reserve
else:
    test_reserve = args.test_reserve
if args.train_data is None or args.train_data == []:
    train_data = default_train_data
else:
    train_data = args.train_data
if args.cuda_device >= 0 and torch.cuda.is_available():
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda:"+str(args.cuda_device))
else:
    device = torch.device('cpu')


# classes

class Printer:
    def __init__(self, tostdout=True, tofile=True, file_path='log'):
        self.tostdout = tostdout
        self.tofile = tofile
        self.file_path = file_path

    def print(self, msg):
        if self.tostdout:
            print(msg)
        if self.tofile:
            with open(self.file_path, 'a') as f:
                f.write(str(msg)+'\n')

# TODO generic model class to implement common functions


class Generator:
    def __init__(self, network = 'Hulb128Net', weights_dict_path = None, model_path = None,
                 device = 'cuda:0', weight_SSIM=default_weight_SSIM,
                 weight_L1=default_weight_L1, activation='PReLU', funit=32,
                 beta1=default_beta1, lr=default_lr, printer=None, compute_SSIM_anyway=False):
        self.p = printer
        self.loss = 1
        self.weight_SSIM = weight_SSIM
        if weight_SSIM > 0 or compute_SSIM_anyway:
            self.criterion_SSIM = pytorch_ssim.SSIM().to(device)
        self.weight_L1 = weight_L1
        if weight_L1 > 0:
            self.criterion_L1 = nn.L1Loss().to(device)
        self.weight_D = 1 - weight_SSIM - weight_L1
        if self.weight_D > 0:
            self.criterion_D = nn.MSELoss().to(device)
        if model_path is not None:
            self.model = torch.load(model_path, map_location=device)
        else:
            if network == 'Hulb128Net':
                self.model = Hulb128Net(funit=funit, activation=activation)
            #elif ...
            else:
                p.print('Error: generator network not properly specified')
                exit(1)
            if weights_dict_path is not None:
                self.model.load_state_dict(torch.load(weights_dict_path))
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.75, verbose=True, threshold=1e-8, patience=5)
        self.device = device
        self.loss = {'SSIM': 1, 'L1': 1, 'D': 1, 'weighted': 1}
        self.compute_SSIM_anyway = compute_SSIM_anyway

    def get_loss(self, pretty_printed=False):
        if pretty_printed:
            return ", ".join(["%s: %.3f"%(key, val) if val != 1 else 'NA' for key,val in self.loss.items()])
        return self.loss

    def denoise_batch(self, noisy_batch):
        return self.model(noisy_batch)

    def learn(self, generated_batch_cropped, clean_batch_cropped, discriminator_predictions=None):
        if self.weight_SSIM > 0 or self.compute_SSIM_anyway:
            loss_SSIM = self.criterion_SSIM(generated_batch_cropped, clean_batch_cropped)
            loss_SSIM = 1-loss_SSIM
            self.loss['SSIM'] = loss_SSIM.item()
        if self.weight_SSIM == 0:
            loss_SSIM = torch.zeros(1).to(device)
        if self.weight_L1 > 0:
            loss_L1 = self.criterion_L1(generated_batch_cropped, clean_batch_cropped)
            self.loss['L1'] = loss_L1.item()
        else:
            loss_L1 = torch.zeros(1).to(device)
        if self.weight_D > 0:
            loss_D = self.criterion_D(discriminator_predictions,
                                      gen_target_probabilities(True, discriminator_predictions.shape,
                                                               device=self.device, noisy=False))
            self.loss['D'] = math.sqrt(loss_D.item())
        else:
            loss_D = torch.zeros(1).to(device)
        loss = loss_SSIM * self.weight_SSIM + loss_L1 * self.weight_L1 + loss_D * self.weight_D
        self.loss['weighted'] = loss.item()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_learning_rate(self, avg_loss):
        self.scheduler.step(metrics=avg_loss)
        lr = self.optimizer.param_groups[0]['lr']
        p.print('Learning rate: %f' % lr)
        return lr

    def save_model(self, model_dir, epoch):
        save_path = os.path.join(model_dir, 'generator_%u.pt' % epoch)
        torch.save(self.model.state_dict(), save_path)


class Discriminator:
    def __init__(self, network='Hul112Disc', weights_dict_path=None,
                 model_path=None, device='cuda:0', loss_function='MSE',
                 activation='PReLU', funit=24, beta1=default_beta1,
                 lr = default_lr, not_conditional = False, printer=None):
        self.p = printer
        self.device = device
        self.loss = 1
        self.loss_function = loss_function
        if loss_function == 'MSE':
            self.criterion = nn.MSELoss().to(device)
        if model_path is not None:
            self.model = torch.load(model_path, map_location=device)
        else:
            if not_conditional:
                input_channels = 3
            else:
                input_channels = 6
            if network == 'Hul112Disc':
                self.model = Hul112Disc(funit=funit, out_activation=activation, input_channels = input_channels)
            # elif ...
            else:
                p.print('Error: generator network not properly specified')
                exit(1)
            self.model = self.model.to(device)
            if weights_dict_path is not None:
                self.model.load_state_dict(torch.load(weights_dict_path))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))
        self.conditional = not not_conditional
        self.predictions_range = None
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.75, verbose=True, threshold=1e-8, patience=5)

    def update_learning_rate(self, avg_loss):
        self.scheduler.step(metrics=avg_loss)
        lr = self.optimizer.param_groups[0]['lr']
        p.print('Learning rate: %f' % lr)
        return lr

    def get_loss(self):
        return self.loss

    def get_predictions_range(self):
        return 'range (r-r+f-f+): '+str(self.predictions_range)

    def update_loss(self, loss_fake, loss_real):
        if self.loss_function == 'MSE':
            self.loss = (math.sqrt(loss_fake)+math.sqrt(loss_real))/2
        else:
            p.print('Error: loss function not implemented: %s'%(self.loss_function))

    def discriminate_batch(self, generated_batch_cropped, noisy_batch_cropped=None):
        if self.conditional:
            fake_batch = torch.cat([noisy_batch_cropped, generated_batch_cropped], 1)
        else:
            fake_batch = generated_batch_cropped
        return self.model(fake_batch)

    def learn(self, generated_batch_cropped, clean_batch_cropped, noisy_batch_cropped=None):
        self.optimizer.zero_grad()
        if self.conditional:
            real_batch = torch.cat([noisy_batch_cropped, clean_batch_cropped], 1)
            fake_batch = torch.cat([noisy_batch_cropped, generated_batch_cropped.detach()], 1)
        else:
            real_batch = clean_batch_cropped
            fake_batch = generated_batch_cropped.detach()
        pred_real = self.model(real_batch)
        loss_real = self.criterion(pred_real,
                                   gen_target_probabilities(True, pred_real.shape,
                                                            device=self.device, noisy=True))
        loss_real_detached = loss_real.item()
        loss_real.backward()
        pred_fake = self.model(fake_batch)
        loss_fake = self.criterion(pred_fake,
                                   gen_target_probabilities(False, pred_fake.shape,
                                                            device=self.device,
                                                            noisy=self.loss < 0.25))
        loss_fake_detached = loss_fake.item()
        loss_fake.backward()
        self.predictions_range = ", ".join(["{:.2}".format(float(i)) for i in (min(pred_real), max(pred_real), min(pred_fake), max(pred_fake))])
        self.update_loss(loss_fake_detached, loss_real_detached)
        self.optimizer.step()

    def save_model(self, model_dir, epoch):
        save_path = os.path.join(model_dir, 'generator_%u.pt' % epoch)
        torch.save(self.model.state_dict(), save_path)


def crop_batch(batch, boundaries):
    return batch[:, :, boundaries[0]:boundaries[1], boundaries[0]:boundaries[1]]


cudnn.benchmark = True

torch.manual_seed(123)
torch.cuda.manual_seed(123)

expname = datetime.datetime.now().isoformat()[:-10]+'_'+'_'.join(sys.argv).replace('/','-')[0:255]
model_dir = os.path.join('models', expname)
txt_path = os.path.join('results', 'train', expname)
os.makedirs(model_dir, exist_ok=True)

frozen_generator = args.freeze_generator

p = Printer(file_path=os.path.join(txt_path))

p.print(args)
p.print("cmd: python3 "+" ".join(sys.argv))

DDataset = DenoisingDataset(train_data, test_reserve=test_reserve)
data_loader = DataLoader(dataset=DDataset, num_workers=args.threads, drop_last=True,
                         batch_size=args.batch_size, shuffle=True)

discriminator = Discriminator(network=args.d_network, weights_dict_path=args.d_weights_dict_path,
                              model_path=args.d_model_path, device=device,
                              loss_function=args.d_loss_function, activation=args.d_activation,
                              funit=args.d_funit, beta1=args.beta1, lr=args.d_lr,
                              not_conditional=args.not_conditional, printer=p)
generator = Generator(network=args.g_network, weights_dict_path=args.g_weights_dict_path,
                      model_path=args.g_model_path, device=device, weight_SSIM=args.weight_SSIM,
                      weight_L1=args.weight_L1, activation=args.g_activation, funit=args.g_funit,
                      beta1=args.beta1, lr=args.g_lr, printer=p,
                      compute_SSIM_anyway=args.compute_SSIM_anyway)

crop_boundaries = get_crop_boundaries(DDataset.cs, DDataset.ucs, args.g_network, args.d_network)

use_D = (args.weight_SSIM + args.weight_L1) < 1
discriminator_predictions = None
generator_learning_rate = args.g_lr
discriminator_learning_rate = args.d_lr

start_time = time.time()
for epoch in range(1, args.epochs):
    loss_D_list = []
    loss_G_list = []
    loss_G_SSIM_list = []
    epoch_start_time = time.time()
    for iteration, batch in enumerate(data_loader, 1):
        iteration_summary = 'Epoch %u batch %u/%u: ' % (epoch, iteration, len(data_loader))
        clean_batch_cropped = crop_batch(batch[0].to(device), crop_boundaries)
        noisy_batch = batch[1].to(device)
        noisy_batch_cropped = crop_batch(noisy_batch, crop_boundaries)
        generated_batch = generator.denoise_batch(noisy_batch)
        generated_batch_cropped = crop_batch(generated_batch, crop_boundaries)
        # train discriminator based on its previous performance
        discriminator_learns = (discriminator.get_loss() > random.random() and use_D) or frozen_generator
        if discriminator_learns:
            discriminator.learn(noisy_batch_cropped=noisy_batch_cropped,
                                generated_batch_cropped=generated_batch_cropped,
                                clean_batch_cropped=clean_batch_cropped)
            loss_D_list.append(discriminator.get_loss())
            iteration_summary += 'loss D: %f (%s)' % (discriminator.get_loss(), discriminator.get_predictions_range())
        # train generator if discriminator didn't learn or discriminator is somewhat useful
        generator_learns = ((not discriminator_learns) or (discriminator.get_loss() < random.random())) and not frozen_generator
        if generator_learns:
            if discriminator_learns:
                iteration_summary += ', '
            while len(iteration_summary) < 100:
                iteration_summary += ' '
            if use_D:
                discriminator_predictions = discriminator.discriminate_batch(
                    generated_batch_cropped=generated_batch_cropped,
                    noisy_batch_cropped=noisy_batch_cropped)
            generator.learn(generated_batch_cropped=generated_batch_cropped,
                            clean_batch_cropped=clean_batch_cropped,
                            discriminator_predictions=discriminator_predictions)
            loss_G_list.append(generator.get_loss()['weighted'])
            loss_G_SSIM_list.append(generator.get_loss()['SSIM'])
            iteration_summary += 'loss G: %s' % generator.get_loss(pretty_printed=True)
        elif frozen_generator:
            frozen_generator = discriminator.get_loss() < 0.5
        p.print(iteration_summary)

    p.print("Epoch %u summary:" % epoch)
    p.print("Time elapsed (s): %u (epoch), %u (total)" % (time.time()-epoch_start_time,
                                                          time.time()-start_time))
    p.print("Generator:")
    if len(loss_G_SSIM_list) > 0:
        p.print("Average SSIM loss: %f" % statistics.mean(loss_G_SSIM_list))
    if len(loss_G_list) > 0:
        average_g_weighted_loss = statistics.mean(loss_G_list)
        p.print("Average weighted loss: %f" % average_g_weighted_loss)
        generator_learning_rate = generator.update_learning_rate(average_g_weighted_loss)
    else:
        p.print("Generator learned nothing")
    p.print("Discriminator:")
    if len(loss_D_list) > 0:
        average_d_loss = statistics.mean(loss_D_list)
        p.print("Average normalized loss: %f" % (average_d_loss))
        discriminator_learning_rate = discriminator.update_learning_rate(average_d_loss)
    generator.save_model(model_dir, epoch)
    discriminator.save_model(model_dir, epoch)
    if args.time_limit < time.time() - start_time:
        p.print("Time is up")
        exit(0)
    if discriminator_learning_rate < args.min_lr and generator_learning_rate < args.min_lr:
        p.print("Minimum learning rate reached")
        exit(0)

