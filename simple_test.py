# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to test the model

import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import torchvision
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='datasets/test/testdata_128', type=str, help='directory of test dataset')
    parser.add_argument('--models_dir', default='models', type=str, help='root directory of models')
    parser.add_argument('--expname', type=str, help='experiment name where the models are located')
    parser.add_argument('--model_fn', default='notset', type=str, help='the model filename, s.a. model_500.pth (last file is autodetected)')
    parser.add_argument('--result_dir', default='results/test', type=str, help='directory where results are saved')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    args = parser.parse_args()
    if not args.expname:
        parser.error('expname argument is required')
    return args


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


class DnCNN(nn.Module):

    def __init__(self, depth=22, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


if __name__ == '__main__':

    args = parse_args()
    totensor = torchvision.transforms.ToTensor()

    # model = DnCNN()
    if not os.path.exists(os.path.join(args.models_dir, args.expname, args.model_fn)):
        model_fn = sorted(os.listdir(os.path.join(args.models_dir, args.expname)))[-1]
    else:
        model_fn = args.model_fn
        # model.load_state_dict(torch.load(os.path.join(args.models_dir, args.model_fn)))
    log('load '+args.expname+'/'+model_fn)
    model = torch.load(os.path.join(args.models_dir, args.expname, model_fn), map_location='cuda:0')
#    params = model.state_dict()
#    print(params.values())
#    print(params.keys())
#
#    for key, value in params.items():
#        print(key)    # parameter name
#    print(params['dncnn.12.running_mean'])
#    print(model.state_dict())

    model.eval()  # evaluation mode
#    model.train()

    if torch.cuda.is_available():
        model = model.cuda()

    set_names = os.listdir(args.set_dir)
    if '.jpg' in set_names[0]:
        set_names = ['.']
    for set_cur in set_names:
        if args.save_result:
            result_dir_img = os.path.join(args.result_dir, args.expname, model_fn, 'img', args.set_dir.split('/')[-1], set_cur)
            result_dir_txt = os.path.join(args.result_dir, args.expname, model_fn, 'txt', args.set_dir.split('/')[-1])
            os.makedirs(result_dir_img, exist_ok=True)
            os.makedirs(result_dir_txt, exist_ok=True)
        if set_cur == 'ISO200' or set_cur == 'UNK':
            comparexy=False
        else:
            comparexy=True
        psnrs = []
        ssims = []

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                # x is clean, y is noise
                #if comparexy:
                #    x = np.array(imread(os.path.join(args.set_dir, 'ISO200', im.replace(set_cur, 'ISO200'))), dtype=np.float32)/255.0
                #x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32)/255.0
                #np.random.seed(seed=0)  # for reproducibility
                #y = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32)/255.0
                #y = x + np.random.normal(0, args.sigma/255.0, x.shape)  # Add Gaussian noise without clipping
                #y = y.astype(np.float32)
                #y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
                y_ = totensor(Image.open(os.path.join(args.set_dir, set_cur, im)))
                y_ = y_.view(1,-1,y_.shape[1], y_.shape[2])
                torch.cuda.synchronize()
                start_time = time.time()
                y_ = y_.cuda()
                x_ = model(y_)  # inference
                #x_ = x_.view(y_.shape[0], y_.shape[1], y_.shape[2])
                #x_ = x_.cpu()
                #x_ = x_.detach().numpy().astype(np.float32)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))
                #if comparexy:
                #    psnr_x_ = compare_psnr(x, x_)
                #    ssim_x_ = compare_ssim(x, x_, multichannel=True)
                name, ext = os.path.splitext(im)
                torchvision.utils.save_image(x_, os.path.join(result_dir_img, name+'_dncnn'+ext))
                    #show(np.hstack((y, x_)))  # show the image
                    #print(x_.shape)
                    #save_result(x_, path=os.path.join(result_dir_img, name+'_dncnn'+ext))
                      # save the denoised image
                #if comparexy:
                    #psnrs.append(psnr_x_)
                    #ssims.append(ssim_x_)
        #if comparexy:
            #psnr_avg = np.mean(psnrs)
            #ssim_avg = np.mean(ssims, )
            #psnrs.append(psnr_avg)
            #ssims.append(ssim_avg)
            #if args.save_result:
                #save_result(np.hstack((psnrs, ssims)), path=os.path.join(result_dir_txt, set_cur+'_results.txt'))
            #log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))








