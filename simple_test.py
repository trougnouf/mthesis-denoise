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
import torch
import torchvision
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_dir', default='datasets/test/testdata_128', type=str, help='directory of test dataset (or any directory containing images to be denoised)')
    parser.add_argument('--model_dir', type=str, help='directory where .th models are saved (latest .th file is autodetected)')
    parser.add_argument('--model_path', type=str, help='the model file path')
    parser.add_argument('--result_dir', default='results/test', type=str, help='directory where results are saved')
    parser.add_argument('--cuda_device', default=0, type=int, help='Device number (default: 0, typically 0-3)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing images')
    parser.add_argument('--uncrop', action='store_true', help='Uncrop denoised images (run uncrop_images.py)')
    parser.add_argument('--ds_dir', default='datasets/dataset', type=str, help='Directory with matching original images to get dimensions when running with uncrop')
    args = parser.parse_args()
    if not args.model_dir and not args.model_path:
        parser.error('model_dir or model_path argument is required')
    return args


if __name__ == '__main__':

    args = parse_args()
    torch.cuda.set_device(args.cuda_device)
    totensor = torchvision.transforms.ToTensor()

    if not args.model_path:
        #model_path = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
        model_path = os.path.join(args.model_dir, "latest_model.pth")
    else:
        model_path = args.model_path
    print('loading '+ model_path)
    model = torch.load(model_path, map_location='cuda:'+str(args.cuda_device))
    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()

    for root, dirs, files in os.walk(args.noisy_dir):
        for name in files:
            cur_img_sav_dir = os.path.join(args.result_dir, '/'.join(model_path.split('/')[-2:]), args.noisy_dir.split('/')[-1], 'img', './'+root.split(args.noisy_dir)[-1])
            cur_img_sav_path = os.path.join(cur_img_sav_dir, name[:-4]+'_denoised.jpg')
            if os.path.isfile(cur_img_sav_path) and not args.overwrite:
                continue
            os.makedirs(cur_img_sav_dir, exist_ok=True)
            y_ = totensor(Image.open(os.path.join(root, name)))
            y_ = y_.view(1,-1,y_.shape[1], y_.shape[2]) # TODO is this correct?
            torch.cuda.synchronize()
            start_time = time.time()
            y_ = y_.cuda()
            x_ = model(y_)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            torchvision.utils.save_image(x_, cur_img_sav_path)
            print('%s : %2.4f second' % (name, elapsed_time))

    if args.uncrop:
        from uncrop_images import *


