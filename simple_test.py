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
import os
import time
import sys
import torch
import torchvision
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_dir', default='datasets/test/ds_128_96', type=str, help='directory of test dataset (or any directory containing images to be denoised), must end with [CROPSIZE]_[USEFULCROPSIZE]')
    parser.add_argument('--model_dir', type=str, help='directory where .th models are saved (latest .th file is autodetected)')
    parser.add_argument('--model_subdir', type=str, help='subdirectory where .th models are saved (latest .th file is autodetected, models dir is assumed)')
    parser.add_argument('--model_path', type=str, help='the specific model file path')
    parser.add_argument('--result_dir', default='results/test', type=str, help='directory where results are saved')
    parser.add_argument('--cuda_device', default=0, type=int, help='Device number (default: 0, typically 0-3)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing images')
    parser.add_argument('--uncrop', action='store_true', help='Uncrop denoised images (run uncrop_images.py)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':


    
    args = parse_args()
    if args.model_path:
        model_path = args.model_path
    elif args.model_dir or args.model_subdir:
        model_dir = args.model_dir if args.model_dir else os.path.join('models', args.model_subdir)
        #model_path = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
        model_path = os.path.join(model_dir, "latest_model.pth")
    else:
        model_path = os.path.join('models', sorted(os.listdir('models'))[-1])
        model_path = os.path.join(model_path, sorted(os.listdir(model_path))[-1])
    noisy_dir = args.noisy_dir[:-1] if args.noisy_dir[-1] == '/' else args.noisy_dir
    cs, ucs = [int(i) for i in noisy_dir.split('_')[-2:]]
    torch.cuda.set_device(args.cuda_device)
    totensor = torchvision.transforms.ToTensor()
    print('loading '+ model_path)
    model = torch.load(model_path, map_location='cuda:'+str(args.cuda_device))
    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()
    
    def get_and_pad_image(path):
        img = Image.open(path)
        if img.size != (cs, cs):
            newimg = Image.new('RGB', (cs, cs))
            xoffset = yoffset = 0
            xpos, ypos, ucs = [int(val.strip('.jpg')) for val in path.split('_')[-3:]] # pos in img
            if xpos == 0:
                xoffset = cs-img.width
            if ypos == 0:
                yoffset = cs.img.height
            newimg.paste(img, (xoffset, yoffset))
            img = newimg
        return totensor(img)
            
        
    
    for root, dirs, files in os.walk(noisy_dir):
        for name in files:
            cur_img_sav_dir = os.path.join(args.result_dir, '/'.join(model_path.split('/')[-2:]), noisy_dir.split('/')[-1], 'img', './'+root.split(noisy_dir)[-1])
            cur_img_sav_path = os.path.join(cur_img_sav_dir, name[:-4]+'_denoised.jpg')
            if os.path.isfile(cur_img_sav_path) and not args.overwrite:
                continue
            os.makedirs(cur_img_sav_dir, exist_ok=True)
            y_ = get_and_pad_image(os.path.join(root, name))
            #y_ = totensor(Image.open(os.path.join(root, name)))
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
        sys.argv.extend(['--crop_dir', os.path.join(args.result_dir, '/'.join(model_path.split('/')[-2:]), noisy_dir.split('/')[-1], 'img')])
        if not '--ds_dir' in sys.argv:
            sys.argv.extend(['--ds_dir', args.ds_dir])
        from uncrop_images import *


