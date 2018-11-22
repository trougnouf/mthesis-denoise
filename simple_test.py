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
    parser.add_argument('--set_dir', default='datasets/test/testdata_128', type=str, help='directory of test dataset')
    parser.add_argument('--models_dir', default='models', type=str, help='root directory of models')
    parser.add_argument('--expname', type=str, help='experiment name where the models are located')
    parser.add_argument('--model_fn', default='notset', type=str, help='the model filename, s.a. model_500.pth (last file is autodetected)')
    parser.add_argument('--result_dir', default='results/test', type=str, help='directory where results are saved')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    parser.add_argument('--cuda_device', default=0, type=int, help='Device number (default: 0, typically 0-3)')
    args = parser.parse_args()
    if not args.expname:
        parser.error('expname argument is required')
    return args


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


if __name__ == '__main__':

    args = parse_args()
    totensor = torchvision.transforms.ToTensor()

    if not os.path.exists(os.path.join(args.models_dir, args.expname, args.model_fn)):
        model_fn = sorted(os.listdir(os.path.join(args.models_dir, args.expname)))[-1]
    else:
        model_fn = args.model_fn
    log('load '+args.expname+'/'+model_fn)
    model = torch.load(os.path.join(args.models_dir, args.expname, model_fn), map_location='cuda:'+args.cuda_device)
    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()

    set_names = os.listdir(args.set_dir)
    if '.jpg' in set_names[0]:
        set_names = ['.']
    for set_cur in set_names:
        result_dir_img = os.path.join(args.result_dir, args.expname, model_fn, 'img', args.set_dir.split('/')[-1], set_cur)
        result_dir_txt = os.path.join(args.result_dir, args.expname, model_fn, 'txt', args.set_dir.split('/')[-1])
        os.makedirs(result_dir_img, exist_ok=True)
        os.makedirs(result_dir_txt, exist_ok=True)

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                y_ = totensor(Image.open(os.path.join(args.set_dir, set_cur, im)))
                y_ = y_.view(1,-1,y_.shape[1], y_.shape[2]) # TODO is this correct?
                torch.cuda.synchronize()
                start_time = time.time()
                y_ = y_.cuda()
                x_ = model(y_)  # inference
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))
                name, ext = os.path.splitext(im)
                torchvision.utils.save_image(x_, os.path.join(result_dir_img, name+'_dncnn'+ext))




