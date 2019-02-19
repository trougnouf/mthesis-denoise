# run this to test the model

import argparse
import os
import time
import sys
import torch
import torchvision
from PIL import Image
from loss import gen_score
import subprocess

# eg python denoise_dir.py --model_subdir ...
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_dir', default='datasets/test/ds_fs', type=str, help='directory of test dataset (or any directory containing images to be denoised), must end with [CROPSIZE]_[USEFULCROPSIZE]')
    parser.add_argument('--model_dir', type=str, help='directory where .th models are saved (latest .th file is autodetected)')
    parser.add_argument('--model_subdir', type=str, help='subdirectory where .th models are saved (latest .th file is autodetected, models dir is assumed)')
    parser.add_argument('--model_path', type=str, help='the specific model file path')
    parser.add_argument('--result_dir', default='results/test', type=str, help='directory where results are saved')
    parser.add_argument('--cuda_device', default=0, type=int, help='Device number (default: 0, typically 0-3)')
    parser.add_argument('--no_scoring', action='store_true', help='Generate SSIM score and MSE loss unless this is set')
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

    sets_to_denoise = os.listdir(args.noisy_dir)

    denoised_save_dir=os.path.join(args.result_dir, model_path.split('/')[-2])
    os.makedirs(denoised_save_dir, exist_ok=True)
    for aset in sets_to_denoise:
        aset_indir = os.path.join(args.noisy_dir, aset)
        for animg in os.listdir(aset_indir):
            inimg_path = os.path.join(aset_indir, animg)
            outimg_path = os.path.join(denoised_save_dir, animg)
            cmd = ['python', 'denoise_image.py', '-i', inimg_path, '-o', outimg_path, '--model_path', model_path, '--ucs', '512', '--cs', '640']
            subprocess.call(cmd)
    if not args.no_scoring:
        gen_score(denoised_save_dir)
