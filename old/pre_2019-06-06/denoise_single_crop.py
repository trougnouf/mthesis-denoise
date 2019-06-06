# run this to test the model on a single crop
#TODO merge w/denoise_dir.py

import argparse
import os
import torch
import torchvision
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='directory where .th models are saved (latest .th file is autodetected)')
    parser.add_argument('--model_subdir', type=str, help='subdirectory where .th models are saved (latest .th file is autodetected, models dir is assumed)')
    parser.add_argument('--model_path', type=str, help='the specific model file path')
    parser.add_argument('--noisy_path', default='noisy.jpg', type=str, help='Path where noisy crop is located')
    parser.add_argument('--result_path', default='denoised.jpg', type=str, help='Path where result is saved')
    parser.add_argument('--cuda_device', default=0, type=int, help='Device number (default: 0, typically 0-3)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing images')
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

    torch.cuda.set_device(args.cuda_device)
    totensor = torchvision.transforms.ToTensor()
    print('loading '+ model_path)
    model = torch.load(model_path, map_location='cuda:'+str(args.cuda_device))
    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()
    
    def get_and_pad_image(path):
        inimg = Image.open(path)
        padamt = int((inimg.width*1.05)/2)
        newimg = Image.new('RGB', (int(inimg.width+padamt*2), int(inimg.height+padamt*2)))
        newimg.paste(inimg, (padamt, padamt))
        return totensor(newimg)
    y_ = get_and_pad_image(args.noisy_path)
    y_ = y_.view(1,-1,y_.shape[1], y_.shape[2])
    torch.cuda.synchronize()
    y_ = y_.cuda()
    x_ = model(y_)
    torch.cuda.synchronize()
    torchvision.utils.save_image(x_, args.result_path) 


