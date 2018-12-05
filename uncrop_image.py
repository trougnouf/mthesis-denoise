import os
from PIL import Image
import math
import argparse
from math import floor

# Params
#parser = argparse.ArgumentParser(description='Images uncropper. Takes a directory containing files named "[FN]_[X_CROPS_NUM]_[Y_CROPS_NUM]_[USEFUL CROP SIZE].jpg" as input, USEFUL_CROP_SIZE being a single integer (eg: 96)')




def get_attr(fn, attr):
    if attr == 'xcnt':
        return int(fn.split('_')[-4])
    elif attr == 'ycnt':
        return int(fn.split('_')[-3])
    elif attr == 'ucs':
        return int(fn.split('_')[-2])
        

def uncrop(crop_dir):
    crops = [os.path.join(crop_dir, crop) for crop in os.listdir(crop_dir)]
    def get_crop_at(xcnt,ycnt):
        for crop in crops:
            if get_attr(crop, 'xcnt') == xcnt and get_attr(crop, 'ycnt') == ycnt:
                return crop
    lastx = max([get_attr(crop, 'xcnt') for crop in crops])
    lasty = max([get_attr(crop, 'ycnt') for crop in crops])
    remx = get_attr(get_crop_at(lastx, 1), 'ucs')
    remy = get_attr(get_crop_at(1, lasty), 'ucs')
    stducs = get_attr(get_crop_at(1, 1), 'ucs')
    cs = Image.open(get_crop_at(1, 1)).width
    width = lastx*stducs+remx
    height = lasty*stducs+remy
    newimg = Image.new('RGB', (width, height))
    for cropfn in crops:
        cropimg = Image.open(cropfn)

        absleft = get_attr(cropfn, 'xcnt')*stducs
        abstop = get_attr(cropfn, 'ycnt')*stducs
        absright = min(absleft+stducs, width)
        absbot = min(abstop+stducs, height)

        #croplength = absright - absleft
        #cropheight = absbottom - abstop

        cropleft = croptop = (cs-stducs)/2

        cropright = min(stducs+cropleft, absright-absleft+cropleft)
        cropbot = min(stducs+croptop, absbot-abstop+cropleft)
        cropimg = cropimg.crop((cropleft, croptop, cropright, cropbot))
        newimg.paste(cropimg, (absleft, abstop, absright, absbot))
    newpath = (crop_dir if crop_dir[-1]!='/' else crop_dir[:-1])+'.jpg'
    newimg.save(newpath, quality=100)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Images uncropper. Takes a directory [SETNAME] which contains files named "[NAME]_[X_CROP_NUM]_[Y_CROP_NUM]_[USEFUL_CROP_SIZE]_denoised.jpg. Outputs [SETNAME].jpg"')
    parser.add_argument('--crops_dir', type=str, help='Directory where crops to be stitched are located', required=True)
    args, _ = parser.parse_known_args()
    
    uncrop(args.crops_dir)
