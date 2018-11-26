import os
from PIL import Image
import math
import argparse

# Params
parser = argparse.ArgumentParser(description='Images uncropper')
parser.add_argument('--crop_dir', default='results/test', type=str, help='Directory where crops to be stitched are located')
parser.add_argument('--ds_dir', default='datasets/dataset', type=str, help='Directory with matching original images to get dimensions')
args = parser.parse_args()

def find_original_img(curitem):
    orname=curitem.split('/')[-1].split('_0_')[0]+'.jpg'
    for root, dirs, files in os.walk(args.ds_dir):
        if orname in files:
            return os.path.join(root, orname)
    print(orname+' not found, skipping.')
    return 404
def find_res(crop_img, fullsizeimg):
    cs = Image.open(crop_img).width
    res = Image.open(fullsizeimg).size
    return tuple([math.floor(i/cs)*cs for i in res])
def uncrop(crop_img_0, res):
    cs = Image.open(crop_img_0).width
    newimg = Image.new('RGB', res)
    for curx in range(int(res[0]/cs)):
        for cury in range(int(res[1]/cs)):
            i = str(int(curx+cury*(res[0]/cs)))
            curcrop =  ('_'+i+'_').join(crop_img_0.split('_0_'))
            newimg.paste(Image.open(curcrop), box=(curx*cs, cury*cs,  curx*cs+cs, cury*cs+cs))
    #newpath = '_'.join(crop_img_0.split('_0_'))
    newpath = '/'.join(crop_img_0.split('/')[:-2]+['_'.join((crop_img_0.split('/')[-1]).split('_0_'))])

    newimg.save(newpath, quality=100)
    print('Uncropped '+newpath+' ('+str(res)+')')
    return 0

todolist = [os.path.join(args.crop_dir, i) for i in os.listdir(args.crop_dir)]
while len(todolist):
    curitem = todolist.pop()
    if os.path.isdir(curitem):
        todolist.extend([os.path.join(curitem, i) for i in os.listdir(curitem)])
        continue
    if curitem[-3:] != 'jpg' and curitem[-3:] != 'png':
        continue
    if curitem.split('_')[-2] == '0':
        fullsizeimg = find_original_img(curitem)
        if fullsizeimg == 404:
            continue
        res = find_res(curitem, fullsizeimg)
        uncrop(curitem, res)
