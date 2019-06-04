#!/usr/bin/env python
from jpegtran import JPEGImage
import sys
import os
import math
from multiprocessing import Pool, TimeoutError
import time
import multiprocessing as mp

def rename_img(fn, cnt):
    return fn.replace('.jpg', "_"+str(cnt)+".jpg")


def get_iso_vals(images_n):
    isovals = [None]*len(images_n)
    for i in range(0, len(images_n)):
        isovals[i] = images_n[i].split("_")[2].split(".")[0]
    return isovals

def crop_an_imageset(imageset_n):
    print(CROPDIM)
    print(os.getcwd())
    os.makedirs(imageset_n, exist_ok=True)
    images_n = os.listdir("../dataset/"+imageset_n)
    isovals = get_iso_vals(images_n)
    for isoval in isovals:
        os.makedirs(imageset_n+"/"+isoval, exist_ok=True)
    images_in = [None]*len(images_n)
    for i in range(0, len(images_n)):
        images_in[i] = JPEGImage("../dataset/"+imageset_n+"/"+images_n[i])
    print(imageset_n)
    image_width = math.floor(images_in[0].width/CROPDIM)*CROPDIM
    image_height = math.floor(images_in[0].height/CROPDIM)*CROPDIM
    curx = cury = cnt = 0
    while cury < image_height:
        for i in range(0, len(images_n)):
            if not os.path.exists(imageset_n+"/"+isovals[i]+"/"+rename_img(images_n[i], cnt)):
                images_in[i].crop(curx, cury, CROPDIM, CROPDIM).save(imageset_n+"/"+isovals[i]+"/"
                                                                     + rename_img(images_n[i], cnt))
        cnt += 1
        curx += CROPDIM
        if curx == image_width:
            curx = 0
            cury += CROPDIM
    return 1


if __name__ == '__main__':
    NUMTHREADS = mp.cpu_count()
    if len(sys.argv) > 3:
        NUMTHREADS = int(sys.argv[3])
    with Pool(processes=NUMTHREADS) as pool:
        CROPDIM = 64
        BASEPATH = '.'
        if len(sys.argv) > 1 and sys.argv[1]:
            BASEPATH = sys.argv[1]
        os.chdir(BASEPATH)
        if len(sys.argv) > 2:
            CROPDIM = int(sys.argv[2])
        print("Usage: python cropdataset.py [PATH] [CROPDIM] [NUMTHREADS]")
        print("PATH must contain a directory named dataset, which contains a"),
        print("directory per set of images named NIND_<aname>_ISO<value>.jpg")
        print("default: python cropdataset.py . 64 cpu_count()")
        imagesets = os.listdir("dataset")
        newds_n = "dataset_"+str(CROPDIM)
        os.makedirs(newds_n, exist_ok=True)
        os.chdir(newds_n)
        print(os.getcwd())
        r = pool.imap_unordered(crop_an_imageset, imagesets)
        [print(p) for p in r]
        time.sleep(10);

        time.sleep(10);
