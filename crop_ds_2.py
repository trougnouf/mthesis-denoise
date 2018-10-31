#!/usr/bin/env python
from jpegtran import JPEGImage
import sys
import os
import math

print("Usage: python cropdataset.py [PATH] [CROPDIM]")
print("PATH must contain a directory named dataset, which contains a"),
print("directory per set of images named NIND_<aname>_ISO<value>.jpg")

CROPDIM = 64


def rename_img(fn, cnt):
    return fn.replace('.jpg', "_"+str(cnt)+".jpg")


def get_iso_vals(images_n):
    isovals = [None]*len(images_n)
    for i in range(0, len(images_n)):
        isovals[i] = images_n[i].split("_")[2].split(".")[0]
    return isovals


basepath = '.'
if len(sys.argv) > 1 and sys.argv[1]:
    basepath = sys.argv[1]
os.chdir(basepath)
if len(sys.argv) > 2:
    CROPDIM = int(sys.argv[2])
imagesets = os.listdir("dataset")
newds_n = "dataset_"+str(CROPDIM)
if not os.path.exists(newds_n):
    os.mkdir(newds_n)
for imageset_n in imagesets:
    if not os.path.exists(newds_n+"/"+imageset_n):
        os.mkdir(newds_n+"/"+imageset_n)
    images_n = os.listdir("dataset/"+imageset_n)
    isovals = get_iso_vals(images_n)
    for isoval in isovals:
        if not os.path.exists(newds_n+"/"+imageset_n+"/"+isoval):
            os.mkdir(newds_n+"/"+imageset_n+"/"+isoval)
    images_in = [None]*len(images_n)
    for i in range(0, len(images_n)):
        images_in[i] = JPEGImage("dataset/"+imageset_n+"/"+images_n[i])
    print(imageset_n)
    image_width = math.floor(images_in[0].width/CROPDIM)*CROPDIM
    image_height = math.floor(images_in[0].height/CROPDIM)*CROPDIM
    curx = cury = cnt = 0
    while cury < image_height:
        for i in range(0, len(images_n)):
            if not os.path.exists(newds_n+"/"+imageset_n+"/"+isovals[i]+"/"
                                  + rename_img(images_n[i], cnt)):
                images_in[i].crop(curx, cury, CROPDIM, CROPDIM)\
                    .save(newds_n+"/"+imageset_n+"/"+isovals[i]+"/"
                          + rename_img(images_n[i], cnt))
        cnt += 1
        curx += CROPDIM
        if curx == image_width:
            curx = 0
            cury += CROPDIM
