import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from random import choice
import torchvision
from random import randint
from math import floor
from io import BytesIO

class DenoisingDataset(Dataset):
    def __init__(self, datadir, docompression=False):
        super(DenoisingDataset, self).__init__()
        self.docompression = docompression
        self.totensor = torchvision.transforms.ToTensor()
        self.datadir = datadir
        self.dataset = []
        cs = int(datadir.split('_')[-2])
        def sortISOs(rawISOs):
            isos = []
            hisos = []
            for iso in rawISOs:
                hisos.append(iso) if 'H' in iso else isos.append(int(iso[3:]))
            isos = sorted(isos)
            hisos = sorted(hisos)
            isos = ['ISO'+str(iso) for iso in isos]
            isos.extend(hisos)
            return isos
        for aset in os.listdir(datadir):
            isos = sortISOs(os.listdir(os.path.join(datadir,aset)))
            for animg in os.listdir(os.path.join(datadir, aset, isos[0])):
                if Image.open(os.path.join(datadir, aset, isos[0], animg)).size == (cs, cs):
                    self.dataset.append([os.path.join(aset,'ISOBASE',animg).replace(isos[0],'ISOBASE'), isos])

    def __getitem__(self, reqindex):
        img = self.dataset[reqindex]
        xpath = os.path.join(self.datadir, img[0].replace('ISOBASE',img[1][0]))
        ypath = os.path.join(self.datadir, img[0].replace('ISOBASE',choice(img[1][1:])))
        ximg = Image.open(xpath)
        yimg = Image.open(ypath)
        # data augmentation
        random_decision = randint(0, 99)
        if random_decision % 10 == 0:
            ximg = ximg.rotate(90)
            yimg = yimg.rotate(90)
        elif random_decision % 10 == 1:
            ximg = ximg.rotate(180)
            yimg = yimg.rotate(180)
        elif random_decision % 10 == 2:
            ximg = ximg.rotate(270)
            yimg = yimg.rotate(270)
        if floor(random_decision/10) == 0 or floor(random_decision/10) == 2:
            ximg = ImageOps.flip(ximg)
            yimg = ImageOps.flip(yimg)
        if floor(random_decision/10) == 1 or floor(random_decision/10) == 2:
            ximg = ImageOps.mirror(ximg)
            yimg = ImageOps.mirror(yimg)
        if self.docompression:
            if self.docompression=='random':
                quality = randint(1,100)
            else:
                quality = int(self.docompression)
            imbuffer = BytesIO()
            yimg.save(imbuffer, 'JPEG', quality=quality)
            yimg = Image.open(imbuffer)
        # return a tensor
        # PIL is H x W x C, totensor is C x H x W
        return (self.totensor(ximg), self.totensor(yimg))

    def __len__(self):
        return len(self.dataset)
