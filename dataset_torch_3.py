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
        # each dataset element is ["<SETNAME>/ISOBASE/<DSNAME>_<SETNAME>_ISOBASE_<XNUM>_<YNUM>_<UCS>.jpg", [<ISOVAL1>,...,<ISOVALN>]]
        self.dataset = []
        self.cs, self.ucs = [int(i) for i in datadir.split('_')[-2:]]
        def sortISOs(rawISOs):
            if any([i[0]=='0' for i in rawISOs]):
                return sorted(rawISOs)
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
                # check for min size
                if all(d >= self.ucs for d in Image.open(os.path.join(datadir, aset, isos[0], animg)).size):
                    self.dataset.append([os.path.join(aset,'ISOBASE',animg).replace('_'+isos[0]+'_','_ISOBASE_'), isos])

    def get_and_pad(self, index):
        img = self.dataset[index]
        xpath = os.path.join(self.datadir, img[0].replace('_ISOBASE_','_'+img[1][0]+'_').replace('/ISOBASE/','/'+img[1][0]+'/'))
        ypath = os.path.join(self.datadir, img[0].replace('_ISOBASE_','_'+choice(img[1][1:])+'_').replace('/ISOBASE/','/'+choice(img[1][1:])+'/'))
        ximg = Image.open(xpath)
        yimg = Image.open(ypath)
        if all(d == self.cs for d in ximg.size):
            return (ximg, yimg)
        xnum, ynum, ucs = [int(i) for i in img[0].strip('.jpg').split('_')[-3:]]
        if xnum == 0:
            # pad left
            ximg = ximg.crop((-self.cs+ximg.width, 0, ximg.width, ximg.height))
            yimg = yimg.crop((-self.cs+yimg.width, 0, yimg.width, yimg.height))
        if ynum == 0:
            # pad top
            ximg = ximg.crop((0, -self.cs+ximg.height, ximg.width, ximg.height))
            yimg = yimg.crop((0, -self.cs+yimg.height, yimg.width, yimg.height))
        if ximg.width < self.cs or ximg.height < self.cs:
            # pad right and bottom
            ximg = ximg.crop((0, 0, self.cs, self.cs))
            yimg = yimg.crop((0, 0, self.cs, self.cs))
        return (ximg, yimg)

    def __getitem__(self, reqindex):
        ximg, yimg = self.get_and_pad(reqindex)
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
