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
        self.datadir = datadir
        self.images = []
        self.start_indices = [0]
        self.Dsize = 0
        for iname in os.listdir(datadir):
            print(iname)
            isos = sortISOs(os.listdir(datadir+'/'+iname))
            ncrops = len(os.listdir(datadir+'/'+iname+'/'+str(isos[0])))
            self.Dsize += ncrops
            curimg = {'name': iname, 'biso': str(isos[0]), 'isos': isos[1:], 'ncrops': ncrops}
            self.images.append(curimg)

    def __getitem__(self, reqindex):
        # find crop #
        i = 0
        for img in self.images:
            if i+img['ncrops'] > reqindex:
                crop_i = reqindex-i
                break
            i += img['ncrops']
        # get image file
        try:
            ximg = Image.open(self.datadir+'/'+img['name']+'/'+img['biso']
                              + '/NIND_'+img['name']+'_'+img['biso']+'_'
                              + str(crop_i)+'.jpg')
        except FileNotFoundError:
            print(' i: ');print(i)
            print(' reqindex: ');print(reqindex)
            print(' crop_i: ');print(crop_i)
            print(self.images)
        niso = choice(img['isos'])
        yimg = Image.open(self.datadir+'/'+img['name']+'/'+niso+'/NIND_'
                          + img['name']+'_'+niso+'_'+str(crop_i)+'.jpg')
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
        return self.Dsize
