import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
#from skimage import io
from random import choice
import torchvision

class DenoisingDataset(Dataset):
    def __init__(self, datadir):
        super(DenoisingDataset, self).__init__()
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
        i = 0
        for img in self.images:
            if i+img['ncrops'] > reqindex:
                crop_i = reqindex-i
                break
            i += img['ncrops']
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
        # PIL is H x W x C, totensor is C x H x W
        return (self.totensor(ximg), self.totensor(yimg))

    def __len__(self):
        return self.Dsize
