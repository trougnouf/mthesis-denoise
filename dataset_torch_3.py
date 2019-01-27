import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision
from random import randint, uniform, choice
from math import floor
from io import BytesIO
import torch

# Sort ISO values (eg ISO200, ISO6400, ...), handles ISOH1, ISOH2, ..., ISOHn as last, handles ISO200-n, ISO6400-n, ... as usable duplicates
def sortISOs(rawISOs):
    # ISO directories can be ISO<NUM>[-REPEATNUM] or ISOH<NUM>[-REPEATNSUM]
    # where ISO<lowest> (and possibly any -REPEATNUM, for example ISO200-1 and
    # ISO200-2) is taken as the base ISO. Other naming conventions will also work
    # so long as the base iso is first alphabetically
    if any([iso[3:] != 'ISO' for iso in rawISOs]):
        biso, *isos = sorted(rawISOs)
        return [biso], isos
    isos = []
    bisos = []
    hisos = []
    dupisos = {}
    for iso in rawISOs:
        if 'H' in iso:
            hisos.append(iso)
        else:
            if '-' in iso:
                isoval, repid = iso[3:].split('-')
                isos.append(int(isoval))
                if isoval in dupisos:
                    dupisos[isoval].append(repid)
                else:
                    dupisos[isoval] = [repid]
            else:
                isos.append(int(iso[3:]))
    bisos,*isos = sorted(isos)
    bisos = [bisos]
    # add duplicates
    while(bisos[0]==isos[0]):
        bisos.append(str(isos.pop(0))+'-'+dupisos[str(bisos[0])].pop())
    for dupiso in dupisos.keys():
        for repid in dupisos[dupiso]:
            isos[isos.index(int(dupiso))] = dupiso+'-'+repid
    hisos = sorted(hisos)
    isos = ['ISO'+str(iso) for iso in isos]
    bisos = ['ISO'+str(iso) for iso in bisos]
    isos.extend(hisos)
    return bisos, isos

class DenoisingDataset(Dataset):
    def __init__(self, datadirs, testreserve=[], yisx=False, compressionmin=100, compressionmax=100, sigmamin=0, sigmamax=0, test_reserve=[]):
        super(DenoisingDataset, self).__init__()
        self.totensor = torchvision.transforms.ToTensor()
        # each dataset element is ["<DATADIR>/<SETNAME>/ISOBASE/<DSNAME>_<SETNAME>_ISOBASE_<XNUM>_<YNUM>_<UCS>.EXT", [<ISOVAL1>,...,<ISOVALN>]]
        self.dataset = []
        self.cs, self.ucs = [int(i) for i in datadirs[0].split('_')[-2:]]
        self.compressionmin, self.compressionmax = compressionmin, compressionmax
        self.sigmamin, self.sigmamax = sigmamin, sigmamax
        for datadir in datadirs:
            for aset in os.listdir(datadir):
                if test_reserve and aset in test_reserve:
                    print('Skipped '+aset+' (test reserve)')
                    continue
                bisos, isos = sortISOs(os.listdir(os.path.join(datadir,aset)))
                if yisx:
                    bisos = isos = bisos[0:1]
                for animg in os.listdir(os.path.join(datadir, aset, isos[0])):
                    # verify that no base-ISO image exceeds CS just because
                    # TODO check time-cost

                    #if any(d > self.cs for d in Image.open(os.path.join(datadir, aset, isos[0], animg)).size):
                    #        print("Warning: excessive crop size for "+aset)
                    # check for min size
                    img4tests=Image.open(os.path.join(datadir, aset, isos[0], animg))
                    if all(d >= self.ucs for d in img4tests.size) and img4tests.getbands() == ('R', 'G', 'B'):
                        self.dataset.append([os.path.join(datadir,aset,'ISOBASE',animg).replace(isos[0]+'_','ISOBASE_'), bisos,isos])
                    else:
                        print('Skipping '+os.path.join(datadir, aset, isos[0], animg))
                print('Added '+aset+str(bisos)+str(isos)+' to the dataset')

    def get_and_pad(self, index):
        img = self.dataset[index]
        xchoice = choice(img[1])
        xpath = os.path.join(img[0].replace('ISOBASE_',xchoice+'_').replace('/ISOBASE/','/'+xchoice+'/'))
        ychoice = choice(img[2])
        ypath = os.path.join(img[0].replace('ISOBASE_',ychoice+'_').replace('/ISOBASE/','/'+ychoice+'/'))
        ximg = Image.open(xpath)
        yimg = Image.open(ypath)
        if all(d == self.cs for d in ximg.size):
            return (ximg, yimg)
        xnum, ynum, ucs = [int(i) for i in img[0].rpartition('.')[0].split('_')[-3:]]
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
        if self.compressionmin < 100:
            quality = randint(self.compressionmin, self.compressionmax)
            imbuffer = BytesIO()
            yimg.save(imbuffer, 'JPEG', quality=quality)
            yimg = Image.open(imbuffer)
        # return a tensor
        # PIL is H x W x C, totensor is C x H x W
        ximg, yimg = self.totensor(ximg), self.totensor(yimg)
        if self.sigmamax > 0:
            noise = torch.randn(yimg.shape).mul_(uniform(self.sigmamin, self.sigmamax)/255)
            yimg = torch.abs(yimg+noise)
        return ximg, yimg
    def __len__(self):
        return len(self.dataset)
