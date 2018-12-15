#!/usr/bin/env python
import os
import requests
import argparse
import subprocess
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--use_wget', action='store_true', help="Use wget instead of python's request library (more likely to succeed)")
args = parser.parse_args()
nindlist = [
    'droid,200,800,3200,6400',
    'gnome,200,800,1600,6400',
    'Ottignies,200,640,3200,6400',
    'MuseeL-turtle,200,800,1250,6400',
    'MuseeL-centrifuge,200,800,2000,6400',
    'MuseeL-shell,400,800,6400',
    'MuseeL-coral,200,800,5000,6400',
    'MuseeL-head,200,640,3200,6400',
    'MuseeL-heads,200,400,3200,6400',
    'MuseeL-mask,640,4000,6400',
    'MuseeL-pig,200,500,2000,6400',
    'MuseeL-inspiredlotus,200,640,2500,6400',
    'MuseeL-pinklotus,200,800,4000,6400',
    'MuseeL-Armlessness,200,800,2000,6400',
    'MuseeL-byMarcGroessens,200,400,3200,6400',
    'MuseeL-SergeVandercam,200,320,1000,6400',
    'MuseeL-Moschophore,200,500,4000,6400',
    'MuseeL-AraPacis,200,800,4000,6400',
    'MuseeL-stele,200,640,1000,6400',
    'MuseeL-cross,200,500,4000,6400',
    'MuseeL-fuite,200,320,4000,6400',
    'MuseeL-RGB,200,250,1000,6400',
    'MuseeL-Vincent,200,400,2500,6400',
    'MuseeL-ambon,200,640,2500,6400',
    'MuseeL-ram,200,800,1000,6400',
    'MuseeL-Ghysels,200,800,6400',
    'MuseeL-pedestal,200,1250,5000,6400',
    'MuseeL-theatre,200,400,2500,6400',
    'MuseeL-text,200,400,5000,6400',
    'MuseeL-painting,200,800,3200,6400',
    'MuseeL-yombe,200,640,3200,6400',
    'MuseeL-roue,200,800,1250,6400',
    'MuseeL-hanging,200,500,4000,6400',
    'MuseeL-snakeAndMask,200,2000,6400,H1',
    'MuseeL-coral2,200,6400,H1,H2',
    'MuseeL-Vanillekipferl,200,6400,H1',
    'MuseeL-clam,200,6400,H1',
    'MuseeL-Ndengese,200,6400,H1',
    'MuseeL-Bobo,200,2500,6400,H1,H2',
    'threebicycles,200,6400',
    'sevenbicycles,200,1600,6400',
    'Stevin,200,4000,6400',
    'wall,200,640,6400',
    'Saint-Remi,200,6400,H1,H2,H3',
    'Saint-Remi-C,200,6400,H1,H2',
    'books,200,1600,6400,H1,H2',
    'bloop,200,3200,6400,H1',
    'schooltop,200,800,6400,H1,H2',
    'Sint-Joris,200,1000,2500,6400,H1,H2,H3',
    'claycreature,200,4000,6400,H1',
    'claycreatures,200,1600,5000,6400,H1,H2,H3',
    'claytools,200,5000,6400,H1,H2,H3',
    'CourtineDeVillersDebris,200,2500,6400,H1,H2',
    'Leonidas,200,400,3200,6400,H1',
    'pastries,200,3200,6400,H1,H2',
    'mugshot,200,6400,H1',
    'holywater,200,1600,4000,6400,H1,H2,H3',
    'chapel,200,1000,6400,H1,H2,H3',
    'directions,200,640,640-2,1250,6400,6400-2,H1,H2,H3',
    'drowning,200,800,6400,H1,H2,H3',
    'keyboard,200,400,800,1600,3200,6400,H1,H2,H3',
    'semicircle,200,320,640,1250,2500,5000,6400,H1,H2,H3',
    'stairs,200,250,320,640,1250,2500,5000,6400,H1,H2',
    'stefantiek,200,250,500,2000,6400,H1,H2',
    'tree1,200,400,1600,3200,6400,H1,H2,H3',
    'tree2,200,800,1600,3200,6400,H1,H2,H3',
    'ursulines-building,200,250,400,1000,4000,6400,H1',
    'ursulines-can,200,200-2,400,800,1600,3200,6400,H1,H2',
    'ursulines-red,200,250,500,4000,6400,H1,H2',
    'vlc,200,250,500,1000,3200,6400,H1,H2,H3',
    'whistle,200,250,500,1000,2000,4000,6400,H1,H2,H3,H4',
    ]
manset = [
    'lightclouds,0denoised,naturalnoise',
    'goldenhoursky,0denoised,naturalnoise'
    ]
os.makedirs('datasets/dataset', exist_ok=True)
os.chdir('datasets/dataset')
burl = 'https://commons.wikimedia.org/wiki/Special:Redirect/file/'
def download(imlist, dsname, targetdir, prefix='ISO'):
    for img in imlist:
        name, *isos = img.split(',')
        os.makedirs(name, exist_ok=True)
        for iso in isos:
            fpath = name+'/'+dsname+'_'+name+'_'+prefix+iso+'.jpg'
            url = burl+fpath.split('/')[1]
            if os.path.isfile(fpath):
                continue
            if args.use_wget:
                subprocess.run(['wget', url, '-O', fpath])
                # TODO add error checking
            else:
                with open(fpath, 'wb') as f:
                    f.write(requests.get(url).content)
                    print('Downloaded '+fpath.split('/')[1])
                    f.flush()
download(nindlist, 'NIND', 'datasets/dataset')
#download(manset, 'MAND', 'datasets/dataset', prefix='')
