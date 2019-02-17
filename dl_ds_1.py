#!/usr/bin/env python
import os
import requests
import argparse
import subprocess
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--use_wget', action='store_true', help="Use wget instead of python's request library (more likely to succeed)")
parser.add_argument('--target_dir', default='datasets/NIND', type=str, help="Target directory (default: datasets/NIND)")
args = parser.parse_args()
nindlist8bit = [
    'droid,200,800,3200,6400',
    'gnome,200,800,1600,6400',
    'Ottignies,200,640,3200,6400',
    'MuseeL-turtle,200,800,1250,6400',
    'MuseeL-centrifuge,200,800,2000,6400',
    'MuseeL-shell,200,400,800,6400',
    'MuseeL-coral,200,800,5000,6400',
    'MuseeL-head,200,640,3200,6400',
    'MuseeL-heads,200,400,3200,6400',
    'MuseeL-mask,640,4000,6400',
    'MuseeL-pig,200,500,2000,6400',
    'MuseeL-inspiredlotus,200,640,2500,6400',
    'MuseeL-pinklotus,200,800,4000,6400',
    'MuseeL-Armlessness,200,800,2000,6400',
    'MuseeL-byMarcGroessens,200,400,3200,6400',
    'MuseeL-Moschophore,200,500,4000,6400',
    'MuseeL-AraPacis,200,800,4000,6400',
    'MuseeL-stele,200,640,1000,6400',
    'MuseeL-cross,200,500,4000,6400',
    'MuseeL-fuite,200,320,4000,6400',
    'MuseeL-RGB,200,250,1000,6400',
    'MuseeL-Vincent,200,400,2500,6400',
    'MuseeL-ambon,200,640,2500,6400',
    'MuseeL-ram,200,800,1000,6400',
    'MuseeL-pedestal,200,1250,5000,6400',
    'MuseeL-theatre,200,400,2500,6400',
    'MuseeL-text,200,400,5000,6400',
    'MuseeL-painting,200,800,3200,6400',
    'MuseeL-yombe,200,640,3200,6400',
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
    'parking-keyboard,200,400,800,1600,3200,6400,H1,H2,H3',
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
    'Homarus-americanus,200,200-2,250,400,800,2000,3200,5000,6400,H1,H2',
    'fruits,200,200-2,800,3200,5000,6400',
    'MVB-Sainte-Anne,200,200-2,250,640,4000,6400,H1',
    'MVB-JardinBotanique,200,200-2,400,1000,2500,3200,6400',
    'MVB-Urania,200,320,500,1000,2500,5000,6400,H1,H2',
    'MVB-1887GrandPlace,200,200-2,400,640,2000,5000,6400,H1,H2',
    'MVB-heraldicLion,200,200-2,320,1000,3200,6400,H1,H2',
    'MVB-LouveFire,200,200-2,400,800,1600,3200,6400,6400-2,H1,H2',
    'MVB-Bombardement,200,200-2,320,800,5000,6400,H1,H2,H3',
    'beads,200,500,1000,3200,6400',
    'shells,200,200-2,250,320,1000,1600,2500,3200,5000,6400,H1,H2,H3',
    'MuseeL-Bobo-C500D,100,200,400,800,1600,3200,H1',
    'MuseeL-yombe-C500D,100,400,800,1600,3200,H1',
    'MuseeL-sol-C500D,100,200,400,800,3200,H1',
    'MuseeL-skull-C500D,100,200,400,800,1600,3200,H1',
    'MuseeL-Sepik-C500D,100,200,800,1600,3200,H1',
    'MuseeL-Saint-Pierre-C500D,100,100-2,200,400,800,1600,3200,H1',
    'MuseeL-mammal-C500D,100,200,400,800,1600,3200,H1',
    'MuseeL-idole-C500D,100,100-2,200,400,800,3200,H1',
    'MuseeL-CopteArch-C500D,100,100-2,200,400,1600,3200',
    'MuseeL-cross-C500D,100,200,400,800,1600,3200,H1',
    'MuseeL-fuite-C500D,100,200,400,800,1600,3200,H1',
    ]
nindlist16bit = [
    'soap,200,200-2,400,800,3200,6400,H1,H2,H3,H4',
    'kibbles,200,200-2,800,5000,6400,H1,H2,H3',
    'bertrixtree,200,400,640,2500,4000,6400,H1',
    'BruegelLibraryS1,200,400,1000,2500,3200,5000,6400,H1,H2',
    'BruegelLibraryS2,200,500,1250,2500,5000,6400,H1,H2,H3,H4',
    'LaptopInLibrary,200,500,800,1600,2500,6400,H1,H2,H3',
    'banana,200,250,500,800,1250,2000,4000,6400,H1,H2,H3',
    'dustyrubberduck,200,1000,1250,2500,5000,6400,H1,H2',
    'partiallyeatenbanana,200,640,1250,2500,4000,5000,6400,H1,H2,H3',
    'corkboard,200,320,1000,2500,5000,6400,H1,H2,H3',
    'fireextinguisher,200,200-2,200-3,800,3200,6400,H1,H2,H3',
    ]
os.makedirs(args.target_dir, exist_ok=True)
os.chdir(args.target_dir)
burl = 'https://commons.wikimedia.org/wiki/Special:Redirect/file/'
def download(imlist, dsname, targetdir, ext, prefix='ISO'):
    for img in imlist:
        name, *isos = img.split(',')
        os.makedirs(name, exist_ok=True)
        for iso in isos:
            fpath = name+'/'+dsname+'_'+name+'_'+prefix+iso+'.'+ext
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
download(nindlist8bit, 'NIND', args.target_dir, ext='jpg')
download(nindlist16bit, 'NIND', args.target_dir, ext='png')
