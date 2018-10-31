#!/usr/bin/env python
import os
import requests
imglist = [
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
    'MuseeL-avelo,200,500,3200,6400',
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
    'MuseeL-snakeAndMast,200,2000,6400,H1',
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
    'books,200,1600,6400,H1,H2'
    ]
os.makedirs('dataset', exist_ok=True)
os.chdir('dataset')
burl = 'https://commons.wikimedia.org/wiki/Special:Redirect/file/'
for img in imglist:
    name, *isos = img.split(',')
    os.makedirs(name, exist_ok=True)
    for iso in isos:
        fpath = name+'/NIND_'+name+'_ISO'+iso+'.jpg'
        if os.path.isfile(fpath):
            continue
        with open(fpath, 'wb') as f:
            f.write(requests.get(burl+fpath.split('/')[1]).content)
            print('Downloaded '+fpath.split('/')[1])
