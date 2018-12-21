import os
import argparse
import matplotlib.pyplot as plt
import numpy
from random import random
from math import floor
from dataset_torch_3 import sortISOs
import csv
import json

# Params
parser = argparse.ArgumentParser(description='Grapher')
parser.add_argument('--res_dir', default='results/test', type=str, help='Results directory')
#parser.add_argument('--res_bfn', default='train_result', type=str, help='Result base filename (s.t. [res_bfn]_[epoch].txt)')
parser.add_argument('--save_dir', default='graphs/auto', type=str, help='Where graphs are saved')
parser.add_argument('--nodisplay', action='store_true')
parser.add_argument('--nosave', action='store_true')
parser.add_argument('--xaxis', type=str, default='ISO')
parser.add_argument('--yaxis', type=str, default='SSIM')
parser.add_argument('--components', nargs='+', help='space-separated line components (eg Noisy NIND Artificial BM3D')
parser.add_argument('--metric', default='ssim', help='Metric shown (ssim, mse)')
#parser.add_argument('--width', default=15, type=int)
#parser.add_argument('--height', default=7, type=int)
parser.add_argument('--run', default=None, type=int, help="Generate a single graph number")
parser.add_argument('--noshow', action='store_true')
parser.add_argument('--nojson', action='store_true')
args = parser.parse_args()

data = dict()
#data = {label: {resfiles: [] , results: {image: {isoval {ssim: , mse, exp=None}}}
#match labels to fn
#load data
#generate a graph for image

components = args.components if args.components else ['Noisy', 'NIND', 'Artificial', 'BM3D']

def find_relevant_experiments(component):
    experiments = os.listdir(args.res_dir)
    data[component] = {'resfiles': []}  # check existing if data is needed earlier
    def add_exp_to_data(experiment):
        respath = os.path.join(args.res_dir, experiment, 'res.txt')
        if os.path.isfile(respath):
            data[component]['resfiles'].append(respath)
    if component == 'Noisy':
        return add_exp_to_data('GT')
    for experiment in experiments:
        if component == 'NIND' and 'run_nn.py' in experiment and '--yisx' not in experiment:
            add_exp_to_data(experiment)
        elif component == 'Artificial' and 'run_nn.py' in experiment and '--yisx' in experiment and '--sigmamax' in experiment:
            add_exp_to_data(experiment)
        elif component == 'BM3D' and 'bm3d' in experiment:
            add_exp_to_data(experiment)

def parse_resfiles(component):
    data[component]['results'] = {}
    for respath in data[component]['resfiles']:
        with open(respath) as f:
            for res in csv.reader(f):
                image,iso = res[0].split('_')[1:3]
                isoval = iso.split('.')[0]
                newssim, newmse = float(res[1]), float(res[2])
                oldssim, oldmse = 0.0, 1.0
                if image not in data[component]['results']:
                    data[component]['results'][image] = {}
                if isoval in data[component]['results'][image]:
                    oldssim = data[component]['results'][image][isoval]['ssim']
                    oldmse = data[component]['results'][image][isoval]['mse']
                if newssim > oldssim:
                    data[component]['results'][image][isoval] = {'ssim': float(res[1]), 'mse': float(res[2]), 'exp': respath.split('/')[-2]}

for component in components:
    find_relevant_experiments(component)
    parse_resfiles(component)

images = list(data[components[0]]['results'].keys())
for image in images:
    _, isos = sortISOs(list(data[components[0]]['results'][image].keys()))
    #isos = baseisos + isos
    for component in components:
        ssimscore = [data[component]['results'][image][iso][args.metric] for iso in isos]
        plt.plot(isos, ssimscore, label=component, marker='.')
        plt.title(image)
        if args.metric == 'ssim':
            plt.ylabel('SSIM score')
        else:
            plt.ylabel('MSE loss')
        plt.xlabel('ISO value')
    plt.grid()
    plt.legend()
    if not args.noshow:
        plt.show()
# TODO use json to handle nested dicts
if not args.nojson:
    with open('data.json', 'w') as f:
        json.dump(data, f)
