import os
import argparse
import matplotlib.pyplot as plt
import numpy
from random import random
from math import floor
from dataset_torch_3 import sortISOs
import csv

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
#parser.add_argument('--width', default=15, type=int)
#parser.add_argument('--height', default=7, type=int)
parser.add_argument('--run', default=None, type=int, help="Generate a single graph number")
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
        ssimscore = [data[component]['results'][image][iso]['ssim'] for iso in isos]
        plt.plot(isos, ssimscore, label=component, marker='.')
        plt.title(image)
        plt.ylabel('SSIM score')
        plt.xlabel('ISO value')
    plt.grid()
    plt.legend()
    plt.show()
       # print(ssimscore)


# x = component image



#if args.xaxis == 'ISO':
    # TODO moved to named function
 #   dirs = os.listdir(args.res_dir)
    #for component in args.legend:





#class Feature:
#    def __init__(feature_names)

# def get_res_i(path):
#     indices = []
#     for fn in os.listdir(path):
#         if args.res_bfn in fn:
#             indices.append(int(fn.split('_')[-1].split('.')[0]))
#     return sorted(indices)


#for expname in os.listdir(args.res_dir):
#    expreslist = []
#    for i in get_res_i(os.path.join(args.res_dir, expname)):
#        ires = dict()
#        with open(os.path.join(args.res_dir, expname, args.res_bfn+'_'+str(i)+'.txt')) as f:
#            ires['epoch'] = float(f.readline())
#            ires['loss'] = float(f.readline())
#            ires['time'] = float(f.readline())
#            if i > 0:
#                ires['time'] += expreslist[-1]['time']
#            expreslist.append(ires)
#    results[expname] = expreslist

# for expname in os.listdir(args.res_dir):
    #expreslist = []
#     epochl = []
#     trainlossl = []
#     testlossl = []
#     timel = []
#     for i in get_res_i(os.path.join(args.res_dir, expname)):
        #ires = dict()
#         with open(os.path.join(args.res_dir, expname, args.res_bfn+'_'+str(i)+'.txt')) as f:
#             epochl.append(float(f.readline()))
#             trainlossl.append(float(f.readline()))
#             testlossl.append(float(f.readline()))
#             timel.append(float(f.readline()))

#     results[expname] = {'epochs': epochl, 'trainloss': trainlossl, 'testloss': testlossl, 'timel': timel}

# def extract_feature_val(expname, feature_n):
#     if type(feature_n) == str:
#         return expname.split(feature_n)[-1].split('_')[1] if len(expname.split(feature_n))>1 else 'default'
#     elif type(feature_n) == list:
#         return ', '.join([expname.split(afeature)[-1].split('_')[1] if len(expname.split(afeature))>1 else 'default' for afeature in feature_n])


# def graph(experiments, feature_n, save=True):
#     for exp in experiments:
#         color = [[random(), random(), random()]]
#         feature_v = extract_feature_val(exp, feature_n)
#         if 'lr' not in feature_n and 'initial_input' not in feature_n and 'n_channels' not in feature_n:
#             plt.scatter(results[exp]["timel"], results[exp]["trainloss"], label=feature_v+' (training)', c=color, marker="^")
#         plt.scatter(results[exp]["timel"], results[exp]["testloss"], label=feature_v+' (testing)', c=color, marker="v")
#         plt.grid()
#     plt.xlabel("time (s)")
#     plt.ylabel("loss")

    #plt.axis([0,300,0,1])
#     plt.legend(title=str(feature_n))
#     if not args.nosave:
        #plt.figure(figsize=(args.width,args.height))
#         os.makedirs(args.save_dir, exist_ok=True)
#         plt.savefig(os.path.join(args.save_dir, ''.join(feature_n)+'.png'))
#     if not args.nodisplay:
#         plt.show()
    # plotting

# params = [
#0
# {
#     'experiments': [
#         '2018-12-09T19:37_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_DnCNN_--time_limit_300',
#         '2018-12-09T19:42_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300',
#         '2018-12-09T19:47_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300'],
#     'feature_n': 'model'
# }, {
#     'experiments': [
#         '2018-12-09T19:52_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300_--initial_input_xwithmask_--network_input_random10',
#         '2018-12-09T19:57_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300_--initial_input_xwithmask_--network_input_prevoutput',
#         '2018-12-09T20:02_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300_--initial_input_xwithmask_--network_input_initial',
#         '2018-12-09T20:07_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300_--initial_input_random10_--network_input_prevoutput',
#         '2018-12-09T20:12_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300_--initial_input_random10_--network_input_initial',
#         '2018-12-09T19:47_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300_--initial_input_random10_--network_input_random10'],
#     'feature_n': ['initial_input', 'network_input']
# }, {
#     'experiments': [
#         '2018-12-09T20:17_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0002_--model_UNet_--time_limit_300',
#         '2018-12-09T20:36_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0003_--model_UNet_--time_limit_300',
#         '2018-12-09T20:41_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0004_--model_UNet_--time_limit_300',
#         '2018-12-09T20:46_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0005_--model_UNet_--time_limit_300',
#         '2018-12-09T20:51_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.000075_--model_UNet_--time_limit_300',
#         '2018-12-09T20:57_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.00005_--model_UNet_--time_limit_300',
#         '2018-12-09T21:02_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.000025_--model_UNet_--time_limit_300',
#         '2018-12-09T21:07_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.00001_--model_UNet_--time_limit_300',
#         '2018-12-09T19:47_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300',
#         '2018-12-10T01:55_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.001_--model_UNet_--time_limit_300',
#         '2018-12-10T02:01_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.01_--model_UNet_--time_limit_300'
#     ], 'feature_n': 'lr'
# }, {
#     'experiments': [
#         '2018-12-09T21:12_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--depth_20',
#         '2018-12-09T21:17_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--depth_18',
#         '2018-12-09T21:22_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--depth_16',
#         '2018-12-09T19:42_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--depth_22'
#     ], 'feature_n': 'depth'
# }, {
#     'experiments': [
#         '2018-12-09T21:27_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--n_channels_112',
#         '2018-12-09T21:32_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--n_channels_96',
#         '2018-12-09T21:37_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--n_channels_80',
#         '2018-12-09T21:42_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--n_channels_144',
#         '2018-12-09T21:47_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--n_channels_160',
#         '2018-12-09T21:52_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--n_channels_176',
#         '2018-12-09T19:42_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--n_channels_128',
#         '2018-12-10T02:13_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--n_channels_32'
#     ], 'feature_n': 'n_channels'
# }, {
#     'experiments': [
#         '2018-12-09T21:57_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--kernel_size_3',
#         '2018-12-09T19:42_run_nn.py_--img_path_dataset-64-kate.png_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_RedCNN_--time_limit_300_--kernel_size_5'
#     ], 'feature_n': 'kernel_size'
# }, {
#     'experiments': [
#         '2018-12-09T22:02_run_nn.py_--img_path_dataset-64-NIND_MuseeL-snakeAndMask_ISO200_554.jpg_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300',
#         '2018-12-09T22:07_run_nn.py_--img_path_dataset-64-NIND_MuseeL-snakeAndMask_ISO2000_554.jpg_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300',
#         '2018-12-09T22:12_run_nn.py_--img_path_dataset-64-NIND_MuseeL-snakeAndMask_ISO6400_554.jpg_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300',
#         '2018-12-09T22:17_run_nn.py_--img_path_dataset-64-NIND_MuseeL-snakeAndMask_ISOH1_554.jpg_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300'
#     ], 'feature_n': 'snakeAndMask'
# }, {
#     'experiments': [
#         '2018-12-09T22:22_run_nn.py_--img_path_dataset-64-NIND_MuseeL-snakeAndMask_ISO200_641.jpg_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300',
#         '2018-12-09T22:27_run_nn.py_--img_path_dataset-64-NIND_MuseeL-snakeAndMask_ISO2000_641.jpg_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300',
#         '2018-12-09T22:32_run_nn.py_--img_path_dataset-64-NIND_MuseeL-snakeAndMask_ISO6400_641.jpg_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300',
#         '2018-12-09T22:37_run_nn.py_--img_path_dataset-64-NIND_MuseeL-snakeAndMask_ISOH1_641.jpg_--mask_path_masks-64-mask1notrans.png_--lr_0.0001_--model_UNet_--time_limit_300'
#     ], 'feature_n': 'snakeAndMask'
# }]

# if args.run is None:
#     for exp in params:
#         graph(**exp)
# else:
#     graph(**params[args.run])
