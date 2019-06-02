import os
import argparse
import matplotlib.pyplot as plt
import numpy
from random import random
from math import floor
from dataset_torch_3 import sortISOs
import csv
import json
import graph_utils

# generate results from log file
# python graph_logfiles.py --experiment Hul112Disc_activation --yaxis "Average MSE" --uneven_graphs
# python graph_logfiles.py --experiment "Hul(b)128Net" --yaxis "Average 1-SSIM loss" --smoothing_factor 20 --uneven_graphs
# python graph_logfiles.py --experiment "Label smoothing" --yaxis "Average MSE loss (discriminator)" --pre "Loss_D: " --post " (" --smoothing_factor 2500 --uneven_graphs
default_smoothing_factor = 50

# Params
parser = argparse.ArgumentParser(description='Grapher')
parser.add_argument('--xaxis', type=str)
parser.add_argument('--yaxis', type=str, default='Average MSE')
parser.add_argument('--experiment', type=str)
parser.add_argument('--list_experiments', action='store_true')
parser.add_argument('--smoothing_factor', type=int, default=default_smoothing_factor)
parser.add_argument('--uneven_graphs', action='store_true', help='Allow some graphs to display more data points')
parser.add_argument('--pre', type=str, default="What precedes the variable to graph")
parser.add_argument('--post', type=str, default="What follows the variable to graph")
args = parser.parse_args()

if args.xaxis is None:
    xaxis = "iterations * %u"%(args.smoothing_factor)
else:
    xaxis = args.xaxis

experiments = dict()
# these should be classes w/ all the parameters, oh well
experiments['Hul112Disc_activation'] = {
#    "None-bc": "results/train/2019-05-22-D_sanity_check_not_conditional_No_activation",
#    "Sigmoid-bc": "results/train/2019-05-22-D_sanity_check_not_conditional_Sigmoid",
#    "PReLU-bc": "results/train/2019-05-22-D_sanity_check_not_conditional",
#    "Final pooling-bc": "results/train/2019-05-22-D_sanity_check_finalpool",
    "PReLU FA, funit=16, final max pooling": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16_finalpool",
    "Sigmoid FA, funit=16":"results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16_sigmoid",
    "None FA, funit=16": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16_no_activation",
"PReLU FA, funit=16, final max pooling, no BN": "results/train/2019-05-26-D_sanity_check_Hulb112Disc_fixed_funit16_finalpool",
"PReLU FA, funit=16":"results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16",
"PReLU FA, funit=32":"results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit32",
"PReLU FA, funit=16, no BN":"results/train/2019-05-26-D_sanity_check_Hulb112Disc_fixed_funit16",
"PReLU FA, funit=32, no BN":"results/train/2019-05-26-D_sanity_check_Hulb112Disc_fixed_funit32",
#"LeakyReLU activations,  funit=24":"results/train/2019-05-26-D_sanity_check_Hull112Disc_fixed_funit24_LeakyReLU",
"LeakyReLU FA, funit=24": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit24_LeakyReLU",
"LeakyReLU FA, funit=24, final max pooling": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit24_LeakyReLU_finalpool"
}
experiments['Hul(b)128Net'] = {"BN, PReLU": "results/train/2019-05-23-Hulb128Net-BN",
"No BN, PReLU": "results/train/2019-05-23-Hulb128Net-NoBN",
"No BN, no final activation": "results/train/2019-05-23-Hulb128Net-NoBN-NoAct"}

experiments["Label smoothing"] = {
"Always noisy labels": "results/train/2019-05-28-Hulb128Disc-very_noisy_probabilities",
"Noisy positive labels": "results/train/2019-05-28-Hulb128Disc-std_noisy_probabilities"
}

if args.list_experiments:
    print(experiments.keys())
    exit(0)

experiment_raw = experiments[args.experiment]

data = dict()
markers = graph_utils.make_markers_dict(components=experiment_raw.keys())
smallest_log = None

for component, path in experiment_raw.items():
    data[component] = graph_utils.parse_log_file(path, smoothing_factor = args.smoothing_factor, pre=args.pre, post=args.post)
    if smallest_log is None or smallest_log > len(data[component]):
        smallest_log = len(data[component])
for component in data:
    if not args.uneven_graphs:
        data[component] = data[component][0:smallest_log]
    plt.plot(data[component], label=component, marker = markers[component])#, marker=markers[component])
plt.title(args.experiment)
plt.ylabel(args.yaxis)
plt.xlabel(xaxis)
plt.grid()
plt.legend()
plt.show()

# images = list(data[components[0]]['results'].keys())
# for image in images:
#     _, isos = sortISOs(list(data[components[0]]['results'][image].keys()))
    #isos = baseisos + isos
#     for component in components:
#         try:
#             ssimscore = [data[component]['results'][image][iso][args.metric] for iso in isos]
#             plt.plot(isos, ssimscore, label=component, marker=markers[component])
#             plt.title(image)
#             if args.metric == 'ssim':
#                 plt.ylabel('SSIM score')
#             else:
#                 plt.ylabel('MSE loss')
#             plt.xlabel('ISO value')
#         except KeyError as err:
#             print(err)
#             continue
#     plt.grid()
#     plt.legend()
#     if not args.noshow:
#         plt.show()
# TODO use json to handle nested dicts
# if not args.nojson:
#     with open('data.json', 'w') as f:
#         json.dump(data, f)
