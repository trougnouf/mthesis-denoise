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
# python graph_logfiles.py --experiment Hul112_activation --yaxis "Average MSE" --uneven_graphs
# python graph_logfiles.py --experiment Hulb128 --yaxis "Average SSIM"

default_smoothing_factor = 50

# Params
parser = argparse.ArgumentParser(description='Grapher')
parser.add_argument('--xaxis', type=str, default="%u iterations"%(default_smoothing_factor))
parser.add_argument('--yaxis', type=str, default='Average MSE')
parser.add_argument('--experiment', type=str)
parser.add_argument('--list_experiments', action='store_true')
parser.add_argument('--smoothing_factor', type=int, default=default_smoothing_factor)
parser.add_argument('--uneven_graphs', action='store_true', help='Allow some graphs to display more data points')
args = parser.parse_args()

experiments = dict()
experiments['Hul112_activation'] = {
#    "None-bc": "results/train/2019-05-22-D_sanity_check_not_conditional_No_activation",
#    "Sigmoid-bc": "results/train/2019-05-22-D_sanity_check_not_conditional_Sigmoid",
#    "PReLU-bc": "results/train/2019-05-22-D_sanity_check_not_conditional",
#    "Final pooling-bc": "results/train/2019-05-22-D_sanity_check_finalpool",
    "Final pooling-16": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16_finalpool",
    "Sigmoid-16":"results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16_sigmoid",
    "None-16": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16_no_activation",
"Final pooling-noBN-16": "results/train/2019-05-26-D_sanity_check_Hulb112Disc_fixed_funit16_finalpool",
"PReLU-16":"results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit16",
"PReLU-32":"results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit32",
"PReLU-noBN-16":"results/train/2019-05-26-D_sanity_check_Hulb112Disc_fixed_funit16",
"PReLU-noBN-32":"results/train/2019-05-26-D_sanity_check_Hulb112Disc_fixed_funit32",
"LeakyReLU everywhere-24":"results/train/2019-05-26-D_sanity_check_Hull112Disc_fixed_funit24_LeakyReLU",
"LeakyReLU-24": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit24_LeakyReLU",
"LeakyReLU-final pooling-24": "results/train/2019-05-26-D_sanity_check_Hul112Disc_fixed_funit24_LeakyReLU_finalpool"
}
experiments['Hulb128'] = {"BN, PReLU": "results/train/2019-05-23-Hulb128Net-BN",
"No BN, PReLU": "results/train/2019-05-23-Hulb128Net-NoBN",
"No BN, no final activation": "results/train/2019-05-23-Hulb128Net-NoBN-NoAct"}
if args.list_experiments:
    print(experiments.keys())
    exit(0)

experiment_raw = experiments[args.experiment]

data = dict()
markers = graph_utils.make_markers_dict(components=experiment_raw.keys())
smallest_log = None

for component, path in experiment_raw.items():
    data[component] = graph_utils.parse_log_file(path, smoothing_factor = args.smoothing_factor)
    if smallest_log is None or smallest_log > len(data[component]):
        smallest_log = len(data[component])
for component in data:
    if not args.uneven_graphs:
        data[component] = data[component][0:smallest_log]
    plt.plot(data[component], label=component, marker = markers[component])#, marker=markers[component])
plt.title(args.experiment)
plt.ylabel(args.yaxis)
plt.xlabel(args.xaxis)
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
