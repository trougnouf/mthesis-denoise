import os
import argparse
import subprocess
from multiprocessing import cpu_count
parser = argparse.ArgumentParser(description='Image cropper with overlap (relies on crop_img.sh)')
parser.add_argument('--cs', default=128, type=int, help='Crop size')
parser.add_argument('--ucs', default=96, type=int, help='Useful crop size')
parser.add_argument('--dsdir', default='datasets/dataset', type=str, help='Input dataset directory')
parser.add_argument('--resdir', default='datasets/train', type=str, help='Output cropped dataset directory ([resdir]/ds_[cs]_[ucs])')
parser.add_argument('--max_threads', type=int, help='Maximum number of active threads, default=#threads')
args = parser.parse_args()

resdir = os.path.join(args.resdir, 'ds_'+str(args.cs)+'_'+str(args.ucs))
sets = os.listdir(args.dsdir)
todolist = []

for aset in sets:
    for image in os.listdir(os.path.join(args.dsdir, aset)):
        inpath=os.path.join(args.dsdir, aset, image)
        isoval=image.split('_')[-1][:-4]
        outdir=os.path.join(resdir, aset, isoval)
        todolist.append(['bash', 'crop_img.sh', str(args.cs), str(args.ucs), inpath, outdir])
processes = set()
max_threads =args.max_threads if args.max_threads else cpu_count()
for task in todolist:
    processes.add(subprocess.Popen(task))
    if len(processes) >= max_threads:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
