import os
import argparse
import subprocess
from multiprocessing import cpu_count
parser = argparse.ArgumentParser(description='Image cropper with overlap (relies on crop_img.sh)')
parser.add_argument('--cs', default=128, type=int, help='Crop size')
parser.add_argument('--ucs', default=96, type=int, help='Useful crop size')
parser.add_argument('--dsdir', default='datasets/NIND', type=str, help='Input dataset directory. Default is datasets/dataset, for test try datasets/noisyonly')
parser.add_argument('--resdir', default='datasets/train', type=str, help='Output cropped dataset directory ([resdir]/dsdir_[cs]_[ucs]). Default is datasets/train, for test try datasets/test')
parser.add_argument('--max_threads', type=int, help='Maximum number of active threads, default=#threads')
args = parser.parse_args()

dsdir = args.dsdir.split('/')[-1]
resdir = os.path.join(args.resdir, dsdir+'_'+str(args.cs)+'_'+str(args.ucs))
todolist = []

def findisoval(fn):
    for split in fn.split('_'):
        if 'ISO' in split or 'GT' in split or 'NOISY' in split:
            return split.split('.')[0]

sets = os.listdir(args.dsdir)
# structured dataset
if os.path.isdir(os.path.join(args.dsdir, sets[0])):
    for aset in sets:
        isovals=[]
        for image in os.listdir(os.path.join(args.dsdir, aset)):
            inpath=os.path.join(args.dsdir, aset, image)
            isoval=findisoval(image)
            # rename duplicate isoval if any is encountered for easier processing (eg SIDD)
            if isoval in isovals:
                oldval=isoval
                while isoval in isovals:
                    isoval=isoval+'-2'
                newpath = inpath.replace(oldval, isoval)
                os.rename(inpath, inpath.replace(oldval, isoval))
                inpath = newpath
            isovals.append(isoval)
            outdir=os.path.join(resdir, aset, isoval)
            todolist.append(['bash', 'crop_img.sh', str(args.cs), str(args.ucs), inpath, outdir])
# or simple image directory
else:
    for image in os.listdir(args.dsdir):
        inpath = os.path.join(args.dsdir, image)
        outdir = os.path.join(resdir, image[:-4])
        todolist.append(['bash', 'crop_img.sh', str(args.cs), str(args.ucs), inpath, outdir])
# TODO or recursively search for all images
        
processes = set()
max_threads =args.max_threads if args.max_threads else cpu_count()
for task in todolist:
    #print(task)#dbg
    processes.add(subprocess.Popen(task))
    if len(processes) >= max_threads:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
