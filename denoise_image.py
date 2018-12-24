import os
import argparse
import torchvision
import torch
from math import ceil, floor
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time


parser = argparse.ArgumentParser(description='Image cropper with overlap (relies on crop_img.sh)')
parser.add_argument('--cs', default=128, type=int, help='Tile size')
parser.add_argument('--ucs', default=96, type=int, help='Useful tile size')
parser.add_argument('-ol', '--overlap', default=8, type=int)
parser.add_argument('-i', '--input', default='datasets/dataset', type=str, help='Input dataset directory. Default is datasets/dataset, for test try datasets/noisyonly')
parser.add_argument('-bs', '--batch_size', type=int, default=1)
# TODO merge these / autodetect
parser.add_argument('--model_dir', type=str, help='directory where .th models are saved (latest .th file is autodetected)')
parser.add_argument('--model_subdir', type=str, help='subdirectory where .th models are saved (latest .th file is autodetected, models dir is assumed)')
parser.add_argument('--model_path', type=str, help='the specific model file path')
args = parser.parse_args()

class OneImageDS(Dataset):
    def __init__(self, inimg, cs, ucs, ol):
        self.inimg = Image.open(inimg)
        self.width, self.height = self.inimg.size
        self.cs, self.ucs, self.ol = cs, ucs, ol    # crop size, useful crop size, overlap
        self.totensor = torchvision.transforms.ToTensor()
        self.iperhl = ceil((self.width - self.ucs) / (self.ucs - self.ol)) # i_per_hline, or crops per line
        self.pad = int((self.cs - self.ucs) / 2)
        ipervl = ceil((self.height - self.ucs) / (self.ucs - self.ol))
        self.size = (self.iperhl+1) * (ipervl+1)
    def __getitem__(self, i):
        # x-y indices (0 to iperhl, 0 to ipervl)
        yi = int(ceil((i+1)/(self.iperhl+1) - 1))   # line number
        xi = i-yi*(self.iperhl+1)
        # x-y start-end position on fs image
        x0 = self.ucs * xi - self.ol * xi - self.pad
        x1 = x0+self.cs
        y0 = self.ucs * yi - self.ol * yi - self.pad
        y1 = y0+self.cs
        ret = Image.new('RGB', (self.cs, self.cs))
        # amount padded to have a cs x cs crop
        x0pad = -min(0, x0)
        x1pad = max(0, x1 - self.width)
        y0pad = -min(0, y0)
        y1pad = max(0, y1 - self.height)
        crop = self.inimg.crop((x0+x0pad, y0+y0pad, x1-x1pad, y1-y1pad))
        ret.paste(crop, (x0pad, y0pad, self.cs-x1pad, self.cs-y1pad))
        usefuldim = (self.pad, self.pad, self.cs-self.pad, self.cs-self.pad)
        usefulstart = (x0+self.pad, y0+self.pad)
        #usefulstart = (x0+self.pad+x0pad, y0+self.pad+y0pad)
        print(str((i, usefuldim, usefulstart)))
        return self.totensor(ret), torch.IntTensor(usefuldim), torch.IntTensor(usefulstart)
    def __len__(self):
        return self.size

if args.model_path:
    model_path = args.model_path
elif args.model_dir or args.model_subdir:
    model_dir = args.model_dir if args.model_dir else os.path.join('models', args.model_subdir)
    #model_path = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
    model_path = os.path.join(model_dir, "latest_model.pth")
else:
    model_path = os.path.join('models', sorted(os.listdir('models'))[-1])
    model_path = os.path.join(model_path, sorted(os.listdir(model_path))[-1])

print('loading '+ model_path)
model = torch.load(model_path, map_location='cuda:'+str(0))
model.eval()  # evaluation mode
if torch.cuda.is_available():
    model = model.cuda()
ds = OneImageDS(args.input, args.cs, args.ucs, args.overlap)
# multiple workers cannot access the same PIL object without crash
DLoader = DataLoader(dataset=ds, num_workers=0, drop_last=False, batch_size=args.batch_size, shuffle=False)
os.makedirs('tmp', exist_ok=True)
topil = torchvision.transforms.ToPILImage()
#newimg = Image.new('RGB', Image.open(args.input).size)
fswidth, fsheight = Image.open(args.input).size
newimg = torch.zeros(3, fsheight, fswidth, dtype=torch.float32)

def make_seamless_edges(tcrop, startcoord):
    y0,x0 = startcoord
    if x0 != 0:#left
        tcrop[:,:,0:args.overlap] = tcrop[:,:,0:args.overlap].div(2)
    if y0 != 0:#top
        tcrop[:,0:args.overlap,:] = tcrop[:,0:args.overlap,:].div(2)
    if x0 + self.ucs < fswidth:#right
        tcrop[:,:,-args.overlap:] = tcrop[:,:,-args.overlap:].div(2)
    if y0 + self.ucs < fswidth:#bottom
        tcrop[:,-args.overlap:,:] = tcrop[:,-args.overlap:,:].div(2)
    return tcrop

start_time = time.time()
for n_count, ydat in enumerate(DLoader):
        print(str(n_count)+'/'+str(int(len(ds)/args.batch_size)))
        ybatch, usefuldims, usefulstarts = ydat
        ybatch = ybatch.cuda()
        xbatch = model(ybatch)
        torch.cuda.synchronize()
        #torchvision.utils.save_image(xbatch, 'tmp/'+str(n_count)+'.jpg')
        for i in range(args.batch_size):
        #print(usefulstarts)
        #for xtens, usefuldim, usefulstart in (xbatch, usefuldims, usefulstarts):
            #print(usefuldims[i])
            #print(usefulstarts[i])
            ud = usefuldims[i]
            # pytorch represents images as [channels, height, width]
            # TODO test leaving on GPU longer
            tensimg = xbatch[i][:,ud[1]:ud[3], ud[0]:ud[2]].cpu().detach()

            absx0, absy0 = tuple(usefulstarts[i].tolist())
            print(tensimg.shape)
            #print(usefulstart)
            #newimg.paste(topil(tensimg), usefulstart)
            newimg[:,absy0:absy0+tensimg.shape[1],absx0:absx0+tensimg.shape[2]] = newimg[:,absy0:absy0+tensimg.shape[1],absx0:absx0+tensimg.shape[2]].add(tensimg)
            # find ucs, starting points
            #tensimg = xbatch[i]

        #for yimg, usefuldim, usefulstart in xbatch, usefuldims, usefulstarts:
        #    pilimg = topil(yimg[1,usefuldim])
        #    newimg.paste(pilimg, usefulstart)
torchvision.utils.save_image(newimg, 'tmp.jpg')
#newimg.save('tmp.jpg')
print('Elapsed time: '+str(time.time()-start_time)+' seconds')
