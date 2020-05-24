# mthesis-denoise

Master thesis on natural image noise removal using Convolutional Neural Networks. Works with the Natural Image Noise Dataset to apply to real photographs, using a UNet network architecture by default.

This repository contains many files from my master thesis. It is unnecessarily large for any productive work. https://github.com/trougnouf/nind-denoise is a lighter repository which should be cloned instead going forward.

## test (denoise an image)

Requirements: pytorch [, exiftool]

```
python3 denoise_image.py python3 denoise_image.py --model_path models/[model.pth] -i <input_image_path> [-o output_image_path]
```

## train

Requirements: pytorch, bash, imagemagick, libjpeg[-turbo] [, wget]

Note that cropping png files is currently extremely slow, therefore you may want to comment out "download(nindlist16bit, 'NIND', args.target_dir, ext='png')" from the dl_dataset_1.py script. This is due to the cropping process opening the whole file every time it makes a new crop, this process is in place to losslessly crop jpeg files but an optimized script should be written for other image types.

```bash
python3 dl_ds_1.py --use_wget   # --use_wget is much less likely to result in half-downloaded files
python3 crop_ds.py              # this will take a long time. Do python3 crop_ds.py --cs 128 --ucs 96 with U-Net model to use all data
# batch_size 94 is for a 11GB NVidia 1080, use a lower batch_size if less memory is available
# train a single U-Net generator:
python3 nn_train.py --g_network UNet --weight_SSIM 1 --batch_size 60 --train_data datasets/train/NIND_128_96
# train a HulbNet generator and HulfDisc discriminator
python3 nn_train.py --d_network Hulf112Disc --batch_size 10
# list options
python3 nn_train.py --help
```

note that run\_nn.py contains slightly more options (such as compression and artificial noise) but it only trains one network at a time. nn\_train.py can currently train one generator and two discriminators.
