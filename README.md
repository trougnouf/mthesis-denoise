# mthesis-denoise

Master thesis on natural image noise removal using Convolutional Neural Networks. Works with the Natural Image Noise Dataset to apply to real photographs, using a UNet network architecture by default.

## test (denoise an image)

Requirements: pytorch [, exiftool]

`python3 denoise_image.py -i <input_image_path> [-o output_image_path] --model_dir models/2018-12-25T20:03_run_nn.py_--batch_size_94_--lr_3e-4`

## train

Requirements: pytorch, bash, imagemagick, libjpeg[-turbo] [, wget]

```bash
python3 dl_ds_1.py --use_wget					# --use_wget is much less likely to result in half-downloaded files
python3 crop_ds.py								# this will take a long time
python3 run_nn.py --lr 3e-4 --batch_size 94		# batch_size 94 is for a 11GB NVidia 1080, use a lower batch_size if less memory is available
```