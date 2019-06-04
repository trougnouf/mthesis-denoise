# Obsolete, use denoise_image.py instead
IMGIN=$1
IMGOUT=$2
# TODO add modular model argument
TMPDIR=tmp${IMGIN}${IMGOUT}
mkdir -p ${TMPDIR}/crops_128_88
mkdir -p ${TMPDIR}/denoised
bash crop_img.sh 128 88 ${IMGIN} ${TMPDIR}/crops_128_88
python denoise_dir.py --noisy_dir ${TMPDIR}/crops_128_88 --model_subdir old/2018-12-08T12:58_run_nn.py_--model_RedCNN_--epoch_16_--cuda_device_0_--batch_size_11_--lr_1e-4 --result_dir ${TMPDIR}/denoised --no_scoring