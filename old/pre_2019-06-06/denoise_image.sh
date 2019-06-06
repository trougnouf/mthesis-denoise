# Obsolete, use denoise_image.py instead
IMGIN=$1
IMGOUT=$2
# TODO add modular model argument
TMPDIR=tmp${IMGIN}${IMGOUT}
mkdir -p ${TMPDIR}/crops_128_80
mkdir -p ${TMPDIR}/denoised
bash crop_img.sh 128 80 ${IMGIN} ${TMPDIR}/crops_128_80
python denoise_dir.py --noisy_dir ${TMPDIR}/crops_128_80 --model_subdir 2018-12-18T08:20_run_nn.py_--test_reserve_ursulines-red_stefantiek_ursulines-building_MuseeL-Bobo_CourtineDeVillersDebris_--batch_size_94_--lr_3e-4 --result_dir ${TMPDIR}/denoised --no_scoring
exiftool -TagsFromFile ${IMGIN} tmp${IMGIN}/${IMGIN} -overwrite_original