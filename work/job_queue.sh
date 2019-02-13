<<<<<<< HEAD
python3 crop_ds.py
python3 run_nn.py --lr 3e-4 --cuda_device 0 --time_limit 259200 --batch_size 94 --test_reserve ursulines-red stefantiek ursulines-building MuseeL-Bobo CourtineDeVillersDebris --skip_sizecheck # NIND for report
python3 run_nn.py --train_data datasets/train/SIDDmed_128_96 datasets/train/NIND_128_96 --lr 3e-4 --cuda_device 0 --time_limit 259200 --batch_size 94 --test_reserve ursulines-red stefantiek ursulines-building MuseeL-Bobo CourtineDeVillersDebris --skip_sizecheck# does SIDD reduce perf
python3 run_nn.py --train_data datasets/train/SIDDmed_128_96 datasets/train/NIND_128_96 --lossf MSE --lr 3e-4 --cuda_device 0 --time_limit 259200 --batch_size 94 --skip_sizecheck # NIND+SIDDmed w/MSE for comp
python3 run_nn.py --lr 3e-4 --cuda_device 0 --time_limit 259200 --batch_size 94 --test_reserve ursulines-red stefantiek ursulines-building MuseeL-Bobo CourtineDeVillersDebris --skip_sizecheck --yval ISO6400# NIND for report
python3 run_nn.py --lr 3e-4 --cuda_device 0 --time_limit 259200 --batch_size 94 # NIND only no reserve
=======
python3 run_nn.py --train_data datasets/train/SIDDmed_128_96 datasets/train/NIND_128_96 --lossf MSE --lr 3e-4 --cuda_device 0 --time_limit 259200 --batch_size 94 # NIND+SIDDmed w/MSE for comp
>>>>>>> aaa6abc0d54e371255cfb98d913720deb3d92180
