python3 run_nn.py --train_data datasets/train/SIDDmed_128_96 datasets/train/NIND_128_96 --lr 3e-4 --cuda_device 2 --time_limit 259200 --batch_size 94 --test_reserve ursulines-red stefantiek ursulines-building MuseeL-Bobo CourtineDeVillersDebris # does SIDD reduce perf
python3 run_nn.py --lr 3e-4 --cuda_device 2 --time_limit 259200 --batch_size 94 --test_reserve ursulines-red stefantiek ursulines-building MuseeL-Bobo CourtineDeVillersDebris # NIND for report
python3 run_nn.py --train_data datasets/train/SIDDmed_128_96 datasets/train/NIND_128_96 --lossf MSE --lr 3e-4 --cuda_device 2 --time_limit 259200 --batch_size 94 # NIND+SIDDmed w/MSE for comp
python3 run_nn.py --lr 3e-4 --cuda_device 2 --time_limit 259200 --batch_size 94 # NIND only no reserve
