import os
import subprocess

for aroot, _, files in os.walk('datasets/NIND'):
    for afile in files:
        if not('ISO100.' in afile or ('ISO200.' in afile and 'C500D' not in aroot)):
            continue
        isoval = afile.split('_')[-1].split('.')[0]
        newisoval = isoval+'-d'
        oldpath = os.path.join(aroot, afile)
        newpath = os.path.join(aroot, afile.replace(isoval, newisoval))
        if 'C500D' in aroot:
            model_subdir = "2019-02-18T20:10_run_nn.py_--time_limit_259200_--batch_size_94_--test_reserve_ursulines-red_stefantiek_ursulines-building_MuseeL-Bobo_CourtineDeVillersDebris_MuseeL-Bobo-C500D_--skip_sizecheck_--lr_3e-4"
        else:
            model_subdir = "2019-02-13T23:30_run_nn.py_--lr_3e-4_--cuda_device_0_--time_limit_259200_--batch_size_94_--test_reserve_ursulines-red_stefantiek_ursulines-building_MuseeL-Bobo_CourtineDeVillersDebris_--skip_sizecheck"
        cmd = ['python3', 'denoise_image.py', '-i', oldpath, '-o', newpath, '--model_subdir', model_subdir]
        subprocess.run(cmd)
