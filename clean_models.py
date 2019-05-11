import os
os.chdir('models')

def find_latest_models(model_dir):
    last = 0
    last_models = []
    for model_file in os.listdir(model_dir):
        if model_file == 'latest_model.pth' or not model_file.endswith('.pth'):
            continue
        current = int(model_file.split('.')[0].split('_')[-1])
        if current > last:
            last = current
            last_models = [model_file]
        elif current == last:
        	last_models.append(model_file)
    return last_models
for model_dir in os.listdir('.'):
    if os.path.isfile(model_dir) or os.path.islink(model_dir):
        continue
    latest_models = find_latest_models(model_dir)
    for model_file in os.listdir(model_dir):
        if model_file not in latest_models and model_file != 'latest_model.pth' and model_file.endswith('.pth'):
            print('rm %s'%(os.path.join(model_dir, model_file)))
            os.remove(os.path.join(model_dir, model_file))
        else:
            print('keep %s'%(os.path.join(model_dir, model_file)))
