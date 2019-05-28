import torch

def get_crop_boundaries(cs, ucs, network=None, discriminator=None):
    if '112' in discriminator:
        loss_crop_lb = int((cs-112)/2)
        loss_crop_up = cs - loss_crop_lb
        assert (loss_crop_up - loss_crop_lb) == 112
    elif network == 'UNet':    # UNet requires huge borders
        loss_crop_lb = int(cs/8)
        loss_crop_up = cs-loss_crop_lb
    else:
        loss_crop_lb = int(cs/16)
        loss_crop_up = cs-loss_crop_lb
    assert (loss_crop_up - loss_crop_lb) >= ucs
    print('Using %s as bounds'%(str((loss_crop_lb, loss_crop_up))))
    return loss_crop_lb, loss_crop_up

def gen_target_probabilities(target_real, target_probabilities_shape, device=None, invert_probabilities=False, noisy = True):
    if (target_real and not invert_probabilities) or (not target_real and invert_probabilities):
        if noisy:
            res = 19/20+torch.rand(target_probabilities_shape)/20
        else:
            res = torch.ones(target_probabilities_shape)
    else:
        if noisy:
            res = torch.rand(target_probabilities_shape)/20
        else:
            res = torch.zeros(target_probabilities_shape)
    if device is None:
        return res
    else:
        return res.to(device)
