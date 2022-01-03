""" Util for model checkpoint loading
Author: Zhao Na
Date: 2021
"""
import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def check_state_dict_consistency(loaded_state_dict, model_state_dict):
    """check consistency between loaded parameters and created model parameters
    """
    valid_state_dict = {}
    for k in loaded_state_dict:
        if k in model_state_dict:
            if loaded_state_dict[k].shape != model_state_dict[k].shape:
                print('\tSkip loading parameter {}, required shape{}, loaded shape{}'.format(
                          k, model_state_dict[k].shape, loaded_state_dict[k].shape))
                valid_state_dict[k] = model_state_dict[k]
            else:
                valid_state_dict[k] = loaded_state_dict[k]
        else:
            print('\tDrop parameter {}.'.format(k))

    for k in model_state_dict:
        if not (k in loaded_state_dict):
            print('\tNo param {}.'.format(k))
            valid_state_dict[k] = model_state_dict[k]

    return valid_state_dict


def load_model_checkpoint(model, model_checkpoint_path, resume=False, optimizer=None):
    start_epoch = 0
    checkpoint = torch.load(model_checkpoint_path)
    pretrained_dict = checkpoint['model_state_dict']
    epoch_ckpt = checkpoint['epoch']

    if resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = epoch_ckpt
        print('-> Resume training from checkpoint %s (epoch %d)' % (model_checkpoint_path, start_epoch))
    else:
        print('-> Load pretrained checkpoint %s (epoch: %d)' % (model_checkpoint_path, epoch_ckpt))

    valid_state_dict = check_state_dict_consistency(pretrained_dict, model.state_dict())
    model.load_state_dict(valid_state_dict)

    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model