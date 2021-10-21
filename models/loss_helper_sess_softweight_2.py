""" Loss Function for Self-Ensembling Semi-Supervised 3D Object Detection

Author: Zhao Na, 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss


def compute_center_consistency_loss(end_points, ema_end_points):
    center = end_points['center'] #(B, num_proposal, 3)
    ema_center = ema_end_points['center'] #(B, num_proposal, 3)
    flip_x_axis = end_points['flip_x_axis'] #(B,)
    flip_y_axis = end_points['flip_y_axis'] #(B,)
    rot_mat = end_points['rot_mat'] #(B,3,3)
    scale_ratio = end_points['scale'] #(B,1,3)

    # align ema_center with center based on the input augmentation steps
    inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
    ema_center[inds_to_flip_x_axis, :, 0] = -ema_center[inds_to_flip_x_axis, :, 0]

    inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
    ema_center[inds_to_flip_y_axis, :, 1] = -ema_center[inds_to_flip_y_axis, :, 1]

    ema_center = torch.bmm(ema_center, rot_mat.transpose(1,2)) #(B, num_proposal, 3)

    ema_center = ema_center * scale_ratio

    dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)  #ind1 (B, num_proposal): ema_center index closest to center

    pred_weights = get_pred_weights(end_points, use_cls=True, use_iou=True)
    ema_pred_weights = get_pred_weights(ema_end_points, use_cls=True, use_iou=True)

    #TODO: use both dist1 and dist2 or only use dist1
    dist = dist1*pred_weights + dist2*ema_pred_weights
    return torch.mean(dist), ind1


def compute_class_consistency_loss(end_points, ema_end_points, map_ind):
    cls_scores = end_points['sem_cls_scores'] #(B, num_proposal, num_class)
    ema_cls_scores = ema_end_points['sem_cls_scores'] #(B, num_proposal, num_class)

    ema_cls_log_prob = F.log_softmax(ema_cls_scores, dim=2) #(B, num_proposal, num_class)
    cls_prob = F.softmax(cls_scores, dim=2) #(B, num_proposal, num_class)

    ema_cls_log_prob_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(ema_cls_log_prob, map_ind)])

    class_consistency_loss = F.kl_div(ema_cls_log_prob_aligned, cls_prob, reduction='none') #(B, num_proposal, num_class)
    class_consistency_loss = torch.mean(class_consistency_loss, dim=2)

    # pred_weights = get_pred_weights(end_points, use_cls=True, use_iou=True)
    ema_pred_weights = get_pred_weights(ema_end_points, use_cls=True)
    pred_weights = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(ema_pred_weights, map_ind)])
    class_consistency_loss = class_consistency_loss * pred_weights
    class_consistency_loss = torch.mean(class_consistency_loss)

    return class_consistency_loss


def compute_size_consistency_loss(end_points, ema_end_points, map_ind, config):
    mean_size_arr = config.mean_size_arr
    mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda() #(num_size_cluster,3)
    B, K = map_ind.shape

    scale_ratio = end_points['scale'] #(B,1,3)
    size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
    size_residual = torch.gather(end_points['size_residuals'], 2, size_class.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,3)) # B,num_proposal,1,3
    size_residual.squeeze_(2)

    ema_size_class = torch.argmax(ema_end_points['size_scores'], -1) # B,num_proposal
    ema_size_residual = torch.gather(ema_end_points['size_residuals'], 2, ema_size_class.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,3)) # B,num_proposal,1,3
    ema_size_residual.squeeze_(2)

    size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
    size_base = size_base.view(B,K,3)
    size = size_base + size_residual

    ema_size_base = torch.index_select(mean_size_arr, 0, ema_size_class.view(-1))
    ema_size_base = ema_size_base.view(B,K,3)
    ema_size = ema_size_base + ema_size_residual
    ema_size = ema_size * scale_ratio

    ema_size_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(ema_size, map_ind)])

    size_consistency_loss = F.mse_loss(size, ema_size_aligned, reduction='none') #(B, num_proposal, 3)
    size_consistency_loss = torch.mean(size_consistency_loss, dim=2)

    # pred_weights = get_pred_weights(end_points, use_iou=True, use_cls=True)
    ema_pred_weights = get_pred_weights(ema_end_points, use_iou=True)
    pred_weights = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(ema_pred_weights, map_ind)])
    size_consistency_loss = size_consistency_loss * pred_weights
    size_consistency_loss = torch.mean(size_consistency_loss)

    return size_consistency_loss


def get_pred_weights(end_points, use_cls=False, use_iou=False):
    # obj score threshold
    pred_objectness = end_points['objectness_scores'].detach()
    pred_objectness = nn.Softmax(dim=2)(pred_objectness)
    # the second element is positive score
    pred_weights = pred_objectness[:, :, 1]

    # cls score threshold
    pred_sem_cls = end_points['sem_cls_scores'].detach()
    pred_sem_cls = nn.Softmax(dim=2)(pred_sem_cls)
    pred_cls_prob, argmax_cls = torch.max(pred_sem_cls, dim=2)

    if use_cls:
        pred_weights *= pred_cls_prob

    if use_iou:
        pred_iou = end_points['iou_scores'].detach()
        pred_iou = nn.Sigmoid()(pred_iou)
        if pred_iou.shape[2] > 1:
            pred_iou_prob = torch.gather(pred_iou, 2, argmax_cls.unsqueeze(-1)).squeeze(-1)  # use pred semantic labels
        else:
            pred_iou_prob = pred_iou.squeeze(-1)
        pred_weights *= pred_iou_prob

    return pred_weights


def get_consistency_loss(end_points, ema_end_points, config):
    """
    Args:
        end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
                flip_x_axis, flip_y_axis, rot_mat
            }
        ema_end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
            }
    Returns:
        consistency_loss: pytorch scalar tensor
        end_points: dict
    """
    center_consistency_loss, map_ind = compute_center_consistency_loss(end_points, ema_end_points)
    class_consistency_loss = compute_class_consistency_loss(end_points, ema_end_points, map_ind)
    size_consistency_loss = compute_size_consistency_loss(end_points, ema_end_points, map_ind, config)

    consistency_loss =  center_consistency_loss +class_consistency_loss + size_consistency_loss

    end_points['center_consistency_loss'] = center_consistency_loss
    end_points['class_consistency_loss'] = class_consistency_loss
    end_points['size_consistency_loss'] = size_consistency_loss
    end_points['consistency_loss'] = consistency_loss

    return consistency_loss, end_points