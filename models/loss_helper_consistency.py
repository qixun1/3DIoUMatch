import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from utils.nn_distance import nn_distance


def compute_center_consistency_loss(end_points, ema_end_points, student_weights=None, ema_weights=None, student=False):
    center = end_points['center']  # (B, num_proposal, 3)
    ema_center = ema_end_points['center']  # (B, num_proposal, 3)
    flip_x_axis = end_points['flip_x_axis']  # (B,)
    flip_y_axis = end_points['flip_y_axis']  # (B,)
    rot_mat = end_points['rot_mat']  # (B,3,3)
    scale_ratio = end_points['scale']  # (B,1,3)

    # align ema_center with center based on the input augmentation steps
    inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
    ema_center[inds_to_flip_x_axis, :, 0] = -ema_center[inds_to_flip_x_axis, :, 0]

    inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
    ema_center[inds_to_flip_y_axis, :, 1] = -ema_center[inds_to_flip_y_axis, :, 1]

    ema_center = torch.bmm(ema_center, rot_mat.transpose(1, 2))  # (B, num_proposal, 3)

    ema_center = ema_center * scale_ratio

    # ind1 need student score then ind2 need ema score
    dist1, ind1, dist2, ind2 = nn_distance(center,
                                           ema_center)  # ind1 (B, num_proposal): ema_center index closest to center

    # TODO: weigh the unlabelled distances according to values
    if student_weights is not None and ema_weights is not None:
        dist1 = student_weights * dist1
        dist2 = ema_weights * dist2
    # use ema weights for the dist2 and student weights for dist 1 on both fully labelled and unlabelled data

    # TODO: use both dist1 and dist2 or only use dist1
    dist = dist1 + dist2

    if student:
        return torch.mean(dist), ind1
    return torch.mean(dist), ind2



def compute_class_consistency_loss(end_points, ema_end_points, map_ind, weight=None, student=False):
    cls_scores = end_points['sem_cls_scores']  # (B, num_proposal, num_class)
    ema_cls_scores = ema_end_points['sem_cls_scores']  # (B, num_proposal, num_class)

    if student:
        target_cls_prob = F.softmax(cls_scores, dim=2)  # (B, num_proposal, num_class)
        ema_cls_log_prob = F.log_softmax(ema_cls_scores, dim=2)  # (B, num_proposal, num_class)
        target_log_prob_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(ema_cls_log_prob, map_ind)])
    else:
        cls_log_prob = F.log_softmax(cls_scores, dim=2)  # (B, num_proposal, num_class)
        target_cls_prob = F.softmax(ema_cls_scores, dim=2)  # (B, num_proposal, num_class)
        target_log_prob_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(cls_log_prob, map_ind)])

    # weighted kl divergence
    if weight is None:
        class_consistency_loss = F.kl_div(target_log_prob_aligned, target_cls_prob)
    else:
        class_consistency_loss = F.kl_div(target_log_prob_aligned, target_cls_prob, reduction='none')
        class_consistency_loss = torch.mean(class_consistency_loss, dim=-1)
        # align weight if required
        if student:
            weight = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(weight, map_ind)])
        class_consistency_loss = torch.mean(weight * class_consistency_loss)
    # class_consistency_loss = F.mse_loss(cls_log_prob_aligned, ema_cls_prob)

    return class_consistency_loss * 2


def compute_size_consistency_loss(end_points, ema_end_points, map_ind, config, weight=None, student=False):
    mean_size_arr = config.mean_size_arr
    mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda()  # (num_size_cluster,3)
    B, K = map_ind.shape

    scale_ratio = end_points['scale']  # (B,1,3)
    size_class = torch.argmax(end_points['size_scores'], -1)  # B,num_proposal
    size_residual = torch.gather(end_points['size_residuals'], 2,
                                 size_class.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3))  # B,num_proposal,1,3
    size_residual.squeeze_(2)

    ema_size_class = torch.argmax(ema_end_points['size_scores'], -1)  # B,num_proposal
    ema_size_residual = torch.gather(ema_end_points['size_residuals'], 2,
                                     ema_size_class.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1,
                                                                                       3))  # B,num_proposal,1,3
    ema_size_residual.squeeze_(2)

    size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
    size_base = size_base.view(B, K, 3)
    size = size_base + size_residual

    ema_size_base = torch.index_select(mean_size_arr, 0, ema_size_class.view(-1))
    ema_size_base = ema_size_base.view(B, K, 3)
    ema_size = ema_size_base + ema_size_residual
    ema_size = ema_size * scale_ratio

    if student:
        target_size_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(ema_size, map_ind)])
        target_size = size
    else:
        target_size_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(size, map_ind)])
        target_size = ema_size

    if weight is None:
        size_consistency_loss = F.mse_loss(target_size_aligned, target_size)
    else:
        # align weight if required
        if student:
            weight = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(weight, map_ind)])
        weight = weight.unsqueeze(-1)
        weight = weight.repeat(1,1,3)
        size_consistency_loss = weighted_mse_loss(target_size_aligned, target_size, weight)

    return size_consistency_loss


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()


def get_consistency_loss(end_points, ema_end_points, config, student=False):
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
    # get the weights (objectness, class, iou_pred) of the student and teacher
    objectness_scores_student = end_points['objectness_scores']
    pred_objectness_student = nn.Softmax(dim=2)(objectness_scores_student)
    objectness_student = pred_objectness_student[:, :, 1]
    sem_cls_scores_student = end_points['sem_cls_scores']
    pred_sem_cls_student = nn.Softmax(dim=2)(sem_cls_scores_student)
    max_cls_student, argmax_cls_student = torch.max(pred_sem_cls_student, dim=2)
    iou_pred_student = nn.Sigmoid()(end_points['iou_scores'])
    if iou_pred_student.shape[2] > 1:
        iou_pred_student = torch.gather(iou_pred_student, 2, argmax_cls_student.unsqueeze(-1)).squeeze(-1)  # use pred semantic labels
    else:
        iou_pred_student = iou_pred_student.squeeze(-1)

    centre_student_weight = objectness_student * max_cls_student * iou_pred_student
    class_student_weight = objectness_student * max_cls_student * iou_pred_student
    size_student_weight = objectness_student * max_cls_student * iou_pred_student

    objectness_scores_ema = ema_end_points['objectness_scores']
    pred_objectness_ema = nn.Softmax(dim=2)(objectness_scores_ema)
    objectness_ema = pred_objectness_ema[:, :, 1]

    sem_cls_scores_ema = ema_end_points['sem_cls_scores']
    pred_sem_cls_ema = nn.Softmax(dim=2)(sem_cls_scores_ema)
    max_cls_ema, argmax_cls_ema = torch.max(pred_sem_cls_ema, dim=2)

    iou_pred_ema = nn.Sigmoid()(ema_end_points['iou_scores'])
    if iou_pred_ema.shape[2] > 1:
        iou_pred_ema = torch.gather(iou_pred_ema, 2, argmax_cls_ema.unsqueeze(-1)).squeeze(-1)  # use pred semantic labels
    else:
        iou_pred_ema = iou_pred_ema.squeeze(-1)

    centre_ema_weight = objectness_ema * max_cls_ema * iou_pred_ema
    class_ema_weight = objectness_ema * max_cls_ema
    size_ema_weight = objectness_ema * iou_pred_ema

    center_consistency_loss, map_ind = compute_center_consistency_loss(end_points, ema_end_points,
                                                                       centre_student_weight, centre_ema_weight, student)
    if student:
        # still use ema weights
        class_consistency_loss = compute_class_consistency_loss(end_points, ema_end_points, map_ind, class_ema_weight)
        size_consistency_loss = compute_size_consistency_loss(end_points, ema_end_points, map_ind, config, size_ema_weight)
    else:
        class_consistency_loss = compute_class_consistency_loss(end_points, ema_end_points, map_ind, class_ema_weight)
        # class_consistency_loss = compute_class_consistency_loss(end_points, ema_end_points, map_ind)
        size_consistency_loss = compute_size_consistency_loss(end_points, ema_end_points, map_ind, config, size_ema_weight)

    consistency_loss = center_consistency_loss + class_consistency_loss + size_consistency_loss

    end_points['center_consistency_loss'] = center_consistency_loss
    end_points['class_consistency_loss'] = class_consistency_loss
    end_points['size_consistency_loss'] = size_consistency_loss
    end_points['consistency_loss'] = consistency_loss

    return consistency_loss, end_points
