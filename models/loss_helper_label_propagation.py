import torch
from torch import nn
import torch.nn.functional as F
import faiss

import numpy as np
from utils.box_util import box3d_iou_batch_gpu


def label_propagate(A, Y, alpha=0.99):
    """ Label Propagation, refer to "Learning with Local and Global Consistency" NeurIPs 2003
    Args:
        A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
        Y: initial label matrix, shape: (num_nodes, n_way)
        alpha: a parameter to control the amount of propagated info.
    Return:
        Z: label predictions, shape: (num_nodes, n_way)
    """
    num_nodes = A.shape[0]

    #compute symmetrically normalized matrix S
    eps = np.finfo(float).eps
    D = A.sum(1) #(num_nodes,)
    D_sqrt_inv = torch.sqrt(1.0/(D+eps))
    D_sqrt_inv = torch.diag_embed(D_sqrt_inv).cuda()
    S = D_sqrt_inv @ A @ D_sqrt_inv

    #close form solution
    Z = torch.inverse(torch.eye(num_nodes).cuda() - alpha*S + eps) @ Y
    return Z

def calculateLocalConstrainedAffinity(node_feat, k=200, method='gaussian', sigma=1.0):
    """
    Calculate the Affinity matrix of the nearest neighbor graph constructed by prototypes and query points,
    It is a efficient way when the number of nodes in the graph is too large.
    Args:
        node_feat: input node features
              shape: (num_nodes, feat_dim)
        k: the number of nearest neighbors for each node to compute the similarity
        method: 'cosine' or 'gaussian', different similarity function
    Return:
        A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
    """
    # kNN search for the graph
    num_nodes = node_feat.shape[0]
    feat_dim = node_feat.shape[1]
    X = node_feat.detach().cpu().numpy()
    # build the index with cpu version
    index = faiss.IndexFlatL2(feat_dim)
    index.add(X)
    _, I = index.search(X, k + 1)
    I = torch.from_numpy(I[:, 1:]).cuda() #(num_nodes, k)

    # create the affinity matrix
    knn_idx = I.unsqueeze(2).expand(-1, -1, feat_dim).contiguous().view(-1, feat_dim)
    knn_feat = torch.gather(node_feat, dim=0, index=knn_idx).contiguous().view(num_nodes, k, feat_dim)

    if method == 'cosine':
        knn_similarity = F.cosine_similarity(node_feat[:,None,:], knn_feat, dim=2)
    elif method == 'gaussian':
        dist = F.pairwise_distance(node_feat[:,:,None], knn_feat.transpose(1,2), p=2)
        knn_similarity = torch.exp(-0.5*(dist/sigma)**2)
    else:
        raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)

    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float).cuda()
    A = A.scatter_(1, I, knn_similarity)
    A = A + A.transpose(0,1)

    identity_matrix = torch.eye(num_nodes, requires_grad=False).cuda()
    A = A * (1 - identity_matrix)
    return A

def get_label_propagation_loss(end_points, prototypes, proto_labels, config_dict, sigma=1.0,
                               obj_threshold=0.8, iou_threshold=0.1):
    """
    Args:
        end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
                flip_x_axis, flip_y_axis, rot_mat
            }
        prototypes: list (num_prototypes*n_classes, feat_dim)
            []
    Returns:
        label_propagation_loss: pytorch scalar tensor
    """

    supervised_mask = end_points['supervised_mask']
    unsupervised_inds = torch.nonzero(1 - supervised_mask).squeeze(1).long()
    num_prototypes = prototypes.shape[0]
    query_feat = end_points['features'][unsupervised_inds, ...]
    batch_size = query_feat.shape[0]
    feat_dim = query_feat.shape[1]
    query_feat = query_feat.view(-1, feat_dim)
    # filter out proposals that do not meet the threshold for the objectness score
    pred_objectness = end_points['objectness_scores'][unsupervised_inds, ...]
    pred_objectness = nn.Softmax(dim=2)(pred_objectness)
    objectness_mask = pred_objectness[:, :, 1].reshape(-1) > obj_threshold


    objectness_mask = objectness_mask.view(-1)
    query_feat = query_feat[objectness_mask]

    num_nodes = num_prototypes + query_feat.shape[0]
    n_classes = proto_labels.shape[-1]
    Y = torch.zeros(num_nodes, n_classes).cuda()
    Y[:num_prototypes] = proto_labels
    # do label propagation
    node_feat = torch.cat((prototypes, query_feat), dim=0) #(num_nodes, feat_dim)
    A = calculateLocalConstrainedAffinity(node_feat, k=100, sigma=sigma)
    propagated_labels = label_propagate(A, Y)

    pseudo_labels = propagated_labels[num_prototypes:, :]
    pseudo_labels = torch.argmax(pseudo_labels, dim=-1)

    # cross entropy of propagated feature vs predicted feature?

    pred_labels = end_points['sem_cls_scores'][unsupervised_inds, ...]

    pred_labels = pred_labels.reshape(-1, pred_labels.shape[-1])[objectness_mask]

    if config_dict['view_stats']:
        # compute gt bboxes
        # box_mask = end_points['box_label_mask'][unsupervised_inds, ...]
        # box_indices = torch.nonzero(box_mask).long()
        center_label = end_points['center_label'][unsupervised_inds, ...]
        heading_class_label = end_points['heading_class_label'][unsupervised_inds, ...]
        heading_residual_label = end_points['heading_residual_label'][unsupervised_inds, ...]
        size_class_label = end_points['size_class_label'][unsupervised_inds, ...]
        size_residual_label = end_points['size_residual_label'][unsupervised_inds, ...]

        gt_size = config_dict['dataset_config'].class2size_gpu(size_class_label, size_residual_label)
        gt_angle = config_dict['dataset_config'].class2angle_gpu(heading_class_label, heading_residual_label)
        gt_bbox = torch.cat([center_label, gt_size, -gt_angle[:, :, None]], dim=2)

        # compute pred bboxes
        pred_center = end_points['center'][unsupervised_inds, ...] #(1, num_proposals, 3)
        pred_size = end_points['size'][unsupervised_inds, ...]
        pred_size[pred_size < 0] = 1e-6
        pred_heading = end_points['heading'][unsupervised_inds, ...]

        pred_bbox = torch.cat([pred_center, pred_size, -pred_heading[:, :, None]], axis=2)

        pred_num = pred_bbox.shape[1]
        gt_num = gt_bbox.shape[1]

        # start = time.time()
        gt_bbox_ = gt_bbox.view(-1, 7)
        pred_bbox_ = pred_bbox.view(-1, 7)

        # compute iou overlap for bboxes and assign each proposal to gt bbox
        iou_labels = box3d_iou_batch_gpu(pred_bbox_, gt_bbox_)
        iou_labels, object_assignment = iou_labels.view(batch_size * pred_num, batch_size, -1).max(dim=2)
        inds = torch.arange(batch_size).cuda().unsqueeze(1).expand(-1, pred_num).contiguous().view(-1, 1)
        iou_labels = iou_labels.gather(dim=1, index=inds).view(batch_size, -1)
        iou_indices = iou_labels.view(-1) > iou_threshold
        iou_labels = iou_labels.detach()
        object_assignment = object_assignment.gather(dim=1, index=inds).view(batch_size, -1)

        # combine indices with previous objectness mask
        combined_indices = iou_indices & objectness_mask
        combined_indices = combined_indices == True

        iou_indices = iou_indices[objectness_mask]

        # combined_indices = iou_indices

        # batch set groundtruth indices to sem indices
        sem_labels = end_points['sem_cls_label'][unsupervised_inds, ...]
        gt_labels = sem_labels.gather(dim=1, index=object_assignment).view(-1)

        # check correctness of pseudo labels
        gt = gt_labels[combined_indices]
        pseudo = pseudo_labels[iou_indices]
        pseudo_correct = pseudo == gt
        # pseudo_labels = pseudo_labels[iou_indices]
        end_points['pseudo_correctness'] = torch.sum(pseudo_correct).detach().cpu().numpy() / pseudo.shape[0]

        # check correctness of predicted sem labels
        preds = torch.argmax(pred_labels, dim=-1)
        pred = preds[iou_indices]
        pred_correct = pred == gt
        # preds = pred_labels[iou_indices]
        end_points['pred_correctness'] = torch.sum(pred_correct).detach().cpu().numpy() / pred.shape[0]



    label_propagation_loss = F.cross_entropy(pred_labels, pseudo_labels)
    end_points['label_propagation_loss'] = label_propagation_loss

    return label_propagation_loss, end_points
