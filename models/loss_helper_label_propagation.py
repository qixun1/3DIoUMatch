import torch
import torch.nn.functional as F
import faiss

import numpy as np
from torch import nn

from utils.vis_utils import visualise_features


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

def get_label_propagation_loss(end_points, prototypes, proto_labels, sigma=1.0, weighing='soft'):
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
    query_feat = query_feat.view(-1, query_feat.shape[1])
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
    proto_labels = torch.argmax(proto_labels, dim=-1)

    # visualise_features(prototypes.detach().cpu().numpy(), proto_labels.detach().cpu().numpy(),
    #                    "assets/labelled_feats")
    # visualise_features(query_feat.detach().cpu().numpy(), pseudo_labels.detach().cpu().numpy(),
    #                    "assets/unlabelled_feats")

    # cross entropy of propagated feature vs predicted feature?

    pred_labels = end_points['sem_cls_scores'][unsupervised_inds, ...]

    pred_labels = pred_labels.reshape(-1, pred_labels.shape[-1])
    pred_objectness = end_points['objectness_scores'][unsupervised_inds, ...]
    pred_objectness = nn.Softmax(dim=2)(pred_objectness)

    if weighing == 'soft':
        weight = pred_objectness[:, :, 1].reshape(-1)
    elif weighing == 'hard':
        objectness_mask = pred_objectness[:, :, 1].reshape(-1) > 0.9
        pred_labels = torch.gather(pred_labels, index=objectness_mask)
        pseudo_labels = torch.gather(pseudo_labels, index=objectness_mask)
        weight = torch.ones(objectness_mask.shape)
    else:
        pass

    if weight is not None:
        label_propagation_loss = F.cross_entropy(pred_labels, pseudo_labels, reduction='none')
        label_propagation_loss = torch.mean(weight * label_propagation_loss)
    else:
        label_propagation_loss = F.cross_entropy(pred_labels, pseudo_labels)
    end_points['label_propagation_loss'] = label_propagation_loss

    return label_propagation_loss, end_points
