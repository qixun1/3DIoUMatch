import os
import sys
import copy
import pickle
import numpy as np
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_cluster import fps

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from vis_utils import visualise_features
from box_util import box3d_iou_batch_gpu
from votenet_iou_branch import VoteNet
from nms import nms_3d_faster_samecls
from ap_helper import predictions2corners3d
from model_util import load_model_checkpoint
from scannet.scannet_prototype_dataset import ScannetPrototypeDataset


def init_detection_model(detector_args, dataset_config):
    model = VoteNet(num_class=dataset_config.num_class,
                    num_heading_bin=dataset_config.num_heading_bin,
                    mean_size_arr=dataset_config.mean_size_arr,
                    num_size_cluster=dataset_config.num_size_cluster,
                    dataset_config=dataset_config,
                    input_feature_dim=int(detector_args.use_color) * 3 + int(not detector_args.no_height) * 1,
                    num_proposal=detector_args.num_target,
                    vote_factor=detector_args.vote_factor)

    model_checkpoint_path = os.path.join(ROOT_DIR, detector_args.detector_checkpoint_path)
    if model_checkpoint_path is not None and os.path.isfile(model_checkpoint_path):
        model = load_model_checkpoint(model, model_checkpoint_path)
    else:
        raise ValueError('The path to the file of detection model checkpoint must be given!')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device


def custom_collate(batch):
    out = {}
    out['point_cloud'] = torch.FloatTensor([item['point_clouds'] for item in batch])
    out['bboxes_param'] = torch.FloatTensor([item['bboxes_param'] for item in batch])
    out['bboxes_cls_label'] = torch.LongTensor([item['bboxes_cls_label'] for item in batch])
    out['bboxes_cls_name'] = [item['bboxes_cls_name'] for item in batch]
    out['scan_name'] = [item['scan_name'] for item in batch]
    return out


def generate_prototypes(model, proto_dataloader, config_dict, fg_iou_thres=0.8, fg_obj_thres=0.9, use_nms=False):
    model.eval()
    num_original_proposals = 0
    all_proposals_feat = []
    all_proposals_cls_label = []
    all_proposal_conf_scores = []
    print('\n------ Begin Generating Proposals ------')
    for batch_idx, input_dict in enumerate(proto_dataloader):
        input_dict['bboxes_param'] = input_dict['bboxes_param'][0].numpy()
        input_dict['bboxes_cls_label'] = input_dict['bboxes_cls_label'][0].numpy()
        input_dict['bboxes_cls_name'] = input_dict['bboxes_cls_name'][0]
        gt_bbox_labels = torch.from_numpy(input_dict['bboxes_cls_label']).cuda()
        num_gt_bboxes = input_dict['bboxes_cls_label'].shape[0]
        num_original_proposals += num_gt_bboxes
        inputs = {'point_clouds': input_dict['point_cloud'].to(device)}

        with torch.no_grad():
            end_points = model(inputs)

        pred_center = end_points['center']  # (1, num_proposals, 3)
        pred_size = end_points['size']
        pred_size[pred_size < 0] = 1e-6
        pred_heading = end_points['heading']
        pred_bbox = torch.cat([pred_center, pred_size, -pred_heading[:, :, None]], dim=2).squeeze(
            0)  # (num_proposals, 7)
        gt_bbox = torch.from_numpy(input_dict['bboxes_param'].astype(np.float32)).cuda()  # (num_gt_bboxes, 7)
        gt_bbox[:, 3:6] = gt_bbox[:, 3:6] / 2
        iou_scores = box3d_iou_batch_gpu(pred_bbox, gt_bbox)  # (num_proposals, num_gt_bboxes)
        iou_score, gt_bbox_ind = iou_scores.max(1)

        pred_sem_logits = F.softmax(end_points['sem_cls_scores'], dim=2).squeeze(0)  # (num_proposals, num_seen_class)
        pred_sem_prob, pred_sem_cls = torch.max(pred_sem_logits, dim=1)

        pred_obj_scores = F.softmax(end_points['objectness_scores'], dim=2).squeeze(0)  # (num_proposals, 2)
        pred_obj_prob = pred_obj_scores[:, 1]

        valid_proposals_mask = iou_score > fg_iou_thres
        valid_proposals_mask = torch.logical_and(valid_proposals_mask, pred_obj_prob > fg_obj_thres)
        valid_proposals_mask = torch.logical_and(valid_proposals_mask, pred_sem_cls == gt_bbox_labels[gt_bbox_ind])

        valid_proposals_inds = torch.nonzero(valid_proposals_mask).squeeze(1)

        if len(valid_proposals_inds) > 0:
            proposal_features = end_points['features'].squeeze(0).transpose(0, 1)  # (num_proposals, 128)
            # add nms:
            if use_nms:
                pred_corners_3d_upright_camera = predictions2corners3d(end_points, config_dict)[0]
                pred_corners_3d_upright_camera = pred_corners_3d_upright_camera[:,
                                                 valid_proposals_inds.detach().cpu().numpy()]
                scores = pred_obj_prob

                scores = scores * iou_score
                K = valid_proposals_inds.shape[0]
                pred_mask = np.zeros(iou_score.shape[0])
                boxes_3d_with_prob = np.zeros((K, 8))
                for j, index in enumerate(valid_proposals_inds):
                    boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[0, j, :, 0])
                    boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[0, j, :, 1])
                    boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[0, j, :, 2])
                    boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[0, j, :, 0])
                    boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[0, j, :, 1])
                    boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[0, j, :, 2])
                    boxes_3d_with_prob[j, 6] = scores[index]
                    boxes_3d_with_prob[j, 7] = pred_sem_cls[
                        index]  # only suppress if the two boxes are of the same class!!
                pick = nms_3d_faster_samecls(boxes_3d_with_prob, 0.25, 64)
                assert (len(pick) > 0)
                pred_mask[valid_proposals_inds.detach().cpu().numpy()[pick]] = 1
                pred_mask = torch.from_numpy(pred_mask).cuda()
                valid_proposals_mask = torch.logical_and(valid_proposals_mask, pred_mask)
                valid_proposals_inds = torch.nonzero(valid_proposals_mask).squeeze(1)

            for valid_proposals_ind in valid_proposals_inds:
                all_proposals_feat.append(proposal_features[valid_proposals_ind])
                all_proposals_cls_label.append(pred_sem_cls[valid_proposals_ind].item())
                all_proposal_conf_scores.append(pred_obj_prob[valid_proposals_ind].item())

        num_fg_proposals = len(valid_proposals_inds)

        print('Scan: %s | %3d GT bboxes | %3d FG proposals' % (batch_idx, num_gt_bboxes, num_fg_proposals))
    print('------- End Generating Proposals -------')

    all_proposals_feat = torch.stack(all_proposals_feat, dim=0).detach().cpu().numpy()
    all_proposals_cls_label = np.array(all_proposals_cls_label)
    all_proposal_conf_scores = np.array(all_proposal_conf_scores)

    # --- statistics over obtained proposals
    num_final_proposals = all_proposals_feat.shape[0]
    print('\nGet %d valid proposals from %d objects' % (num_final_proposals, num_original_proposals))
    unique_class, unique_class_count = np.unique(all_proposals_cls_label, return_counts=True)
    for (class_id, count) in zip(unique_class, unique_class_count):
        print('\tClass %d: %s| %d proposals' % (class_id, dataset_config.class2type[class_id], count))

    all_proposals = {}
    all_proposals['features'] = all_proposals_feat  # np.array
    all_proposals['class_labels'] = all_proposals_cls_label  # np.array
    all_proposals['conf_scores'] = all_proposal_conf_scores

    return all_proposals


def generate_prototypes_v1(model, proto_dataloader):
    model.eval()
    num_original_propsals = 0
    all_proposals_feat = []
    all_proposals_cls_label = []
    all_proposals_cls_name = []
    print('\n------ Begin Generating Proposals ------')
    for batch_idx, input_dict in enumerate(proto_dataloader):
        input_dict['point_cloud'] = input_dict['point_cloud'].to(device)
        input_dict['bboxes_param'] = input_dict['bboxes_param'][0].numpy()
        input_dict['bboxes_cls_label'] = input_dict['bboxes_cls_label'][0].numpy()
        input_dict['bboxes_cls_name'] = input_dict['bboxes_cls_name'][0]
        num_original_propsals += input_dict['bboxes_cls_label'].shape[0]
        with torch.no_grad():
            proposals_feat, proposals_cls_label, proposals_cls_name = model.generate_proposal_features(input_dict)
        if proposals_feat is not None:
            all_proposals_feat.append(proposals_feat)
            all_proposals_cls_label.extend(proposals_cls_label)
            all_proposals_cls_name.extend(proposals_cls_name)

    all_proposals_feat = torch.cat(all_proposals_feat, dim=0).detach().cpu().numpy()
    all_proposals_cls_label = np.array(all_proposals_cls_label)
    all_proposals_cls_name = np.array(all_proposals_cls_name)
    assert all_proposals_feat.shape[0] == len(all_proposals_cls_label)
    print('------- End Generating Proposals -------')

    # --- statistics over obtained proposals
    num_final_proposals = all_proposals_feat.shape[0]
    print('\nGet %d valid proposals from %d objects' % (num_final_proposals, num_original_propsals))
    unique_class, unique_class_count = np.unique(all_proposals_cls_label, return_counts=True)
    for (class_id, count) in zip(unique_class, unique_class_count):
        print('\tClass %d: %s| %d proposals' % (class_id, dataset_config.class2type[class_id], count))

    all_proposals = {}
    all_proposals['features'] = all_proposals_feat  # np.array
    all_proposals['class_labels'] = all_proposals_cls_label  # np.array
    all_proposals['class_names'] = all_proposals_cls_name

    return all_proposals

def process_prototypes(obj, k=100, use_clustering=False):
    prototypes = []
    proto_labels = []
    original_prototypes = obj['prototypes']
    original_proto_labels = obj['proto_labels']
    labels = set(obj['proto_labels'])
    proto_dict = {label: np.where(original_proto_labels == label) for label in labels}
    for cls, inds in proto_dict.items():
        feats = torch.from_numpy(original_prototypes[inds]).to(device)
        if use_clustering:
            prototype = get_multiple_prototypes(feats, k)
        else:
            prototype = feats

        prototypes.append(prototype)
        num_prototypes = prototype.shape[0]
        class_labels = torch.zeros(num_prototypes, len(proto_dict))
        class_labels[:, cls] = 1
        proto_labels.append(class_labels)
    prototypes = torch.cat(prototypes, dim=0)
    proto_labels = torch.cat(proto_labels, dim=0)
    return prototypes, proto_labels

def combine_prototypes(v1, v2):
    v1['prototypes'] = torch.from_numpy(v1['prototypes'])
    v2['prototypes'] = torch.from_numpy(v2['prototypes'])
    v1['proto_labels'] = torch.from_numpy(v1['proto_labels'])
    v2['proto_labels'] = torch.from_numpy(v2['proto_labels'])
    prototypes = torch.cat([v1['prototypes'], v2['prototypes']], dim=0).detach().cpu().numpy()
    labels = np.array(torch.cat([v1['proto_labels'], v2['proto_labels']], dim=0))
    return {'prototypes': prototypes, 'proto_labels': labels}


def get_multiple_prototypes(feat, k):
    """
    Extract multiple prototypes by points separation and assembly
    Args:
        feat: input point features, shape:(n_points, feat_dim)
    Return:
        prototypes: output prototypes, shape: (n_prototypes, feat_dim)
    """
    # sample k seeds as initial centers with Farthest Point Sampling (FPS)
    n = feat.shape[0]
    feat_dim = feat.shape[1]
    assert n > 0
    ratio = k / n
    if ratio < 1:
        fps_index = fps(feat, None, ratio=ratio, random_start=False).unique()
        num_prototypes = len(fps_index)
        farthest_seeds = feat[fps_index]

        # compute the point-to-seed distance
        distances = F.pairwise_distance(feat[..., None], farthest_seeds.transpose(0, 1)[None, ...],
                                        p=2)  # (n_points, n_prototypes)

        # hard assignment for each point
        assignments = torch.argmin(distances, dim=1)  # (n_points,)

        # aggregating each cluster to form prototype
        prototypes = torch.zeros((num_prototypes, feat_dim)).cuda()
        for i in range(num_prototypes):
            selected = torch.nonzero(assignments == i).squeeze(1)
            selected = feat[selected, :]
            prototypes[i] = selected.mean(0)
        return prototypes
    else:
        return feat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 40000]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_nms', action='store_true', help='Use NMS to suppress images.')
    parser.add_argument('--use_v1', action='store_true', help='Use normal prototypes generation.')
    parser.add_argument('--num_target', type=int, default=1024, help='Proposal number [default: 1024]')
    parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
    parser.add_argument('--labeled_sample_list', default='scannetv2_train.txt',
                        help='Labeled sample list from a certain percentage of training [static]')
    parser.add_argument('--detector_checkpoint_path', default=None,
                        help='Pretrained votenet (with labeled data) checkpoint path [Must Given!]')
    parser.add_argument('--proto_file', default='proto.pt', help='location to store prototypes')

    args = parser.parse_args()

    PROTO_FILE = args.proto_file

    PROTOTYPES_DATASET = ScannetPrototypeDataset(labeled_sample_list=args.labeled_sample_list,
                                                 num_points=args.num_point,
                                                 use_color=args.use_color,
                                                 use_height=(not args.no_height))

    PROTOTYPES_DATALOADER = DataLoader(PROTOTYPES_DATASET, batch_size=1, shuffle=False, num_workers=0,
                                       collate_fn=custom_collate)

    dataset_config = PROTOTYPES_DATASET.dataset_config

    model, device = init_detection_model(args, dataset_config)

    config_dict = {'dataset_config': dataset_config}

    if args.use_v1:
        all_proposals = generate_prototypes_v1(model, PROTOTYPES_DATALOADER)
    else:
        all_proposals = generate_prototypes(model, PROTOTYPES_DATALOADER, config_dict, use_nms=args.use_nms)

    prototypes, labels = all_proposals['features'], all_proposals['class_labels']

    if args.use_v1:
        visualise_features(prototypes, labels, 'assets/original_prototypes')
    else:
        visualise_features(prototypes, labels, 'assets/original_prototypes_overlap_front')

    torch.save({'prototypes': prototypes, 'proto_labels': labels}, PROTO_FILE)


# if __name__ == '__main__':
#     v1 = torch.load('v1_proto.pt')
#     v2 = torch.load('v2_proto.pt')
#     obj = combine_prototypes(v1, v2)
#     visualise_features(obj['prototypes'], obj['proto_labels'], 'assets/combined_prototypes')
#     torch.save(obj, 'combined_proto.pt')