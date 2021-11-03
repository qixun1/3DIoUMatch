import os

import numpy as np
import torch
import vtk
from torch import nn

import vis_utils as vis
from utils.box_util import get_3d_box_depth


def visualize_votes(example):
    gt_bboxes = example['bboxes'].squeeze(0).detach().cpu().numpy()
    gt_box3d_list = []
    for i in range(gt_bboxes.shape[0]):
        box = gt_bboxes[i]
        box3d = get_3d_box_depth(box[3:6], 0, box[0:3])
        gt_box3d_list.append(box3d)

    sa1_indices = example['original_indices'][0][0]
    sa2_indices = example['original_indices'][1][0]
    bbox_indices = example['bbox_idx']

    vtk_actors = []
    vtk_sa1 = []
    vtk_sa2 = []
    vtk_aggregated_votes = []
    vtk_gt_boxes = []
    vtk_pc = vis.VtkPointCloud(example['point_clouds'].squeeze(0)[sa1_indices].detach().cpu().numpy(),
                               color=vis.Color.Gray, point_size=5)
    vtk_actors.append(vtk_pc.vtk_actor)

    # Visualize sa1
    vtk_sa1_pred = vis.VtkPointCloud(example['sa1_xyz'].squeeze(0)[sa2_indices].detach().cpu().numpy(), point_size=15,
                                     color=vis.Color.Red)
    vtk_sa1.append(vtk_sa1_pred.vtk_actor)

    # Visualize sa2
    vtk_sa2_pred = vis.VtkPointCloud(example['sa2_xyz'].squeeze(0)[bbox_indices].detach().cpu().numpy(), point_size=10,
                                     color=vis.Color.Purple)
    vtk_sa2.append(vtk_sa2_pred.vtk_actor)

    # Visualize aggregated votes
    vtk_agg_votes = vis.VtkPointCloud(example['aggregated_vote_xyz'].squeeze(0).detach().cpu().numpy(),
                                      point_size=5, color=vis.Color.Green)
    vtk_aggregated_votes.append(vtk_agg_votes.vtk_actor)

    # Visualize GT 3D boxes
    for box3d in gt_box3d_list:
        vtk_box3D = vis.vtk_box_3D(box3d, line_width=2, color=vis.Color.Green)
        vtk_gt_boxes.append(vtk_box3D)

    if 'center' in example:
        vtk_pred_boxes = []
        center = example['center'].detach().cpu().numpy()
        center = np.mean(center, axis=1).reshape(-1)
        pred_bbox = get_3d_box_depth((example['size']*2).detach().cpu().numpy().reshape(-1), 0, center)
        vtk_predbox3D = vis.vtk_box_3D(pred_bbox, line_width=2, color=vis.Color.Red)
        vtk_pred_boxes.append(vtk_predbox3D)

        label = vtk.vtkVectorText()
        obj_score = nn.Softmax(dim=2)(example['objectness_scores'])[:, :, 1]
        sem_cls_score = nn.Softmax(dim=2)(example['sem_cls_scores'])
        pred_cls_prob, pred_cls = torch.max(sem_cls_score, -1)
        cls = example['cls']
        act_cls_prob = sem_cls_score[:, :, cls]
        label.SetText(f'objectness score: {obj_score.detach().cpu().numpy().reshape(-1)}\n'
                      f'pred sem class: {pred_cls.detach().cpu().numpy().reshape(-1)} ({pred_cls_prob.detach().cpu().numpy().reshape(-1)})\n'
                      f'actual sem class: {cls.detach().cpu().numpy()} ({act_cls_prob.detach().cpu().numpy().reshape(-1)})')
        lblMapper = vtk.vtkPolyDataMapper()
        lblMapper.SetInputConnection(label.GetOutputPort())
        vtk_obj = vtk.vtkFollower()
        vtk_obj.SetMapper(lblMapper)
        vtk_obj.SetScale(0.15,0.15,0.15)
        vtk_obj.SetPosition(0,0,0)

        vtk_actors.append(vtk_obj)


    key_to_actors_to_hide = {'g': vtk_aggregated_votes, 'p': vtk_sa1, 'k': vtk_sa2, 'b': vtk_gt_boxes, 'r': vtk_pred_boxes}
    vis.start_render(vtk_actors, key_to_actors_to_hide=key_to_actors_to_hide, background_col=vis.Color.White)


def visualise_indices(bboxes, pc, indices):
    gt_bboxes = bboxes
    gt_box3d_list = []
    for i in range(gt_bboxes.shape[0]):
        box = gt_bboxes[i]
        box3d = get_3d_box_depth(box[3:6], 0, box[0:3])
        gt_box3d_list.append(box3d)

    vtk_actors = []
    vtk_gt_boxes = []
    vtk_pc = vis.VtkPointCloud(pc[indices],
                               color=vis.Color.Gray, point_size=5)
    vtk_actors.append(vtk_pc.vtk_actor)

    # Visualize GT 3D boxes
    for box3d in gt_box3d_list:
        vtk_box3D = vis.vtk_box_3D(box3d, line_width=2, color=vis.Color.Green)
        vtk_gt_boxes.append(vtk_box3D)

    key_to_actors_to_hide = {'b': vtk_gt_boxes}
    vis.start_render(vtk_actors, key_to_actors_to_hide=key_to_actors_to_hide, background_col=vis.Color.White)
