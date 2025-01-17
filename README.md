# 3DIoUMatch

![teaser](imgs/teaser.png)

## Introduction

News: Our paper has been accepted by CVPR 2021!

This is a ***pre-release*** of the code of our paper *3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection*. (arXiv report [here](https://arxiv.org/abs/2012.04355v3)).

In this repository, we provide 3DIoUMatch implementation (with Pytorch) based on [VoteNet](https://github.com/facebookresearch/votenet) and [SESS](https://github.com/Na-Z/sess), as well as the training and evaluation scripts on SUNRGB-D and ScanNet.

Please refer to our [project page](https://thu17cyz.github.io/3DIoUMatch/) for more information.

## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{wang20213dioumatch,
  title={3DIoUMatch: Leveraging iou prediction for semi-supervised 3d object detection},
  author={Wang, He and Cong, Yezhen and Litany, Or and Gao, Yue and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14615--14624},
  year={2021}
}
```

## Installation

Install `Python` -- This repo is tested with Python 3.7.6.

Install `NumPy` -- This repo is tested with NumPy 1.18.5. Please make sure your NumPy version is at least 1.18.

Install `PyTorch` with `CUDA` -- This repo is tested with 
PyTorch 1.5.1, CUDA 10.1. It may work with newer versions, 
but that is not guaranteed. A lower version may be problematic.

Install `TensorFlow` (for `TensorBoard`) -- This repo is tested with TensorFlow 2.2.0.

Compile the CUDA code for [PointNet++](https://arxiv.org/abs/1706.02413), which is used in the backbone network:
```
cd pointnet2
python setup.py install
```

If there is a problem, please refer to [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch#building-only-the-cuda-kernels)

Compile the CUDA code for general 3D IoU calculation in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet):
```
cd OpenPCDet
python setup.py install
```

I deleted the CUDA kernels except 3D IoU calculation in OpenPCDet
for faster installation.

Install dependencies:
```
pip install -r requirements.txt
```

## Datasets

### ScanNet
Please follow the instructions in `scannet/README.md`.

### SUNRGB-D
Please follow the instructions in `sunrgbd/README.md`. 

## Download Pre-trained and Trained Models

We provide the pre-trained models of 
[ScanNet 10%](https://drive.google.com/file/d/1GewYk_XMFtKCG-fpChKraEVO3ClROSZG/view?usp=sharing) 
and [SUNRGB-D 5%](https://drive.google.com/file/d/1UzDllmfx-p2KsHUQJ0mT0maYkyNJwZyK/view?usp=sharing). 

We also provide the trained models of 
[ScanNet 10%](https://drive.google.com/file/d/1M2kRKWWMCVIRPTAMsQbehaFaMFG9vOv-/view?usp=sharing)
 and [SUNRGB-D 5%](https://drive.google.com/file/d/1RdTgSOEmn4CcZEjrGJff2z8raSkDxVbr/view?usp=sharing).

You may download the models and put them into `ckpts`.

We provide 2 data splits of ScanNet 10% and SUNRGB-D 5% in `scannet/meta_data` and `sunrgbd/sunrgbd_trainval`, respectively.

## Pre-training

Please run:
```shell script
sh run_pretrain.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST>
```

For example:
```shell script
sh run_pretrain.sh 0 pretrain_scannet scannet scannetv2_train_0.1.txt
``` 

```shell script
sh run_pretrain.sh 0 pretrain_sunrgbd sunrgbd sunrgbd_v1_train_0.05.txt
``` 

## Training

Please run:
```shell script
sh run_train.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <PRETRAIN_CKPT>
```

For example, use the downloaded models:
```shell script
sh run_train.sh 0 train_scannet scannet scannetv2_train_0.1.txt ckpts/scan_0.1_pretrain.tar
``` 

```shell script
sh run_train.sh 0 train_sunrgbd sunrgbd sunrgbd_v1_train_0.05.txt ckpts/sun_0.05_pretrain.tar
``` 
You may modify the script by adding `--view_stats`  to load labels on unlabeled data and view the statistics on the unlabeled data (e.g. average IoU, class prediction accuracy).


## Evaluation

Please run:
```shell script
sh run_eval.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <CKPT>
```

For example, use the downloaded models:
```shell script
sh run_eval.sh 0 eval_scannet scannet scannetv2_train_0.1.txt ckpts/scan_0.1.tar
``` 

```shell script
sh run_eval.sh 0 eval_sunrgbd sunrgbd sunrgbd_v1_train_0.05.txt ckpts/sun_0.05.tar
``` 

For evaluation with IoU optimization, please run:

Please run:
```shell script
sh run_eval_opt.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <CKPT> <OPT_RATE>
```

The number of steps (of optimization) is by default 10.


## Acknowledgements
Our implementation uses code from the following repositories:
- [Deep Hough Voting for 3D Object Detection in Point Clouds](https://github.com/facebookresearch/votenet)
- [SESS: Self-Ensembling Semi-Supervised 3D Object Detection](https://github.com/Na-Z/sess)
- [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

## Change log
12/31/2020: Pre-release.
