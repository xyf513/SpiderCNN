# SpiderCNN
**SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filters.** ECCV 2018  
Yifan Xu, Tianqi Fan, Mingye Xu, Long Zeng, Yu Qiao.

## Introduction
This project is based on our ECCV18 paper. You can find the [arXiv](https://arxiv.org/abs/1803.11527) version here.
```
@article{xu2018spidercnn,
  title={SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filters},
  author={Xu, Yifan and Fan, Tianqi and Xu, Mingye and Zeng, Long and Qiao, Yu},
  journal={arXiv preprint arXiv:1803.11527},
  year={2018}
}
```
SpiderCNN is a convolutional neural network that can process signals on point clouds.

## Installation
The code is based on [PointNet](https://github.com/charlesq34/pointnet)， and [PointNet++](https://github.com/charlesq34/pointnet2). Please install [TensorFlow](https://www.tensorflow.org/install/), and follow the instruction in [PointNet++](https://github.com/charlesq34/pointnet2) to compile the customized TF operators.  
The code has been tested with Python 2.7, TensorFlow 1.3.0, CUDA 8.0 and cuDNN 6.0 on Ubuntu 14.04.

## Usage
### Classification
Preprocessed ModelNet40 dataset can be downloaded [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).  
To train a SpiderCNN model (with input XYZ coordinates and normal vectors) to classify shapes in ModelNet40:
```
python train.py
```
To train a SpiderCNN model (with input XYZ coordinates) with multi GPU to classify shapes in ModelNet40:
```
python train_xyz.py
```

### Part Segmentation
Preprocessed ShapeNetPart dataset can be downloaded [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip).
To train a model to segment object parts for ShapeNet models (with input XYZ coordinates and normal vectors):
```
cd part_seg
python train.py
```

## License
This repository is released under MIT License (see LICENSE file for details).