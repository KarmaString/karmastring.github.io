---
title: "ICCV19 - Skeleton-Aware 3D Human Shape Reconstruction From Point Clouds"
date: 2021-02-04 17:15:00 -0400
categories: [Research]
tags: [PaperReading, Unfinished]
image: /assets/img/blog/SAShape.png
math: true
---

# 概述
---

这篇paper"_Skeleton-Aware 3D Human Shape Reconstruction From Point Clouds_"是在ICCV2019上发表的。根据作者的叙述，这可能是**第一篇用点云数据并辅以SMPL Model来做3D human reconstruction的文章**。以下用`SAShape`来指代这篇paper。

### 目标
用3D Point Cloud数据生成人的3D Shape （在具体实现中，作者先在synthesized dataset上做了supervised training，再于unseen dataset上做unsupervised fine-tuning）

### 主要贡献：
- Skeleton awareness，即在直接的point feature到SMPL parameter的映射中间，加了一层skeleton joint feature。作者在Introduction中提到，最直接达成目标的方案就是先用PointNet/PointNet++提取点云的feature，再将feature直接feed到SMPL中来得到3D Mesh。不过，由于SMPL中的shape和pose interact in a highly nonlinear way[16,31]，作者就提出用这个skeleton aware的方式，先预估出骨架的位置（也就是pose的信息），再进一步得到对应的SMPL model的pose和shape parameter。
- 三个module：在原始的PointNet++中加入graph aggregation module；在PointNet++后加入attention module；在计算skeleton feature的过程中加入skeleton graph module。

### 核心reference：
- [34;`PointNet`] Pointnet: Deep learning on point sets for 3d classiﬁcation and segmentation [CVPR17]
  - 经典点云处理方案
- [35;`PointNet++`] Pointnet++: Deep hierarchical feature learning on point sets in a metric space [NIPS17]
  - 经典点云处理方案
- [21; `SMPL`] SMPL: A skinned multiperson linear model [SIGGRAPHAsia15]
  - 3D Human Mesh是基于SMPL model的
  - 作者指出了SMPL的好处如下：
    - SMPL allows independent analysis or control of shape or pose
    - SMPL avoids the direct modeling of rugged and twisted shapes
    - SMPL is differentiable and thus can be easily integrated with neural networks

---

# 方法

### 概述

大体由三个模块组成：

- a modified PointNet++ module： 用来提取point cloud feature
- an attention module：将unordered point features 映射为ordered skeleton joint features
- a skeleton graph module： 用graph convolution从skeleton joint features中生成SMPL需要的参数

### Modified PointNet++ Module

具体实现是在PointNet++中每次MLP之前，做一下基于k-nearest neighbor生成的graph的spectral convolution。目的是为了explore point interactions。

### Attention Module

这个部分就是用基础的attention mechanism来计算skeleton joint features。作者在这里提到了permutation invariance的概念，并且例举了一些没法achieve permutation invariance的方法。

### Skeleton Graph Module

作者说之所以没有直接生成SMPL的参数是因为：SMPL shape and pose parameters interact in a nonlinear way. Shape parameters are used for joint predictions in the rest pose, which are further coupled with joint transformations derived from pose parameters.

### Offline Training + Online Tuning

Offline Training: 

- synthesized dataset
- 2 loss: vertex distance; Laplacian term to regularize/smooth over-bent shapes

Online Tuning:

- loss: Chamfer distance between the input point cloud and the reconstructed mesh.

### 总结

基本上没有太多的知识点，methodology依赖于以下知识/概念：

- PointNet/PointNet++ [34, 35] 来处理基础的点云
- 基于fast spectral convolution [36, 17, 11] 的graph convolution来处理点云中点之间的关系，以及joint之间的关系
- Attention mechanism [常见的neural network中aggregation的方法] 来将unordered point features映射为ordered skeleton joint features

---

# 实验

### dataset

- Synthesized Dataset: SURREAL dataset[43] provides SMPL shape and pose parameters for models captured in real scenarios, which enables generations of large numbers of human shapes with large variations and reasonable poses.
- Dyna Dataset [32] offers registered meshes with SMPL topology.
- DFAUST Dataset [6] provides raw scans of several persons in different motions.
- Berkeley MHAD dataset [26] provides two depth sequences from Kinect with human joint locations.

### Metrics

- Synthesized dataset & Dyna Dataset (who have ground truth): the average vertex-wise Euclidean distance. (Note that we use vertex-wise distance rather than vertex-to-surface distance as it can better reﬂect the distortion of reconstructed results.)
- DFAUST dataset & MHAD dataset (unknown ground truth): the average point-to-vertex distance.

### Ablation Study

- 借由替换提出的三个模块，作者证明了每个模块的有用性
- 列出了initial results和fine tuning results，证明了fine tuning的有用性
- 值得注意的是，作者**将身体不同地方loss用颜色可视化在了Mesh上**，并展示了fine tuning后身体各处的loss减少了

### Comparisons to the State-of-the-art

- Baseline是 3DCODED [13] 和 SMPLify [5]  (SMPLify-mesh & SMPLify-pcd)。其中SMPLify-mesh is in fact the upper bound, indicating the best that SMPL model can produce given the ground-truth mesh with the same topology. 通过数值比较证明了提出的方法的有效性和准确性。
- 在没有ground truth的数据集上给出了visualize的结果。通过visualization证明了提出的方法的准确性。

### Limitation

- 作者说our method has relatively large errors in female shape reconstruction，especially in chest, belly, and hip part。We conjecture this is inherited from the SMPL representation, since SMPLify-mesh exhibits similar problems. The possible reason may be that SMPL model only uses 10 parameters for shapes, which makes it hard to model body parts with a larger derivation from neutral shapes.
- Another major limitation is that our method is restricted to SMPL model and can only reconstruct naked human shapes.

# 值得注意的reference
---

- [46] Dynamic graph CNN for learning on point clouds
  - require a global knn graph, which results in $O(N^2)$ complexity in both space and time
- [13]
  - directly learn to reconstruct 3D human from point clouds
  - directly learned to deform a given template for human reconstruction, but often obtained twisted human shapes, especially in the shape arms.
  - **a Laplacian term to regularize/smooth over-bent shapes**
  - Baseline
- [20]
  - directly learn to reconstruct 3D human from point clouds
  - proposed a variational auto-encoder to learn for deformable shape completion, which often results in rugged surfaces.
- [16]
  - hard to directly regress SMPL parameters from image features
- [31]
  - hard to directly regress SMPL parameters from image features
- [48, 15, 14]
  - temporal consistency of acting persons
- [54, 4, 1]
  - reconstruct clothes and textures
- [55, 45]
  - IMU sensor is introduced for robust pose estimation in recent works
- [42, 31, 16, 5, 27, 1, 28]
  - reconstruct 3D human from images
- [40, 33, 47]
  - reconstruct human shape by predicting dense correspondence to a body surface
- [19]
  - proposed to use LSTM to leverage joint relations, but it only allows to propagate features from parent joints to their children.
- [5] SMPLify
  - proposed several important pose priors to prevent over-bent shapes and achieve successful reconstruction.
  - SMPLify-mesh 可以给出reconstruction的上界
- 文中提到了三类graph convolutional networks:
  - spatial-based methods [25, 41] learn features by directly filtering local neighbors on graph, and only a limited number of neighbors can be considered in each layer because of memory restriction.
  - Spectral-based methods [7, 23] learn features in Fourier domain constructed by the eigen-decomposition of Laplacian matrix. However, the unstable and computationally expensive eigen-decomposition makes it unsuitable to process noisy point data.
  - A compromise is the fast spectral convolution on graphs [36, 17, 11], which uses a k-order Chebyshev polynomial to approximate the spectral convolution and thus avoids eigen-decomposition



# 未解决的疑问
---
- 如何做graph convolutional networks？（基础知识欠缺）
- Chamfer distance是如何计算的？



