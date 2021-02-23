---
title: "CVPR18 - End-to-end Recovery of Human Shape and Pose"
date: 2021-02-16 18:45:00 -0400
categories: [Research]
tags: [PaperReading, Unfinished]
image: /assets/img/blog/E2ERecovery.png
math: true
---

# 概述
---

这篇paper"_End-to-end Recovery of Human Shape and Pose_"是在CVPR2018上发表的。在给出图片中人的Bounding box后， 这篇paper里的方法可以real-time地从2D image中重构出人的3D Mesh。

### 目标

从单张的2D RGB图像中重构出图像中人的3D mesh。作者提出的model可以使用2D-to-3D的supervision进行训练，也可以不使用任何paired 2D-to-3D supervision来进行weakly-supervised training。

### 面临的问题

作者列出了三个challenges：

- 缺少3D的对应数据。The lack of large-scale ground truth 3D annotation for in-the-wild images.
- 2D到3D映射的ambiguities。The inherent ambiguities in single-view 2D-to-3D mapping.
- rotation matrices比较难regress出来。已有的一些工作是将其离散化，并变成一个分类问题解决的 [46]。

### 主要贡献：

- 作者认为其核心贡献是：take advantage of these unpaired 2D keypoint annotations and 3D scans in a conditional generative adversarial manner.
- Infer 3D mesh parameters directly from image features, while previous approaches infer them from 3D keypoints. 作者认为这样做avoids the need for two stage training以及avoids throwing away a lot of image information.
- 输出Mesh。Output meshes instead of skeletons.
- The framework is end-to-end.
- The model can be trained without paired 2D-to-3D data.

### 核心reference：

- [24; `SMPL`] SMPL: A skinned multiperson linear model [SIGGRAPHAsia15]
  - SMPL parameterizes the mesh by 3D joint angles and a low-dimensional linear shape space.
  - The output mesh can be immediately used by animators, modified, measured, manipulated and retargeted.
  - It is non-trivial to estimate the full pose of the body from only the 3D joint locations, since joint locations alone do not constrain the full DoF at each joint.
  - Predicting rotations also ensures that limbs are symmetric and of valid length.

---

# 方法

### 概述

原理： 使用adversarial NN，借由对大量3D scans的数据训练，discriminator可以分辨出生成的SMPL的parameter是否plausible （与传统中一样，这个discriminator的训练是和model其他部分训练交叉进行的）。 然后作者直接使用 unpaired 2D keypoint annotations 的数据用reprojection loss 来进行训练，并加入discriminator的weakly-supervision， 让其能够单张的2D RGB图像中重构出图像中人的3D mesh。如果paired 3D annotation也存在的话，还会加上3D keypoint的loss。

### 详细过程

- 第一步：生成必要的参数
  - 作者先将原始图片直接输入`ResNet-50`，得到一个2048维的向量
  - 将这个2048维的向量，通过`3D Regression Module`得到85个参数 $\Theta$
  - $\Theta=\{\theta, beta, R, t, s\}$；其中$\theta$（23x3个pose参数）和$\beta$（10个shape参数）是SMPLmodel的输入参数；$R$（可以由一个长度为3的旋转向量表示）是global rotation；$t$（2个参数）是x，y平面上的translation；$s$ （1个参数）mesh的scale。
- 第二步：用SMPL生成Mesh，并计算reprojection loss，3D loss和adversarial loss。
  - 目标的Mesh是用SMPL生成的，即$M(\theta, \beta)$。另外SMPL还会给出对应的3D joints，即$X(\theta, \beta)$
  - 有了3D joints之后，可以将其映射回2D的image: $$\hat{x} = s\Pi (RX(\theta, \beta))+t$$
  - 对于所有的数据，我们有了一个reprejection的loss：$$L_{reproj}=\sum_i||v_i(x_i-\hat{x}_i)||_1$$ 其中$x_i \in \mathbb{R}^{2 \times K}$，为第$i$个ground truth 2D joints。$v_i \in \{0, 1\}^K$是visibility，当对应joint可见时为1，不可见时为0。
  - 额外的，对于那些有着对应ground truth 3D joints的数据，还有3D loss： $$L_{3D}=L_{3D joints}+L_{3D smpl}$$其中$$L_{3D joints}=||X_i-\hat{X}_i||^2_2$$，即生成的3D joints与ground truth 3D joints的欧氏距离的平方。$$L_{3D smpl}=||[\beta_i, \theta_i]-[\hat{\beta_i}, \hat{\theta_i}]||^2_2$$，即参数之差的平方和。
  - 对于所有的数据，还有一个adversarial loss。对encoder来说，目标是：$$\min L_{adv}(E)=\sum_i \mathbb{E}_{\Theta \sim p_E}[(D_i(E(I))-1)^2]$$即希望encoder能够让discriminator将其生成的参数判断为真。相对的对于discriminator，目标是：$$\min L_{dis}(D_i)=\mathbb{E}_{\Theta \sim data}[(D_i(\Theta)-1)^2]+\mathbb{E}_{\Theta \sim p_E}[D_i(E(I))^2]$$，即希望discriminator能将生成的3D Mesh参数判断为假，同时将数据中3D Mesh参数判断为真。
  - 值得注意是，paper中用了多个discriminator。1个discriminator针对shape参数；1个discriminator针对所有的joints参数；23个discriminator针对每个joint。
  - 综上，overall objective of encoder：$$L=\lambda(L_{reproj}+\mathbb{1}L_{3D})+L_{adv}$$

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

- Baseline: [5, 20]

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



