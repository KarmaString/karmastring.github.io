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
- [47] Adversarial inverse graphics networks: Learning 2d-to-3d lifting and image-to-image translation from unpaired supervision.
  - Adversarial Prior
- [7] Human pose estimation with iterative error feedback
  - Iterative error feedback loop.

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
  - 有了3D joints之后，可以将其映射回2D的image: $\hat{x} = s\Pi (RX(\theta, \beta))+t$  
  - 对于所有的数据，我们有了一个reprejection的loss: $L_{reproj}=\sum_i \Vert v_i(x_i-\hat{x}_i) \Vert_1$
    其中$x_i \in \mathbb{R}^{2 \times K}$，为第$i$个ground truth 2D joints；$v_i \in \{0, 1 \}^K$是visibility，当对应joint可见时为1，不可见时为0。
  - 额外的，对于那些有着对应ground truth 3D joints的数据，还有3D loss: $L_{3D}=L_{3D joints}+L_{3D smpl}$  
    其中  
       $$L_{3DJoints} = \Vert X_i - \hat{X}_i \Vert^2_2$$  
    即生成的3D joints与ground truth 3D joints的欧氏距离的平方。  
       $$L_{3DSMPL}=\Vert [\beta_i, \theta_i]-[\hat{\beta_i}, \hat{\theta_i}] \Vert^2_2$$  
    即参数之差的平方和。
  - 对于所有的数据，还有一个adversarial loss。对encoder来说，目标是：  
       $$\min L_{adv}(E)=\sum_i \mathbb{E}_{\Theta \sim p_E}[(D_i(E(I))-1)^2]$$  
    即希望encoder能够让discriminator将其生成的参数判断为真。相对的对于discriminator，目标是：  
       $$\min L_{dis}(D_i)=\mathbb{E}_{\Theta \sim data}[(D_i(\Theta)-1)^2]+\mathbb{E}_{\Theta \sim p_E}[D_i(E(I))^2]$$  
    即希望discriminator能将生成的3D Mesh参数判断为假，同时将数据中3D Mesh参数判断为真。
  - 值得注意是，paper中用了多个discriminator。1个discriminator针对shape参数；1个discriminator针对所有的joints参数；23个discriminator针对每个joint。
  - 综上，overall objective of encoder：$L=\lambda(L_{reproj}+\mathbb{1}L_{3D})+L_{adv}$

### 总结

这个paper中最核心的思想是利用adversarial learning，在没有paired 2D/3D skeleton的情况下，学习3D human mesh。 如果我们将2D skeleton和3D skeleton看成是两个domains的话，这个思想是和`Cycle-GAN`的思想是一致的。

methodology的理解依赖于以下知识/概念：  
- Adversarial prior [47]。
- 基于SMPL [24] 的human mesh model。
- 基于ResNet [15] 的encoder网络。
- iterative error feedback (IEF) loop [7, 9, 31]。

---

# 实验

### dataset

- MS COCO, Human3.6M, MPI-INF-3DHP, LSP

### Metrics

- Mean per joint position error (MPJPE)
- Reconstruction error: MPJPE after rigid alignment of the prediction with ground truth via Procrustes Analysis [11]
- Percentage of Correct Keypoints(PCK) thresholded at 150mm
- the Area Under the Curve (AUC) over a rage of PCK thresholds [27].

### Experiments

- 作者对比了与baseline的结果
- 计算了时间，证明可以实时
- 做了Human Body Segmentation的实验
- 有/无 paired 3D supervision的实验

# 未解决的疑问
---

- iterative error feedback (IEF) loop [7, 9, 31] 是一个什么样的过程？（进一步调研）
- [5, 20]中对于SMPL和2D joints数目不匹配的问题的解决方案（进一步调研）
- 什么是Reconstruction error: MPJPE after rigid alignment of the prediction with ground truth via Procrustes Analysis [11] ; Reconstruction error removes global misalignments and evaluates the quality of the reconstructed 3D skeleton （进一步调研）
- [20]中好像有使用SMPL做body segment

# 启发意义的点
---

- iterative error feedback (IEF) loop [7, 9, 31] 可以帮助优化3D mesh的重构过程。
- 在解决问题的时候可以把要解决的问题考虑为多个domain，用adversarial learning的方式协助解决问题。
- 对于SMPL和2D joints数目不匹配的问题，[5, 20]似乎有相应的解决方案。
- Reconstruction error[11] 似乎可以给出没有global misalignments的结果
- Metrics中的PCK和AUC值得参考
- SMPL脸上的点与2D的标注有对应关系，如(在0-index下)
  - nose: 333
  - left eye: 2801
  - right eye: 6261
  - left ear: 584
  - right ear: 4072



