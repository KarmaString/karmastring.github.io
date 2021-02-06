---
title: "CVPR20 - VIBE: Video Inference for Human Body Pose and Shape Estimation"
date: 2020-10-05 15:15:00 -0400
categories: [Research]
tags: [PaperReading, Unfinished]
image: /assets/img/blog/VIBE.png
math: true
---

# 概述
---

这篇paper"_VIBE: Video Inference for Human Body Pose and Shape Estimation_"是由Max Planck Institute（MPI）的人在CVPR2020上发表的，并提供了[code和pretrained model](https://github.com/mkocabas/VIBE)。以下用`VIBE`来指代这篇paper。

### 目标
用monocular video数据生成人的3D motion的Pose和Shape

### 主要贡献：
- 使用了AMASS dataset对model进行了adversarial training。（这个方法其实已经是老生常谈的了，只是这种类似AMASS的3D motion capture dataset不常有。这个AMASS dataset看上去也主要是由MPI的人做的。）
- 在discriminator中使用了attention mechanism来weight the contribution of different frames。（使用attention也是比较常见的方法）
- quantitatively地与一些temporal architectures做了些比较。
- 达到了state-of-the-art的结果

### 核心reference：
- [29;`E2ERecovery`] End-to-end recovery of human shape and pose [CVPR18] 
  - 作者认为`E2ERecovery` train a single-image pose estimator using only 2D keypoints and an unpaired dataset of **static** 3D human shapes and poses using an adversarial training approach，从而能比较好的生成真实的**static**3D Human Mesh。而这篇`VIBE`正是用这个思想，并用了一个 large-scale 3D **motion-capture** dataset (AMASS [41]) 来生成真实的**dynamic** 3D Human Mesh。类似于所有的有经典adversarial net的model，`VIBE`对adversarial也有一个叙述，姑且将其记录下来：Here, given the video of a person, we train a temporal model to predict the parameters of the SMPL body model for each frame while a motion discriminator tries to distinguish between real and regressed sequences. By doing so, the regressor is encouraged to output poses that represent plausible motions through minimizing an adversarial training loss while the discriminator acts as weak supervision. The motion discriminator implicitly learns to account for the statics, physics and kinematics of the human body in motion using the ground-truth motion-capture data.
  - 作者用了`E2ERecovery` 中regressor的结构
- [36; `MFLoop`] Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop [ICCV19]
  - 是`VIBE`的前驱pretrained CNN网络，来从single-image生成body pose和shape的estimation。
- [40; `SMPL`] SMPL: A skinned multiperson linear model [SIGGRAPHAsia15]
  - 这篇paper `VIBE` 的3D Human Mesh就是基于SMPL model的

# 实验
---

- Criteria：
  - 3DPW [61]
  - MPI-INF-3DHP [42]

# 需要进一步调研的reference
---

- [30] Learning 3D human dynamics from video [CVPR19]
  - 这是这篇paper主要的baseline，与这篇paper有一样的目标。
- [29] End-to-end recovery of human shape and pose [CVPR18]
  - paper中有不少思想借鉴了这篇paper，其中最重要的是：[29] train a single-image pose estimator using only 2D keypoints and an unpaired dataset of static 3D human shapes and poses using an adversarial training approach。
- [53] Human Mesh Recovery From Monocular Images via a Skeleton-Disentangled Representation [ICCV19]
  - input是image或video；spatial and temporal information in a decoupling manor; self-attention; a transformer-based temporal model

# 未解决的疑问
---
- [29][36] 中的gender-neutral shape model是什么?

# Save for later
---

- 3D Human Mesh from image(s):
  - [11] Keep it SMPL: Automatic estimation of 3D human pose and shape from a single image [ECCV16]
     - 始祖级
  - [21] HoloPose: Holistic 3D Human Reconstruction In-The-Wild [CVPR19]
     - Loss看上去很漂亮；DensePose；**Euler angle limited to the convex hull to enforce attainable joint rotations**.
  - [25] Towards Accurate Marker-less Human Shape and Pose Estimation over Time [3DV17]
     - **Multi-view videos**；**temporal prior to handle the left and right side swapping issue**；silhouette
  - [36] Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop [ICCV19]
     - combine optimization and regression approaches; 生成的动作看上去比较复杂而且极具二义性
     - 同时也作为`VIBE`的前驱pretrained CNN网络，来从single-image生成body pose和shape的estimation。
  - [38] Unite the People: Closing the loop between 3D and 2D human representations [CVPR17]
  - [45] Neural Body Fitting: Unifying Deep Learning and Model Based Human Pose and Shape Estimation [3DV18]
     - 从part segmentation到pose estimation
  - [48] Learning to Estimate 3D Human Pose and Shape From a Single Color Image [CVPR18]
     - 最初接触的有关SMPL的文章；silhouette & keypoints
  - [57] Self-supervised Learning of Motion Capture [NIPS17]
  - [6] Exploiting Temporal Context for 3D Human Pose Estimation in the Wild [CVPR19]
     - 使用了temporal信息

- 3D Human Pose (skeleton)：
  - [35] Self-Supervised Learning of 3D Human Pose using Multi-view Geometry [CVPR19]
     - Multi-view images；no 3D ground truth (self-supervise)
  - [15] Learning 3D Human Pose from Structure and Motion [ECCV18]
  - [24] Exploiting temporal information for 3D human pose estimation [ECCV18]
  - [43] Single-shot multi-person 3D pose estimation from monocular RGB [3DV18]
  - [44] VNect: Real-time 3D human pose estimation with a single RGB camera [SIGGRAPH17]
  - [49] 3D human pose estimation in video with temporal convolutions and semi-supervised training [CVPR19]

- Parametric 3D human body models:
  - [4; `Scape`] Scape: Shape completion and animation of people [SIGGRAPH05]
     - 除了SMPL以外的另一种方法，也需要知晓一下
  - [40; `SMPL`] SMPL: A skinned multiperson linear model [SIGGRAPHAsia15]
     - 经典方法
  - [47; `SMPL-X`] Expressive Body Capture: 3D Hands, Face, and Body from a Single Image [CVPR19]
     - 还是MPI做的，包含了手指、脚、面部表情的SMPL加强版

- Non-parametric body mesh reconstruction methods:
  - [37] Convolutional Mesh Regression for Single-Image Human Shape Reconstruction [CVPR19]
  - [51] PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization [ICCV19]
  - DeepCap: Monocular Human Performance Capture  Using Weak Supervision [CVPR20, Best Student Paper Honorable Mention]
  - LiveCap: Real-time Human Performance Capture from Monocular Video [TOG19]
  - [59] BodyNet: Volumetric Inference of 3D Human Body Shapes [ECCV18]

- GAN for modeling:
  - [9] HP-GAN: Probabilistic 3D human motion prediction via GAN [CVPR18]
  - [20] Adversarial geometry-aware human motion prediction [ECCV18]
  - [2] Structured prediction helps 3D human motion modeling [ICCV19]

- 2D Human Keypoint Estimation:
  - [50] DeepCut: Joint subset partition and labeling for multi person pose estimation [CVPR16]
