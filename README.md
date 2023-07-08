# ParametricNet
ParametricNet is a new 6DoF pose estimation network for parametric shapes with keypoint learning and Hough voting scheme.

This is the code of tensorflow version for our ICRA2021 paper: [ParametricNet: 6DoF Pose Estimation Network for Parametric Shapes in Stacked Scenarios](https://ieeexplore.ieee.org/abstract/document/9561181).

# Environment
Ubuntu 18.04/20.04

python3.6, tensorflow1.5, opencv-python, sklearn, h5py, et al.

Our backbone PointSIFT is borrowed from [pointSIFT](https://github.com/MVIG-SJTU/pointSIFT).

# Dataset
Siléane dataset is available at [here](http://rbregier.github.io/dataset2017).

Parametric dataset is available at [here](https://github.com/lvwj19/ParametricNet/tree/main/ParametricDataset).

# Evaluation metric
The python code of evaluation metric is available at [here](https://github.com/rbregier/pose_recovery_evaluation).

# Citation
If you use this codebase in your research, please cite:

```
@inproceedings{zeng2021parametricnet,
  title={ParametricNet: 6DoF Pose Estimation Network for Parametric Shapes in Stacked Scenarios},
  author={Zeng, Long and Lv, Wei Jie and Zhang, Xin Yu and Liu, Yong Jin},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={772--778},
  year={2021},
  organization={IEEE}
}
```
