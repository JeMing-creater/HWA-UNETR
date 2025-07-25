# HWA-UNETR: Hierarchical Window Aggregate UNETR for 3D Multimodal Gastric Lesion Segmentation

[![MICCAI](https://img.shields.io/badge/MICCAI-Target_Conference-blue.svg)](https://miccai.org/)
[![GCM 2025](https://img.shields.io/badge/Dataset_Reference-Awesome_Medical_Dataset-orange.svg)](https://drive.google.com/file/d/1S_ld7KqXcq0BbmkfXdPd8aLQFLsvZ2xR/view?usp=drive_link)

### 🎉 This work is provisionally accepted by [MICCAI 2025](https://conferences.miccai.org/2025/en/default.asp)!

![Overview](./img/Overview.jpg)


# How to run this code
All of the following operations must be entered in the operating system's command line.

## Get project
```
git clone git@github.com:JeMing-creater/HWA-UNETR.git
```

## Environment install
### It is recommended to use Python 3.10.x to avoid ImportError

```
cd requirements

# Necessary environment requirements
pip install -r requirements.txt

# Mamba base environment
cd /requirements/Mamba/causal-conv1d
python setup.py install
cd /requirements/Mamba/mamba
python setup.py install
# TFM method setting
# Find the mamba_sample.py file and replace it with requirements\mamba_sample.py
```

## Training
Single-GPU training
```
python3 train_seg_GCM.py
```
Multi-GPU training. User can change the GPU settings in train.sh. 
```
sh train.sh
```


# Data Description
## New dataset:  GCM 2025
**Dataset Name**: Gastric Cancer MRI, GCM 2025

**Modality**: MRI

**Size**: 500 3D volumes (400 Training + 100 Validation)

- Download the official GCM 2025 dataset from the link below: <br>
    https://drive.google.com/file/d/1S_ld7KqXcq0BbmkfXdPd8aLQFLsvZ2xR/view?usp=drive_link

![GCM_V](./img/GCM_V.jpg)

**How to use it:**
Users need to download the dataset file `GCM2025.zip` from above link, and put it in the same directory as this file and run the following command:
```
unzip GCM2025.zip
```
and then users can note that there will be a dir name `data` that includes all data and a `GCM2025.xlxs` that records all detailed information.

**Introduction**: GCM 2025 includes three types of MRI stomach scans along with their corresponding professional annotations for tumors: fat-suppressed T2-weighted (FS-T2W), apparent diffusion coefficient (ADC), and venous phase contrast-enhanced T1-weighted (CE-T1W). Challengers need to integrate and utilize the modal information to achieve precise lesion segmentation.

**Data Acquisition Criteria**: The technical scanning parameters for all data samples in the GCM 2025 dataset are comprehensively summarized in the following table:<br>
![Data Acquisition Criteria](./img/DAC.jpg)

**Data Acquisition Criteria**: The acquisition criteria of GCM 2025 dataset are as follows: (a) histopathologically confirmed gastric cancer; (b) preoperative MRI within 2 weeks prior to surgery; (c) with no history of other malignancies. Exclusion criteria comprised: (a) suboptimal image quality and (b) tumor lesion size too small to be segmented. <br>



## publicly BraTS 2021

**Modality**: MRI

**Size**: 1470 3D volumes (1251 Training + 219 Validation)

**Challenge**: RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge

- Register and download the official BraTS 21 dataset from the link below and place them into "TrainingData" in the dataset folder:

  https://www.synapse.org/#!Synapse:syn27046444/wiki/616992

**Introduction**: The sub-regions considered for evaluation in BraTS 21 challenge are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT). The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (NCR) parts of the tumor. The appearance of NCR is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edematous/invaded tissue (ED), which is typically depicted by hyper-intense signal in FLAIR [[BraTS 21]](http://braintumorsegmentation.org/).

# Experiments
![Table1](./img/Table1.jpg)
![Table2](./img/Table2.jpg)

# Visual Comparisons 
![Visual1](./img/Visual.png)
![Visual2](./img/Heatmap.png)

## Notion
If there are any problems, please don't hesitate to contact us by email: [jaming.work@gmail.com
](mailto:jaming.work@gmail.com
)

## Citation
If you find this repository useful or utilize the GCM 2025 Dataset, please consider citing this paper:
```
@article{liang2025hwa,
  title={HWA-UNETR: Hierarchical Window Aggregate UNETR for 3D Multimodal Gastric Lesion Segmentation},
  author={Liang, Jiaming and Dai, Lihuan and Sheng, Xiaoqi and Chen, Xiangguang and Yao, Chun and Tao, Guihua and Leng, Qibin and Cai, Honming and Zhong, Xi},
  journal={arXiv preprint arXiv:2505.10464},
  year={2025}
}
```

## Acknowledgement
Many thanks for these repos for their great contribution!

https://github.com/Project-MONAI/MONAI

https://github.com/MIC-DKFZ/nnUNet

https://github.com/ge-xing/SegMamba

https://github.com/lhaof/nnMamba

https://github.com/LeonidAlekseev/Swin-UNETR


