# HWA-UNETR: Hierarchical Window Aggregate UNETR for 3D Multimodal Stomach Lesion Segmentation
All code be released after the paper is accepted.


## Overview
![Overview](./img/Overview.png)



# Data Description
## New dataset:  SCM 2025
Dataset Name: Stomach Cancer MRI, SCM 2025

Modality: MRI

Size: 500 3D volumes (400 Training + 100 Validation)

- Download the official SCM 2025 dataset from the link below and place then into "TrainingData" in the dataset folder:
    通过网盘分享的文件：SCM 2025
    链接: https://pan.baidu.com/s/1LC9RYFeYNC1kIh2W7EGOUw?pwd=Hscm 提取码: Hscm 

SCM 2025 includes three types of MRI stomach scans along with their corresponding professional annotations for tumors: fat-suppressed T2-weighted (FS-T2W), apparent diffusion coefficient (ADC), and venous phase contrast-enhanced T1-weighted (CE-T1W). Challengers need to integrate and utilize the modal information to achieve precise lesion segmentation.

## publicly BraTS2021

Modality: MRI

Size: 1470 3D volumes (1251 Training + 219 Validation)

Challenge: RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge

- Register and download the official BraTS 21 dataset from the link below and place then into "TrainingData" in the dataset folder:

  https://www.synapse.org/#!Synapse:syn27046444/wiki/616992

The sub-regions considered for evaluation in BraTS 21 challenge are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT). The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (NCR) parts of the tumor. The appearance of NCR is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edematous/invaded tissue (ED), which is typically depicted by hyper-intense signal in FLAIR [[BraTS 21]](http://braintumorsegmentation.org/).
