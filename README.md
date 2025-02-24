# HWA-UNETR: Hierarchical Window Aggregate UNETR for 3D Multimodal Gastric Lesion Segmentation
All dataset and code will be released after the paper is accepted.

## Abstract
In the realm of existing multimodal medical image segmentation approaches, analysis of gastric cancerous lesions poses unique challenges. As a clinical scenario marked by the lack of independent multimodal datasets and the requirement to integrate inherently misaligned modalities, it forces algorithms to train only on approximated data and rely on application migration, thus consuming significant resources and potentially compromising analysis accuracy. To address those challenges, we have made two major contributions: First, we publicly disseminate the GCM 2025 dataset, which serves as the first large-scale, open-source collection of gastric cancer multimodal MRI scans, featuring professionally annotated FS-T2W, ADC, and CE-T1W images from 500 patients. Second, we introduce HWA-UNETR, a novel 3D segmentation framework that employs an originality HWA block with learnable window aggregation layers to establish dynamic feature correspondences between different modalities' anatomical structures, and leverages the innovative tri-orientated fusion mamba mechanism for context modeling and capturing long-range spatial dependencies. Extensive experiments on our GCM dataset and the publicly BraTS 2021 dataset validate the performance of our framework, demonstrating that the new approach surpasses existing methods by up to 1.68\% in the Dice score while maintaining solid robustness.
<br>

![Overview](./img/Overview.png)



# Data Description
## New dataset:  GCM 2025
Dataset Name: Gastric Cancer MRI, GCM 2025

Modality: MRI

Size: 500 3D volumes (400 Training + 100 Validation)

- Download the official GCM 2025 dataset from the link below and place it into "TrainingData" in the dataset folder: <br>
    通过网盘分享的文件：GCM 2025<br>
    链接: https://pan.baidu.com/s/1ZxxGyaMb4CJlZH5bDSqHJw?pwd=gcmm 提取码: gcmm

GCM 2025 includes three types of MRI stomach scans along with their corresponding professional annotations for tumors: fat-suppressed T2-weighted (FS-T2W), apparent diffusion coefficient (ADC), and venous phase contrast-enhanced T1-weighted (CE-T1W). Challengers need to integrate and utilize the modal information to achieve precise lesion segmentation.

**Data Acquisition Criteria**: The technical scanning parameters for all data samples in the GCM 2025 dataset are comprehensively summarized in the following table:<br>
![Overview](./img/Table.jpg)

**Data Acquisition Criteria**: The acquisition criteria of GCM 2025 dataset are as follows: (a) histopathologically confirmed gastric cancer; (b) preoperative MRI within 2 weeks prior to surgery; (c) with no history of other malignancies. Exclusion criteria comprised: (a) suboptimal image quality and (b) tumor lesion size too small to be segmented. <br>

## publicly BraTS2021

Modality: MRI

Size: 1470 3D volumes (1251 Training + 219 Validation)

Challenge: RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge

- Register and download the official BraTS 21 dataset from the link below and place them into "TrainingData" in the dataset folder:

  https://www.synapse.org/#!Synapse:syn27046444/wiki/616992

The sub-regions considered for evaluation in BraTS 21 challenge are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT). The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (NCR) parts of the tumor. The appearance of NCR is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edematous/invaded tissue (ED), which is typically depicted by hyper-intense signal in FLAIR [[BraTS 21]](http://braintumorsegmentation.org/).
