import os
import math
import yaml
import torch
import monai
import random
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from datetime import datetime
from easydict import EasyDict
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep
from monai.networks.utils import one_hot
sitk.ProcessObject.SetGlobalWarningDisplay(False)
from typing import Tuple, List, Mapping, Hashable, Dict
from monai.transforms import (
    LoadImaged, MapTransform, ScaleIntensityRanged, EnsureChannelFirstd, Spacingd, Orientationd,ResampleToMatchd, ResizeWithPadOrCropd, Resize, Resized, RandFlipd, NormalizeIntensityd, ToTensord,RandScaleIntensityd,RandShiftIntensityd
)


class ConvertToMultiChannelBasedOnBratsClassesd_For_BraTS(monai.transforms.MapTransform):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(self, keys: monai.config.KeysCollection, is2019: bool = False, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.is2019 = is2019

    def converter(self, img: monai.config.NdarrayOrTensor):
        # TC WT ET
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if self.is2019:
            result = [(img == 2) | (img == 3), (img == 1) | (img == 2) | (img == 3), (img == 2)]
        else:
            # TC WT ET
            result = [(img == 1) | (img == 4), (img == 1) | (img == 4) | (img == 2), img == 4]
            # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
            # label 4 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

    def __call__(self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]) -> Dict[
        Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

def read_csv_for_GCM(config):
    csv_path = config.GCM_loader.root + '/' + 'Classification.xlsx'
    # 定义dtype转换，将第二列（索引为1）读作str
    dtype_converters = {1: str}
    
    df1 = pd.read_excel(csv_path, engine='openpyxl', dtype=dtype_converters, sheet_name='腹膜转移分类')
    df2 = pd.read_excel(csv_path, engine='openpyxl', dtype=dtype_converters, sheet_name='淋巴结同时序（手术）')
    df3 = pd.read_excel(csv_path, engine='openpyxl', dtype=dtype_converters, sheet_name='淋巴结异时序（化疗后）')

    # 创建空字典
    content_dict1 = {}
    content_dict2 = {}
    content_dict3 = {}
    # 遍历DataFrame的每一行，从第二行开始
    for index, row in df1.iterrows():
        
        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        time = row[4]  # 第4列的数据读为time，用于划分数据
        content_dict1[key] = [values,time]
    
    for index, row in df2.iterrows():
        
        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        time = row[4] 
        content_dict2[key] = [values,time]

    for index, row in df3.iterrows():
        
        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        time = row[4] 
        content_dict3[key] = [values,time]
    
    return content_dict1, content_dict2, content_dict3
def read_usedata(file_path):
    read_flas = False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if read_flas == True:
                result = line.replace('\n', '').split(',')
                result = [data.replace(' ', '') for data in result]
                return result
            elif 'Useful data' in line:
                read_flas = True
                continue
def load_MR_dataset_images(root, use_data, use_models):
    images_path = os.listdir(root)
    images_list = []
    images_lack_list = []
    
    for path in use_data:
        if path in images_path:
            models = os.listdir(root + '/' + path + '/')
        else:
            continue
        image = []
        label = []
        
        for model in models:
            if model in use_models:
                if not os.path.exists(root + '/' + path + '/' + model ):
                    print(f"{path} does not have {model} file. ")
                    break
                elif not os.path.exists(root + '/' + path + '/' + model + '/' + path + '.nii.gz'):
                    print(f"{path} does not have {model} image file.")
                    break

                image.append(root + '/' + path + '/' + model + '/' + path + '.nii.gz')
                if not os.path.exists(root + '/' + path + '/' + model + '/' + path + 'seg.nii.gz'):
                    print(f"Label file not found for {path} in model {model}. ")
                    label.append(root + '/' + path + '/' + model + '/' + path + '.nii.gz')
                else:
                    label.append(root + '/' + path + '/' + model + '/' + path + 'seg.nii.gz')
        
        images_list.append({
                'image': image,
                'label': label,
            })
        
    return images_list
def load_brats2021_dataset_images(root):
    images_path = os.listdir(root)
    images_list = []
    for path in images_path:
        image_path = root + '/' + path + '/' + path
        flair_img = image_path + '_flair.nii.gz'
        t1_img = image_path + '_t1.nii.gz'
        t1ce_img = image_path + '_t1ce.nii.gz'
        t2_img = image_path + '_t2.nii.gz'
        seg_img = image_path + '_seg.nii.gz'
        images_list.append({
            'image': [flair_img, t1_img, t1ce_img, t2_img],
            'label': seg_img
        })
    return images_list
def get_GCM_transforms(config: EasyDict) -> Tuple[
    monai.transforms.Compose, monai.transforms.Compose]:
    load_transform = []
    for model_scale in config.GCM_loader.model_scale:
        load_transform.append(
            monai.transforms.Compose([
                LoadImaged(keys=["image", "label"], image_only=False, simple_keys=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                Resized(keys=["image", "label"], spatial_size=config.GCM_loader.target_size, mode=("trilinear", "nearest-exact")),
                
                ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,            # 输出图像的最小强度值
                        b_max=1.0,            # 输出图像的最大强度值
                        clip=True             # 是否裁剪超出范围的值
                    ),
                ToTensord(keys=['image', 'label'])
            ])
        )
    
    train_transform = monai.transforms.Compose([
        # 训练集的额外增强
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=['image', 'label'])
    ])
    val_transform = monai.transforms.Compose([
        ToTensord(keys=["image", "label"]),
    ])
    return load_transform, train_transform, val_transform

class MultiModalityDataset(monai.data.Dataset):
    def __init__(self, data, loadforms, transforms, use_class=True):
        self.data = data
        self.transforms = transforms
        self.loadforms = loadforms
        self.use_class = use_class
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        combined_data = {}
        
        for i in range(0, len(item['image'])):
            # print('Loading ', item['image'][i])
            globals()[f'data_{i}'] = self.loadforms[i]({
                'image': item['image'][i],
                'label': item['label'][i]
            })

            combined_data[f'model_{i}_image'] = globals()[f'data_{i}']['image']
            combined_data[f'model_{i}_label'] = globals()[f'data_{i}']['label']

            
            imgae = combined_data[f'model_{i}_image']
            combined_data[f'model_{i}_image'] = imgae
            
        images = []
        labels = []
        
        for i in range(0, len(item['image'])):
            images.append(combined_data[f'model_{i}_image'])
            labels.append(combined_data[f'model_{i}_label'])
            image_tensor = torch.cat(images, dim=0)
            label_tensor = torch.cat(labels, dim=0)
        
        result = {'image': image_tensor, 'label': label_tensor}
        result = self.transforms(result)

        if self.use_class == True:
            return {'image': result['image'], 'label': result['label'], 'class_label': torch.tensor(item['class_label']).unsqueeze(0).long()}
        else:
            return {'image': result['image'], 'label': result['label']}
def split_list(data, ratios):
    # 计算每个部分的大小
    sizes = [math.ceil(len(data) * r) for r in ratios]
    
    # 调整大小以确保总大小与原列表长度匹配
    total_size = sum(sizes)
    if total_size != len(data):
        sizes[-1] -= (total_size - len(data))
    
    # 分割列表
    start = 0
    parts = []
    for size in sizes:
        end = start + size
        parts.append(data[start:end])
        start = end
    
    return parts

def split_examples_to_data(data, config, lack_flag=False, loding=False):
    def read_file_to_list(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # 去除每行末尾的换行符
        lines = [line.strip() for line in lines]
        return lines

    def select_example_to_data(data, example_list):
        selected_data = []
        for d in data:
            num = d['image'][0].split('/')[-1].split('.')[0]
            if num in example_list:
                selected_data.append(d)
        return selected_data

    def load_example_to_data(data, example_path, loding=False):
        data_list = read_file_to_list(example_path)
        print(f'Loading examples from {example_path}')
        if loding == True:
            data_list = select_example_to_data(data, data_list)
        return data_list

    train_example = config.GCM_loader.root + '/' + 'train_examples.txt'
    val_example = config.GCM_loader.root + '/' + 'val_examples.txt'
    test_example = config.GCM_loader.root + '/' + 'test_examples.txt'
    
    train_data, val_data, test_data = load_example_to_data(data, train_example, loding), load_example_to_data(data, val_example, loding), load_example_to_data(data, test_example, loding)

    if lack_flag == True:
        train_data_lack, val_data_lack, test_data_lack = load_example_to_data(data, train_example, loding), load_example_to_data(data, val_example, loding), load_example_to_data(data, test_example, loding)
        return train_data, val_data, test_data, train_data_lack, val_data_lack, test_data_lack
    
    return train_data, val_data, test_data 
    
def get_dataloader_GCM(config: EasyDict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.GCM_loader.root
    datapath = root
    use_models = config.GCM_loader.checkModels
    
    # 按时间顺序划分数据集
    use_data_list = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]

    # 剔除不需要的病历号
    remove_list = config.GCM_loader.leapfrag
    use_data = [item for item in use_data_list if item not in remove_list]
    
    # 在use_data处划分数据，避免并行导致的读取问题
    random.shuffle(use_data)
    print('Random Loading!')
    train_use_data, val_use_data, test_use_data = split_list(use_data, [config.GCM_loader.train_ratio, config.GCM_loader.val_ratio, config.GCM_loader.test_ratio]) 
    if config.GCM_loader.fusion == True:
        need_val_data = val_use_data + test_use_data
        val_use_data = need_val_data
        test_use_data = need_val_data
    
    # 加载MR数据
    train_data = load_MR_dataset_images(datapath, train_use_data, use_models)
    val_data = load_MR_dataset_images(datapath, val_use_data, use_models)
    test_data = load_MR_dataset_images(datapath, test_use_data, use_models)

    load_transform, train_transform, val_transform = get_GCM_transforms(config)

    train_dataset = MultiModalityDataset(data=train_data,  
                                         loadforms = load_transform,
                                         transforms=train_transform,
                                         use_class=False)
    val_dataset   = MultiModalityDataset(data=val_data, 
                                         loadforms = load_transform,
                                         transforms=val_transform,
                                         use_class=False)
    test_dataset   = MultiModalityDataset(data=test_data,
                                         loadforms = load_transform,
                                         transforms=val_transform,
                                         use_class=False)
    
    train_loader = monai.data.DataLoader(train_dataset, num_workers=config.GCM_loader.num_workers,batch_size=config.trainer.batch_size, shuffle=True)
    val_loader = monai.data.DataLoader(val_dataset, num_workers=config.GCM_loader.num_workers, batch_size=config.trainer.batch_size, shuffle=False)
    test_loader = monai.data.DataLoader(test_dataset, num_workers=config.GCM_loader.num_workers, batch_size=config.trainer.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    # TODO: debug, you need to set your config path here.
    config = EasyDict(yaml.load(open('/workspace/Jeming/HWA_UNETR/config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    train_loader, val_loader, test_loader = get_dataloader_GCM(config)
    
    for i, batch in enumerate(train_loader):
        try:
            print(batch['image'].shape)
            print(batch['label'].shape)
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue 

    for i, batch in enumerate(val_loader):
        try:
            print(batch['image'].shape)
            print(batch['label'].shape)
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue 

    for i, batch in enumerate(test_loader):
        try:
            print(batch['image'].shape)
            print(batch['label'].shape)
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue