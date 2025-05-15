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
                        keys=["image"],  
                        a_min=model_scale[0],  
                        a_max=model_scale[1],  
                        b_min=0.0,            
                        b_max=1.0,           
                        clip=True            
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
    sizes = [math.ceil(len(data) * r) for r in ratios]
    
    total_size = sum(sizes)
    if total_size != len(data):
        sizes[-1] -= (total_size - len(data))
    
    start = 0
    parts = []
    for size in sizes:
        end = start + size
        parts.append(data[start:end])
        start = end
    
    return parts

def get_dataloader_GCM(config: EasyDict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.GCM_loader.root
    datapath = root
    use_models = config.GCM_loader.checkModels
    
    # Split the dataset into chronological order
    use_data_list = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]

    # Eliminate unnecessary medical record numbers
    remove_list = config.GCM_loader.leapfrag
    use_data = [item for item in use_data_list if item not in remove_list]
    
    random.shuffle(use_data)
    print('Random Loading!')
    train_use_data, val_use_data, test_use_data = split_list(use_data, [config.GCM_loader.train_ratio, config.GCM_loader.val_ratio, config.GCM_loader.test_ratio]) 
    if config.GCM_loader.fusion == True:
        need_val_data = val_use_data + test_use_data
        val_use_data = need_val_data
        test_use_data = need_val_data
    
    # Load MR data
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