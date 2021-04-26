from __future__ import print_function, division
import os
import string
import itertools
import pickle
import shutil
from skimage.morphology import remove_small_objects, remove_small_holes
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils import data
from torch.autograd import Variable
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    ShiftScaleRotate,
    RandomBrightness
)


class DatasetProcessor(data.Dataset):
    
    def __init__(self, root_path, file_list, is_test=False, as_torch_tensor=True, augmentations=False, mask_weight=True):
        self.is_test = is_test
        self.mask_weight = mask_weight
        self.root_path = root_path
        self.file_list = file_list
        self.as_torch_tensor = as_torch_tensor
        self.augmentations = augmentations
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        self.been = []
        
    def clear_buff(self):
        self.been = []
    
    def __len__(self):
        return len(self.file_list)

    def transform(self, image, mask):
        aug = Compose([
            HorizontalFlip(p=0.9),
            RandomBrightness(p=.5,limit=0.3),
            RandomContrast(p=.5,limit=0.3),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, 
                             p=0.7,  border_mode=0, interpolation=4)
        ])
        
        augmented = aug(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
    def get_mask_weight(self, mask):
        mask_ = cv2.erode(mask, kernel=np.ones((8,8),np.uint8), iterations=1)
        mask_ = mask-mask_
        return mask_ + 1
    
    def __getitem__(self, index):
        
        file_id = index
        if type(index) != str:
            file_id = self.file_list[index]
        
        #image_folder = self.root_path # original
        image_folder = self.root_path
        image_path = os.path.join(image_folder, file_id + ".jpg")
        
        #mask_folder = self.root_path[:-1] + "_mask/" # original
        mask_folder = self.root_path[:-7] + "\masks" #-1 goes one character back like cd..
        mask_path = os.path.join(mask_folder, file_id + ".png")
        
        if self.as_torch_tensor:
                    
            if not self.is_test:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(mask_path))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
                #resize to 320x256
                image = cv2.resize(image, (256, 320), interpolation=cv2.INTER_LANCZOS4)
                mask = cv2.resize(mask, (256, 320), interpolation=cv2.INTER_LANCZOS4)
                
                if self.augmentations:
                    if file_id not in self.been:
                        self.been.append(file_id)
                    else:
                        image, mask = self.transform(image, mask)
                    
                mask = mask // 255
                mask = mask[:, :, np.newaxis]
                if self.mask_weight:
                    mask_w = self.get_mask_weight(np.squeeze(mask))
                else: 
                    mask_w = np.ones((mask.shape[:-1]))
                mask_w = mask_w[:, :, np.newaxis]
                    
                mask = torch.from_numpy(np.transpose(mask, (2, 0, 1)).astype('float32'))
                mask_w = torch.from_numpy(np.transpose(mask_w, (2, 0, 1)).astype('float32'))
                image = self.norm(image)
                return image, mask, mask_w

            else:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 320), interpolation=cv2.INTER_LANCZOS4)
                image = self.norm(image)
                return image
            
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image, dtype=np.uint8)
            if not self.is_test:
                mask = cv2.imread(str(mask_path))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if self.augmentations:
                    if file_id not in self.been:
                        self.been.append(file_id)
                    else:
                        image, mask = self.transform(image, mask)
                return image, mask
            
            else:
                if self.augmentations:
                    if file_id not in self.been:
                        self.been.append(file_id)
                    else:
                        image = self.transform(image)
                return image