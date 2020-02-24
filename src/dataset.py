import pandas as pd
import numpy as np

import albumentations
import torch

from PIL import Image


class APTOS2019Dataset:
    def __init__(self, folds, img_height, img_width, training=True, mean=[], std=[]):
        df = pd.read_csv("../input/train.csv")

        self.image_ids = df.id_code.values
        self.diagnosis = df.diagnosis.values

        self.img_width, self.img_height = img_width, img_height

        df = df[df.kfold.isin(folds)].reset_index(drop=True)

        if training is True:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Blur(p=0.9),
                albumentations.ShiftScaleRotate(),
                albumentations.Normalize(mean, std, always_apply=True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_width, img_height, always_apply=True),
                albumentations.Normalize(mean, std, always_apply=True)
            ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image = Image.open(f'../input/train_images/{self.image_ids[idx]}.png').convert("RGB")
        image = image.reshape(self.img_height, self.img_width).astype(float)
        image = self.aug(np.array(image))["image"]
        image = np.tranpose(image, (2, 0, 1)).astype(np.float32)

        if self.training is True:
            return {
                'image': torch.tensor(image, dtype=torch.float),
                'diagnosis': torch.tensor(diagnosis, dtype=torch.float)
            }
        else:
            return {
                'image': torch.tensor(image, dtype=torch.float)
            }


