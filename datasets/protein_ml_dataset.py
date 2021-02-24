# https://github.com/CellProfiling/HPA-competition-solutions/blob/master/bestfitting/src/datasets/protein_ml_dataset.py
import os
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
from utils.common_util import *
import pandas as pd
from config.config import *
from datasets.tool import *
from utils.augment_util import *
from tqdm import tqdm
import shutil

class ProteinMLDataset(Dataset):
    def __init__(self,
                 split_file,
                 img_size=512,
                 transform=None,
                 return_label=True,
                 is_trainset=True,
                 in_channels=3,
                 ):
        self.is_trainset = is_trainset
        self.img_size = img_size
        self.return_label = return_label
        self.in_channels = in_channels
        self.transform = transform
        data_type = 'train' if is_trainset else 'test'
        split_df = pd.read_csv(split_file)
        base_dir = DATA_DIR

        if EXTERNAL not in split_df.columns:
            split_df[EXTERNAL] = False

        self.is_trainset = is_trainset

        self.split_df = split_df
        self.is_external = self.split_df[EXTERNAL].values
        self.img_ids = self.split_df[ID].values

        if self.return_label:
            self.labels = self.split_df[ANTIBODY_CODE].values.astype(int)

        self.num = len(self.img_ids)

        self.set_img_dir(base_dir, data_type)

        if ANTIBODY in self.split_df.columns:
            print(self.split_df[ANTIBODY].nunique())

    def set_img_dir(self, base_dir, data_type):
        if self.img_size<=512:
            self.img_dir = opj(base_dir, data_type, 'images')
            self.external_img_dir = opj(base_dir, 'train', 'external_v18_512')
        elif self.img_size<=768:
            self.img_dir = opj(base_dir, data_type, 'images_768')
            self.external_img_dir = opj(base_dir, 'train', 'external_v18_768')
        else:
            self.img_dir = opj(base_dir, data_type, 'images_1536')
            self.external_img_dir = opj(base_dir, 'train', 'external_v18_1536')
        print(self.img_dir)
        print(self.external_img_dir)

    def read_rgby(self, img_dir, img_id, index):
        if self.is_external[index]:
            img_is_external = True
        else:
            img_is_external = False

        suffix = '.jpg' if img_is_external else '.png'
        if self.in_channels == 3:
            colors = ['red', 'green', 'blue']
        else:
            colors = ['red', 'green', 'blue', 'yellow']

        flags = cv2.IMREAD_GRAYSCALE
        img = [cv2.imread(opj(img_dir, img_id + '_' + color + suffix), flags)
               for color in colors]
        img = np.stack(img, axis=-1)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img / 255.0
        return img

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        if self.is_external[index]:
            img_dir = self.external_img_dir
        else:
            img_dir = self.img_dir

        image = self.read_rgby(img_dir, img_id, index)
        if image is None:
            print(img_dir, img_id)

        if self.transform is not None:
            image = self.transform(image)

        image = image_to_tensor(image)
        if self.return_label:
            label = self.labels[index]
            return image, label, index
        else:
            return image, index

    def __len__(self):
        return self.num
