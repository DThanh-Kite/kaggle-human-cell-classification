import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from loguru import logger

from .tool import image_to_tensor, label_to_tensor


# # creating duplicates for rare classes in train set
# class Oversampling:
#     def __init__(self, path):
#         self.train_labels = pd.read_csv(path).set_index('Id')
#         self.train_labels['Target'] = [[int(i) for i in s.split()] 
#                                        for s in self.train_labels['Target']]  
        
#         # Figure this out
#         #set the minimum number of duplicates for each class
#         self.multi = [1,1,1,1,1,1,1,1,
#                       4,4,4,1,1,1,1,4,
#                       1,1,1,1,2,1,1,1,
#                       1,1,1,4]

#     def get(self, image_id):
#         labels = self.train_labels.loc[image_id,'Target'] if image_id \
#           in self.train_labels.index else []
#         m = 1
#         for l in labels:
#             if m < self.multi[l]: m = self.multi[l]
#         return m
    
# s = Oversampling(os.path.join(PATH, LABELS))
# tr_n = [idx for idx in tr_n for _ in range(s.get(idx))]
# print(len(tr_n), flush=True)


class HPADataset(Dataset):
    def __init__(self,
                 root_dir='data/hpa-single-cell-image-classification',
                 img_size=512,
                 is_training=True,
                 return_label=True,
                 in_channels=3,
                 transform=None,
                 aug_version=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(
            root_dir,
            'train'
        )
        self.is_training = is_training
        self.in_channels = in_channels # RGB or RGBY
        self.img_size = img_size
        self.return_label = return_label

        # Load data from csv
        self.train_df = pd.read_csv(os.path.join(
            root_dir,
            'train.csv'
        ))
        logger.info(self.train_df.head())
        self.image_ids = self.train_df['ID'].values
        self.labels = self.train_df['Label'].values
        logger.info('Dataset loaded with {:d} images'.format(len(self.image_ids)))

        self.class_names = {
            0:  "Nucleoplasm", 
            1:  "Nuclear membrane",   
            2:  "Nucleoli",   
            3:  "Nucleoli fibrillar center" ,  
            4:  "Nuclear speckles",
            5:  "Nuclear bodies",
            6:  "Endoplasmic reticulum",   
            7:  "Golgi apparatus",
            8:  "Intermediate filaments",
            9:  "Actin filaments", 
            10: "Microtubules",
            11:  "Mitotic spindle",
            12:  "Centrosome",   
            13:  "Plasma membrane",
            14:  "Mitochondria",   
            15:  "Aggresome",
            16:  "Cytosol",   
            17:  "Vesicles and punctate cytosolic patterns",   
            18:  "Negative"
        }
        self.num_classes = len(self.class_names.keys())
        logger.info('{:d} available classes: {}'.format(self.num_classes,
                                                        self.class_names))
        
        # Augmentation
        self.augmentation = None

        # To Tensor
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def read_crop_img(self, img):
        random_crop_size = int(np.random.uniform(self.crop_size, self.img_size))
        x = int(np.random.uniform(0, self.img_size - random_crop_size))
        y = int(np.random.uniform(0, self.img_size - random_crop_size))
        crop_img = img[x:x + random_crop_size, y:y + random_crop_size]
        return crop_img

    def read_rgby(self, img_dir, img_id, index):
        suffix = '.png'
        if self.in_channels == 3:
            colors = ['red', 'green', 'blue']
        else:
            colors = ['red', 'green', 'blue', 'yellow']

        flags = cv2.IMREAD_GRAYSCALE
        img = [cv2.imread(os.path.join(img_dir, img_id + '_' + color + suffix), flags)
               for color in colors]
        img = np.stack(img, axis=-1)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, index):
        """ Generate one sample of data
        """
        image_dir = self.image_dir
        image_id = self.image_ids[index]
        image_path = os.path.join(self.image_dir, image_id)

        img = self.read_rgby(image_dir, image_id, index)
        if img is None:
            logger.error(image_dir, image_id)

        # Image Augmentation
        if self.augmentation is not None:
            img = self.augmentation.augment_image(img)
        
        # h, w = img.shape[:2]
        # if self.crop_size > 0:
        #     if self.crop_size != h or self.crop_size != w:
        #         image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        # else:
        #     if self.img_size != h or self.img_size != w:
        #         image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        img = img / 255.0
        img = image_to_tensor(img)

        # Create label 
        label = self.labels[index]
        # logger.debug('Label: {}'.format(label))
        label = label.split('|')
        label = list(map(int, label))
        label = np.eye(self.num_classes, dtype='float')[label]
        label = label.sum(axis=0)
        # logger.debug('Label: {}'.format(label))

        if self.return_label:
            return img, label, index
        else:
            return img, index

if __name__ == '__main__':
    from torchvision import datasets, transforms
    trsfm = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = HPADataset(
        root_dir='../data/hpa-single-cell-image-classification',
        in_channels=3,
        transform=trsfm
    )
    img, index = dataset.__getitem__(0)
