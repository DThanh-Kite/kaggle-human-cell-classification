from torchvision import datasets, transforms

from base import BaseDataLoader
from datasets.hpa_dataset import HPADataset


class HPADataLoader(BaseDataLoader):
    """
    DataLoader for Human Cell Classification Dataset
    """
    def __init__(self, data_dir,
                 batch_size,
                 shuffle=True,
                 training=True,
                 img_size=512,
                 in_channels=3,
                 validation_split=0.0,
                 num_workers=1,
                 aug_version=None):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir

        self.dataset = HPADataset(self.data_dir,
                                img_size=img_size,
                                is_training=training,
                                in_channels=in_channels,
                                transform=trsfm,
                                aug_version=aug_version)
        super().__init__(self.dataset,
                         batch_size,
                         shuffle,
                         validation_split,
                         num_workers)
