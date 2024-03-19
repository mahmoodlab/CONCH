from PIL import Image
import os
import torch
from torch.utils.data import Dataset

class TileClassificationDataset(Dataset):
    def __init__(self, 
                 df, 
                 data_source = None, 
                 img_transforms = None,
                 index_col = 'image_name',
                 subdir_col = None,
                 target_col = 'label', 
                 target_transforms = None,
                 label_map = None,
                 dummy_size = 0):

        self.label_map = label_map
        self.data_source = data_source
        self.index_col = index_col
        self.target_col = target_col
        self.subdir_col = subdir_col
        self.img_transforms = img_transforms
        self.target_transforms = target_transforms
        self.data = df
        self.dummy_size = dummy_size

    def __len__(self):
        return len(self.data)

    def get_ids(self, ids):
        return self.data.loc[ids, self.index_col]

    def get_labels(self, ids):
        return self.data.loc[ids, self.target_col]

    def __getitem__(self, idx):
        img_name = self.get_ids(idx)
        label = self.get_labels(idx)

        if self.label_map is not None:
            label = self.label_map[label]
        if self.target_transforms is not None:
            label = self.target_transforms(label)

        if self.dummy_size > 0:
            img = torch.rand(3, self.dummy_size, self.dummy_size)
        else:
            if self.data_source is not None:
                if self.subdir_col is not None:
                    subdir = self.data.loc[idx, self.subdir_col]
                    if not isinstance(subdir, str):
                        subdir = ""
                    img_path = os.path.join(self.data_source, subdir, img_name)
                else:
                    img_path = os.path.join(self.data_source, img_name)
            else:
                img_path = img_name
            img = Image.open(img_path).convert('RGB')
            if self.img_transforms is not None:
                img = self.img_transforms(img)

        return {'img': img,  'label': label}