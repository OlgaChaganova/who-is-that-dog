import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from skimage import color, transform
import sklearn
import yaml


with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)


class ImagesDataset(Dataset):
    '''
    The class of the images dataset.
    '''
    def __init__(self, df: pd.DataFrame, labels_col: str, encoder: sklearn.preprocessing.LabelEncoder(), mode: str):
        '''
        :param df: pandas dataframe with images
        :mode: train / valid
        :param size: required size of images in the shape of (height, width), using in the resizing function
        '''
        assert mode in ['train', 'valid'], "Mode must be on of the followings: train, valid"
        assert labels_col in df.columns, f"labels_col must be one of the followings: {list(df.columns)}"

        super().__init__()
        self.mode = False if mode == 'train' else True
        self.df = df[df['is_valid'] == self.mode].reset_index()
        self.labels_col = labels_col
        self.len_ = len(self.df)
        self.encoder = encoder

    def __len__(self):
        return self.len_

    @staticmethod
    def load_sample(file):
        image = Image.open(file)
        image.load()
        return image

    @staticmethod
    def transform(img, size):
        img = np.array(img)
        if img.shape[-1] == 4:
            img = color.rgba2rgb(img)
        elif len(img.shape) == 2:
            img = color.gray2rgb(img)
        img = transform.resize(img, (size, size))
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32)
        return img

    def __getitem__(self, index):
        img = self.load_sample(config['PATH'] + self.df.loc[index, 'path'])
        label = self.df.loc[index, self.labels_col]
        label = self.encoder.transform(np.array([label]))
        img = self.transform(img, config['SIZE'])
        return img, label
