import torch
import torch.nn as nn
from torchvision import models
from sklearn import preprocessing
import numpy as np
import yaml

from dataset import ImagesDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

label_convertor = {
    'n02089973': 'English foxhound',
    'n02088364': 'Beagle',
    'n02096294': 'Australian terrier',
    'n02115641': 'Dingo',
    'n02086240': 'Shih-Tzu',
    'n02099601': 'Golden retriever',
    'n02087394': 'Rhodesian ridgeback',
    'n02093754': 'Border terrier',
    'n02111889': 'Samoyed',
    'n02105641': 'Old English sheepdog'
}


class DogClassifierModel:
    def __init__(self, n_classes):
        print("Loading model...")
        self.model = models.efficientnet_b4(pretrained=False)
        self.model.classifier = nn.Sequential(
                                             nn.Dropout(p=0.4, inplace=True),
                                             nn.Linear(in_features=1792, out_features=n_classes, bias=True)
                                             )
        checkpoint = torch.load(config['model_name'], map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.classes_ = np.load(config['encoder_labels'], allow_pickle=True)
        print("Model is ready!")

    def predict(self, image_path):
        self.model.eval()
        img = ImagesDataset.load_sample(image_path)
        img = ImagesDataset.transform(img, config['img_size']).unsqueeze(0)
        img = img.to(DEVICE)

        res = self.model(img)
        y_hat = res.softmax(dim=1).argmax(dim=1)
        proba = torch.max(res.softmax(dim=1))
        label = self.label_encoder.inverse_transform([y_hat])[0].split('_')
        breed = label_convertor[label[0]]
        return breed, proba.item()

