
import networkx as nx
import numpy as np
import re
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import cv2
import matplotlib.pyplot as plt

from config import *

class MVTecDataset(Dataset): 
    def __init__(self,data_dir, transform=None):
        self.label_name = data_label
        self.transform = transform
        self.data_info = self.get_img_info(data_dir)

    def __len__(self):
        return len(self.data_info)
       
    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        image = plt.imread(path_img)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    # 从目录中获取图片及标签
    # Retrieve images and labels from the directory.
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            # Iterate through all the labels in the directory.
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))

                # 遍历图片
                # Iterate through all the images in the directory of labels.
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = data_label[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info


def detectTrueLabel(mask_dir):
    true_labels = []
    # Iterate over each mask file
    for mask_file in os.listdir(mask_dir):
        # Load mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Threshold mask to convert to binary
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

        # Determine label based on the presence of anomalies
        label = 1 if np.max(binary_mask) == 255 else 0

        true_labels.append(label)

    return true_labels

def training_data_mvtec():
    # Load the dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = MVTecDataset(training_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def testing_data_mvtec():
    # Load the dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = MVTecDataset(testing_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    return test_loader

