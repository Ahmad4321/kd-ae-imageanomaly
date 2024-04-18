from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image
import os

from config import *

normMean = [0.6979763, 0.66827846, 0.6554924]
normStd = [0.23617877, 0.25674772, 0.25802118]

# Mlist数据集
# 从Mlist中调取特定数据
class Mnisttox(Dataset):
    def __init__(self, datasets ,labels:list):
        self.dataset = [datasets[i][0] for i in range(len(datasets))
                        if datasets[i][1] in labels ]
        self.labels = labels
        self.len_oneclass = int(len(self.dataset)/10)

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, index):
        img = self.dataset[index]
        return img,[]

def Train_DataLoader_Mnist(numbers:list):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))  # [0,1] => [-1,1]
    ])
    dataset = MNIST('./data', download=True,train=True, transform=img_transform)
    filter = Mnisttox(dataset,numbers)
    loader = DataLoader(filter, batch_size=batch_size, shuffle=True)
    return loader

def Test_DataLoader_Mnist(numbers:list):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))  # [0,1] => [-1,1]
    ])
    dataset = MNIST('./data', download=True,train=False, transform=img_transform)
    filter = Mnisttox(dataset,numbers)
    loader = DataLoader(filter, batch_size=len(dataset), shuffle=True)
    return loader

# 异常检测-capsule数据集
class Capsule(Dataset):
    def __init__(self,data_dir, transform=None):
        self.label_name = capsule_label
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            # 在这里做transform，转为tensor等等
            img = self.transform(img)

        return img, label
    
    #从目录中获取图片及标签
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = capsule_label[sub_dir]
                    data_info.append((path_img, int(label)))
        # 有了data_info，就可以返回上面的__getitem__()函数中的self.data_info[index]，
        # 根据index索取图片和标签
        return data_info
    
def Train_DataLoader_Capsule():
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)), #缩放数据
        transforms.ToTensor(), #转换成张量数据
        # transforms.Normalize(normMean, normStd), #数据标准化
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    dataset = Capsule(data_dir='./data/capsule/train', transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def Test_DataLoader_Capsule():
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)), #缩放数据
        transforms.ToTensor(), #转换成张量数据
        # transforms.Normalize(normMean, normStd), #数据标准化
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    dataset = Capsule(data_dir='./data/capsule/test', transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader