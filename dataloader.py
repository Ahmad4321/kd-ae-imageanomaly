
import networkx as nx
import numpy as np
import os
import re

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from config import *

# 从当前目录下读取文件
def read_graphfile(dataname:str, datadir='dataset'):
    prefix = os.path.join(datadir, dataname, dataname)

    # 注意：整张图编号从1开始。节点编号从0开始。

    # 图编号。第i行数值为x，即表示第i个节点属于编号为x的图
    # The value of the ith row is x, which means that the ith node belongs to the graph numbered x
    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_cnt = set()
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=0
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            graph_cnt.add(int(line))
            i+=1
    
    # 图标签。第i行表示第i张图的标签。
    # Graph Labels. Row i denotes the label of the ith graph.
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    # print(graph_labels)

    # 读取边数据。两个字典分别用来存储每张图包含的边以及节点编号。
    # Read edge data. Use two dictionaries to store the edges and the node IDs contained in each graph respectively.
    filename_adj=prefix + '_A.txt'
    edge_graph={i:[[],[]] for i in range(1,len(graph_cnt)+1)}
    index_graph={i:[] for i in range(1,len(graph_cnt)+1)}


    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            e0,e1 = (e0-1,e1-1)
            (edge_graph[graph_indic[e0]])[0].append(e1)
            (edge_graph[graph_indic[e0]])[1].append(e0)
            # edge_graph[graph_indic[e0]][0].append(e0)
            # edge_graph[graph_indic[e0]][1].append(e1)

            index_graph[graph_indic[e0]]+=[e0,e1]
            # 在graph_indic字典中找到节点e0对应的图编号。从而该条边数据与对应的节点数据放入对应的图中。
            # Find the graph number corresponding to node e0 in the graph_indic dictionary. 
            # Then place this edge data and the corresponding node data into the corresponding graph.

        # 使节点编号从0开始并去重。
        # Start the node numbering from 0.
        for k in index_graph.keys():
            index_graph[k]=[u-1 for u in set(index_graph[k])]
    
    # 节点标签，第i行表示第i个节点。具体含义需要参考数据集说明。
    # Node labels. Row i denotes the ith node.
    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels.append(int(line))
    except IOError:
        print('No node labels')

    # 节点属性，第i行表示第i个节点。将数据转换为numpy数组进行存储。
    # Node attributes. row i represents the ith node. The data is converted into numpy arrays.
    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\n")
                attrs = [float(attr) for attr in re.split(r"[,\s]+", line) if not attr == '']
                node_attrs.append(attrs)
    except IOError:
        print('No node attributes')
       
    graph_list = []
    for i in range(1,1+len(edge_graph)):
        x = torch.tensor([node_attrs[a] for a in index_graph[i]],dtype=torch.float)
        edge_index = torch.tensor(edge_graph[i],dtype=torch.long)
        y = torch.tensor([node_labels[a] for a in index_graph[i]],dtype=torch.long)
        
        graph = Data(x=x,edge_index=edge_index,y=y)
        graph_list.append(graph)

    return graph_list


def Dataloader(dataname:str, datadir='dataset'):
    graph_list = read_graphfile(dataname, datadir)
    loader = DataLoader(graph_list, batch_size=32)
    return loader

# print(read_graphfile('test'))
# print(DataLoader('test'))


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
    train_dataset = MVTecDataset('./data/capsule/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def testing_data_mvtec():
    # Load the dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = MVTecDataset('./data/capsule/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    return test_loader

