import os
import numpy as np
import torch

from torch import nn
from torch.autograd import Variable

import pylab
import matplotlib.pyplot as plt

from config import *
from dataloader import *
from model import *

AE = Autoencoder().to(device)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(AE.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)


train_loader = training_data_mvtec()

losses = np.zeros(num_epochs_teacher)

for epoch in range(num_epochs_teacher):
    for img,_ in train_loader:

        Xi = img.to(device)

        xhat = AE(Xi).to(device)
        loss = mse_loss(xhat, Xi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss: {:.4f}'.format(
        epoch + 1,
        num_epochs_teacher,
        loss))

torch.save({
    'epoch': num_epochs_teacher,
    'teacher_state_dict': AE.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': learning_rate,
}, CKPT_teacher)