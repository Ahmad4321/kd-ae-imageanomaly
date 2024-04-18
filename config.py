import torch.nn as nn
import torch

# Dataset Configure
# ===============
data_label = {
    'good':0,
    'bad':1,
    'crack':1,
    'squeeze':1,
    'poke':1,
    'faulty_imprint':1,
    'scratch':1,
}

# Train Configure
# ===============
training_data_path = '{Path to training dataset}'
testing_data_path = '{Path to test dataset}'

mse = nn.MSELoss()
cos = nn.CosineSimilarity()

# batch Size
batch_size = 64

# learning Rate
learning_rate = 4.0e-4

# Number of loops
num_epochs_teacher = 50
num_epochs_kd = 100

# hyper-Parameters
lambda_param = 0.5

# weight for optimizer
weight_decay=1e-5

# model
CKPT_PATH = "./checkpoint/model_KD.pt"
CKPT_teacher = "./checkpoint/model_teacher.pt"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Test Configure
# ===============
anomaly_threshold = 0.5