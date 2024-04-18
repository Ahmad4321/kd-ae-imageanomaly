import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
import cv2


# Define the encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


# Define the decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define a custom dataset class
class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = plt.imread(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 50
lambda_param = 0.5

# Load the dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = MVTecDataset(root_dir='./capsule/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
teacher_model = Autoencoder().to(device)
student_model = Autoencoder().to(device)

mse = nn.MSELoss()
cos = nn.CosineSimilarity()


optimizerT = torch.optim.Adam(teacher_model.parameters(), lr=learning_rate) # Vk in my case
optimizerS = torch.optim.Adam(student_model.encoder.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    teacher_running_loss = 0.0
    Student_running_loss = 0.0
    for data in train_loader:
        img = data.to(device)
        teacher_recon = teacher_model(img)
        zs = student_model.encoder(img)


        optimizerT.zero_grad()
        loss = mse(teacher_recon, img)
        loss.backward()
        optimizerT.step()

        # X_ti = teacher_model.decoder(z_t)


        zt = teacher_model.encoder(img).detach()

        lossS = (lambda_param * mse(zs, zt)) + ((1 - lambda_param) * (1 - cos(zs, zt)))
        studentloss = lossS.sum()
        optimizerS.zero_grad()
        studentloss.backward()
        optimizerS.step()

        # zs = encodedzs.detach()

        teacher_running_loss += loss.item()
        Student_running_loss += studentloss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {teacher_running_loss/ len(train_loader):.4f} , SLoss: {Student_running_loss / len(train_loader):.4f}")



teacher_model.eval()

# Evaluate on test images
test_dataset = MVTecDataset(root_dir='./capsule/test/mixture', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

anomaly_scores = []
for data in test_loader:
    img = data.to(device)
    zt = teacher_model.encoder(img)
    zs = student_model.encoder(img)
    # loss = mse(recon, img)


    score = (0.3 * mse(zs,zt) + 0.3 * (1 - cos(zs,zt)) + 0.4 * mse(teacher_model.decoder(zs),img))
    scoremean = torch.mean(score)
    anomaly_scores.append(scoremean.item())


print(anomaly_scores)

mask_dir = './capsule/ground_truth/mixture'
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


        # Print or store the label along with the corresponding file name
        # print(f"Image: {mask_file}, Label: {label}")
true_labels_dataset = detectTrueLabel(mask_dir)
print(true_labels_dataset)
# Perform anomaly detection evaluation

print(anomaly_scores)
anomaly_threshold = 0.5  # Set an anomaly threshold
predicted_labels = [1 if score > anomaly_threshold else 0 for score in anomaly_scores]
true_labels = [1,0]


accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels,zero_division=1)
recall = recall_score(true_labels, predicted_labels)

print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Visualize anomaly scores
plt.plot(anomaly_scores, marker='o')
plt.title('Anomaly Scores')
plt.xlabel('Image Index')
plt.ylabel('Anomaly Score')
plt.grid(True)
plt.show()



plt.savefig('./capsule/foo1.png')
