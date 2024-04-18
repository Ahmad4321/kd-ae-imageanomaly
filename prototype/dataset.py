from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import matplotlib.pyplot as plt

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