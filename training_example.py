import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np

from lasma_dataloader import LASMA_Augmentation

class OCTClassificationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.ones((256, 256), dtype=np.uint8) * 100
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:150, 100:150] = 255
        label = self.labels[idx]

        if self.transform:
            img_aug = self.transform(img, mask)
        else:
            img_aug = img

        img_tensor = torch.tensor(img_aug, dtype=torch.float32).unsqueeze(0)
        img_tensor = img_tensor.repeat(3, 1, 1) / 255.0

        return img_tensor, label

def train_oct_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    lasma_aug = LASMA_Augmentation(
        d_pixels=15, r1=0.5, r2=1.5, r3=3.0, epochs=15, device=device
    )

    dummy_imgs = ["path/to/img"] * 100
    dummy_masks = ["path/to/mask"] * 100
    dummy_labels = [np.random.randint(0, 2) for _ in range(100)]

    train_dataset = OCTClassificationDataset(dummy_imgs, dummy_masks, dummy_labels, transform=lasma_aug)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = models.resnet18(weights=None)
    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx == 1:
                break

        print(f"Epoch [{epoch+1}/{num_epochs}] done, Loss: {running_loss:.4f}")

if __name__ == "__main__":
    train_oct_classifier()
