import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.resnet import ResNet, BasicBlock
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LightsDataset(Dataset):
    def __init__(self, csv_file, root_dir, data_transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = data_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Definicja ResNet10
class ResNet10(ResNet):
    def __init__(self, num_classes=5):
        super().__init__(block=BasicBlock, layers=[1,1,1,1], num_classes=num_classes)

# Klasa modelu
class LightLevelClassifier(nn.Module):
    def __init__(self, n_classes=5, dropout_prob=0.5):
        super(LightLevelClassifier, self).__init__()
        self.resnet = ResNet10(num_classes=n_classes)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.resnet.fc(x)
        return x

# Klasa do zwiększenia zbioru poprzez dodanie obróconych wersji każdego obrazu
class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, angles):
        self.original_dataset = original_dataset
        self.angles = angles
        self.augmented_mapping = []

        # Dla każdego indeksu w oryginalnym zbiorze, tworzymy 1 oryginał + kopie obrócone
        for i in range(len(original_dataset)):
            # Oryginał
            self.augmented_mapping.append((i, None))
            # Obrócone wersje
            for angle in self.angles:
                self.augmented_mapping.append((i, angle))

    def __len__(self):
        return len(self.augmented_mapping)

    def __getitem__(self, idx):
        orig_idx, angle = self.augmented_mapping[idx]
        image, label = self.original_dataset[orig_idx]
        if angle is not None:
            # Obracamy obraz o zadany kąt
            image = transforms.functional.rotate(image, angle)
        return image, label

# Hiperparametry
num_classes = 5
learning_rate = 0.001
batch_size = 32
num_epochs = 20
weight_decay = 1e-4  # L2 regularization

# Transformacje bez losowej rotacji - tylko normalizacja i zmiana rozmiaru
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Wczytanie danych
dataset = LightsDataset(
    csv_file="output_labels.csv",
    root_dir="light",
    data_transform=base_transform
)

# Uzyskanie etykiet i indeksów
labels = dataset.annotations.iloc[:, 1].values
indices = np.arange(len(dataset))

# Stratyfikowany podział danych na treningowy (70%) i tymczasowy (30%)
train_indices, temp_indices, train_labels, temp_labels = train_test_split(
    indices,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

# Stratyfikowany podział tymczasowego zbioru na walidacyjny (15%) i testowy (15%)
val_indices, test_indices, val_labels, test_labels = train_test_split(
    temp_indices,
    temp_labels,
    test_size=0.5,  # 50% z 30% to 15%
    random_state=42,
    stratify=temp_labels
)

# Tworzenie podzbiorów
train_set = Subset(dataset, train_indices)
val_set = Subset(dataset, val_indices)
test_set = Subset(dataset, test_indices)

# Definiujemy kąty obrotów
rotation_angles = [-40, -30, -20, -10, 10, 20, 30, 40]

# Tworzymy powiększony zbiór treningowy z obrotami
augmented_train_set = AugmentedDataset(train_set, angles=rotation_angles)

# Loadery
train_loader = DataLoader(dataset=augmented_train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Inicjalizacja modelu
model = LightLevelClassifier(n_classes=num_classes, dropout_prob=0.5).to(device)

# Funkcja straty
criterion = nn.CrossEntropyLoss()

# Optymalizator z L2 regularizacją
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Trenowanie
train_losses, val_losses = [], []
for epoch in range(num_epochs):
    # Faza treningu
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Faza walidacji
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1} Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

# Wizualizacja strat
plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
