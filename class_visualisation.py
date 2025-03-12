import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dataset
class LightsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
batch_size = 32

# Transformacje
transform = transforms.Compose([
    transforms.Resize((700, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
dataset = LightsDataset(csv_file="output_labels.csv", root_dir="light", transform=transform)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

# Model (bez końcowej warstwy klasyfikacyjnej)
model = models.densenet121(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Usuwamy ostatnią warstwę klasyfikacyjną
model.to(device)
model.eval()

# Wyciąganie cech i etykiet
features = []
labels = []
with torch.no_grad():
    for data, target in data_loader:
        data = data.to(device)
        output = model(data)
        features.append(output.view(output.size(0), -1).cpu())  # Wyjście modelu ResNet50 to tensor 2048-elementowy
        labels.extend(target.numpy())

features = torch.cat(features, dim=0)
labels = torch.tensor(labels)

# Implementacja PCA w PyTorch
def pca_torch(X, k=3):
    # Normalizacja danych
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean

    # Obliczanie macierzy kowariancji
    covariance_matrix = torch.mm(X_centered.T, X_centered) / (X_centered.size(0) - 1)

    # Obliczanie wartości własnych i wektorów własnych
    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
    eigenvectors = eigenvectors[:, :k].real  # Wybieramy tylko k pierwszych wektorów własnych i bierzemy część rzeczywistą

    # Transformacja danych na nowe osie
    reduced_data = torch.mm(X_centered, eigenvectors)
    return reduced_data

# Redukcja wymiarów do 3D za pomocą PCA w PyTorch
reduced_features = pca_torch(features, k=3).numpy()

# Wizualizacja 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=labels, cmap="tab10", alpha=0.7, edgecolor="k")

ax.set_xlabel("Wymiar 1")
ax.set_ylabel("Wymiar 2")
ax.set_zlabel("Wymiar 3")
plt.title("Wizualizacja cech w 3D")
plt.colorbar(scatter, label="Klasa")
plt.show()
