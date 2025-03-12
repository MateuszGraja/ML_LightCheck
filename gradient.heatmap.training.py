import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import cm
import warnings
import gc

# Importy Captum
from captum.attr import IntegratedGradients, NoiseTunnel
from matplotlib.colors import LinearSegmentedColormap

# Ignorowanie ostrzeżeń
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Tworzenie folderów na wykresy i heatmaps, jeśli nie istnieją
os.makedirs("wykresy/heatmaps_attributions", exist_ok=True)
os.makedirs("modele", exist_ok=True)


class LightsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparametry
num_classes = 5
learning_rate = 3e-5
batch_size = 32
num_epochs = 40
model_name = "efficientnet_b0"

# Definicja custom colormap
custom_cmap = LinearSegmentedColormap.from_list('custom_black_white',
                                                 [(0, '#ffffff'),  # Białe dla niskich wartości
                                                  (0.25, '#000000'),  # Czarne dla średnich wartości
                                                  (1, '#000000')],  # Czarne dla wysokich wartości
                                                 N=256)

# Transformacje
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Wczytanie danych
dataset = LightsDataset(
    csv_file="output_labels.csv",
    root_dir="light",
    transform=transform,
)

# Uzyskanie etykiet i indeksów
labels = dataset.annotations.iloc[:, 1].values
indices = np.arange(len(dataset))

# Unikalne klasy i liczność każdej z nich
classes, class_counts = np.unique(labels, return_counts=True)

# Proporcje zbiorów
train_prop = 0.7
val_prop = 0.15
test_prop = 0.15

# Listy na indeksy
train_indices = []
val_indices = []
test_indices = []

# Ustawienie ziarna losowości dla powtarzalności
np.random.seed(42)

# Podział danych z zachowaniem proporcji klas
for cls in classes:
    # Indeksy próbek dla danej klasy
    cls_indices = indices[labels == cls]
    # Pomieszanie indeksów
    np.random.shuffle(cls_indices)
    # Obliczenie liczby próbek dla każdego zbioru
    n_total = len(cls_indices)
    n_train = int(train_prop * n_total)
    n_val = int(val_prop * n_total)
    n_test = n_total - n_train - n_val  # Reszta do zbioru testowego

    # Podział indeksów
    cls_train_indices = cls_indices[:n_train]
    cls_val_indices = cls_indices[n_train:n_train + n_val]
    cls_test_indices = cls_indices[n_train + n_val:]

    # Dodanie do ogólnych list indeksów
    train_indices.extend(cls_train_indices)
    val_indices.extend(cls_val_indices)
    test_indices.extend(cls_test_indices)

# Konwersja na tablice numpy i pomieszanie
train_indices = np.array(train_indices)
val_indices = np.array(val_indices)
test_indices = np.array(test_indices)

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)
np.random.shuffle(test_indices)

# Tworzenie podzbiorów
train_set = Subset(dataset, train_indices)
val_set = Subset(dataset, val_indices)
test_set = Subset(dataset, test_indices)

# Loadery
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Model
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = models.efficientnet_b0(weights=weights)

# Zamrożenie wszystkich warstw poza ostatnią
for param in model.parameters():
    param.requires_grad = False

# Odblokowanie warstwy docelowej dla Grad-CAM
for param in model.features[-1][0].parameters():
    param.requires_grad = True

# EfficientNet_B0 ma classifier = nn.Sequential(
#     nn.Dropout(p=0.2, inplace=True),
#     nn.Linear(in_features=1280, out_features=1000),
# )
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=num_classes)
)
model.to(device)

# Ładowanie wytrenowanego modelu, jeśli istnieje
pretrained_model_path = f"efficientnet_b0_epoka40.pth"
if os.path.exists(pretrained_model_path):
    state_dict = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Załadowano wytrenowany model z {pretrained_model_path}")
else:
    print(f"Nie znaleziono wytrenowanego modelu w {pretrained_model_path}, rozpoczynanie treningu.")

# Funkcja straty i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Dodanie scheduler'a ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.1, patience=5, verbose=True)

# Inicjalizacja list do przechowywania metryk
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
test_accuracies = []


# Funkcja do ewaluacji modelu na danym loaderze
def evaluate(loader, model, criterion):
    num_correct = 0
    num_samples = 0
    losses = []
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            loss = criterion(scores, y)
            losses.append(loss.item())

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

            all_preds.append(scores.cpu())
            all_labels.append(y.cpu())

    accuracy = float(num_correct) / float(num_samples) * 100
    avg_loss = sum(losses) / len(losses)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    model.train()
    return accuracy, avg_loss, all_preds, all_labels


# Grad-CAM Implementacja
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Rejestracja hooków i przechowywanie uchwytów
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
        print("Hooks registered.")

    def save_activation(self, module, input, output):
        self.activations = output.detach()
        print("Activation saved.")

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        print("Gradient saved.")

    def generate_heatmap(self, input_image, class_idx):
        # Forward pass
        output = self.model(input_image)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot.to(device), retain_graph=True)

        if self.gradients is None or self.activations is None:
            print("Gradients or activations are None.")
            return None

        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # [C, H, W]
        activations = self.activations[0].cpu().numpy()  # [C, H, W]

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # [C]

        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam

        return cam


# Inicjalizacja GradCAM (dla EfficientNet_B0, ostatnia warstwa konwolucyjna to features[-1][0])
grad_cam = GradCAM(model, model.features[-1][0])

# Reużywanie obiektów IntegratedGradients i NoiseTunnel
integrated_gradients = IntegratedGradients(model)
noise_tunnel = NoiseTunnel(integrated_gradients)


# Funkcja do generowania i zapisywania heatmap oraz atrybucji
def save_heatmap_and_attributions(image_path, cam, input_image, pred_class, output_dir, cls, original_idx, sample_num):
    try:
        # Oryginalny obraz
        img = cv2.imread(image_path)
        if img is None:
            print(f"Nie można wczytać obrazu z {image_path}.")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, 224, 224))

        # Heatmap
        heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cm.jet(heatmap)[:, :, :3] * 255  # Kolorowanie

        # Nakładanie heatmap na obraz
        superimposed_img = heatmap_color * 0.4 + img * 0.6
        superimposed_img = superimposed_img.astype(np.uint8)

        # Generowanie Gradient-Based Attribution
        attributions_ig = integrated_gradients.attribute(input_image, target=pred_class, n_steps=50)
        attributions_ig = attributions_ig.squeeze().cpu().detach().numpy()

        # Generowanie Noise Tunnel Attribution
        attributions_nt = noise_tunnel.attribute(input_image, nt_samples=5, nt_type='smoothgrad_sq', target=pred_class)
        attributions_nt = attributions_nt.squeeze().cpu().detach().numpy()

        # Normalizacja atrybucji
        attributions_ig = np.sum(np.abs(attributions_ig), axis=0)
        attributions_ig = (attributions_ig - np.min(attributions_ig)) / (np.max(attributions_ig) - np.min(attributions_ig) + 1e-8)

        attributions_nt = np.sum(np.abs(attributions_nt), axis=0)
        attributions_nt = (attributions_nt - np.min(attributions_nt)) / (np.max(attributions_nt) - np.min(attributions_nt) + 1e-8)

        # Zastosowanie custom colormap
        attributions_ig_cmap = custom_cmap(attributions_ig)
        attributions_nt_cmap = custom_cmap(attributions_nt)

        # Tworzenie figure z 4 subplots
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        # Oryginalny obraz
        axs[0].imshow(img)
        axs[0].axis('off')
        axs[0].set_title('Oryginalny Obraz')

        # Heatmapa Grad-CAM
        axs[1].imshow(superimposed_img)
        axs[1].axis('off')
        axs[1].set_title('Grad-CAM Heatmap')

        # Gradient-Based Attribution
        axs[2].imshow(attributions_ig_cmap)
        axs[2].axis('off')
        axs[2].set_title('Gradient-Based Attribution')

        # Noise Tunnel Attribution
        axs[3].imshow(attributions_nt_cmap)
        axs[3].axis('off')
        axs[3].set_title('Noise Tunnel Attribution')

        # Zapisanie figure z unikalną nazwą
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"heatmap_cls_{cls}_sample_{sample_num}_idx_{original_idx}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Zapisano porównanie w {output_path}")
    except Exception as e:
        print(f"Błąd podczas zapisywania atrybucji dla {image_path}: {e}")
    finally:
        # Usunięcie zmiennych i wyczyszczenie pamięci
        del img, heatmap, heatmap_color, superimposed_img
        del attributions_ig, attributions_nt, attributions_ig_cmap, attributions_nt_cmap
        torch.cuda.empty_cache()
        gc.collect()


# Funkcja do generowania Heatmaps oraz Atrybucji i zapisywania porównania
def generate_and_save_heatmaps_and_attributions(model, dataset, num_samples_per_class=10):
    model.eval()
    os.makedirs("wykresy/heatmaps_attributions", exist_ok=True)

    # Tworzymy słownik z listami indeksów dla każdej klasy
    class_indices = {cls: [] for cls in classes}

    # Przeglądamy cały zbiór testowy i przypisujemy indeksy do odpowiednich klas
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label.item()].append(idx)

    # Generujemy heatmapy i atrybucje dla wybranych próbek każdej klasy
    for cls in classes:
        cls_indices = class_indices[cls]
        if len(cls_indices) == 0:
            print(f"Brak próbek dla klasy {cls}.")
            continue
        selected_indices = np.random.choice(cls_indices,
                                           size=min(num_samples_per_class, len(cls_indices)),
                                           replace=False)
        for sample_num, idx in enumerate(selected_indices, 1):
            image, label = dataset[idx]
            input_image = image.unsqueeze(0).to(device)

            # Forward pass
            output = model(input_image)
            pred_class = output.argmax(dim=1).item()

            # Generate heatmap
            cam = grad_cam.generate_heatmap(input_image, pred_class)

            if cam is None:
                print(f"Nie udało się wygenerować heatmapy dla indeksu {idx}")
                continue

            # Get original image path
            original_idx = dataset.indices[idx]
            img_path = dataset.dataset.annotations.iloc[original_idx, 0]
            full_img_path = os.path.join(dataset.dataset.root_dir, img_path)

            # Define output directory for combined plots
            output_dir = "wykresy/heatmaps_attributions"
            os.makedirs(output_dir, exist_ok=True)

            # Save heatmap and attributions with unikalnymi nazwami
            save_heatmap_and_attributions(full_img_path, cam, input_image, pred_class, output_dir, cls, original_idx, sample_num)

            # Usunięcie zmiennych i wyczyszczenie pamięci
            del input_image, image, label, output, cam
            torch.cuda.empty_cache()
            gc.collect()


# Trenowanie sieci (jeśli nie załadowano wytrenowanego modelu)
if not os.path.exists(pretrained_model_path):
    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = sum(losses) / len(losses)
        train_losses.append(avg_train_loss)

        # Ewaluacja na zbiorze treningowym
        train_acc, _, _, _ = evaluate(train_loader, model, criterion)
        train_accuracies.append(train_acc)

        # Ewaluacja na zbiorze walidacyjnym
        val_acc, val_loss, _, _ = evaluate(val_loader, model, criterion)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        # Ewaluacja na zbiorze testowym
        test_acc, test_loss, test_preds, test_labels = evaluate(test_loader, model, criterion)
        test_accuracies.append(test_acc)

        # Krok scheduler'a na podstawie straty walidacyjnej
        scheduler.step(val_loss)

        # Zapisywanie modelu z unikalną nazwą
        torch.save(model.state_dict(), f"modele/{model_name}_epoka_{epoch + 1}.pth")

        # Wyświetlanie aktualnego learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoka {epoch + 1}/{num_epochs}, LR: {current_lr:.6f}, Strata trening: {avg_train_loss:.4f}, "
              f"Strata walidacja: {val_loss:.4f}, Dokł. trening: {train_acc:.2f}%, "
              f"Dokł. walidacja: {val_acc:.2f}%, Dokł. test: {test_acc:.2f}%")

    # Po treningu, zapisujemy model
    final_model_path = f"modele/{model_name}_epoka_{num_epochs}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model zapisany jako {final_model_path}")
else:
    print("Model został załadowany, pomijanie treningu.")

# Generowanie Heatmaps oraz Atrybucji
generate_and_save_heatmaps_and_attributions(model, test_set, num_samples_per_class=10)

# Końcowa ewaluacja na zbiorze testowym
print("Końcowa ewaluacja na zbiorze testowym:")
test_accuracy, _, test_preds, test_labels = evaluate(test_loader, model, criterion)
print(f"Dokładność na zbiorze testowym: {test_accuracy:.2f}%")

# Plotowanie strat treningowych i walidacyjnych
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Strata Trening')
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', label='Strata Walidacja')
plt.title('Strata vs. Epoki')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.grid(True)
plt.savefig(f"wykresy/strata-{num_epochs}epok-{model_name}.png")
# plt.show()

# Plotowanie dokładności
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='x', label='Dokładność Trening')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='x', label='Dokładność Walidacja')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='x', label='Dokładność Test')
plt.title('Dokładność vs. Epoki')
plt.xlabel('Epoka')
plt.ylabel('Dokładność (%)')
plt.legend()
plt.grid(True)
plt.savefig(f"wykresy/dokladnosc-{num_epochs}epok-{model_name}.png")
# plt.show()

# Przygotowanie danych do PR Curves
test_labels_np = test_labels.numpy()
test_preds_np = test_preds.numpy()

# Funkcja do obliczania Precision-Recall Curve
def compute_pr_curve(labels, scores):
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]

    tp = 0
    fp = 0
    fn = np.sum(sorted_labels)
    tn = len(sorted_labels) - fn

    precisions = []
    recalls = []

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    unique_recalls, indices = np.unique(recalls, return_index=True)
    unique_precisions = np.array(precisions)[indices]

    return unique_recalls, unique_precisions

# Wykres Precision-Recall dla każdej klasy
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    binary_labels = (test_labels_np == i).astype(int)
    scores = test_preds_np[:, i]

    recalls, precisions = compute_pr_curve(binary_labels, scores)

    ap = np.trapz(precisions, recalls)

    plt.plot(recalls, precisions, label=f'Klasa {i} (AP={ap:.2f})')

plt.xlabel('Czułość (Recall)')
plt.ylabel('Precyzja (Precision)')
plt.title('Precision-Recall Curves dla każdej klasy')
plt.legend()
plt.grid(True)
plt.savefig(f"wykresy/precision-recall-{num_epochs}epok-{model_name}.png")
# plt.show()

# Wykres strat i dokładności w czasie trenowania
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epoka')
ax1.set_ylabel('Strata', color=color)
ax1.plot(range(1, len(train_losses) + 1), train_losses, marker='o', color=color, label='Strata Trening')
ax1.plot(range(1, len(val_losses) + 1), val_losses, marker='o', color='tab:orange', label='Strata Walidacja')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Dokładność (%)', color=color)
ax2.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='x', color=color, label='Dokładność Trening')
ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='s', color='tab:green', label='Dokładność Walidacja')
ax2.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='^', color='tab:purple', label='Dokładność Test')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title('Strata i Dokładność w czasie Trenowania')
plt.savefig(f"wykresy/strata_i_dokladnosc-{num_epochs}epok-{model_name}.png")
plt.show()
