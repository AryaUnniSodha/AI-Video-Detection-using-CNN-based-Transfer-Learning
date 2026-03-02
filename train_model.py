import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, random_split

# --------------------
# SETTINGS
# --------------------
DATA_DIR = "dataset"
BATCH_SIZE = 16
EPOCHS = 5
MODEL_PATH = "deepfake_model.pth"

device = torch.device("cpu")
print("Using device:", device)

# --------------------
# TRANSFORMS
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------
# DATASET
# --------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --------------------
# MODEL
# --------------------
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Freeze all layers first
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last 2 convolution blocks
for param in model.features[-2:].parameters():
    param.requires_grad = True

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, 2)

model = model.to(device)

# --------------------
# LOSS & OPTIMIZER
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)

# --------------------
# TRAINING LOOP
# --------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Training Accuracy: {train_acc:.2f}%")

    # --------------------
    # VALIDATION
    # --------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

# --------------------
# SAVE MODEL
# --------------------
torch.save(model.state_dict(), MODEL_PATH)
print("\nModel saved as", MODEL_PATH)