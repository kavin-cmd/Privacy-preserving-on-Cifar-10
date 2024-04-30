# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import numpy as np
from tqdm import tqdm
import time

# Parameters
EPSILON = 3.0
DELTA = 1e-5
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 256  # Adjusted for practical memory constraints and training stability
MAX_PHYSICAL_BATCH_SIZE = 256  # Adjusted to match BATCH_SIZE for simplicity
MAX_GRAD_NORM = 1.5  # Initial gradient norm
DATA_ROOT = './cifar10'

# CIFAR10 normalization values and transformations including advanced augmentations
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

# Datasets and DataLoaders
train_dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup with adjustments for CIFAR10
model = models.densenet121(pretrained=False, num_classes=10)  # Using DenseNet121
model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # Adjust input channels

# Validate and fix the model for differential privacy if necessary
errors = ModuleValidator.validate(model, strict=False)
if errors:  # If there are errors, fix them
    model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)

# Attach the privacy engine to the optimizer
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)

def adjust_privacy_parameters(epoch):
    """ Adjust privacy parameters dynamically based on the epoch """
    # Simple example: Linear decay of max_grad_norm
    initial_norm = 1.5
    final_norm = 0.5
    total_epochs = EPOCHS
    new_norm = initial_norm - (epoch / total_epochs) * (initial_norm - final_norm)
    privacy_engine.max_grad_norm = new_norm

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Optionally adjust privacy parameters at some interval or condition
        if batch_idx % 100 == 0:  # Example condition
            adjust_privacy_parameters(epoch)

    return train_loss / len(train_loader), 100. * correct / total

# Training loop with dynamic privacy adjustments
for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    end_time = time.time()
    print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Duration: {end_time - start_time:.2f}s')

# Function to evaluate the model on the test dataset
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Evaluate model performance
test_accuracy = test(model, test_loader, device)
