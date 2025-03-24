import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.emotion_cnn import EmotionCNN  # Ensure this is your model class

# Define the class mapping for each emotion
class_mapping = {
    'angry': 0,
    'fear': 1,
    'happy': 2,
    'sad': 3,
    'neutral': 4
}

# Image transformation with more augmentation
transform_train = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transform_test = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the FER2013 dataset
train_data_path = "D:/Facial_Emotion_Detection/data/FER2013/train"
test_data_path = "D:/Facial_Emotion_Detection/data/FER2013/test"

train_dataset = datasets.ImageFolder(train_data_path, transform=transform_train)
test_dataset = datasets.ImageFolder(test_data_path, transform=transform_test)

# DataLoader with adjusted batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Device Configuration
device = torch.device("cpu")  # Using CPU for training

# Initialize model, loss function, and optimizer
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Changed to AdamW with weight decay

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=50,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos'
)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train():
    num_epochs = 50
    best_accuracy = 0.0
    patience = 15  # Increased patience
    patience_counter = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print batch progress
            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        train_accuracy = 100 * correct / total
        epoch_time = time.time() - start_time

        # Evaluate on test set
        test_accuracy = evaluate(model, test_loader, device)
        
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, "
              f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")

        # Save the model if test accuracy improves
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"New best accuracy: {best_accuracy:.2f}%")
            print("Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
            }, "model/emotion_cnn_best.pth")
            print("Model saved!")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("Training Completed!")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    train()
