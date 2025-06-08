import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from dataset import ChestXrayDataset
from model import ChestXrayCNN
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_dataloaders(normal_dir, tb_dir, additional_dir, additional_dir_type, test_size, batch_size=32, img_size=256):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Dataset 1: Normal directory (used fully for training)
    normal_dataset = ChestXrayDataset(normal_dir, None, transform=transform, img_size=img_size)
    
    # Dataset 2: TB directory (used fully for training)
    tb_dataset = ChestXrayDataset(None, tb_dir, transform=transform, img_size=img_size)
    
    # Dataset 3: Additional directory (used for training and testing)
    if additional_dir_type == 'normal':
        additional_dataset = ChestXrayDataset(additional_dir, None, transform=transform, img_size=img_size)
    elif additional_dir_type == 'tb':
        additional_dataset = ChestXrayDataset(None, additional_dir, transform=transform, img_size=img_size)
    else:
        raise ValueError("additional_dir_type must be either 'normal' or 'tb'.")

    # Get dataset indices for the additional directory
    additional_indices = list(range(len(additional_dataset)))
    
    # Train-test split for the additional directory
    train_indices, test_indices = train_test_split(additional_indices, test_size=test_size, stratify=additional_dataset.labels, random_state=0)

    # Create subsets for training and testing from the additional dataset
    train_additional_dataset = Subset(additional_dataset, train_indices)
    test_additional_dataset = Subset(additional_dataset, test_indices)

    # Combine the normal and TB datasets fully for training
    train_datasets = [normal_dataset, tb_dataset, train_additional_dataset]

    # Concatenate the datasets
    train_dataset_combined = ConcatDataset(train_datasets)
    test_dataset_combined = test_additional_dataset

    # Print dataset details
    print(f'Total training set size: {len(train_dataset_combined)}')
    print(f'Total test set size: {len(test_dataset_combined)}')

    # Check if datasets are empty
    if len(train_dataset_combined) == 0:
        raise ValueError("Combined training dataset is empty.")
    if len(test_dataset_combined) == 0:
        raise ValueError("Testing dataset is empty.")

    # Create dataloaders
    train_loader = DataLoader(train_dataset_combined, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset_combined, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs, side_outputs = model(images)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        scheduler.step(epoch_loss)  # Adjust the learning rate based on the epoch loss
        print(f"Loss: {epoch_loss:.4f}")

    # Save the model after training
    torch.save(model.state_dict(), 'fineTune_chest_xray_cnn.pth')
    print('Model saved as fineTune_chest_xray_cnn.pth')

def evaluate_model(model, dataloader):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device, dtype=torch.float)
            outputs, side_outputs = model(images)
            outputs = outputs.squeeze(dim=1)
            preds = (outputs > 0.5).float()

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    print('Classification Report:')
    print(classification_report(true_labels, pred_labels))
    print('Confusion Matrix:')
    print(confusion_matrix(true_labels, pred_labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chest X-ray CNN for TB detection")
    parser.add_argument("--normal_dir", type=str, required=True, help="Directory with normal chest X-ray images (used fully for training)")
    parser.add_argument("--tb_dir", type=str, required=True, help="Directory with TB chest X-ray images (used fully for training)")
    parser.add_argument("--additional_dir", type=str, required=True, help="Directory with additional chest X-ray images (split between training and testing)")
    parser.add_argument("--additional_dir_type", type=str, required=True, choices=['normal', 'tb'], help="Specify if the additional directory contains 'normal' or 'tb' images")
    parser.add_argument("--test_size", type=float, required=True, help="Test size for splitting the additional directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for resizing")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Device to use for training")

    args = parser.parse_args()

    set_seed(0)

    normal_dir = args.normal_dir
    tb_dir = args.tb_dir
    additional_dir = args.additional_dir
    additional_dir_type = args.additional_dir_type
    test_size = args.test_size
    batch_size = args.batch_size
    img_size = args.img_size
    num_epochs = args.epochs
    learning_rate = args.lr
    device = args.device

    train_loader, test_loader = get_dataloaders(normal_dir, tb_dir, additional_dir, additional_dir_type, test_size, batch_size, img_size)

    # Load the model
    task_model = ChestXrayCNN(img_size)
    task_model.load_state_dict(torch.load('chest_xray_cnn.pth'))
    task_model = task_model.to(device)

    # Freeze all convolutional layers
    for name, param in task_model.named_parameters():
        if 'conv' in name:
            param.requires_grad = False

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, task_model.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-5, verbose=True)

    # Train the model
    train_model(task_model, train_loader, criterion, optimizer, scheduler, num_epochs)

    # Evaluate the model
    evaluate_model(task_model, test_loader)
