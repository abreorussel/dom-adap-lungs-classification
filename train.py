import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset import ChestXrayDataset, BalancedBatchSampler
from model import ChestXrayCNN, Autoencoder
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=1, min_lr=1e-5, verbose=1):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best = None
        self.num_bad_epochs = 0

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
        elif (self.mode == 'min' and metrics < self.best) or (self.mode == 'max' and metrics > self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                for param_group in self.optimizer.param_groups:
                    new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                    if param_group['lr'] > new_lr:
                        param_group['lr'] = new_lr
                        if self.verbose:
                            print(f'Reducing learning rate to {new_lr:.6f}.')
                self.num_bad_epochs = 0

def get_dataloaders(normal_dir, tb_dir, batch_size=32, img_size=256, test_size=0.8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust for single channel
    ])

    dataset = ChestXrayDataset(normal_dir, tb_dir, transform=transform, img_size=img_size)

    # Split indices into train and test sets
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, stratify=dataset.labels)

    # Subset the dataset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Separate normal and TB indices in the training set
    train_normal_indices = [i for i, idx in enumerate(train_indices) if dataset.labels[idx] == 0]
    train_tb_indices = [i for i, idx in enumerate(train_indices) if dataset.labels[idx] == 1]

    # Create balanced batch sampler for training
    train_sampler = BalancedBatchSampler(train_normal_indices, train_tb_indices, batch_size)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Print dataset details
    print(f'Total dataset size: {len(dataset)}')
    print(f'Training set size: {len(train_dataset)}')
    print(f'Test set size: {len(test_dataset)}')
    print(f'Number of normal images in training set: {len(train_normal_indices)}')
    print(f'Number of TB images in training set: {len(train_tb_indices)}')

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
        scheduler.step(epoch_loss)
        print(f"Loss: {epoch_loss:.4f}")

    # Save the model after training
    torch.save(model.state_dict(), 'chest_xray_tbx11_80_20.pth')
    print('Model saved as chest_xray_tbx11_80_20.pth')
    

def train_autoencoder(model, autoencoders, dataloader, device, num_epochs=20):
    criteria = [nn.MSELoss() for _ in autoencoders]
    optimizers = [torch.optim.Adam(ae.parameters(), lr=0.001) for ae in autoencoders]

    for autoencoder in autoencoders:
        autoencoder.to(device)

    for epoch in range(num_epochs):
        running_losses = [0.0] * len(autoencoders)
        for images, _ in dataloader:
            images = images.to(device)
            with torch.no_grad():
                _, side_outputs = model(images)

            for idx, autoencoder in enumerate(autoencoders):
                side_out = side_outputs[idx].detach()
                autoencoder.train()
                optimizers[idx].zero_grad()
                recon = autoencoder(side_out)
                loss = criteria[idx](recon, side_out)
                loss.backward()
                optimizers[idx].step()
                running_losses[idx] += loss.item() * images.size(0)

        epoch_losses = [running_loss / len(dataloader.dataset) for running_loss in running_losses]
        print(f"Epoch {epoch + 1}/{num_epochs}, Losses: {epoch_losses}")

    # Save the trained autoencoders
    for idx, autoencoder in enumerate(autoencoders):
        torch.save(autoencoder.state_dict(), f'./autoencoders/autoencoder_layer{idx}.pth')

def evaluate_model(model, dataloader):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device, dtype=torch.float)
            outputs, side_outputs = model(images)
            outputs = outputs.squeeze()
            preds = (outputs > 0.5).float()
            
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
    
    print('Classification Report:')
    print(classification_report(true_labels, pred_labels))
    print('Confusion Matrix:')
    print(confusion_matrix(true_labels, pred_labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chest X-ray CNN for TB detection")
    parser.add_argument("--normal_dir", type=str, required=True, help="Directory with normal chest X-ray images")
    parser.add_argument("--tb_dir", type=str, required=True, help="Directory with TB chest X-ray images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for resizing")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--test_size", type=float, default=0.8, help="Proportion of dataset to use for testing")

    args = parser.parse_args()

    normal_dir = args.normal_dir
    tb_dir = args.tb_dir
    batch_size = args.batch_size
    img_size = args.img_size
    num_epochs = args.epochs
    learning_rate = args.lr
    device = args.device
    test_size = args.test_size

    train_loader, test_loader = get_dataloaders(normal_dir, tb_dir, batch_size, img_size, test_size)

    task_model = ChestXrayCNN(img_size).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(task_model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1, min_lr=1e-5, verbose=1)

    # Train the model
    train_model(task_model, train_loader, criterion, optimizer, scheduler, num_epochs)

    # # Define and train autoencoders for each side output layer
    # layer_dims = [1, 16, 32, 64]  # Example dimensions of the task network layers
    # autoencoders = [Autoencoder(dim) for dim in layer_dims]

    # train_autoencoder(task_model, autoencoders, train_loader, device=device)

    # # Evaluate the model
    evaluate_model(task_model, test_loader)



'''
    python train.py --tb_dir "/home/russel/rnd2/lung_classifier/chestxray/Datasets/mendeley/tb/" --normal_dir "/home/russel/rnd2/lung_classifier/chestxray/Datasets/mendeley/nontb/" --test_size 0.20
'''
    


'''
python train.py --tb_dir "/home/russel/rnd2/lung_classifier/chestxray/Datasets/tbx11k_cropped/TB_cropped" 
            --normal_dir /home/russel/rnd2/lung_classifier/chestxray/Datasets/tbx11k_cropped/NON_TB_cropped 
            --test_size 0.8

python finetune.py --tb_dir "/home/russel/rnd2/lung_classifier/chestxray/Datasets/Kaggle/tb" 
    --normal_dir "/home/russel/rnd2/lung_classifier/chestxray/Datasets/Kaggle/nontb"


python finetune.py --tb_dir "/home/russel/rnd2/lung_classifier/chestxray/Datasets/Dataset of Tuberculosis Chest X-rays Images/TB Chest X-rays" 
    --normal_dir "/home/russel/rnd2/lung_classifier/chestxray/Datasets/Dataset of Tuberculosis Chest X-rays Images/Normal Chest X-rays"


'''


'''
    conda env : dom-new



'''
