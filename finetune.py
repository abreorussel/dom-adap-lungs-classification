import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from dataset import ChestXrayDataset, BalancedBatchSampler
from model import ChestXrayCNN
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Function to load data
def get_dataloaders(normal_dir, tb_dir, batch_size=32, img_size=256, test_size=0.99, seed=0):
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = ChestXrayDataset(normal_dir, tb_dir, transform=transform, img_size=img_size)

    dataset_indices = list(range(len(dataset)))

    # Train-test split
    train_indices, test_indices = train_test_split(dataset_indices, test_size=test_size, stratify=dataset.labels, random_state=seed)
    
    print(f"Total training samples: {len(train_indices)}")

    # Subset the dataset for train and test
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Now the indices are relative to the subset, adjust accordingly:
    train_normal_indices = [i for i in range(len(train_dataset)) if train_dataset.dataset.labels[train_indices[i]] == 0]
    train_tb_indices = [i for i in range(len(train_dataset)) if train_dataset.dataset.labels[train_indices[i]] == 1]

    print(f"Normal samples in training set: {len(train_normal_indices)}")
    print(f"TB samples in training set: {len(train_tb_indices)}")

    # Create balanced batch sampler for training
    train_sampler = BalancedBatchSampler(train_normal_indices, train_tb_indices, batch_size)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*60, shuffle=False, num_workers=8)

    print(f"Total batches in training loader: {len(train_loader)}")

    return train_loader, test_loader

# Function to load data
def get_dataloaders_num(normal_dir, tb_dir, num_normal, num_tb, batch_size=32, img_size=256, seed=0):
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load the dataset
    dataset = ChestXrayDataset(normal_dir, tb_dir, transform=transform, img_size=img_size)
    
    # Set the seed 
    set_seed(seed)
    
    # Get indices for normal and TB images
    normal_indices = [i for i in range(len(dataset)) if dataset.labels[i] == 0]
    tb_indices = [i for i in range(len(dataset)) if dataset.labels[i] == 1]
    
    # Ensure we don't request more images than are available
    num_normal = min(len(normal_indices), num_normal)
    num_tb = min(len(tb_indices), num_tb)
    
    # Randomly select the required number of images for training
    train_normal_indices = random.sample(normal_indices, num_normal)
    train_tb_indices = random.sample(tb_indices, num_tb)
    
    # Remaining images will be used for testing
    test_normal_indices = list(set(normal_indices) - set(train_normal_indices))
    test_tb_indices = list(set(tb_indices) - set(train_tb_indices))
    
    # Combine training and test indices
    train_indices = train_normal_indices + train_tb_indices
    test_indices = test_normal_indices + test_tb_indices
    
    # Shuffle the indices
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    
    # Subset the dataset for train and test
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*60, shuffle=False, num_workers=8)
    
    print(f"Training: {len(train_normal_indices)} normal, {len(train_tb_indices)} TB images")
    print(f"Testing: {len(test_normal_indices)} normal, {len(test_tb_indices)} TB images")
    
    return train_loader, test_loader

# Function to get dataloaders for Active Learning with model output thresholding
def get_dataloaders_active_learning(normal_dir, tb_dir, model, lower_limit, upper_limit, batch_size=32, img_size=256):
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load the dataset
    dataset = ChestXrayDataset(normal_dir, tb_dir, transform=transform, img_size=img_size)

    # Set the model to evaluation mode to do forward pass
    model.eval()
    
    uncertain_indices = []
    confident_indices = []
    
    # DataLoader for full dataset without shuffling (for forward pass)
    full_loader = DataLoader(dataset, batch_size=batch_size*80, shuffle=False, num_workers=8)
    
    # Directory to save the plot
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Collect model outputs
    all_outputs = []

    # Perform forward pass and collect outputs
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(full_loader, desc="Identify uncertain samples")):
            images = images.to(device)
            outputs, _ = model(images)
            outputs = outputs.squeeze(dim=1).cpu().numpy()
            all_outputs.extend(outputs)
            
            # Get global indices directly from DataLoader (safer)
            global_indices = range(i * batch_size * 80, i * batch_size * 80 + len(images))
            for idx, output in zip(global_indices, outputs):
                if lower_limit <= output <= upper_limit:
                    uncertain_indices.append(idx)
                else:
                    confident_indices.append(idx)

    # Plot the histogram of model outputs
    plt.hist(all_outputs, bins=100, alpha=0.75)
    plt.axvline(lower_limit, color='r', linestyle='dashed', linewidth=2, label=f'Lower Limit: {lower_limit}')
    plt.axvline(upper_limit, color='g', linestyle='dashed', linewidth=2, label=f'Upper Limit: {upper_limit}')
    plt.xlabel('Model output')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Outputs')
    plt.legend()
    plt.text(0.1, max(plt.gca().get_ylim()) * 0.9, f'Uncertain samples: {len(uncertain_indices)}', color='red')
    plt.text(0.1, max(plt.gca().get_ylim()) * 0.8, f'Confident samples: {len(confident_indices)}', color='green')

    # Save the plot to the results directory
    plot_path = os.path.join(results_dir, 'output_distribution.png')
    plt.savefig(plot_path)  # Save the figure
    plt.close()  # Close the figure to free up memory

    print(f"Plot saved at: {plot_path}")
    print(f"Uncertain samples (train set): {len(uncertain_indices)}")
    print(f"Confident samples (test set): {len(confident_indices)}")
    
    # Subset the dataset for uncertain (train) and confident (test) samples
    train_dataset = Subset(dataset, uncertain_indices)
    test_dataset = Subset(dataset, confident_indices)

    # Now the indices are relative to the subset, adjust accordingly:
    train_normal_indices = [i for i in range(len(train_dataset)) if train_dataset.dataset.labels[uncertain_indices[i]] == 0]
    train_tb_indices = [i for i in range(len(train_dataset)) if train_dataset.dataset.labels[uncertain_indices[i]] == 1]

    print(f"Normal samples in training set: {len(train_normal_indices)}")
    print(f"TB samples in training set: {len(train_tb_indices)}")

    # Create balanced batch sampler for training
    train_sampler = BalancedBatchSampler(train_normal_indices, train_tb_indices, batch_size)

    # Create DataLoaders for uncertain training data and confident test data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*80, shuffle=False, num_workers=8)

    print(f"Training set: {len(train_loader)}")
    print(f"Test set: {len(test_loader)}")

    return train_loader, test_loader


# Function to train model
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device, dtype=torch.float)
            
            optimizer.zero_grad()
            outputs, side_outputs = model(images)
            outputs = outputs.squeeze(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        scheduler.step(epoch_loss)  # Adjust the learning rate based on the epoch loss
        print(f"Loss: {epoch_loss:.4f}")


# Function to evaluate model and return metrics
def evaluate_model(model, dataloader):
    model.eval()
    model = model.to(device)
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
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    return accuracy, precision, recall, f1

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chest X-ray CNN for TB detection")
    parser.add_argument("--normal_dir", type=str, default=None, help="Directory with normal chest X-ray images")
    parser.add_argument("--tb_dir", type=str, default=None, help="Directory with TB chest X-ray images")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for resizing")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Device to use for training")

    args = parser.parse_args()

    normal_dir = args.normal_dir
    tb_dir = args.tb_dir
    batch_size = args.batch_size
    img_size = args.img_size
    num_epochs = args.epochs
    learning_rate = args.lr
    device = args.device

    # Metrics storage for different runs
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    if not os.path.exists("result"):
        os.makedirs("result")

    # Number of iteration
    it = 10
    for seed in range(it):
        print(f"Training with random seed: {seed}")

        # Set seed and get dataloaders for the current run
        train_loader, test_loader = get_dataloaders(normal_dir, tb_dir, batch_size, img_size, seed=seed)
        
        # Load the model
        task_model = ChestXrayCNN(img_size) 
        task_model.load_state_dict(torch.load('chest_xray_kaggle_80_20.pth'))
        task_model = task_model.to(device,non_blocking=True)

        # train_loader, test_loader = get_dataloaders_active_learning(normal_dir, tb_dir, task_model, 0.45, 0.55, batch_size, img_size)

        # Freeze all convolutional layers
        for name, param in task_model.named_parameters():
            if 'conv' in name:
                param.requires_grad = False

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, task_model.parameters()), lr=learning_rate)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-5, verbose=True)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-5)

        # Train the model
        train_model(task_model, train_loader, criterion, optimizer, scheduler, num_epochs)

        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(task_model, test_loader)
        print(f'Accuracy:{accuracy:.4f}, Precision : {precision:.4f}, Recall: {recall:.4f}, f1-score: {f1:.4f}')

        # Append the metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Plotting box plots
    metrics_data = [accuracies, precisions, recalls, f1_scores]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    
    metrics_df = pd.DataFrame({
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    metrics_df.to_csv(os.path.join('result', 'metrics.csv'), index=False)
    
    # Mean values
    print(f"Accuracy: {np.mean(accuracies):.4f}\nPrecision: {np.mean(precisions):.4f}\nRecall: {np.mean(recalls):.4f}\nF1-Score: {np.mean(f1_scores):.4f}")

    plt.figure(figsize=(10, 6))
    plt.boxplot(metrics_data, tick_labels=labels)
    plt.title(f'Box Plot of Model Performance Metrics Over {it} Runs on different samples')
    plt.ylabel('Scores')
    plt.grid(True)
    
    # Save the plot as a PNG in the results folder
    plt.savefig(os.path.join('result', 'performance_metrics_box_plot.png'))
    plt.show()


    '''

        python finetune.py --tb_dir "/home/russel/rnd2/lung_classifier/chestxray/Datasets/Kaggle/tb" 
        --normal_dir "/home/russel/rnd2/lung_classifier/chestxray/Datasets/Kaggle/nontb"

    '''
