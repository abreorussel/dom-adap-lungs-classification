import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ChestXrayDataset
from model import ChestXrayCNN, Autoencoder
from adapmodel import ANet
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import argparse
import torchvision.utils as vutils
#from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(model_path, img_size, device):
    model = ChestXrayCNN(img_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_autoencoders(autoencoder_paths, device, layer_dims):
    if len(autoencoder_paths) != len(layer_dims):
        raise ValueError("Number of autoencoder paths must match the number of layer dimensions")
    
    autoencoders = []
    for ae_path, dim in zip(autoencoder_paths, layer_dims):
        ae_model = Autoencoder(dim)  # Initialize Autoencoder with the correct dimension
        ae_model.load_state_dict(torch.load(ae_path))
        ae_model.to(device)
        ae_model.eval()
        autoencoders.append(ae_model)
    
    return autoencoders

def get_test_dataloader(normal_dir, tb_dir, batch_size=1, img_size=256):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = ChestXrayDataset(normal_dir, tb_dir, transform=transform, img_size=img_size)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader, len(dataset.normal_images), len(dataset.tb_images)

def evaluate_model(model, dataloader, device, AENet, AELoss, orthw, anet_epochs):
    model.eval()  
    for ae in AENet:
        ae.eval() 

    true_labels = []
    pred_labels = []
    anet = ANet().to(device)
    optimizer_ANet = optim.Adam(anet.parameters(), lr=0.01)

    i = 0
    total_loss = 0
    num_images = 0
   
    # output_img_dir = "output_img_Indian"
    # os.makedirs(output_img_dir, exist_ok=True)  # Ensure the output directory exists

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Optimize ANet for each test image
        anet.reset(device)  # Reset ANet to its initial state
        anet_loss_list = anet.opt_ANet(images, model, AENet, optimizer_ANet, AELoss, orthw, epochs=anet_epochs)

        total_loss += anet_loss_list[0]
        
        num_images += len(images)  # Increment the count by the batch size

        # Forward pass through task model and adaptors
        with torch.no_grad():
            outputs, side_out = anet(images, model)

            # input_img_path = os.path.join(output_img_dir, f"input_img_{i}.png")
            # adapted_img_path = os.path.join(output_img_dir, f"adapted_img_{i}.png")

            # if i % 100 == 0:
            #     # Save the input image
            #     vutils.save_image(images[0], input_img_path)
            #     # Save the adapted image
            #     vutils.save_image(side_out[0][0], adapted_img_path)

            i += 1
            preds = (outputs.squeeze(dim=1) > 0.5).float()
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    # Calculate and print the mean reconstruction loss over all images
    mean_loss = total_loss / num_images
    print(f"Mean reconstruction loss: {mean_loss}")

    print(classification_report(true_labels, pred_labels))
    print(confusion_matrix(true_labels, pred_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Chest X-ray Classification Model with Adaptors")
    parser.add_argument("--task_model_path", type=str, required=True, help="Path to the trained task model")
    parser.add_argument("--normal_dir", type=str,default=None, help="Directory with normal chest X-ray images")
    parser.add_argument("--tb_dir", type=str,default=None, help="Directory with TB chest X-ray images")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--anet_epochs", type=int, default=10, help="Number of epochs to update ANet for each image")

    args = parser.parse_args()

    set_seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task_model = load_model(args.task_model_path, args.img_size, device)
    test_loader, num_normal, num_tb = get_test_dataloader(args.normal_dir, args.tb_dir, args.batch_size, args.img_size)

    # Define the directory containing the autoencoders
    directory = '/home/tirthesh/chestxray/Code/autoencoders'

    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        exit(1)

    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter out only .pth files and sort them
    all_files = [f for f in all_files if f.endswith('.pth') and os.path.isfile(os.path.join(directory, f))]
    all_files.sort()  # Sort the list of .pth files

    # Construct the full paths
    autoencoder_paths = [os.path.join(directory, f) for f in all_files]

    # Define the dimensions of the task network layers
    layer_dims = [1, 16, 32, 64]

    # Load the autoencoders
    AENet = load_autoencoders(autoencoder_paths, device, layer_dims)
    AELoss = nn.MSELoss()

    evaluate_model(task_model, test_loader, device, AENet, AELoss, orthw=1, anet_epochs=args.anet_epochs)
