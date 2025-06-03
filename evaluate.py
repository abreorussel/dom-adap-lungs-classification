import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ChestXrayDataset
from model import ChestXrayCNN
from sklearn.metrics import classification_report, confusion_matrix

# Load the pre-trained task model
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
task_model = ChestXrayCNN(imagesize=256).to(device)
task_model.load_state_dict(torch.load('chest_xray_mendeley_5_95.pth'))
task_model.eval()

# Define the new dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor()
])

new_dataset = ChestXrayDataset(
    # normal_dir='/home/russel/rnd2/lung_classifier/chestxray/Datasets/Kaggle/nontb', 
    # tb_dir='/home/russel/rnd2/lung_classifier/chestxray/Datasets/Kaggle/tb', 
    # normal_dir = '/home/russel/rnd2/lung_classifier/chestxray/Datasets/mendeley/nontb' ,
    # tb_dir = '/home/russel/rnd2/lung_classifier/chestxray/Datasets/mendeley/tb' ,
    normal_dir = '/home/russel/rnd2/lung_classifier/chestxray/Datasets/tbx11k-simplified/nontb' ,
    tb_dir = '/home/russel/rnd2/lung_classifier/chestxray/Datasets/tbx11k-simplified/tb' ,
    transform=transform, 
    img_size=256
)
new_dataloader = DataLoader(new_dataset, batch_size=1, shuffle=False)

true_labels = []
pred_labels = []

# Evaluate the task model on the new dataset
for idx, (images, labels) in enumerate(tqdm(new_dataloader, desc="Evaluating")):
    images, labels = images.to(device), labels.to(device, dtype=torch.float)
    
    with torch.no_grad():  # Ensure no gradients are calculated
        outputs, side_outputs = task_model(images)
    
    outputs = outputs.squeeze(dim=1)  # Squeeze the output if necessary
    preds = (outputs > 0.5).float()  # Apply the threshold to get binary predictions
    
    true_labels.extend(labels.cpu().numpy())
    pred_labels.extend(preds.cpu().numpy())

# Print evaluation metrics
print('Classification Report:')
print(classification_report(true_labels, pred_labels))
print('Confusion Matrix:')
print(confusion_matrix(true_labels, pred_labels))
