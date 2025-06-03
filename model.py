import torch
import torch.nn as nn
import torch.nn.functional as F

class ChestXrayCNN(nn.Module):
    def __init__(self, imagesize):
        super(ChestXrayCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        fc_input_size = 64 * (imagesize // 8) * (imagesize // 8)
        self.fc1 = nn.Linear(fc_input_size, 64)
        
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        side_outputs = []
        side_outputs.append(x)
        x = self.pool1(F.relu(self.conv1(x)))
        side_outputs.append(x)
        x = self.pool2(F.relu(self.conv2(x)))
        side_outputs.append(x)
        x = self.pool3(F.relu(self.conv3(x)))
        side_outputs.append(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x, side_outputs
    

class Autoencoder(nn.Module):
    def __init__(self, in_channels):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.Sigmoid()  # Use Sigmoid to output values between 0 and 1 if input images are normalized in this range
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
