import os
import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torch
import random


class ChestXrayDataset(Dataset):
    def __init__(self, normal_dir, tb_dir, transform=None, img_size=256):
        self.normal_dir = normal_dir
        self.tb_dir = tb_dir
        self.transform = transform
        self.img_size = img_size

        if tb_dir and os.path.exists(tb_dir):
            self.tb_images = [os.path.join(tb_dir, img) for img in os.listdir(tb_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        else:
            self.tb_images = [] 

        if normal_dir and os.path.exists(normal_dir):
            self.normal_images = [os.path.join(normal_dir, img) for img in os.listdir(normal_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        else:
            self.normal_images = [] 
        
        self.all_images = self.normal_images + self.tb_images
        self.labels = [0] * len(self.normal_images) + [1] * len(self.tb_images)  # 0 for normal, 1 for TB

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, (self.img_size, self.img_size))
        image = Image.fromarray(image)  # Convert to PIL image
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


class BalancedBatchSampler(Sampler):
    def __init__(self, normal_indices, tb_indices, batch_size):
        self.normal_indices = normal_indices
        self.tb_indices = tb_indices
        self.batch_size = batch_size // 2  # half batch for each class
        self.normal_index_pointer = 0
        self.tb_index_pointer = 0
        self.num_batches = max(len(normal_indices),len(tb_indices)) // self.batch_size

    def __iter__(self):
        random.shuffle(self.normal_indices)
        random.shuffle(self.tb_indices)

        for _ in range(self.num_batches):
            normal_batch = self._get_next_indices(self.normal_indices, self.normal_index_pointer, self.batch_size)
            self.normal_index_pointer = (self.normal_index_pointer + self.batch_size) % len(self.normal_indices)
            
            tb_batch = self._get_next_indices(self.tb_indices, self.tb_index_pointer, self.batch_size)
            self.tb_index_pointer = (self.tb_index_pointer + self.batch_size) % len(self.tb_indices)
            
            batch = normal_batch + tb_batch
            random.shuffle(batch)
            for idx in batch:
                yield idx

    def _get_next_indices(self, indices, start_pointer, count):
        end_pointer = start_pointer + count
        if end_pointer <= len(indices):
            return indices[start_pointer:end_pointer]
        else:
            return indices[start_pointer:] + indices[:end_pointer % len(indices)]

    def __len__(self):
        return self.num_batches * self.batch_size * 2

