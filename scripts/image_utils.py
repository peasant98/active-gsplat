import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import os

DATASET_FOLDERS = ['bike_dataset', 'bonsai_dataset', 'counter_dataset', 'garden_dataset', 'stump_dataset', 'dataset', 'room_dataset']

# Define custom dataset for image pairs
class ImagePairFullDataset(Dataset):
    def __init__(self, csv_files, dataset_folders=DATASET_FOLDERS, transform=None):
        """
        full dataset with all pairs from diferent folders
        """
        pairs = None
        
        # for each folder, combine pairs
        for folder in dataset_folders:
            csv_file = os.path.join(folder, csv_files)
            pair_folder = pd.read_csv(csv_file)
            if pairs is None:
                pairs = pair_folder
            else:
                pairs = pd.concat([pairs, pair_folder])
        
        self.pairs = pairs
        self.dataset_folders = dataset_folders
        self.transform = transform

    def __len__(self):
        return len(self.pairs) * 2

    def __getitem__(self, idx):
        
        real_idx = idx // 2
        should_swap = idx % 2 == 1
        
        row = self.pairs.iloc[real_idx]
        image1_path = os.path.join(row["Image_1"])
        image2_path = os.path.join(row["Image_2"])
        label = int(row["Preference"])

        # Load images
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Concatenate the two images as input
        combined_images = torch.cat((image1, image2), dim=0)  # Concatenate along the channel dimension
        
        if should_swap:
            return (image2, image1), torch.tensor(1 - label, dtype=torch.float)

        return (image1, image2), torch.tensor(label, dtype=torch.float)
