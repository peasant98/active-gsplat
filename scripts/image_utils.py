import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import os

# DATASET_FOLDERS = ['bike_dataset', 'bonsai_dataset', 'counter_dataset', 'garden_dataset', 'stump_dataset', 'dataset', 'room_dataset']
DATASET_FOLDERS = ['basement1', 'basement2', 'bathroom1', 'bathroom5', 'bathroom6']

# Define custom dataset for image pairs
class ImagePairFullDataset(Dataset):
    def __init__(self, root_dataset_path = "datasets", dataset_folders=DATASET_FOLDERS, transform=None):
        """
        Full dataset with all pairs from different folders.
        Assume the root dataset folder is "datasets".
        Only include rows where 'Human Preference' is populated.
        """
        pairs = None
        self.root = root_dataset_path
        # For each folder, combine pairs filtering for valid Human Preference
        for folder in dataset_folders:
            csv_file = os.path.join(root_dataset_path, folder, folder + ".csv")
            df = pd.read_csv(csv_file)
            
            # Filter out rows with missing 'Human Preference'
            df = df.dropna(subset=["Human Preference"])
            
            # Add a column to record which folder the images belong to
            df["folder"] = folder
            
            if pairs is None:
                pairs = df
            else:
                pairs = pd.concat([pairs, df], ignore_index=True)
        
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        # Retrieve the folder recorded earlier
        folder = row["folder"]
        image1_path = os.path.join(self.root, folder, row["Image_1"])
        image2_path = os.path.join(self.root, folder, row["Image_2"])
        label = int(row["Human Preference"])

        # Load images
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), torch.tensor(label, dtype=torch.float)

class ImagePairSceneDataset(Dataset):
    def __init__(self, csv_file, dataset_folder, transform=None):
        self.pairs = pd.read_csv(csv_file)
        self.dataset_folder = dataset_folder
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        image1_path = os.path.join(self.dataset_folder, row["Image_1"])
        image2_path = os.path.join(self.dataset_folder, row["Image_2"])
        label = int(row["Preference"])

        # Load images
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), torch.tensor(label, dtype=torch.float)