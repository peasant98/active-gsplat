from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
import os

class ImagePairDataset(Dataset):
    def __init__(self, csv_file, dataset_folder, transform=None):
        self.pairs = pd.read_csv(csv_file)
        self.dataset_folder = dataset_folder

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                            ]) # Normalization for concatenated input

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

        # Concatenate the two images as input
        combined_images = torch.cat((image1, image2), dim=0)  # Concatenate along the channel dimension

        return combined_images, torch.tensor(label, dtype=torch.float)
    
if __name__ == "__main__":
    dataset = ImagePairDataset("../kitchen_small_dataset/kitchen_all_image_pairs.csv", "../kitchen_small_dataset/")

    print(f"Dataset Size: {len(dataset)}")

    # images, labels = dataset[0]
    # print()
