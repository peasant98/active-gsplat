import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import os
import argparse
import tqdm

# Define custom dataset for image pairs
class ImagePairDataset(Dataset):
    def __init__(self, csv_file, dataset_folder, transform=None, label="Human Preference"):
        self.pairs = pd.read_csv(csv_file)
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.pairs) * 2

    def __getitem__(self, idx):
        
        real_idx = idx // 2
        should_swap = idx % 2 == 1
        
        row = self.pairs.iloc[real_idx]
        image1_path = os.path.join(self.dataset_folder, row["Image_1"])
        image2_path = os.path.join(self.dataset_folder, row["Image_2"])
        label = int(row[self.label])

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

class ResNetPreferenceModel(nn.Module):
    def __init__(self, pretrained_model_name="resnet50"):
        super(ResNetPreferenceModel, self).__init__()
        
        # Load pre-trained ResNet for two independent feature heads
        self.resnet1 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.resnet2 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Modify input layers of both ResNets to accept single 3-channel image
        self.resnet1.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet2.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract number of features from ResNet output
        num_features = self.resnet1.fc.in_features
        
        # Remove the classification heads; we only need features
        self.resnet1.fc = nn.Identity()
        self.resnet2.fc = nn.Identity()
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(num_features * 2, 1),  # Binary classification
            nn.Sigmoid()  # Output probability
        )
        
        self.linear = nn.Linear(num_features, 1)

    def forward(self, img1, img2):
        # Extract features from both images
        features1 = self.resnet1(img1)
        features2 = self.resnet2(img2)
        
        x1 = self.linear(features1)
        x2 = self.linear(features2)
        
        output = torch.exp(x1) / (torch.exp(x1) + torch.exp(x2))
        
        # Element-wise max of the two feature vectors
        # combined_features = torch.cat((features1, features2), dim=1)
        
        # Classification based on the selected features
        # output = self.fc(combined_features)
        return output

# Load pre-trained ResNet and modify it
class ResNetBinaryClassifier(nn.Module):
    def __init__(self, pretrained_model_name="resnet50"):
        super(ResNetBinaryClassifier, self).__init__()
        # Load pre-trained ResNet
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify input layer to accept concatenated images (6 channels instead of 3)
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the classification head for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1),  # Binary classification
            nn.Sigmoid()  # Output probability
        )

        # self.criterion = nn.BCELoss()
        # self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        return self.resnet(x)

def train_resnet(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    num_epochs=10,
    save_path = "kitchen_resnet_big.pth"
):

    print("Beginning training... \n")
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0

        # Training phase
        model.train()
        train_progress = tqdm.tqdm(train_loader, desc="Training", leave=False)
        for images, labels in train_progress:
            optimizer.zero_grad()
            image1, image2 = images

            # Forward pass
            outputs = model(image1, image2)
            outputs = outputs.squeeze()  # Remove extra dimension: [batch_size, 1] -> [batch_size]

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_progress.set_postfix({"Batch Loss": loss.item()})

        # Validation phase
        val_loss = 0
        correct = 0
        total = 0
        model.eval()
        val_progress = tqdm.tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, labels in val_progress:
                image1, image2 = images
                outputs = model(image1, image2)
                outputs = outputs.squeeze()

                # Compute loss
                val_loss += criterion(outputs, labels).item()

                # Compute accuracy
                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_progress.set_postfix({"Batch Loss": val_loss})

        val_accuracy = correct / total
        print(f"Train Loss: {epoch_loss / len(train_loader):.4f}, "
                f"Val Loss: {val_loss / len(val_loader):.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}")
    
        # Save the model after training
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Save the model after training
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def test_resnet(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_progress = tqdm.tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for images, labels in test_progress:
            image1, image2 = images
            outputs = model(image1, image2)
            outputs = outputs.squeeze()

            # Compute accuracy
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            test_progress.set_postfix({"Accuracy": correct / total})

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Resnet on human pref dataset for other view selection.")
    parser.add_argument("--dataset_folder", "-d", type=str, required=True, default=None, help="Path to dataset where images are stored.")
    parser.add_argument("--csv_file", "-c", type=str, required=True, default=None, help="Path to CSV file with generated image pairs.")
    parser.add_argument("--save_path", "-s", default = ".pth", type=str, help=" Path to save model checkpoint .pth")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help = "Batch Size for model training")
    parser.add_argument("--label", "-l", type=str, default="Human Preference", help="Determines whether to use LLM agent to label preferences.")
    args = parser.parse_args()

    if args.dataset_folder is None or args.dataset_folder == "" or args.dataset_folder == " ":
        # print(args.dataset_folder)
        raise ValueError("Invalid Dataset path")
    
    if args.csv_file is None or args.csv_file == "" or args.csv_file == " ":
        raise ValueError("Invalid CSV File path")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for concatenated input
    ])

    # Load dataset
    full_dataset = ImagePairDataset(args.csv_file, args.dataset_folder, transform=transform, label = args.label)

    # Define split sizes
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Dataset created... \n")

    # Initialize model, loss, and optimizer
    model = ResNetPreferenceModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Running training
    train_resnet(model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, save_path=args.save_path, num_epochs=5)

    # Running test
    test_resnet(model, test_loader=test_loader)