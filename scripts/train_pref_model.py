import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import argparse
import tqdm

from pref_models import *
from image_utils import *

def train(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    num_epochs=5,
    save_path="kitchen_big.pth",
    device=None
):
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
            image1 = image1.to(device)
            image2 = image2.to(device)
            labels = labels.to(device)

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
                image1 = image1.to(device)
                image2 = image2.to(device)
                labels = labels.to(device)
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

def test(model, test_loader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.eval()
    correct = 0
    total = 0
    test_progress = tqdm.tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for images, labels in test_progress:
            image1, image2 = images
            image1 = image1.to(device)
            image2 = image2.to(device)
            labels = labels.to(device)
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
    parser = argparse.ArgumentParser(description="Train Preference Model on Human Pref dataset for Next Best View selection.")
    parser.add_argument("--model", "-m", type=str, required=False, default="resnet", help="Model to use for training. Options: resnet, dino")    
    parser.add_argument("--csv-file", "-c", type=str, nargs="+", required=False, help="Path(s) to CSV file(s) with generated image pairs.")
    parser.add_argument("--dataset-folder", "-d", type=str, nargs="+", required=False, default="datasets", help="Path(s) to dataset folder(s) where images are stored.")
    parser.add_argument("--save-path", "-s", type=str, help=" Path to save model checkpoint .pth")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help = "Batch Size for model training")
    parser.add_argument("--all-scenes", action="store_true", default=False, help="Use all scenes in the dataset. Default is False.")

    args = parser.parse_args()

    if args.all_scenes:
        # Check if args.dataset_folder is not empty
        if args.dataset_folder is None or args.dataset_folder == "" or args.dataset_folder == " ":
            raise ValueError("Invalid dataset folder path when all scenes has been selected")
    else:
        if args.csv_file is None or args.csv_file == "" or args.csv_file == " ":
            raise ValueError("Invalid CSV File path")
    
    if args.model not in PREF_MODELS:
        raise ValueError(f"Invalid Model type. Choose between {PREF_MODELS}")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)) if args.model == "resnet" else transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset
    if args.all_scenes:
        full_dataset = ImagePairFullDataset(transform=transform)
    else:
        full_dataset = ImagePairSceneDataset(args.csv_file, args.dataset_folder, transform=transform)

    # Define split sizes
    dataset_size = len(full_dataset)
    print(f"Dataset size: {dataset_size}")
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

    # Device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, loss, and optimizer
    if args.model == "resnet":
        model = ResNetPreferenceModel() 
    elif args.model == "dino":
        model = Dinov2PreferenceModel()
    elif args.model == "hiera":
        model = HieraPreferenceModel()
    else:
        raise ValueError(f"Invalid Model type. Choose between {PREF_MODELS}")
        
    model = model.to(device)  # move model to GPU if available
    criterion = nn.BCELoss() # Neeed for Bradley-Terry model
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Training... {args.model} model\n")

    # Running training
    train(model, train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        save_path=args.save_path,
        device=device
    )

    print(f"Testing... {args.model} model \n")
    # Running test
    test(model, test_loader=test_loader, device=device)