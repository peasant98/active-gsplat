import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import argparse

from pref_models import *
from image_utils import *
import random

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Visualize trained model predictions")
    parser.add_argument("--model-path", type=str, default="models/hiera/all_scenes.pth", help="Path to the trained model")
    parser.add_argument("--model", "-m", type=str, default="hiera", help="Model to use for visualization. Options: resnet, dino, hiera")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU evaluation using CUDA")
    parser.add_argument("--num-samples", "-n", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--test_dataset", "-t", type=str, help="Path to the test dataset folder")
    
    args = parser.parse_args()

    print(f"Using model: {args.model}")

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)) if args.model == "resnet" else transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if args.test_dataset is not None:
        folder = args.test_dataset.split("/")[-2] #Ex: [datasets, bathroom, bathroom_16, " "]
        csv_file = os.path.join(args.test_dataset, folder + ".csv")
        full_dataset = ImagePairSceneDataset(csv_file=csv_file, dataset_folder=args.test_dataset, transform=transform)
    else:
        full_dataset = ImagePairFullDataset(root_dataset_path="datasets", transform=transform)

    num_samples = args.num_samples #random.randint(1, len(full_dataset))

    if args.model == "hiera":
        model = HieraPreferenceModel()
    elif args.model == "dino":
        model = Dinov2PreferenceModel()
    else:
        model = ResNetPreferenceModel()

    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    
    indices = random.sample(range(len(full_dataset)), num_samples)
    for idx in indices:
        # Assume each sample is a tuple (image1, image2) already processed by the dataset’s transform.
        (image1, image2), label = full_dataset[idx]
        # Add batch dimension and send to device
        image1_tensor = image1.unsqueeze(0).to(device)
        image2_tensor = image2.unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = model(image1_tensor, image2_tensor).cpu().item()

        # # Convert tensors to numpy arrays for visualization (H, W, C)
        image1 = image1.cpu().permute(1, 2, 0).numpy()
        image2 = image2.cpu().permute(1, 2, 0).numpy()

        # Visualize the pair and the prediction
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image1)
        axes[0].axis("off")
        axes[0].set_title("Image 1")
        axes[1].imshow(image2)
        axes[1].axis("off")
        axes[1].set_title("Image 2")
        if prediction > 0.5:
            plt.suptitle(f"{args.model}'s Prediction: {prediction:.4f} Image 1 is preferred", fontsize=16)
        else:
            plt.suptitle(f"{args.model}'s Prediction: {prediction:.4f} Image 2 is preferred", fontsize=16)
        plt.show()