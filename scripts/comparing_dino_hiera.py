import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pref_models import *
from image_utils import *

# Visualize the pair and the prediction
import matplotlib.patches as patches

if __name__=="__main__":

    dino_model_path = "models/dino/dino_all_scenes.pth"
    hiera_model_path = "models/hiera/all_scenes.pth"
    num_samples = 5 #random.randint(1, len(full_dataset))

    print(f"Comparing Hiera and Dino on {num_samples} test Images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    full_dataset = ImagePairFullDataset(root_dataset_path="datasets", transform=transform)

   
    df = full_dataset.pairs.sample(n=num_samples)
    hiera_model = HieraPreferenceModel()
    dino_model = Dinov2PreferenceModel()

    hiera_model.load_state_dict(torch.load(hiera_model_path))
    hiera_model.to(device)

    dino_model.load_state_dict(torch.load(dino_model_path))
    dino_model.to(device)

    hiera_model.eval()
    dino_model.eval()
    for _, row in df.iterrows():
        folder = row["folder"]
        image1_path = os.path.join(full_dataset.root, folder, row["Image_1"])
        image2_path = os.path.join(full_dataset.root, folder, row["Image_2"])

        # Load and preprocess images
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")
        image1_tensor = transform(image1).unsqueeze(0).to(device)  # Add batch dimension
        image2_tensor = transform(image2).unsqueeze(0).to(device)  # Add batch dimension

        # print(image1_tensor.shape, image2_tensor.shape)

        # Make prediction
        with torch.no_grad():
            hiera_pred = hiera_model(image1_tensor, image2_tensor).cpu().item()
            dino_pred = dino_model(image1_tensor, image2_tensor).cpu().item()
            # print(prediction)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        im1 = axes[0].imshow(image1)
        axes[0].axis("off")
        axes[0].set_title("Image 1")
        
        im2 = axes[1].imshow(image2)
        axes[1].axis("off")
        axes[1].set_title("Image 2")
        
        # For dino model: red box around the preferred image
        if dino_pred > 0.5:
            preferred_ax = axes[0]
            im = im1
        else:
            preferred_ax = axes[1]
            im = im2
        extent = im.get_extent()  # [xmin, xmax, ymin, ymax]
        width = extent[1] - extent[0]
        height = extent[3] - extent[2]
        red_rect = patches.Rectangle((extent[0], extent[2]), width, height,
                         linewidth=3, edgecolor='red', facecolor='none')
        preferred_ax.add_patch(red_rect)
        
        # For hiera model: blue box around the preferred image
        if hiera_pred > 0.5:
            preferred_ax = axes[0]
            im = im1
        else:
            preferred_ax = axes[1]
            im = im2
        extent = im.get_extent()
        width = extent[1] - extent[0]
        height = extent[3] - extent[2]
        blue_rect = patches.Rectangle((extent[0], extent[2]), width, height,
                          linewidth=3, edgecolor='blue', facecolor='none')
        preferred_ax.add_patch(blue_rect)
        
        plt.suptitle(f"Dino Prediction (Red): {dino_pred:.4f} | Hiera Prediction (Blue): {hiera_pred:.4f}", fontsize=16)
        plt.show()