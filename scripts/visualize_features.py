import os
import os.path as osp

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

import torch
import torchvision.transforms as T

from pref_models import *
from image_utils import ImagePairSceneDataset

import argparse

def main(save_fg_mask=False, img_size=224, output_folder="outputs", model_name = "dino", model_path="models/dino/dino_all_scenes_old.pth"):
    os.makedirs(output_folder, exist_ok=True)

    assert img_size % 14 == 0, "The image size must be exactly divisible by 14"

    if model_name == "dino":
        model = Dinov2PreferenceModel()
    elif model_name == "hiera":
        model = HieraPreferenceModel()
    else:
        model = ResNetPreferenceModel ()
    
    model.load_state_dict(torch.load(model_path))
    model.to("cuda")

    transform = T.Compose([
                    T.ToTensor(), 
                    T.Resize(img_size+int(img_size*0.01)*10), 
                    T.CenterCrop(img_size), 
                    T.Normalize([0.5], [0.5]), 
                ])
    
    patch_h = patch_w = img_size // 14
    
    # Images to visualize

    img_folder = "datasets/bathroom/bathroom_16"
    base_name = img_folder.split("/")[-1] # bathroom_16
    csv_file = osp.join(img_folder, base_name + ".csv")
    image_dataset = ImagePairSceneDataset(csv_file=csv_file, dataset_folder=img_folder, transform=transform)
    img_cnt = len(image_dataset)

    if img_cnt == 0:
        raise ValueError(f"No images found in {img_folder}.")

    print("Image count: ", img_cnt)

    images = []
    images_plot = []
    for i in range(img_cnt):
        # image_dataset returns (image1, image2, label)
        (img1, img2), _ = image_dataset[i]
        images.append(img1)
        images.append(img2)

        # Unnormalize the images for plotting (assumes normalization was [-0.5, 0.5])
        img1_np = (img1.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
        img2_np = (img2.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
        images_plot.append((img1_np.astype(np.uint8), img2_np.astype(np.uint8)))
    images = torch.stack(images).cuda()
    
    img_cnt =len(images)
    print("New image count: ", img_cnt)
    
    print("Dataset created...")

    with torch.no_grad():

        if model_name == "hiera":
            import pdb
            pdb.set_trace()
            embeddings = model.forward_features(images, 1) # [batch_size, seq_length, hidden_dim]
            print(embeddings.shape)
            x_norm_patchtokens = embeddings.cpu().numpy()
        else:
            embeddings = model.dino1.forward_features(images)
            # print(embeddings.keys())
            x_norm_patchtokens = embeddings["x_norm_patchtokens"].cpu().numpy() # [batch_size, seq_length, hidden_dim]

    x_norm_1616_patches = x_norm_patchtokens.reshape(img_cnt*patch_h*patch_w, -1)

    fg_pca = PCA(n_components=1)
    fg_pca_images = fg_pca.fit_transform(x_norm_1616_patches)
    fg_pca_images = minmax_scale(fg_pca_images)
    fg_pca_images = fg_pca_images.reshape(img_cnt, patch_h*patch_w)

    masks = []
    for i in range(img_cnt):
        image_patches = fg_pca_images[i,:]
        # mask = (image_patches < 0.4).ravel()
        mask = (image_patches > 0.6).ravel()
        masks.append(mask)

    if save_fg_mask:
        # Number of image pairs (each entry in images_plot is a tuple of two images)
        n_pairs = len(images_plot)
        plt.figure(figsize=(10, 5*n_pairs))
        for i in range(n_pairs):
            idx1 = 2 * i
            idx2 = 2 * i + 1

            # Process the first image in the pair
            image_patches1 = fg_pca_images[idx1, :].copy()
            mask1 = masks[idx1]
            image_patches1[~mask1] = 0
            image_patches1 = image_patches1.reshape(patch_h, patch_w)

            plt.subplot(n_pairs, 2, 2*i+1)
            plt.axis("off")
            plt.imshow(images_plot[i][0])
            plt.imshow(image_patches1, extent=(0, img_size, img_size, 0), alpha=0.5, cmap="jet")

            # Process the second image in the pair
            image_patches2 = fg_pca_images[idx2, :].copy()
            mask2 = masks[idx2]
            image_patches2[~mask2] = 0
            image_patches2 = image_patches2.reshape(patch_h, patch_w)

            plt.subplot(n_pairs, 2, 2*i+2)
            plt.axis("off")
            plt.imshow(images_plot[i][1])
            plt.imshow(image_patches2, extent=(0, img_size, img_size, 0), alpha=0.5, cmap="jet")
        plt.savefig(osp.join(output_folder, "fg_mask.jpg"))
        plt.close()

    pca = PCA(n_components=3)
    fg_patches = np.vstack([x_norm_patchtokens[i,masks[i],:] for i in range(img_cnt)])
    pca_features = pca.fit_transform(fg_patches)
    fg_result = minmax_scale(pca_features)

    mask_indices = [0, *np.cumsum([np.sum(m) for m in masks]), -1]

    n_pairs = len(images_plot)
    plt.figure(figsize=(12, 6 * n_pairs))
    for i in range(n_pairs):
        idx1 = 2 * i
        idx2 = 2 * i + 1

        # Process the first image in the pair
        pca_results1 = np.zeros((patch_h * patch_w, 3), dtype='float32')
        pca_results1[masks[idx1]] = fg_result[mask_indices[idx1]:mask_indices[idx1 + 1]]
        pca_results1 = pca_results1.reshape(patch_h, patch_w, 3)
        plt.subplot(n_pairs, 2, 2 * i + 1)
        plt.axis("off")
        plt.imshow(images_plot[i][0])
        plt.imshow(pca_results1, extent=(0, img_size, img_size, 0), alpha=0.5, cmap="jet")

        # Process the second image in the pair
        pca_results2 = np.zeros((patch_h * patch_w, 3), dtype='float32')
        pca_results2[masks[idx2]] = fg_result[mask_indices[idx2]:mask_indices[idx2 + 1]]
        pca_results2 = pca_results2.reshape(patch_h, patch_w, 3)
        plt.subplot(n_pairs, 2, 2 * i + 2)
        plt.axis("off")
        plt.imshow(images_plot[i][1])
        plt.imshow(pca_results2, extent=(0, img_size, img_size, 0), alpha=0.5, cmap="jet")
    plt.savefig(osp.join(output_folder, "results.jpg"))
    plt.close()

    ### Raw
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(x_norm_patchtokens.reshape(img_cnt * patch_h * patch_w, -1))
    pca_features = pca_features.reshape(img_cnt, patch_h * patch_w, 3)
    for i in range(img_cnt):
        mask = masks[i]
        pca_features[i, ~mask] = np.min(pca_features[i])
    pca_features = pca_features.reshape(img_cnt * patch_h * patch_w, 3)
    fg_result = minmax_scale(pca_features)
    fg_result = fg_result.reshape(img_cnt, patch_h, patch_w, 3)

    n_pairs = len(images_plot)
    plt.figure(figsize=(12, 6 * n_pairs))
    for i in range(n_pairs):
        idx1 = 2 * i
        idx2 = 2 * i + 1

        # Process the first image in the pair
        plt.subplot(n_pairs, 2, 2 * i + 1)
        plt.axis("off")
        plt.imshow(images_plot[i][0])
        plt.imshow(fg_result[idx1], extent=(0, img_size, img_size, 0), alpha=0.5, cmap="jet")

        # Process the second image in the pair
        plt.subplot(n_pairs, 2, 2 * i + 2)
        plt.axis("off")
        plt.imshow(images_plot[i][1])
        plt.imshow(fg_result[idx2], extent=(0, img_size, img_size, 0), alpha=0.5, cmap="jet")
    plt.savefig(osp.join(output_folder, "results_raw.jpg"))
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize features from the trained model")
    parser.add_argument("--save-fg-mask", action="store_true", default=True, help="Save the foreground mask")
    parser.add_argument("--img-size", type=int, default=224, help="Image size for visualization")
    parser.add_argument("--output-folder", type=str, default="outputs", help="Output folder for saving results")
    parser.add_argument("--model-name", "-m", type=str, default="dino", help="Model name for visualization")
    parser.add_argument("--model-path", type=str, default="models/dino/dino_all_scenes_old.pth", help="Path to the model weights")

    args = parser.parse_args()

    main(save_fg_mask=args.save_fg_mask,
         img_size=args.img_size,
         output_folder=args.output_folder,
         model_name=args.model_name,
         model_path=args.model_path)