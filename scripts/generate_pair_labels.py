import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def display_images_with_input(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    preference = {"value": None}

    def on_key(event):
        if event.key in ['1', '2', '0']:
            preference["value"] = int(event.key)
            plt.close()

    # Display the images
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title("Image 1 (Press 1 for this)")
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title("Image 2 (Press 2 or 0 for this)")
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    
    

    return preference["value"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label pairs of images with human preferences.")
    parser.add_argument("--dataset_folder", "-d", type=str, required=True, help="Path to dataset where images are stored.")
    parser.add_argument("--csv_file", "-c", type=str, required=True, help="Path to CSV file with generated image pairs.")
    args = parser.parse_args()

    if not args.dataset_folder or not os.path.isdir(args.dataset_folder):
        raise ValueError("Invalid dataset folder path.")
    if not args.csv_file or not os.path.isfile(args.csv_file):
        raise ValueError("Invalid CSV file path.")

    # Load dataset and CSV
    dataset_folder = args.dataset_folder
    pairs_csv = args.csv_file
    pairs_df = pd.read_csv(pairs_csv)

    # Add a column for labels if it doesn't exist
    if "Preference" not in pairs_df.columns:
        pairs_df["Preference"] = None

    # Iterate through each pair for labeling
    for index, row in pairs_df.iterrows():
        # Skip if preference is already input
        if not pd.isnull(row["Preference"]):
            continue

        image1_path = os.path.join(dataset_folder, row["Image_1"])
        image2_path = os.path.join(dataset_folder, row["Image_2"])

        # Display the images and get preference
        preference = display_images_with_input(image1_path, image2_path)
        
        if preference == 2:
            preference = 0

        # Save the preference
        pairs_df.at[index, "Preference"] = preference if preference is not None else None

        # Save progress in-place
        pairs_df.to_csv(pairs_csv, index=False)
        print(f"Preference saved for pair {index + 1}/{len(pairs_df)}.")

    print("All pairs labeled.")
