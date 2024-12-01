import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def display_images(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Display the images
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title("Image 1")
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title("Image 2")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Label pairs of images with human preferences.")
    parser.add_argument("--dataset_folder", "-d", type=str, required=True, default=None, help="Path to dataset where images are stored.")
    parser.add_argument("--csv_file", "-c", type=str, required=True, default=None, help="Path to CSV file with generated image pairs.")
    args = parser.parse_args()

    if args.dataset_folder is None or "" or " ":
        raise ValueError("Invalid Dataset path")
    
    if args.csv_file is None or "" or  " ":
        raise ValueError("Invalid CSV File path")

    # Paths
    dataset_folder = args.dataset_folder # "../one_percent_dataset"
    pairs_csv =  args.csv_file # (dataset_folder + "/onepercent_all_image_pairs.csv")
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

        # Display the images
        display_images(image1_path, image2_path)

        # Input preference
        while True:
            try:
                preference = int(input("Enter your preference (1 for Image 1, 2 or 0 for Image 2): "))
                if preference in [0, 1, 2]:
                    break
                else:
                    print("Invalid input. Please enter 1 or 0.")
            except ValueError:
                print("Invalid input. Please enter 1 or 0.")

        # Save the preference
        pairs_df.at[index, "Preference"] = preference if preference == 1 or preference == 0 else 0

        # Save progress in-place
        pairs_df.to_csv(pairs_csv, index=False)
        print(f"Preference saved for pair {index + 1}/{len(pairs_df)}.")

    print("All pairs labeled.")
