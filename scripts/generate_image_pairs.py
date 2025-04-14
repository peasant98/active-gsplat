import os
import itertools
import random
import argparse
import pandas as pd

def generate_image_pairs(dataset_folder, output_csv, n_pairs=1000):

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure the output directory exists

    # Empty the CSV file if it exists
    if os.path.exists(output_csv):
        open(output_csv, 'w').close()  # Clear the file

    # Get a list of all images in the dataset folder
    images = [f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))]

    # Generate all possible pairs: (nC2)
    pairs = list(itertools.combinations(images, 2))

    # Select approximately n_pairs randomly
    pairs = random.sample(pairs, min(n_pairs, len(pairs)))

    # Convert pairs to a DataFrame for saving
    pairs_df = pd.DataFrame(pairs, columns=["Image_1", "Image_2"])

    # Save pairs to the CSV file
    pairs_df.to_csv(output_csv, index=False)

    print(f"Generated {len(pairs)} pairs and saved to {output_csv}")

if __name__=="__main__":


    parser = argparse.ArgumentParser(description="Generate image pairs from a dataset folder.")
    parser.add_argument("--dataset_folder", "-d", type=str, required=True, help="Path to dataset where images are stored.")
    parser.add_argument("--output_csv", "-o", type=str, required=True, help="Path to output CSV file for image pairs.")
    parser.add_argument("--n_pairs", "-n", type=int, default=1000, help="Number of pairs to generate (default: 1000).")
    args = parser.parse_args()

    generate_image_pairs(args.dataset_folder, args.output_csv, args.n_pairs)