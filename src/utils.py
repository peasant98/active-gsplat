import os
import itertools
import pandas as pd
import random
import shutil

def make_smaller_dataset(original_folder, new_folder, percent):

    os.makedirs(new_folder, exist_ok=True)

    all_images = [f for f in os.listdir(original_folder) if os.path.isfile(os.path.join(original_folder, f))]
    num_images_to_sample = max(1, int(len(all_images) * percent))  # Ensure at least one image is selected
    random_sample = random.sample(all_images, num_images_to_sample)

    for image in random_sample:
        shutil.copy(os.path.join(original_folder, image), os.path.join(new_folder, image))

    print(f"Copied {len(random_sample)} images to {new_folder}")

def generate_image_pairs(dataset_folder, output_csv):

    images = [f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))]

    pairs = list(itertools.combinations(images, 2))
    pairs_df = pd.DataFrame(pairs, columns=["Image_1", "Image_2"])
    pairs_df.to_csv(output_csv, index=False)

    print(f"Generated {len(pairs)} pairs and saved to {output_csv}")