import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)

from PIL import Image
import os
import pandas as pd
import argparse

# Dataset preparation
class PairwiseImageDataset(Dataset):
    def __init__(self, pairs_csv, dataset_folder, image_processor):
        self.pairs = pd.read_csv(pairs_csv)
        self.dataset_folder = dataset_folder
        self.image_processor = image_processor

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

        # Combine images into a single tensor (e.g., side-by-side)
        combined_image = Image.new("RGB", (image1.width + image2.width, image1.height))
        combined_image.paste(image1, (0, 0))
        combined_image.paste(image2, (image1.width, 0))

        # Apply feature extractor
        inputs = self.image_processor(images=combined_image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
        inputs["labels"] = torch.tensor(label, dtype=torch.long)

        return inputs

def train(dataset, model):
    # Split into train and validation datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="../models/base-ViT/results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_dir="../models/base-ViT/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    # import pdb
    # pdb.set_trace()

    trainer.train()

    # Save the model
    trainer.save_model("../models/base-ViT/fine_tuned_base-ViT")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Label pairs of images with human preferences.")
    parser.add_argument("--dataset_folder", "-d", type=str, required=True, default=None, help="Path to dataset where images are stored.")
    parser.add_argument("--csv_file", "-c", type=str, required=True, default=None, help="Path to CSV file with generated image pairs.")
    args = parser.parse_args()

    if args.dataset_folder is None or args.dataset_folder == "" or args.dataset_folder == " ":
        # print(args.dataset_folder)
        raise ValueError("Invalid Dataset path")
    
    if args.csv_file is None or args.csv_file == "" or args.csv_file == " ":
        raise ValueError("Invalid CSV File path")
    

    # Initialize the feature extractor and model
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=2  # Binary classification
    )

    # Load dataset
    dataset = PairwiseImageDataset(
        pairs_csv = args.csv_file, #"labeled_image_pairs.csv",
        dataset_folder = args.dataset_folder, #"path_to_new_1_percent_dataset",
        image_processor = image_processor
    )

    # print(dataset[0]['pixel_values'].shape)
    # Running train
    train(dataset = dataset, model = model)
    
