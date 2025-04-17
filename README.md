# Active Learning for Gaussian Splatting

This repo is a minimal implementation of active learning for Gaussian Splatting. It is customizable and allows for real-time training of splats.

This work supports the paper [HP-GS: Human Preference Next Best View Selection for 3D Gaussian Splatting](https://www.youtube.com/watch?v=t3gCQJGRSSY). We link the paper in this repo [here](https://github.com/peasant98/active-gsplat/blob/main/papers/HP-GS.pdf), but will be uploading the paper to Arxiv next month with some additional experiments.

![image](https://github.com/user-attachments/assets/9c963de4-67d8-490b-9581-541055ada916)


## Generating Image Pair dataset

To generate an image pair dataset from a given directory of images of a scene, run the following script:

```bash

python3 scripts/generate_image_pairs.py -d <path to image folder> -o <path to output csv file> -n <number of pairs>

```

Then, to label the image pairs using binary preference labels, run:

```bash

python3 scripts/generate_pair_labels.py -d <path to image folder> -c <path to csv file>

```

This will open an interactive UI where you can select (1s or 0s) for each image pair. The script will save the labels in the same CSV file. If you'd like an LLM to automatically label the dataset then add the flag `--label_llm`. This will run Gemini 2.0 Flash on the image pairs. For further modifications to the model, you can check out the `generate_pair_labels.py` script.

## Training the Preference Model

Next, to train a specific preference model (resnet or dino), run the following script:

```bash

python3 scripts/train_pref_model.py -m <resnet or dino> -c <path to csv file> -d <path to image folder> -s <path to output model checkpoint .pth> -b 16

```

Use the flag `--all-scenes` if you would like to train the model over all possible scenes. If you wish to add more scenes, modify the DATASET_FOLDERS list at the top of `scripts/image_utils.py`.

The preference models are stored in `scripts/pref_models.py`.

## Visualizing the Preference Model
You can use the following script to visualize predictions of the trained model:

```bash
python3 scripts/visualize_pref_model.py -m <resnet or dino or hiera> --model-path <path to model ckpt .pth> --gpu -n <number of samples you want to visualize> -t <path to test dataset folder>

```

Note that the if the test dataset flag is not passed, it will automatically use random samples from the full dataset it was trained on. Also ensure that the test dataset folder contains a csv with some image pairs. You can create this csv using the `generate_image_pairs.py` script. 

## Running View Selection

Run the below bash and fill it in with your customizations:

```bash
cd strong_gsplat/

python3 examples/simple_active_trainer.py default --disable_viewer --data_factor <specify> \\
--render_traj_path ellipse --data_dir data/360_v2/<object>/ \\
--result_dir results/<specify>/ --view_selection_method <view selection method> --should_render False

```

`view_selection_method` is one of `random`, `fisher`, or `pref_model` (you will need to specify a path to the pref model)
