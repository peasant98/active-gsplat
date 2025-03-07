# Active Learning for Gaussian Splatting

This repo is a minimal implementation of active learning for Gaussian Splatting. It is customizable and allows for real-time training of splats.

This work supports the paper [HP-GS: Human Preference Next Best View Selection for 3D Gaussian Splatting](https://www.youtube.com/watch?v=t3gCQJGRSSY). We link the paper in this repo [here](https://github.com/peasant98/active-gsplat/blob/main/papers/HP-GS.pdf), but will be uploading the paper to Arxiv next month with some additional experiments.

![image](https://github.com/user-attachments/assets/9c963de4-67d8-490b-9581-541055ada916)



## Training the Preference Model

To train a specific preference model (resnet or dino), run the following script:

```bash

python3 scripts/train_pref_model.py -m resnet -c [path to csv file] -d [path to images] -s [path to model checkpoint .pth] -b 16

```


## Running View Selection

Run the below bash and fill it in with your customizations:

```bash
cd strong_gsplat/

python3 examples/simple_active_trainer.py default --disable_viewer --data_factor <specify> \\
--render_traj_path ellipse --data_dir data/360_v2/<object>/ \\
--result_dir results/<specify>/ --view_selection_method <view selection method> --should_render False

```

`view_selection_method` is one of `random`, `fisher`, or `pref_model` (you will need to specify a path to the pref model)
