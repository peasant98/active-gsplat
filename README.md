# Active Learning for Gaussian Splatting

This repo is a minimal implementation of active learning for Gaussian Splatting. It is customizable and allows for real-time training of splats.

This work supports the paper [HP-GS: Human Preference Next Best View Selection for 3D Gaussian Splatting](https://www.youtube.com/watch?v=t3gCQJGRSSY). We link the paper in this repo [here](https://github.com/peasant98/active-gsplat/blob/main/papers/HP-GS.pdf), but will be uploading the paper to Arxiv next month with some additional experiments.

![image](https://github.com/user-attachments/assets/9c963de4-67d8-490b-9581-541055ada916)



## Training the Preference Model

Docs in progress!


## Running View Selection

Run the below bash and fill it in with your customizations:

```bash
cd strong_gsplat/

python3 examples/simple_active_trainer.py default --disable_viewer --data_factor <specify> \\
--render_traj_path ellipse --data_dir data/360_v2/<object>/ \\
--result_dir results/<specify>/ --view_selection_method <view selection method> --should_render False

```

`view_selection_method` is one of `random`, `fisher`, or `pref_model` (you will need to specify a path to the pref model)

# Habitat-Sim Setup and Dataset Download

To use Habitat-Sim and download the hssd-hab dataset for generating environments and datasets, follow these steps:

1. **Create and activate a compatible conda environment:**
	```bash
	conda create -n habitat_sim python=3.9 cmake=3.27
	conda activate habitat_sim
	```

2. **Install Habitat-Sim with display and headless rendering support:**
	```bash
	conda install habitat-sim withbullet -c conda-forge -c aihabitat
	```

3. **Download the hssd-hab dataset:**
	```bash
	python -m habitat_sim.utils.datasets_download --uids hssd-hab --data-path /path/to/dataset/
	```

	*Note: To view all possible downloadable datasets, run:*
	```bash
	python -m habitat_sim.utils.datasets_download --list
	```

4. **To use the installed viewer:**
    ```bash
    habitat-viewer --dataset /path/to/hssd-hab/hssd-hab.scene_dataset_config.json -- 102344280
    ```

This will set up Habitat-Sim and place the hssd-hab dataset in the `datasets/` directory. You can now use these assets for generating your own datasets and running your data processing scripts.