# DeepLearningTennis3DUplift

This repository contains the code base for the master thesis:

"From Broadcast to 3D: A Deep Learning Approach for Tennis Trajectory and Spin Estimation"

The project combines:

- A highly accurate MuJoCo-based tennis simulation that models aerodynamic effects (including Magnus force) and realistic ball-court interactions.
- Synthetic data generation for both single trajectories and stitched rallies.
- Neural networks that predict 3D ball position and spin from 2D ball coordinates and 16 court keypoints (see `images/keypoints.png`).

Across the explored model variants, the rally model in the `dynamicAnkle` configuration showed the best overall performance.

Publication note: the thesis is complete, but the corresponding paper is not yet published.

## Usage via torch.hub

This repository provides a PyTorch Hub interface for loading trained uplifting models directly.

### 1. Install Requirements

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Adapt the Path to Weights

Edit `paths.py` and set `weights_path` to the base directory that contains these four subfolders:

- `trajectories_dynamic`
- `trajectories_dynamicAnkle`
- `rallies_dynamic`
- `rallies_dynamicAnkle`

Each subfolder must contain a `model.pt` file.

### 3. Load Model via torch.hub

```python
import torch

model = torch.hub.load(
    "https://github.com/xSpaceCoder/DeepLearningTennis3DUplift",
    "tennis_uplifting",
    input_type="rallies",
    mode="dynamicAnkle",
)
```

Supported options:

- `input_type`: `trajectories` or `rallies`
- `mode`: `dynamic` or `dynamicAnkle`

Predict 3D position and spin with for mode dynamic (ankle_v can be added but will be ignored):

```python
model.predict(ball_coords, court_coords, times)
```

Predict 3D position and spin with for mode dynamicAnkle:

```python
model.predict(r_img, court_img, mask, times, ankle_v)
```

The model expects 2D ball coordinates, 16 court keypoints, and timestamps.

For API details, see `torchhub/interface_rallies_dynamicAnkle.py`.

Further reading:

- https://docs.pytorch.org/docs/stable/hub.html#loading-models-from-hub

## Download Datasets from Hugging Face
All used datasets are saved and documented as public on Hugging Face

### Augmented TrackNet Datasets (ACE)
As base for the real datasets the TrackNet datasets were used. The were augmented and saved as npy files:

Single Trajectories (inlcuding ball tosses before serves): https://huggingface.co/datasets/XSpaceCoderX/ACE-Trajectories_withTosses
Single Trajectories (excluding ball tosses before serves): https://huggingface.co/datasets/XSpaceCoderX/ACE-Trajectories_noTosses
Whole Rallies: https://huggingface.co/datasets/XSpaceCoderX/AD-Rallies

### Artificial Datasets (AD)
For the training datasets this code simulates a dataset with 187,200 single trajectories with hit types serve, groundstroke, lob, smash, volley and short (AD-Trajectories). 

Single Trajectories: https://huggingface.co/datasets/XSpaceCoderX/AD-Trajectories

For taining the Rally model this thesis used the simulated trajectories from the AD-Trajectories dataset as bases and stitched them together. This was done with a recursive function that branched multiple times to receive a max of 160 rallies based on one ball toss. The result is a dataset with approxemately 3.2 Milion Rallies for the AD-Rallies Dataset:

Rallies: https://huggingface.co/datasets/XSpaceCoderX/AD-Rallies


## Workspace Setup

Create and activate a Python virtual environment, then install the full dependency set:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_full.txt
```

After installing dependencies, configure paths in `paths.py`:

If you do not want to simulate datasets locally, pretrained/simulated assets are available on Hugging Face:

- https://huggingface.co/XSpaceCoderX

This includes:

- Enriched TrackNet datasets (ACE) for inference
- Artificial datasets (AD) for training

## Dataset Generation

### 1. Generate Single-Trajectory Synthetic Data

Use the provided script:

```bash
bash create_synthetic_dataset.sh
```

This runs the trajectory simulation pipeline from `syntheticdataset/mujocosimulation.py` for multiple shot types and directions.

### 2. Generate Stitched Synthetic Rallies

After trajectory pools are available, generate stitched rally trees by executing:

```bash
python -m syntheticdataset.rallyStitching.generate_stitched_rallies
```

This creates branched rally trajectories by stitching toss, serve, and return segments. An example rally is visible in images/3d_rally.png. This view as well as a top view can be created with the script syntheticdataset/rallyStitching/vizualize_rally.py

## Training

### Train Trajectory Model (`uplifting`)

For single-trajectory training, run:

```bash
python -m uplifting.train --gpu 0 --folder results --token_mode dynamic --time_rotation new
```
*folder*: define name of the folder includding the simulated trajectories
*token_mode*: dynamic or dynamicAnkle (dynamic takes in the ball and court keypoint positions on the 2d image, dynamicAnkle takes in the same but also requires an ankle_v coodrinate that correspons to the v-coordinate of the hitting player at the moment of hitting the trajectory)

Example:

```bash
python -m uplifting.train --gpu 0 --folder results --token_mode dynamicAnkle --time_rotation new
```

### Train Rally Model (`uplifting_rallies`)

For rally training, run:

```bash
python -m uplifting_rallies.train --gpu 0 --folder results --token_mode <token_mode> --time_rotation new
```

*folder*: define name of the folder includding the simulated trajectories
*token_mode*: dynamic or dynamicAnkle (dynamic takes in the ball and court keypoint positions on the 2d image, dynamicAnkle takes in the same but also requires an ankle_v coodrinate that correspons to the v-coordinate of the serving player at the moment of throwing the ball toss at the start of the rally)

Recommended configuration from the thesis:

```bash
python -m uplifting_rallies.train --gpu 0 --folder results --token_mode dynamicAnkle --time_rotation new
```

Optional resume from latest checkpoint, since training takes long:

```bash
python -m uplifting_rallies.train --gpu 0 --folder results --token_mode dynamicAnkle --time_rotation new --resume <path_to_latest.pt>
```


