# DeepLearningTennis3DUplift
A Deep Learning Python project for 3D tennis analytics based on 2D broadcasting videos, featuring modules for data processing, model training, and camera parameter verification. Includes tools for uplifting 2D to 3D keypoints and analyzing tennis rallies.

# Usage via torch.hub
This repository provides a PyTorch Hub interface for easy loading and inference with the Tennis 3D Uplifting model.

1. Install Requirements
First, create and activate a virtual environment, then install the required packages according to requirements.txt

2. Adapt the Path to Weights
Edit the file paths.py and set the correct path to your model weights. weights_path should be the base path that contains four subfoldders:
   - trajectories_dynamic
   - trajectories_dynamicAnkle
   - rallies_dynamic
   - rallies_dynamicAnkle
  
    The subfoulders each contain the model.pt file that includes the pre-trained weights.

1. Using torch.hub
You can load the model directly in your Python code using torch.hub:
```
import torch

model = torch.hub.load(
    'https://github.com/xSpaceCoder/DeepLearningTennis3DUplift', 
    'tennis_uplifting',
    input_type='rallies',
    mode='dynamicAnkle'
)
```
There are two input_types: trajectories and rallies
There are two modes: dynamic and dynamicAnkle

If you want to model to predict the 3D position and the spin use:
```
model.predict(ball_coords, court_courts, times)
```
The model expects 2D ball coordinates, court keypoints, and timestamps as input.
See the class UpliftingModel in interface_rallies_dynamicAnkle.py for details on input/output formats.

For further information follow: https://docs.pytorch.org/docs/stable/hub.html#loading-models-from-hub
