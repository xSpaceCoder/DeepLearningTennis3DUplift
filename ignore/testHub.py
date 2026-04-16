import torch

# Replace with your actual path
model = torch.hub.load(
    "/home/mmc-user/tennisuplifting/DeepLearningTennis3DUplift",
    "tennis_uplifting",
    input_type="rallies",
    mode="dynamic",
    source="local",
)
print(model)
