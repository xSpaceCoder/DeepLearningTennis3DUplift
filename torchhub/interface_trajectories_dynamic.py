import torch
import numpy as np
import os

# uplifting
from torchhub.helper import load_model as load_uplifting_model
from uplifting.helper import transform_rotationaxes

import paths


class UpliftingModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Architecture
        model_path = os.path.join(
            paths.weights_path, "trajectories_dynamic", "model.pt"
        )
        self.model, self.transform, self.transform_mode = load_uplifting_model(
            model_path=model_path
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, ball_coords, court_courts, times):
        """
        Input:
            - ball_coords: torch tensor (N, 2) of 2D ball coordinates in pixel space
            - court_courts: torch tensor (16, 3) of 2D table keypoints in pixel space -> (x, y, visibility)
            - times: torch tensor (N,) of time stamps in seconds -> they have to match the ball coordinates
        Returns:
            - pred_spin: torch tensor (n, 3) of predicted spin vector in local coordinate system
            - pred_pos_3d: torch tensor (N, 3) of predicted 3D ball positions in world coordinates
        """
        # Prepare inputs
        data = {
            "r_img": ball_coords,
            "court_img": court_courts,
        }
        data = self.transform(data)
        ball_coords, court_courts = data["r_img"], data["court_img"]

        mask = np.zeros((ball_coords.shape[0] + 1,), dtype=np.float32)
        mask[:-1] = 1.0  # True for all ball points

        return self.predict_without_normalization(
            ball_coords, court_courts, torch.tensor(mask).to(self.device), times
        )

    def predict_without_normalization(self, ball_coords, court_courts, mask, times):
        """Assume coords are already normalized
        Input:
            - ball_coords: torch tensor (N, 2) of 2D ball coordinates in pixel space
            - court_courts: torch tensor (16, 3) of 2D court keypoints in pixel space -> (x, y, visibility)
            - mask: torch tensor (N,) of mask for ball coordinates
            - times: torch tensor (N,) of time stamps in seconds -> they have to match the ball coordinates
        Returns:
            - pred_spin: torch tensor (N, 3) of predicted spin vector in local coordinate system
            - pred_pos: torch tensor (N, 3) of predicted 3D ball positions in world coordinates
        """
        ball_coords, court_courts, mask, times = (
            ball_coords.to(self.device),
            court_courts.to(self.device),
            mask.to(self.device),
            times.to(self.device),
        )

        with torch.no_grad():
            pred_rotation, pred_position = self.model(
                ball_coords, court_courts, mask, times
            )

        # transform prediction into local coordinate system
        if self.transform_mode == "global":
            pred_rotation_local = transform_rotationaxes(
                pred_rotation, pred_position.clone()
            )
        else:
            pred_rotation_local = pred_rotation

        # remove padding
        T_prime = int(mask.sum().item())
        pred_position = pred_position[:, :T_prime, :].cpu().numpy()

        return pred_rotation_local.squeeze(0), pred_position.squeeze(0)


if __name__ == "__main__":
    # Simple test to check if weights can be loaded
    uplifting_model = UpliftingModel()
    print("All models loaded successfully.")
