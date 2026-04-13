# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Optional
import json
import os

from uplifting.model import get_model
from uplifting.transformations import get_transforms, UnNormalizeImgCoords
from uplifting.data import (
    SynthTennisDataset,
    RealInferenceDataset as TrajectoriesNoTosses,
)
from paths import (
    weights_path,
    data_path_real_trajectories_no_tosses,
    data_path_real_trajectories_with_tosses,
)
from uplifting_rallies.data import RealInferenceDataset as TrajectoriesWithTosses
from uplifting.helper import WIDTH, HEIGHT
from uplifting.helper import world2cam, cam2img
from helper import court_connections, court_points

csv_file_path = "/home/mmc-user/tennisapplication/trajectory_clip_mapping.csv"
DATA_PATH = "/data/goeppeal/data_tennis"
IMAGE_PATH = "/data/goeppeal/data_tennis/tracknet/Dataset"
KEYPOINTS_CSV = "/home/mmc-user/tennisapplication/2D_pts_correctedPoints.json"

# final models
t_dynamic_model = weights_path +    "trajectories_dynamic/model_trajectory.pt"
t_dynamic_ankle_model = weights_path + "trajectories_dynamicAnkle/model_trajectory.pt"
r_dynamic_model = weights_path + "rallies_dynamic/model_trajectory.pt"
r_dynamic_ankle_model = weights_path + "rallies_dynamicAnkle/model_trajectory.pt"

MODEL_PATH = r_dynamic_model
data_format = "withToss" if "rallies" in MODEL_PATH else "noToss"

match MODEL_PATH:
    case _ if "rallies_mode-dynamicAnkle" in MODEL_PATH:
        trajectory_color = "#2556AC"
        RealInferenceDataset = TrajectoriesWithTosses
        DATA_PATH_REAL_TRAJECTORIES = data_path_real_trajectories_with_tosses
    case _ if "rallies_mode-dynamic" in MODEL_PATH:
        trajectory_color = "#DC3241"
        RealInferenceDataset = TrajectoriesWithTosses
        DATA_PATH_REAL_TRAJECTORIES = data_path_real_trajectories_with_tosses
    case _ if "trajectories_mode-dynamicAnkle" in MODEL_PATH:
        trajectory_color = "#00AE5B"
        RealInferenceDataset = TrajectoriesNoTosses
        DATA_PATH_REAL_TRAJECTORIES = data_path_real_trajectories_no_tosses
    case _ if "trajectories_mode-dynamic" in MODEL_PATH:
        trajectory_color = "#A33CA2"
        RealInferenceDataset = TrajectoriesNoTosses
        DATA_PATH_REAL_TRAJECTORIES = data_path_real_trajectories_no_tosses


WIDTH = 1920
HEIGHT = 1080


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_connections = [
    (0, 2),
    (2, 4),
    (4, 5),
    (5, 3),
    (3, 1),
    (1, 0),  # outer rectangle
    (2, 3),
    (3, 9),
    (9, 8),
    (8, 7),
    (7, 2),
    (10, 12),
    (11, 14),
    (13, 15),
    (16, 17),
    (17, 19),
    (19, 18),
    (18, 16),
]


def load_model(model_path):
    """
    Load the uplifting model from the given path.
    Args:
        model_path (str): Path to the saved model.
    Returns:
        model (torch.nn.Module): Loaded uplifting model.
        transform (callable): Transformation function for input images.
        transform_mode (str): Spin transformation mode used during training ('global' or 'local').
    """
    loaded_dict = torch.load(model_path, weights_only=False)
    model_name = loaded_dict["additional_info"]["name"]
    model_size = loaded_dict["additional_info"]["size"]
    identifier = loaded_dict["identifier"]
    tabletoken_mode = loaded_dict["additional_info"]["tabletoken_mode"]
    time_rotation = loaded_dict["additional_info"]["time_rotation"]
    transform_mode = loaded_dict["additional_info"]["transform_mode"]
    randdet_prob, randmiss_prob, tablemiss_prob = (
        loaded_dict["additional_info"]["randdet_prob"],
        loaded_dict["additional_info"]["randmiss_prob"],
        loaded_dict["additional_info"]["tablemiss_prob"],
    )
    uplifting_model = get_model(
        model_name,
        size=model_size,
        mode=tabletoken_mode,
        time_rotation=time_rotation,
    )

    # Load state dict with strict=False to handle missing ankle embedding parameters
    missing_keys, unexpected_keys = uplifting_model.load_state_dict(
        loaded_dict["model_state_dict"], strict=False
    )

    uplifting_model.eval()
    print(
        f"Loaded Uplifting model: {model_name} with size {model_size}, tabletoken_mode: {tabletoken_mode}, time_rotation: {time_rotation}, transform_mode: {transform_mode}"
    )
    print(
        f"Noise settings during training - randdet_prob: {randdet_prob}, randmiss_prob: {randmiss_prob}, tablemiss_prob: {tablemiss_prob}"
    )
    return uplifting_model, transform_mode


def metric_pos2D_fn(pred_2D, gt_2D, mask):
    T, __ = pred_2D.shape
    for t in range(T):
        if mask[t] == 0:
            pred_2D[t], gt_2D[t] = np.array([0, 0]), np.array(
                [0, 0]
            )  # set to same value such that difference is 0
    return np.sum(np.sqrt(np.sum((pred_2D - gt_2D) ** 2, axis=-1))) / np.sum(mask)


def get_trajectory(
    model_path,
):
    denorm = UnNormalizeImgCoords()
    # Model
    model, transform_mode = load_model(model_path)
    model = model.to(device)
    model.eval()
    metrics_2D = []

    # Dataset
    val_transforms = get_transforms(None, "val")
    valset = SynthTennisDataset("test", transforms=val_transforms)

    # groundstroke, serve, volley, smash, lob, short
    # direction far_to_close
    trajectories = [
        1,
        0,
        4,
        8,
        17,
        2,
        46,
        145,
        11,
    ]  # groundstroke, serve, volley, smash, lob, short

    (
        gt_trajectory_3D,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        ankle_v,
        ankle_3D,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for traj_id in range(1000):
        (
            r_img,
            court_img,
            mask,
            r_world,
            rotation,
            times,
            bounces,
            ankle_v,
            ankle_pos,
            Mint,
            Mext,
        ) = valset[traj_id]

        r_img = r_img.unsqueeze(0).to(device)
        court_img = court_img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        times = times.unsqueeze(0).to(device)
        r_world = r_world.unsqueeze(0).to(device)
        ankle_v = ankle_v.unsqueeze(0).to(device)

        with torch.no_grad():
            _, pred_position = model(
                r_img, court_img, mask, times, ankle_v
            )  # should return position (not normalized)

        pred_position = pred_position.squeeze(0).cpu().numpy()
        # Ensure pred_position only contains valid points based on the mask
        pred_position = pred_position[: int(mask.squeeze(0).cpu().numpy().sum())]

        # Reproject predicted points
        M_ext_np = Mext.numpy()
        M_int_np = Mint.numpy()
        projected_pred_img = cam2img(world2cam(pred_position, M_ext_np), M_int_np)

        # Denormalize original images and court points
        stuff = denorm(
            {
                "r_img": r_img.squeeze(0).cpu().numpy(),
                "court_img": court_img.squeeze(0).cpu().numpy(),
                "ankle_v": ankle_v.squeeze(0).cpu().numpy(),
            }
        )
        original_r_img_coords = stuff["r_img"]
        original_court_img_coords = stuff["court_img"]
        original_ankle_v = stuff["ankle_v"]
        len_trajectory = projected_pred_img.shape[0]

        metric_2d = metric_pos2D_fn(
            projected_pred_img,
            original_r_img_coords[:len_trajectory],
            mask.squeeze(0).cpu().numpy(),
        )

        metrics_2D.append(metric_2d)

        len_trajectory = np.sum(mask.squeeze(0).cpu().numpy()).astype(int)
        gt_trajectory_3D.append(r_world.squeeze(0).cpu().numpy()[:len_trajectory])
        gt_img_coords.append(original_r_img_coords[:len_trajectory])
        pred_trajectory_3D.append(pred_position)
        pred_img_coords.append(projected_pred_img)
        court_img_coords.append(original_court_img_coords[:, :2])
        ankle_v.append(original_ankle_v)
        ankle_3D.append(ankle_pos.cpu().numpy())

    return (
        metrics_2D,
        gt_trajectory_3D,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        ankle_v,
        ankle_3D,
    )


def get_worst_real_trajectory(
    model_path,
):

    denorm = UnNormalizeImgCoords()
    # Model
    model, transform_mode = load_model(model_path)
    model = model.to(device)
    model.eval()

    # Dataset
    val_transforms = get_transforms(None, "val")
    valset = RealInferenceDataset("test", transforms=val_transforms)

    (
        metrics_2D,
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        ankle_v_coords,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for traj_id in range(len(valset)):
        if data_format == "noToss":
            r_img, court_img, mask, times, hits, Mint, Mext, _, ankle_v = valset[
                traj_id
            ]
        else:
            r_img, court_img, mask, times, hits, Mint, Mext, _, _, ankle_v = valset[
                traj_id
            ]

        r_img = r_img.unsqueeze(0).to(device)
        court_img = court_img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        times = times.unsqueeze(0).to(device)
        ankle_v = ankle_v.unsqueeze(0).to(device)

        with torch.no_grad():
            _, pred_position = model(
                r_img, court_img, mask, times, ankle_v
            )  # should return position (not normalized) , ankle_x_img

        pred_position = pred_position.squeeze(0).cpu().numpy()

        # Reproject predicted points
        M_ext_np = Mext.numpy()
        M_int_np = Mint.numpy()
        projected_pred_img = cam2img(world2cam(pred_position, M_ext_np), M_int_np)

        # Denormalize original images and court points
        stuff = denorm(
            {
                "r_img": r_img.squeeze(0).cpu().numpy(),
                "court_img": court_img.squeeze(0).cpu().numpy(),
                "ankle_v": ankle_v.squeeze(0).cpu().numpy(),
            }
        )
        original_r_img_coords = stuff["r_img"]
        original_court_img_coords = stuff["court_img"]
        original_ankle_v = stuff["ankle_v"]

        metric_2d = metric_pos2D_fn(
            projected_pred_img, original_r_img_coords, mask.squeeze(0).cpu().numpy()
        )

        if hits[0] != -1:  # metric_2d < 20.0 or metric_2d > 20.0:
            continue

        metrics_2D.append(metric_2d)

        # Ensure pred_position only contains valid points based on the mask
        pred_position = pred_position[: int(mask.squeeze(0).cpu().numpy().sum())]

        len_trajectory = np.sum(mask.squeeze(0).cpu().numpy()).astype(int)
        trajectory_ids.append(traj_id)
        gt_img_coords.append(original_r_img_coords[:len_trajectory])
        pred_trajectory_3D.append(pred_position)
        pred_img_coords.append(projected_pred_img[:len_trajectory])
        court_img_coords.append(original_court_img_coords[:, :2])
        ankle_v_coords.append(original_ankle_v)

    return (
        metrics_2D,
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        ankle_v_coords,
    )


def get_specific_real_trajectory(
    model_path,
    trajectory_ids,
):
    denorm = UnNormalizeImgCoords()
    # Model
    model, transform_mode = load_model(model_path)
    model = model.to(device)
    model.eval()

    # Dataset
    val_transforms = get_transforms(None, "val")
    valset = RealInferenceDataset("test", transforms=val_transforms)

    (
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        ankle_v_coord,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )
    for traj_id in trajectory_ids:
        if data_format == "noToss":
            r_img, court_img, mask, times, hits, Mint, Mext, _, ankle_v = valset[
                traj_id
            ]
        else:
            r_img, court_img, mask, times, hits, Mint, Mext, _, _, ankle_v = valset[
                traj_id
            ]

        r_img = r_img.unsqueeze(0).to(device)
        court_img = court_img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        times = times.unsqueeze(0).to(device)
        ankle_v = ankle_v.unsqueeze(0).to(device)

        with torch.no_grad():
            _, pred_position = model(
                r_img, court_img, mask, times, ankle_v
            )  # should return position (not normalized)

        pred_position = pred_position.squeeze(0).cpu().numpy()

        # Reproject predicted points
        M_ext_np = Mext.numpy()
        M_int_np = Mint.numpy()
        projected_pred_img = cam2img(world2cam(pred_position, M_ext_np), M_int_np)

        # Denormalize original images and court points
        stuff = denorm(
            {
                "r_img": r_img.squeeze(0).cpu().numpy(),
                "court_img": court_img.squeeze(0).cpu().numpy(),
                "ankle_v": ankle_v.squeeze(0).cpu().numpy(),
            }
        )
        original_r_img_coords = stuff["r_img"]
        original_court_img_coords = stuff["court_img"]

        # Ensure pred_position only contains valid points based on the mask
        pred_position = pred_position[: int(mask.squeeze(0).cpu().numpy().sum())]

        len_trajectory = np.sum(mask.squeeze(0).cpu().numpy()).astype(int)
        gt_img_coords.append(original_r_img_coords[:len_trajectory])
        pred_trajectory_3D.append(pred_position)
        pred_img_coords.append(projected_pred_img[:len_trajectory])
        court_img_coords.append(original_court_img_coords[:, :2])
        ankle_v_coord.append(stuff["ankle_v"])

    return (
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        ankle_v_coord,
    )


def _model_plot_color(model_path: str) -> str:
    if "rallies_mode-dynamicAnkle" in model_path:
        return "#2556AC"
    if "rallies_mode-dynamic" in model_path:
        return "#DC3241"
    if "trajectories_mode-dynamicAnkle" in model_path:
        return "#00AE5B"
    if "trajectories_mode-dynamic" in model_path:
        return "#A33CA2"
    raise ValueError(f"Unsupported model path: {model_path}")


def _infer_single_model_trajectory(model_path: str, trajectory_id: int) -> np.ndarray:
    """Infer one trajectory for one model, using the correct dataset type from its mode."""
    model, _ = load_model(model_path)
    model = model.to(device)
    model.eval()

    val_transforms = get_transforms(None, "val")
    if "rallies_mode-" in model_path:
        valset = TrajectoriesWithTosses("test", transforms=val_transforms)
        sample = valset[trajectory_id]
        r_img, court_img, mask, times, _, _, _, _, _, ankle_v = sample
    elif "trajectories_mode-" in model_path:
        valset = TrajectoriesNoTosses("test", transforms=val_transforms)
        sample = valset[trajectory_id]
        r_img, court_img, mask, times, _, _, _, _, ankle_v = sample
    else:
        raise ValueError(
            "model_path must be one of rallies_mode-* or trajectories_mode-* models."
        )

    if trajectory_id < 0 or trajectory_id >= len(valset):
        raise ValueError(
            f"trajectory_id {trajectory_id} out of range. Dataset has {len(valset)} trajectories."
        )

    r_img = r_img.unsqueeze(0).to(device)
    court_img = court_img.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    times = times.unsqueeze(0).to(device)
    ankle_v = ankle_v.unsqueeze(0).to(device)

    with torch.no_grad():
        _, pred_position = model(r_img, court_img, mask, times, ankle_v)

    pred_position = pred_position.squeeze(0).cpu().numpy()
    valid_len = int(mask.squeeze(0).cpu().numpy().sum())
    return pred_position[:valid_len]


def compare_rallies_models_sideview(
    trajectory_id: int,
    include_trajectory_models: bool = False,
    separate_subplots: bool = False,
):
    """
    Compare one trajectory in a single xz side-view plot.

    Always includes:
    - r_dynamic_model (red)
    - r_dynamic_ankle_model (blue)

    Optionally includes when include_trajectory_models=True:
    - t_dynamic_model (purple)
    - t_dynamic_ankle_model (green)

    Layout:
    - separate_subplots=False: all selected models in one image
    - separate_subplots=True: one subplot per model with shared x/y axes
    """
    model_paths = [r_dynamic_model, r_dynamic_ankle_model]
    if include_trajectory_models:
        model_paths.extend([t_dynamic_model, t_dynamic_ankle_model])

    model_predictions = []
    for model_path in model_paths:
        prediction = _infer_single_model_trajectory(model_path, trajectory_id)
        model_predictions.append((model_path, prediction))

    labels = []
    for model_path, _ in model_predictions:
        if "rallies_mode-dynamicAnkle" in model_path:
            labels.append("r_dynamic_ankle_model")
        elif "rallies_mode-dynamic" in model_path:
            labels.append("r_dynamic_model")
        elif "trajectories_mode-dynamicAnkle" in model_path:
            labels.append("t_dynamic_ankle_model")
        elif "trajectories_mode-dynamic" in model_path:
            labels.append("t_dynamic_model")
        else:
            labels.append("model")
    num_id = trajectory_id + 170
    if not separate_subplots:
        fig, ax = plt.subplots()
        ax.set_title(f"xz Graph of trajectory {num_id}")
        ax.grid(visible="true", which="minor")
        ax.set_aspect("equal")
        ax.margins(0.2, 0.2)

        for connection in court_connections:
            ax.plot(
                court_points[connection, 0],
                court_points[connection, 2],
                "k",
            )

        for (model_path, pred), label in zip(model_predictions, labels):
            color = _model_plot_color(model_path)

            ax.plot(
                pred[:, 0],
                pred[:, 2],
                color=color,
                linewidth=2,
                label=label,
            )
            ax.scatter(
                pred[0, 0],
                pred[0, 2],
                color="black",
                s=24,
                zorder=3,
            )
            ax.set_xlim(-14, 14)
            ax.set_ylim(-0.2, 4)

        ax.grid(True)
        # ax.legend()
        plt.show()
        return

    x_min = min(np.min(pred[:, 0]) for _, pred in model_predictions)
    x_max = max(np.max(pred[:, 0]) for _, pred in model_predictions)
    z_min = min(np.min(pred[:, 2]) for _, pred in model_predictions)
    z_max = max(np.max(pred[:, 2]) for _, pred in model_predictions)
    num_id = trajectory_id + 170

    for (model_path, pred), label in zip(model_predictions, labels):
        fig, ax = plt.subplots()
        color = _model_plot_color(model_path)
        ax.set_title(f"xz Graph of trajectory {num_id} ({label})")
        ax.grid(visible="true", which="minor")
        ax.set_aspect("equal")
        ax.margins(0.2, 0.2)
        ax.set_xlim(-14, 14)
        ax.set_ylim(-0.2, z_max)

        for connection in court_connections:
            ax.plot(
                court_points[connection, 0],
                court_points[connection, 2],
                "k",
            )

        ax.plot(
            pred[:, 0],
            pred[:, 2],
            color=color,
            linewidth=2,
        )
        ax.scatter(
            pred[0, 0],
            pred[0, 2],
            color=color,
            s=24,
            zorder=3,
        )
        ax.grid(True)
        plt.tight_layout()
        plt.show()


def get_Game_Clip_from_trajectoryID(trajectory_id: int):
    # Build path to trajectory folder
    trajectory_folder = os.path.join(
        DATA_PATH_REAL_TRAJECTORIES, f"trajectory_{trajectory_id:04d}"
    )
    info_path = os.path.join(trajectory_folder, "info.json")

    # Handle edge cases
    if trajectory_id < 0 or not os.path.exists(info_path):
        return None

    # Read info.json
    with open(info_path, "r") as f:
        info = json.load(f)

    game = info["game"]
    clip = info["clip"]

    game = "game" + str(game)
    clip = "Clip" + str(clip)

    print(f"Trajectory ID {trajectory_id} corresponds to Game: {game}, Clip: {clip}")
    return game, clip


def get_background_image(trajectory_id: int):

    game, clip = get_Game_Clip_from_trajectoryID(trajectory_id)
    image_file_path = f"{IMAGE_PATH}/{game}/{clip}/0002.jpg"

    bg_image = mpimg.imread(image_file_path)

    return bg_image


def get_annotated_keypoints(trajectory_id: int):
    game, clip = get_Game_Clip_from_trajectoryID(trajectory_id)

    if game is None or clip is None:
        return np.array([])

    # Load JSON data
    with open(KEYPOINTS_CSV, "r") as f:
        keypoints_data = json.load(f)

    # Construct search string
    search_string = f"{game}_{clip}"

    # Find matching entry
    for entry in keypoints_data:
        if entry["name"] == search_string:
            keypoints = entry["2D_keypoints"]
            # Scale keypoints from 720x1280 to 1080x1920 resolution
            keypoints_array = np.array(keypoints)
            if keypoints_array.size > 0:
                # Scale factor: width 1920/1280=1.5, height 1080/720=1.5
                keypoints_array = keypoints_array * 1.5
            return keypoints_array

    # Return empty array if not found
    return np.array([])


def plot_prediction_on_image(
    traj_idx,
    num_traj,
    gt_trajectory_2D,
    pred_trajectory_2D,
    court_img,
    ankle_img: Optional[np.ndarray] = None,
):
    fig, ax = plt.subplots()
    plt.gca().invert_yaxis()
    # ax.set_title(f"2D Trajectory {num_traj} Prediction (cyan) vs Ground Truth (black)")
    ax.set_xlim(0, WIDTH),
    ax.set_ylim(HEIGHT, 0)

    background_image = get_background_image(num_traj)

    # Display background image if provided
    if background_image is not None:
        # Display background image
        ax.imshow(
            background_image,
            extent=[0, WIDTH, HEIGHT, 0],
            aspect="auto",
            zorder=0,
            alpha=0.7,
        )

    ax.plot(
        gt_trajectory_2D[traj_idx][:, 0],
        gt_trajectory_2D[traj_idx][:, 1],
        color="black",
        linewidth=2,
        label="GT",
        zorder=2,
    )
    """ ax.plot(
        pred_trajectory_2D[traj_idx][:, 0],
        pred_trajectory_2D[traj_idx][:, 1],
        color=trajectory_color,
        linewidth=2,
        label="Pred",
        zorder=2,
    ) 
    if ankle_img is not None:
        ax.axhline(y=ankle_img[traj_idx], color="#F87825", label="ankle v", zorder=3)
    """
    annotated_pts = get_annotated_keypoints(num_traj)
    if annotated_pts.shape[0] > 0:
        ax.scatter(
            annotated_pts[:, 0],
            annotated_pts[:, 1],
            s=3,
            color="#F87825",
            label="court KP",
            zorder=3,
        )

    # Draw court lines using connections
    for connection in img_connections:
        pt1, pt2 = connection
        if pt1 < len(court_img[traj_idx]) and pt2 < len(court_img[traj_idx]):
            ax.plot(
                [court_img[traj_idx][pt1, 0], court_img[traj_idx][pt2, 0]],
                [court_img[traj_idx][pt1, 1], court_img[traj_idx][pt2, 1]],
                "k-",
                linewidth=2,
                zorder=1,
            )

    ax.legend()
    plt.show()


def visualize_trajectories(
    pred_trajectory_3D,
    gt_trajectory_2D,
    pred_trajectory_2D,
    court_img,
    gt_trajectory_3D: Optional[list] = None,
    trajectory_ids: Optional[list] = None,
    ankle_img: Optional[np.ndarray] = None,
    ankle_pos: Optional[np.ndarray] = None,
):
    """
    Visualize predicted vs ground truth trajectories.
    Args:
        pred_file (str): Path to the prediction file.
        gt_file (str): Path to the ground truth file.
        output_file (str): Path to save the output visualization.
    """

    for traj_idx in range(len(pred_trajectory_3D)):
        pred_positions = np.array(pred_trajectory_3D[traj_idx])  # shape: (T, 3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-8, 8)
        ax.set_zlim(0.0, 3.0)

        num_traj = (
            trajectory_ids[traj_idx] + 170 if trajectory_ids is not None else traj_idx
        )

        if gt_trajectory_3D is not None:
            gt_positions = np.array(gt_trajectory_3D[traj_idx])  # shape: (T, 3)
            ax.set_title(f"3D Trajectory {num_traj} Prediction vs Ground Truth (black)")
            ax.plot(
                gt_positions[:, 0],
                gt_positions[:, 1],
                gt_positions[:, 2],
                color="black",
            )
            ax.scatter(
                gt_positions[0, 0],
                gt_positions[0, 1],
                gt_positions[0, 2],
                color="black",
                label="Start",
                zorder=3,
            )

        ax.plot(
            pred_positions[:, 0],
            pred_positions[:, 1],
            pred_positions[:, 2],
            color=trajectory_color,
        )
        ax.scatter(
            pred_positions[0, 0],
            pred_positions[0, 1],
            pred_positions[0, 2],
            color="black",
            label="Start",
            zorder=3,
        )
        for connection in court_connections:
            ax.plot(
                court_points[connection, 0],
                court_points[connection, 1],
                court_points[connection, 2],
                "k",
            )

        if ankle_pos is not None:
            ax.scatter(
                ankle_pos[traj_idx][0],
                ankle_pos[traj_idx][1],
                ankle_pos[traj_idx][2],
                color="red",
                label="Ankle Position",
            )

        ax.legend()
        plt.show()

        plot_prediction_on_image(
            traj_idx,
            num_traj,
            gt_trajectory_2D,
            pred_trajectory_2D,
            court_img,
            ankle_img=ankle_img if ankle_img is not None else None,
        )

        fig, ax = plt.subplots()
        ax.set_title(f"xz Graph of trajectory {num_traj}")
        ax.grid(visible="true", which="minor")
        ax.set_aspect("equal")
        ax.margins(0.2, 0.2)
        ax.scatter(
            pred_positions[0, 0],
            pred_positions[0, 2],
            color="black",
            label="Start",
            zorder=3,
        )
        if ankle_pos is not None:
            ax.scatter(ankle_pos[traj_idx][0], ankle_pos[traj_idx][2], color="red")
        for connection in court_connections:
            ax.plot(
                court_points[connection, 0],
                court_points[connection, 2],
                "k",
            )
        ax.grid(True)
        ax.plot(pred_positions[:, 0], pred_positions[:, 2], color=trajectory_color)
        plt.show()

        """ 
        # xy plane
        fig, ax = plt.subplots()
        ax.set(xlim=(-15, 15), ylim=(-8, 8))
        ax.scatter(gt_positions[0, 0], gt_positions[0, 1], color="red", label="Start")
        ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], color="green", label="End")
        for connection in court_connections:
            ax.plot(
                court_points[connection, 0],
                court_points[connection, 1],
                "k",
            )
        ax.grid(True)
        ax.plot(gt_positions[:, 0], gt_positions[:, 1])
        plt.show()

        # xz plane
        fig, ax = plt.subplots()
        ax.set(xlim=(-15, 15), ylim=(0, 3))
        ax.scatter(gt_positions[0, 0], gt_positions[0, 2], color="red", label="Start")
        ax.scatter(gt_positions[-1, 0], gt_positions[-1, 2], color="green", label="End")
        for connection in court_connections:
            ax.plot(
                court_points[connection, 0],
                court_points[connection, 2],
                "k",
            )
        ax.grid(True)
        ax.plot(gt_positions[:, 0], gt_positions[:, 2])
        plt.show() """


def create_histogram(metrics):
    # Create histogram of metrics_2D distribution
    metrics_array = np.array(metrics)
    n_bins = 9
    bins = np.linspace(0, 45, n_bins + 1)
    counts, bin_edges = np.histogram(metrics_array, bins=bins)

    # Create column chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins)]

    ax.bar(range(n_bins), counts, tick_label=bin_labels, color="#2556AC")
    ax.set_ylabel("Count")
    # ax.set_title("Distribution of 2D Reprojection Error Metrics")
    ax.grid(True, alpha=0.3)

    # Add count labels on top of each bar
    for i, count in enumerate(counts):
        ax.text(i, count, str(int(count)), ha="center", va="bottom", fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def calcualte_metric(metrics):
    metric_2D = np.sum(np.array(metrics)) / len(metrics)
    print(
        f"\nAverage 2D Reprojection Error over {len(metrics)} trajectories: {metric_2D:.3f} pixels"
    )

    metric_2D_normed = metric_2D / np.sqrt(WIDTH**2 + HEIGHT**2)
    print(f"Normalized Average 2D Reprojection Error: {metric_2D_normed:.3f}\n")


if __name__ == "__main__":
    """(
        metrics,
        gt_trajectory_3D,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img,
        ankle_img,
        ankle_pos,
    ) = get_trajectory(MODEL_PATH)
    visualize_trajectories(
        pred_trajectory_3D[:10],
        gt_img_coords,
        pred_img_coords,
        court_img,
        gt_trajectory_3D=gt_trajectory_3D,
        ankle_img=ankle_img,
        ankle_pos=ankle_pos,
    )"""

    (
        metrics,
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img,
        ankle_v,
    ) = get_worst_real_trajectory(MODEL_PATH)

    # calcualte_metric(metrics)

    # create_histogram(metrics)

    compare_rallies_models_sideview(88)

    """ (
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img,
        ankle_v,
    ) = get_specific_real_trajectory(
        MODEL_PATH,
        [156],
    ) """

    if "dynamicAnkle" in MODEL_PATH:
        visualize_trajectories(
            pred_trajectory_3D[:10],
            gt_img_coords,
            pred_img_coords,
            court_img,
            trajectory_ids=trajectory_ids,
            ankle_img=ankle_v,
        )
    else:
        visualize_trajectories(
            pred_trajectory_3D[:10],
            gt_img_coords,
            pred_img_coords,
            court_img,
            trajectory_ids=trajectory_ids,
        )


# %%
