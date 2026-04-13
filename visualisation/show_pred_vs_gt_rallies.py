# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Optional
import json
import os

from uplifting_rallies.model import get_model
from uplifting_rallies.transformations import get_transforms, UnNormalizeImgCoords
from uplifting_rallies.data import (
    StitchedRallyDataset,
    RealInferenceDataset,
    RealInferenceRalliesDataset,
)
from uplifting_rallies.helper import WIDTH, HEIGHT, BALL_VISIBLE, transform_rotationaxes
from uplifting_rallies.helper import world2cam, cam2img
from helper import court_connections, court_points
from paths import data_path_real_rallies

csv_file_path = "/home/mmc-user/tennisapplication/trajectory_clip_mapping.csv"
IMAGE_PATH = "/data/goeppeal/data_tennis/tracknet/Dataset"
KEYPOINTS_CSV = "/home/mmc-user/tennisapplication/2D_pts_correctedPoints.json"
r_dynamic_model = "/data/goeppeal/logs_tennis/uplifting/final_models/rallies/rallies_mode-dynamic_tokenrotation-new_03052026-104317/model_trajectory.pt"
r_dynamic_ankle_model = "/data/goeppeal/logs_tennis/uplifting/final_models/rallies/dynamicAnkle/rallies_mode-dynamicAnkle_tokenrotation-new_03102026-113735_Part2/model_trajectory.pt"

MODEL_PATH = r_dynamic_ankle_model

match MODEL_PATH:
    case _ if "rallies_mode-dynamicAnkle" in MODEL_PATH:
        trajectory_color = "#2556AC"
        ankle = True
    case _ if "rallies_mode-dynamic" in MODEL_PATH:
        trajectory_color = "#DC3241"
        ankle = False
    case _:
        raise ValueError(
            "MODEL_PATH must point to either r_dynamic_model or r_dynamic_ankle_model."
        )

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
    (9, 7),
    (7, 2),
    (10, 12),
    (11, 14),
    (13, 15),
]


def _sort_visible_first_like_model(r_img, mask, times):
    """Sort frames as in model.FirstStage when interpolate_missing=False."""
    visibilities = r_img[:, :, 2]
    keep_mask = visibilities == BALL_VISIBLE
    sorted_keep_mask, sort_indices = torch.sort(
        keep_mask.int(), dim=1, descending=True, stable=True
    )
    sorted_keep_mask = sorted_keep_mask.bool()

    b, t, d = r_img.shape
    sort_indices_d = sort_indices.unsqueeze(-1).expand(b, t, d)
    r_img = torch.gather(r_img, dim=1, index=sort_indices_d)
    times = torch.gather(times, dim=1, index=sort_indices)
    mask = torch.gather(mask, dim=1, index=sort_indices)
    mask = torch.where(sorted_keep_mask, mask, torch.zeros_like(mask))
    return r_img, mask, times


def load_model(model_path):
    """
    Load the uplifting model from the given path.
    Args:
        model_path (str): Path to the saved model.
    Returns:
        model (torch.nn.Module): Loaded uplifting model.
        transform (callable): Transformation function for input images.
    """
    loaded_dict = torch.load(model_path, weights_only=False)
    model_name = loaded_dict["additional_info"]["name"]
    model_size = loaded_dict["additional_info"]["size"]
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
        interpolate_missing=False,
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


def get_trajectory(
    model_path,
):
    denorm = UnNormalizeImgCoords()
    # Model
    model, transform_mode = load_model(model_path)
    model = model.to(device)
    model.eval()

    # Dataset
    val_transforms = get_transforms(None, "val")
    valset = StitchedRallyDataset("val", transforms=val_transforms)

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
        ankle_2D,
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

    for traj_id in trajectories:
        (
            r_img,
            court_img,
            mask,
            r_world,
            rotation,
            times,
            bounces,
            ankle_img,
            ankle_pos,
            Mint,
            Mext,
        ) = valset[traj_id]

        r_img = r_img.unsqueeze(0).to(device)
        court_img = court_img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        times = times.unsqueeze(0).to(device)
        r_world = r_world.unsqueeze(0).to(device)
        ankle_img = ankle_img.unsqueeze(0).to(device)

        with torch.no_grad():
            _, pred_position = model(
                r_img, court_img, mask, times, ankle_img[:, 1:2]
            )  # should return position (not normalized)

        pred_position = pred_position.squeeze(0).cpu().numpy()
        # Ensure pred_position only contains valid points based on the mask
        pred_position = pred_position[: int(mask.squeeze(0).cpu().numpy().sum())]

        # Reproject predicted points
        M_ext_np = Mext.numpy()
        M_int_np = Mint.numpy()
        projected_pred_img = cam2img(world2cam(pred_position, M_ext_np), M_int_np)

        # Denormalize original images and table points
        stuff = denorm(
            {
                "r_img": r_img.squeeze(0).cpu().numpy(),
                "court_img": court_img.squeeze(0).cpu().numpy(),
                "ankle_img": ankle_img.squeeze(0).cpu().numpy(),
            }
        )
        original_r_img_coords = stuff["r_img"]
        original_court_img_coords = stuff["court_img"]
        original_ankle_img = stuff["ankle_img"]

        len_trajectory = np.sum(mask.squeeze(0).cpu().numpy()).astype(int)
        gt_trajectory_3D.append(r_world.squeeze(0).cpu().numpy()[:len_trajectory])
        gt_img_coords.append(original_r_img_coords[:len_trajectory])
        pred_trajectory_3D.append(pred_position)
        pred_img_coords.append(projected_pred_img)
        court_img_coords.append(original_court_img_coords[:, :2])
        ankle_2D.append(original_ankle_img)
        ankle_3D.append(ankle_pos.cpu().numpy())

    return (
        gt_trajectory_3D,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        ankle_2D,
        ankle_3D,
    )


def get_real_trajectory(
    model_path,
):
    denorm = UnNormalizeImgCoords()
    # Model
    model, transform_mode = load_model(model_path)
    model = model.to(device)
    model.eval()

    # Dataset
    val_transforms = get_transforms(None, "val")
    valset = RealInferenceDataset("val", transforms=val_transforms)

    trajectories = [
        1,  # groundstroke, backhand slice
        0,  # serve
        49,
        86,
        115,
        168,
    ]

    (
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        ankle_2D,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )

    for traj_id in trajectories:
        r_img, court_img, mask, times, _, Mint, Mext, _, ankle_img = valset[traj_id]

        r_img = r_img.unsqueeze(0).to(device)
        court_img = court_img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        times = times.unsqueeze(0).to(device)
        ankle_img = ankle_img.to(device)

        with torch.no_grad():
            _, pred_position = model(
                r_img, court_img, mask, times, ankle_img[:, 1:2]
            )  # should return position (not normalized)

        pred_position = pred_position.squeeze(0).cpu().numpy()
        # Ensure pred_position only contains valid points based on the mask
        pred_position = pred_position[: int(mask.squeeze(0).cpu().numpy().sum())]

        # Reproject predicted points
        M_ext_np = Mext.numpy()
        M_int_np = Mint.numpy()
        projected_pred_img = cam2img(world2cam(pred_position, M_ext_np), M_int_np)

        # Denormalize original images and table points
        stuff = denorm(
            {
                "r_img": r_img.squeeze(0).cpu().numpy(),
                "court_img": court_img.squeeze(0).cpu().numpy(),
                "ankle_img": ankle_img.squeeze(0).cpu().numpy(),
            }
        )
        original_r_img_coords = stuff["r_img"]
        original_court_img_coords = stuff["court_img"]
        oiginal_ankle_img = stuff["ankle_img"]

        len_trajectory = np.sum(mask.squeeze(0).cpu().numpy()).astype(int)
        gt_img_coords.append(original_r_img_coords[:len_trajectory])
        pred_trajectory_3D.append(pred_position)
        pred_img_coords.append(projected_pred_img)
        court_img_coords.append(original_court_img_coords[:, :2])
        ankle_2D.append(oiginal_ankle_img)

    return (
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        ankle_2D,
    )


def get_worst_real_trajectory(
    model_path,
):
    def metric_pos2D_fn(pred_2D, gt_2D, mask):
        T, __ = pred_2D.shape
        for t in range(T):
            if mask[t] == 0:
                pred_2D[t], gt_2D[t] = np.array([0, 0]), np.array(
                    [0, 0]
                )  # set to same value such that difference is 0
        return np.sum(np.sqrt(np.sum((pred_2D - gt_2D) ** 2, axis=-1))) / np.sum(mask)

    denorm = UnNormalizeImgCoords()
    # Model
    model, transform_mode = load_model(model_path)
    model = model.to(device)
    model.eval()

    # Dataset
    val_transforms = get_transforms(None, "val")
    valset = RealInferenceDataset("val", transforms=val_transforms)

    (
        metrics_2D,
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        spin_gt,
        spin_pred,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for traj_id in range(len(valset)):
        r_img, court_img, mask, times, hits, Mint, Mext, spin_class, start_serve = (
            valset[traj_id]
        )

        r_img = r_img.unsqueeze(0).to(device)
        court_img = court_img.unsqueeze(0).to(device)
        spin_class = spin_class.cpu().numpy()
        mask = mask.unsqueeze(0).to(device)
        times = times.unsqueeze(0).to(device)
        start_serve = int(start_serve.cpu().numpy())

        with torch.no_grad():
            pred_rotation, pred_position = model(
                r_img, court_img, mask, times
            )  # should return position (not normalized) , ankle_img[:, 1:2]

        pred_rotation = transform_rotationaxes(pred_rotation, pred_position)

        pred_position = pred_position.squeeze(0).cpu().numpy()
        pred_rotation = pred_rotation.squeeze(0).cpu().numpy()

        # Reproject predicted points
        M_ext_np = Mext.numpy()
        M_int_np = Mint.numpy()
        projected_pred_img = cam2img(world2cam(pred_position, M_ext_np), M_int_np)

        # Denormalize original images and table points
        stuff = denorm(
            {
                "r_img": r_img.squeeze(0).cpu().numpy(),
                "court_img": court_img.squeeze(0).cpu().numpy(),
            }
        )
        original_r_img_coords = stuff["r_img"]
        original_court_img_coords = stuff["court_img"]

        metric_2d = metric_pos2D_fn(
            projected_pred_img,
            original_r_img_coords[:, :2],
            mask.squeeze(0).cpu().numpy(),
        )

        if hits[0] == -1:  # metric_2d < 20.0 or metric_2d > 20.0:
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
        spin_gt.append(spin_class)
        spin_pred.append(pred_rotation[start_serve, 1])

    return (
        metrics_2D,
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
        spin_gt,
        spin_pred,
    )


def get_specific_real_trajectory(
    model_path,
    trajectory_id: int,
):
    denorm = UnNormalizeImgCoords()
    # Model
    model, transform_mode = load_model(model_path)
    model = model.to(device)
    model.eval()

    # Dataset
    val_transforms = get_transforms(None, "val")
    valset = RealInferenceDataset("val", transforms=val_transforms)

    (
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )

    r_img, court_img, mask, times, _, Mint, Mext, _, ankle_x_img = valset[trajectory_id]

    r_img = r_img.unsqueeze(0).to(device)
    court_img = court_img.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    times = times.unsqueeze(0).to(device)
    ankle_x_img = ankle_x_img.unsqueeze(0).to(device)

    with torch.no_grad():
        _, pred_position = model(
            r_img, court_img, mask, times, ankle_x_img
        )  # should return position (not normalized)

    pred_position = pred_position.squeeze(0).cpu().numpy()

    # Reproject predicted points
    M_ext_np = Mext.numpy()
    M_int_np = Mint.numpy()
    projected_pred_img = cam2img(world2cam(pred_position, M_ext_np), M_int_np)

    # Denormalize original images and table points
    stuff = denorm(
        {
            "r_img": r_img.squeeze(0).cpu().numpy(),
            "court_img": court_img.squeeze(0).cpu().numpy(),
        }
    )
    original_r_img_coords = stuff["r_img"]
    original_court_img_coords = stuff["court_img"]

    # Ensure pred_position only contains valid points based on the mask
    pred_position = pred_position[: int(mask.squeeze(0).cpu().numpy().sum())]

    len_trajectory = np.sum(mask.squeeze(0).cpu().numpy()).astype(int)
    trajectory_ids.append(trajectory_id)
    gt_img_coords.append(original_r_img_coords[:len_trajectory])
    pred_trajectory_3D.append(pred_position)
    pred_img_coords.append(projected_pred_img[:len_trajectory])
    court_img_coords.append(original_court_img_coords[:, :2])

    return (
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img_coords,
    )


def get_Game_Clip_from_rallyID(rally_id: int):
    trajectory_folder = os.path.join(data_path_real_rallies, f"rally_{rally_id:04d}")
    info_path = os.path.join(trajectory_folder, "info.json")

    if rally_id < 0 or not os.path.exists(info_path):
        return None, None

    with open(info_path, "r") as f:
        info = json.load(f)

    game = "game" + str(info["game"])
    clip = "Clip" + str(info["clip"])

    print(f"Trajectory ID {rally_id} corresponds to Game: {game}, Clip: {clip}")
    return game, clip


def get_background_image(trajectory_id: int):
    game, clip = get_Game_Clip_from_rallyID(trajectory_id)
    if game is None or clip is None:
        return None

    image_file_path = f"{IMAGE_PATH}/{game}/{clip}/0001.jpg"
    if not os.path.exists(image_file_path):
        return None

    return mpimg.imread(image_file_path)


def get_annotated_keypoints(trajectory_id: int):
    game, clip = get_Game_Clip_from_rallyID(trajectory_id)

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
    ax.plot(
        pred_trajectory_2D[traj_idx][:, 0],
        pred_trajectory_2D[traj_idx][:, 1],
        color=trajectory_color,
        linewidth=2,
        label="Pred",
        zorder=2,
    )
    if ankle_img is not None:
        ax.axhline(y=ankle_img[traj_idx], color="#F87825", label="ankle v", zorder=2)

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
    gt_r_img_coords,
    pred_img_coords,
    pred_3D_coords,
    court_img_coords,
    rally_ids,
    new_trajectory_frame_idx,
    single_trajectories: bool = False,
    ankle_img: Optional[np.ndarray] = None,
):
    """
    Visualize predicted vs ground truth trajectories.
    Args:
        pred_file (str): Path to the prediction file.
        gt_file (str): Path to the ground truth file.
        output_file (str): Path to save the output visualization.
    """

    def _plot_3d_and_xz(pred_positions: np.ndarray, num_traj: int, segment_idx=None):
        suffix = f" (segment {segment_idx + 1})" if segment_idx is not None else ""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"3D trajectory {num_traj}{suffix}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-8, 8)
        ax.set_zlim(0.0, 3.0)

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
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()
        ax.set_title(f"xz Graph of trajectory {num_traj}{suffix}")
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
        for connection in court_connections:
            ax.plot(
                court_points[connection, 0],
                court_points[connection, 2],
                "k",
            )
        ax.grid(True)
        ax.plot(pred_positions[:, 0], pred_positions[:, 2], color=trajectory_color)
        plt.show()

    for traj_idx in range(len(rally_ids)):
        num_traj = rally_ids[traj_idx] + 30
        pred_positions = np.array(pred_3D_coords[traj_idx])  # shape: (T, 3)

        if not single_trajectories:
            _plot_3d_and_xz(pred_positions, num_traj)
            plot_prediction_on_image(
                traj_idx,
                num_traj,
                gt_r_img_coords,
                pred_img_coords,
                court_img_coords,
                ankle_img=ankle_img if ankle_img is not None else None,
            )
            continue

        starts_raw = np.array(new_trajectory_frame_idx[traj_idx]).astype(int)
        starts_positive = starts_raw[starts_raw > 0]
        starts_raw -= 1

        trajectory_length = pred_positions.shape[0]
        boundaries = np.concatenate(
            [np.array([0]), starts_positive, np.array([trajectory_length])]
        )
        boundaries = np.unique(boundaries)
        boundaries = boundaries[(boundaries >= 0) & (boundaries <= trajectory_length)]

        for segment_idx in range(len(boundaries) - 1):
            start_frame, end_frame = (
                boundaries[segment_idx],
                boundaries[segment_idx + 1],
            )
            if end_frame <= start_frame:
                continue

            pred_segment = pred_positions[start_frame:end_frame]
            gt_segment = np.array(gt_r_img_coords[traj_idx])[start_frame:end_frame]
            pred_img_segment = np.array(pred_img_coords[traj_idx])[
                start_frame:end_frame
            ]

            if pred_segment.shape[0] == 0:
                continue

            _plot_3d_and_xz(pred_segment, num_traj, segment_idx=segment_idx)

            show_ankle = ankle_img is not None and segment_idx < 2
            image_ankle = [ankle_img[traj_idx]] if show_ankle else None

            plot_prediction_on_image(
                0,
                num_traj,
                [gt_segment],
                [pred_img_segment],
                [court_img_coords[traj_idx]],
                ankle_img=image_ankle,
            )


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

    ax.bar(range(n_bins), counts, tick_label=bin_labels)
    ax.set_xlabel("Metric 2D Range (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of 2D Reprojection Error Metrics")
    ax.grid(True, alpha=0.3)

    # Add count labels on top of each bar
    for i, count in enumerate(counts):
        ax.text(i, count, str(int(count)), ha="center", va="bottom", fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def calcualte_metric(metrics):
    metrics_array = np.array(metrics)
    metric_2D = np.mean(metrics_array)
    metric_2D_std = np.std(metrics_array)
    print(
        f"\nAverage 2D Reprojection Error over {len(metrics)} trajectories: {metric_2D:.3f} +/- {metric_2D_std:.3f} pixels"
    )

    metric_2D_normed = metric_2D / np.sqrt(WIDTH**2 + HEIGHT**2)
    metric_2D_std_normed = metric_2D_std / np.sqrt(WIDTH**2 + HEIGHT**2)
    print(
        f"Normalized Average 2D Reprojection Error: {metric_2D_normed:.3f} +/- {metric_2D_std_normed:.3f}\n"
    )


def evaluate_rallies_2d_error(model_path: str):
    """
    Evaluate the 2D reprojection error over all rallies in RealInferenceRalliesDataset.

    For each rally, calculates the mean per-frame Euclidean distance between
    predicted 2D positions and ground truth 2D detections.

    Args:
        model_path: Path to the model checkpoint.

    Returns:
        rally_errors: List of mean 2D errors for each rally.
        rally_ids: List of rally indices.
    """

    def mean_per_frame_error(
        pred_2D: np.ndarray, gt_2D: np.ndarray, mask: np.ndarray
    ) -> float:
        """Calculate mean Euclidean distance for valid frames."""
        valid_mask = mask.astype(bool)
        if np.sum(valid_mask) == 0:
            return 0.0

        # Compute frame-wise Euclidean distance only for valid frames
        distances = np.sqrt(
            np.sum((pred_2D[valid_mask] - gt_2D[valid_mask, :2]) ** 2, axis=-1)
        )
        return np.mean(distances)

    denorm = UnNormalizeImgCoords()

    # Load model
    model, transform_mode = load_model(model_path)
    model = model.to(device)
    model.eval()

    # Dataset
    val_transforms = get_transforms(None, "val")
    valset = RealInferenceRalliesDataset("test", transforms=val_transforms)

    (
        rally_errors,
        rally_ids,
        gt_r_img_coords,
        court_img_coords,
        pred_3D_coords,
        ankle_v_coords,
        pred_img_coords,
        frames_single_trajectories,
        ankle_v,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    print(f"\nEvaluating 2D error over {len(valset)} rallies...")

    for rally_id in range(len(valset)):
        (
            r_img,
            court_img,
            mask,
            times,
            Mint,
            Mext,
            _,
            _,
            new_trajectory_frame_idx,
            ankle_v,
        ) = valset[rally_id]

        # Move to device
        r_img = r_img.unsqueeze(0).to(device)
        court_img = court_img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        times = times.unsqueeze(0).to(device)
        ankle_v = ankle_v.unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            if ankle:
                _, pred_position = model(r_img, court_img, mask, times, ankle_v)
            else:
                _, pred_position = model(r_img, court_img, mask, times)

        pred_position = pred_position.squeeze(0).cpu().numpy()

        r_img, mask, times = _sort_visible_first_like_model(r_img, mask, times)
        r_img_np = r_img.squeeze(0).cpu().numpy()
        mask_np = mask.squeeze(0).cpu().numpy()

        # Reproject to 2D
        M_ext_np = Mext.numpy()
        M_int_np = Mint.numpy()
        projected_pred_img = cam2img(world2cam(pred_position, M_ext_np), M_int_np)

        # Denormalize ground truth 2D coordinates
        stuff = denorm(
            {
                "r_img": r_img_np,
                "court_img": court_img.squeeze(0).cpu().numpy(),
                "ankle_v": ankle_v.squeeze(0).cpu().numpy(),
            }
        )

        valid_mask = mask_np.astype(bool)

        # Calculate error using model-aligned sorted mask
        error = mean_per_frame_error(projected_pred_img, stuff["r_img"], mask_np)

        gt_valid = stuff["r_img"][valid_mask]
        pred_img_valid = projected_pred_img[valid_mask]
        pred_3d_valid = pred_position[valid_mask]

        rally_errors.append(error)
        rally_ids.append(rally_id)
        gt_r_img_coords.append(gt_valid)
        court_img_coords.append(stuff["court_img"])
        pred_3D_coords.append(pred_3d_valid)
        ankle_v_coords.append(stuff["ankle_v"])
        pred_img_coords.append(pred_img_valid)
        frames_single_trajectories.append(new_trajectory_frame_idx)

    return (
        rally_errors,
        rally_ids,
        gt_r_img_coords,
        court_img_coords,
        pred_3D_coords,
        ankle_v_coords,
        pred_img_coords,
        frames_single_trajectories,
    )


def create_rally_histogram(
    errors: list, title: str = "Distribution of Mean 2D Error per Rally"
):
    """
    Create histogram of rally-level 2D errors.

    Args:
        errors: List of mean 2D errors per rally.
        title: Title for the histogram.
    """
    errors_array = np.array(errors)
    n_bins = 9
    bins = np.linspace(0, 45, n_bins + 1)
    counts, bin_edges = np.histogram(errors_array, bins=bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins)]

    ax.bar(range(n_bins), counts, tick_label=bin_labels)
    ax.set_xlabel("Mean 2D Error per Rally (pixels)")
    ax.set_ylabel("Number of Rallies")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add count labels on top of each bar
    for i, count in enumerate(counts):
        ax.text(i, count, str(int(count)), ha="center", va="bottom", fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def print_rally_evaluation_summary(errors: list):
    """
    Print summary statistics for rally-level 2D errors.

    Args:
        errors: List of mean 2D errors per rally.
    """
    errors_array = np.array(errors)

    print("\n" + "=" * 60)
    print("RALLY-LEVEL 2D REPROJECTION ERROR SUMMARY")
    print("=" * 60)
    print(f"Number of rallies evaluated: {len(errors)}")
    print(f"Mean error across rallies:   {np.mean(errors_array):.3f} pixels")
    print(f"Median error:                {np.median(errors_array):.3f} pixels")
    print(f"Std deviation:               {np.std(errors_array):.3f} pixels")
    print(
        f"Min error of rally {np.argmin(errors_array)}:       {np.min(errors_array):.3f} pixels"
    )
    print(
        f"Max error of rally {np.argmax(errors_array)}:       {np.max(errors_array):.3f} pixels"
    )

    # Normalized by image diagonal
    diagonal = np.sqrt(WIDTH**2 + HEIGHT**2)
    normalized_mean = np.mean(errors_array) / diagonal
    print(
        f"\nNormalized mean error:       {normalized_mean:.4f} (relative to image diagonal)"
    )
    print("=" * 60 + "\n")


def _infer_single_rally_prediction(model_path: str, rally_id: int):
    """Run one rally through one model and return valid predicted 3D points and segment boundaries."""
    model, _ = load_model(model_path)
    model = model.to(device)
    model.eval()

    val_transforms = get_transforms(None, "val")
    valset = RealInferenceRalliesDataset("test", transforms=val_transforms)

    if rally_id < 0 or rally_id >= len(valset):
        raise ValueError(
            f"rally_id {rally_id} out of range. Dataset has {len(valset)} rallies."
        )

    (
        r_img,
        court_img,
        mask,
        times,
        _,
        _,
        _,
        _,
        new_trajectory_frame_idx,
        ankle_v,
    ) = valset[rally_id]

    r_img = r_img.unsqueeze(0).to(device)
    court_img = court_img.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    times = times.unsqueeze(0).to(device)
    ankle_v = ankle_v.unsqueeze(0).to(device)

    use_ankle = "rallies_mode-dynamicAnkle" in model_path

    with torch.no_grad():
        if use_ankle:
            _, pred_position = model(r_img, court_img, mask, times, ankle_v)
        else:
            _, pred_position = model(r_img, court_img, mask, times)

    pred_position = pred_position.squeeze(0).cpu().numpy()

    _, sorted_mask, _ = _sort_visible_first_like_model(r_img, mask, times)
    valid_mask = sorted_mask.squeeze(0).cpu().numpy().astype(bool)
    pred_position_valid = pred_position[valid_mask]

    starts_raw = np.array(new_trajectory_frame_idx).astype(int)
    starts_positive = starts_raw[starts_raw > 0]

    trajectory_length = pred_position_valid.shape[0]
    boundaries = np.concatenate(
        [np.array([0]), starts_positive, np.array([trajectory_length])]
    )
    boundaries = np.unique(boundaries)
    boundaries = boundaries[(boundaries >= 0) & (boundaries <= trajectory_length)]

    return pred_position_valid, boundaries


def compare_two_models_sideview(rally_id: int):
    """
    Compare one selected rally in xz side view for both models.

    Creates one figure per detected trajectory segment with:
    - dynamicAnkle model in blue
    - dynamic model in red
    """
    pred_dynamic, boundaries = _infer_single_rally_prediction(r_dynamic_model, rally_id)
    pred_ankle, _ = _infer_single_rally_prediction(r_dynamic_ankle_model, rally_id)

    segment_count = len(boundaries) - 1

    combined_dynamic_segments = []
    combined_ankle_segments = []

    for segment_idx in range(segment_count):
        start_frame = boundaries[segment_idx]
        end_frame = boundaries[segment_idx + 1]
        if end_frame <= start_frame:
            continue

        if start_frame == 0:
            dynamic_segment = pred_dynamic[start_frame:end_frame]
            ankle_segment = pred_ankle[start_frame:end_frame]
        else:
            dynamic_segment = pred_dynamic[start_frame - 1 : end_frame]
            ankle_segment = pred_ankle[start_frame - 1 : end_frame]

        if dynamic_segment.shape[0] == 0 or ankle_segment.shape[0] == 0:
            continue

        if segment_idx < 2:
            combined_dynamic_segments.append(dynamic_segment)
            combined_ankle_segments.append(ankle_segment)
            continue

        fig, ax = plt.subplots()
        num_id = rally_id + 30
        ax.set_title(
            f"xz side view rally {num_id} (segment {segment_idx + 1}/{segment_count})"
        )
        ax.grid(visible="true", which="minor")
        ax.set_aspect("equal")
        ax.margins(0.2, 0.2)
        ax.set_xlim(-14, 14)
        ax.set_ylim(-0.2, 4)

        for connection in court_connections:
            ax.plot(court_points[connection, 0], court_points[connection, 2], "k")

        ax.plot(
            dynamic_segment[:, 0],
            dynamic_segment[:, 2],
            color="#DC3241",
            label="dynamic",
            linewidth=2,
        )
        ax.plot(
            ankle_segment[:, 0],
            ankle_segment[:, 2],
            color="#2556AC",
            label="dynamicAnkle",
            linewidth=2,
        )
        ax.scatter(
            dynamic_segment[0, 0],
            dynamic_segment[0, 2],
            color="black",
            s=20,
            zorder=3,
        )
        ax.scatter(
            ankle_segment[0, 0],
            ankle_segment[0, 2],
            color="black",
            s=20,
            zorder=3,
        )
        ax.grid(True)
        # ax.legend()
        plt.show()

    if combined_dynamic_segments and combined_ankle_segments:
        fig, ax = plt.subplots()
        num_id = rally_id + 30
        ax.set_title(f"xz side view rally {num_id} (segments 1-2/{segment_count})")
        ax.grid(visible="true", which="minor")
        ax.set_aspect("equal")
        ax.margins(0.2, 0.2)
        ax.set_xlim(-14, 14)
        ax.set_ylim(-0.2, 4)

        for connection in court_connections:
            ax.plot(court_points[connection, 0], court_points[connection, 2], "k")

        for seg_idx, (dynamic_segment, ankle_segment) in enumerate(
            zip(combined_dynamic_segments, combined_ankle_segments)
        ):
            dyn_label = "dynamic (seg 1)" if seg_idx == 0 else "dynamic (seg 2)"
            ankle_label = (
                "dynamicAnkle (seg 1)" if seg_idx == 0 else "dynamicAnkle (seg 2)"
            )

            ax.plot(
                dynamic_segment[:, 0],
                dynamic_segment[:, 2],
                color="#DC3241",
                label=dyn_label,
                linewidth=2,
                alpha=1.0,
            )
            ax.plot(
                ankle_segment[:, 0],
                ankle_segment[:, 2],
                color="#2556AC",
                label=ankle_label,
                linewidth=2,
                alpha=1.0,
            )
            ax.scatter(
                dynamic_segment[0, 0],
                dynamic_segment[0, 2],
                color="black",
                s=20,
                zorder=3,
            )
            ax.scatter(
                ankle_segment[0, 0],
                ankle_segment[0, 2],
                color="black",
                s=20,
                zorder=3,
            )

        ax.grid(True)
        # ax.legend()
        plt.show()


if __name__ == "__main__":
    compare_two_models_sideview(15)

    # Evaluate 2D error over all rallies
    """(
        rally_errors,
        rally_ids,
        gt_r_img_coords,
        court_img_coords,
        pred_3d_coords,
        ankle_v_coords,
        pred_img_coords,
        new_trajectory_frame_idx,
    ) = evaluate_rallies_2d_error(MODEL_PATH)
    print_rally_evaluation_summary(rally_errors)
    create_rally_histogram(rally_errors)

    if ankle:
        visualize_trajectories(
            gt_r_img_coords[13:14],
            pred_img_coords[13:14],
            pred_3d_coords[13:14],
            court_img_coords[13:14],
            rally_ids[13:14],
            new_trajectory_frame_idx=new_trajectory_frame_idx[13:14],
            single_trajectories=True,
            ankle_img=ankle_v_coords[13:14],
        )
    else:
        visualize_trajectories(
            gt_r_img_coords[13:14],
            pred_img_coords[13:14],
            pred_3d_coords[13:14],
            court_img_coords[13:14],
            rally_ids[13:14],
            new_trajectory_frame_idx=new_trajectory_frame_idx[13:14],
            single_trajectories=True,
        )

    (
        gt_trajectory_3D,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img,
        ankle_img,
        ankle_pos,
    ) = get_trajectory(MODEL_PATH)
    visualize_trajectories(
        pred_trajectory_3D,
        gt_img_coords,
        pred_img_coords,
        court_img,
        gt_trajectory_3D=gt_trajectory_3D,
        ankle_img=ankle_img,
        ankle_pos=ankle_pos,
    )

    (
        metrics,
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img,
        spin_gt,
        spin_pred,
    ) = get_worst_real_trajectory(MODEL_PATH)

    calcualte_metric(metrics)

    create_histogram(metrics)

    (
        trajectory_ids,
        gt_img_coords,
        pred_trajectory_3D,
        pred_img_coords,
        court_img,
    ) = get_specific_real_trajectory(
        MODEL_PATH,
        1,
    )"""


# %%
