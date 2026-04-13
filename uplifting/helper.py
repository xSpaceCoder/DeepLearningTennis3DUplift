import numpy as np
import einops as eo
import torch
from torch.utils.tensorboard.summary import hparams
import random
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from paths import (
    data_path as DATA_PATH,
    data_path_ram as DATA_PATH_RAM,
    logs_path as LOGS_PATH,
)

KEYPOINT_VISIBLE = 1
KEYPOINT_INVISIBLE = 0

TENNIS_COURT_LENGTH = 23.774  # baseline to baseline
TENNIS_COURT_WIDTH = 8.233  # sideline to sideline

# Net Dimensions
NET_TENNIS_WIDTH = TENNIS_COURT_WIDTH + (2 * 0.914)  # Length of the net
NET_POLE_TENNIS_HEIGHT = 1.07  # Height of the net pole
NET_TENNIS_HEIGHT = 0.914  # Height of the net at the center
# NET_TENNIS_DEPTH = 0.008  # Thickness/width of the net material (e.g., 0.8 cm)

# Derived Court and Net Positions
BASELINE_X_CLOSE = TENNIS_COURT_LENGTH / 2
BASELINE_X_FAR = -TENNIS_COURT_LENGTH / 2
SERVICELINE_X_CLOSE = 6.401
SERVICELINE_X_FAR = -6.401
NET_X_POS = 0.0

SINGLE_SIDELIINE_Y_LEFT = -TENNIS_COURT_WIDTH / 2
SINGLE_SIDELIINE_Y_RIGHT = TENNIS_COURT_WIDTH / 2
NETPOST_Y_LEFT = -NET_TENNIS_WIDTH / 2
NETPOST_Y_RIGHT = NET_TENNIS_WIDTH / 2

HEIGHT, WIDTH = (
    1080,
    1920,
)  # train, val_real and val_synth will be processed with HD resolution
MAX_FPS = 500  # used to calculate the rotation angles in the rotary positional embedding -> Videos should be recorded with a lower fps

# fx and fy regressed from a real tennis broadcast video
base_f = 1900


court_points = np.array(
    [
        [BASELINE_X_CLOSE, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 0 close left
        [BASELINE_X_CLOSE, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 1 close right
        [0.0, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 2 center left
        [0.0, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 3 center right
        [BASELINE_X_FAR, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 4 far left
        [BASELINE_X_FAR, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 5 far right
        [0.0, 0.0, 0.0],  # 6 net center bottom
        [0.0, NETPOST_Y_LEFT, NET_POLE_TENNIS_HEIGHT],  # 7 net left top
        [0.0, 0.0, NET_TENNIS_HEIGHT],  # 8 net center top
        [0.0, NETPOST_Y_RIGHT, NET_POLE_TENNIS_HEIGHT],  # 9 net right top
        [
            SERVICELINE_X_CLOSE,
            SINGLE_SIDELIINE_Y_LEFT,
            0.0,
        ],  # 10 service line close left
        [SERVICELINE_X_CLOSE, 0.0, 0.0],  # 11 service line close center
        [
            SERVICELINE_X_CLOSE,
            SINGLE_SIDELIINE_Y_RIGHT,
            0.0,
        ],  # 12 service line close right
        [SERVICELINE_X_FAR, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 13 service line far left
        [SERVICELINE_X_FAR, 0.0, 0.0],  # 14 service line far center
        [SERVICELINE_X_FAR, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 15 service line far right
    ]
)

# Connections for drawing the tennis court lines
court_connections = [
    (0, 2),  # left side
    (2, 4),  # left side
    (1, 3),  # right side
    (3, 5),  # right side
    (0, 1),  # front side
    (4, 5),  # back side
    (6, 2),  # center line
    (2, 3),  # center line
    (3, 7),  # center line
    (6, 9),  # net
    (9, 10),  # net
    (10, 7),  # net
    (12, 8),  # center service line
    (8, 15),  # center service line
    (11, 12),  # close service line
    (12, 13),  # close service line
    (14, 15),  # far service line
    (15, 16),  # far service line
]


def get_cameralocations(Mexts):
    """Get the camera location from the extrinsic matrix."""
    if len(Mexts.shape) == 3:
        R_transposed = eo.rearrange(Mexts[:, :3, :3], "t i j -> t j i")  # R^-1 = R^T
        c = np.einsum("t i j, t j -> t i", -R_transposed, Mexts[:, :3, 3])  # R^-1 * -t
    elif len(Mexts.shape) == 2:
        R_transposed = Mexts[:3, :3].T
        c = -R_transposed @ Mexts[:3, 3]
    else:
        raise ValueError("Shape not supported.")
    return c


def get_forwards(Mexts):
    """Get the normalized forward direction from the extrinsic matrix."""
    forwards = Mexts[..., 2, :3]
    forwards /= np.linalg.norm(forwards, axis=-1)[..., np.newaxis]
    return forwards


def get_ups(Mexts):
    """Get the normalized up direction from the extrinsic matrix."""
    ups = -Mexts[..., 1, :3]
    ups /= np.linalg.norm(ups, axis=-1)[..., np.newaxis]
    return ups


def get_rights(Mexts):
    """Get the normalized right direction from the extrinsic matrix."""
    rights = Mexts[..., 0, :3]
    rights /= np.linalg.norm(rights, axis=-1)[..., np.newaxis]
    return rights


def get_Mext(c, f, r):
    """Get the extrinsic matrix from the camera location, forward and right directions."""
    if len(c.shape) == len(f.shape) == len(r.shape) == 1:
        up = np.cross(f, r)
        up /= np.linalg.norm(up)
        R = np.zeros((3, 3))
        R[0, :] = r
        R[1, :] = up
        R[2, :] = f
        t = -R @ c
        Mext = np.eye(4)
        Mext[:3, :3] = R
        Mext[:3, 3] = t
        return Mext
    elif len(c.shape) == len(f.shape) == len(r.shape) == 2:
        up = np.cross(f, r)
        up /= np.linalg.norm(up, axis=1)[:, np.newaxis]
        R = np.zeros((len(c), 3, 3))
        R[:, 0, :] = r
        R[:, 1, :] = up
        R[:, 2, :] = f
        t = -np.einsum("t i j, t j -> t i", R, c)
        Mext = np.zeros((len(c), 4, 4))
        Mext[:, :3, :3] = R
        Mext[:, :3, 3] = t
        Mext[:, 3, 3] = 1
        return Mext
    else:
        raise ValueError("Shape not supported.")


def cam2img(r_cam, Mints):
    """Project a batch of 3D points to image coordinates."""
    if len(r_cam.shape) == 1:
        if len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, "i j, j -> i")
            r_img = r_img[:2] / r_img[2]
        else:
            raise ValueError("Shape not supported.")
    elif len(r_cam.shape) == 2:
        if len(Mints.shape) == 3:
            r_img = eo.einsum(Mints[:, :3, :3], r_cam, "b i j, b j -> b i")
            r_img = r_img[:, :2] / r_img[:, 2:3]
        elif len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, "i j, b j -> b i")
            r_img = r_img[:, :2] / r_img[:, 2:3]
        else:
            raise ValueError("Shape not supported.")
    elif len(r_cam.shape) == 3:
        if len(Mints.shape) == 3:
            r_img = eo.einsum(Mints[:, :3, :3], r_cam, "b i j, b t j -> b t i")
            r_img = r_img[:, :, :2] / r_img[:, :, 2:3]
        elif len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, "i j, b t j -> b t i")
            r_img = r_img[:, :, :2] / r_img[:, :, 2:3]
        else:
            raise ValueError("Shape not supported.")
    else:
        raise ValueError("Shape not supported.")
    return r_img


def world2cam(r_world, Mexts):
    """Transform a batch of 3D points from world to camera coordinates."""
    if len(r_world.shape) == 1:
        D = r_world.shape
        if len(Mexts.shape) == 2:
            r_world = concat(r_world, (D,))
            r_cam = eo.einsum(Mexts, r_world, "i j, j -> i")
            r_cam = r_cam[:3] / r_cam[3]
        else:
            raise ValueError("Shape not supported.")
    elif len(r_world.shape) == 2:
        T, D = r_world.shape
        if len(Mexts.shape) == 3:
            r_world = concat(r_world, (T, D))
            r_cam = eo.einsum(Mexts, r_world, "b i j, b j -> b i")
            r_cam = r_cam[:, :3] / r_cam[:, 3:4]
        elif len(Mexts.shape) == 2:
            r_world = concat(r_world, (T, D))
            r_cam = eo.einsum(Mexts, r_world, "i j, b j -> b i")
            r_cam = r_cam[:, :3] / r_cam[:, 3:4]
        else:
            raise ValueError("Shape not supported.")
    elif len(r_world.shape) == 3:
        B, T, D = r_world.shape
        if len(Mexts.shape) == 3:
            r_world = concat(r_world, (B, T, D))
            r_cam = eo.einsum(Mexts, r_world, "b i j, b t j -> b t i")
            r_cam = r_cam[:, :, :3] / r_cam[:, :, 3:4]
        elif len(Mexts.shape) == 2:
            r_world = concat(r_world, (B, T, D))
            r_cam = eo.einsum(Mexts, r_world, "i j, b t j -> b t i")
            r_cam = r_cam[:, :, :3] / r_cam[:, :, 3:4]
        else:
            raise ValueError("Shape not supported.")
    else:
        raise ValueError("Shape not supported.")
    return r_cam


def concat(x, shape):
    """
    Concatenates a tensor `x` with a tensor of ones along the last dimension.
    Parameters:
    - x: Input tensor, either a numpy array or a PyTorch tensor.
    - shape: Desired shape of the tensor to concatenate (should match `x` except for the last dimension).
    Returns:
    - Concatenated tensor with an additional column of ones.
    """
    if isinstance(x, np.ndarray):
        ones = np.ones((*shape[:-1], 1))
        return np.concatenate([x, ones], axis=-1)
    elif isinstance(x, torch.Tensor):
        ones = torch.ones((*shape[:-1], 1), device=x.device)
        return torch.cat([x, ones], dim=-1)
    else:
        raise TypeError("Input must be either a numpy array or a torch tensor.")


class SummaryWriter(torch.utils.tensorboard.SummaryWriter):
    """Fixes the add_hparams function of SummaryWriter

    For more information: https://github.com/pytorch/pytorch/issues/32651#issuecomment-648340103
    """

    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    ):
        """Fixed version of `add_hparams`"""
        torch._C._log_api_usage_once(
            "tensorboard.logging.add_hparams"
        )  # pylint: disable=protected-access
        if not isinstance(hparam_dict, dict) or not isinstance(metric_dict, dict):
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)

    def add_hparams2(self, hparam_dict, metric_dict):
        """A variant of the `add_hparams` function that doesn't write a scalar.
        At least to me, this additional metric in the scalars tab is annoying.
        """
        exp, ssi, sei = hparams(hparam_dict, metric_dict, None)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)


def get_logs_path():
    return os.path.join(LOGS_PATH, "uplifting")


def get_data_path():
    tmp_exists = os.path.exists(DATA_PATH_RAM)
    if tmp_exists:
        return DATA_PATH_RAM
        # return os.path.join(DATA_PATH_RAM, "syntheticdata_03-04")
    else:
        return os.path.join(DATA_PATH, "syntheticdata_03-04")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def binary_metrics(rot_gt, rot_pred):
    """Calculates the true/false positive/negative for the direction of each spin component (e.g. topspin vs backspin).
    Args:
        rot_gt (torch.tensor): ground truth rotation, shape (B, 3)
        rot_pred (torch.tensor): predicted rotation, shape (B, 3)
    Returns:
        TP (torch.tensor): true positives, shape (3)
        TN (torch.tensor): true negatives, shape (3)
        FP (torch.tensor): false positives, shape (3)
        FN (torch.tensor): false negatives, shape (3)
    """
    rot_gt = torch.sign(rot_gt)
    rot_pred = torch.sign(rot_pred)
    TP = torch.sum(torch.logical_and(rot_gt[:, :] == 1, rot_pred[:, :] == 1), dim=0)
    TN = torch.sum(torch.logical_and(rot_gt[:, :] == -1, rot_pred[:, :] == -1), dim=0)
    FP = torch.sum(torch.logical_and(rot_gt[:, :] == -1, rot_pred[:, :] == 1), dim=0)
    FN = torch.sum(torch.logical_and(rot_gt[:, :] == 1, rot_pred[:, :] == -1), dim=0)

    return TP, TN, FP, FN


def update_ema(model, model_ema, alpha=0.95):
    """Update the EMA model with the current model.
    Args:
        model (torch.nn.Module): current model
        model_ema (torch.nn.Module): EMA model
        alpha (float): EMA decay factor
    Returns:
        model_ema (torch.nn.Module): updated EMA model
    """
    with torch.no_grad():
        model_state = model.state_dict()  # Call once!
        for name, param in model_ema.named_parameters():
            param.data = alpha * param.data + (1 - alpha) * model_state[name].data
        for name, param in model_ema.named_buffers():
            param.data = alpha * param.data + (1 - alpha) * model_state[name].data

    return model_ema


def create_confusion_matrix(TP, TN, FP, FN, title="Confusion Matrix", dpi=100):
    """Create a confusion matrix from true/false positives/negatives and display it as an image.
    Args:
        TP (int): true positives
        TN (int): true negatives
        FP (int): false positives
        FN (int): false negatives
        title (str): title of the plot
    Returns:
        img (np.array): image of the confusion matrix
    """
    cm = np.array([[TP, FN], [FP, TN]])

    # Create a figure with reduced size
    fig, ax = plt.subplots(figsize=(2, 2), dpi=dpi)
    canvas = FigureCanvas(fig)
    im = ax.imshow(cm, cmap="Blues")  # Store the image object

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["Pred +", "Pred -"])
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(["GT +", "GT -"])

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations with adjustable font size and color based on cell brightness
    threshold = im.norm(cm.max()) / 2.0  # Threshold based on max value's normalization
    for i in range(2):
        for j in range(2):
            text_color = (
                "white" if im.norm(cm[i, j]) > threshold else "black"
            )  # Determine text color
            ax.text(
                j, i, cm[i, j], ha="center", va="center", color=text_color, fontsize=12
            )

    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    canvas.draw()

    img = np.array(canvas.buffer_rgba())
    plt.close()
    return img


def save_model(model, config, epoch, debug, name="model.pt"):
    """Saves the model weights and additional information about the run
    Args:
        model (nn.Module): model to save
        config (MyConfig): configuration object
        epoch (int): epoch number
        debug (bool): debug mode
    """
    save_path = config.get_pathforsaving(debug)
    os.makedirs(save_path, exist_ok=True)
    h_params = config.get_hparams()
    additional_info = {"epoch": epoch, **h_params}

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "identifier": config.get_identifier(),
            "additional_info": additional_info,
        },
        os.path.join(save_path, name),
    )


def transform_rotationaxes(rotation, r_gt):
    """Rotates the axes such that the x axis is in flight direction, y axis in the worlds x-y plane.
    Args:
        rotation (torch.tensor): rotation of the ball, shape (B, 3) or (3,)
        r_gt (torch.tensor): ground truth trajectory, shape (B, T, 3) or (T, 3)
    """
    if len(r_gt.shape) == 3:
        B, T, _ = r_gt.shape
        v0 = torch.zeros((B, 3), device=r_gt.device, dtype=r_gt.dtype)
        # Create e_z directly with the correct shape - much faster than torch.tensor + repeat
        e_z = torch.zeros((B, 3), device=r_gt.device, dtype=r_gt.dtype)
        e_z[:, 2] = -1.0
    elif len(r_gt.shape) == 2:
        T, __ = r_gt.shape
        v0 = torch.zeros((3,), device=r_gt.device, dtype=r_gt.dtype)
        # Create e_z directly - avoids torch.tensor overhead
        e_z = torch.zeros((3,), device=r_gt.device, dtype=r_gt.dtype)
        e_z[2] = -1.0
    else:
        raise ValueError("Shape not supported.")
    # get axes
    v0[..., :2] = r_gt[..., 1, :2] - r_gt[..., 0, :2]
    e_x = v0 / torch.linalg.norm(v0, dim=-1, keepdim=True)
    e_y = torch.cross(e_z, e_x, dim=-1)

    # get elements of rotation in axis directions
    w_0 = torch.einsum("... i, ... i -> ...", rotation, e_x)
    w_1 = torch.einsum("... i, ... i -> ...", rotation, e_y)
    w_2 = torch.einsum("... i, ...i -> ...", rotation, e_z)

    return torch.stack([w_0, w_1, w_2], dim=-1)


def inversetransform_rotationaxes(local_rotation, r_gt):
    """Inversely rotates the axes such that the rotation is transformed from local into the global coordinate system.
    Args:
        local_rotation (torch.tensor): rotation of the ball in local coordinate system, shape (B, 3) or (3,)
        r_gt (torch.tensor): ground truth trajectory in global coordinate system, shape (B, T, 3) or (T, 3)
    """
    if len(r_gt.shape) == 3:
        B, T, _ = r_gt.shape
        v0 = torch.zeros((B, 3), device=r_gt.device)
        e_z = torch.tensor([0, 0, 1], device=r_gt.device, dtype=torch.float32).repeat(
            B, 1
        )
        R_T = torch.zeros((B, 3, 3), device=r_gt.device, dtype=r_gt.dtype)
    elif len(r_gt.shape) == 2:
        T, _ = r_gt.shape
        v0 = torch.zeros((3,), device=r_gt.device)
        e_z = torch.tensor([0, 0, 1], device=r_gt.device, dtype=torch.float32)
        R_T = torch.zeros((3, 3), device=r_gt.device, dtype=r_gt.dtype)
    else:
        raise ValueError("Shape not supported.")

    # Compute the local basis vectors
    v0[..., :2] = r_gt[..., 1, :2] - r_gt[..., 0, :2]
    e_x = v0 / torch.linalg.norm(v0, dim=-1, keepdim=True)
    e_y = torch.cross(e_z, e_x, dim=-1)

    # Combine the local basis vectors into a matrix
    # The basis vectors are columns of the rotation matrix
    R_T[..., :, 0] = e_x
    R_T[..., :, 1] = e_y
    R_T[..., :, 2] = e_z

    # Transform the local rotation into the global coordinate system
    global_rotation = torch.einsum("... i j, ... j -> ... i", R_T, local_rotation)

    return global_rotation
