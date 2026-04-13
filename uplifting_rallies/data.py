import numpy as np
import random
import torch
import os
import glob

from uplifting_rallies.helper import get_Mext, cam2img, world2cam
from uplifting_rallies.helper import HEIGHT, WIDTH, base_f
from uplifting_rallies.helper import court_points
from uplifting_rallies.helper import (
    KEYPOINT_VISIBLE,
    BALL_VISIBLE,
)
from paths import data_path as DATA_PATH
from paths import data_path_synth_rallies as TMPFS_SYNTH_RALLIES
from paths import data_path_real_trajectories_with_tosses as TMPFS_REAL_TRAJECTORIES
from paths import data_path_real_rallies as TMPFS_REAL_RALLIES


BACKSPIN_CLASS = 2
TOPSPIN_CLASS = 1
NOT_ANNOTATED_CLASS = 0

OFFSET_CENTER_Y = 1

CAMERA_VIEW_POINTS = np.array(
    [[-14, 7, 0], [-14, -7, 0], [13, 6, 0], [13, -6, 0], [0, 6, 4], [0, -6, 4]]
)


def pad_or_crop_sequence(values, sequence_len, feature_dim=None, fill_value=0):
    """Pad or crop a sequence to a fixed length."""
    max_t = min(values.shape[0], sequence_len)
    if feature_dim is None:
        padded = np.full((sequence_len,), fill_value, dtype=values.dtype)
        padded[:max_t] = values[:max_t]
    else:
        padded = np.full((sequence_len, feature_dim), fill_value, dtype=values.dtype)
        padded[:max_t] = values[:max_t, :feature_dim]
    return padded, max_t


def build_sequence_mask(valid_length, sequence_len):
    """Create a mask for valid (non-padded) sequence elements."""
    mask = np.zeros((sequence_len,), dtype=bool)
    mask[:valid_length] = True
    return mask


def to_float32_tensor(values):
    """Create a torch float32 tensor without extra copies when possible."""
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def to_int64_tensor(values):
    """Create a torch int64 tensor without extra copies when possible."""
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.int64))


class StitchedRallyDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading stitched tennis rallies.

    Each item in the dataset corresponds to one full .npz file,
    with all segments (shots) concatenated into a single trajectory.
    """

    def __init__(self, mode="train", transforms=None):
        self.mode = mode
        self.transforms = transforms

        self.sequence_len = 600  # maximum length of the sequence (maximum number of tokens for the transformer) -> pad/crop to this length

        data_name = "reduced_stitched_rallies"
        if os.path.exists(TMPFS_SYNTH_RALLIES):
            self.root_dir = TMPFS_SYNTH_RALLIES
            print("Loading data from RAM using tmpfs...")
        else:
            self.root_dir = os.path.join(DATA_PATH, data_name)
            print(
                "WARNING: Loading data from disk, consider using tmpfs for faster loading!"
            )
        self.original_fps = 500  # Original data framerate in Hz

        # Load the list of all "branches" (which are the .npz files), do not include dead ends or the unused indices file
        search_pattern = os.path.join(self.root_dir, "toss_*_branch_*.npz")
        self.rally_files = sorted(glob.glob(search_pattern))
        # Filter out dead-end files explicitly
        self.rally_files = [f for f in self.rally_files if "_deadend.npz" not in f]

        # choose data according to mode
        if mode == "train":
            self.rally_files = self.rally_files[: int(0.88 * len(self.rally_files))]
        elif mode == "val":
            self.rally_files = self.rally_files[
                int(0.88 * len(self.rally_files)) : int(0.9 * len(self.rally_files))
            ]
        elif mode == "test":
            self.rally_files = self.rally_files[int(0.9 * len(self.rally_files)) :]
        else:
            raise ValueError(f"Unknown mode {mode}")

        # minimum and maximum phi that is sampled for the camera position
        self.sampled_phis = (
            3,
            -3,
        )
        self.sampled_distances = (
            12,
            60,
        )  # minimum and maximum distance of the camera to the origin
        self.sampled_thetas = (
            85,
            65,
        )  # minimum and maximum theta that is sampled for the camera position
        self.sampled_f = (
            0.3 * base_f,
            2.0 * base_f,
        )  # minimum and maximum focal length that is sampled for the camera intrinsics

        self.fps_bounds = (20, 65)  # minimum and maximum framerate
        # The resolution of the simulated video frames -> rescale the coordinates from this resolution to the working resolution (WIDTH, HEIGHT)
        self.original_resolution = (1920, 1080)

    def __len__(self):
        """Returns the total number of rally files."""
        return len(self.rally_files)

    def __getitem__(self, idx):
        """
        Loads one full rally, concatenates its segments, and returns
        all requested data as a dictionary of tensors.
        """
        filepath = self.rally_files[idx]
        # Load the data, allowing pickles for the object array
        with np.load(filepath, allow_pickle=True) as data:
            positions = np.concatenate(data["positions"], axis=0)
            velocities = np.concatenate(data["velocities"], axis=0)
            rotation = np.concatenate(data["rotations"], axis=0)
            ankle_position = self.get_ankle_position(positions[0])

        # Generate the time vector
        num_frames = positions.shape[0]
        blur_times = np.arange(num_frames) / self.original_fps

        # sample a framerate and calculate times and r_world based on the new framerate
        fps = random.randint(self.fps_bounds[0], self.fps_bounds[1])
        start_time = blur_times[0]
        end_time = blur_times[-1]
        times = np.arange(start_time, end_time, 1.0 / fps)
        insertion_points = np.searchsorted(blur_times, times)
        idx_right = np.clip(insertion_points, 0, len(blur_times) - 1)
        idx_left = np.clip(insertion_points - 1, 0, len(blur_times) - 1)
        diff_left = np.abs(blur_times[idx_left] - times)
        diff_right = np.abs(blur_times[idx_right] - times)
        nearest_frame_indices = np.where(diff_right < diff_left, idx_right, idx_left)
        r_world = positions[nearest_frame_indices]
        rotation = rotation[nearest_frame_indices]

        # sample camera matrices
        Mint, Mext, r_img, court_img, ankle_v = self.sample_camera(
            r_world, ankle_position
        )

        # crop/pad sequences and build mask for valid timesteps
        r_img, max_t = pad_or_crop_sequence(r_img, self.sequence_len, feature_dim=2)
        mask = build_sequence_mask(max_t, self.sequence_len)
        r_world, __ = pad_or_crop_sequence(r_world, self.sequence_len, feature_dim=3)
        times, __ = pad_or_crop_sequence(times, self.sequence_len)
        rotation, __ = pad_or_crop_sequence(rotation, self.sequence_len, feature_dim=3)

        # Add visibility to court keypoints as extra dimension: (13, 2) --> (13, 3) ; All keypoints are visible at the moment
        court_img = np.concatenate(
            [
                court_img,
                np.full(
                    (court_img.shape[0], 1), KEYPOINT_VISIBLE, dtype=court_img.dtype
                ),
            ],
            axis=1,
        )
        # Add additional dimension to image positions that encode the visibility
        r_img = np.concatenate(
            [r_img, np.full((r_img.shape[0], 1), BALL_VISIBLE, dtype=r_img.dtype)],
            axis=1,
        )

        # rescale coordinates from original resolution to processing resolution
        data = {
            "r_img": r_img,
            "court_img": court_img,
            "ankle_v": ankle_v,
            "Mint": Mint,
        }
        data = transform_resolution(data, self.original_resolution, (WIDTH, HEIGHT))
        r_img, court_img, ankle_v, Mint = (
            data["r_img"],
            data["court_img"],
            data["ankle_v"],
            data["Mint"],
        )

        # apply transforms
        if self.transforms is not None:
            data = {
                "r_img": r_img,
                "r_world": r_world,
                "times": times,
                "mask": mask,
                "court_img": court_img,
                "Mint": Mint,
                "Mext": Mext,
                "blur_positions": positions,
                "blur_times": blur_times,
                "rotation": rotation,
                "ankle_v": ankle_v,
            }
            data = self.transforms(data)
            r_img, court_img, mask, r_world, ankle_v = (
                data["r_img"],
                data["court_img"],
                data["mask"],
                data["r_world"],
                data["ankle_v"],
            )
            times, Mint, Mext = data["times"], data["Mint"], data["Mext"]
            positions, blur_times = data["blur_positions"], data["blur_times"]
            rotation = data["rotation"]

        r_img, court_img, mask, ankle_v = (
            to_float32_tensor(r_img),
            to_float32_tensor(court_img),
            to_float32_tensor(mask),
            to_float32_tensor(ankle_v),
        )
        r_world, rotation, times = (
            to_float32_tensor(r_world),
            to_float32_tensor(rotation),
            to_float32_tensor(times),
        )
        Mint, Mext = to_float32_tensor(Mint), to_float32_tensor(Mext)

        return r_img, court_img, mask, r_world, rotation, times, Mint, Mext, ankle_v

    def sample_camera(self, r_world, ankle_position):
        valid = False
        try_num = 0
        max_tries = 100
        while not valid and try_num < max_tries:
            f = random.uniform(self.sampled_f[0], self.sampled_f[1])
            Mint = np.array(
                [[f, 0, (WIDTH - 1) / 2], [0, f, (HEIGHT - 1) / 2], [0, 0, 1]]
            )

            # extrinsic matrix
            distance = random.uniform(
                self.sampled_distances[0], self.sampled_distances[1]
            )
            # phi angle between 20 and 160 degrees
            phi = random.uniform(self.sampled_phis[0], self.sampled_phis[1])
            # theta angle between 30 and 70 degrees
            theta = random.uniform(self.sampled_thetas[0], self.sampled_thetas[1])
            # lookat point somewhere around the center of the court
            lookat = np.array(
                (
                    random.uniform(-1, 7),
                    random.uniform(-OFFSET_CENTER_Y, OFFSET_CENTER_Y),
                    0,
                )
            )

            # camera location
            c = np.array(
                [
                    distance * np.sin(np.radians(theta)) * np.cos(np.radians(phi)),
                    distance * np.sin(np.radians(theta)) * np.sin(np.radians(phi)),
                    distance * np.cos(np.radians(theta)),
                ]
            )
            # forward direction
            f = -(c - lookat) / np.linalg.norm(c - lookat)
            # right direction (choose a random vector approximately in the x-y plane)
            epsilon = random.uniform(
                -0.1, 0.1
            )  # small random value that controls the deviation from the x-y plane
            r = np.array([-f[1] / f[0] - f[2] / f[0] * epsilon, 1, epsilon])
            r /= np.linalg.norm(r)
            # up direction
            u = -np.cross(f, r)
            if u[2] < 0:  # The up vector has to be in the positive z direction
                r = np.array(
                    [f[1] / f[0] - f[2] / f[0] * epsilon, -1, epsilon]
                )  # choose the other direction for r_y
                r /= np.linalg.norm(r)
                u = -np.cross(f, r)
            u /= np.linalg.norm(u)
            # extrinsic matrix
            Mext = get_Mext(c, f, r)

            # calculate image coordinates of trajectory with estimated camera matrices
            r_cam = world2cam(r_world, Mext)
            r_img = cam2img(r_cam, Mint)
            # calculate image coordinates of ankle position
            ankle_cam = world2cam(ankle_position, Mext)
            ankle_img = cam2img(ankle_cam, Mint)

            # verify if PPA is in the camera view
            ppa_cam = world2cam(CAMERA_VIEW_POINTS, Mext)
            ppa_img = cam2img(ppa_cam, Mint)
            # check if field is completely inside the image
            valid = np.all((ppa_img >= 0) & (ppa_img < np.array([WIDTH, HEIGHT])))
            # calculate the court position in image coordinates
            court_cam = world2cam(court_points, Mext)
            court_img = cam2img(court_cam, Mint)
            # check if field is completely inside the image
            valid = valid and (
                np.all((court_img >= 0) & (court_img < np.array([WIDTH, HEIGHT])))
            )
            # check if close base line takes up at least 50% of the image width
            valid = valid and (
                np.abs(np.subtract(court_img[0, 0], court_img[1, 0])) > 0.5 * WIDTH
            )
            # check if trajectory is not too small in the image
            valid = valid and (
                r_img[:, 0].max() - r_img[:, 0].min() > 0.15 * WIDTH
                or r_img[:, 1].max() - r_img[:, 1].min() > 0.15 * HEIGHT
            )
            try_num += 1
        return Mint, Mext, r_img, court_img, ankle_img[1]

    def get_ankle_position(self, start_position):
        """
        Returns a 3D ankle position. This corresponds to the foot that is closest to the net.
        """
        sign_x = 1 if start_position[0] > 0 else -1
        # differenciate between over head and other strokes
        if start_position[2] < 2:
            # a ball is always hit in front of the body
            x_ankle = start_position[0] + sign_x * (random.random() * 1.5)
            y_ankle = start_position[1] + random.random() * 4 - 2
            z_ankle = 0.1
        else:  # overhead strokes are often hit within a jump
            x_ankle = start_position[0] + sign_x * (random.random() * 0.7)
            y_ankle = start_position[1] + random.random() * 0.8 - 0.4
            z_ankle = 0.1 + random.random() * 0.2
        return np.array([x_ankle, y_ankle, z_ankle])


class RealInferenceDataset(torch.utils.data.Dataset):
    """Dataset for real data inference"""

    def __init__(self, mode, transforms=None):
        self.path = TMPFS_REAL_TRAJECTORIES
        self.transforms = transforms
        assert mode in ["val", "test"], "mode must be one of: 'val', 'test'"

        self.data_paths = sorted(
            [
                os.path.join(self.path, foldername)
                for i, foldername in enumerate(os.listdir(self.path))
                if foldername.startswith("trajectory_")
            ]
        )

        self.sequence_len = (
            120
            # crop sequence if it is longer, else padding of sequence
        )

        self.original_resolution = (1280, 720)

        if mode == "val":
            self.data_paths = self.data_paths[: int(0.33 * len(self.data_paths))]
        elif mode == "test":
            self.data_paths = self.data_paths[int(0.33 * len(self.data_paths)) :]
        else:
            raise ValueError(f"Unknown mode {mode}")
        self.length = len(self.data_paths)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        r_img = np.load(os.path.join(data_path, "r_img.npy"))
        times = np.load(os.path.join(data_path, "times.npy"))
        bounces = np.load(os.path.join(data_path, "hits.npy"))
        ankle_v = np.load(os.path.join(data_path, "2dPoseEstimation.npy"))[0, 16, 1:2]
        if bounces.size == 0:
            bounces = np.array([-1], dtype=np.float32)
        elif bounces.size > 1:
            bounces = bounces[0:1]  # only return first hit
        Mint = np.load(os.path.join(data_path, "Mint.npy"))
        Mext = np.load(os.path.join(data_path, "Mext.npy"))
        spin_class = np.array([np.load(os.path.join(data_path, "spin_class.npy"))[0]])
        start_serve = np.load(os.path.join(data_path, "start_serve.npy"))

        court_cam = world2cam(court_points, Mext)
        court_img = cam2img(court_cam, Mint)

        # crop/pad sequences and build mask for valid timesteps
        r_img, max_t = pad_or_crop_sequence(r_img, self.sequence_len, feature_dim=3)
        mask = build_sequence_mask(max_t, self.sequence_len)
        times, __ = pad_or_crop_sequence(times, self.sequence_len)

        # Add visibility to court keypoints as extra dimension: (16, 2) --> (16, 3) ; All keypoints are visible at the moment
        court_img = np.concatenate(
            [
                court_img,
                np.full(
                    (court_img.shape[0], 1), KEYPOINT_VISIBLE, dtype=court_img.dtype
                ),
            ],
            axis=1,
        )

        # rescale coordinates from original resolution to processing resolution
        data = {
            "r_img": r_img,
            "court_img": court_img,
            "Mint": Mint,
            "ankle_v": ankle_v,
        }
        data = transform_resolution(data, self.original_resolution, (WIDTH, HEIGHT))
        r_img, court_img, Mint, ankle_v = (
            data["r_img"],
            data["court_img"],
            data["Mint"],
            data["ankle_v"],
        )

        # apply transforms
        data = {
            "r_img": r_img,
            "times": times,
            "hits": bounces,
            "mask": mask,
            "court_img": court_img,
            "Mint": Mint,
            "Mext": Mext,
            "spin_class": spin_class,
            "ankle_v": ankle_v,
        }
        if self.transforms is not None:
            data = self.transforms(data)
        r_img, court_img, mask, ankle_v = (
            data["r_img"],
            data["court_img"],
            data["mask"],
            data["ankle_v"],
        )
        times, bounces, Mint, Mext, spin_class = (
            data["times"],
            data["hits"],
            data["Mint"],
            data["Mext"],
            data["spin_class"],
        )

        r_img, court_img, mask, ankle_v = (
            to_float32_tensor(r_img),
            to_float32_tensor(court_img),
            to_float32_tensor(mask),
            to_float32_tensor(ankle_v),
        )
        times, start_serve, bounces, Mint, Mext = (
            to_float32_tensor(times),
            to_float32_tensor(start_serve),
            to_float32_tensor(bounces),
            to_float32_tensor(Mint),
            to_float32_tensor(Mext),
        )
        spin_class = to_float32_tensor(spin_class)

        return (
            r_img,
            court_img,
            mask,
            times,
            bounces,
            Mint,
            Mext,
            spin_class,
            start_serve,
            ankle_v,
        )


class RealInferenceRalliesDataset(torch.utils.data.Dataset):
    """Dataset for real data inference"""

    def __init__(self, mode, transforms=None):
        self.path = TMPFS_REAL_RALLIES
        self.transforms = transforms
        assert mode in ["val", "test"], "mode must be one of: 'val', 'test'"

        self.data_paths = sorted(
            [
                os.path.join(self.path, foldername)
                for i, foldername in enumerate(os.listdir(self.path))
                if foldername.startswith("rally_") and foldername != "rally_0059"
            ]
        )

        self.sequence_len = 600
        self.max_trajectories = 25

        self.original_resolution = (1280, 720)

        if mode == "val":
            self.data_paths = self.data_paths[: int(0.33 * len(self.data_paths))]
        elif mode == "test":
            self.data_paths = self.data_paths[int(0.33 * len(self.data_paths)) :]

        else:
            raise ValueError(f"Unknown mode {mode}")
        self.length = len(self.data_paths)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        r_img = np.load(os.path.join(data_path, "r_img.npy"))
        times = np.load(os.path.join(data_path, "times.npy"))
        Mint = np.load(os.path.join(data_path, "Mint.npy"))
        Mext = np.load(os.path.join(data_path, "Mext.npy"))
        spin_class_per_frame = np.load(
            os.path.join(data_path, "spin_class_per_frame.npy")
        )
        spin_class_per_shot = np.load(
            os.path.join(data_path, "spin_class_per_shot.npy")
        )
        new_trajectory_frame_idx = np.load(
            os.path.join(data_path, "new_trajectory_frame_idx.npy")
        )
        ankle_v = np.load(os.path.join(data_path, "2dPoseEstimation.npy"))[16, 1:2]

        court_cam = world2cam(court_points, Mext)
        court_img = cam2img(court_cam, Mint)

        # crop/pad sequences and build mask for valid timesteps
        r_img, max_t = pad_or_crop_sequence(r_img, self.sequence_len, feature_dim=3)
        mask = build_sequence_mask(max_t, self.sequence_len)
        times, __ = pad_or_crop_sequence(times, self.sequence_len)
        spin_class_per_frame, __ = pad_or_crop_sequence(
            spin_class_per_frame, self.sequence_len
        )

        # crop/pad per-shot arrays to the fixed maximum number of trajectories
        spin_class_per_shot, __ = pad_or_crop_sequence(
            spin_class_per_shot, self.max_trajectories, fill_value=-1
        )
        new_trajectory_frame_idx, __ = pad_or_crop_sequence(
            new_trajectory_frame_idx, self.max_trajectories, fill_value=-1
        )

        # Add visibility to court keypoints as extra dimension: (16, 2) --> (16, 3) ; All keypoints are visible at the moment
        court_img = np.concatenate(
            [
                court_img,
                np.full(
                    (court_img.shape[0], 1), KEYPOINT_VISIBLE, dtype=court_img.dtype
                ),
            ],
            axis=1,
        )

        # rescale coordinates from original resolution to processing resolution
        data = {
            "r_img": r_img,
            "court_img": court_img,
            "Mint": Mint,
            "ankle_v": ankle_v,
        }
        data = transform_resolution(data, self.original_resolution, (WIDTH, HEIGHT))
        r_img, court_img, Mint, ankle_v = (
            data["r_img"],
            data["court_img"],
            data["Mint"],
            data["ankle_v"],
        )

        # apply transforms
        data = {
            "r_img": r_img,
            "times": times,
            "mask": mask,
            "court_img": court_img,
            "Mint": Mint,
            "Mext": Mext,
            "ankle_v": ankle_v,
        }
        if self.transforms is not None:
            data = self.transforms(data)
        r_img, court_img, mask, ankle_v = (
            data["r_img"],
            data["court_img"],
            data["mask"],
            data["ankle_v"],
        )
        times, Mint, Mext = (
            data["times"],
            data["Mint"],
            data["Mext"],
        )

        r_img, court_img, mask, ankle_v = (
            to_float32_tensor(r_img),
            to_float32_tensor(court_img),
            to_float32_tensor(mask),
            to_float32_tensor(ankle_v),
        )
        times, Mint, Mext, new_trajectory_frame_idx = (
            to_float32_tensor(times),
            to_float32_tensor(Mint),
            to_float32_tensor(Mext),
            to_int64_tensor(new_trajectory_frame_idx),
        )
        spin_class_per_frame = to_float32_tensor(spin_class_per_frame)
        spin_class_per_shot = to_float32_tensor(spin_class_per_shot)

        return (
            r_img,
            court_img,
            mask,
            times,
            Mint,
            Mext,
            spin_class_per_shot,
            spin_class_per_frame,
            new_trajectory_frame_idx,
            ankle_v,
        )


def transform_resolution(data, original_resolution, processing_resolution):
    """Transform coordinates from original resolution to processing resolution.
    Arguments:
        data: dict with keys 'r_img', 'court_img', Mint
        original_resolution: tuple (width, height)
        processing_resolution: tuple (width, height)
    """
    assert (
        "r_img" in data and "court_img" in data and "Mint" in data and "ankle_v" in data
    ), "data must contain keys 'r_img', 'court_img', 'Mint' and 'ankle_v'"
    orig_w, orig_h = original_resolution
    proc_w, proc_h = processing_resolution
    scale_x = proc_w / orig_w
    scale_y = proc_h / orig_h
    r_img = data["r_img"]
    r_img[..., 0] = (r_img[..., 0] + 0.5) * scale_x - 0.5
    r_img[..., 1] = (r_img[..., 1] + 0.5) * scale_y - 0.5
    data["r_img"] = r_img
    court_img = data["court_img"]
    court_img[..., 0] = (court_img[..., 0] + 0.5) * scale_x - 0.5
    court_img[..., 1] = (court_img[..., 1] + 0.5) * scale_y - 0.5
    data["court_img"] = court_img
    Mint = data["Mint"]
    Mint[0, 0] = Mint[0, 0] * scale_x
    Mint[1, 1] = Mint[1, 1] * scale_y
    Mint[0, 2] = (Mint[0, 2] + 0.5) * scale_x - 0.5
    Mint[1, 2] = (Mint[1, 2] + 0.5) * scale_y - 0.5
    data["Mint"] = Mint
    ankle_v = data["ankle_v"]
    ankle_v = (ankle_v + 0.5) * scale_y - 0.5
    data["ankle_v"] = ankle_v

    return data
