import numpy as np
import random
import torch
import os

from uplifting.helper import get_Mext, cam2img, world2cam
from uplifting.helper import HEIGHT, WIDTH, base_f
from uplifting.helper import get_data_path
from uplifting.helper import court_points
from uplifting.helper import KEYPOINT_VISIBLE
from paths import (
    data_path_real_trajectories_no_tosses as DATA_PATH_REAL_TRAJECTORIES,
)


BACKSPIN_CLASS = 2
TOPSPIN_CLASS = 1
NOT_ANNOTATED_CLASS = 0

OFFSET_CENTER_Y = 1

CAMERA_VIEW_POINTS = np.array(
    [[-14, 7, 0], [-14, -7, 0], [13, 6, 0], [13, -6, 0], [0, 6, 4], [0, -6, 4]]
)


def pad_or_crop_sequence(values, sequence_len, feature_dim=None):
    """Pad or crop a sequence to a fixed length."""
    max_t = min(values.shape[0], sequence_len)
    if feature_dim is None:
        padded = np.zeros((sequence_len,), dtype=values.dtype)
        padded[:max_t] = values[:max_t]
    else:
        padded = np.zeros((sequence_len, feature_dim), dtype=values.dtype)
        padded[:max_t] = values[:max_t, :feature_dim]
    return padded, max_t


def build_sequence_mask(valid_length, sequence_len):
    """Create a mask for valid (non-padded) sequence elements."""
    mask = np.zeros((sequence_len,), dtype=bool)
    mask[:valid_length] = True
    return mask


def to_float32_tensor(values):
    """Create a torch tensor without extra copies when possible."""
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


class SynthTennisDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", transforms=None):
        self.mode = mode
        path = get_data_path()

        trajectory_modes = ["groundstroke", "serve", "volley", "smash", "lob", "short"]
        directions = ["far_to_close", "close_to_far"]
        in_out = ["in", "out"]
        data_paths = []
        for tm in trajectory_modes:
            for direction in directions:
                for state in in_out:
                    dps = sorted(
                        [
                            os.path.join(
                                path, tm, direction, state, f"trajectory_{i:04}"
                            )
                            for i, __ in enumerate(
                                os.listdir(os.path.join(path, tm, direction, state))
                            )
                        ]
                    )
                    rnd = random.Random(0)
                    rnd.shuffle(data_paths)
                    # get the same number of train/val/test data for each mode and direction
                    if mode == "train":
                        dps = dps[: int(0.7 * len(dps))]
                    elif mode == "val":
                        dps = dps[int(0.7 * len(dps)) : int(0.8 * len(dps))]
                    elif mode == "test":
                        dps = dps[int(0.8 * len(dps)) :]
                    else:
                        raise ValueError(f"Unknown mode {mode}")
                    data_paths.extend(dps)
        self.data_paths = data_paths
        self.length = len(self.data_paths)

        self.sequence_len = (
            120  # crop sequence if it is longer, else padding of sequence
        )

        # The resolution of the simulated video frames
        # -> rescale the coordinates from this resolution to the working resolution (WIDTH, HEIGHT)
        self.original_resolution = (1920, 1080)

        self.transforms = transforms

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

        self.num_cameras = 1 if mode in ["val", "test"] else 1
        self.cam_num = 0

        self.fps_bounds = (20, 65)  # minimum and maximum framerate
        self.eval_fps = 30  # frames per second for evaluation

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        blur_positions = np.load(os.path.join(data_path, "positions.npy"))
        blur_times = np.load(os.path.join(data_path, "times.npy"))
        bounces = np.atleast_1d(np.load(os.path.join(data_path, "bounces.npy")))
        rotation = np.load(os.path.join(data_path, "rotations.npy"))[0]
        ankle_position = self.get_ankle_position(blur_positions[0])

        # sample a framerate and calculate times and r_world based on the new framerate
        fps = (
            random.randint(self.fps_bounds[0], self.fps_bounds[1])
            if self.mode == "train"
            else self.eval_fps
        )
        start_time = blur_times[0]
        end_time = blur_times[-1]
        times = np.arange(start_time, end_time, 1.0 / fps)
        insertion_points = np.searchsorted(blur_times, times)
        idx_right = np.clip(insertion_points, 0, len(blur_times) - 1)
        idx_left = np.clip(insertion_points - 1, 0, len(blur_times) - 1)
        diff_left = np.abs(blur_times[idx_left] - times)
        diff_right = np.abs(blur_times[idx_right] - times)
        nearest_frame_indices = np.where(diff_right < diff_left, idx_right, idx_left)
        r_world = blur_positions[nearest_frame_indices]

        if self.mode == "train":
            Mint, Mext, r_img, court_img, ankle_v = self.sample_camera(
                r_world, ankle_position
            )
        else:
            Mint = np.load(os.path.join(data_path, "Mint.npy"))
            Mext = np.load(os.path.join(data_path, "Mext.npy"))
            Mint, Mext = Mint[0], Mext[0]
            r_cam = world2cam(r_world, Mext)
            r_img = cam2img(r_cam, Mint)
            court_cam = world2cam(court_points, Mext)
            court_img = cam2img(court_cam, Mint)
            ankle_cam = world2cam(ankle_position, Mext)
            ankle_v = cam2img(ankle_cam, Mint)[1]

        # crop/pad sequences and build mask for valid timesteps
        r_img, max_t = pad_or_crop_sequence(r_img, self.sequence_len, feature_dim=2)
        mask = build_sequence_mask(max_t, self.sequence_len)
        r_world, __ = pad_or_crop_sequence(r_world, self.sequence_len, feature_dim=3)
        times, __ = pad_or_crop_sequence(times, self.sequence_len)

        # if no bounces are present, set to -1  --> Relevant for RandomStopAugmentation
        if len(bounces) == 0:
            bounces = np.array(
                [
                    -1,
                ],
                dtype=np.float32,
            )

        # Add visibility to court keypoints as extra dimension: (17, 2) --> (17, 3) ; All keypoints are visible at the moment
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
        data = {
            "r_img": r_img,
            "r_world": r_world,
            "times": times,
            "hits": bounces,
            "rotation": rotation,
            "mask": mask,
            "court_img": court_img,
            "ankle_v": ankle_v,
            "Mint": Mint,
            "Mext": Mext,
            "blur_positions": blur_positions,
            "blur_times": blur_times,
        }
        if self.transforms is not None:
            data = self.transforms(data)
        r_img, court_img, mask, r_world, rotation = (
            data["r_img"],
            data["court_img"],
            data["mask"],
            data["r_world"],
            data["rotation"],
        )
        times, bounces, ankle_v, Mint, Mext = (
            data["times"],
            data["hits"],
            data["ankle_v"],
            data["Mint"],
            data["Mext"],
        )

        r_img, court_img, mask = (
            to_float32_tensor(r_img),
            to_float32_tensor(court_img),
            to_float32_tensor(mask),
        )
        r_world, rotation, times = (
            to_float32_tensor(r_world),
            to_float32_tensor(rotation),
            to_float32_tensor(times),
        )
        bounces, ankle_pos, ankle_v, Mint, Mext = (
            to_float32_tensor(bounces),
            to_float32_tensor(ankle_position),
            to_float32_tensor(ankle_v),
            to_float32_tensor(Mint),
            to_float32_tensor(Mext),
        )

        # Just return the first bounce --> Matters for RandomStopAugmentation; negative if no bounce in trajectory
        bounces = bounces[0:1]
        return (
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
        )

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
            phi = random.uniform(self.sampled_phis[0], self.sampled_phis[1])
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
            c += np.array([0.0, 0.0, 0])
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
            z_ankle = 0.1 + random.random() * 0.3
        else:
            x_ankle = start_position[0] + sign_x * (random.random() * 0.7)
            y_ankle = start_position[1] + random.random() * 0.8 - 0.4
            z_ankle = 0.1 + random.random() * 0.2
        return np.array([x_ankle, y_ankle, z_ankle])


class RealInferenceDataset(torch.utils.data.Dataset):
    """Dataset for real data inference"""

    def __init__(self, mode, transforms=None):
        self.path = DATA_PATH_REAL_TRAJECTORIES
        self.transforms = transforms
        assert mode in ["val", "test"], "mode must be one of: 'val', 'test'"

        self.data_paths = sorted(
            [
                os.path.join(self.path, foldername)
                for i, foldername in enumerate(os.listdir(self.path))
                if foldername.startswith("trajectory_")
            ]
        )

        self.sequence_len = 120

        self.original_resolution = (1280, 720)

        if mode == "val":
            self.data_paths = self.data_paths[
                : int(0.33 * len(self.data_paths))
            ]  # 33% for validation
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

        court_cam = world2cam(court_points, Mext)
        court_img = cam2img(court_cam, Mint)

        # crop/pad sequences and build mask for valid timesteps
        r_img, max_t = pad_or_crop_sequence(r_img, self.sequence_len, feature_dim=2)
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
        times, bounces, Mint, Mext = (
            to_float32_tensor(times),
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
            ankle_v,
        )


def transform_resolution(data, original_resolution, processing_resolution):
    """Transform coordinates from original resolution to processing resolution.
    Arguments:
        data: dict with keys 'r_img', 'court_img', Mint, ankle_v
        original_resolution: tuple (width, height)
        processing_resolution: tuple (width, height)
    """
    assert (
        "r_img" in data and "court_img" in data and "Mint" in data and "ankle_v" in data
    ), "data must contain keys 'r_img', 'court_img', 'Mint', 'ankle_v'"
    orig_w, orig_h = original_resolution
    proc_w, proc_h = processing_resolution
    if orig_w == proc_w and orig_h == proc_h:
        return data  # no transformation needed
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
