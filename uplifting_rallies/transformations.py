import numpy as np
from uplifting_rallies.helper import HEIGHT, WIDTH
from uplifting_rallies.helper import cam2img, world2cam
from uplifting_rallies.helper import (
    KEYPOINT_VISIBLE,
    KEYPOINT_INVISIBLE,
)


IMG_SIZE = np.array([WIDTH, HEIGHT])


def _active_length(mask):
    return int(np.sum(mask))


class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        """
        transforms (list of objects) – list of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img' and 'court_img'.
        """
        for t in self.transforms:
            data = t(data)
        return data


class RandomizeDetections:
    """
    Randomize image coordinates to simulate noisy detections.
    """

    def __init__(self, std=5):
        """
        seed (int) – Seed for the random number generator.
        """
        self.std = std

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img' and 'court_img'.
        """
        r_img = data["r_img"]
        court_img = data["court_img"]
        r_img[:, :2] = r_img[:, :2] + np.random.normal(
            loc=0, scale=self.std, size=r_img[:, :2].shape
        )
        court_img[:, :2] = court_img[:, :2] + np.random.normal(
            loc=0, scale=self.std, size=court_img[:, :2].shape
        )
        data["r_img"] = r_img
        data["court_img"] = court_img
        return data


class RandomCut:
    """
    Cuts a part out of the trajectory.
    Args:
        min_cut_length (int): Minimum length of the cut part.
        max_cut_length (int): Maximum length of the cut part.
    """

    def __init__(self, min_length=15, max_length=150):
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img' and 'court_img'.
        """
        r_img = data["r_img"]
        r_world = data["r_world"]
        times = data["times"]
        mask = data["mask"]
        blur_times = data["blur_times"]
        rotation = data["rotation"]

        # decide where to cut
        length = _active_length(mask)
        upper, lower = min(length, self.max_length), min(self.min_length, length)
        cut_length = int(round(np.random.random() * (upper - lower) + lower))
        start_ind = np.random.randint(0, length - cut_length + 1)
        cut_slice = slice(start_ind, start_ind + cut_length)

        # create new arrays
        new_r_img = np.zeros_like(r_img)
        new_r_world = np.zeros_like(r_world)
        new_times = np.zeros_like(times)
        new_mask = np.zeros_like(mask)  # 0 is False
        new_rotation = np.zeros_like(rotation)
        new_r_img[0:cut_length] = r_img[cut_slice]
        new_r_world[0:cut_length] = r_world[cut_slice]
        new_times[0:cut_length] = times[cut_slice]
        new_rotation[0:cut_length] = rotation[cut_slice]
        new_mask[0:cut_length] = True

        # shift times (blur time too to ensure that further augmentations work correctly)
        time_shift = times[start_ind]
        new_times[0:cut_length] = new_times[0:cut_length] - time_shift
        blur_times = blur_times - time_shift

        # return data
        data["r_img"] = new_r_img
        data["r_world"] = new_r_world
        data["times"] = new_times
        data["mask"] = new_mask
        data["blur_times"] = blur_times
        data["rotation"] = new_rotation
        return data


class SimulateAce:
    """
    Cuts the rally after the the serve in order to simulate an ace
    """

    def __init__(self, ace_prob=0.1):
        self.ace_prob = ace_prob

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img' and 'court_img'.
        """
        if np.random.random() > self.ace_prob:  # don't do anything
            return data
        r_img = data["r_img"]
        r_world = data["r_world"]
        times = data["times"]
        mask = data["mask"]
        blur_times = data["blur_times"]
        rotation = data["rotation"]

        # decide where to cut
        velocity_z = np.diff(r_world[:, 2])
        bounces = (
            np.where(
                (velocity_z[:-1] < 0) & (velocity_z[1:] > 0) & (r_world[1:-1, 2] < 0.2)
            )[0]
            + 1
        )
        if len(bounces) > 0:
            first_bounce_idx = bounces[0]
        else:
            return data

        velocity_x = np.diff(r_world[:, 0])
        if data["r_world"][0, 0] > 0:
            # Return: x velocity changes from negative to positive
            try:
                return_index = (
                    np.where((velocity_x[:-1] < 0) & (velocity_x[1:] > 0))[0][0] + 1
                )
            except IndexError:
                return data
        else:
            # Return: x velocity changes from positive to negative
            try:
                return_index = (
                    np.where((velocity_x[:-1] > 0) & (velocity_x[1:] < 0))[0][0] + 1
                )
            except IndexError:
                return data

        if return_index < first_bounce_idx:
            return data

        # create new arrays
        new_r_img = np.zeros_like(r_img)
        new_r_world = np.zeros_like(r_world)
        new_times = np.zeros_like(times)
        new_mask = np.zeros_like(mask)  # 0 is False
        new_rotation = np.zeros_like(rotation)
        keep_slice = slice(0, return_index)
        new_r_img[keep_slice] = r_img[keep_slice]
        new_r_world[keep_slice] = r_world[keep_slice]
        new_times[keep_slice] = times[keep_slice]
        new_rotation[keep_slice] = rotation[keep_slice]
        new_mask[0:return_index] = True

        # return data
        data["r_img"] = new_r_img
        data["r_world"] = new_r_world
        data["times"] = new_times
        data["mask"] = new_mask
        data["blur_times"] = blur_times
        data["rotation"] = new_rotation
        return data


class MotionBlur:
    """
    Simulate noisy detections due to motion blur by adding noise along the trajectory in image space.
    Should be applied before RandomizeDetections.
    """

    def __init__(self, blur_strength=0.5):
        # if 1, the motion blur will be chosen from full range between the previous and next frame
        # if 0, there is basically no motion blur
        self.blur_strength = blur_strength
        assert (
            0.1 <= blur_strength < 0.5 or blur_strength == 0
        ), "blur_strength should be in the range [0.1, 0.5) or 0."

    def __call__(self, data):
        if self.blur_strength == 0:
            return data

        r_worlds = data["r_world"]
        r_imgs = data["r_img"]
        Mint = data["Mint"]
        Mext = data["Mext"]

        length = _active_length(data["mask"])
        times = data["times"]
        blur_r_world = data["blur_positions"]
        blur_times = data["blur_times"]

        # Calculate time boundaries before and after.
        before, after = np.copy(times), np.copy(times)
        before[1:length] = times[: length - 1]
        after[: length - 1] = times[1:length]

        b = times[:length] + self.blur_strength * (before[:length] - times[:length])
        a = times[:length] + self.blur_strength * (after[:length] - times[:length])

        # Vectorized binary search for high-resolution index bounds.
        idx_start = np.searchsorted(blur_times, b, side="left")
        idx_end = np.searchsorted(blur_times, a, side="right")

        # Ensure each range has at least one candidate.
        idx_end = np.maximum(idx_start + 1, idx_end)

        # Vectorized random sampling of indices between bounds.
        rand_offsets = np.random.rand(length) * (idx_end - idx_start)
        rand_idx = idx_start + rand_offsets.astype(int)

        # Guard against floating-point edge effects.
        rand_idx = np.clip(rand_idx, 0, len(blur_times) - 1)

        # Fetch all 3D points at once and do batched projection.
        blur_r = blur_r_world[rand_idx]  # Shape: (length, 3)
        blur_r_cam = world2cam(blur_r, Mext)
        blur_r_img = cam2img(blur_r_cam, Mint)

        # Update data arrays in-place.
        r_worlds[:length] = blur_r
        r_imgs[:length, :2] = blur_r_img

        return data


class RandomMissing:
    """The ball was not detected in the image, so we set the ball coordinates to invisible."""

    def __init__(self, randmiss_prob):
        """
        miss_prob (float) – Probability of a miss detection. Should be in the range [0, 1].
        """
        self.randmiss_prob = randmiss_prob

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img' and 'court_img'.
        """
        r_img = data["r_img"]
        mask = data["mask"]

        T = _active_length(mask)
        visibility_probs = np.random.rand(T)
        r_img[:T, 2] = np.where(
            visibility_probs < self.randmiss_prob, KEYPOINT_INVISIBLE, KEYPOINT_VISIBLE
        )

        data["r_img"] = r_img
        data["mask"] = mask
        return data


class NormalizeImgCoords:
    """
    Normalize image coordinates to the range [0, 1] using HEIGHT and WIDTH. Apply transformation after RandomizeDetections!!!
    """

    def __init__(self):
        pass

    def __call__(self, data):
        r_img = data["r_img"]
        court_img = data["court_img"]
        ankle_v = data["ankle_v"]
        r_img[..., :2] = r_img[..., :2] / IMG_SIZE
        court_img[..., :2] = court_img[..., :2] / IMG_SIZE
        ankle_v = ankle_v / HEIGHT
        data["r_img"] = r_img
        data["court_img"] = court_img
        data["ankle_v"] = ankle_v
        return data


class UnNormalizeImgCoords:
    """
    Unnormalize image coordinates to the range [0, WIDTH] and [0, HEIGHT]. Apply transformation after RandomizeDetections!!!
    """

    def __init__(self):
        pass

    def __call__(self, data):
        r_img = data["r_img"]
        court_img = data["court_img"]
        ankle_v = data["ankle_v"]
        r_img[..., :2] = r_img[..., :2] * IMG_SIZE
        court_img[..., :2] = court_img[..., :2] * IMG_SIZE
        ankle_v = ankle_v * HEIGHT
        data["r_img"] = r_img
        data["court_img"] = court_img
        data["ankle_v"] = ankle_v
        return data


def get_transforms(config, mode="train"):
    """
    Get the transforms for the dataset.
    mode (str) – Mode of the dataset. Can be 'train', 'val', or 'test'.
    """
    transforms = []
    if mode == "train":
        transforms.append(SimulateAce(config.ace_prob))
        transforms.append(MotionBlur(config.blur_strength))
        transforms.append(RandomizeDetections(config.randomize_std))
        transforms.append(
            RandomCut(config.randcut_min_length, config.randcut_max_length)
        )
        transforms.append(RandomMissing(config.randmiss_prob))
    transforms.append(NormalizeImgCoords())
    return Compose(transforms)
