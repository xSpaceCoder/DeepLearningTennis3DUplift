import numpy as np
from uplifting.helper import HEIGHT, WIDTH
from uplifting.helper import cam2img, world2cam
from uplifting.helper import KEYPOINT_INVISIBLE


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
        r_img = r_img + np.random.normal(loc=0, scale=self.std, size=r_img.shape)
        court_img[:, :2] = court_img[:, :2] + np.random.normal(
            loc=0, scale=self.std, size=court_img[:, :2].shape
        )
        data["r_img"] = r_img
        data["court_img"] = court_img
        return data


class RandomStop:
    """
    Randomly stop the sequence after the bounce to simulate that the oposing player hit the ball.
    """

    def __init__(self, stop_prob=0.5):
        """
        stop_prob (float) – Probability to stop the sequence after the bounce.
        """
        self.stop_prob = stop_prob

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img' and 'court_img'.
        """
        if np.random.random() > self.stop_prob:
            return data

        hits = data["hits"]
        times = data["times"]
        mask = data["mask"]
        r_img = data["r_img"]
        r_world = data["r_world"]

        hit_time = hits[0]  # take the first bounce
        if hit_time <= 0:  # there is no bounce
            return data

        hit_ind = np.argmin(np.abs(times - hit_time))
        seq_len = _active_length(mask)
        if seq_len - hit_ind < 4:  # minimum sequence length after hit
            return data

        len_after_hit = np.random.randint(4, seq_len - hit_ind + 1)
        mask[hit_ind + len_after_hit :] = False
        invalid_mask = ~mask

        # set coordinates to 0 after the new end
        r_img[invalid_mask] = 0
        r_world[invalid_mask] = 0
        times[invalid_mask] = 0

        data["mask"] = mask
        data["r_img"] = r_img
        data["r_world"] = r_world
        data["times"] = times
        return data


class MotionBlur:
    """
    Simulate noisy detections due to motion blur by adding noise along the trajectory in image space. Should be applied before RandomizeDetections.
    """

    def __init__(self, blur_strength=0.5):
        # if 1, the motion blur will be chosen from full range between the previous and next frame
        # if 0, there is basically no motion blur
        self.blur_strength = blur_strength
        assert (
            0.1 <= blur_strength < 0.5 or blur_strength == 0
        ), "blur_strength should be in the range [0.1, 0.5) or 0."

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img' and 'court_img'.
        """
        if self.blur_strength == 0:
            return data
        r_worlds = data["r_world"]  # Shape (T, 3)
        r_imgs = data["r_img"]  # Shape (T, 2)
        Mint = data["Mint"]
        Mext = data["Mext"]
        mask = data["mask"]
        length = _active_length(mask)
        times = data["times"]  # Shape (T,)
        blur_r_world = data["blur_positions"]  # Shape (T, 3)
        blur_times = data["blur_times"]  # Shape (T,)
        # easily access the time of the previous and next frame
        before, after = times.copy(), times.copy()
        before[1:length] = times[: length - 1]
        after[: length - 1] = times[1:length]
        # evaluate the time boundary before and after (with the strength parameter)
        before[:length] = (
            times[:length] + self.blur_strength * (before - times)[:length]
        )
        after[:length] = times[:length] + self.blur_strength * (after - times)[:length]
        # iterate over trajectory (TODO: this is slow, can be optimized)
        for i in range(length):
            b, a = before[i], after[i]  # Scalars
            # get all the coordinates that are in the range of the blur
            in_blur_range = (blur_times >= b) & (blur_times <= a)
            valid_blur_times = blur_times[in_blur_range]
            valid_blur_r = blur_r_world[in_blur_range]
            # choose a random coordinate that is inside the blur range
            blur_t = np.random.choice(valid_blur_times)
            blur_r = valid_blur_r[valid_blur_times == blur_t]
            blur_r = blur_r[0]
            blur_r_cam = world2cam(blur_r, Mext)
            blur_r_img = cam2img(blur_r_cam, Mint)
            # update the r and r_img
            r_worlds[i] = blur_r
            r_imgs[i] = blur_r_img
        data["r_world"] = r_worlds
        data["r_img"] = r_imgs
        return data


class RandomDetection:
    """
    Instead of the ball or court keypoint, a random image point is chosen.
    """

    def __init__(self, randdet_prob):
        """
        miss_prob (float) – Probability of a miss detection. Should be in the range [0, 1].
        """
        self.randdet_prob = randdet_prob

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img' and 'court_img'.
        """
        r_img = data["r_img"]
        court_img = data["court_img"]
        mask = data["mask"]

        T = _active_length(mask)
        ball_replace = np.random.random(T) < self.randdet_prob
        ball_replace_count = int(np.sum(ball_replace))
        if ball_replace_count > 0:
            # choose random points in the image
            r_img_indices = np.flatnonzero(ball_replace)
            r_img[r_img_indices] = np.random.rand(ball_replace_count, 2) * IMG_SIZE

        court_replace = np.random.random(court_img.shape[0]) < self.randdet_prob
        court_replace_count = int(np.sum(court_replace))
        if court_replace_count > 0:
            # choose random points in the image
            court_img[court_replace, :2] = (
                np.random.rand(court_replace_count, 2) * IMG_SIZE
            )
        data["r_img"] = r_img
        data["court_img"] = court_img
        return data


class RandomMissing:
    """The ball was not detected in the image, so we remove the ball coordinates."""

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
        r_world = data["r_world"]
        times = data["times"]
        mask = data["mask"]

        new_mask = np.zeros_like(mask)
        new_r_img = np.zeros_like(r_img)
        new_r_world = np.zeros_like(r_world)
        new_times = np.zeros_like(times)

        T = _active_length(mask)
        keep = np.random.random(T) >= self.randmiss_prob
        kept_count = int(np.sum(keep))

        new_mask[:kept_count] = True
        new_r_img[:kept_count] = r_img[:T][keep]
        new_r_world[:kept_count] = r_world[:T][keep]
        new_times[:kept_count] = times[:T][keep]

        # update the mask, r_img, r_world, and times
        data["mask"] = new_mask
        data["r_img"] = new_r_img
        data["r_world"] = new_r_world
        data["times"] = new_times
        return data


class CourtKPMissing:
    """Sometimes a court keypoint is not detected. Thus, we randomly set the visibility to 0."""

    def __init__(self, tablemiss_prob):
        """
        miss_prob (float) – Probability of a miss detection. Should be in the range [0, 1].
        """
        self.tablemiss_prob = tablemiss_prob

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img' and 'court_img'.
        """
        court_img = data["court_img"]
        missing = np.random.random(court_img.shape[0]) < self.tablemiss_prob
        missing_count = int(np.sum(missing))
        if missing_count > 0:
            court_img[missing, 2] = KEYPOINT_INVISIBLE
            # Set the coordinates to random values within the image.
            court_img[missing, :2] = np.random.rand(missing_count, 2) * IMG_SIZE
        data["court_img"] = court_img
        return data


class TableMissing(CourtKPMissing):
    """Backward-compatible alias for CourtKPMissing."""

    pass


class BounceMissing:
    """Sometimes a trajectory is retuned before the ball bounces. Thus, we randomly cut the trajectory off before the bounce."""

    def __init__(self, bouncemiss_prob):
        """
        bounce_prob (float) – Probability of a missing bounce. Should be in the range [0, 1].
        """
        self.bouncemiss_prob = bouncemiss_prob

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img' and 'court_img'.
        """
        r_img = data["r_img"]
        r_world = data["r_world"]
        times = data["times"]
        mask = data["mask"]
        hits = data["hits"]

        if np.random.random() < self.bouncemiss_prob:
            # verify that a serves are not shortend
            if not (np.abs(r_world[0, 0]) > 10 and r_world[0, 2] > 2):
                # look for intervall to cutoff the trajectory
                # start is when the ball has crossed the net (sign of x changes)
                sign_changes = np.where(np.diff(np.sign(r_world[:, 0])) != 0)[0]
                if len(sign_changes) > 0:
                    start_idx = sign_changes[0] + 1
                else:
                    return data
                start_idx = np.max(
                    [start_idx, 30]
                )  # at least index 30 (this corresponds to reaction time of players)
                # end is the first bounce
                end_idx = np.argmin(np.abs(times - hits[0])) - 1

                if start_idx < end_idx:
                    cutoff_idx = np.random.randint(start_idx, end_idx)
                else:
                    return data

                mask[cutoff_idx:] = False
                times[cutoff_idx:] = np.float64(0.0)
                r_img[cutoff_idx:] = [0.0, 0.0]
                r_world[cutoff_idx:] = [0.0, 0.0, 0.0]
                hits = np.array([-1.0])
                data["mask"] = mask
                data["r_img"] = r_img
                data["r_world"] = r_world
                data["hits"] = hits
                data["times"] = times
        return data


class DeleteOOF:
    """
    Delete out-of-frame ball detections
    """

    def __init__(self):
        pass

    def __call__(self, data):
        """
        data (dict) – Dictionary with keys 'r_img', 'times', 'mask' and 'r_world' (for synthetic validation data only).
        """
        val_synth = "r_world" in data
        r_img = data["r_img"]
        times = data["times"]
        mask = data["mask"]

        new_mask = np.zeros_like(mask)
        new_r_img = np.zeros_like(r_img)
        new_times = np.zeros_like(times)

        if val_synth:
            r_world = data["r_world"]
            new_r_world = np.zeros_like(r_world)

        T = _active_length(mask)
        in_frame = (
            (r_img[:T, 0] >= 0)
            & (r_img[:T, 0] <= WIDTH)
            & (r_img[:T, 1] >= 0)
            & (r_img[:T, 1] <= HEIGHT)
        )
        kept_count = int(np.sum(in_frame))

        new_mask[:kept_count] = True
        new_r_img[:kept_count] = r_img[:T][in_frame]
        new_times[:kept_count] = times[:T][in_frame]
        if val_synth:
            new_r_world[:kept_count] = r_world[:T][in_frame]

        # update the mask, r_img, r_world, and times
        data["mask"] = new_mask
        data["r_img"] = new_r_img
        data["times"] = new_times
        if val_synth:
            data["r_world"] = new_r_world
        return data


class Identity:
    """
    Identity transform.
    """

    def __init__(self):
        pass

    def __call__(self, data):
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
        r_img = r_img[..., :2] / IMG_SIZE
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
        r_img = r_img * IMG_SIZE
        court_img[..., :2] = court_img[..., :2] * IMG_SIZE
        ankle_v[..., :2] = ankle_v * HEIGHT
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
        transforms.append(MotionBlur(config.blur_strength))
        transforms.append(RandomizeDetections(config.randomize_std))
        transforms.append(RandomStop(config.stop_prob))
        transforms.append(RandomDetection(config.randdet_prob))
        transforms.append(RandomMissing(config.randmiss_prob))
        transforms.append(TableMissing(config.tablemiss_prob))
    if mode == "val":
        transforms.append(DeleteOOF())
    transforms.append(NormalizeImgCoords())
    return Compose(transforms)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
