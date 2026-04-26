import numpy as np
import einops as eo
import torch
import mujoco
import os

# -------------------------------------
# --- Physical & Simulation Globals ---
# -------------------------------------

# Screen and Camera Configuration
HEIGHT, WIDTH = 1080, 1920
CAMERA_NAME = "main"

# Ball Parameters for Tennis
BALL_RADIUS = 0.0335  # Radius of a type 2 tennis ball in meters.
BALL_MASS = 0.0577  # Average mass of a type 2 tennis ball in kilograms.

# Tennis Court Dimensions (in meters, standard single court)
# Vizualisation in world_coordinates.png in images folder
TENNIS_COURT_LENGTH = 23.774  # baseline to baseline
TENNIS_COURT_WIDTH = 8.233  # sideline to sideline
TENNIS_COURT_DOUBLE_WIDTH = 10.97

# Principle Playing Area (PPA)
# This is the larger plane the ball can interact with, extending beyond the court lines.
PPA_LENGTH = 30.0
PPA_WIDTH = 15.0

# Net Dimensions
NET_TENNIS_WIDTH = TENNIS_COURT_WIDTH + (2 * 0.914)  # Length of the net
NET_TENNIS_HEIGHT = 1.07  # Height of the net at the center
NET_TENNIS_DEPTH = 0.008  # Thickness/width of the net material (e.g., 0.8 cm)

# Derived Court and Net Positions
BASELINE_X_CLOSE = TENNIS_COURT_LENGTH / 2
BASELINE_X_FAR = -TENNIS_COURT_LENGTH / 2
SERVICELINE_X_CLOSE = 6.401
SERVICELINE_X_FAR = -6.401
NET_X_POS = 0.0

SINGLE_SIDELIINE_Y_LEFT = -TENNIS_COURT_WIDTH / 2
SINGLE_SIDELIINE_Y_RIGHT = TENNIS_COURT_WIDTH / 2
DOUBLE_SIDELIINE_Y_LEFT = -TENNIS_COURT_DOUBLE_WIDTH / 2
DOUBLE_SIDELIINE_Y_RIGHT = TENNIS_COURT_DOUBLE_WIDTH / 2
NETPOST_Y_LEFT = -NET_TENNIS_WIDTH / 2
NETPOST_Y_RIGHT = NET_TENNIS_WIDTH / 2

# MuJoCo Simulation Parameters
TIMESTEP = 0.001  # Simulation timestep in seconds. Smaller values increase accuracy but are slower.
MAX_SIMULATION_TIME = 4.0  # Maximum duration of a single trajectory simulation.
FPS = 500  # Frames per second for saving the trajectory data.

# -------------------------------------
# --- Trajectory Analysis Constants ---
# -------------------------------------

# Parameters for Hit Detection in the `_count_hits` function
HIT_DETECTION_Z_THRESHOLD_COURT = (
    BALL_RADIUS + 0.02
)  # Z-height threshold to robustly detect a court bounce (since court is at z=0).
HIT_DETECTION_X_MARGIN = 0.01  # A small margin to avoid detecting hits exactly at the net line (x=NET_X_POS).
HIT_TIME_INTERPOLATION_WEIGHTS = (
    0.75,
    0.25,
)  # Weights for interpolating the exact hit time, combining the interval midpoint and the point of minimum height.

# -------------------------------------
# --- Camera & Visualization Setup ---
# -------------------------------------

# Camera Pose and Intrinsics
# These values define the camera's position, orientation, and lens properties.
# These values where derived from the average of reverse engineered camera extrinsics and intrinsics from the TrackNet dataset.
fx, fy = 1400, 1400
camera_pos = np.array([21.1, 0.0, 10.0])
camera_z = np.array([11.0, 0.0, 8.0])  # camera "looks" in the negative z direction
camera_x = np.array([0, 1, 0])  # right direction - u-axis
camera_y = np.array([-8, 0, 11])  # up direction - negative v-axis

# These points define the keypoints of a single tennis court.
# Position of the keypoints see keypoints-png in images folder
court_points = np.array(
    [
        [BASELINE_X_CLOSE, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 0 close left
        [BASELINE_X_CLOSE, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 1 close right
        [0.0, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 2 center left
        [0.0, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 3 center right
        [BASELINE_X_FAR, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 4 far left
        [BASELINE_X_FAR, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 5 far right
        [0.0, NETPOST_Y_LEFT, 0.0],  # 6 net left bottom
        [0.0, NETPOST_Y_RIGHT, 0.0],  # 7 net right bottom
        [0.0, 0.0, 0.0],  # 8 net center bottom
        [0.0, NETPOST_Y_LEFT, NET_TENNIS_HEIGHT],  # 9 net left top
        [0.0, NETPOST_Y_RIGHT, NET_TENNIS_HEIGHT],  # 10 net right top
        [
            SERVICELINE_X_CLOSE,
            SINGLE_SIDELIINE_Y_LEFT,
            0.0,
        ],  # 11 service line close left
        [SERVICELINE_X_CLOSE, 0.0, 0.0],  # 12 service line close center
        [
            SERVICELINE_X_CLOSE,
            SINGLE_SIDELIINE_Y_RIGHT,
            0.0,
        ],  # 13 service line close right
        [SERVICELINE_X_FAR, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 14 service line far left
        [SERVICELINE_X_FAR, 0.0, 0.0],  # 15 service line far center
        [SERVICELINE_X_FAR, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 16 service line far right
        [0.0, 0.0, 0.914],  # 17 net center top
        [BASELINE_X_CLOSE, DOUBLE_SIDELIINE_Y_LEFT, 0.0],  # 18 double line close left
        [BASELINE_X_CLOSE, DOUBLE_SIDELIINE_Y_RIGHT, 0.0],  # 19 double line close right
        [BASELINE_X_FAR, DOUBLE_SIDELIINE_Y_LEFT, 0.0],  # 20 double line far left
        [BASELINE_X_FAR, DOUBLE_SIDELIINE_Y_RIGHT, 0.0],  # 21 double line far right
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
    (9, 17),  # net
    (17, 10),
    (10, 7),  # net
    (12, 8),  # center service line
    (8, 15),  # center service line
    (11, 12),  # close service line
    (12, 13),  # close service line
    (14, 15),  # far service line
    (15, 16),  # far service line
    (18, 19),
    (18, 20),
    (20, 21),
    (19, 21),
]


# --- MuJoCo XML Definition ---
XML = f"""
<mujoco>
  <option cone="elliptic" gravity="0 0 -9.81" integrator="implicitfast" timestep="{TIMESTEP}" density="1.225" viscosity="0.00001789" />
  <asset>
    <material name="ball_material" reflectance="0" rgba="1 1 1 1"/>
    <material name="net_material" reflectance="0"  texuniform="false" texrepeat="1 1" />
    <material name="court_material" reflectance="0.05" texrepeat="1 1" texuniform="true"/>
  </asset>
  <visual>
    <global offheight="{HEIGHT}" offwidth="{WIDTH}"/>
  </visual>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="-2 -2 10" dir="0.1 0.1 -1"/>
    <geom name="floor_geom" type="plane" size="800 800 0.1" rgba="0.6 0.6 0.6 1" pos="0 0 -0.1" material="ball_material"/>
    <geom name="court_plane_geom" type="plane" size="30 30 1" rgba="0.6 0.6 0.6 1" pos="0 0 0" material="court_material"/>
    <body name="ball_body">
      <freejoint/>  
      <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}" material="ball_material" mass="{BALL_MASS}" fluidshape="ellipsoid" fluidcoef="0.6 0.6 0.6 1.0 0.2"/>
    </body>
    <geom name="net_geom" type="box" pos="0 0 {NET_TENNIS_HEIGHT / 2}" size="{NET_TENNIS_DEPTH / 2} {NET_TENNIS_WIDTH / 2} {NET_TENNIS_HEIGHT / 2}" material="net_material" rgba="1 1 1 0.6" />
    <camera name="{CAMERA_NAME}"
            focal="{fx/WIDTH} {fy/HEIGHT}"
            resolution="{WIDTH} {HEIGHT}"
            sensorsize="1 1"
            pos="{camera_pos[0]} {camera_pos[1]} {camera_pos[2]}"
            mode="fixed"
            xyaxes= "{camera_x[0]} {camera_x[1]} {camera_x[2]} {camera_y[0]} {camera_y[1]} {camera_y[2]}"/>
  </worldbody>
  <default>
    <pair solref="-48688 -21.52" solreffriction="0.0 -100.0" friction="0.7 0.7 0.005 0.04 0.04" solimp="0.98 0.99 0.001 0.5 2"/>
  </default>
  <contact>
    <pair geom1="ball_geom" geom2="court_plane_geom"/>
    <pair geom1="ball_geom" geom2="net_geom"/>
  </contact>
</mujoco>
"""

# --- Utility Functions (Unchanged) ---


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
    """
    if isinstance(x, np.ndarray):
        ones = np.ones((*shape[:-1], 1))
        return np.concatenate([x, ones], axis=-1)
    elif isinstance(x, torch.Tensor):
        ones = torch.ones((*shape[:-1], 1), device=x.device)
        return torch.cat([x, ones], dim=-1)
    else:
        raise TypeError("Input must be either a numpy array or a torch tensor.")


def _calc_cammatrices(data, camera_name):
    """Calculate the camera extrinsic and intrinsic matrices from the MuJoCo data, based on camera name."""
    camera_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    weird_R = eo.rearrange(data.cam_xmat[camera_id], "(i j) -> i j", i=3, j=3).T
    R = np.eye(3)
    R[0, :] = weird_R[0, :]
    R[1, :] = -weird_R[1, :]
    R[2, :] = -weird_R[2, :]
    cam_pos = data.cam_xpos[camera_id]
    t = -np.dot(R, cam_pos)
    ex_mat = np.eye(4)
    ex_mat[:3, :3] = R
    ex_mat[:3, 3] = t
    fx = (
        data.model.cam_intrinsic[camera_id][0]
        / data.model.cam_sensorsize[camera_id][0]
        * data.model.cam_resolution[camera_id][0]
    )
    fy = (
        data.model.cam_intrinsic[camera_id][1]
        / data.model.cam_sensorsize[camera_id][1]
        * data.model.cam_resolution[camera_id][1]
    )
    cx = (WIDTH - 1) / 2
    cy = (HEIGHT - 1) / 2
    in_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1], [0, 0, 0]])
    return ex_mat, in_mat


def _count_hits(positions, direction, fps=FPS):
    """
    Counts bounces on own side, opponent side, and out.
    """
    hits_own = []
    hits_opponent = []
    hits_out = []

    # Define X-axis checks based on direction and net position
    if direction == "far_to_close":
        # Own side is negative x, opponent side is positive x
        opposite_side_x_check = (
            lambda x: x > NET_X_POS + HIT_DETECTION_X_MARGIN and x < BASELINE_X_CLOSE
        )
        own_side_x_check = (
            lambda x: x > BASELINE_X_FAR and x < NET_X_POS - HIT_DETECTION_X_MARGIN
        )
    else:  # 'close_to_far'
        # Own side is positive x, opponent side is negative x
        opposite_side_x_check = (
            lambda x: x > BASELINE_X_FAR and x < NET_X_POS - HIT_DETECTION_X_MARGIN
        )
        own_side_x_check = (
            lambda x: x > NET_X_POS + HIT_DETECTION_X_MARGIN and x < BASELINE_X_CLOSE
        )

    # Y-axis check (within court width)
    y_check = lambda y: y > SINGLE_SIDELIINE_Y_LEFT and y < SINGLE_SIDELIINE_Y_RIGHT

    binary_mask_z = np.array(
        [pos[2] < HIT_DETECTION_Z_THRESHOLD_COURT for pos in positions]
    )
    binary_mask_y = np.array([y_check(pos[1]) for pos in positions])
    binary_mask_x_opponent = np.array(
        [opposite_side_x_check(pos[0]) for pos in positions]
    )
    binary_mask_x_own = np.array([own_side_x_check(pos[0]) for pos in positions])
    binary_mask_opponent = binary_mask_z & binary_mask_y & binary_mask_x_opponent
    binary_mask_own = binary_mask_z & binary_mask_y & binary_mask_x_own
    binary_mask_out = binary_mask_z & ~(binary_mask_opponent | binary_mask_own)

    positions = np.array(positions)
    w1, w2 = HIT_TIME_INTERPOLATION_WEIGHTS

    for mask, hit_list in [
        (binary_mask_opponent, hits_opponent),
        (binary_mask_own, hits_own),
        (binary_mask_out, hits_out),
    ]:
        start, end = None, None
        for i, b in enumerate(mask):
            if i == 0 and b:
                start = i
            elif b and (
                i == 0 or not mask[i - 1]
            ):  # Ensure start is set even if i-1 is out of bounds
                start = i
            if not b and mask[i - 1] and i != 0:
                end = i - 1
                # Interpolate hit time based on a weighted average of the interval midpoint and the time of minimum height
                mid_point_time = (end + start) / 2 / fps
                min_height_time = (
                    np.argmin(positions[start : end + 1, 2]) + start
                ) / fps
                hit_list.append(w1 * mid_point_time + w2 * min_height_time)

    hits_out.extend(hits_own)
    hits_out.sort()

    all_hits = hits_opponent + hits_out
    all_hits.sort()
    return hits_opponent, hits_out, all_hits


if __name__ == "__main__":
    pass
