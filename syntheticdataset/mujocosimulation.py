# %%
import mujoco
import numpy as np
import random
import math
import os
import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt

# Import from the new refactored helper file
from syntheticdataset.helper import (
    cam2img,
    world2cam,
    _calc_cammatrices,
    _count_hits,
    XML,
    HEIGHT,
    WIDTH,
    TENNIS_COURT_LENGTH,
    TENNIS_COURT_WIDTH,
    NET_TENNIS_HEIGHT,
    NET_TENNIS_WIDTH,
    TIMESTEP,
    MAX_SIMULATION_TIME,
    FPS,
    CAMERA_NAME,
    BALL_RADIUS,
    PPA_LENGTH,
    PPA_WIDTH,
    BALL_RADIUS,
    SERVICELINE_X_CLOSE,
    SINGLE_SIDELIINE_Y_LEFT,
    SINGLE_SIDELIINE_Y_RIGHT,
    court_connections,
    court_points,
)


# --- Constants for Trajectory Generation and Validation ---

INIT_POS_RANGES = {
    # based on scientific literature for pro players
    # - added margin for non-pro players and to ensure enough valid trajectories
    "serve": {
        "x": (TENNIS_COURT_LENGTH / 2 - 1.0, TENNIS_COURT_LENGTH / 2 + 1),
        "y": (-TENNIS_COURT_WIDTH / 2 + 0.5, TENNIS_COURT_WIDTH / 2 - 0.5),
        "z": (2.00, 3.00),
    },
    "groundstroke": {
        "x": (0.2, PPA_LENGTH - 0.5),
        "y": (-PPA_WIDTH / 2 + 0.5, PPA_WIDTH / 2 - 0.5),
        "z": (0.2, 1.5),
    },
    "volley": {
        "x": (0.1, SERVICELINE_X_CLOSE + 1),
        "y": (-TENNIS_COURT_WIDTH / 2, TENNIS_COURT_WIDTH / 2),
        "z": (0.2, 1.8),
    },
    "smash": {
        "x": (0.5, TENNIS_COURT_LENGTH / 2 + 1),
        "y": (-TENNIS_COURT_WIDTH / 2 - 1, TENNIS_COURT_WIDTH / 2 + 1),
        "z": (2.0, 2.8),
    },
    "lob": {
        "x": (0.5, PPA_LENGTH / 2),
        "y": (-PPA_WIDTH / 2 - 2, PPA_WIDTH / 2 + 2),
        "z": (0.2, 1.5),
    },
    "short": {
        "x": (0.2, PPA_LENGTH / 2),
        "y": (-PPA_WIDTH / 2 - 2, PPA_WIDTH / 2 + 2),
        "z": (0.2, 2.5),
    },
    "toss": {
        "x": (TENNIS_COURT_LENGTH / 2 - 0.5, TENNIS_COURT_LENGTH / 2 + 1),
        "y": (-TENNIS_COURT_WIDTH / 2 + 0.5, TENNIS_COURT_WIDTH / 2 - 0.5),
        "z": (1.00, 1.80),
    },
}
INIT_VEL_SPEED_RANGE = (12.0, 40.0)  # from scientific literature
INIT_VEL_SPEED_RANGE_SERVE = (30.0, 55.0)  # from scientific literature for pro players
INIT_VEL_PHI_DEVIATION_DEG = 60.0
INIT_VEL_THETA_DEVIATION_DEG = {"below": (25.0, 60.0), "above": (25.0, 60.0)}
INIT_X_ANG_VEL_RANGE = (-50.0, 50.0)
INIT_Y_ANG_VEL_RANGE = (-500, 500.0)
INIT_Z_ANG_VEL_RANGE = (-50.0, 50.0)

THROW_VEL_X_RANGE = (0, 2)
THROW_VEL_Y_RANGE = (-0.5, 0.5)  # "Nearly vertical"
THROW_VEL_Z_RANGE = (4, 8)  # ball toss height between 2 and 4 m
THROW_ANG_VEL_RANGE = (-2.0, 2.0)  # Low spin from open palm
MIN_THROW_HEIGHT = 2.00  # at least 2 m height for racket ball contact

# avg time ball is in air during grand slam ground strokes is 1.3
MIN_TRAJ_DURATION_SEC = 0.5
MIN_TRAJ_LEN_FRAMES = int(round(MIN_TRAJ_DURATION_SEC * FPS))
MIN_TRAJ_CUT_TIME_RATIO = 0.2
# International Tennis federation requires a free height at net of 9 meters for indoor tennis courts
# - added margin to represent also outside courts
LOB_MAX_HEIGHT_BALL = 15
# for a ball to not be consideres a lob
MAX_HEIGHT_BALL = 5
NET_CLEARANCE_X_MARGIN = BALL_RADIUS + 0.02
OOB_DEFINITIONS = (PPA_LENGTH / 2, PPA_WIDTH / 2, -1.0)

# Save trajectories in chunks to reduce memory usage
CHUNK_SIZE = 500


def _init_simulation(seed, mode, direction, model=None):
    """Initialize simulation state; reuse provided model if given to avoid rebuilding XML each time."""
    rng_py = random.Random(seed)
    model = model if model is not None else mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    sign_x = -1 if direction == "far_to_close" else 1
    r = np.empty(3, dtype=np.float64)
    if "serve" in mode:
        ranges = INIT_POS_RANGES["serve"]
        r[0] = rng_py.uniform(*ranges["x"]) * sign_x
        r[1] = rng_py.uniform(*ranges["y"])
        r[2] = rng_py.uniform(*ranges["z"])
    elif "groundstroke" in mode:
        ranges = INIT_POS_RANGES["groundstroke"]
        r[0] = rng_py.uniform(*ranges["x"]) * sign_x
        r[1] = rng_py.uniform(*ranges["y"])
        r[2] = rng_py.uniform(*ranges["z"])
    elif "volley" in mode:
        ranges = INIT_POS_RANGES["volley"]
        r[0] = rng_py.uniform(*ranges["x"]) * sign_x
        r[1] = rng_py.uniform(*ranges["y"])
        r[2] = rng_py.uniform(*ranges["z"])
    elif "smash" in mode:
        ranges = INIT_POS_RANGES["smash"]
        r[0] = rng_py.uniform(*ranges["x"]) * sign_x
        r[1] = rng_py.uniform(*ranges["y"])
        r[2] = rng_py.uniform(*ranges["z"])
    elif "lob" in mode:
        ranges = INIT_POS_RANGES["lob"]
        r[0] = rng_py.uniform(*ranges["x"]) * sign_x
        r[1] = rng_py.uniform(*ranges["y"])
        r[2] = rng_py.uniform(*ranges["z"])
    elif "short" in mode:
        ranges = INIT_POS_RANGES["short"]
        r[0] = rng_py.uniform(*ranges["x"]) * sign_x
        r[1] = rng_py.uniform(*ranges["y"])
        r[2] = rng_py.uniform(*ranges["z"])
    else:
        ranges = INIT_POS_RANGES["toss"]
        r[0] = rng_py.uniform(*ranges["x"]) * sign_x
        r[1] = rng_py.uniform(*ranges["y"])
        r[2] = rng_py.uniform(*ranges["z"])

        # tosses have a different velocity and angular velocity
        v = np.empty(3, dtype=np.float64)
        v[0] = rng_py.uniform(*THROW_VEL_X_RANGE) * sign_x * -1
        v[1] = rng_py.uniform(*THROW_VEL_Y_RANGE)
        v[2] = rng_py.uniform(*THROW_VEL_Z_RANGE)
        w = np.empty(3, dtype=np.float64)
        w[0] = rng_py.uniform(*THROW_ANG_VEL_RANGE)
        w[1] = rng_py.uniform(*THROW_ANG_VEL_RANGE)
        w[2] = rng_py.uniform(*THROW_ANG_VEL_RANGE)

    if mode != "toss":
        # calculation for phi (how high over net) and theta (right, left)

        # 1. Define reference points:
        match mode:
            case "serve":
                end_1 = np.array([SERVICELINE_X_CLOSE - 0.5, 0.0])
                y = TENNIS_COURT_WIDTH / 2 if r[1] < 0 else -TENNIS_COURT_WIDTH / 2
                end_2 = np.array([SERVICELINE_X_CLOSE - 0.5, y])
            case "short":
                end_1 = np.array([1.0, SINGLE_SIDELIINE_Y_RIGHT])
                end_2 = np.array([1.0, SINGLE_SIDELIINE_Y_LEFT])
            case _:
                end_1 = np.array([SERVICELINE_X_CLOSE - 1, SINGLE_SIDELIINE_Y_RIGHT])
                end_2 = np.array([SERVICELINE_X_CLOSE - 1, SINGLE_SIDELIINE_Y_LEFT])

        # 2. calcualte delta_x = End_x - Start_x, delta_y = End_y - Start_y and delta_z = net_z - Start_z
        delta_x = np.abs(end_1[0]) + np.abs(r[0])
        delta_y_1 = end_1[1] - r[1]
        delta_y_2 = end_2[1] - r[1]

        # a lob has to be at least 3.5 m high at the net
        if mode == "lob":
            net_z = 3.5
        else:
            net_z = NET_TENNIS_HEIGHT + BALL_RADIUS + 0.2
        delta_z = net_z - r[2]

        # 3. calculate base_phi in degree
        base_phi_1 = np.rad2deg(math.atan2(delta_y_1, delta_x))
        base_phi_2 = np.rad2deg(math.atan2(delta_y_2, delta_x))
        base_theta = 90 - np.rad2deg(
            math.atan2(delta_z, np.abs(r[0]))
        )  # polar angle (0°=up, 90°=horizontal, 180°=down)

        # 4. adapt angle based on incomming or outgoing direction
        phi_1 = 180 - base_phi_1 if direction == "close_to_far" else base_phi_1
        phi_2 = 180 - base_phi_2 if direction == "close_to_far" else base_phi_2

        # 5. choose phi randomly between phi_1 and phi_2
        phi = np.deg2rad(rng_py.uniform(phi_1, phi_2))
        if mode == "serve" or mode == "smash":
            theta = np.deg2rad(rng_py.uniform(base_theta, 80))
        elif mode == "lob":
            theta = np.minimum(
                np.deg2rad(45), (np.deg2rad(rng_py.uniform(base_theta, 5)))
            )
        else:
            theta = np.deg2rad(rng_py.uniform(base_theta, 45))

        speed = rng_py.uniform(*INIT_VEL_SPEED_RANGE)
        v = np.empty(3, dtype=np.float64)
        v[0] = speed * math.sin(theta) * math.cos(phi)
        v[1] = speed * math.sin(theta) * math.sin(phi)
        v[2] = speed * math.cos(theta)
        w = np.zeros(3, dtype=np.float64)
        w[0] = -rng_py.uniform(*INIT_X_ANG_VEL_RANGE)
        w[1] = -rng_py.uniform(*INIT_Y_ANG_VEL_RANGE)
        w[2] = -rng_py.uniform(*INIT_Z_ANG_VEL_RANGE)
    data.qpos[0:3] = r
    data.qvel[0:3] = v
    data.qvel[3:6] = w
    return model, data


def find_valid_trajectories_worker(seeds_and_mode):
    seeds, mode, direction = seeds_and_mode
    valid_trajectory_data = []
    out_trajectory_data = []
    # Build model once per worker to avoid repeated XML parsing
    worker_model = mujoco.MjModel.from_xml_string(XML)
    for seed in seeds:
        model, data = _init_simulation(seed, mode, direction, model=worker_model)
        mujoco.mj_step(model, data)
        discard_trajectory = False

        # Calculate camera matrices
        ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA_NAME)

        positions, velocities, rotations, times, r_imgs = (
            [],
            [],
            [],
            [],
            [],
        )
        next_save_time = 0.0
        out = False
        while next_save_time < MAX_SIMULATION_TIME:
            steps = round((next_save_time - data.time) / TIMESTEP)
            mujoco.mj_step(model, data, steps)

            # Check if ball is out of Playable Area (PPA)
            if mode != "toss":
                oob_x, oob_y, _ = OOB_DEFINITIONS
                if abs(data.qpos[0]) > oob_x or abs(data.qpos[1]) > oob_y:
                    break
            else:
                if steps > 0:
                    if (abs(data.qpos[0]) > abs(positions[0][0])) or (
                        abs(data.qpos[0] - positions[0][0]) > 1.0
                    ):  # ball moved more than 1 m in x-direction during toss
                        discard_trajectory = True
                        break
                    if (
                        abs(data.qpos[1] - positions[0][1]) > 0.4
                    ):  # ball moved more than 0.4 m in y-direction during toss
                        discard_trajectory = True
                        break
                    if data.qpos[2] < positions[0][2]:
                        break  # Toss is over (ball fell back to hand height)

            # Check if ball exceeded maximum height
            if (data.qpos[2] > MAX_HEIGHT_BALL) and (mode != "lob"):
                discard_trajectory = True
                break
            elif (data.qpos[2] > LOB_MAX_HEIGHT_BALL) and (mode == "lob"):
                discard_trajectory = True
                break

            # Use pre-calculated camera matrices for view checking
            r_cam = world2cam(data.qpos[0:3].copy(), ex_mat)
            r_img = cam2img(r_cam, in_mat[:3, :3])

            # Check if ball is within camera view
            if mode != "serve" and mode != "lob" and mode != "toss":
                if not np.all((r_img >= 0) & (r_img < np.array([WIDTH, HEIGHT]))):
                    break

            # Append data
            positions.append(data.qpos[0:3].copy())
            velocities.append(data.qvel[0:3].copy())
            rotations.append(data.qvel[3:6].copy())
            times.append(next_save_time)
            r_imgs.append(r_img)
            next_save_time += 1 / FPS

        if discard_trajectory:
            discard_trajectory = False
            positions, velocities, rotations, times, r_imgs, ex_mat, in_mat = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            continue
        # Check duration of trajectory
        if len(positions) < MIN_TRAJ_LEN_FRAMES:
            positions, velocities, rotations, times, r_imgs, ex_mat, in_mat = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            continue

        # determine hits
        hits_opponent, hits_out, all_hits = _count_hits(positions, direction)

        if mode == "toss":
            if not positions:
                continue  # Skip if list is empty
            apex_height = np.max(np.array(positions)[:, 2])
            is_valid_height = apex_height > MIN_THROW_HEIGHT
            is_valid_bounces = len(all_hits) == 0

            if is_valid_height and is_valid_bounces:
                valid_trajectory_data.append(
                    {
                        "positions": np.array(positions),
                        "velocities": np.array(velocities),
                        "rotations": np.array(rotations),
                        "times": np.array(times),
                        "Mext": ex_mat[np.newaxis, :, :],
                        "Mint": in_mat[:3, :3][np.newaxis, :, :],
                        "bounces": np.array(hits_opponent),
                        "seed": seed,
                    }
                )
            continue  # Skip all other validation

        if len(all_hits) == 0:
            positions, velocities, rotations, times, r_imgs, ex_mat, in_mat = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            continue

        if mode == "short":
            # for short: check if first hit is with in the T service boxes
            bounce_index = np.sum(np.where(np.array(times) < all_hits[0], 1, 0)) - 1
            position_hit = positions[bounce_index]
            if abs(position_hit[0]) > SERVICELINE_X_CLOSE:
                positions, velocities, rotations, times, r_imgs, ex_mat, in_mat = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                continue

        if (len(hits_out) > 0) and (all_hits[0] == hits_out[0]):
            out = True

        if mode == "serve":
            # only for serve: check if first hit is in correct service box
            bounce_index = np.sum(np.where(np.array(times) < all_hits[0], 1, 0)) - 1
            position_hit = positions[bounce_index]
            if not (
                abs(position_hit[1]) < TENNIS_COURT_WIDTH / 2
                and abs(position_hit[0]) < SERVICELINE_X_CLOSE
            ):
                out = True

        # Cut trajectory before second hit (if exists)
        cut_index = -1
        if len(all_hits) > 1:
            cut_index = np.sum(np.where(np.array(times) < all_hits[1], 1, 0)) - 1
        hits_out = []
        hits_opponent = all_hits[0]
        positions, velocities, rotations, times = (
            positions[:cut_index],
            velocities[:cut_index],
            rotations[:cut_index],
            times[:cut_index],
        )
        if mode == "serve":
            r_imgs = r_imgs[:cut_index]
            num_valid_frames = np.sum(
                np.all(
                    (r_imgs >= np.array([0, 0])) & (r_imgs < np.array([WIDTH, HEIGHT])),
                    axis=1,
                )
            )
            if num_valid_frames / len(r_imgs) < 0.8:
                positions, velocities, rotations, times, r_imgs, ex_mat, in_mat = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                continue

        # Check duration of trajectory
        if len(positions) < MIN_TRAJ_LEN_FRAMES:
            positions, velocities, rotations, times, r_imgs, ex_mat, in_mat = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            continue

        # Check net clearance
        positions_arr = np.array(positions)
        close_to_net_mask = np.abs(positions_arr[:, 0]) < NET_CLEARANCE_X_MARGIN
        if np.any(close_to_net_mask):
            heights_close_to_net = positions_arr[close_to_net_mask, 2]
            widths_close_to_net = positions_arr[close_to_net_mask, 1]
            if (
                np.max(heights_close_to_net) < NET_TENNIS_HEIGHT
                and np.min(np.abs(widths_close_to_net)) < NET_TENNIS_WIDTH / 2
            ):
                positions, velocities, rotations, times, r_imgs, ex_mat, in_mat = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                continue
        # Prepare data of valid trajectory
        if out:
            out_trajectory_data.append(
                {
                    "positions": np.array(positions),
                    "velocities": np.array(velocities),
                    "rotations": np.array(rotations),
                    "times": np.array(times),
                    "Mext": ex_mat[np.newaxis, :, :],
                    "Mint": in_mat[:3, :3][np.newaxis, :, :],
                    "bounces": np.array(hits_opponent),
                    "seed": seed,
                }
            )
        else:
            valid_trajectory_data.append(
                {
                    "positions": np.array(positions),
                    "velocities": np.array(velocities),
                    "rotations": np.array(rotations),
                    "times": np.array(times),
                    "Mext": ex_mat[np.newaxis, :, :],
                    "Mint": in_mat[:3, :3][np.newaxis, :, :],
                    "bounces": np.array(hits_opponent),
                    "seed": seed,
                }
            )
    return valid_trajectory_data, out_trajectory_data


def get_valid_trajectories(
    num_trajectories, num_processes, mode, direction, output_path
):
    valid_trajectories = []
    out_trajectories = []
    total_saved_valid = 0
    total_saved_out = 0
    if mode == "toss":
        max_out_trajectories = 0
    else:
        max_out_trajectories = (
            num_trajectories // 5
        )  # save 20% of "out trajectories" per mode and direction
    current_seed, batch_size = 0, min(1024, num_trajectories)

    with tqdm.tqdm(total=num_trajectories, unit="trajectory", disable=False) as pbar:
        while total_saved_valid < num_trajectories:
            seeds = [
                list(range(current_seed + j, current_seed + batch_size, num_processes))
                for j in range(num_processes)
            ]
            # Prepare containers for worker results (lists of lists)
            results_from_pool = []
            out_trajectory_data = []

            if num_processes == 1:
                # Run workers sequentially in the main process and collect their two-part return values
                for s in seeds[0]:
                    valid_list, out_list = find_valid_trajectories_worker(
                        ([s], mode, direction)
                    )
                    results_from_pool.append(valid_list)
                    out_trajectory_data.append(out_list)
            else:
                with Pool(num_processes) as p:
                    worker_results = p.map(
                        find_valid_trajectories_worker,
                        [(s, mode, direction) for s in seeds if s],
                    )
                    results_from_pool = [valid for valid, _ in worker_results]
                    out_trajectory_data = [out for _, out in worker_results]

            for trajectory_list in results_from_pool:
                valid_trajectories.extend(trajectory_list)
            remaining_out_quota = max_out_trajectories - total_saved_out
            if remaining_out_quota > 0:
                for trajectory_list in out_trajectory_data:
                    if remaining_out_quota <= 0:
                        break
                    if len(trajectory_list) <= remaining_out_quota:
                        out_trajectories.extend(trajectory_list)
                        remaining_out_quota -= len(trajectory_list)
                    else:
                        out_trajectories.extend(trajectory_list[:remaining_out_quota])
                        remaining_out_quota = 0

            current_seed += batch_size

            # Save valid trajectories in chunks
            if len(valid_trajectories) >= CHUNK_SIZE:
                chunk_to_save = valid_trajectories[:CHUNK_SIZE]
                save_dataset(
                    output_path, chunk_to_save, [], total_saved_valid, total_saved_out
                )
                total_saved_valid += len(chunk_to_save)
                valid_trajectories = valid_trajectories[CHUNK_SIZE:]

            # Save out trajectories in chunks
            if (
                len(out_trajectories) >= CHUNK_SIZE
                and total_saved_out < max_out_trajectories
            ):
                remaining_out_quota = max_out_trajectories - total_saved_out
                chunk_size = min(CHUNK_SIZE, remaining_out_quota)
                chunk_to_save_out = out_trajectories[:chunk_size]
                save_dataset(
                    output_path,
                    [],
                    chunk_to_save_out,
                    total_saved_valid,
                    total_saved_out,
                )
                total_saved_out += len(chunk_to_save_out)
                out_trajectories = out_trajectories[chunk_size:]

            pbar.n = total_saved_valid + len(valid_trajectories)
            pbar.set_postfix_str(f"{total_saved_valid + len(valid_trajectories)} found")
            pbar.refresh()

    # Save remaining valid trajectories (trim to exact count needed)
    remaining_needed = num_trajectories - total_saved_valid
    if remaining_needed > 0 and len(valid_trajectories) > 0:
        final_chunk = valid_trajectories[:remaining_needed]
        save_dataset(output_path, final_chunk, [], total_saved_valid, total_saved_out)
        total_saved_valid += len(final_chunk)

    # Save any remaining out trajectories
    if len(out_trajectories) > 0 and total_saved_out < max_out_trajectories:
        remaining_out_quota = max_out_trajectories - total_saved_out
        final_out_chunk = out_trajectories[:remaining_out_quota]
        if len(final_out_chunk) > 0:
            save_dataset(
                output_path,
                [],
                final_out_chunk,
                total_saved_valid,
                total_saved_out,
            )
            total_saved_out += len(final_out_chunk)

    print(
        f"Saved {total_saved_valid} valid trajectories and {total_saved_out} out trajectories."
    )
    return total_saved_valid, total_saved_out


def save_dataset(
    path,
    trajectories_data,
    out_trajectories_data,
    start_index_valid=0,
    start_index_out=0,
):
    """Save trajectories with continuous numbering across chunks.

    Args:
        path: Base path for saving
        trajectories_data: List of valid trajectory data to save
        out_trajectories_data: List of out trajectory data to save
        start_index_valid: Starting index for valid trajectory numbering
        start_index_out: Starting index for out trajectory numbering
    """
    os.makedirs(path, exist_ok=True)

    if len(trajectories_data) > 0:
        print(
            f"Saving {len(trajectories_data)} valid trajectories starting at index {start_index_valid}..."
        )
        for i, traj_data in tqdm.tqdm(
            enumerate(trajectories_data),
            total=len(trajectories_data),
            desc="Saving valid",
        ):
            save_path = os.path.join(path, f"in/trajectory_{start_index_valid + i:04}")
            os.makedirs(save_path, exist_ok=True)
            for key, value in traj_data.items():
                if key != "seed":
                    np.save(os.path.join(save_path, f"{key}.npy"), value)

    if len(out_trajectories_data) > 0:
        print(
            f"Saving {len(out_trajectories_data)} out trajectories starting at index {start_index_out}..."
        )
        for i, traj_data in tqdm.tqdm(
            enumerate(out_trajectories_data),
            total=len(out_trajectories_data),
            desc="Saving out",
        ):
            save_path = os.path.join(path, f"out/trajectory_{start_index_out + i:04}")
            os.makedirs(save_path, exist_ok=True)
            for key, value in traj_data.items():
                if key != "seed":
                    np.save(os.path.join(save_path, f"{key}.npy"), value)


### for testing purposes (from XML mujoco def)
def test_COR(drop_height=2.54, duration=3.0, timestep=TIMESTEP):
    """Run a deterministic vertical drop from `drop_height` onto the ground plane and report rebound height and estimated COR.

    Parameters:
    - drop_height: initial drop height in meters
    - duration: maximum simulation time in seconds
    - timestep: simulation timestep (seconds)

    """
    print(
        f"Running drop test: drop_height={drop_height} m, duration={duration}s, timestep={timestep}s"
    )
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Place ball directly at x,y=3 so it falls onto the plane geom at z=0
    data.qpos[0:3] = np.array([3.0, 3.0, drop_height])  # x,y=3 to avoid net
    data.qvel[0:3] = np.zeros(3)
    data.qvel[3:6] = np.zeros(3)

    positions = []
    velocities = []
    contacts = []
    max_steps = int(duration / timestep)
    for _ in range(max_steps):
        mujoco.mj_step(model, data)
        positions.append(data.qpos[0:3].copy())
        velocities.append(data.qvel[0:3].copy())
        contacts.append(data.ncon > 0)

    zs = np.array(positions)[:, 2]
    if zs.size == 0:
        print("No data recorded in drop test.")
        return

    # times = np.array(times)
    z_vel = np.array(velocities)[:, 2]

    plot_over_idx(zs)
    plot_over_idx(z_vel)

    # Find the frame just before contact and just after contact
    contact_start_idx = contacts.index(True)
    # Find the first index that is False after contact starts
    contact_end_idx = contacts.index(False, contact_start_idx)
    print(f"Contact detected from index {contact_start_idx} to {contact_end_idx}")

    # Find the frame where maximum height after the contact (rebound height)
    rebound_max_idx = int(np.argmax(zs[contact_end_idx:])) + contact_end_idx

    rebound_height = float(zs[rebound_max_idx])
    rebound_x_offset = float(positions[rebound_max_idx][0]) - 3.0
    rebound_y_offset = float(positions[rebound_max_idx][1]) - 3.0

    combined_offset = math.sqrt(rebound_x_offset**2 + rebound_y_offset**2)
    deviation_angle_in_degrees = (
        math.atan2(combined_offset, rebound_height) * 180.0 / math.pi
    )

    if deviation_angle_in_degrees > 5.0:
        print(
            f"Warning: Significant horizontal deviation detected during drop test: {deviation_angle_in_degrees:.2f} degrees"
        )
        return

    # Estimate COR: -v_after / v_before
    # The lowest velocity just before the minimum height index is the impact velocity.
    # The highest velocity just after is the rebound velocity.
    v_before = z_vel[contact_start_idx - 1]
    v_after = z_vel[contact_end_idx + 1]

    # COR is the ratio of relative speeds, hence the negative sign.
    cor_est = -v_after / v_before if v_before != 0 else 0

    print(
        f"Drop test results:\n  min_z (impact) start index={contact_start_idx}, z={zs[contact_start_idx]:.4f} with deviation in degrees = {deviation_angle_in_degrees:.2f}\n  max_z (rebound) index={rebound_max_idx}, rebound_height={rebound_height:.4f} m\n  estimated COR based on velocities {cor_est:.4f}"
    )
    print(
        f"velocity before bounce = {v_before:.4f} m/s, velocity after bounce = {v_after:.4f} m/s"
    )
    return


def test_COF(
    initial_speed=30.0,
    initial_height=0.5,
    angle_of_incidence_deg=16.0,
    duration=0.5,
    initial_angular_velocity=0,
    timestep=TIMESTEP,
):
    """
    Runs a bounce simulation to estimate the Coefficient of Friction (COF).
    The ball is incident on the court surface at a specified angle.

    Parameters:
    - initial_speed (float): Initial speed of the ball in m/s. (30m/s is proposed by ITF)
    - initial_height (float): Initial height of the ball in meters.
    - angle_of_incidence_deg (float): Angle of incidence to the surface in degrees. (16° is proposed by ITF)
    - duration (float): Maximum simulation time in seconds.
    - initial_angular_velocity (float): Initial angular velocity around the y-axis in rad/s.
    - timestep (float): Simulation timestep in seconds.
    """
    print(
        f"Running COF test: initial_speed={initial_speed} m/s, initial_height={initial_height} m, angle={angle_of_incidence_deg}°, initial_angular_velocity={initial_angular_velocity} rad/s, duration={duration}s"
    )
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Set initial position and velocity for the bounce
    angle_rad = np.deg2rad(angle_of_incidence_deg)
    vx1_initial = initial_speed * np.cos(angle_rad)
    vz1_initial = -initial_speed * np.sin(angle_rad)

    data.qpos[0:3] = np.array(
        [-7.0, 0.0, initial_height]
    )  # Start away from origin to avoid net
    data.qvel[0:3] = np.array([vx1_initial, 0.0, vz1_initial])
    data.qvel[3:6] = np.array([0.0, -initial_angular_velocity, 0.0])  # np.zeros(3)

    positions = []
    velocities = []
    angular_velocities = []
    contacts = []

    max_steps = int(duration / timestep)
    for step in range(max_steps):
        mujoco.mj_step(model, data)
        positions.append(data.qpos[0:3].copy())
        velocities.append(data.qvel[0:3].copy())
        angular_velocities.append(data.qvel[3:6].copy())

        if data.ncon == 0:
            contacts.append(False)
        elif data.ncon > 0:
            contacts.append(True)

    if not velocities:
        print("No data recorded in COF test.")
        return
    zs = np.array(positions)[:, 2]
    ys = np.array(positions)[:, 1]
    xs = np.array(positions)[:, 0]
    z_vel = np.array(velocities)[:, 2]
    x_vel = np.array(velocities)[:, 0]
    xz_vel = np.sqrt(x_vel**2 + z_vel**2)
    omegaY = np.array(angular_velocities)[:, 1]  # y-component of angular velocity

    # -----------------------------------
    # Plots for visualization
    # ----------------------------------
    print("\n x over index")
    plot_over_idx(xs)
    print("y over index")
    plot_over_idx(ys)
    print("z over index")
    plot_over_idx(zs)
    print("z_vel over index")
    plot_over_idx(z_vel)
    print("x_vel over index")
    plot_over_idx(x_vel)
    print("xz_vel over index")
    plot_over_idx(xz_vel)
    print("omegaY over index")
    plot_over_idx(omegaY)
    print("trajecotry of ball")
    show_trajectory(xs, ys, zs)

    try:
        # Find the frame just before contact and just after contact
        contact_start_idx = contacts.index(True)
        # Find where contact ends
        contact_end_idx = len(contacts) - 1 - contacts[::-1].index(True)
        show_trajectory(
            xs[contact_start_idx - 3 : contact_end_idx + 3],
            ys[contact_start_idx - 3 : contact_end_idx + 3],
            zs[contact_start_idx - 3 : contact_end_idx + 3],
        )
        v_before = velocities[contact_start_idx - 1]
        v_after = velocities[contact_end_idx + 1]
        omega1 = angular_velocities[contact_start_idx - 1]
        omega2 = angular_velocities[contact_end_idx + 1]

        vx1, vz1 = v_before[0], v_before[2]
        vx2, vz2 = v_after[0], v_after[2]

        # verify sliding criteria
        sliding_criteria = False
        if vx2 > BALL_RADIUS * abs(omega2[1]):
            sliding_criteria = True

        # Calculate COF using µ = (vx1 − vx2) / (vz2 − vz1)
        # pdb.set_trace()
        cof_numerator = vx1 - vx2
        cof_denominator = vz2 - vz1
        cof = cof_numerator / cof_denominator if cof_denominator != 0 else float("inf")
        cor = -vz2 / vz1 if vz1 != 0 else 0
        cor_t = cor + 0.003 * 8
        cpr = 100 * (1 - cof) + 150 * (0.81 - cor_t)

        print(f"COF test results:")
        print(f"  Bounce detected in from {contact_start_idx} to {contact_end_idx}")
        print(f"  Velocities before bounce (vx1, vz1): ({vx1:.4f}, {vz1:.4f}) m/s")
        print(f"  Velocities after bounce (vx2, vz2): ({vx2:.4f}, {vz2:.4f}) m/s")
        print(f"  Angular velocity before bounce (omega_y): {omega1[1]:.4f} rad/s")
        print(f"  Angular velocityafter bounce (omega_y): {omega2[1]:.4f} rad/s")
        print(f"  Sliding criteria: {sliding_criteria}")
        print(f"  Estimated COF (µ): {cof:.4f}")
        print(f"  estimted COR (e)): {cor:.4f}")
        print(f"  estimated COR_T (e_T): {cor_t:.4f}")
        print(f"  estimated CPR: {cpr:.4f}")

        return

    except (ValueError, IndexError):
        print("Could not determine bounce event from simulation data.")


def plot_over_idx(data_array):
    """
    Plots the values of a 1D array over their indices.

    This is a utility function for visualizing data like velocities or positions
    against their array index, which often corresponds to timesteps.

    Parameters:
    - data_array (array-like): A 1D list or numpy array of numerical data.
    """
    input_data = np.array(data_array)
    if input_data.ndim != 1:
        print(
            f"Error: plot_over_idx expects a 1D array, but got an array with shape {input_data.shape}"
        )
        return

    if len(input_data) == 0:
        print("Warning: plot_over_idx received an empty array. Nothing to plot.")
        return

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(input_data, color="#F87825")
    # ax.set_title("Drop Test from 254 cm at rigid smooth surface")
    ax.set_xlabel("Timesteps in ms", fontsize=16)
    ax.set_ylabel("Height in m", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    # ax.vlines(x=88, ymin=np.min(input_data), ymax=np.max(input_data), colors="r")
    # ax.vlines(x=106, ymin=np.min(input_data), ymax=np.max(input_data), colors="r")

    # Set axis limits as requested
    ax.set_xlim(0, 1500)  # ax.set_xlim(0, len(input_data) + 5)
    ax.set_ylim(np.min(input_data), np.max(input_data) + 0.3)
    ax.grid(True, which="both")
    plt.show()


def show_trajectory(xs, ys, zs):
    # Create a new figure and a 3D subplot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the trajectory line
    ax.plot(xs, ys, zs, label="Ball Trajectory")

    # Set labels for the axes
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position (Height)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-8, 8)
    ax.set_zlim(0.0, 15.0)

    for connection in court_connections:
        ax.plot(
            court_points[connection, 0],
            court_points[connection, 1],
            court_points[connection, 2],
            "k",
        )

    # Set a title for the plot
    ax.set_title("Tennis Ball Trajectory")

    # Add a legend
    ax.legend()
    plt.show()
    fig.savefig("trajectory.png")


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Generate inference dataset")
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=500,
        help="Number of trajectories to generate. If size is < 400 consider reducing the CHUNK_SIZE",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="tmp",
        help="Folder to save the dataset",
    )
    parser.add_argument(
        "--num_processes", type=int, default=1, help="Number of cpu processes to use"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="groundstroke",
        help="Mode of the syntheticdataset, e.g. groundstroke, serve, volley, smash, lob, short, toss",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="close_to_far",
        help="Direction of the syntheticdataset, e.g. far_to_close, close_to_far",
    )
    parser.add_argument(
        "--test_COR",
        action="store_true",
        help="Run the COR test instead of generating the dataset",
    )
    parser.add_argument(
        "--test_COF",
        action="store_true",
        help="Run the COF test instead of generating the dataset",
    )
    args = parser.parse_args()

    if args.test_COR and args.test_COF:
        test_COR()
        test_COF()
        exit(0)

    if args.test_COR:
        test_COR()
        exit(0)

    if args.test_COF:
        test_COF()
        exit(0)

    mode = args.mode
    assert (
        mode in INIT_POS_RANGES
    ), f"Mode {mode} not supported. Choose from {list(INIT_POS_RANGES.keys())}"
    direction = args.direction
    assert direction in [
        "far_to_close",
        "close_to_far",
    ], f"Direction {direction} not supported. Choose from ['far_to_close', 'close_to_far']"

    from paths import data_path

    # Build output path
    output_path = os.path.join(data_path, args.folder, mode, direction)

    # --- Logic from mujocosimulation_modular.py ---
    print(f"Searching for {args.num_trajectories} valid trajectories...")
    total_valid, total_out = get_valid_trajectories(
        num_trajectories=args.num_trajectories,
        num_processes=args.num_processes,
        mode=mode,
        direction=direction,
        output_path=output_path,
    )

    print(
        f"Dataset generation complete. Total: {total_valid} valid, {total_out} out trajectories."
    )
