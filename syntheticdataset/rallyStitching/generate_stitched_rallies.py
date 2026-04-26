"""
Generates a synthetic, branched dataset of stitched table tennis rallies.

This script uses a MuJoCo physics simulation to create realistic ball
trajectories. It relies on a pre-computed classification of serves and returns
from `classify_rallies.py` AND new synthetic data pools for tosses, serves,
and returns.

The logic is as follows:
1. Load all data pools (synthetic tosses, synthetic/DeepMind serves,
   synthetic/DeepMind returns).
2. Create sets of all available IDs for tracking.
3. Start each rally by simulating a 'toss' segment.
4. Stitch a 'serve hit' segment to the end of the toss.
5. Recursively stitch 'intermediate' segments to build a rally tree.
6. All segments are validated against a "ghost net".
7. After generation, report which data pool states were unused.
"""

import os
import random
import shutil
import dataclasses as dc
import multiprocessing

import mujoco
import numpy as np
import pandas as pd
from tqdm import tqdm

from paths import data_path
from syntheticdataset.helper import (
    XML,
    MAX_SIMULATION_TIME,
    BASELINE_X_CLOSE,
    SINGLE_SIDELIINE_Y_RIGHT,
    NET_TENNIS_HEIGHT,
    NET_TENNIS_WIDTH,
    BALL_RADIUS,
)

DATASET_FOLDER = "syntheticdata"

# ==============================================================================
# 1. CONFIGURATION AT-A-GLANCE
# ==============================================================================

# --- DEBUGGING & SCALING ---
NUM_WORKERS = 64  # <-- SET THE NUMBER OF WORKER PROCESSES HERE

# --- RALLY STRUCTURE ---
RALLY_LENGTH = 6  # 70% of tennis rallies end in 0-4 shots
BRANCHING_FACTOR = 4
MAX_TRAJECTORIES_PER_TOSS = 160  # Maximum number of saved rallies per toss

# --- SIMULATION PARAMETERS (SHARED) ---
FPS = 500
TIMESTEP = 0.001
MAX_TIME_PER_ROLLOUT = MAX_SIMULATION_TIME

# --- PHYSICS CONSTANTS (SHARED) ---
NET_CLEARANCE_X_MARGIN = 0.04

# --- INPUT/OUTPUT ---
RALLIES_PATH = "syntheticdataset/rallyStitching/jsons/rallies.json"
SERVES_PATH = "syntheticdataset/rallyStitching/jsons/serves.json"
TOSSES_PATH = "syntheticdataset/rallyStitching/jsons/tosses.json"

OUTDIR = os.path.join(data_path, "stitched_rallies")

MAX_TRIES_PER_SEGMENT = 10
NEIGHBOR_EPSILON = 0.1  # Max distance for 'sample_closest'
HIT_WINDOW_Z_RANGE = (3.0, 3.8)  # Valid Z-heights to "hit" the ball after a toss

# ==============================================================================
# 2. DATA STRUCTURES and HELPERS
# ==============================================================================


@dc.dataclass(frozen=True, slots=True)
class BallState:
    id: str
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    w_vel_x: float
    w_vel_y: float
    w_vel_z: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.pos_x, self.pos_y, self.pos_z])

    @property
    def linear_velocity(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y, self.vel_z])

    @property
    def angular_velocity(self) -> np.ndarray:
        return np.array([self.w_vel_x, self.w_vel_y, self.w_vel_z])


def _load_and_augment(
    path: str, id_prefix: str, apply_transform: bool
) -> tuple[list[BallState], dict]:
    """Helper to load one JSON, apply transform if needed, and augment."""
    rallies_pd = pd.read_json(path)

    def row_to_ballstate(r):
        return BallState(
            id=f"{id_prefix}{r.id}",
            pos_x=r.pos_x,
            pos_y=r.pos_y,
            pos_z=r.pos_z,
            vel_x=r.vel_x,
            vel_y=r.vel_y,
            vel_z=r.vel_z,
            w_vel_x=r.w_vel_x,
            w_vel_y=r.w_vel_y,
            w_vel_z=r.w_vel_z,
        )

    rallies = rallies_pd.apply(row_to_ballstate, axis=1).tolist()

    rallies_map = {bs.id: bs for bs in rallies}
    return rallies, rallies_map


def load_all_data_pools() -> dict:
    """
    Loads all synhic data sources
    """

    # --- Load synthetic data pools ---
    print("Loading synthetic data pools...")
    try:
        toss_states, _ = _load_and_augment(TOSSES_PATH, "t_", False)
        print(f"Added {len(toss_states)} tosses.")

        serve_states, _ = _load_and_augment(SERVES_PATH, "s_", False)
        print(f"  + {len(serve_states)} serves added to pool.")

        return_states, _ = _load_and_augment(RALLIES_PATH, "r_", False)
        print(f"  + {len(return_states)} returns added to pool.")

    except FileNotFoundError as e:
        print(f"Error: Could not load required file: {e.filename}")
        print("Please run `generate_synthetic_states.py --mode toss` first.")
        raise

    # --- Create final data structures for sampling ---
    data_pools = {
        "toss_states": toss_states,
        "serve_states": serve_states,
        "return_states": return_states,
        "serve_pos": np.asarray([bs.position for bs in serve_states]),
        "return_pos": np.asarray([bs.position for bs in return_states]),
        # Sets for tracking usage
        "all_toss_ids": {bs.id for bs in toss_states},
        "all_serve_ids": {bs.id for bs in serve_states},
        "all_return_ids": {bs.id for bs in return_states},
    }

    return data_pools


def _get_window_start_end(positions: np.ndarray, player: str):
    mask = positions[:, 2] < 0.1

    all_bounce_indices = np.where(mask)[0]
    if len(all_bounce_indices) == 0:
        return None, None

    run_starts = [all_bounce_indices[0]]
    run_ends = []
    for i in range(1, len(all_bounce_indices)):
        if all_bounce_indices[i] > all_bounce_indices[i - 1] + 1:
            run_ends.append(all_bounce_indices[i - 1])
            run_starts.append(all_bounce_indices[i])
    run_ends.append(all_bounce_indices[-1])

    window_start = run_ends[0]
    window_end = run_starts[1] if len(run_starts) > 1 else len(positions) - 1
    return window_start, window_end


def _get_trajectory_from_id(id) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper to load the full trajectory (pos, vel, rot) for a given state ID.
    The ID format encodes mode, direction, and trajectory ID.
    """
    cut_id = id.split("_", 5)
    mode = cut_id[1]
    direction = cut_id[2] + "_" + cut_id[3] + "_" + cut_id[4]
    traj_id = "trajectory_" + cut_id[5]
    folder = os.path.join(data_path, DATASET_FOLDER, f"{mode}/{direction}/in/{traj_id}")

    pos_file = os.path.join(folder, "positions.npy")
    vel_file = os.path.join(folder, "velocities.npy")
    rot_file = os.path.join(folder, "rotations.npy")

    if not (
        os.path.isfile(pos_file)
        and os.path.isfile(vel_file)
        and os.path.isfile(rot_file)
    ):
        raise FileNotFoundError(
            f"Missing trajectory files for ID {id} in folder {folder}"
        )

    positions = np.load(pos_file)
    velocities = np.load(vel_file)
    rotations = np.load(rot_file)

    return positions, velocities, rotations


def sample_closest(
    q: np.ndarray, states: list[BallState], positions: np.ndarray
) -> BallState:
    dists = np.linalg.norm(positions - q[None, :], axis=1)
    idx = np.where(dists <= NEIGHBOR_EPSILON)[0]
    j = int(np.random.choice(idx)) if idx.size > 0 else int(np.argmin(dists))
    return states[j]


def compute_rollout(model, data, timestep) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos, vel, rot = [], [], []
    next_save = 0.0
    while next_save <= MAX_TIME_PER_ROLLOUT:
        steps = max(0, round((next_save - data.time) / timestep))
        for _ in range(steps):
            mujoco.mj_step(model, data)
        if abs(data.qpos[0]) > 15 or abs(data.qpos[1]) > 8 or data.qpos[2] < -0.1:
            break
        pos.append(data.qpos[0:3].copy())
        vel.append(data.qvel[0:3].copy())
        rot.append(data.qvel[3:6].copy())
        next_save += 1.0 / FPS
    return np.array(pos), np.array(vel), np.array(rot)


# ==============================================================================
# 3. CORE LOGIC
# ==============================================================================


def _validate_net_clearance(positions: np.ndarray) -> bool:
    """Checks if the trajectory flew through the 'ghost net'."""
    close_to_net_mask = np.abs(positions[:, 0]) < NET_CLEARANCE_X_MARGIN
    if np.any(close_to_net_mask):
        heights_close_to_net = positions[close_to_net_mask, 2]
        widths_close_to_net = positions[close_to_net_mask, 1]
        if (
            np.max(heights_close_to_net) < (NET_TENNIS_HEIGHT + BALL_RADIUS + 0.02)
        ) and (
            np.min(np.abs(widths_close_to_net))
            < (NET_TENNIS_WIDTH / 2 + BALL_RADIUS + 0.02)
        ):
            return False
    return True


def is_bounce_valid(positions: np.ndarray, player: str) -> bool:
    """
    Validates if a trajectory has the correct number of bounces on the
    opponent's side.
    """
    mask_z = positions[:, 2] < 0.1
    mask_y = np.abs(positions[:, 1]) < SINGLE_SIDELIINE_Y_RIGHT

    if player == "close":
        mask_x_opp = (positions[:, 0] < -0.01) & (positions[:, 0] > -BASELINE_X_CLOSE)
    else:
        mask_x_opp = (positions[:, 0] > 0.01) & (positions[:, 0] < BASELINE_X_CLOSE)

    bounce_mask = mask_z & mask_y & mask_x_opp
    valid_trajectory = len(bounce_mask) > 0
    return valid_trajectory


def _validate_serve_bounces(positions: np.ndarray, is_from_close_side: bool) -> bool:
    """Helper to check for opponent side' bounce in service box."""
    mask_z = positions[:, 2] < 0.1
    if positions[0, 1] < 0:
        mask_y = (positions[:, 1] >= 0) & (positions[:, 1] <= SINGLE_SIDELIINE_Y_RIGHT)
    else:
        mask_y = (positions[:, 1] >= -SINGLE_SIDELIINE_Y_RIGHT) & (positions[:, 1] <= 0)
    if is_from_close_side:
        mask_x_opp = (positions[:, 0] < -0.01) & (positions[:, 0] > -BASELINE_X_CLOSE)
    else:
        mask_x_opp = (positions[:, 0] > 0.01) & (positions[:, 0] < BASELINE_X_CLOSE)

    bounce_mask = mask_z & mask_y & mask_x_opp
    valid_trajectory = len(bounce_mask) > 0
    return valid_trajectory


def create_toss_segment(
    toss_states: list[BallState],
) -> tuple[dict | None, dict | None, str | None]:
    """
    Creates the first segment of a rally: the ball toss.
    Simulates the toss, cleans any bounces, and finds a random hit frame.
    """
    for _ in range(MAX_TRIES_PER_SEGMENT):
        initial_state = random.choice(toss_states)
        is_from_close_side = initial_state.position[0] > 0
        clean_positions, clean_velocities, clean_rotations = _get_trajectory_from_id(
            initial_state.id
        )

        # Find apex and all valid hit frames
        try:
            apex_index = np.argmax(clean_positions[:, 2])
        except ValueError:
            continue  # Empty positions list

        valid_hit_indices = []
        for i in range(apex_index, len(clean_positions)):
            if HIT_WINDOW_Z_RANGE[0] <= clean_positions[i, 2] <= HIT_WINDOW_Z_RANGE[1]:
                valid_hit_indices.append(i)

        if not valid_hit_indices:
            continue  # Toss was bad (e.g., too low, never entered hit window)

        # 5. Clip and Return
        hit_frame_index = random.choice(valid_hit_indices)

        clipped_pos = clean_positions[: hit_frame_index + 1]

        next_player = "close" if is_from_close_side else "far"
        next_state = {"initial_position": clipped_pos[-1], "player": next_player}
        segment_data = {
            "positions": clipped_pos,
            "velocities": clean_velocities[: hit_frame_index + 1],
            "rotations": clean_rotations[: hit_frame_index + 1],
        }
        return segment_data, next_state, initial_state.id

    return None, None, None


def create_stitched_serve(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    current_state: dict,
    serve_states: list[BallState],
    serve_positions: np.ndarray,
) -> tuple[dict | None, dict | None, str | None]:
    """
    Stitches a 'serve hit' state onto the end of a 'toss' segment.
    Validates that the resulting trajectory is a 2-bounce serve.
    """
    player = current_state["player"]
    target_position = current_state["initial_position"]
    is_from_close_side = player == "close"

    for _ in range(MAX_TRIES_PER_SEGMENT):
        initial_state = sample_closest(target_position, serve_states, serve_positions)

        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = target_position
        data.qvel[0:3] = initial_state.linear_velocity
        data.qvel[3:6] = initial_state.angular_velocity

        mujoco.mj_forward(model, data)
        positions, velocities, rotations = compute_rollout(
            model, data, model.opt.timestep
        )

        if len(positions) < 10:
            continue

        if not _validate_net_clearance(positions):
            continue

        if not _validate_serve_bounces(positions, is_from_close_side):
            continue

        window_start, window_end = _get_window_start_end(positions, player)
        if window_start is None or window_end is None:
            continue
        if window_start >= window_end:
            continue

        end_idx = random.randint(window_start + 2, window_end - 2)
        clipped_pos = positions[: end_idx + 1]

        next_player = "far" if is_from_close_side else "close"
        next_state = {"initial_position": clipped_pos[-1], "player": next_player}
        segment_data = {
            "positions": clipped_pos,
            "velocities": velocities[: end_idx + 1],
            "rotations": rotations[: end_idx + 1],
        }
        return segment_data, next_state, initial_state.id

    return None, None, None


def create_stitched_return(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    current_state: dict,
    return_states: list[BallState],
    return_positions: np.ndarray,
) -> tuple[dict | None, dict | None, str | None]:
    """
    Creates a stitched return segment.
    Returns the segment data, next state, and the ID of the state used.
    """
    player = current_state["player"]
    target_position = current_state["initial_position"]

    for _ in range(MAX_TRIES_PER_SEGMENT):
        initial_state = sample_closest(target_position, return_states, return_positions)

        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = target_position
        data.qvel[0:3] = initial_state.linear_velocity
        data.qvel[3:6] = initial_state.angular_velocity

        mujoco.mj_forward(model, data)
        positions, velocities, rotations = compute_rollout(
            model, data, model.opt.timestep
        )

        if len(positions) < 10:
            continue

        if not _validate_net_clearance(positions):
            continue

        if not is_bounce_valid(positions, player):
            continue

        window_start, window_end = _get_window_start_end(positions, player)
        if window_start is None or window_end is None:
            continue

        if window_start >= window_end:
            continue

        end_idx = random.randint(window_start, window_end)
        clipped_pos = positions[: end_idx + 1]

        next_player = "far" if player == "close" else "close"
        next_state = {"initial_position": clipped_pos[-1], "player": next_player}
        segment_data = {
            "positions": clipped_pos,
            "velocities": velocities[: end_idx + 1],
            "rotations": rotations[: end_idx + 1],
        }
        return segment_data, next_state, initial_state.id

    return None, None, None


def _save_rally(rally_so_far: list[dict], filename: str, outdir: str):
    path = os.path.join(outdir, filename)
    # save data as float16 to save disk space. Float16 is sufficient for metric coordinates.
    np.savez_compressed(
        path,
        positions=np.array(
            [s["positions"].astype(np.float16) for s in rally_so_far], dtype=object
        ),
        velocities=np.array(
            [s["velocities"].astype(np.float16) for s in rally_so_far], dtype=object
        ),
        rotations=np.array(
            [s["rotations"].astype(np.float16) for s in rally_so_far], dtype=object
        ),
    )


def build_rally_branches(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rally_so_far: list[dict],
    base_filename: str,
    data_pools: dict,
    counter: int,
    tracking_sets: dict,
) -> tuple[int, dict]:
    """
    Recursively builds and saves rally branches, tracking used state IDs.
    """

    # --- Stop condition: Max trajectories per toss reached ---
    if counter >= MAX_TRAJECTORIES_PER_TOSS:
        return counter, tracking_sets

    # --- Stop condition: Rally is at max length ---
    if len(rally_so_far) >= RALLY_LENGTH:
        _save_rally(rally_so_far, f"{base_filename}_branch_{counter}.npz", OUTDIR)
        counter += 1
        return counter, tracking_sets

    parent_segment = rally_so_far[-1]
    next_player = "far" if parent_segment["player"] == "close" else "close"
    next_state_template = {
        "initial_position": parent_segment["positions"][-1],
        "player": next_player,
    }

    # --- NORMAL RECURSION LOGIC ---
    child_segments = []
    attempts = 0
    while (
        len(child_segments) < BRANCHING_FACTOR and attempts < MAX_TRIES_PER_SEGMENT * 2
    ):
        seg_data, _, used_return_id = create_stitched_return(
            model,
            data,
            next_state_template,
            data_pools["return_states"],
            data_pools["return_pos"],
        )
        if seg_data:
            seg_data["player"] = next_player
            child_segments.append(seg_data)
            if used_return_id:  # Track used ID
                tracking_sets["returns"].add(used_return_id)
        attempts += 1

    if not child_segments:
        # Save the rally here if it's a dead end but still valid
        _save_rally(
            rally_so_far, f"{base_filename}_branch_{counter}_deadend.npz", OUTDIR
        )
        counter += 1
        return counter, tracking_sets  # Dead end

    for child in child_segments:
        # Pass tracking sets down the recursive call
        counter, tracking_sets = build_rally_branches(
            model,
            data,
            rally_so_far + [child],
            base_filename,
            data_pools,
            counter,
            tracking_sets,
        )
    return counter, tracking_sets


# ==============================================================================
# 4. WORKER FUNCTION FOR MULTIPROCESSING
# ==============================================================================
# Global variable to hold the data pools for each worker process
worker_data_pools = None


def init_worker(shared_pools: dict):
    """
    Initializer function for the multiprocessing pool.

    This runs exactly once per worker process when it starts up.
    It saves the massive data dictionary into the worker's local global memory,
    preventing the main process from having to serialize and pipe gigabytes of
    data for every single toss.
    """
    global worker_data_pools
    worker_data_pools = shared_pools


def generate_rally_tree_worker(toss_index: int) -> tuple[int, dict]:
    """
    Worker function to generate a full rally tree, starting from a ball toss.
    Reads the necessary state data from the globally initialized `worker_data_pools`.

    Args:
        toss_index: The integer index of the toss state to use as the root.

    Returns:
        A tuple containing:
        - num_files_generated (int): The number of valid .npz files saved.
        - tracking_sets (dict): Sets of IDs of the tosses, serves, and returns used.
    """
    global worker_data_pools

    # Each worker needs its own independent model and data instance
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Initialize tracking sets for this worker
    tracking_sets = {"tosses": set(), "serves": set(), "returns": set()}

    # 1. Get a *single* toss state for this worker from the global pools
    toss_state = worker_data_pools["toss_states"][toss_index]

    # 2. Create the toss segment
    toss_segment, next_state, toss_id = create_toss_segment([toss_state])
    if toss_segment is None:
        return 0, tracking_sets  # Failed to create a valid toss

    if toss_id:
        tracking_sets["tosses"].add(toss_id)
    player = "close" if toss_state.position[0] > 0 else "far"
    toss_segment["player"] = player

    # 3. Create the stitched serve segment
    serve_segment, next_state, serve_id = create_stitched_serve(
        model,
        data,
        next_state,
        worker_data_pools["serve_states"],
        worker_data_pools["serve_pos"],
    )
    if serve_segment is None:
        return 0, tracking_sets  # Failed to stitch a valid serve

    if serve_id:
        tracking_sets["serves"].add(serve_id)
    serve_segment["player"] = player  # Serve is by the same player

    # 4. We have a valid root, start branching for returns
    rally_so_far = [toss_segment, serve_segment]
    base_filename = f"toss_{toss_index}"

    files_generated, tracking_sets = build_rally_branches(
        model=model,
        data=data,
        rally_so_far=rally_so_far,
        base_filename=base_filename,
        data_pools=worker_data_pools,  # Pass the global pool down into recursion
        counter=0,
        tracking_sets=tracking_sets,
    )
    return files_generated, tracking_sets


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    print("--- Starting Branched Rally Generation ---")
    if os.path.exists(OUTDIR):
        print(f"Clearing existing output directory: {OUTDIR}")
        shutil.rmtree(OUTDIR)
    os.makedirs(OUTDIR)

    # Load all data pools once in the main process
    try:
        data_pools = load_all_data_pools()
    except FileNotFoundError:
        return  # Error message already printed in loader

    toss_states = data_pools["toss_states"]
    if not toss_states:
        print("Error: No valid tosses found. Cannot generate rallies.")
        return
    if not data_pools["serve_states"]:
        print("Error: No valid serves found. Cannot generate rallies.")
        return
    if not data_pools["return_states"]:
        print("Error: No valid returns found. Cannot generate rallies.")
        return

    # Prepare a list of tasks for the worker pool
    # The tasks list is now just a lightweight list of integers (indices)!
    tasks = list(range(len(toss_states)))

    print(
        f"\nStarting generation... {len(tasks)} root tosses to process with {NUM_WORKERS} workers."
    )

    # Create a pool of workers, initializing each with the massive data_pools dict
    with multiprocessing.Pool(
        processes=NUM_WORKERS, initializer=init_worker, initargs=(data_pools,)
    ) as pool:
        results = list(
            tqdm(pool.imap(generate_rally_tree_worker, tasks), total=len(tasks))
        )

    print("\n--- Generation Complete ---")

    # Aggregate results and tracking sets
    total_files = 0
    all_used_tosss = set()
    all_used_serves = set()
    all_used_returns = set()

    for files, sets in results:
        total_files += files
        all_used_tosss.update(sets["tosses"])
        all_used_serves.update(sets["serves"])
        all_used_returns.update(sets["returns"])

    print(f"Successfully generated {total_files} rally files in '{OUTDIR}'.")

    # --- Final Usage Report ---
    unused_tosss = data_pools["all_toss_ids"] - all_used_tosss
    unused_serves = data_pools["all_serve_ids"] - all_used_serves
    unused_returns = data_pools["all_return_ids"] - all_used_returns

    print("\n--- Data Pool Usage Report ---")
    print(f"Tosses:   {len(all_used_tosss)} used ({len(unused_tosss)} unused)")
    print(f"Serves:   {len(all_used_serves)} used ({len(unused_serves)} unused)")
    print(f"Returns:  {len(all_used_returns)} used ({len(unused_returns)} unused)")

    # Optional: Save unused IDs to a file for inspection
    try:
        np.savez_compressed(
            os.path.join(OUTDIR, "_unused_ids.npz"),
            unused_tosss=np.array(list(unused_tosss), dtype=object),
            unused_serves=np.array(list(unused_serves), dtype=object),
            unused_returns=np.array(list(unused_returns), dtype=object),
        )
        print("Saved lists of unused IDs to '_unused_ids.npz'.")
    except Exception as e:
        print(f"Could not save unused IDs: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method(
        "fork", force=True
    )  # Use spawn instead of fork if not on Linux
    main()
