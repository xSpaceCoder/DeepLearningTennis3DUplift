"""Transform the Uplifting Tennis data into the Deepmind json format"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import data_path as requested
from paths import data_path as DATA_PATH

data_path = os.path.join(DATA_PATH, "syntheticdata_03-04")


def load_tennis_trajectories(target_modes):
    """
    Scans data_path for specific modes and extracts initial states. Only valid trajectories are used.

    Args:
        target_modes: List of mode strings (e.g., ['groundstroke, serve, volley, smash, lob, short, toss'])

    Returns:
        List of dictionaries containing initial state data.
    """
    data_rows = []
    directions = ["far_to_close", "close_to_far"]

    print(f"Scanning for modes: {target_modes} in {data_path}...")

    for mode in target_modes:
        for direction in directions:
            # Construct path: data_path/mode/direction/in/trajectory_*
            search_pattern = os.path.join(data_path, mode, direction, "in/trajectory_*")
            traj_folders = glob.glob(search_pattern)

            if not traj_folders:
                print(f"No trajectory files found in {search_pattern}.")
                continue

            print(
                f"Found {len(traj_folders)} trajectories for mode '{mode}' ({direction})."
            )

            for folder in tqdm(traj_folders, desc=f"Processing {mode}"):
                try:
                    # Load necessary files
                    # Tennis data saves full trajectory (T, 3). We only need the first frame.
                    pos_file = os.path.join(folder, "positions.npy")
                    vel_file = os.path.join(folder, "velocities.npy")
                    rot_file = os.path.join(folder, "rotations.npy")

                    if not (os.path.exists(pos_file) and os.path.exists(vel_file)):
                        continue

                    # Load only the first frame (index 0)
                    pos = np.load(pos_file)[0]
                    vel = np.load(vel_file)[0]
                    rot = np.load(rot_file)[0]

                    # Generate a unique ID
                    traj_id = os.path.basename(folder).split("_")[-1]  # e.g., '0001'
                    unique_id = f"{mode}_{direction}_{traj_id}"

                    row = {
                        "id": unique_id,
                        # Position
                        "pos_x": float(pos[0]),
                        "pos_y": float(pos[1]),
                        "pos_z": float(pos[2]),
                        # Linear Velocity
                        "vel_x": float(vel[0]),
                        "vel_y": float(vel[1]),
                        "vel_z": float(vel[2]),
                        # Angular Velocity
                        "w_vel_x": float(rot[0]),
                        "w_vel_y": float(rot[1]),
                        "w_vel_z": float(rot[2]),
                    }
                    data_rows.append(row)
                except Exception as e:
                    # Skip corrupted files
                    print(f"Skipping trajectory in {folder} due to error: {e}")
                    continue

    return data_rows


def main():
    # 1. Generate Rallies (Returns)
    # The 'intermediate' mode corresponds to regular rallies/returns.
    print("--- Generaring Rallies Dataset ---")
    rallies_data = load_tennis_trajectories(
        target_modes=["groundstroke", "volley", "smash", "lob", "short"]
    )
    if rallies_data:
        out_rallies = os.path.join(data_path, "rallies.json")
        pd.DataFrame(rallies_data).to_json(out_rallies, orient="records")
        print(f"Saved {len(rallies_data)}/116.000 entries to {out_rallies}")
    else:
        print("Warning: No data found. 'rallies.json' was not created.")

    # 2. Generate Serves
    print("\n--- Generaring Serves Dataset ---")
    serves_data = load_tennis_trajectories(target_modes=["serve"])
    if serves_data:
        out_serves = os.path.join(data_path, "serves.json")
        pd.DataFrame(serves_data).to_json(out_serves, orient="records")
        print(f"Saved {len(serves_data)}/40.000 entries to {out_serves}")
    else:
        print("Warning: No serve data found. 'serves.json' was not created.")

    # 2. Generate ball tosses
    print("\n--- Generaring Toss Dataset ---")
    serves_data = load_tennis_trajectories(target_modes=["toss"])
    if serves_data:
        out_tosses = os.path.join(data_path, "tosses.json")
        pd.DataFrame(serves_data).to_json(out_tosses, orient="records")
        print(f"Saved {len(serves_data)}/20.000 entries to {out_tosses}")
    else:
        print("Warning: No toss data found. 'tosses.json' was not created.")


if __name__ == "__main__":
    main()
