# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from syntheticdataset.helper import court_connections, court_points


def backspin_vs_topspin(backspin_path, topspin_path):

    if not os.path.exists(backspin_path):
        raise ValueError(f"Path {backspin_path} does not exist.")
    if not os.path.exists(topspin_path):
        raise ValueError(f"Path {topspin_path} does not exist.")

    # Load data
    pos_ts = np.load(os.path.join(topspin_path, "positions.npy"))
    rot_ts = np.load(os.path.join(topspin_path, "rotations.npy"))
    pos_bs = np.load(os.path.join(backspin_path, "positions.npy"))
    rot_bs = np.load(os.path.join(backspin_path, "rotations.npy"))

    apex_ts = np.max(pos_ts[650:, 2])

    print(f"Topspin inital positions: {pos_ts[0]}")
    print(f"Backspin inital positions: {pos_bs[0]}")

    print(f"\n Topspin inital rotations: {rot_ts[0]}")
    print(f"Backspin inital rotations: {rot_bs[0]}")

    fig, ax = plt.subplots()
    ax.set(xlim=(-15, 15), ylim=(0, 3.5))
    ax.set_aspect("equal")
    plt.axhline(y=apex_ts, color="#333333", linestyle="--")
    for connection in court_connections:
        ax.plot(
            court_points[connection, 0],
            court_points[connection, 2],
            "k",
        )
    ax.grid(True)
    ax.plot(pos_ts[:, 0], pos_ts[:, 2], color="#f87825", label="topspin")
    ax.plot(pos_bs[:, 0], pos_bs[:, 2], color="#A33CA2", label="backspin")
    ax.legend()
    plt.show()


def sidespin(far_to_close_path, close_to_far_path):

    if not os.path.exists(far_to_close_path):
        raise ValueError(f"Path {far_to_close_path} does not exist.")
    if not os.path.exists(close_to_far_path):
        raise ValueError(f"Path {close_to_far_path} does not exist.")

    # Load data
    pos_close = np.load(os.path.join(close_to_far_path, "positions.npy"))
    rot_close = np.load(os.path.join(close_to_far_path, "rotations.npy"))
    pos_far = np.load(os.path.join(far_to_close_path, "positions.npy"))
    rot_far = np.load(os.path.join(far_to_close_path, "rotations.npy"))
    vel_close = np.load(os.path.join(close_to_far_path, "velocities.npy"))
    vel_far = np.load(os.path.join(far_to_close_path, "velocities.npy"))

    print(f"Close to far inital positions: {pos_close[0]}")
    print(f"Far to close inital positions: {pos_far[0]}")

    print(f"\n Close to far inital rotations: {rot_close[0]}")
    print(f"Far to close inital rotations: {rot_far[0]}")

    print(f"\n Close to far inital velocities: {vel_close[0]}")
    print(f"Far to close inital velocities: {vel_far[0]}")

    fig, ax = plt.subplots()
    ax.set_title(f"XY View for different spin trajectories")
    ax.set_aspect("equal")
    for connection in court_connections:
        ax.plot(
            court_points[connection, 0],
            court_points[connection, 1],
            "k",
        )
    ax.grid(True)
    ax.plot(pos_close[:, 0], pos_close[:, 1], color="#f87825", label="close to far")
    ax.plot(pos_far[:, 0], pos_far[:, 1], color="#A33CA2", label="far to close")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    base_path = "/data/goeppeal/data_tennis/tmp"
    # far to close
    print("Far to close")
    backspin_path = os.path.join(
        base_path,
        "topspin/groundstroke/close_to_far/in/trajectory_0000",
    )
    topspin_path = os.path.join(
        base_path,
        "topspin/groundstroke/close_to_far/in/trajectory_0001",
    )
    backspin_vs_topspin(backspin_path, topspin_path)

    # close to far
    print("Close to far")
    backspin_path = os.path.join(
        base_path,
        "groundstrok/close_to_far/-200/groundstroke/close_to_far/in/trajectory_0001",
    )
    topspin_path = os.path.join(
        base_path,
        "groundstrok/close_to_far/200/groundstroke/close_to_far/in/trajectory_0001",
    )
    backspin_vs_topspin(backspin_path, topspin_path)

    far_to_close_path = os.path.join(
        base_path,
        "corkspin/groundstroke/far_to_close/in/trajectory_0001",
    )
    close_to_far_path = os.path.join(
        base_path,
        "corkspin/groundstroke/close_to_far/in/trajectory_0001",
    )

    sidespin(far_to_close_path, close_to_far_path)
# %%
