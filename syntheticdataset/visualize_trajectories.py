# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from paths import data_path
from syntheticdataset.helper import court_connections, court_points


def show_trajectory(folder, mode, number, direction, save_path=None):
    print(
        f"Loading trajectory {number} from {os.path.join(data_path, folder, mode, direction)}"
    )
    path = os.path.join(data_path, folder, mode, direction, f"trajectory_{number:04}")
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist.")
    file_name = os.path.join("positions.npy")
    file_path = os.path.join(path, file_name)

    positions = np.load(file_path)  # shape: (T, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-13, 13)
    ax.set_ylim(-6, 6)
    ax.set_zlim(0.0, 2.0)
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="#f87825",
    )
    ax.scatter(
        positions[0, 0], positions[0, 1], positions[0, 2], color="red", label="Start"
    )
    ax.scatter(
        positions[-1, 0], positions[-1, 1], positions[-1, 2], color="green", label="End"
    )
    for connection in court_connections:
        ax.plot(
            court_points[connection, 0],
            court_points[connection, 1],
            court_points[connection, 2],
            "k",
        )
    ax.legend()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()

    fig, ax = plt.subplots()
    ax.set(positions[:, 0], positions[:, 1])
    plt.show()


def visualize_trajectory_with_camera_view(traj_path, save_path=None):
    """Visualize trajectory in 3D and from camera viewpoint using camera matrices.

    Args:
        traj_path: Path to trajectory directory containing positions.npy, Mint.npy, Mext.npy
        save_path: Optional path to save the figure. If None, displays interactively.
    """
    from syntheticdataset.helper import world2cam, cam2img, HEIGHT, WIDTH

    if not os.path.exists(traj_path):
        raise ValueError(f"Path {traj_path} does not exist.")

    # Load data
    positions = np.load(os.path.join(traj_path, "positions.npy"))
    rotation = np.load(os.path.join(traj_path, "rotations.npy"))
    velocities = np.load(os.path.join(traj_path, "velocities.npy"))
    rotation2 = np.load(
        os.path.join(
            "/data/goeppeal/data_tennis/tmp/groundstrok/close_to_far/-200/groundstroke/close_to_far/in/trajectory_0001",
            "rotations.npy",
        )
    )
    positions2 = np.load(
        os.path.join(
            "/data/goeppeal/data_tennis/tmp/groundstrok/close_to_far/-200/groundstroke/close_to_far/in/trajectory_0001",
            "positions.npy",
        )
    )
    velocities2 = np.load(
        os.path.join(
            "/data/goeppeal/data_tennis/tmp/groundstrok/close_to_far/-200/groundstroke/close_to_far/in/trajectory_0001",
            "velocities.npy",
        )
    )

    print(f"Orange Trajectory: {traj_path}")
    print(
        f"Purple Trajectory: /data/goeppeal/data_tennis/tmp/groundstrok/close_to_far/-200/groundstroke/close_to_far/in/trajectory_0001"
    )
    print(f"orange velocities: {velocities[0]}, \npurple velocities: {velocities2[0]}")
    print(
        f"Orange Start position: {positions[0]}, \nPurple Start position: {positions2[0]}"
    )
    print(
        f"Orange Start rotation: {rotation[0]}, \nPurple Start rotation: {rotation2[0]}"
    )
    Mint = np.load(os.path.join(traj_path, "Mint.npy"))
    Mext = np.load(os.path.join(traj_path, "Mext.npy"))

    Mint = Mint[0]
    Mext = Mext[0]

    # Create figure with two subplots: 3D view and camera view
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.set_box_aspect(
        [3, 1.6, 0.8]
    )  # Aspect ratio based on court dimensions and height
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(-13, 13)
    ax1.set_ylim(-6, 6)
    ax1.set_zlim(0.0, 2.0)
    ax1.set_zticks([0, 0.5, 1.0, 1.5, 2.0])

    ax1.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="#f87825",
        zorder=3,
        label="Topspin",
    )
    ax1.plot(
        positions2[:, 0],
        positions2[:, 1],
        positions2[:, 2],
        color="#A33CA2",
        zorder=4,
        label="Backspin",
    )
    ax1.legend()

    # Draw court in 3D
    for connection in court_connections:
        ax1.plot(
            court_points[connection, 0],
            court_points[connection, 1],
            court_points[connection, 2],
            "k",
            zorder=2,
        )

    """ # Draw rectangle in 3D
    rectangle_points = np.array(
        [
            [-15, -7, 0],
            [15, -7, 0],
            [15, 7, 0],
            [-15, 7, 0],
            [-15, -7, 0],  # close the rectangle
        ]
    )
    ax1.plot(
        rectangle_points[:, 0],
        rectangle_points[:, 1],
        rectangle_points[:, 2],
        color="grey",
        linewidth=2,
    ) """

    fig = plt.figure(figsize=(16, 8))
    # Camera view (right subplot)
    ax2 = fig.add_subplot(111)
    ax2.set_xlim(0, WIDTH)
    ax2.set_ylim(HEIGHT, 0)  # Inverted y-axis for image coordinates
    ax2.set_xlabel("Image X (pixels)")
    ax2.set_ylabel("Image Y (pixels)")
    ax2.set_title("Camera View")

    # Project trajectory positions to image
    positions_cam = world2cam(positions, Mext)
    positions_img = cam2img(positions_cam, Mint)

    # Project court points to image
    court_points_cam = world2cam(court_points, Mext)
    court_points_img = cam2img(court_points_cam, Mint)

    """ # Project rectangle points to image
    rectangle_cam = world2cam(rectangle_points, Mext)
    rectangle_img = cam2img(rectangle_cam, Mint) """

    # Draw projected trajectory
    ax2.plot(
        positions_img[:, 0], positions_img[:, 1], "b-", linewidth=2, label="Trajectory"
    )
    """
    ax2.scatter(
        positions_img[0, 0],
        positions_img[0, 1],
        color="red",
        s=100,
        label="Start",
        zorder=5,
    )
    ax2.scatter(
        positions_img[-1, 0],
        positions_img[-1, 1],
        color="green",
        s=100,
        label="End",
        zorder=5,
    )"""

    # Draw projected court lines
    for connection in court_connections:
        ax2.plot(
            court_points_img[connection, 0],
            court_points_img[connection, 1],
            "k-",
            linewidth=1,
        )

    """ # Draw projected rectangle
    ax2.plot(
        rectangle_img[:, 0],
        rectangle_img[:, 1],
        color="grey",
        linewidth=2,
        label="Rectangle",
    ) """

    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory visualization to {save_path}")
    else:
        plt.show()


def show_trajectory_from_path(traj_path, save_path=None):
    """Load a trajectory directory directly (absolute path) and plot/save it.

    traj_path should be the directory that contains positions.npy (e.g., .../trajectory_0000)
    """
    if not os.path.exists(traj_path):
        raise ValueError(f"Path {traj_path} does not exist.")
    file_path = os.path.join(traj_path, "positions.npy")
    if not os.path.exists(file_path):
        raise ValueError(f"positions.npy not found in {traj_path}")

    positions = np.load(file_path)
    positions2 = np.load(
        os.path.join(
            "/data/goeppeal/data_tennis/tmp/groundstroke/close_to_far/in/trajectory_0000",
            "positions.npy",
        )
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-13, 13)
    ax.set_ylim(-6, 6)
    ax.set_zlim(0.0, 2.0)
    ax.set_title(os.path.basename(traj_path))
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.scatter(
        positions[0, 0], positions[0, 1], positions[0, 2], color="red", label="Start"
    )
    ax.scatter(
        positions[-1, 0], positions[-1, 1], positions[-1, 2], color="green", label="End"
    )
    for connection in court_connections:
        ax.plot(
            court_points[connection, 0],
            court_points[connection, 1],
            court_points[connection, 2],
            "k",
        )
    ax.legend()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()

    # xy plane
    fig, ax = plt.subplots()
    ax.set(xlim=(-15, 15), ylim=(-8, 8))
    ax.scatter(positions[0, 0], positions[0, 1], color="red", label="Start")
    ax.scatter(positions[-1, 0], positions[-1, 1], color="green", label="End")
    for connection in court_connections:
        ax.plot(
            court_points[connection, 0],
            court_points[connection, 1],
            "k",
        )
    ax.grid(True)
    ax.plot(positions[:, 0], positions[:, 1])
    plt.show()

    # xz plane
    fig, ax = plt.subplots()
    ax.set(xlim=(-15, 15), ylim=(0, 3.5))
    plt.axhline(y=1.55, color="black", linestyle="--")
    for connection in court_connections:
        ax.plot(
            court_points[connection, 0],
            court_points[connection, 2],
            "k",
        )
    ax.grid(True)
    ax.plot(positions[:, 0], positions[:, 2], color="#f87825", label="topspin")
    ax.plot(positions2[:, 0], positions2[:, 2], color="#A33CA2", label="backspin")
    ax.legend()
    plt.show()

    rotations_path = os.path.join(traj_path, "rotations.npy")
    rotations = np.load(rotations_path)
    print(f"rotation at the beginning: {rotations[0]}")
    print(f"Rotation at the end: {rotations[-1]}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Visualize a synthetic trajectory (plot or save)."
    )
    p.add_argument(
        "-t",
        "--trajectory",
        help="Absolute path to a trajectory directory (contains positions.npy)",
    )
    p.add_argument(
        "-s",
        "--save",
        help="If provided, save the plot to this PNG path instead of showing",
    )
    # legacy options (optional) to use the original data_path-based loader
    p.add_argument(
        "--folder", help="folder under data_path (legacy)", default="sanitycheck"
    )
    p.add_argument("--mode", help="mode under folder (legacy)", default="final_win")
    p.add_argument(
        "--direction", help="direction under folder (legacy)", default="close_to_far"
    )
    p.add_argument("--number", help="trajectory number (legacy)", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    traj_path = "/data/goeppeal/data_tennis/tmp/groundstrok/close_to_far/200/groundstroke/close_to_far/in/trajectory_0001"
    visualize_trajectory_with_camera_view(traj_path)
    show_trajectory_from_path(traj_path)

    """ 
    # If a trajectory absolute path is provided, prefer it
    if args.trajectory:
        show_trajectory_from_path(args.trajectory, save_path=args.save)
    else:
        # fallback to original behavior (looping was original; keep it minimal)
        show_trajectory(
            args.folder, args.mode, args.number, args.direction, save_path=args.save
        ) """

# %%
