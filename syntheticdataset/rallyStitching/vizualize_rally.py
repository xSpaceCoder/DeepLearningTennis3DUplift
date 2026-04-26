# %%
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("/home/mmc-user/tennisuplifting/DeepLearningTennis3DUplift")
print(os.getcwd())
from syntheticdataset.helper import court_connections, court_points


def generate_rally_vizualize_rally(rally_path):
    positions = np.load(rally_path, allow_pickle=True)["positions"]
    name = rally_path.split("/")[-1].split(".")[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-13, 13)
    ax.set_ylim(-6, 6)
    ax.set_zlim(0.0, 2.0)
    ax.set_title(f"Rally Visualization: {name}")
    for i in range(len(positions)):
        ax.plot(
            positions[i][:, 0],
            positions[i][:, 1],
            positions[i][:, 2],
            label=f"Shot {i+1}",
        )
    for connection in court_connections:
        ax.plot(
            court_points[connection, 0],
            court_points[connection, 1],
            court_points[connection, 2],
            "k",
        )
    ax.legend()
    plt.show()

    # xy plane
    fig, ax = plt.subplots()
    ax.set(xlim=(-15, 15), ylim=(-8, 8))
    ax.set_title(f"Rally Visualization: {name}")
    for i in range(len(positions)):
        ax.plot(positions[i][:, 0], positions[i][:, 1], label=f"Shot {i+1}")
    for connection in court_connections:
        ax.plot(
            court_points[connection, 0],
            court_points[connection, 1],
            "k",
        )
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    rally_path = (
        "/data/goeppeal/data_tennis/stitched_rallies_02-24/toss_21_branch_230.npz"
    )
    generate_rally_vizualize_rally(rally_path)
# %%
