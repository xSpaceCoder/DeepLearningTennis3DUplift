# %%
import matplotlib.pyplot as plt
from uplifting.helper import HEIGHT, WIDTH
from visualization.show_pred_vs_gt_trajectory import (
    get_background_image,
    img_connections,
)


def verifyCameraParameters(trajectory_2d, table_img, trajectory_id, Mint, Mext):
    """
    Visualize predicted vs ground truth trajectories.
    Args:
        trajectory_2d (np.ndarray): 2D trajectory points.
        table_img (np.ndarray): Image of the table.
        trajectory_id (int): Identifier for the trajectory.
    """
    from uplifting.helper import cam2img, world2cam
    from uplifting.data import CAMERA_VIEW_POINTS

    fig, ax = plt.subplots()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(-HEIGHT, 0)

    ppa_cam = world2cam(CAMERA_VIEW_POINTS, Mext)
    ppa_img = cam2img(ppa_cam, Mint)

    ax.scatter(
        ppa_img[:, 0], -ppa_img[:, 1], c="r", marker="o", s=25, zorder=2, clip_on=False
    )
    ax.plot(
        trajectory_2d[:, 0] * WIDTH,
        -trajectory_2d[:, 1] * HEIGHT,
        color="black",
        linewidth=2,
        zorder=2,
    )

    # Draw court lines using connections
    for connection in img_connections:
        pt1, pt2 = connection
        if pt1 < len(table_img) and pt2 < len(table_img):
            ax.plot(
                [table_img[pt1, 0] * WIDTH, table_img[pt2, 0] * WIDTH],
                [-table_img[pt1, 1] * HEIGHT, -table_img[pt2, 1] * HEIGHT],
                "k-",
                linewidth=2,
                zorder=1,
            )
    plt.show()
    fig.savefig("table_img.png")


# %%
