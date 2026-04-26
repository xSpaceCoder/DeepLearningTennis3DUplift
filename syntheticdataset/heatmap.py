"""
This module provides functions to load tennis ball trajectory data from a structured directory, process it,
and visualize it as density heatmaps. The visualizations include:
1. Ball XY Density (Top-down)
2. Bounce Point Density (Top-down)
3. Ball XZ Density (Side view)
4. Starting Positions (Top-down)

If the `save` option is enabled, the generated figures are saved to disk as png and svg.

The filter which trajectories to include in the heatmaps can be configured starting line 503.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
from tqdm import tqdm

from syntheticdataset.helper import (
    TENNIS_COURT_LENGTH,
    TENNIS_COURT_WIDTH,
    NET_TENNIS_HEIGHT,
    SERVICELINE_X_CLOSE,
)

from paths import data_path as ROOT_DIR, vizualisation_path


THESIS_COLORS = [
    "#2556ac",
    "#00ae5b",
    "#ffe200",
    "#f87825",
    "#dc3241",
    "#a33ca2",
]
THESIS_CMAP = LinearSegmentedColormap.from_list("thesis", THESIS_COLORS)
BACKGROUND_COLOR = "#ffffff"
FOREGROUND_COLOR = "#000000"
TITLE_FONT_SIZE = 22
AXIS_LABEL_FONT_SIZE = 18
AXIS_TICK_FONT_SIZE = 14
COLORBAR_LABEL_FONT_SIZE = 18
COLORBAR_TICK_FONT_SIZE = 14


def get_trajectory_paths(root_dir, stroke_categories, direction, in_out_options):
    """
    Constructs the list of trajectory folder paths based on user filters.

    Structure: root / stroke / direction / in_out / trajectoryXXXX / ...
    """
    search_paths = []

    for stroke in stroke_categories:
        for status in in_out_options:
            # Construct path pattern: root/stroke/direction/status/trajectory*
            path_pattern = os.path.join(
                root_dir, stroke, direction, status, "trajectory*"
            )
            found_folders = glob(path_pattern)
            search_paths.extend(found_folders)

    return search_paths


def load_and_process_data(root_dir, stroke_categories, direction, in_out_options):
    """
    Reads .npy files based on the complex directory structure.
    """

    # Storage for plotting
    all_x = []
    all_y = []
    all_z = []

    bounce_x = []
    bounce_y = []
    # starting positions (first frame of each trajectory)
    start_x = []
    start_y = []

    # 1. Gather all folder paths based on filters
    traj_folders = get_trajectory_paths(
        root_dir, stroke_categories, direction, in_out_options
    )

    print(
        f"Found {len(traj_folders)} trajectories matching filters: {stroke_categories}, {direction}, {in_out_options}"
    )

    for folder_path in tqdm(traj_folders):
        try:
            # Load files
            # shape check: positions is often 3xN, we want consistent indexing
            pos = np.load(os.path.join(folder_path, "positions.npy"))
            times = np.load(os.path.join(folder_path, "times.npy"))

            # Ensure 3xN shape (Rows: x, y, z)
            if pos.shape[0] != 3:
                pos = pos.T

            # Times is 1D
            times = times.flatten()

            # Check for bounces
            bounce_file = os.path.join(folder_path, "bounces.npy")
            bounces = []
            if os.path.exists(bounce_file):
                bounces = np.load(bounce_file)

            # --- 1. Aggregating Ball Positions ---
            all_x.append(pos[0, :])
            all_y.append(pos[1, :])
            all_z.append(pos[2, :])

            # --- Start Positions (first frame of this trajectory) ---
            # Ensure there is at least one frame
            if pos.shape[1] > 0:
                start_x.append(pos[0, 0])
                start_y.append(pos[1, 0])

            # --- 2. Processing Bounces ---
            if bounces.size > 0:
                # Assuming times are sorted and bounces match exact frames or are very close
                # searchsorted finds the insertion point to maintain order, which corresponds to the index
                # if the timestamp exists in the array.
                bounces = np.atleast_1d(bounces)
                indices = np.searchsorted(times, bounces)

                # Clip indices to be safe (in case bounce is slightly past last frame due to float precision)
                indices = np.clip(indices, 0, len(times) - 1)

                # Append positions at these indices
                # Using 'extend' because one trajectory might have multiple bounces
                bounce_x.extend(pos[0, indices])
                bounce_y.extend(pos[1, indices])

        except Exception as e:
            # print(f"Skipping {folder_path}: {e}")
            continue

    if not all_x:
        print("No data found! Check your directory structure or filters.")
        return None, None

    print("Concatenating data...")
    flat_x = np.concatenate(all_x)
    flat_y = np.concatenate(all_y)
    flat_z = np.concatenate(all_z)

    return (
        (flat_x, flat_y, flat_z),
        (np.array(bounce_x), np.array(bounce_y)),
        (np.array(start_x), np.array(start_y)),
    )


def draw_tennis_court(ax, view="top"):
    """
    Draws Lawn Tennis court dimensions (in meters).
    Assumes (0,0) is the center of the net.
    """
    # Dimensions in meters
    length = TENNIS_COURT_LENGTH
    width_singles = TENNIS_COURT_WIDTH
    width_doubles = 10.97
    half_len = length / 2
    half_wid_s = width_singles / 2
    half_wid_d = width_doubles / 2
    service_line_dist = SERVICELINE_X_CLOSE
    net_height_center = NET_TENNIS_HEIGHT

    if view == "top":
        # Outer Boundary (Doubles)
        ax.add_patch(
            patches.Rectangle(
                (-half_len, -half_wid_d),
                length,
                width_doubles,
                fill=False,
                edgecolor=FOREGROUND_COLOR,
                lw=1.5,
            )
        )
        # Singles Sidelines
        ax.add_patch(
            patches.Rectangle(
                (-half_len, -half_wid_s),
                length,
                width_singles,
                fill=False,
                edgecolor=FOREGROUND_COLOR,
                lw=1,
            )
        )
        # Service Lines
        ax.plot(
            [-service_line_dist, -service_line_dist],
            [-half_wid_s, half_wid_s],
            color=FOREGROUND_COLOR,
            lw=1,
        )
        ax.plot(
            [service_line_dist, service_line_dist],
            [-half_wid_s, half_wid_s],
            color=FOREGROUND_COLOR,
            lw=1,
        )
        # Center Service Line
        ax.plot(
            [-service_line_dist, service_line_dist],
            [0, 0],
            color=FOREGROUND_COLOR,
            lw=1,
        )
        # Net
        ax.plot(
            [0, 0],
            [-half_wid_d, half_wid_d],
            color=FOREGROUND_COLOR,
            ls=":",
            lw=2,
        )

    elif view == "side":
        # Side view typically X vs Z
        # Ground line
        ax.plot([-half_len, half_len], [0, 0], color=FOREGROUND_COLOR, lw=2)
        # Net
        ax.plot([0, 0], [0, net_height_center], color=FOREGROUND_COLOR, lw=2)
        # Net visual guide (top of net across width)
        rect = patches.Rectangle(
            (-0.1, 0), 0.2, net_height_center, color=FOREGROUND_COLOR, alpha=0.35
        )
        ax.add_patch(rect)


def style_plot(fig, ax):
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.tick_params(colors=FOREGROUND_COLOR, labelsize=AXIS_TICK_FONT_SIZE)
    ax.xaxis.label.set_color(FOREGROUND_COLOR)
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_color(FOREGROUND_COLOR)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.title.set_color(FOREGROUND_COLOR)
    ax.title.set_size(TITLE_FONT_SIZE)

    for spine in ax.spines.values():
        spine.set_color(FOREGROUND_COLOR)


def style_colorbar(colorbar):
    colorbar.ax.set_facecolor(BACKGROUND_COLOR)
    colorbar.ax.yaxis.label.set_color(FOREGROUND_COLOR)
    colorbar.ax.yaxis.label.set_size(COLORBAR_LABEL_FONT_SIZE)
    colorbar.ax.tick_params(colors=FOREGROUND_COLOR, labelsize=COLORBAR_TICK_FONT_SIZE)
    colorbar.outline.set_edgecolor(FOREGROUND_COLOR)


def add_aligned_colorbar(fig, ax, mappable, label="Density"):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.12)
    colorbar = fig.colorbar(mappable, cax=cax, label=label)
    style_colorbar(colorbar)
    return colorbar


def save_heatmap_figures(
    figures,
    save_root,
    selected_strokes,
    direction,
    in_out_select,
    dpi=300,
    formats=("png", "svg"),
):
    """Save a set of matplotlib figures to disk with the folder structure:
    save_root / <StrokesConcatenatedCamelCase> / <direction> / <in_out_folder>

    Parameters
    - figures: dict of {title: figure} to save. Titles are sanitized into filenames.
    - save_root: base directory where images will be stored.
    - selected_strokes: list of stroke strings (will be concatenated in CamelCase).
    - direction: string (used as folder name).
    - in_out_select: list containing 'in' and/or 'out' (folder will be 'in', 'out' or 'in_out').
    - dpi: resolution for saved raster images.
    - formats: iterable of formats to save, e.g. ('png','svg').

    The function overwrites existing files with the same name.
    """

    # Build strokes folder name by concatenating CamelCase versions
    def camelcase_join(items):
        parts = []
        for it in items:
            # remove non-alphanum, split on underscores/dashes, capitalize
            clean = str(it).replace("-", "_")
            sub = "".join([p.capitalize() for p in clean.split("_") if p])
            if sub:
                parts.append(sub)
        return "".join(parts) if parts else "AllStrokes"

    strokes_folder = camelcase_join(selected_strokes)

    # Build in_out folder
    in_out_folder = "both" if len(in_out_select) > 1 else in_out_select[0]

    # Full destination dir
    dest_dir = os.path.join(save_root, strokes_folder, direction, in_out_folder)
    os.makedirs(dest_dir, exist_ok=True)

    def sanitize_filename(s):
        # Keep alphanum, dash and underscore; replace spaces with underscore
        import re

        s = s.strip()
        s = s.replace(" ", "_")
        s = re.sub(r"[^0-9A-Za-z_\-\.]+", "", s)
        return s

    for title, fig in figures.items():
        fname_base = sanitize_filename(title)
        if not fname_base:
            fname_base = "heatmap"

        for fmt in formats:
            out_path = os.path.join(dest_dir, f"{fname_base}.{fmt}")
            try:
                # For vector formats dpi is ignored, but matplotlib accepts it
                fig.savefig(
                    out_path,
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                )
            except Exception:
                # Best-effort: attempt saving via canvas if direct save fails
                try:
                    fig.canvas.print_figure(
                        out_path,
                        dpi=dpi,
                        facecolor=fig.get_facecolor(),
                    )
                except Exception:
                    # If saving fails, continue to next file
                    continue


def plot_heatmaps(
    traj_data,
    bounce_data,
    start_data=None,
    save=True,
    save_root="./heatmap_outputs",
    selected_strokes=None,
    direction_label=None,
    in_out_label=None,
    dpi=300,
    formats=("png", "svg"),
):
    """Plot heatmaps. Optionally save the produced figures.

    Parameters
    - traj_data, bounce_data: as before
    - save: bool, whether to save the three generated figures
    - save_root: str path where images will be saved (if save=True). If None, defaults to './heatmap_outputs'
    - selected_strokes: list of stroke names; defaults to module `SELECTED_STROKES`
    - direction_label: direction string; defaults to module `DIRECTION`
    - in_out_label: list of in/out selections; defaults to module `IN_OUT_SELECT`
    - dpi: save dpi
    - formats: formats to save (png/svg)
    """
    if traj_data is None:
        return

    # Use module defaults when not provided
    if selected_strokes is None:
        selected_strokes = SELECTED_STROKES
    if direction_label is None:
        direction_label = DIRECTION
    if in_out_label is None:
        in_out_label = IN_OUT_SELECT

    tx, ty, tz = traj_data
    bx, by = bounce_data
    # start positions (may be empty)
    if start_data is not None:
        sx, sy = start_data
    else:
        sx, sy = np.array([]), np.array([])

    plt.style.use("default")

    cmap = THESIS_CMAP
    bins = 150

    # Define plot limits (Meters)
    xlim = [-15, 15]  # Slightly larger than half-court (11.89m)
    ylim = [-8, 8]  # Slightly wider than doubles width (10.97m/2 = 5.48)
    zlim = [0, 5]  # Height up to 5 meters

    # --- 1. Ball XY Density (Top-down) ---
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    h1 = ax1.hist2d(tx, ty, bins=bins, cmap=cmap, range=[xlim, ylim], density=True)
    draw_tennis_court(ax1, view="top")
    # ax1.set_title("Top View")
    ax1.set_xlabel("Length (m)")
    ax1.set_ylabel("Width (m)")
    ax1.set_aspect("equal")
    style_plot(fig1, ax1)
    add_aligned_colorbar(fig1, ax1, h1[3])

    # --- 2. Bounce Point Density (Top-down) ---
    fig2, ax2 = plt.subplots(figsize=(14, 10))

    # Using specific range to match court size
    h2 = ax2.hist2d(bx, by, bins=bins, cmap=cmap, range=[xlim, ylim], density=True)
    draw_tennis_court(ax2, view="top")
    # ax2.set_title("Bounce Density (Top View)")
    ax2.set_xlabel("Length (m)")
    ax2.set_ylabel("Width (m)")
    ax2.set_aspect("equal")
    style_plot(fig2, ax2)
    add_aligned_colorbar(fig2, ax2, h2[3])

    # --- 3. Ball XZ Density (Side view) ---
    fig3, ax3 = plt.subplots(figsize=(16, 6))
    h3 = ax3.hist2d(tx, tz, bins=bins, cmap=cmap, range=[xlim, zlim], density=True)
    draw_tennis_court(ax3, view="side")
    # ax3.set_title("Side View")
    ax3.set_xlabel("Length (m)")
    ax3.set_ylabel("Height (m)")
    # Use automatic aspect for the side view so the full vertical range is visible
    # (using 'equal' here with a large X-range and a small Z-range produced a
    # very short axis height and appeared clipped in the figure layout)
    ax3.set_aspect("auto")
    style_plot(fig3, ax3)
    colorbar3 = fig3.colorbar(h3[3], ax=ax3, label="Density")
    style_colorbar(colorbar3)

    # --- 4. Starting Positions (Top-down) ---
    figures_add = {}
    if sx.size > 0:
        fig4, ax4 = plt.subplots(figsize=(14, 10))
        h4 = ax4.hist2d(sx, sy, bins=bins, cmap=cmap, range=[xlim, ylim], density=True)
        draw_tennis_court(ax4, view="top")
        # ax4.set_title("Starting Positions (Top View)")
        ax4.set_xlabel("Length (m)")
        ax4.set_ylabel("Width (m)")
        ax4.set_aspect("equal")
        style_plot(fig4, ax4)
        add_aligned_colorbar(fig4, ax4, h4[3])
        figures_add[ax4.get_title()] = fig4

    # Collect figures and titles for saving if requested
    figures = {
        ax1.get_title(): fig1,
        ax2.get_title(): fig2,
        ax3.get_title(): fig3,
    }

    # merge optional figures (e.g., starting positions)
    if figures_add:
        figures.update(figures_add)

    if save:
        if save_root is None:
            save_root = os.path.join(vizualisation_path, "heatmap_outputs")
        # Ensure in_out_label is list
        if isinstance(in_out_label, str):
            in_out_list = [in_out_label]
        else:
            in_out_list = list(in_out_label)

        # If both in and out are selected and the list contains both, use both
        # (save hierarchy will use 'in', 'out' or 'in_out')
        save_heatmap_figures(
            figures,
            save_root,
            selected_strokes,
            direction_label,
            in_out_list,
            dpi=dpi,
            formats=formats,
        )

    plt.tight_layout()
    plt.show()


# --- CONFIGURATION ---

# FILTERS
# Options: 'groundstroke', 'serve', 'volley', 'smash', 'lob', 'short', toss
SELECTED_STROKES = ["smash"]

# Options: 'far_to_close', 'close_to_far'
DIRECTION = "far_to_close"

# Options: 'in', 'out' (Can select one or both)
IN_OUT_SELECT = ["in", "out"]

# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure ROOT_DIR exists before running or replace with actual path
    if os.path.exists(ROOT_DIR):
        t_data, b_data, s_data = load_and_process_data(
            ROOT_DIR,
            SELECTED_STROKES,
            DIRECTION,
            IN_OUT_SELECT,
        )
        plot_heatmaps(t_data, b_data, start_data=s_data)
    else:
        print(f"Please update ROOT_DIR variable. Path not found: {ROOT_DIR}")

# %%
