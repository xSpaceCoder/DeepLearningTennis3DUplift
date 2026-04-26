import numpy as np
import mujoco


from syntheticdataset.mujocosimulation import (
    _calc_cammatrices,
    world2cam,
    cam2img,
    _count_hits,
    NET_TENNIS_HEIGHT,
    NET_TENNIS_WIDTH,
    MAX_TIME,
    TIMESTEP,
    FPS,
    CAMERA,
    WIDTH,
    HEIGHT,
)


def simulate_trajectory(model, data, mode, direction):
    # renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    mujoco.mj_step(model, data)
    # renderer.update_scene(data, camera=CAMERA)

    positions, velocities, rotations = [], [], []
    ex_mats = []
    in_mats = []
    times = []
    next_save_time = 0.0
    while next_save_time < MAX_TIME:
        steps = round((next_save_time - data.time) / TIMESTEP)
        mujoco.mj_step(model, data, steps)

        correct_side = (
            data.qpos[0] < 0 if direction == "far_to_close" else data.qpos[0] > 0
        )  # check if ball is on opponents side
        # check if ball is out of bounds
        if mode == "final_lose":  # no bounce
            if abs(data.qpos[0]) > 6 or abs(data.qpos[1]) > 3:
                break
        elif mode == "final_win":  # two bounces on opponent side
            if correct_side and (
                abs(data.qpos[0]) > 1.38
                or abs(data.qpos[1]) > 0.77
                or data.qpos[2] < 0.7
            ):
                break
        elif mode == "groundstroke":  # one bounce on opponent side
            if correct_side and (abs(data.qpos[0]) > 4.5 or abs(data.qpos[1]) > 2.5):
                break
        elif (
            mode == "first_good"
        ):  # first bounce on players side, second bounce on opponent side
            if correct_side and (abs(data.qpos[0]) > 2.5 or abs(data.qpos[1]) > 1.5):
                break
        elif (
            mode == "first_short"
        ):  # first bounce on players side, second bounce on players side
            if abs(data.qpos[0]) > 2.5 or abs(data.qpos[1]) > 1.5 or data.qpos[2] < 0.5:
                break
        elif mode == "first_long":  # one bounce on players side
            if correct_side and abs(data.qpos[0]) > 2.5 or abs(data.qpos[1]) > 1.5:
                break

        # check if ball is still in the image plane
        # renderer.update_scene(data, camera=CAMERA)
        ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA)
        r_cam = world2cam(data.qpos[0:3].copy(), ex_mat)
        r_img = cam2img(r_cam, in_mat[:3, :3])
        if not np.all((r_img >= 0) & (r_img < np.array([WIDTH, HEIGHT]))):
            break

        positions.append(data.qpos[0:3].copy())
        velocities.append(data.qvel[0:3].copy())
        rotations.append(data.qvel[3:6].copy())
        times.append(next_save_time)
        ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA)
        in_mat = in_mat[:3, :3]
        ex_mats.append(ex_mat)
        in_mats.append(in_mat)
        next_save_time += 1 / FPS

    # check if list is empty -> I do the check later, but it is also needed now because the following code would fail otherwise
    minimum_length = int(round(0.2 * FPS))  # less than 0.2 seconds is unreasonable
    if len(positions) < minimum_length:
        print(
            f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
        )

    # calculate the number of bounces
    hits_opponent, hits_own, hits_ground = _count_hits(positions, direction)
    # check maximum height
    max_height = 1.4 if "first" in mode else 1.8
    if np.max(np.array(positions)[:, 2]) > max_height:
        print(
            f"Trajectory too high: {np.max(np.array(positions)[:, 2])} m, mode: {mode}, direction: {direction}"
        )

    min_percent = (
        0.2  # minimum percentage of time before cutting for a trajectory to be valid
    )
    # cut trajectory
    if mode == "final_lose":  # no bounce
        if len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(
                    f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
                )
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = (
                positions[:cut_index],
                velocities[:cut_index],
                rotations[:cut_index],
                times[:cut_index],
            )
            hits_ground = []
    elif mode == "final_win":  # two bounces on opponent side
        if len(hits_opponent) > 2:
            cut_time = hits_opponent[2]
            if cut_time < min_percent * MAX_TIME:
                print(
                    f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
                )
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = (
                positions[:cut_index],
                velocities[:cut_index],
                rotations[:cut_index],
                times[:cut_index],
            )
            hits_opponent = hits_opponent[:2]
            hits_ground = []
        elif len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(
                    f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
                )
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = (
                positions[:cut_index],
                velocities[:cut_index],
                rotations[:cut_index],
                times[:cut_index],
            )
            hits_ground = []
    elif mode == "groundstroke":  # one bounce on opponent side
        if len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(
                    f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
                )
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = (
                positions[:cut_index],
                velocities[:cut_index],
                rotations[:cut_index],
                times[:cut_index],
            )
            hits_ground = []
    elif (
        mode == "first_good"
    ):  # first bounce on players side, second bounce on opponent side
        if len(hits_opponent) > 1:
            cut_time = hits_opponent[1]
            if cut_time < min_percent * MAX_TIME:
                print(
                    f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
                )
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = (
                positions[:cut_index],
                velocities[:cut_index],
                rotations[:cut_index],
                times[:cut_index],
            )
            hits_opponent = hits_opponent[:1]
            hits_ground = []
        elif len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(
                    f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
                )
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = (
                positions[:cut_index],
                velocities[:cut_index],
                rotations[:cut_index],
                times[:cut_index],
            )
            hits_ground = []
    elif (
        mode == "first_short"
    ):  # first bounce on players side, second bounce on players side
        if len(hits_own) > 2:
            cut_time = hits_own[2]
            if cut_time < min_percent * MAX_TIME:
                print(
                    f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
                )
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = (
                positions[:cut_index],
                velocities[:cut_index],
                rotations[:cut_index],
                times[:cut_index],
            )
            hits_own = hits_own[:2]
            hits_opponent = []
            hits_ground = []
        elif len(hits_opponent) > 0:
            cut_time = hits_opponent[0]
            if cut_time < min_percent * MAX_TIME:
                print(
                    f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
                )
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = (
                positions[:cut_index],
                velocities[:cut_index],
                rotations[:cut_index],
                times[:cut_index],
            )
            hits_opponent = []
            hits_ground = []
        elif len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(
                    f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
                )
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = (
                positions[:cut_index],
                velocities[:cut_index],
                rotations[:cut_index],
                times[:cut_index],
            )
            hits_ground = []
    elif mode == "first_long":  # one bounce on players side
        if len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(
                    f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}"
                )
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = (
                positions[:cut_index],
                velocities[:cut_index],
                rotations[:cut_index],
                times[:cut_index],
            )
            hits_ground = []

    # ensure minimum length of 7 frames
    if len(positions) < minimum_length:
        print(
            f"Trajectory too short after cutting: {len(positions)} frames, mode: {mode}, direction: {direction}"
        )

    # check if ball is above the net
    heights_close_to_net = np.array(positions)[:, 2][
        np.abs(np.array(positions)[:, 0]) < 0.04
    ]
    widths_close_to_net = np.array(positions)[:, 1][
        np.abs(np.array(positions)[:, 0]) < 0.04
    ]
    if (
        len(heights_close_to_net) > 0
        and np.max(heights_close_to_net) < NET_TENNIS_HEIGHT
        and np.min(np.abs(widths_close_to_net)) < NET_TENNIS_WIDTH / 2
    ):
        print(
            f"Trajectory too low: max height {np.max(heights_close_to_net)} m, mode: {mode}, direction: {direction}"
        )

    # check if final ball position is on the correct side
    is_opposite_site = lambda x: x < 0 if direction == "far_to_close" else x > 0
    if mode == "final_lose":
        if not is_opposite_site(positions[-1][0]):
            print(
                f"Final position not on opponent side: {positions[-1][0]}, mode: {mode}, direction: {direction}"
            )
    elif mode == "first_long":
        if not is_opposite_site(positions[-1][0]):
            print(
                f"Final position not on opponent side: {positions[-1][0]}, mode: {mode}, direction: {direction}"
            )
    # maybe add first_short -> should it be on players side? np.max(positions[:, 0]) < 0

    # check if number of bounces is fitting to the mode
    if mode == "final_lose":  # no bounce
        if len(hits_opponent) == 0 and len(hits_own) == 0 and len(hits_ground) == 0:
            pass
        else:
            print(
                f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}"
            )
    elif mode == "final_win":  # two bounces on opponent side
        if len(hits_opponent) == 2 and len(hits_own) == 0 and len(hits_ground) == 0:
            pass
        else:
            print(
                f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}"
            )
    elif mode == "groundstroke":  # one bounce on opponent side
        if len(hits_opponent) == 1 and len(hits_own) == 0 and len(hits_ground) == 0:
            pass
        else:
            print(
                f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}"
            )
    elif (
        mode == "first_good"
    ):  # first bounce on players side, second bounce on opponent side
        if len(hits_opponent) == 1 and len(hits_own) == 1 and len(hits_ground) == 0:
            pass
        else:
            print(
                f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}"
            )
    elif (
        mode == "first_short"
    ):  # first bounce on players side, second bounce on players side
        if len(hits_opponent) == 0 and len(hits_own) == 2 and len(hits_ground) == 0:
            pass
        else:
            print(
                f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}"
            )
    elif mode == "first_long":  # one bounce on players side
        if len(hits_opponent) == 0 and len(hits_own) == 1 and len(hits_ground) == 0:
            pass
        else:
            print(
                f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}"
            )

    return (
        np.array(positions),
        np.array(velocities),
        np.array(rotations),
        np.array(times),
        np.array(ex_mats),
        np.array(in_mats),
    )


if __name__ == "__main__":
    pass
