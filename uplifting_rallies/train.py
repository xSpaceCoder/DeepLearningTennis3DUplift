import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--folder", type=str, default="tmp")
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--blur_strength", type=float, default=0.4)
    parser.add_argument("--randmiss_prob", type=float, default=0.05)
    parser.add_argument("--randomize_std", type=float, default=2)
    parser.add_argument("--model_name", type=str, default="connectstage")
    parser.add_argument("--model_size", type=str, default="large")
    parser.add_argument("--token_mode", type=str, default="dynamicAnkle")
    parser.add_argument("--transform_mode", type=str, default="global")
    parser.add_argument("--time_rotation", type=str, default="new")
    parser.add_argument("--interpolate_missing", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to latest.pt checkpoint to resume training",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch

if __name__ == "__main__" and args.debug:
    torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import roc_auc_score

from uplifting_rallies.helper import SummaryWriter, seed_worker
from uplifting_rallies.helper import binary_metrics
from uplifting_rallies.helper import update_ema
from uplifting_rallies.helper import (
    create_confusion_matrix,
    transform_rotationaxes,
)
from uplifting_rallies.helper import (
    save_model,
    WIDTH,
    HEIGHT,
    BALL_VISIBLE,
)
from uplifting_rallies.helper import world2cam, cam2img
from uplifting_rallies.data import (
    StitchedRallyDataset,
    RealInferenceDataset,
    RealInferenceRalliesDataset,
    TOPSPIN_CLASS,
    BACKSPIN_CLASS,
)
from uplifting_rallies.transformations import get_transforms, UnNormalizeImgCoords
from uplifting_rallies.model import get_model
from uplifting_rallies.config import TrainConfig

device = "cuda:0"
debug = False


def run(config, resume_path=None):
    start_epoch = 0
    checkpoint = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        config.ident = checkpoint["identifier"]
        start_epoch = checkpoint["epoch"] + 1
        print(
            f"Resuming from epoch {checkpoint['epoch']}, starting epoch {start_epoch}"
        )

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    logs_path = config.get_logs_path(debug)
    writer = SummaryWriter(logs_path)

    model = get_model(
        config.name,
        config.size,
        config.tabletoken_mode,
        config.time_rotation,
        config.interpolate_missing,
    ).to(device)
    model_ema = get_model(
        config.name,
        config.size,
        config.tabletoken_mode,
        config.time_rotation,
        config.interpolate_missing,
    ).to(device)
    model_ema = update_ema(
        model, model_ema, 0
    )  # copies the model to model_ema completely

    num_workers = 0 if debug else min(config.BATCH_SIZE, 16)
    train_transforms = get_transforms(config, "train")
    val_transforms = get_transforms(config, "val")
    trainset = StitchedRallyDataset("train", transforms=train_transforms)
    valset = StitchedRallyDataset("val", transforms=val_transforms)
    valset_real = RealInferenceDataset(transforms=val_transforms, mode="val")
    valset_real_rallies = RealInferenceRalliesDataset(
        transforms=val_transforms, mode="val"
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valloader_real = torch.utils.data.DataLoader(
        valset_real,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valloader_real_rallies = torch.utils.data.DataLoader(
        valset_real_rallies,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda")

    best_metric_trajectory, best_metric_spin, best_metric_synthetic = 1e8, 0, 1e8
    threshold_trajectory_metric = 0.007
    best_metric_spin_mixed, best_metric_trajectory_mixed = 0, 1e8

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        model_ema.load_state_dict(checkpoint["model_ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_metric_trajectory = checkpoint["best_metric_trajectory"]
        best_metric_spin = checkpoint["best_metric_spin"]
        best_metric_synthetic = checkpoint["best_metric_synthetic"]
        best_metric_spin_mixed = checkpoint["best_metric_spin_mixed"]
        best_metric_trajectory_mixed = checkpoint["best_metric_trajectory_mixed"]
        del checkpoint

    if start_epoch == 0:
        val(model_ema, valloader, writer, -1, device, config)
        val_real(model_ema, valloader_real, writer, -1, device, config)
        val_real_rallies(model_ema, valloader_real_rallies, writer, -1, device, config)
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model_ema = train(
            model,
            model_ema,
            trainloader,
            optimizer,
            scaler,
            writer,
            epoch,
            device,
            config,
        )
        metric_synthetic = val(model_ema, valloader, writer, epoch, device, config)
        metric_trajectory, metric_spin = val_real(
            model_ema, valloader_real, writer, epoch, device, config
        )
        metric_rallies, rallies_spin = val_real_rallies(
            model_ema, valloader_real_rallies, writer, epoch, device, config
        )

        # save model if metric improved
        if metric_trajectory < best_metric_trajectory:
            best_metric_trajectory = metric_trajectory
            save_model(model_ema, config, epoch, debug, name="model_trajectory.pt")
        if metric_spin >= best_metric_spin:  # larger F1 is better
            best_metric_spin = metric_spin
            save_model(model_ema, config, epoch, debug, name="model_spin.pt")
        if metric_synthetic < best_metric_synthetic:
            best_metric_synthetic = metric_synthetic
            save_model(model_ema, config, epoch, debug, name="model_synthetic.pt")
        if metric_trajectory <= threshold_trajectory_metric:
            if metric_spin > best_metric_spin_mixed:
                best_metric_spin_mixed = metric_spin
                save_model(model_ema, config, epoch, debug, name="model.pt")
                best_metric_trajectory_mixed = (
                    metric_trajectory  # update with value of saved model
                )
            elif (
                metric_spin == best_metric_spin_mixed
            ):  # important case, since 100% is realistic in our setting
                if (
                    metric_trajectory < best_metric_trajectory_mixed
                ):  # If spin metric is already maxed out, decide using trajectory metric
                    best_metric_trajectory_mixed = metric_trajectory
                    save_model(model_ema, config, epoch, debug, name="model.pt")

        # Save latest checkpoint for resuming
        save_path = config.get_pathforsaving(debug)
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_ema_state_dict": model_ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "identifier": config.get_identifier(),
                "best_metric_trajectory": best_metric_trajectory,
                "best_metric_spin": best_metric_spin,
                "best_metric_synthetic": best_metric_synthetic,
                "best_metric_spin_mixed": best_metric_spin_mixed,
                "best_metric_trajectory_mixed": best_metric_trajectory_mixed,
            },
            os.path.join(save_path, "latest.pt"),
        )


def train(
    model,
    model_ema,
    trainloader,
    optimizer,
    scaler,
    writer,
    epoch,
    device,
    config,
):
    # If we are using the multistage model, we want to prevent that the rotation influences the learning of the positions
    if config.name == "multistage":
        model.full_backprop = False

    # for mid-epoch validation
    mid_epoch_idx = len(trainloader) // 2

    model.train()
    iteration = epoch * len(trainloader)
    for i, data in enumerate(tqdm(trainloader)):
        r_img, court_img, mask, r_world, rotation, times, __, __, ankle_v = data
        r_img, court_img, mask, rotation, ankle_v = (
            r_img.to(device),
            court_img.to(device),
            mask.to(device),
            rotation.to(device),
            ankle_v.to(device),
        )
        r_world, times = r_world.to(device), times.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred_rotation, pred_position = model(r_img, court_img, mask, times, ankle_v)

            # remove image and gt world coordinates if detections are not visible and are not interpolated
            r_img, visibilities = r_img[:, :, :2], r_img[:, :, 2]
            if not config.interpolate_missing:
                # Sort such that visible balls are first
                keep_mask = visibilities == BALL_VISIBLE
                sorted_keep_mask, sort_indices = torch.sort(
                    keep_mask, dim=1, descending=True
                )
                # Apply sorting to image coordinates
                B, T, D = r_img.shape
                sort_indices_d = sort_indices.clone().unsqueeze(-1).expand(B, T, D)
                r_img = torch.gather(r_img, dim=1, index=sort_indices_d)
                # Apply sorting to world coordinates
                B, T, D = r_world.shape
                sort_indices_d = sort_indices.clone().unsqueeze(-1).expand(B, T, D)
                r_world = torch.gather(r_world, dim=1, index=sort_indices_d)
                # Apply sorting to times and mask
                times = torch.gather(times, dim=1, index=sort_indices)
                mask = torch.gather(mask, dim=1, index=sort_indices)
                mask = torch.where(
                    sorted_keep_mask, mask, torch.tensor(0, device=mask.device)
                )

            # If network output is in ball's coordinate system, transform the ground truth to the local coordinate system
            if config.transform_mode == "local":
                rotation = transform_rotationaxes(rotation, r_world)
            loss_rot = torch.sum(
                torch.sqrt(torch.sum((rotation - pred_rotation) ** 2, dim=-1)) * mask
            ) / torch.sum(mask)
            loss_pos = torch.sum(
                torch.nn.functional.mse_loss(pred_position, r_world, reduction="none")
                * mask.unsqueeze(-1)
            ) / torch.sum(mask)
            loss = loss_rot + loss_pos
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # gradient clipping
        scaler.step(optimizer)
        scaler.update()

        model_ema = update_ema(model, model_ema, config.ema_decay)

        writer.add_scalar("train/loss", loss.item(), iteration + i)
        writer.add_scalar("train/loss rotation", loss_rot.item(), iteration + i)
        writer.add_scalar("train/loss position", loss_pos.item(), iteration + i)

    return model_ema


def val(model, valloader, writer, epoch, device, config, step=None):
    if step is None:
        step = int(
            epoch * 2
        )  # Convert to integer: epoch 0 -> 0, epoch 0.5 -> 1, epoch 1 -> 2, etc.
    loss_fn = torch.nn.MSELoss()
    metric_fn = lambda x, y: torch.sum(torch.sqrt(torch.sum((x - y) ** 2, dim=-1)))
    metric_pos_fn = lambda x, y, mask: torch.sum(
        torch.sum(torch.sqrt(torch.sum((x - y) ** 2, dim=-1)) * mask, dim=1)
        / torch.sum(mask, dim=1)
    )

    def metric_2D_fn(pred_pos, gt_pos, mask, Mint, Mext):
        pred_pos_2D = cam2img(world2cam(pred_pos, Mext), Mint)
        gt_pos_2D = cam2img(world2cam(gt_pos, Mext), Mint)
        return torch.sum(
            torch.sum(
                torch.sqrt(torch.sum((pred_pos_2D - gt_pos_2D) ** 2, dim=-1)) * mask,
                dim=1,
            )
            / torch.sum(mask, dim=1)
        )

    model.eval()

    cam_num = 0
    loss, metric = 0, 0
    metric_trajectory = 0
    metric_2D = 0
    metric_position = 0
    TPs, TNs, FPs, FNs = 0, 0, 0, 0
    number = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(valloader)):
            r_img, court_img, mask, r_world, rotation, times, Mint, Mext, ankle_v = data
            r_img, court_img, mask, rotation, ankle_v = (
                r_img.to(device),
                court_img.to(device),
                mask.to(device),
                rotation.to(device),
                ankle_v.to(device),
            )
            r_world, times = r_world.to(device), times.to(device)
            Mint, Mext = Mint.to(device), Mext.to(device)
            B, T, D = r_img.shape
            pred_rotation, pred_position = model(r_img, court_img, mask, times, ankle_v)

            # remove image and gt world coordinates if detections are not visible and are not interpolated
            r_img, visibilities = r_img[:, :, :2], r_img[:, :, 2]
            if not config.interpolate_missing:
                # Sort such that visible balls are first
                keep_mask = visibilities == BALL_VISIBLE
                sorted_keep_mask, sort_indices = torch.sort(
                    keep_mask, dim=1, descending=True
                )
                # Apply sorting to image coordinates
                B, T, D = r_img.shape
                sort_indices_d = sort_indices.clone().unsqueeze(-1).expand(B, T, D)
                r_img = torch.gather(r_img, dim=1, index=sort_indices_d)
                # Apply sorting to world coordinates
                B, T, D = r_world.shape
                sort_indices_d = sort_indices.clone().unsqueeze(-1).expand(B, T, D)
                r_world = torch.gather(r_world, dim=1, index=sort_indices_d)
                # Apply sorting to ground truth rotations
                B, T, D = rotation.shape
                sort_indices_d = sort_indices.clone().unsqueeze(-1).expand(B, T, D)
                rotation = torch.gather(rotation, dim=1, index=sort_indices_d)
                # Apply sorting to predicted rotations
                B, T, D = pred_rotation.shape
                sort_indices_d = sort_indices.clone().unsqueeze(-1).expand(B, T, D)
                pred_rotation = torch.gather(pred_rotation, dim=1, index=sort_indices_d)
                # Apply sorting to times and mask
                times = torch.gather(times, dim=1, index=sort_indices)
                mask = torch.gather(mask, dim=1, index=sort_indices)
                mask = torch.where(
                    sorted_keep_mask, mask, torch.tensor(0, device=mask.device)
                )

            # All metrics are calculated in the ball's coordinate system -> transform (predicted and) gt rotations accordingly
            rotation = transform_rotationaxes(rotation, r_world, mask)
            if config.transform_mode == "global":
                pred_rotation = transform_rotationaxes(pred_rotation, r_world, mask)

            loss = loss_fn(pred_rotation, rotation)
            metric += metric_fn(pred_rotation, rotation)
            metric_position += metric_pos_fn(pred_position, r_world, mask)
            metric_2D += metric_2D_fn(pred_position, r_world, mask, Mint, Mext)
            tmp = binary_metrics(pred_rotation, rotation)
            TPs, TNs, FPs, FNs = TPs + tmp[0], TNs + tmp[1], FPs + tmp[2], FNs + tmp[3]
            number += B
        loss /= number
        metric /= number
        metric_trajectory /= number
        metric_position /= number
        metric_2D /= number
        normed_metric_2D = metric_2D / (WIDTH**2 + HEIGHT**2) ** 0.5
        accuracy = (TPs + TNs) / (TPs + TNs + FPs + FNs)

    writer.add_scalar(f"val{cam_num}/loss", loss.item(), step)
    writer.add_scalar(f"val{cam_num}/metric", metric.item(), step)
    writer.add_scalar(f"val{cam_num}/accuracy x", accuracy[0].item(), step)
    writer.add_scalar(f"val{cam_num}/accuracy y", accuracy[1].item(), step)
    writer.add_scalar(f"val{cam_num}/accuracy z", accuracy[2].item(), step)
    writer.add_scalar(f"val{cam_num}/metric position", metric_position.item(), step)
    writer.add_scalar(f"val{cam_num}/metric 2D", metric_2D.item(), step)
    writer.add_scalar(f"val{cam_num}/metric 2D normed", normed_metric_2D.item(), step)

    TPs, TNs, FPs, FNs = (
        TPs.cpu().numpy(),
        TNs.cpu().numpy(),
        FPs.cpu().numpy(),
        FNs.cpu().numpy(),
    )
    writer.add_image(
        f"val{cam_num}/confusion matrix x",
        create_confusion_matrix(TPs[0], TNs[0], FPs[0], FNs[0]),
        step,
        dataformats="HWC",
    )
    writer.add_image(
        f"val{cam_num}/confusion matrix y",
        create_confusion_matrix(TPs[1], TNs[1], FPs[1], FNs[1]),
        step,
        dataformats="HWC",
    )
    writer.add_image(
        f"val{cam_num}/confusion matrix z",
        create_confusion_matrix(TPs[2], TNs[2], FPs[2], FNs[2]),
        step,
        dataformats="HWC",
    )
    # Add predicted and gt rotations as text to tensorboard
    writer.add_text(
        f"Predicted Rotations",
        str(np.round(pred_rotation[:4].cpu().numpy(), 1)),
        step,
    )
    writer.add_text(f"GT Rotations", str(np.round(rotation[:4].cpu().numpy(), 1)), step)
    # Add hparams to tensorboard
    writer.add_hparams2(config.get_hparams(), {"metric": metric.item()})

    model.train()

    return metric.item()


def val_real(model, valloader, writer, epoch, device, config, step=None):
    if step is None:
        step = int(
            epoch * 2
        )  # Convert to integer: epoch 0 -> 0, epoch 0.5 -> 1, epoch 1 -> 2, etc.

    def metric_pos2D_fn(pred_2D, gt_2D, mask):
        B, T, __ = pred_2D.shape
        for b in range(B):
            for t in range(T):
                if mask[b, t] == 0:
                    pred_2D[b, t], gt_2D[b, t] = np.array([0, 0]), np.array(
                        [0, 0]
                    )  # set to same value such that difference is 0
        return np.sum(
            np.sum(np.sqrt(np.sum((pred_2D - gt_2D) ** 2, axis=-1)) * mask, axis=1)
            / np.sum(mask, axis=1)
        )

    denorm = UnNormalizeImgCoords()
    model.eval()

    metric_2D = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    scores = []  # needed for ROC-AUC and number of missortings
    labels = []  # needed for ROC-AUC and number of missortings
    number = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(valloader)):
            (
                r_img,
                court_img,
                mask,
                times,
                hits,
                Mint,
                Mext,
                spin_class,
                start_serve,
                ankle_v,
            ) = data
            r_img, court_img, mask, times, start_serve, ankle_v = (
                r_img.to(device),
                court_img.to(device),
                mask.to(device),
                times.to(device),
                start_serve.to(device),
                ankle_v.to(device),
            )
            B, T, D = r_img.shape
            pred_rotation, pred_position = model(r_img, court_img, mask, times, ankle_v)
            # If network output is in ball's coordinate system, transform the ground truth to the local coordinate system
            if config.transform_mode == "global":
                pred_rotation = transform_rotationaxes(
                    pred_rotation, pred_position, mask
                )

            # We can only calculate a metric for the first predicted spin
            pred_rot = []
            for b in range(B):
                serve_idx = int(start_serve[b].item()) + 1
                pred_rot.append(pred_rotation[b, serve_idx, :])  # (b, 3)

            pred_rotation = pred_rot

            # remove image and gt world coordinates if detections are not visible and are not interpolated
            r_img, visibilities = r_img[:, :, :2], r_img[:, :, 2]
            if not config.interpolate_missing:
                # Sort such that visible balls are first
                keep_mask = visibilities == BALL_VISIBLE
                sorted_keep_mask, sort_indices = torch.sort(
                    keep_mask, dim=1, descending=True
                )
                # Apply sorting to image coordinates
                B, T, D = r_img.shape
                sort_indices_d = sort_indices.clone().unsqueeze(-1).expand(B, T, D)
                r_img = torch.gather(r_img, dim=1, index=sort_indices_d)
                # Apply sorting to times and mask
                times = torch.gather(times, dim=1, index=sort_indices)
                mask = torch.gather(mask, dim=1, index=sort_indices)
                mask = torch.where(
                    sorted_keep_mask, mask, torch.tensor(0, device=mask.device)
                )

            # denormalization of ground truth image coordinates to calculate the 2D metric
            data_gt = denorm(
                {
                    "r_img": r_img.cpu().numpy(),
                    "court_img": court_img.cpu().numpy(),
                    "ankle_v": ankle_v.cpu().numpy(),
                }
            )
            r_img, court_img, ankle_v = (
                data_gt["r_img"],
                data_gt["court_img"],
                data_gt["ankle_v"],
            )

            # calculate metrics
            pred_pos_2D = cam2img(
                world2cam(pred_position, Mext.to(device)), Mint.to(device)
            )
            m = metric_pos2D_fn(pred_pos_2D.cpu().numpy(), r_img, mask.cpu().numpy())
            metric_2D += m

            # binary metrics: Front- vs Backspin ; ROC-AUC ; Number of missortings
            for b in range(B):
                # binary metrics
                if spin_class[b] == TOPSPIN_CLASS:  # Frontspin
                    if pred_rotation[b][1] > 0:
                        TP += 1
                    else:
                        FN += 1
                elif spin_class[b] == BACKSPIN_CLASS:  # Backspin
                    if pred_rotation[b][1] < 0:
                        TN += 1
                    else:
                        FP += 1
                # ROC-AUC and missortings
                if spin_class[b] in [
                    BACKSPIN_CLASS,
                    TOPSPIN_CLASS,
                ]:  # only consider if spin class is annotated
                    scores.append(pred_rotation[b][1].item())
                    labels.append(
                        1 if spin_class[b] == 1 else 0
                    )  # The methods use Frontspin=1 and Backspin=0

            number += B

        metric_2D /= number
        normed_metric_2D = (
            metric_2D / (WIDTH**2 + HEIGHT**2) ** 0.5
        )  # normalize by image size
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1_plus = 2 * TP / (2 * TP + FP + FN)
        f1_minus = 2 * TN / (2 * TN + FN + FP)
        macro_f1 = (f1_plus + f1_minus) / 2
        roc_auc = roc_auc_score(labels, scores)

    writer.add_scalar("val real/metric 2D", metric_2D, step)
    writer.add_scalar("val real/metric 2D normed", normed_metric_2D, step)
    writer.add_scalar("val real/accuracy", accuracy, step)
    writer.add_scalar("val real/macro f1", macro_f1, step)
    writer.add_scalar("val real/ROC AUC", roc_auc, step)

    return normed_metric_2D, macro_f1


def val_real_rallies(model, valloader, writer, epoch, device, config, step=None):
    if step is None:
        step = int(
            epoch * 2
        )  # Convert to integer: epoch 0 -> 0, epoch 0.5 -> 1, epoch 1 -> 2, etc.

    def metric_pos2D_fn(pred_2D, gt_2D, mask):
        B, T, __ = pred_2D.shape
        for b in range(B):
            for t in range(T):
                if mask[b, t] == 0:
                    pred_2D[b, t], gt_2D[b, t] = np.array([0, 0]), np.array(
                        [0, 0]
                    )  # set to same value such that difference is 0
        return np.sum(
            np.sum(np.sqrt(np.sum((pred_2D - gt_2D) ** 2, axis=-1)) * mask, axis=1)
            / np.sum(mask, axis=1)
        )

    denorm = UnNormalizeImgCoords()
    model.eval()

    metric_2D = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    scores = []  # needed for ROC-AUC and number of missortings
    labels = []  # needed for ROC-AUC and number of missortings
    number = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(valloader)):
            (
                r_img,
                court_img,
                mask,
                times,
                Mint,
                Mext,
                spin_class_per_shot,
                spin_class_per_frame,
                new_trajectory_frame_idx,
                ankle_v,
            ) = data
            (
                r_img,
                court_img,
                mask,
                times,
                new_trajectory_frame_idx,
                ankle_v,
            ) = (
                r_img.to(device),
                court_img.to(device),
                mask.to(device),
                times.to(device),
                new_trajectory_frame_idx.to(device),
                ankle_v.to(device),
            )
            B, T, D = r_img.shape
            pred_rotation, pred_position = model(r_img, court_img, mask, times, ankle_v)
            # If network output is in ball's coordinate system, transform the ground truth to the local coordinate system
            if config.transform_mode == "global":
                pred_rotation = transform_rotationaxes(
                    pred_rotation, pred_position, mask
                )

            # We can only calculate a metric for the first predicted spin
            pred_rotation = pred_rotation[:, :, 1]  # (B, T)

            # remove image and gt world coordinates if detections are not visible and are not interpolated
            r_img, visibilities = r_img[:, :, :2], r_img[:, :, 2]
            if not config.interpolate_missing:
                # Sort such that visible balls are first
                keep_mask = visibilities == BALL_VISIBLE
                sorted_keep_mask, sort_indices = torch.sort(
                    keep_mask, dim=1, descending=True
                )
                # Apply sorting to image coordinates
                B, T, D = r_img.shape
                sort_indices_d = sort_indices.clone().unsqueeze(-1).expand(B, T, D)
                r_img = torch.gather(r_img, dim=1, index=sort_indices_d)
                # Apply sorting to times and mask
                times = torch.gather(times, dim=1, index=sort_indices)
                mask = torch.gather(mask, dim=1, index=sort_indices)
                mask = torch.where(
                    sorted_keep_mask, mask, torch.tensor(0, device=mask.device)
                )

            # denormalization of ground truth image coordinates to calculate the 2D metric
            data_gt = denorm(
                {
                    "r_img": r_img.cpu().numpy(),
                    "court_img": court_img.cpu(),
                    "ankle_v": ankle_v.cpu().numpy(),
                }
            )
            r_img, court_img, ankle_v = (
                data_gt["r_img"],
                data_gt["court_img"],
                data_gt["ankle_v"],
            )

            # calculate metrics
            pred_pos_2D = cam2img(
                world2cam(pred_position, Mext.to(device)), Mint.to(device)
            )
            m = metric_pos2D_fn(pred_pos_2D.cpu().numpy(), r_img, mask.cpu().numpy())
            metric_2D += m

            # binary metrics: Front- vs Backspin ; ROC-AUC ; Number of missortings
            for b in range(B):
                pred_rot_rally = pred_rotation[b]
                spin_cls_for_rally = spin_class_per_shot[b]
                num_shots = torch.where(new_trajectory_frame_idx[b] == -1)[0][
                    0
                ].item()  # find first occurrence of -1 which indicates end of shots
                for idx, frame in enumerate(new_trajectory_frame_idx[b, :num_shots]):
                    # binary metrics
                    if spin_cls_for_rally[idx] == TOPSPIN_CLASS:  # Frontspin
                        if pred_rot_rally[int(frame)] > 0:
                            TP += 1
                        else:
                            FN += 1
                    elif spin_cls_for_rally[idx] == BACKSPIN_CLASS:  # Backspin
                        if pred_rot_rally[int(frame)] < 0:
                            TN += 1
                        else:
                            FP += 1
                    # ROC-AUC and missortings
                    if spin_cls_for_rally[idx] in [
                        BACKSPIN_CLASS,
                        TOPSPIN_CLASS,
                    ]:  # only consider if spin class is annotated
                        scores.append(pred_rot_rally[int(frame)].item())
                        labels.append(
                            1 if spin_cls_for_rally[idx] == TOPSPIN_CLASS else 0
                        )  # The methods use Frontspin=1 and Backspin=0

            number += B

        metric_2D /= number
        normed_metric_2D = (
            metric_2D / (WIDTH**2 + HEIGHT**2) ** 0.5
        )  # normalize by image size
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1_plus = 2 * TP / (2 * TP + FP + FN)
        f1_minus = 2 * TN / (2 * TN + FN + FP)
        macro_f1 = (f1_plus + f1_minus) / 2
        roc_auc = roc_auc_score(labels, scores)

    writer.add_scalar("val real rallies/metric 2D", metric_2D, step)
    writer.add_scalar("val real rallies/metric 2D normed", normed_metric_2D, step)
    writer.add_scalar("val real rallies/accuracy", accuracy, step)
    writer.add_scalar("val real rallies/macro f1", macro_f1, step)
    writer.add_scalar("val real rallies/ROC AUC", roc_auc, step)

    return normed_metric_2D, macro_f1


def main():
    global debug
    debug = args.debug
    config = TrainConfig(
        args.lr, args.model_name, args.model_size, debug, args.folder, args.exp_id
    )
    config.blur_strength = args.blur_strength
    config.randomize_std = args.randomize_std
    config.randmiss_prob = args.randmiss_prob
    config.tabletoken_mode = args.token_mode
    config.transform_mode = args.transform_mode
    config.time_rotation = args.time_rotation
    config.interpolate_missing = args.interpolate_missing
    run(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
