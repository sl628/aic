"""Train ResNet18 late-fusion phase classifier on labeled aic episodes.

Handles class imbalance via:
  - Class-weighted cross-entropy loss
  - Optional subsampling of majority classes (P3)
  - Per-class precision/recall/F1 + weighted/macro F1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T

from aic_xvla.phase_classifier_model import PhaseClassifierNet, PHASES, IMAGE_MEAN, IMAGE_STD

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
log = logging.getLogger("train_phase_classifier")

CAMERA_KEYS = ["left_camera", "center_camera", "right_camera"]

_TRAIN_TF = T.Compose([
    T.Resize((224, 224)),
    T.RandomRotation(5),
    T.ColorJitter(brightness=0.05, contrast=0.05),
    T.ToTensor(),
    T.Normalize(IMAGE_MEAN, IMAGE_STD),
])
_VAL_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGE_MEAN, IMAGE_STD),
])


class PhaseDataset(Dataset):
    def __init__(self, episode_ids: list[str], labels: dict[str, list[int]],
                 data_root: str, train: bool = True,
                 max_samples: int | None = None,
                 subsample_p3: bool = False,
                 target_p3_ratio: float = 0.3):
        self.data_root = Path(data_root)
        self.tf = _TRAIN_TF if train else _VAL_TF
        self.samples: list[tuple[str, int, int]] = []

        for ep_id in episode_ids:
            ep_labels = labels.get(ep_id)
            if ep_labels is None:
                continue
            for t, lbl in enumerate(ep_labels):
                self.samples.append((ep_id, t, lbl))

        # Subsample majority classes (e.g. P3/insert).
        if subsample_p3 and train:
            counts = np.zeros(4, dtype=np.int64)
            for _, _, lbl in self.samples:
                counts[lbl] += 1
            # Target: keep all samples of minority classes, cap majority.
            max_per_class = int(counts.max() * target_p3_ratio)
            filtered = []
            per_class_count = np.zeros(4, dtype=np.int64)
            for s in self.samples:
                lbl = s[2]
                if lbl != 3 or per_class_count[3] < max_per_class:
                    filtered.append(s)
                    per_class_count[lbl] += 1
            self.samples = filtered

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ep_id, frame_idx, label = self.samples[idx]
        views = []
        for cam in CAMERA_KEYS:
            path = (self.data_root / "episodes" / ep_id / "images" / cam /
                    f"frame_{frame_idx:04d}.jpg")
            pil = Image.open(path).convert("RGB")
            views.append(self.tf(pil))
        return torch.stack(views, dim=0), label


def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs, dim=0), torch.tensor(labels, dtype=torch.long)


def compute_class_weights(labels_dict: dict[str, list[int]],
                          episode_ids: list[str]) -> torch.Tensor:
    """Inverse-frequency class weights."""
    counts = np.zeros(4, dtype=np.float64)
    for ep_id in episode_ids:
        ep_lbl = labels_dict.get(ep_id, [])
        for l in ep_lbl:
            counts[l] += 1
    total = counts.sum()
    weights = total / (4 * counts + 1e-8)
    return torch.from_numpy(weights).float()


def compute_metrics(gt: np.ndarray, pred: np.ndarray, n_classes: int = 4) -> dict:
    """Per-class precision, recall, F1 + macro/weighted F1."""
    metrics = {}
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(gt, pred):
        conf[t, p] += 1

    metrics["confusion_matrix"] = conf.tolist()
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c in range(n_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        support = conf[c, :].sum()

        prec = tp / max(tp + fp, 1e-8)
        rec = tp / max(tp + fn, 1e-8)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        metrics[f"P{c}_precision"] = prec
        metrics[f"P{c}_recall"] = rec
        metrics[f"P{c}_f1"] = f1
        metrics[f"P{c}_support"] = int(support)

    total_support = sum(supports) or 1
    metrics["macro_f1"] = np.mean(f1s)
    metrics["weighted_f1"] = sum(f * s for f, s in zip(f1s, supports)) / total_support
    metrics["accuracy"] = conf.trace() / max(conf.sum(), 1)
    return metrics


def maybe_init_wandb(args) -> None:
    if not args.wandb_project:
        return
    import wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/yifeng/aic_xvla_data")
    p.add_argument("--labels", default="/home/yifeng/aic_xvla_data/episode_labels.json")
    p.add_argument("--train-meta", default="/home/yifeng/aic_xvla_data/train_meta.json")
    p.add_argument("--val-meta", default="/home/yifeng/aic_xvla_data/val_meta.json")
    p.add_argument("--out", default="/home/yifeng/aic_xvla_data/phase_classifier_v2.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=None,
                   help="limit training samples for quick testing")
    p.add_argument("--log-interval", type=int, default=500,
                   help="log progress every N batches")
    p.add_argument("--subsample-p3", action="store_true",
                   help="subsample majority class (P3) for balanced training")
    p.add_argument("--target-p3-ratio", type=float, default=0.3,
                   help="target ratio of P3 relative to largest minority class")
    p.add_argument("--class-weights", action="store_true", default=True,
                   help="use inverse-frequency class weights in loss")
    p.add_argument("--wandb-project", default=None, help="W&B project name")
    p.add_argument("--wandb-entity", default=None, help="W&B entity (user/team)")
    p.add_argument("--wandb-run-name", default=None, help="W&B run name")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    with open(args.labels) as f:
        labels = json.load(f)
    log.info("Loaded labels for %d episodes", len(labels))

    with open(args.train_meta) as f:
        train_meta = json.load(f)
    with open(args.val_meta) as f:
        val_meta = json.load(f)

    train_ep_ids = [Path(e["parquet_path"]).parent.name for e in train_meta["datalist"]]
    val_ep_ids = [Path(e["parquet_path"]).parent.name for e in val_meta["datalist"]]

    train_ds = PhaseDataset(train_ep_ids, labels, args.data_root, train=True,
                            max_samples=args.max_samples,
                            subsample_p3=args.subsample_p3,
                            target_p3_ratio=args.target_p3_ratio)
    val_ds = PhaseDataset(val_ep_ids, labels, args.data_root, train=False,
                          max_samples=args.max_samples)
    log.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    if len(train_ds) == 0:
        log.error("No training samples!")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True)

    model = PhaseClassifierNet().to(device)

    # Class-weighted loss.
    if args.class_weights:
        weight = compute_class_weights(labels, train_ep_ids).to(device)
        log.info("Class weights: %s", np.round(weight.cpu().numpy(), 3).tolist())
    else:
        weight = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    maybe_init_wandb(args)
    log.info("Starting training...")
    sys.stdout.flush()
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()
        num_batches = len(train_loader)

        for batch_idx, (images, gt_labels) in enumerate(train_loader):
            images, gt_labels = images.to(device), gt_labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, gt_labels, weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = logits.max(1)
            train_loss += loss.item() * images.size(0)
            train_correct += preds.eq(gt_labels).sum().item()
            train_total += images.size(0)

            if batch_idx % args.log_interval == 0:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / max(elapsed, 1e-6)
                live_acc = train_correct / max(train_total, 1)
                log.info(
                    f"  batch {batch_idx:5d}/{num_batches}  "
                    f"loss={train_loss/train_total:.4f}  acc={live_acc:.4f}  "
                    f"{rate:.1f} batch/s  {elapsed:.0f}s"
                )
                sys.stdout.flush()

        train_acc = train_correct / train_total
        train_loss_avg = train_loss / train_total

        # Validation with full metrics.
        model.eval()
        val_loss_total = 0.0
        all_gt, all_pred = [], []

        with torch.no_grad():
            for images, gt_labels in val_loader:
                images, gt_labels = images.to(device), gt_labels.to(device)
                logits = model(images)
                loss = F.cross_entropy(logits, gt_labels)
                _, preds = logits.max(1)
                val_loss_total += loss.item() * images.size(0)
                all_gt.extend(gt_labels.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())

        val_metrics = compute_metrics(np.array(all_gt), np.array(all_pred))
        val_loss_avg = val_loss_total / max(len(all_gt), 1)
        scheduler.step()

        log.info(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train_loss={train_loss_avg:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss_avg:.4f}  acc={val_metrics['accuracy']:.4f}  "
            f"macro_f1={val_metrics['macro_f1']:.4f}  w_f1={val_metrics['weighted_f1']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"{(time.time()-t0)/60:.1f}min"
        )
        log.info(
            f"  P0: prec={val_metrics['P0_precision']:.3f} rec={val_metrics['P0_recall']:.3f} "
            f"f1={val_metrics['P0_f1']:.3f}  "
            f"P1: prec={val_metrics['P1_precision']:.3f} rec={val_metrics['P1_recall']:.3f} "
            f"f1={val_metrics['P1_f1']:.3f}  "
            f"P2: prec={val_metrics['P2_precision']:.3f} rec={val_metrics['P2_recall']:.3f} "
            f"f1={val_metrics['P2_f1']:.3f}  "
            f"P3: prec={val_metrics['P3_precision']:.3f} rec={val_metrics['P3_recall']:.3f} "
            f"f1={val_metrics['P3_f1']:.3f}"
        )
        sys.stdout.flush()

        if args.wandb_project:
            import wandb
            wandb.log({
                "train/loss": train_loss_avg,
                "train/acc": train_acc,
                "val/loss": val_loss_avg,
                "val/acc": val_metrics["accuracy"],
                "val/macro_f1": val_metrics["macro_f1"],
                "val/weighted_f1": val_metrics["weighted_f1"],
                "val/precision_p0": val_metrics["P0_precision"],
                "val/precision_p1": val_metrics["P1_precision"],
                "val/precision_p2": val_metrics["P2_precision"],
                "val/precision_p3": val_metrics["P3_precision"],
                "val/recall_p0": val_metrics["P0_recall"],
                "val/recall_p1": val_metrics["P1_recall"],
                "val/recall_p2": val_metrics["P2_recall"],
                "val/recall_p3": val_metrics["P3_recall"],
                "val/f1_p0": val_metrics["P0_f1"],
                "val/f1_p1": val_metrics["P1_f1"],
                "val/f1_p2": val_metrics["P2_f1"],
                "val/f1_p3": val_metrics["P3_f1"],
                "lr": scheduler.get_last_lr()[0],
            })

        # Save on best macro F1.
        if val_metrics["weighted_f1"] > best_f1:
            best_f1 = val_metrics["weighted_f1"]
            torch.save(model.state_dict(), args.out)
            log.info("  → saved (weighted_f1=%.4f)", best_f1)
            sys.stdout.flush()

    log.info("Done. Best weighted F1: %.4f", best_f1)
    log.info("Saved to: %s", args.out)


if __name__ == "__main__":
    main()
