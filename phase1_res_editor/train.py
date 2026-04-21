# train.py
import os
import csv
import math
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model
from detector_wrapper import FrozenDetectorScorer
from generator import ResidualUNet
from losses import phase1_loss


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARCH_ALIASES = {
    "res18": "Imagenet:resnet18",
    "res34": "Imagenet:resnet34",
    "res50": "Imagenet:resnet50",
    "res101": "Imagenet:resnet101",
    "res152": "Imagenet:resnet152",
    "resnet18": "Imagenet:resnet18",
    "resnet34": "Imagenet:resnet34",
    "resnet50": "Imagenet:resnet50",
    "resnet101": "Imagenet:resnet101",
    "resnet152": "Imagenet:resnet152",
    "vit_b_16": "Imagenet:vit_b_16",
    "vit_b_32": "Imagenet:vit_b_32",
    "vit_l_16": "Imagenet:vit_l_16",
    "vit_l_32": "Imagenet:vit_l_32",
    "clip_rn50": "CLIP:RN50",
    "clip_rn101": "CLIP:RN101",
    "clip_vitb16": "CLIP:ViT-B/16",
    "clip_vitb32": "CLIP:ViT-B/32",
    "clip_vitl14": "CLIP:ViT-L/14",
    "clip_vitl14_336": "CLIP:ViT-L/14@336px",
}


class SimpleImageFolder(Dataset):
    def __init__(self, root: str, image_size: int = 256):
        self.root = Path(root)
        self.paths = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTS])
        if not self.paths:
            raise RuntimeError(f"No images found in: {root}")

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),   # => [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        return x, path.name


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_detector_arch(arch: str) -> str:
    arch = arch.strip()
    return ARCH_ALIASES.get(arch, arch)


def strip_state_dict_prefix(state_dict, prefixes=("module.", "model.")):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def build_detector_from_ufd(args, device: torch.device) -> FrozenDetectorScorer:
    detector_arch = normalize_detector_arch(args.detector_arch)
    obj = torch.load(args.detector_ckpt, map_location="cpu")

    if isinstance(obj, nn.Module):
        raw_detector = obj
    else:
        raw_detector = get_model(detector_arch)

        if not isinstance(obj, dict):
            raise RuntimeError(
                f"Unsupported checkpoint type: {type(obj)}. "
                "Expected either an nn.Module or a state_dict-like dict."
            )

        load_ok = False
        if "model" in obj and isinstance(obj["model"], nn.Module):
            raw_detector = obj["model"]
            load_ok = True
        elif "model" in obj and isinstance(obj["model"], dict):
            raw_detector.load_state_dict(strip_state_dict_prefix(obj["model"]), strict=True)
            load_ok = True
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            raw_detector.load_state_dict(strip_state_dict_prefix(obj["state_dict"]), strict=True)
            load_ok = True
        elif {"weight", "bias"}.issubset(obj.keys()):
            raw_detector.fc.load_state_dict(obj, strict=True)
            load_ok = True
        elif all(isinstance(k, str) for k in obj.keys()):
            cleaned_state = strip_state_dict_prefix(obj)
            if any(k.startswith("fc.") for k in cleaned_state):
                raw_detector.load_state_dict(cleaned_state, strict=True)
            else:
                raw_detector.fc.load_state_dict(cleaned_state, strict=True)
            load_ok = True

        if not load_ok:
            raise RuntimeError(
                "Could not infer how to load the detector checkpoint. "
                "Supported formats: full nn.Module, {'model': state_dict}, "
                "{'state_dict': state_dict}, full model state_dict, or fc-only state_dict."
            )

    raw_detector = raw_detector.to(device).eval()

    scorer = FrozenDetectorScorer(
        raw_detector=raw_detector,
        input_size=args.detector_input_size,
        fake_index=args.fake_index,
    ).to(device)
    return scorer


@torch.no_grad()
def evaluate(
    generator: ResidualUNet,
    detector: FrozenDetectorScorer,
    loader: DataLoader,
    device: torch.device,
    eps: float,
    out_csv: str,
):
    generator.eval()
    rows: List[Tuple[str, float, float, float]] = []

    score_before_sum = 0.0
    score_after_sum = 0.0
    l1_sum = 0.0
    n = 0

    for x, names in loader:
        x = x.to(device)

        score_before = detector(x)

        raw_r = generator(x)
        x_adv, bounded_r = generator.apply_residual(x, raw_r, eps)
        score_after = detector(x_adv)

        l1 = torch.abs(x_adv - x).mean(dim=(1, 2, 3))

        bs = x.size(0)
        score_before_sum += score_before.sum().item()
        score_after_sum += score_after.sum().item()
        l1_sum += l1.sum().item()
        n += bs

        for i in range(bs):
            rows.append((
                names[i],
                float(score_before[i].item()),
                float(score_after[i].item()),
                float(l1[i].item()),
            ))

    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "score_before", "score_after", "l1_diff"])
        writer.writerows(rows)

    stats = {
        "mean_score_before": score_before_sum / max(n, 1),
        "mean_score_after": score_after_sum / max(n, 1),
        "mean_l1": l1_sum / max(n, 1),
        "num_samples": n,
    }
    return stats


def save_visual_batch(
    x: torch.Tensor,
    x_adv: torch.Tensor,
    bounded_r: torch.Tensor,
    save_path: str,
    max_items: int = 4,
    eps: float = 8 / 255.0,
):
    """
    Save [original | edited | residual_vis] grid
    """
    ensure_dir(os.path.dirname(save_path))
    k = min(max_items, x.size(0))

    # map residual from [-eps, eps] -> [0,1]
    r_vis = torch.clamp((bounded_r[:k] / (2 * eps)) + 0.5, 0.0, 1.0)

    grid = torch.cat([x[:k].cpu(), x_adv[:k].cpu(), r_vis.cpu()], dim=0)
    save_image(grid, save_path, nrow=k)


def train_one_epoch(
    generator: ResidualUNet,
    detector: FrozenDetectorScorer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
):
    generator.train()

    log = {
        "loss_total": 0.0,
        "loss_rec": 0.0,
        "loss_det": 0.0,
        "loss_tv": 0.0,
        "loss_res": 0.0,
        "score_before": 0.0,
        "score_after": 0.0,
        "num_batches": 0,
    }

    last_batch = None

    for x, names in loader:
        x = x.to(device)

        with torch.no_grad():
            score_before = detector(x)

        raw_r = generator(x)
        x_adv, bounded_r = generator.apply_residual(x, raw_r, args.eps)

        score_after = detector(x_adv)

        loss, stats = phase1_loss(
            x=x,
            x_adv=x_adv,
            bounded_r=bounded_r,
            score_after=score_after,
            lambda_rec=args.lambda_rec,
            lambda_det=args.lambda_det,
            lambda_tv=args.lambda_tv,
            lambda_res=args.lambda_res,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        log["loss_total"] += stats["loss_total"].item()
        log["loss_rec"] += stats["loss_rec"].item()
        log["loss_det"] += stats["loss_det"].item()
        log["loss_tv"] += stats["loss_tv"].item()
        log["loss_res"] += stats["loss_res"].item()
        log["score_before"] += score_before.mean().item()
        log["score_after"] += score_after.mean().item()
        log["num_batches"] += 1

        last_batch = (x.detach(), x_adv.detach(), bounded_r.detach())

    nb = max(log["num_batches"], 1)
    for k in list(log.keys()):
        if k != "num_batches":
            log[k] /= nb
    return log, last_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs")

    parser.add_argument("--detector_ckpt", type=str, required=True)
    parser.add_argument("--detector_arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--detector_input_size", type=int, default=224)
    parser.add_argument("--fake_index", type=int, default=1)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_ch", type=int, default=32)

    parser.add_argument("--eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_det", type=float, default=0.5)
    parser.add_argument("--lambda_tv", type=float, default=0.01)
    parser.add_argument("--lambda_res", type=float, default=0.01)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "checkpoints"))
    ensure_dir(os.path.join(args.out_dir, "samples"))
    ensure_dir(os.path.join(args.out_dir, "scores"))

    train_ds = SimpleImageFolder(args.train_dir, image_size=args.image_size)
    val_ds = SimpleImageFolder(args.val_dir, image_size=args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    detector = build_detector_from_ufd(args, device)
    generator = ResidualUNet(in_ch=3, base_ch=args.base_ch).to(device)

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    best_metric = math.inf

    for epoch in range(1, args.epochs + 1):
        train_log, last_batch = train_one_epoch(
            generator=generator,
            detector=detector,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
        )

        val_csv = os.path.join(args.out_dir, "scores", f"val_epoch_{epoch:03d}.csv")
        val_stats = evaluate(
            generator=generator,
            detector=detector,
            loader=val_loader,
            device=device,
            eps=args.eps,
            out_csv=val_csv,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_total={train_log['loss_total']:.4f} "
            f"train_det={train_log['loss_det']:.4f} "
            f"train_before={train_log['score_before']:.4f} "
            f"train_after={train_log['score_after']:.4f} | "
            f"val_before={val_stats['mean_score_before']:.4f} "
            f"val_after={val_stats['mean_score_after']:.4f} "
            f"val_l1={val_stats['mean_l1']:.6f}"
        )

        if last_batch is not None:
            x, x_adv, bounded_r = last_batch
            sample_path = os.path.join(args.out_dir, "samples", f"epoch_{epoch:03d}.png")
            save_visual_batch(x, x_adv, bounded_r, sample_path, eps=args.eps)

        ckpt = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "val_stats": val_stats,
        }
        torch.save(ckpt, os.path.join(args.out_dir, "checkpoints", f"epoch_{epoch:03d}.pt"))

        # choose lower mean_score_after as better
        metric = val_stats["mean_score_after"]
        if metric < best_metric:
            best_metric = metric
            torch.save(
                ckpt,
                os.path.join(args.out_dir, "checkpoints", "best.pt")
            )


if __name__ == "__main__":
    main()
