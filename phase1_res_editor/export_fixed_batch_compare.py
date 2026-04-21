import argparse
import csv
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torchvision.utils import save_image

from generator import ResidualUNet
from train import SimpleImageFolder, build_detector_from_ufd, ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def checkpoint_sort_key(path: Path) -> Tuple[int, int, str]:
    stem = path.stem
    if stem.startswith("epoch_"):
        try:
            return (0, int(stem.split("_")[-1]), stem)
        except ValueError:
            pass
    if stem == "best":
        return (1, 10**9, stem)
    return (2, 10**9, stem)


def load_torch_checkpoint(path: Path):
    return torch.load(path, map_location="cpu")


def load_checkpoint_args(path: Path) -> Dict:
    obj = load_torch_checkpoint(path)
    if isinstance(obj, dict) and isinstance(obj.get("args"), dict):
        return obj["args"]
    return {}


def choose_value(cli_value, ckpt_args: Dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(key, default)


def resolve_project_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def collect_checkpoints(
    checkpoints_dir: Path,
    checkpoint_glob: str,
    explicit_paths: Sequence[str],
    include_best: bool,
) -> List[Path]:
    if explicit_paths:
        paths = [Path(p) for p in explicit_paths]
    else:
        paths = sorted(checkpoints_dir.glob(checkpoint_glob), key=checkpoint_sort_key)
        if include_best and (checkpoints_dir / "best.pt").exists():
            paths.append(checkpoints_dir / "best.pt")

    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        deduped.append(path)
        seen.add(resolved)

    if not deduped:
        raise RuntimeError("No checkpoints found for comparison.")
    return deduped


def pick_fixed_indices(dataset_len: int, num_samples: int, seed: int, mode: str) -> List[int]:
    count = min(dataset_len, num_samples)
    if mode == "first":
        return list(range(count))

    rng = random.Random(seed)
    indices = rng.sample(range(dataset_len), count)
    return sorted(indices)


def build_fixed_batch(
    dataset: SimpleImageFolder,
    indices: Iterable[int],
) -> Tuple[torch.Tensor, List[str], List[str]]:
    images = []
    names = []
    paths = []
    for idx in indices:
        x, name = dataset[idx]
        images.append(x)
        names.append(name)
        paths.append(str(dataset.paths[idx]))
    return torch.stack(images, dim=0), names, paths


def resolve_generator_state(ckpt_obj) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("generator"), dict):
        return ckpt_obj["generator"]
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("state_dict"), dict):
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict) and all(isinstance(k, str) for k in ckpt_obj.keys()):
        return ckpt_obj
    raise RuntimeError("Unsupported generator checkpoint format.")


def save_triplet_grid(
    x: torch.Tensor,
    x_adv: torch.Tensor,
    bounded_r: torch.Tensor,
    eps: float,
    save_path: Path,
):
    ensure_dir(str(save_path.parent))
    residual_vis = torch.clamp((bounded_r / (2 * eps)) + 0.5, 0.0, 1.0)
    grid = torch.cat([x.cpu(), x_adv.cpu(), residual_vis.cpu()], dim=0)
    save_image(grid, str(save_path), nrow=x.size(0))


def write_manifest(
    save_path: Path,
    names: Sequence[str],
    paths: Sequence[str],
    score_before: torch.Tensor,
):
    ensure_dir(str(save_path.parent))
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image_name", "source_path", "score_before"])
        for idx, (name, src, score) in enumerate(zip(names, paths, score_before.tolist())):
            writer.writerow([idx, name, src, float(score)])


def write_per_checkpoint_csv(
    save_path: Path,
    names: Sequence[str],
    score_before: torch.Tensor,
    score_after: torch.Tensor,
    l1: torch.Tensor,
):
    ensure_dir(str(save_path.parent))
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "score_before", "score_after", "delta", "l1_diff"])
        for name, before, after, l1_value in zip(
            names,
            score_before.tolist(),
            score_after.tolist(),
            l1.tolist(),
        ):
            writer.writerow([
                name,
                float(before),
                float(after),
                float(after - before),
                float(l1_value),
            ])


def append_summary_row(
    rows: List[Dict],
    checkpoint_name: str,
    score_before: torch.Tensor,
    score_after: torch.Tensor,
    l1: torch.Tensor,
):
    delta = score_after - score_before
    rows.append({
        "checkpoint": checkpoint_name,
        "mean_score_before": float(score_before.mean().item()),
        "mean_score_after": float(score_after.mean().item()),
        "mean_delta": float(delta.mean().item()),
        "mean_l1": float(l1.mean().item()),
        "best_delta": float(delta.min().item()),
        "worst_delta": float(delta.max().item()),
    })


def write_summary(save_path: Path, rows: Sequence[Dict]):
    ensure_dir(str(save_path.parent))
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "mean_score_before",
                "mean_score_after",
                "mean_delta",
                "mean_l1",
                "best_delta",
                "worst_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--checkpoint_glob", type=str, default="epoch_*.pt")
    parser.add_argument("--checkpoint_paths", nargs="*", default=None)
    parser.add_argument("--include_best", action="store_true")

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--sample_mode", choices=("random", "first"), default="random")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--detector_ckpt", type=str, default=None)
    parser.add_argument("--detector_arch", type=str, default=None)
    parser.add_argument("--detector_input_size", type=int, default=None)
    parser.add_argument("--fake_index", type=int, default=None)

    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--base_ch", type=int, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    default_checkpoints_dir = PROJECT_ROOT / "outputs_phase1" / "checkpoints"
    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else default_checkpoints_dir
    checkpoint_paths = collect_checkpoints(
        checkpoints_dir=checkpoints_dir,
        checkpoint_glob=args.checkpoint_glob,
        explicit_paths=args.checkpoint_paths or [],
        include_best=args.include_best,
    )

    first_ckpt_args = load_checkpoint_args(checkpoint_paths[0])
    image_size = choose_value(args.image_size, first_ckpt_args, "image_size", 256)
    base_ch = choose_value(args.base_ch, first_ckpt_args, "base_ch", 32)
    eps = choose_value(args.eps, first_ckpt_args, "eps", 8.0 / 255.0)
    detector_ckpt = choose_value(args.detector_ckpt, first_ckpt_args, "detector_ckpt", None)
    detector_arch = choose_value(args.detector_arch, first_ckpt_args, "detector_arch", "CLIP:ViT-L/14")
    detector_input_size = choose_value(args.detector_input_size, first_ckpt_args, "detector_input_size", 224)
    fake_index = choose_value(args.fake_index, first_ckpt_args, "fake_index", 1)

    if detector_ckpt is None:
        raise RuntimeError("detector_ckpt is required. Pass it explicitly or use checkpoints that store args.")
    detector_ckpt = resolve_project_path(detector_ckpt)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    detector_args = SimpleNamespace(
        detector_ckpt=detector_ckpt,
        detector_arch=detector_arch,
        detector_input_size=detector_input_size,
        fake_index=fake_index,
    )
    detector = build_detector_from_ufd(detector_args, device)

    dataset = SimpleImageFolder(args.data_dir, image_size=image_size)
    indices = pick_fixed_indices(len(dataset), args.num_samples, args.seed, args.sample_mode)
    x_cpu, names, source_paths = build_fixed_batch(dataset, indices)
    x = x_cpu.to(device)

    with torch.no_grad():
        score_before = detector(x).detach().cpu()

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    write_manifest(out_dir / "fixed_batch_manifest.csv", names, source_paths, score_before)

    summary_rows: List[Dict] = []
    for ckpt_path in checkpoint_paths:
        ckpt_obj = load_torch_checkpoint(ckpt_path)
        ckpt_args = ckpt_obj.get("args", {}) if isinstance(ckpt_obj, dict) else {}
        generator_base_ch = choose_value(args.base_ch, ckpt_args, "base_ch", base_ch)
        generator_eps = choose_value(args.eps, ckpt_args, "eps", eps)

        generator = ResidualUNet(in_ch=3, base_ch=generator_base_ch).to(device)
        generator.load_state_dict(resolve_generator_state(ckpt_obj), strict=True)
        generator.eval()

        with torch.no_grad():
            raw_r = generator(x)
            x_adv, bounded_r = generator.apply_residual(x, raw_r, generator_eps)
            score_after = detector(x_adv).detach().cpu()
            bounded_r_cpu = bounded_r.detach().cpu()
            x_adv_cpu = x_adv.detach().cpu()
            l1 = (x_adv_cpu - x_cpu).abs().mean(dim=(1, 2, 3))

        ckpt_name = ckpt_path.stem
        save_triplet_grid(
            x=x_cpu,
            x_adv=x_adv_cpu,
            bounded_r=bounded_r_cpu,
            eps=generator_eps,
            save_path=out_dir / "grids" / f"{ckpt_name}.png",
        )
        write_per_checkpoint_csv(
            save_path=out_dir / "scores" / f"{ckpt_name}.csv",
            names=names,
            score_before=score_before,
            score_after=score_after,
            l1=l1,
        )
        append_summary_row(
            rows=summary_rows,
            checkpoint_name=ckpt_name,
            score_before=score_before,
            score_after=score_after,
            l1=l1,
        )

    write_summary(out_dir / "summary.csv", summary_rows)
    print(f"Saved fixed-batch comparison to: {out_dir}")


if __name__ == "__main__":
    main()
