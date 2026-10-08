import argparse
import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .resnet_bs import load_resnet_bs


def prepare_qureai(path: Path, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
    arr = arr.astype(np.float32)
    x = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return (x - x.min()) / (x.max() - x.min() + 1e-10), arr.shape


def prepare_rajaraman(path: Path, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as image:
        shape = (image.height, image.width)
        resized = image.convert("L").resize((size, size), Image.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0, shape


PREPARE = {"qureai": prepare_qureai, "rajaraman_resnetbs": prepare_rajaraman}


class Radiographs(Dataset):
    def __init__(self, paths: list[Path], size: int, prepare):
        self.paths, self.size, self.prepare = paths, size, prepare

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x, shape = self.prepare(self.paths[i], self.size)
        return torch.from_numpy(x)[None], torch.tensor(shape), i


def load_model(cfg: dict, weights: Path, device: torch.device, dtype: torch.dtype):
    if cfg["model"] == "rajaraman_resnetbs":
        net = load_resnet_bs(weights).to(device, dtype)

        def split_soft_first(x):
            soft = net(x)
            return soft, x - soft

        return split_soft_first
    net = torch.jit.load(str(weights), map_location=device).eval().to(dtype)

    def split_bone_first(x):
        bone = net(x)[0]
        return x - bone, bone

    return split_bone_first


def to_u8(x: np.ndarray, shape: tuple[int, int] | None) -> np.ndarray:
    y = (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if shape is None:
        return y
    return cv2.resize(y, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def save_png(path: Path, image: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.png")
    cv2.imwrite(str(tmp), image)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = args.config.parent
    root = (base / cfg["dataset_root"]).resolve()
    out = (base / cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=out / cfg["log"], level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")

    with (root / cfg["manifest"]).open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["view"] == cfg["view"]]
    suffixes = ["soft_tissue", "bone"] if cfg["save_bone"] else ["soft_tissue"]
    todo = [root / r["path"] for r in rows if not all((out / f"{Path(r['path']).stem}_{s}.png").exists() for s in suffixes)]
    todo = todo[: args.limit]
    logging.info("model %s, frontal %d, todo %d", cfg["model"], len(rows), len(todo))
    if not todo:
        return

    device = torch.device(cfg["device"])
    dtype = torch.float16 if cfg["half_precision"] else torch.float32
    torch.backends.cudnn.benchmark = True
    suppress = load_model(cfg, (base / cfg["weights"]).resolve(), device, dtype)

    dataset = Radiographs(todo, cfg["input_size"], PREPARE[cfg["model"]])
    loader = DataLoader(dataset, batch_size=1, num_workers=cfg["num_workers"], pin_memory=True)
    pool = ThreadPoolExecutor(cfg["writer_threads"])
    pending = []
    with torch.inference_mode():
        for x, shape, i in tqdm(loader, desc=cfg["model"]):
            soft, bone = (m[0, 0].float().cpu().numpy() for m in suppress(x.to(device, dtype)))
            stem = todo[int(i)].stem
            size = tuple(int(s) for s in shape[0]) if cfg["resize_to_original"] else None
            pending.append(pool.submit(save_png, out / f"{stem}_soft_tissue.png", to_u8(soft, size)))
            if cfg["save_bone"]:
                pending.append(pool.submit(save_png, out / f"{stem}_bone.png", to_u8(bone, size)))
            logging.info("done %s", stem)
    for p in pending:
        p.result()
    pool.shutdown()
    if device.type == "cuda":
        logging.info("peak GPU memory %.0f MiB", torch.cuda.max_memory_allocated() / 2**20)


if __name__ == "__main__":
    main()
