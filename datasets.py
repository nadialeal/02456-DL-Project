# DATA PREPARATION
import json, random, re
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader
import cv2


# --------------------- preprocessing (ROI + normalize) ---------------------

# cache for npy mask index (number -> path)
_NPY_MASK_INDEX = None


def _load_grayscale(img_path: Path) -> np.ndarray:
    """Robust grayscale loader for TIFF/PNG/JPG. Returns float32 in [0,1]."""
    img_path = Path(img_path)
    if img_path.suffix.lower() in {".tif", ".tiff"}:
        arr = tiff.imread(str(img_path))
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        arr = arr.astype(np.float32)
        if arr.max() > 255:  # likely 16-bit
            arr = arr / 65535.0
        else:
            arr = arr / (255.0 if arr.max() > 1.0 else 1.0)
        return arr
    else:
        return np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0


def _normalize_percentile_safe(img01: np.ndarray, p_low=1.0, p_high=99.5) -> np.ndarray:
    """Clip to percentiles computed on non-zero pixels (fallback to image percentiles)."""
    x = img01
    nz = x[x > 0]
    if nz.size >= 16:
        lo, hi = np.percentile(nz, (p_low, p_high))
    else:
        lo, hi = np.percentile(x, (p_low, p_high))
    if hi <= lo + 1e-8:
        lo, hi = float(x.min()), float(x.max())
    y = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return y


def preprocess_pipeline(img_path, mask_path=None, roi=None):
    """
    Loads image (and mask if provided), applies ROI if valid, percentile-normalizes image to [0,1],
    and returns PIL uint8 images (img 0..255, mask 0 or 255).
    """
    # ---- image ----
    img = _load_grayscale(Path(img_path))
    H, W = img.shape

    if roi is not None:
        y1, y2, x1, x2 = roi["y1"], roi["y2"], roi["x1"], roi["x2"]
        # clamp ROI to bounds; if invalid we skip
        if 0 <= y1 < y2 <= H and 0 <= x1 < x2 <= W:
            img = img[y1:y2, x1:x2]

    img = _normalize_percentile_safe(img, p_low=1.0, p_high=99.5)
    img_u8 = (img * 255.0).astype(np.uint8)

    if mask_path is None:
        return Image.fromarray(img_u8)

    # ---- mask ----
    mask_path = Path(mask_path)
    if mask_path.suffix.lower() == ".npy":
        m = np.squeeze(np.asarray(np.load(mask_path)))
        # masks are 0/1 with FG=1; binarize robustly
        if m.dtype == bool:
            m = m.astype(np.uint8)
        else:
            m = (m > 0.5 if m.max() <= 1.0 else m > 127).astype(np.uint8)
        if roi is not None and "y1" in roi:
            y1, y2, x1, x2 = roi["y1"], roi["y2"], roi["x1"], roi["x2"]
            if 0 <= y1 < y2 <= H and 0 <= x1 < x2 <= W:
                m = m[y1:y2, x1:x2]
        mask_u8 = (m * 255).astype(np.uint8)
        return Image.fromarray(img_u8), Image.fromarray(mask_u8)

    # png/tif masks
    mask = np.array(Image.open(mask_path).convert("L"))
    if roi is not None and "y1" in roi:
        y1, y2, x1, x2 = roi["y1"], roi["y2"], roi["x1"], roi["x2"]
        if 0 <= y1 < y2 <= H and 0 <= x1 < x2 <= W:
            mask = mask[y1:y2, x1:x2]
    mask = (mask > 127).astype(np.uint8) * 255
    return Image.fromarray(img_u8), Image.fromarray(mask)


# --------------------- crop helper (shared for weak/strong) ---------------------


def _sample_crop_coords(mask, H, W, size):
    """One crop for both branches; prefer FG center if mask is given."""
    if mask is not None and np.any(mask > 0):
        ys, xs = np.where(mask > 0)
        i = np.random.randint(len(xs))
        cy, cx = int(ys[i]), int(xs[i])
    else:
        cy = np.random.randint(size // 2, max(size // 2 + 1, H - size // 2))
        cx = np.random.randint(size // 2, max(size // 2 + 1, W - size // 2))
    y1 = int(np.clip(cy - size // 2, 0, H - size))
    x1 = int(np.clip(cx - size // 2, 0, W - size))
    return y1, x1


# --------------------- dataset ---------------------


class MaterialDataset(Dataset):
    def __init__(self, samples, mode, cfg, use_strong_weak=False, full_image=False):
        """
        samples: list of dicts
          labeled/val/test -> {"img": path, "mask": path}
          unlabeled        -> {"img": path}
        mode: "labeled" | "val" | "test" | "unlabeled"
        use_strong_weak: if True, return two views (weak/strong) for mean-teacher
        full_image: if True (val/test only), return full ROI image (for sliding-window eval)
        """
        self.samples = samples
        self.mode = mode
        self.cfg = cfg
        self.use_strong_weak = use_strong_weak
        self.full_image = full_image
        ps = cfg["patch_size"]

        # shared geometry (record & replay same transform for weak/strong)
        self.geom = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        ])

        # intensity heads (image only)
        self.intensity_weak = A.Compose([])  # keep weak nearly clean
        self.intensity_strong = A.Compose([
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur()], p=0.5),
            # A.CLAHE(clip_limit=2.0, p=0.3),  # optional; disable if preprocess already normalizes well
        ])

        # deterministic val/test crop + normalize
        self.val_crop = A.Compose([
            A.PadIfNeeded(min_height=ps, min_width=ps, border_mode=cv2.BORDER_REFLECT_101),
            A.CenterCrop(ps, ps, p=1.0),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ], additional_targets={"mask": "mask"})

        self.normalize_only = A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
        self.to_tensor = ToTensorV2()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        # If you want ROI only for unlabeled big scans, change to:
        # roi = self.cfg.get("roi_unlabeled") if self.mode == "unlabeled" else None
        roi = self.cfg.get("roi")
        ps = self.cfg["patch_size"]

        if self.mode in ("labeled", "val", "test"):
            img_pil, mask_pil = preprocess_pipeline(rec["img"], rec["mask"], roi=roi)
            img = np.array(img_pil)
            mask = (np.array(mask_pil) // 255).astype(np.uint8)

            # full-ROI pass-through for sliding-window eval
            if self.mode in ("val", "test") and self.full_image:
                t = self.to_tensor(image=self.normalize_only(image=img)["image"], mask=mask)
                return {"x": t["image"], "y": t["mask"].unsqueeze(0).float()}

            # deterministic val/test patch path (no weak/strong)
            if self.mode in ("val", "test") and not self.use_strong_weak:
                aug = self.val_crop(image=img, mask=mask)
                t = self.to_tensor(image=aug["image"], mask=aug["mask"])
                return {"x": t["image"], "y": t["mask"].unsqueeze(0).float()}

            if self.use_strong_weak:
                # 1) ONE base crop for both branches
                H, W = img.shape[:2]
                y1, x1 = _sample_crop_coords(mask, H, W, ps)
                crop = A.Crop(x_min=x1, y_min=y1, x_max=x1 + ps, y_max=y1 + ps)
                base = crop(image=img, mask=mask)

                # 2) SAME geometry (record + replay) for weak & strong
                g_w = self.geom(image=base["image"], mask=base["mask"])
                g_s = A.ReplayCompose.replay(g_w["replay"], image=base["image"], mask=base["mask"])

                # 3) intensity only on images; strong is heavier
                iw = self.intensity_weak(image=g_w["image"])["image"]
                is_ = self.intensity_strong(image=g_s["image"])["image"]

                # 4) normalize and to tensor
                iw = self.normalize_only(image=iw)["image"]
                is_ = self.normalize_only(image=is_)["image"]

                tw = self.to_tensor(image=iw, mask=g_w["mask"])
                ts = self.to_tensor(image=is_, mask=g_s["mask"])

                # supervised loss uses the STRONG branch (mask aligned to strong)
                return {"x_w": tw["image"], "x_s": ts["image"], "y": ts["mask"].unsqueeze(0).float()}

            # fallback
            aug = self.val_crop(image=img, mask=mask)
            t = self.to_tensor(image=aug["image"], mask=aug["mask"])
            return {"x": t["image"], "y": t["mask"].unsqueeze(0).float()}

        # -------- unlabeled --------
        img_pil = preprocess_pipeline(rec["img"], roi=roi)
        img = np.array(img_pil)

        if self.use_strong_weak:
            H, W = img.shape[:2]
            y1, x1 = _sample_crop_coords(None, H, W, ps)
            crop = A.Crop(x_min=x1, y_min=y1, x_max=x1 + ps, y_max=y1 + ps)
            base = crop(image=img)

            # same geometry for both, then different intensity
            g = self.geom(image=base["image"])  # record geometry once
            iw = self.intensity_weak(image=g["image"])["image"]
            is_ = self.intensity_strong(image=g["image"])["image"]

            iw = self.normalize_only(image=iw)["image"]
            is_ = self.normalize_only(image=is_)["image"]

            return {
                "x_w": self.to_tensor(image=iw)["image"],
                "x_s": self.to_tensor(image=is_)["image"],
            }

        # single-view unlabeled (not used in mean-teacher, but kept for completeness)
        aug = self.val_crop(image=img)
        x = self.to_tensor(image=aug["image"])["image"]
        return {"x": x}


# --------------------- helpers to match folder layout ---------------------


def _index_npy_masks(masks_dir: Path):
    """
    Build an index mapping the LAST integer in the filename to its .npy path.
    Accepts names like: mask_000.npy, mask_000_inv.npy, anything*123*.npy
    """
    idx = {}
    for p in masks_dir.glob("*.npy"):
        nums = re.findall(r"(\d+)", p.stem)
        if nums:
            idx[int(nums[-1])] = p
    return idx


def _find_mask_for_image(img_path: Path, masks_dir: Path):
    """
    Try (1) same stem with common image extensions, then
        (2) map image's LAST numeric id to a .npy mask (e.g., image_v2_21 -> id 21 -> mask_021_inv.npy).
    """
    stem = img_path.stem

    # 1) same-stem image masks (png/tif/tiff)
    for ext in (".png", ".tif", ".tiff"):
        cand = masks_dir / f"{stem}{ext}"
        if cand.exists():
            return cand

    # 2) numeric id mapping to .npy  -> use the LAST number in the image stem
    nums = re.findall(r"(\d+)", stem)
    if nums:
        num = int(nums[-1])
        global _NPY_MASK_INDEX
        if _NPY_MASK_INDEX is None:
            _NPY_MASK_INDEX = _index_npy_masks(masks_dir)
        if num in _NPY_MASK_INDEX:
            return _NPY_MASK_INDEX[num]

        # fallbacks with common paddings
        for name in (f"mask_{num:03d}.npy", f"mask_{num:03d}_inv.npy",
                     f"mask_{num:02d}.npy", f"mask_{num:02d}_inv.npy"):
            cand = masks_dir / name
            if cand.exists():
                return cand

    raise FileNotFoundError(f"No mask found for image '{img_path.name}' in {masks_dir}")


def _list_labeled_pairs(labeled_dir: str):
    img_dir = Path(labeled_dir) / "Original_images"
    mask_dir = Path(labeled_dir) / "Original_masks"
    assert img_dir.exists(), f"Missing: {img_dir}"
    assert mask_dir.exists(), f"Missing: {mask_dir}"
    pairs = []
    imgs = sorted(list(img_dir.glob("*.tif")) + list(img_dir.glob("*.tiff")))
    for img in imgs:
        mask = _find_mask_for_image(img, mask_dir)
        pairs.append({"img": str(img), "mask": str(mask), "stem": img.stem})
    return pairs


# --------------------- splits + loaders ---------------------


def ensure_split(CFG):
    """
    Writes split.json with lists of **stems** from labeled/Original_images.
    Only run once; delete the file to reshuffle.
    """
    sp = Path(CFG["paths"]["split_json"])
    if sp.exists():
        return
    pairs = _list_labeled_pairs(CFG["paths"]["labeled_dir"])
    stems = [p["stem"] for p in pairs]
    rng = random.Random(CFG["seed"])
    rng.shuffle(stems)

    n = len(stems)
    n_test = min(5, max(1, n // 5))
    n_val = min(4, max(1, n // 5))
    test = stems[:n_test]
    val = stems[n_test:n_test + n_val]
    train = stems[n_test + n_val:]

    sp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"train": train, "val": val, "test": test}, open(sp, "w"), indent=2)


def _collect_labeled_samples(labeled_dir: str, stems: list[str]):
    img_dir = Path(labeled_dir) / "Original_images"
    mask_dir = Path(labeled_dir) / "Original_masks"
    out = []
    for stem in stems:
        img = img_dir / f"{stem}.tif"
        if not img.exists():
            alt = img_dir / f"{stem}.tiff"
            if alt.exists():
                img = alt
            else:
                raise FileNotFoundError(f"Missing image for stem {stem} in {img_dir}")
        mask = _find_mask_for_image(img, mask_dir)
        out.append({"img": str(img), "mask": str(mask)})
    return out


def _collect_unlabeled_samples(unlabeled_dir: str):
    unl_dir = Path(unlabeled_dir)
    imgs = sorted(list(unl_dir.glob("*.tif")) + list(unl_dir.glob("*.tiff")))
    return [{"img": str(p)} for p in imgs]


def make_loaders(CFG):
    split = json.load(open(CFG["paths"]["split_json"]))
    lab_dir, unl_dir = CFG["paths"]["labeled_dir"], CFG["paths"]["unlabeled_dir"]

    train = _collect_labeled_samples(lab_dir, split["train"])
    val = _collect_labeled_samples(lab_dir, split["val"])
    test = _collect_labeled_samples(lab_dir, split["test"])
    unl = _collect_unlabeled_samples(unl_dir)

    ds_train = MaterialDataset(train, "labeled", CFG, use_strong_weak=True)
    ds_val = MaterialDataset(val, "val", CFG, use_strong_weak=False)
    ds_test = MaterialDataset(test, "test", CFG, use_strong_weak=False)
    ds_unl = MaterialDataset(unl, "unlabeled", CFG, use_strong_weak=True)
    ds_val_full = MaterialDataset(val, "val", CFG, use_strong_weak=False, full_image=True)
    ds_test_full = MaterialDataset(test, "test", CFG, use_strong_weak=False, full_image=True)

    bs, nw = CFG["batch_size"], CFG["num_workers"]
    return dict(
        labeled=DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True),
        val=DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        test=DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        unlabeled=DataLoader(ds_unl, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True),
        val_full=DataLoader(ds_val_full, batch_size=1, shuffle=False, num_workers=nw, pin_memory=True),
        test_full=DataLoader(ds_test_full, batch_size=1, shuffle=False, num_workers=nw, pin_memory=True),
    )
