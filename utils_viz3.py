import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def _to01(img: torch.Tensor) -> torch.Tensor:
    """Per-image min-max to [0,1] for visualization (torch in, torch out)."""
    img = img.float()
    img = img - img.amin(dim=(-2, -1), keepdim=True)
    img = img / (img.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return img.clamp(0, 1)

def _to01_np(x) -> np.ndarray:
    """Same as _to01 but returns numpy (accepts torch OR numpy)."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().float()
    else:  # numpy -> torch
        t = torch.from_numpy(x).float()
    t = t - t.amin(dim=(-2, -1), keepdim=True)
    t = t / (t.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return t.clamp(0, 1).numpy()

def overlay_mask(img, mask, alpha=0.35):
    # ensure numpy for stacking
    img01 = _to01_np(img)
    mask  = (np.asarray(mask) > 0.5).astype(np.float32)
    rgb   = np.stack([img01, img01, img01], axis=-1)
    tint  = np.zeros_like(rgb); tint[..., 0] = mask  # red
    out   = (1 - alpha) * rgb + alpha * tint
    return np.clip(out, 0.0, 1.0)

def show_augmented_labeled(batch, n=2):
    # keep tensors as tensors
    xw = batch["x_w"].detach().cpu()  # [B,1,H,W] torch
    xs = batch["x_s"].detach().cpu()
    y  = batch["y"].detach().cpu().numpy()  # mask is fine as numpy

    fig, axes = plt.subplots(n, 4, figsize=(16, 8))
    fig.suptitle("Labeled augmentations (weak vs strong + mask)")

    for i in range(n):
        iw = _to01_np(xw[i,0])   # -> numpy in [0,1]
        is_ = _to01_np(xs[i,0])
        m  = (y[i,0] > 0.5).astype(np.float32)

        axes[i,0].imshow(iw, cmap="gray", vmin=0, vmax=1); axes[i,0].set_title("weak");  axes[i,0].axis("off")
        axes[i,1].imshow(is_, cmap="gray", vmin=0, vmax=1); axes[i,1].set_title("strong");axes[i,1].axis("off")
        axes[i,2].imshow(overlay_mask(is_, m), vmin=0, vmax=1); axes[i,2].set_title("strong + mask"); axes[i,2].axis("off")
        axes[i,3].imshow(overlay_mask(iw, m),  vmin=0, vmax=1); axes[i,3].set_title("weak + mask");   axes[i,3].axis("off")

    plt.tight_layout(); plt.show()

def show_augmented_unlabeled(batch, n=2):
    xw = batch["x_w"].detach().cpu()
    xs = batch["x_s"].detach().cpu()

    fig, axes = plt.subplots(n, 2, figsize=(9, 9))
    fig.suptitle("Unlabeled augmentations (weak vs strong)")
    for i in range(n):
        iw = _to01_np(xw[i,0])
        is_ = _to01_np(xs[i,0])

        axes[i,0].imshow(iw, cmap="gray", vmin=0, vmax=1); axes[i,0].set_title("weak");  axes[i,0].axis("off")
        axes[i,1].imshow(is_, cmap="gray", vmin=0, vmax=1); axes[i,1].set_title("strong");axes[i,1].axis("off")

    plt.tight_layout(); plt.show()

@torch.no_grad()
def _as_prob(t: torch.Tensor) -> torch.Tensor:
    """If already ~[0,1] treat as prob; else apply sigmoid."""
    t = t.float()
    t_min = t.min().item(); t_max = t.max().item()
    return t if (t_min >= -1e-3 and t_max <= 1 + 1e-3) else torch.sigmoid(t)

def save_pred_grid(x, y, pred_like, out_path, thr: float = 0.5, max_rows: int = 4):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x_vis = _to01(x.detach().cpu())
    y_vis = y.detach().cpu().float()
    p_vis = _as_prob(pred_like.detach().cpu())
    pred  = (p_vis > thr).float()

    B = min(max_rows, x_vis.size(0))
    fig, axes = plt.subplots(B, 3, figsize=(9, 3 * B))
    if B == 1: axes = axes[None, :]
    for i in range(B):
        axes[i,0].imshow(x_vis[i,0].numpy(), cmap="gray", vmin=0, vmax=1); axes[i,0].set_title("image"); axes[i,0].axis("off")
        axes[i,1].imshow(y_vis[i,0].numpy(), cmap="gray", vmin=0, vmax=1); axes[i,1].set_title("GT");    axes[i,1].axis("off")
        axes[i,2].imshow(pred[i,0].numpy(),  cmap="gray", vmin=0, vmax=1); axes[i,2].set_title(f"pred @{thr}"); axes[i,2].axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)

# (the 5-panel helpers below stay the same as in your file)
def _overlay_rgb(img01: np.ndarray, mask01: np.ndarray, color=(1.0, 0.0, 0.0), alpha=0.35):
    """Overlay a binary mask on a grayscale image (HxW) -> RGB with tint."""
    rgb = np.stack([img01, img01, img01], axis=-1)
    m = mask01 > 0.5
    tint = np.zeros_like(rgb); tint[..., 0], tint[..., 1], tint[..., 2] = color
    rgb[m] = (1 - alpha) * rgb[m] + alpha * tint[m]
    return np.clip(rgb, 0, 1)

def _fp_fn_rgb(img01: np.ndarray, pred01: np.ndarray, gt01: np.ndarray):
    """Return RGB where FP=red, FN=cyan, otherwise grayscale."""
    rgb = np.stack([img01, img01, img01], axis=-1)
    fp = (pred01 > 0.5) & (gt01 < 0.5)
    fn = (pred01 < 0.5) & (gt01 > 0.5)
    rgb[fp] = [1.0, 0.0, 0.0]     # red
    rgb[fn] = [0.0, 1.0, 1.0]     # cyan
    return rgb

def save_patch_vis_5panel(x, y, pred_like, out_path, thr: float = 0.5, max_rows: int = 4):
    """
    Save a 5-column grid per patch: image, GT, prob, overlay, FP/FN.
    x: (B,1,H,W)  normalized like training (we rescale to [0,1] for display)
    y: (B,1,H,W)  {0,1}
    pred_like: (B,1,H,W) logits OR probabilities
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x_vis = _to01(x.detach().cpu())
    y_vis = y.detach().cpu().float()
    p_vis = _as_prob(pred_like.detach().cpu())  # [0,1]
    pred  = (p_vis > thr).float()

    B = min(max_rows, x_vis.size(0))
    fig, axes = plt.subplots(B, 5, figsize=(15, 3.2 * B))
    if B == 1: axes = axes[None, :]

    for i in range(B):
        img = x_vis[i, 0].numpy()
        gt  = y_vis[i, 0].numpy()
        prob= p_vis[i, 0].numpy()
        pr  = pred[i, 0].numpy()

        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1);        axes[i, 0].set_title("image");       axes[i, 0].axis("off")
        axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1);         axes[i, 1].set_title("GT mask");     axes[i, 1].axis("off")
        axes[i, 2].imshow(prob, vmin=0, vmax=1);                    axes[i, 2].set_title("prob");        axes[i, 2].axis("off")
        axes[i, 3].imshow(_overlay_rgb(img, pr, color=(1,0,0)));    axes[i, 3].set_title(f"pred @{thr}");axes[i, 3].axis("off")
        axes[i, 4].imshow(_fp_fn_rgb(img, pr, gt), vmin=0, vmax=1); axes[i, 4].set_title("FP=red, FN=cyan"); axes[i, 4].axis("off")

    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)