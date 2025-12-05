
import os, csv
import random, numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CSVLogger:
    def __init__(self, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if write_header: self.w.writerow(header)
    def log(self, row):
        self.w.writerow(row); self.f.flush()
    def close(self):
        self.f.close()

@torch.no_grad()
def dice_iou_from_logits(logits, y, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p*y).sum(dim=(1,2,3))
    dice = (2*inter + eps) / (p.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3)) + eps)
    iou  = (inter + eps) / (p.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3)) - inter + eps)
    return float(dice.mean().item()), float(iou.mean().item())

@torch.no_grad()
def run_validation(model, loader, device):
    model.eval()
    dices, ious = [], []
    for b in loader:
        x = b["x"].to(device); y = b["y"].to(device)
        logits = model(x)
        d,i = dice_iou_from_logits(logits, y)
        dices.append(d); ious.append(i)
    return float(np.mean(dices)), float(np.mean(ious))

@torch.no_grad()
def sliding_window_predict(model, x_full, patch=256, overlap=0.5, device="cuda"):
    """
    x_full: [1,H,W] float tensor in [0,1]; returns [1,H,W] logits stitched by averaging overlaps
    """
    model.eval()
    _, H, W = x_full.shape
    stride = max(1, int(patch * (1 - overlap)))
    out  = torch.zeros((1, H, W), device=device)
    norm = torch.zeros((1, H, W), device=device)

    for y in range(0, max(1, H - patch + 1), stride):
        for x in range(0, max(1, W - patch + 1), stride):
            tile = x_full[:, y:y+patch, x:x+patch].unsqueeze(0).to(device)  # [1,1,p,p]
            if tile.shape[-1] < patch or tile.shape[-2] < patch:
                tile = F.pad(tile, (0, patch - tile.shape[-1], 0, patch - tile.shape[-2]))
            logits = model(tile)[:, :, :min(patch, H-y), :min(patch, W-x)]
            out[:,  y:y+logits.shape[-2], x:x+logits.shape[-1]] += logits.squeeze(0)
            norm[:, y:y+logits.shape[-2], x:x+logits.shape[-1]] += 1.0

    return out / norm.clamp_min(1.0)

@torch.no_grad()
def eval_full_images(model, loader_full, patch, overlap, device, dice_iou_fn):
    dices, ious = [], []
    for b in loader_full:
        x_full = b["x"].to(device)  # [1,1,H,W]
        y_full = b["y"].to(device)
        logits = sliding_window_predict(model, x_full[0], patch=patch, overlap=overlap, device=device)
        d,i = dice_iou_fn(logits.unsqueeze(0), y_full[0].unsqueeze(0))
        dices.append(d); ious.append(i)
    return float(np.mean(dices)), float(np.mean(ious))

def plot_curves(run_dir):
    sup_csv = os.path.join(run_dir, "sup_metrics.csv")
    ssl_csv = os.path.join(run_dir, "ssl_metrics.csv")
    if not (os.path.exists(sup_csv) or os.path.exists(ssl_csv)):
        print("No metrics found."); return

    if os.path.exists(sup_csv) or os.path.exists(ssl_csv):
        plt.figure()
        if os.path.exists(sup_csv):
            df = pd.read_csv(sup_csv)
            plt.plot(df.epoch, df.val_dice, label='sup val Dice')
        if os.path.exists(ssl_csv):
            df = pd.read_csv(ssl_csv)
            plt.plot(df.epoch, df.val_dice, label='ssl val Dice')
        plt.xlabel('epoch'); plt.ylabel('Dice'); plt.legend(); plt.title('Validation Dice'); plt.show()

    for csv_path, title in [(sup_csv, 'Supervised'), (ssl_csv, 'Semi-supervised')]:
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        plt.figure()
        plt.plot(df.epoch, df.loss_sup, label='loss_sup')
        if 'loss_cons' in df.columns:
            plt.plot(df.epoch, df.loss_cons, label='loss_cons')
        plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title(title+' Losses'); plt.show()
