"""Torchvision baselines (MobileNetV3, EfficientNet, ResNet) trained under the YOLO protocol."""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import CLASSES
from .data import list_folder_dataset

ARCHS = {
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "resnet18": ("resnet18", "ResNet18_Weights"),
}


def build(arch, pretrained=True, num_classes=2):
    import torchvision.models as tvm
    fn_name, w_name = ARCHS[arch]
    weights = getattr(tvm, w_name).DEFAULT if pretrained else None
    m = getattr(tvm, fn_name)(weights=weights)
    if arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        last = m.classifier[-1]
        m.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return m


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(ck["arch"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m


class _FolderSet(Dataset):
    def __init__(self, root, split, augment=False):
        from .models import IMAGENET_MEAN, IMAGENET_STD, load_tensor
        self.rows = list_folder_dataset(root, split)
        self.load, self.mean, self.std = load_tensor, IMAGENET_MEAN[0], IMAGENET_STD[0]
        if augment:  # same online blur/colour augmentation as the YOLO runs (mexp.train)
            from PIL import Image
            from .train import train_transforms
            tf = train_transforms()
            self.load = lambda path: tf(Image.open(path).convert("RGB"))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return (self.load(r["path"]) - self.mean) / self.std, CLASSES.index(r["label"])


def _evaluate(model, loader, device):
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            n += len(y)
    return correct / max(n, 1)


def train(arch, data, out_dir, epochs=30, patience=5, batch=64, lr=1e-3, weight_decay=0.01,
          label_smoothing=0.1, workers=4, seed=0, device=None, pretrained=True, augment=False):
    """Cosine LR and early stopping on validation accuracy, same as the YOLO protocol. No online
    augmentation unless ``augment`` (Roboflow already augmented the training split)."""
    torch.manual_seed(seed)
    if device is not None and str(device).isdigit():  # Ultralytics-style GPU index
        device = f"cuda:{device}"
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tr = DataLoader(_FolderSet(data, "train", augment), batch, shuffle=True, num_workers=workers)
    va = DataLoader(_FolderSet(data, "valid"), batch, num_workers=workers)
    model = build(arch, pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.5 * (1 + math.cos(math.pi * e / epochs)))
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best, bad, history = -1.0, 0, []
    for ep in range(epochs):
        model.train(); t0 = time.time(); loss_sum = 0.0
        for x, y in tr:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(y)
        sched.step()
        acc = _evaluate(model, va, device)
        history.append({"epoch": ep + 1, "train_loss": loss_sum / len(tr.dataset),
                        "val_acc": acc, "time_s": time.time() - t0})
        print(f"  [{arch}] epoch {ep + 1}/{epochs} loss={history[-1]['train_loss']:.4f} val_acc={acc:.4f}")
        if acc > best:
            best, bad = acc, 0
            torch.save({"arch": arch, "state_dict": model.state_dict()}, out_dir / "best.pth")
        else:
            bad += 1
            if bad >= patience:
                break
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return out_dir / "best.pth"


def export_onnx(pth, out, batch_dynamic=True):
    from . import IMG_H, IMG_W
    m = load_checkpoint(pth).eval()
    x = torch.zeros(1, 3, IMG_H, IMG_W)
    axes = {"images": {0: "batch"}, "output0": {0: "batch"}} if batch_dynamic else None
    torch.onnx.export(m, x, out, input_names=["images"], output_names=["output0"],
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    return out


def complexity(model):
    """Parameter count (M) and GFLOPs of one 96x320 image; works for any nn.Module."""
    import copy

    from torch.utils.flop_counter import FlopCounterMode

    from . import IMG_H, IMG_W
    model = copy.deepcopy(model).float().eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, 3, IMG_H, IMG_W))
    return params, counter.get_total_flops() / 1e9
