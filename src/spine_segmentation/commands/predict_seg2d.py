from pathlib import Path
import click
import torch
import numpy as np
import cv2

from .root import cli
from ..core.models.unet import UNet
from ..core.drr_utils import device
from ..core.overlay import save_u8_gray, overlay_mask_on_gray


def _read_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


@cli.command("predict-seg2d")
@click.option("--model", type=click.Path(path_type=Path), required=True, help="Path to model.pt")
@click.option("--frame", type=click.Path(path_type=Path), required=True, help="Frame image or a directory")
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("outputs/seg2d_pred"), show_default=True)
@click.option("--thresh", type=float, default=0.5, show_default=True)
def predict_seg2d(model, frame, out_dir, thresh):
    dev = device()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(model, map_location=dev)
    net = UNet(in_ch=1, out_ch=1, base=32).to(dev)
    net.load_state_dict(ckpt["model"])  # type: ignore[index]
    net.eval()

    paths = []
    frame = Path(frame)
    if frame.is_dir():
        paths.extend(sorted(list(frame.glob("**/*.png"))))
        paths.extend(sorted(list(frame.glob("**/*.jpg"))))
    else:
        paths = [frame]
    if not paths:
        raise click.ClickException("No images found to predict.")

    for p in paths:
        img = _read_gray(p)
        x = torch.from_numpy((img / 255.0).astype(np.float32))[None, None, ...].to(dev)
        with torch.no_grad():
            logits = net(x)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        mask = (prob > float(thresh)).astype(np.uint8) * 255
        save_u8_gray(out_dir / f"{p.stem}_mask.png", mask)
        over = overlay_mask_on_gray(img, mask, alpha=0.35, color=(0, 255, 0))
        cv2.imwrite(str(out_dir / f"{p.stem}_overlay.png"), over)

    click.echo(f"✅ Saved predictions to {out_dir}")

