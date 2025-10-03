from __future__ import annotations

from pathlib import Path

import click
import cv2

from .root import cli
from ..core.deepdrr_bridge import render_deepdrr_projection


@cli.command("deepdrr-project")
@click.option("--ct", "ct_path", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="Path to the CT volume (NIfTI) in Hounsfield units")
@click.option("--out", "out_path", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="Destination path for the rendered DRR (PNG)")
@click.option("--yaw", type=float, default=0.0, show_default=True)
@click.option("--pitch", type=float, default=0.0, show_default=True)
@click.option("--roll", type=float, default=0.0, show_default=True)
@click.option("--size", "sensor_height", type=int, default=512, show_default=True,
              help="Detector height in pixels (legacy square output size)")
@click.option("--sensor-width", type=int, default=None,
              help="Detector width in pixels (defaults to --size when omitted)")
@click.option("--pixel-mm", type=float, default=1.2, show_default=True,
              help="Detector pixel size (mm/pixel)")
@click.option("--sdd", "source_to_detector_distance", type=float, default=1600.0, show_default=True,
              help="Source-to-detector distance (mm)")
@click.option("--no-noise", is_flag=True, help="Disable DeepDRR noise injection")
@click.option("--spectrum", default="90KV_AL40", show_default=True,
              help="Named X-ray spectrum to use (DeepDRR preset)")
@click.option("--tone", type=click.Choice(["smooth", "raw"], case_sensitive=False), default="smooth", show_default=True,
              help="Tone-mapping style (default: smooth gamma)")
@click.option("--clahe/--no-clahe", default=False, show_default=True,
              help="Apply CLAHE after tone mapping")
@click.option("--native-resolution/--no-native-resolution", default=True, show_default=True,
              help="Keep the CT-derived projection grid (disable to force a square detector)")
def deepdrr_project(ct_path: Path, out_path: Path, yaw: float, pitch: float, roll: float,
                    sensor_height: int, sensor_width: int | None,
                    pixel_mm: float, source_to_detector_distance: float,
                    no_noise: bool, spectrum: str, tone: str, clahe: bool,
                    native_resolution: bool) -> None:
    """Render a single DeepDRR projection for quick inspection."""

    if not native_resolution and sensor_height <= 0:
        raise click.ClickException("--size must be > 0 when requesting --no-native-resolution")
    if sensor_width is not None and sensor_width <= 0:
        raise click.ClickException("--sensor-width must be > 0 when provided")

    img = render_deepdrr_projection(
        ct_path=ct_path,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        sensor_size_px=sensor_height,
        sensor_width_px=sensor_width,
        native_resolution=native_resolution,
        pixel_size_mm=pixel_mm,
        source_to_detector_distance_mm=source_to_detector_distance,
        add_noise=not no_noise,
        spectrum=spectrum,
        tone_style=tone,
        apply_clahe=clahe,
    )

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), img):
        raise click.ClickException(f"Failed to save projection to {out_path}")

    click.echo(f"✅ DeepDRR projection written to {out_path}")
