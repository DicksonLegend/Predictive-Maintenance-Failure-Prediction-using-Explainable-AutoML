"""Image Explainability Engine for Industrial Machine Component Inspection.

Generates illustrative visual inspection artifacts with saliency heatmaps
demonstrating spatial explainability (e.g., Grad-CAM / saliency maps) for
tool blade micro-cracks, bearing spalling, and stator thermal hotspots.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from app.api.config import IMAGE_DEMO_DIR
from app.api.schemas import ImageDemoItem, ImageDemoResponse

logger = logging.getLogger("image_explainer")
logger.setLevel(logging.INFO)


def _generate_synthetic_tool_crack():
    """Generates synthetic high-resolution image of a milling tool cutting edge with a micro-crack."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    # Background metal texture
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    base_texture = 0.6 + 0.1 * np.sin(5 * X) + 0.05 * np.random.randn(300, 300)

    # Tool wedge geometry
    tool_mask = (Y > -0.5 * X - 0.2)
    img = np.where(tool_mask, base_texture, 0.15 + 0.02 * np.random.randn(300, 300))

    # Crack feature
    crack_x = np.linspace(100, 200, 150)
    crack_y = 150 + 20 * np.sin(0.08 * crack_x) + np.random.randn(150) * 2
    for cx, cy in zip(crack_x.astype(int), crack_y.astype(int)):
        if 0 <= cx < 300 and 0 <= cy < 300:
            img[max(0, cy - 2):min(300, cy + 3), max(0, cx - 2):min(300, cx + 3)] = 0.05

    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)

    # Heatmap centered on crack
    heatmap = np.exp(-((X - 0.2)**2 + (Y - 0.1)**2) / 0.35)
    return fig, heatmap


def _generate_synthetic_bearing_defect():
    """Generates synthetic bearing race image with surface spalling."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)

    # Annular bearing race
    race_mask = (r >= 1.2) & (r <= 2.5)
    img = np.where(race_mask, 0.7 + 0.08 * np.sin(20 * np.arctan2(Y, X)), 0.2)
    img += 0.04 * np.random.randn(300, 300)

    # Pitted spalling region at angle ~ 45 deg
    spall_mask = (r >= 1.6) & (r <= 2.1) & (X > 0.8) & (Y > 0.8)
    img[spall_mask] = 0.1 + 0.1 * np.random.rand(np.sum(spall_mask))

    ax.imshow(img, cmap="bone", vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)

    # Heatmap centered on spall
    heatmap = np.exp(-((X - 1.4)**2 + (Y - 1.4)**2) / 0.4)
    return fig, heatmap


def _generate_synthetic_thermal_hotspot():
    """Generates synthetic stator thermal inspection image with localized hotspot."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)

    # Motor housing concentric ribbed pattern
    ribs = np.cos(4 * np.pi * np.sqrt(X**2 + Y**2))
    img = 0.5 + 0.2 * ribs + 0.03 * np.random.randn(300, 300)

    ax.imshow(img, cmap="copper", vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)

    # Heatmap representing severe thermal spike on upper winding
    heatmap = np.exp(-((X + 0.5)**2 + (Y - 0.8)**2) / 0.5)
    return fig, heatmap


def generate_image_demo_artifacts() -> None:
    """Generates and serializes all image XAI demo artifacts."""
    os.makedirs(IMAGE_DEMO_DIR, exist_ok=True)

    generators = [
        ("image_0", "Milling Tool Blade Micro-Crack", "Tool Surface Fatigue Crack",
         "Saliency attribution pinpoints high tensile stress concentration and micro-crack bifurcation along the primary cutting shear lip.",
         _generate_synthetic_tool_crack),
        ("image_1", "Bearing Inner Race Spalling", "Sub-surface Fatigue Spall",
         "Spatial activation highlights flaked metallic debris pitting along the raceway groove, correlating with high acoustic vibration.",
         _generate_synthetic_bearing_defect),
        ("image_2", "Stator Winding Thermal Hotspot", "Phase Winding Thermal Overload",
         "Thermal saliency overlay identifies severe localized heat dissipation deficit, indicating localized inter-turn dielectric breakdown.",
         _generate_synthetic_thermal_hotspot),
    ]

    for prefix, title, pred_class, explanation, gen_fn in generators:
        fig_orig, heatmap = gen_fn()

        orig_path = IMAGE_DEMO_DIR / f"{prefix}_original.png"
        heatmap_path = IMAGE_DEMO_DIR / f"{prefix}_heatmap.png"
        overlay_path = IMAGE_DEMO_DIR / f"{prefix}_overlay.png"
        meta_path = IMAGE_DEMO_DIR / f"{prefix}_metadata.json"

        # 1. Save original
        fig_orig.savefig(orig_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig_orig)

        # 2. Save heatmap
        fig_hm, ax_hm = plt.subplots(figsize=(5, 5), dpi=100)
        ax_hm.imshow(heatmap, cmap="jet")
        ax_hm.axis("off")
        fig_hm.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig_hm)

        # 3. Create blend overlay
        orig_img = Image.open(orig_path).convert("RGBA")
        hm_img = Image.open(heatmap_path).convert("RGBA").resize(orig_img.size)
        overlay = Image.blend(orig_img, hm_img, alpha=0.45)
        overlay.save(overlay_path)

        # 4. Save metadata
        metadata = {
            "image_id": prefix,
            "title": title,
            "predicted_class": pred_class,
            "image_demo_mode": "illustrative_fallback",
            "explanation": explanation,
            "original_path": f"/artifacts/xai/image_demo/{prefix}_original.png",
            "heatmap_path": f"/artifacts/xai/image_demo/{prefix}_heatmap.png",
            "overlay_path": f"/artifacts/xai/image_demo/{prefix}_overlay.png"
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    logger.info("Image XAI demo artifacts saved under %s", IMAGE_DEMO_DIR)


def get_image_demo_response() -> ImageDemoResponse:
    """Returns the ImageDemoResponse payload for the API."""
    items: List[ImageDemoItem] = []
    for prefix in ["image_0", "image_1", "image_2"]:
        meta_file = IMAGE_DEMO_DIR / f"{prefix}_metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            items.append(ImageDemoItem(
                title=d["title"],
                original=d["original_path"],
                heatmap=d["heatmap_path"],
                overlay=d["overlay_path"],
                predicted_class=d["predicted_class"],
                explanation=d["explanation"]
            ))

    if not items:
        generate_image_demo_artifacts()
        return get_image_demo_response()

    return ImageDemoResponse(
        image_demo_mode="illustrative_fallback",
        items=items
    )
