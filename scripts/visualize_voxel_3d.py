"""
3D Medical Voxel Visualization — BDMAP Organ Segmentation

使用 Plotly 交互式可视化 voxel_labels 数据
- Marching Cubes 提取器官表面网格
- 按解剖分组归类，支持分组 / 单器官开关
- 科研配色方案（Nature / Lancet 风格）

Usage:
    python visualize_voxel_3d.py [--file PATH] [--downsample N]

Author: auto-generated
Date: 2026-04-14
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes

# ---------------------------------------------------------------------------
# 项目内器官映射
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.organ_mapping import LABEL_TO_ORGAN

# ---------------------------------------------------------------------------
# 解剖分组 & 科研配色
# ---------------------------------------------------------------------------
ORGAN_GROUPS = {
    "Solid Organs": list(range(1, 14)),       # liver … spinal_cord
    "Lung Lobes": list(range(14, 19)),
    "Digestive Tract": list(range(19, 24)),
    "Adrenal Glands": [24, 25],
    "Vertebrae": list(range(26, 52)),
    "Left Ribs": list(range(52, 64)),
    "Right Ribs": list(range(64, 76)),
    "Other Bones": list(range(76, 89)),
    "Muscles": list(range(89, 99)),
    "Arteries": list(range(99, 108)),
    "Veins & Cardiac": list(range(108, 117)),
    "Body Composition": [119, 120, 121, 122],
}

# 每个分组一个主色调，组内器官通过明度/饱和度微调区分
# 科研级调色板 — 参考 Nature Methods / Lancet 配色
GROUP_PALETTES = {
    "Solid Organs":     {"h": 10,  "s": [0.65, 0.85], "l": [0.45, 0.65]},  # warm red-brown
    "Lung Lobes":       {"h": 200, "s": [0.50, 0.70], "l": [0.55, 0.72]},  # sky blue
    "Digestive Tract":  {"h": 35,  "s": [0.60, 0.80], "l": [0.48, 0.62]},  # amber
    "Adrenal Glands":   {"h": 55,  "s": [0.55, 0.70], "l": [0.50, 0.60]},  # olive
    "Vertebrae":        {"h": 45,  "s": [0.20, 0.40], "l": [0.60, 0.78]},  # bone ivory
    "Left Ribs":        {"h": 40,  "s": [0.18, 0.35], "l": [0.62, 0.80]},  # light bone
    "Right Ribs":       {"h": 50,  "s": [0.18, 0.35], "l": [0.62, 0.80]},  # light bone
    "Other Bones":      {"h": 42,  "s": [0.25, 0.45], "l": [0.55, 0.72]},  # tan
    "Muscles":          {"h": 0,   "s": [0.55, 0.75], "l": [0.40, 0.55]},  # deep red
    "Arteries":         {"h": 355, "s": [0.75, 0.90], "l": [0.42, 0.55]},  # bright red
    "Veins & Cardiac":  {"h": 240, "s": [0.50, 0.70], "l": [0.40, 0.58]},  # blue-violet
    "Body Composition": {"h": 90,  "s": [0.15, 0.30], "l": [0.65, 0.80]},  # muted green-gray
}


def _color_for_organ(group_name: str, idx_in_group: int, group_size: int) -> str:
    """Generate an HSL color for an organ within its anatomical group."""
    pal = GROUP_PALETTES.get(group_name)
    if pal is None:
        return "hsl(0, 50%, 50%)"
    t = idx_in_group / max(group_size - 1, 1)
    s = pal["s"][0] + t * (pal["s"][1] - pal["s"][0])
    l = pal["l"][0] + t * (pal["l"][1] - pal["l"][0])
    return f"hsl({pal['h']}, {s*100:.0f}%, {l*100:.0f}%)"


# ---------------------------------------------------------------------------
# Marching Cubes mesh extraction
# ---------------------------------------------------------------------------
def extract_organ_mesh(voxel_labels: np.ndarray, label_id: int,
                       world_min: np.ndarray, voxel_size: np.ndarray,
                       downsample: int = 1):
    """
    Extract surface mesh for a single organ label using marching cubes.

    Returns (vertices_world, faces) or None if organ not present / too small.
    """
    mask = (voxel_labels == label_id).astype(np.float32)
    if mask.sum() < 4:
        return None

    # Optional spatial downsampling for speed
    if downsample > 1:
        d = downsample
        mask = mask[::d, ::d, ::d]
        effective_voxel = voxel_size * d
    else:
        effective_voxel = voxel_size

    try:
        verts, faces, _, _ = marching_cubes(mask, level=0.5,
                                            spacing=tuple(effective_voxel))
    except (RuntimeError, ValueError):
        return None

    # voxel index → world coordinates
    verts_world = verts + world_min
    return verts_world, faces


# ---------------------------------------------------------------------------
# Build Plotly figure
# ---------------------------------------------------------------------------
def build_figure(npz_path: str, downsample: int = 1,
                 show_groups: list | None = None,
                 opacity: float = 1.0) -> go.Figure:
    """
    Build an interactive 3D Plotly figure.

    Parameters
    ----------
    npz_path : path to the .npz file
    downsample : spatial downsample factor (1 = full res, 2 = half, …)
    show_groups : list of group names to show initially (None = all)
    opacity : mesh opacity (1.0 = fully opaque)
    """
    data = np.load(npz_path)
    voxel_labels = data["voxel_labels"]
    world_min = data["grid_world_min"]
    voxel_size = data["grid_voxel_size"]

    present_labels = set(np.unique(voxel_labels)) - {0}

    traces = []
    buttons_group = []  # group-level toggle buttons

    for group_name, label_ids in ORGAN_GROUPS.items():
        group_ids_present = [lid for lid in label_ids if lid in present_labels]
        if not group_ids_present:
            continue

        group_visible = (show_groups is None or group_name in show_groups)
        group_trace_indices = []

        for i, lid in enumerate(group_ids_present):
            result = extract_organ_mesh(voxel_labels, lid, world_min,
                                        voxel_size, downsample)
            if result is None:
                continue

            verts, faces = result
            organ_name = LABEL_TO_ORGAN.get(lid, f"label_{lid}")
            color = _color_for_organ(group_name, i, len(group_ids_present))

            trace = go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color,
                opacity=opacity,
                name=organ_name,
                legendgroup=group_name,
                legendgrouptitle_text=group_name,
                hovertemplate=(
                    f"<b>{organ_name}</b> (id {lid})<br>"
                    f"Group: {group_name}<br>"
                    "x: %{x:.1f} mm<br>y: %{y:.1f} mm<br>z: %{z:.1f} mm"
                    "<extra></extra>"
                ),
                visible=True if group_visible else "legendonly",
                flatshading=True,
                lighting=dict(
                    ambient=0.3,
                    diffuse=0.7,
                    specular=0.2,
                    roughness=0.5,
                    fresnel=0.1,
                ),
                lightposition=dict(x=1000, y=1000, z=1000),
            )
            group_trace_indices.append(len(traces))
            traces.append(trace)

        # group toggle button
        if group_trace_indices:
            buttons_group.append(
                dict(label=group_name, method="restyle",
                     args=[{"visible": True}, group_trace_indices]),
                # We'll build proper toggle buttons below
            )

    if not traces:
        raise RuntimeError("No organ meshes could be extracted — data may be empty.")

    fig = go.Figure(data=traces)

    # ----- Dropdown: group visibility presets -----
    n = len(traces)

    def _visibility_for(groups_on):
        vis = []
        idx = 0
        for gname, lids in ORGAN_GROUPS.items():
            present = [lid for lid in lids if lid in present_labels]
            for lid in present:
                # check if mesh was actually generated
                if idx < n:
                    vis.append(True if gname in groups_on else "legendonly")
                    idx += 1
        # pad remaining (shouldn't happen)
        while len(vis) < n:
            vis.append("legendonly")
        return vis

    preset_buttons = [
        dict(label="All Organs",
             method="restyle",
             args=[{"visible": [True] * n}]),
        dict(label="Solid Organs Only",
             method="restyle",
             args=[{"visible": _visibility_for({"Solid Organs"})}]),
        dict(label="Skeleton",
             method="restyle",
             args=[{"visible": _visibility_for(
                 {"Vertebrae", "Left Ribs", "Right Ribs", "Other Bones"})}]),
        dict(label="Thorax (Lungs + Heart + Ribs)",
             method="restyle",
             args=[{"visible": _visibility_for(
                 {"Lung Lobes", "Solid Organs", "Left Ribs", "Right Ribs",
                  "Other Bones"})}]),
        dict(label="Vasculature",
             method="restyle",
             args=[{"visible": _visibility_for(
                 {"Arteries", "Veins & Cardiac"})}]),
        dict(label="Muscles + Bones",
             method="restyle",
             args=[{"visible": _visibility_for(
                 {"Muscles", "Vertebrae", "Left Ribs", "Right Ribs",
                  "Other Bones"})}]),
        dict(label="Soft Tissue Only",
             method="restyle",
             args=[{"visible": _visibility_for(
                 {"Solid Organs", "Lung Lobes", "Digestive Tract",
                  "Muscles"})}]),
        dict(label="Hide All",
             method="restyle",
             args=[{"visible": ["legendonly"] * n}]),
    ]

    # ----- Opacity slider -----
    opacity_steps = []
    for op_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        opacity_steps.append(
            dict(label=f"{op_val:.1f}",
                 method="restyle",
                 args=[{"opacity": op_val}])
        )

    # ----- Layout -----
    fig.update_layout(
        title=dict(
            text=(f"<b>3D Organ Segmentation</b> — "
                  f"{Path(npz_path).stem}<br>"
                  f"<sup style='color:#666'>Voxel size: "
                  f"{voxel_size[0]:.1f} mm | "
                  f"Grid: {voxel_labels.shape} | "
                  f"Organs: {len(present_labels)}</sup>"),
            x=0.5,
            font=dict(size=18, family="Arial, Helvetica, sans-serif"),
        ),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            bgcolor="#fafafa",
            xaxis=dict(backgroundcolor="#f0f0f0", gridcolor="#ddd",
                       showbackground=True, zeroline=False),
            yaxis=dict(backgroundcolor="#f0f0f0", gridcolor="#ddd",
                       showbackground=True, zeroline=False),
            zaxis=dict(backgroundcolor="#f0f0f0", gridcolor="#ddd",
                       showbackground=True, zeroline=False),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(
            title="<b>Organs</b>  (click to toggle)",
            groupclick="togglegroup",
            itemsizing="constant",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
        updatemenus=[
            # Group presets dropdown
            dict(
                type="dropdown",
                direction="down",
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
                buttons=preset_buttons,
                bgcolor="white",
                bordercolor="#aaa",
                font=dict(size=12),
                pad=dict(r=10, t=10),
            ),
            # Opacity dropdown
            dict(
                type="dropdown",
                direction="down",
                x=0.01, y=0.88,
                xanchor="left", yanchor="top",
                buttons=opacity_steps,
                bgcolor="white",
                bordercolor="#aaa",
                font=dict(size=12),
                pad=dict(r=10, t=10),
                active=4,  # default = 1.0
            ),
        ],
        annotations=[
            dict(text="<b>View Preset:</b>", x=0.01, y=1.02,
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12)),
            dict(text="<b>Opacity:</b>", x=0.01, y=0.91,
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12)),
        ],
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=80, b=10),
        height=850,
    )

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="3D Interactive Organ Visualization (Plotly)")
    parser.add_argument(
        "--file", type=str,
        default="test-output/train/BDMAP_00000001.npz",
        help="Path to .npz voxel file")
    parser.add_argument(
        "--downsample", type=int, default=1,
        help="Spatial downsample factor (1=full, 2=half res for speed)")
    parser.add_argument(
        "--opacity", type=float, default=1.0,
        help="Initial mesh opacity (0.0-1.0)")
    parser.add_argument(
        "--no-body-comp", action="store_true",
        help="Hide body composition (fat/muscle) on startup for cleaner view")
    parser.add_argument(
        "--save-html", type=str, default=None,
        help="Save as standalone HTML file instead of opening browser")
    args = parser.parse_args()

    show_groups = None
    if args.no_body_comp:
        show_groups = [g for g in ORGAN_GROUPS if g != "Body Composition"]

    print(f"Loading {args.file} ...")
    fig = build_figure(
        args.file,
        downsample=args.downsample,
        show_groups=show_groups,
        opacity=args.opacity,
    )

    if args.save_html:
        fig.write_html(args.save_html, include_plotlyjs="cdn")
        print(f"Saved -> {args.save_html}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
