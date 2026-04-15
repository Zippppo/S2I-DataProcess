"""
3D Medical Voxel Visualization — BDMAP Organ Segmentation

使用 Plotly 交互式可视化 voxel_labels 数据
- Marching Cubes 提取器官表面网格
- 按解剖分组归类，支持分组 / 单器官开关
- 科研配色方案（Nature / Lancet 风格）

Usage:
    python visualize_voxel_3d.py [--file PATH] [--downsample N]
    python scripts/visualize_voxel_3d.py --file test-output\train\BDMAP_00000001.npz 
    

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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.organ_mapping import (
    INSIDE_BODY_EMPTY_LABEL,
    LABEL_TO_ORGAN,
    NUM_CLASSES,
    OUTSIDE_BODY_BACKGROUND_LABEL,
)

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

# ---------------------------------------------------------------------------
# 配色：黄金角度色相分散，保证每个 label 颜色差异最大化
# ---------------------------------------------------------------------------
_GOLDEN_ANGLE = 137.508


def _color_for_label(label_id: int) -> str:
    """Assign a visually distinct color per label using golden-angle hue spacing."""
    hue = (label_id * _GOLDEN_ANGLE) % 360
    return f"hsl({hue:.0f}, 78%, 55%)"


def validate_labels(voxel_labels: np.ndarray) -> dict:
    """
    Validate voxel label array and print diagnostics.

    Checks:
    - Label 0 (inside_body_empty) and 255 (outside_body_background) exist
    - All labels are within expected range (0-122 or 255)
    - Reports any unexpected labels

    Returns summary dict with label statistics.
    """
    unique, counts = np.unique(voxel_labels, return_counts=True)
    total = voxel_labels.size
    label_counts = dict(zip(unique.tolist(), counts.tolist()))

    n_inside_empty = label_counts.get(INSIDE_BODY_EMPTY_LABEL, 0)
    n_outside_bg = label_counts.get(OUTSIDE_BODY_BACKGROUND_LABEL, 0)
    organ_labels = {l for l in unique if l != INSIDE_BODY_EMPTY_LABEL
                    and l != OUTSIDE_BODY_BACKGROUND_LABEL}
    valid_organ_range = set(range(1, NUM_CLASSES))  # 1-122
    unexpected = organ_labels - valid_organ_range

    n_organ = sum(label_counts[l] for l in organ_labels)

    print("=" * 60)
    print("  Label Validation Report")
    print("=" * 60)
    print(f"  Grid shape     : {voxel_labels.shape}")
    print(f"  Total voxels   : {total:,}")
    print(f"  Unique labels  : {len(unique)}")
    print("-" * 60)
    print(f"  [Label   0] inside_body_empty   : {n_inside_empty:>10,}  "
          f"({n_inside_empty/total*100:5.1f}%)")
    print(f"  [Label 255] outside_body_bg      : {n_outside_bg:>10,}  "
          f"({n_outside_bg/total*100:5.1f}%)")
    print(f"  [Label 1-122] organ voxels       : {n_organ:>10,}  "
          f"({n_organ/total*100:5.1f}%)")
    print(f"  Organ types present              : {len(organ_labels)}")
    print("-" * 60)

    ok = True
    if n_inside_empty == 0:
        print("  [WARN] Label 0 (inside_body_empty) not found!")
        ok = False
    if n_outside_bg == 0:
        print("  [WARN] Label 255 (outside_body_background) not found!")
        ok = False
    if unexpected:
        print(f"  [ERROR] Unexpected labels outside [0-122, 255]: {sorted(unexpected)}")
        ok = False

    if ok:
        print("  [OK] All labels valid.")
    print("=" * 60)

    return {
        "total": total,
        "n_inside_empty": n_inside_empty,
        "n_outside_bg": n_outside_bg,
        "n_organ": n_organ,
        "organ_labels": sorted(organ_labels),
        "unexpected": sorted(unexpected),
        "label_counts": label_counts,
    }


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
                 opacity: float = 1.0,
                 show_skin: bool = True,
                 pc_downsample: int = 1) -> go.Figure:
    """
    Build an interactive 3D Plotly figure.

    Parameters
    ----------
    npz_path : path to the .npz file
    downsample : spatial downsample factor (1 = full res, 2 = half, …)
    show_groups : list of group names to show initially (None = all)
    opacity : mesh opacity (1.0 = fully opaque)
    show_skin : whether to display the sensor_pc skin point cloud
    pc_downsample : point cloud downsample factor (1 = all points)
    """
    data = np.load(npz_path)
    voxel_labels = data["voxel_labels"]
    world_min = data["grid_world_min"]
    voxel_size = data["grid_voxel_size"]
    sensor_pc = data["sensor_pc"]

    # Validate labels (0=inside empty, 255=outside bg, 1-122=organs)
    report = validate_labels(voxel_labels)
    if report["unexpected"]:
        print(f"[WARN] Proceeding with {len(report['unexpected'])} unexpected labels")

    present_labels = set(report["organ_labels"])

    traces = []
    trace_groups = []  # group name per trace, for visibility presets
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
            color = _color_for_label(lid)

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
            trace_groups.append(group_name)

        # group toggle button
        if group_trace_indices:
            buttons_group.append(
                dict(label=group_name, method="restyle",
                     args=[{"visible": True}, group_trace_indices]),
            )

    # --- Skin (sensor_pc) point cloud trace ---
    skin_trace_idx = None
    if sensor_pc is not None and len(sensor_pc) > 0:
        pc = sensor_pc[::pc_downsample]
        skin_trace = go.Scatter3d(
            x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
            mode='markers',
            marker=dict(size=1, color='rgba(180,180,180,0.3)'),
            name=f'Skin (Input, {len(pc):,} pts)',
            legendgroup='Skin (Input)',
            legendgrouptitle_text='Skin (Input)',
            hovertemplate=(
                "<b>Skin point</b><br>"
                "x: %{x:.1f} mm<br>y: %{y:.1f} mm<br>z: %{z:.1f} mm"
                "<extra></extra>"
            ),
            visible=True if show_skin else "legendonly",
        )
        skin_trace_idx = len(traces)
        traces.append(skin_trace)
        trace_groups.append('Skin (Input)')

    if not traces:
        raise RuntimeError("No organ meshes could be extracted — data may be empty.")

    fig = go.Figure(data=traces)

    # ----- Dropdown: group visibility presets -----
    n = len(traces)

    def _visibility_for(groups_on):
        return [True if g in groups_on else "legendonly" for g in trace_groups]

    preset_buttons = [
        dict(label="All + Skin",
             method="restyle",
             args=[{"visible": [True] * n}]),
        dict(label="All Organs",
             method="restyle",
             args=[{"visible": _visibility_for(
                 set(ORGAN_GROUPS.keys()))}]),
        dict(label="Solid Organs Only",
             method="restyle",
             args=[{"visible": _visibility_for({"Solid Organs"})}]),
        dict(label="Skeleton + Skin",
             method="restyle",
             args=[{"visible": _visibility_for(
                 {"Vertebrae", "Left Ribs", "Right Ribs", "Other Bones",
                  "Skin (Input)"})}]),
        dict(label="Thorax (Lungs + Heart + Ribs)",
             method="restyle",
             args=[{"visible": _visibility_for(
                 {"Lung Lobes", "Solid Organs", "Left Ribs", "Right Ribs",
                  "Other Bones"})}]),
        dict(label="Vasculature",
             method="restyle",
             args=[{"visible": _visibility_for(
                 {"Arteries", "Veins & Cardiac"})}]),
        dict(label="Skin Only",
             method="restyle",
             args=[{"visible": _visibility_for({"Skin (Input)"})}]),
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
                  f"Organs: {len(present_labels)} | "
                  f"Skin pts: {len(sensor_pc):,} | "
                  f"Inside empty: {report['n_inside_empty']/report['total']*100:.1f}% | "
                  f"Outside bg: {report['n_outside_bg']/report['total']*100:.1f}%</sup>"),
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
        "--no-skin", action="store_true",
        help="Hide skin point cloud on startup")
    parser.add_argument(
        "--pc-downsample", type=int, default=1,
        help="Point cloud downsample factor (1=all points, 4=every 4th point)")
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
        show_skin=not args.no_skin,
        pc_downsample=args.pc_downsample,
    )

    if args.save_html:
        fig.write_html(args.save_html, include_plotlyjs="cdn")
        print(f"Saved -> {args.save_html}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
