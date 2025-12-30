import os
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

# Reuse core discovery + layout/crop logic from the CLI implementation.
import compare


@dataclass
class _Session:
    comparator: compare.InteractiveCropComparator
    source: str
    root: str
    group: str
    dataset: str
    pair: str
    structure: str
    effective_structure: str
    reference_key: str
    pending_point: Optional[Tuple[int, int]] = None


def _parse_bg_color(layout_bg_color: str) -> tuple:
    # Return RGBA tuple or a transparent-like default.
    try:
        if isinstance(layout_bg_color, str) and layout_bg_color.strip().lower() != "transparent":
            parts = [int(x) for x in layout_bg_color.replace(' ', '').split(',') if x != '']
            if len(parts) not in (3, 4):
                raise ValueError
            return tuple(max(0, min(255, v)) for v in parts)
    except Exception:
        pass
    return (0, 0, 0, 0)


def _bgr_to_rgb(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _next_available_roi_id(rois: Dict[int, Dict[str, Any]]) -> int:
    used = {int(k) for k in (rois or {}).keys() if isinstance(k, int) or str(k).isdigit()}
    n = 1
    while n in used:
        n += 1
    return n


def _draw_rois_on_bgr(
    img_bgr: np.ndarray,
    rois: Dict[int, Dict[str, Any]],
    thickness: int = 2,
    active_roi: Optional[int] = None,
    cmp_: Optional[compare.InteractiveCropComparator] = None,
) -> np.ndarray:
    out = img_bgr.copy()
    for rid in sorted(rois.keys()):
        rect = rois[rid].get("rect")
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        color = tuple(int(c) for c in rois[rid].get("color", (0, 0, 255)))

        # Prefer reusing compare.py drawing helpers when available.
        if cmp_ is not None and hasattr(cmp_, "draw_dashed_rect") and getattr(cmp_, "mode", "selection") == "selection" and active_roi == rid:
            try:
                cmp_.draw_dashed_rect(out, (x1, y1), (x2, y2), color, thickness=max(1, int(thickness)))
            except Exception:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, int(thickness)))
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, int(thickness)))

        # Match compare.py label positions:
        # - active ROI: circled label near (x1+20, max(20, y1-20))
        # - inactive ROI: text near (x1+3, max(0, y1-5))
        if active_roi is not None and rid == active_roi:
            cx = int(x1 + 20)
            cy = int(max(20, y1 - 20))
            if cmp_ is not None and hasattr(cmp_, "draw_circled_label"):
                try:
                    cmp_.draw_circled_label(out, (cx, cy), rid, color)
                except Exception:
                    pass
            else:
                radius = 12
                cv2.circle(out, (cx, cy), radius, color, thickness=max(1, int(thickness)))
                cv2.circle(out, (cx, cy), radius - 2, (0, 0, 0), thickness=-1)
                text = str(rid)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(
                    out,
                    text,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
        else:
            cv2.putText(out, f"{rid}", (x1 + 3, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def _list_methods_from_root(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    names = []
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p) and not d.startswith('.'):
            names.append(d)
    return sorted(names)


def _load_methods_from_timeline(path: str = "./timeline_methods.txt") -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.read().splitlines() if line.strip()]


def _build_input_folder(
    source: str,
    root: str,
    group: str,
    dataset: str,
    pair: str,
    structure: str,
) -> Dict[str, Any]:
    methods = _load_methods_from_timeline("./timeline_methods.txt")
    if source == "external":
        if not methods:
            raise ValueError("timeline_methods.txt is required for external source")
        input_folder = {m: f"/data/user/results/{m}/{dataset}/pred/{pair}" for m in methods}
        return input_folder

    if not methods:
        methods = _list_methods_from_root(root)

    input_folder = compare.discover_local_inputs(root, methods, group=group, dataset=dataset, pair=pair, structure=structure)
    if not input_folder:
        raise ValueError(
            "No valid method inputs found. Checked layouts: "
            "<method>/<group>/<dataset>[/<pair>], <method>/<dataset>, <method>/, "
            "and shared-folder layout (image-id folders containing per-method files)."
        )
    return input_folder


def _infer_effective_structure(
    source: str,
    root: str,
    structure: str,
    input_folder: Dict[str, Any],
) -> str:
    if source == "external":
        return "external"
    if structure == "shared":
        return "shared"
    try:
        # shared detection: method -> list-of-files
        if any(isinstance(v, list) for v in input_folder.values()):
            return "shared"
    except Exception:
        pass

    try:
        method = next(iter(input_folder.keys()))
        cand = input_folder[method]
        if not isinstance(cand, str):
            return structure
        method_root = os.path.join(root, method)
        rel = os.path.relpath(cand, method_root)
        if rel in (".", ""):
            return "flat"
        parts = [p for p in rel.split(os.sep) if p and p != "."]
        if len(parts) >= 3:
            return "group-dataset-pair"
        if len(parts) == 2:
            return "group-dataset"
        if len(parts) == 1:
            return "dataset-only"
        return "flat"
    except Exception:
        return structure


def _structure_uses_fields(source: str, eff_structure: str) -> Dict[str, bool]:
    # Controls whether fields are relevant to *path resolution*.
    if source == "external":
        return {"root": False, "group": False, "dataset": True, "pair": True, "structure": False}

    s = (eff_structure or "auto").lower()
    if s == "group-dataset-pair":
        return {"root": True, "group": True, "dataset": True, "pair": True, "structure": True}
    if s == "group-dataset":
        return {"root": True, "group": True, "dataset": True, "pair": False, "structure": True}
    if s == "dataset-only":
        return {"root": True, "group": False, "dataset": True, "pair": False, "structure": True}
    if s == "flat":
        return {"root": True, "group": False, "dataset": False, "pair": False, "structure": True}
    if s == "shared":
        return {"root": True, "group": False, "dataset": False, "pair": False, "structure": True}
    # auto/unknown: keep everything editable (except pair, which is only meaningful for group-dataset-pair)
    return {"root": True, "group": True, "dataset": True, "pair": True, "structure": True}


def ui_update_param_relevance(
    sess: Optional[_Session],
    source: str,
    root: str,
    group: str,
    dataset: str,
    pair: str,
    structure: str,
) -> Tuple[gr.update, gr.update, gr.update, gr.update, gr.update]:
    # Determine effective structure for UI hinting.
    eff = structure
    if source == "external":
        eff = "external"
    elif structure == "auto":
        # On initial page load (no session yet), avoid any filesystem inference.
        if sess is None:
            eff = "auto"
        elif sess is not None and sess.source == source and sess.root == root and sess.group == group and sess.dataset == dataset and sess.pair == pair and sess.structure == structure:
            eff = sess.effective_structure
        else:
            # Best-effort inference (may be slow on large folders; safe fallback).
            try:
                inputs = _build_input_folder(source, root, group, dataset, pair, structure)
                eff = _infer_effective_structure(source, root, structure, inputs)
            except Exception:
                eff = "auto"

    uses = _structure_uses_fields(source, eff)
    # Gray out by disabling the field (matches inactive input styling).
    return (
        gr.update(interactive=bool(uses.get("root", True))),
        gr.update(interactive=bool(uses.get("group", True))),
        gr.update(interactive=bool(uses.get("dataset", True))),
        gr.update(interactive=bool(uses.get("pair", True))),
        gr.update(interactive=bool(uses.get("structure", True))),
    )


def _frame_choices(sess: _Session) -> List[str]:
    files = sess.comparator.image_files.get(sess.reference_key, [])
    return [os.path.basename(p) for p in files]


def _roi_table_from_comparator(cmp_: compare.InteractiveCropComparator) -> List[List[int]]:
    rows: List[List[int]] = []
    for rid in sorted(cmp_.rois.keys()):
        rect = cmp_.rois[rid].get("rect")
        if rect is None:
            rows.append([rid, 0, 0, 0, 0])
        else:
            x1, y1, x2, y2 = rect
            rows.append([rid, int(x1), int(y1), int(x2), int(y2)])
    return rows


def _apply_roi_table_to_comparator(
    cmp_: compare.InteractiveCropComparator,
    table: Any,
) -> None:
    def _to_rows(obj: Any) -> List[List[Any]]:
        if obj is None:
            return []
        if isinstance(obj, list):
            return obj
        # Gradio Dataframe may return a pandas.DataFrame depending on version.
        try:
            # pandas.DataFrame has .values
            values = getattr(obj, "values", None)
            if values is not None:
                return values.tolist()
        except Exception:
            pass
        try:
            to_numpy = getattr(obj, "to_numpy", None)
            if callable(to_numpy):
                return to_numpy().tolist()
        except Exception:
            pass
        return []

    def _safe_int(v: Any, default: int = 0) -> int:
        if v is None:
            return default
        try:
            if isinstance(v, float) and np.isnan(v):
                return default
        except Exception:
            pass
        try:
            return int(v)
        except Exception:
            return default

    rois: Dict[int, Dict[str, Any]] = {}
    rows = _to_rows(table)
    for row in rows:
        if row is None or len(row) < 5:
            continue
        rid = _safe_int(row[0], default=0)
        if rid <= 0:
            continue
        x1, y1, x2, y2 = (
            _safe_int(row[1]),
            _safe_int(row[2]),
            _safe_int(row[3]),
            _safe_int(row[4]),
        )
        color = cmp_.color_for_id(rid)
        rect = None
        if any(v != 0 for v in (x1, y1, x2, y2)):
            rect = (x1, y1, x2, y2)
        rois[rid] = {"rect": rect, "color": color}
    cmp_.rois = rois
    cmp_.active_roi = max(rois.keys()) if rois else None
    cmp_.selection_start = None


def _render_outputs(sess: _Session, preview_key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[List[int]]]:
    cmp_ = sess.comparator
    ref = cmp_.read_frame(sess.reference_key, cmp_.current_frame)
    if ref is None:
        return None, None, None, _roi_table_from_comparator(cmp_)

    ref_overlay = _draw_rois_on_bgr(ref, cmp_.rois, thickness=cmp_.line_thickness, active_roi=cmp_.active_roi, cmp_=cmp_)
    grid = cmp_.build_grid()
    # In the CLI, display_scale/magnify is used when showing the OpenCV window.
    # In Gradio, we need to explicitly scale the rendered grid image.
    if grid is not None:
        try:
            dscale = float(getattr(cmp_, "display_scale", 1.0) or 1.0)
            if dscale > 0 and dscale != 1.0:
                grid = cv2.resize(grid, dsize=None, fx=dscale, fy=dscale, interpolation=cv2.INTER_NEAREST)
        except Exception:
            pass
    final = cmp_.build_final_layout_for_key(preview_key or sess.reference_key, sort_mode=cmp_.sort_mode, reverse_sort=cmp_.sort_reverse)

    return _bgr_to_rgb(ref_overlay), _bgr_to_rgb(grid) if grid is not None else None, _bgr_to_rgb(final) if final is not None else None, _roi_table_from_comparator(cmp_)


def ui_load(
    source: str,
    root: str,
    group: str,
    dataset: str,
    pair: str,
    structure: str,
    output_dir: str,
    columns: int,
    grid_gap: int,
    magnify: float,
    thickness: int,
    layout: str,
    layout_gap: int,
    layout_border_scale: float,
    layout_bg_color: str,
) -> Tuple[Any, ...]:
    input_folder = _build_input_folder(source, root, group, dataset, pair, structure)

    # Parse background color
    bg = _parse_bg_color(layout_bg_color)

    # Instantiate comparator (no OpenCV windows used in Gradio mode)
    cmp_ = compare.InteractiveCropComparator(
        input_folder,
        output_folder=output_dir or "./.crop_grid/",
        reference_key=('GT' if 'GT' in input_folder else ('input' if 'input' in input_folder else next(iter(input_folder.keys())))),
        columns=columns,
        grid_gap=grid_gap,
        display_scale=magnify,
        line_thickness=thickness,
        layout_border_scale=layout_border_scale,
        layout_gap=layout_gap,
        layout_bg_color=bg,
        current_group=group,
        current_dataset=dataset,
    )
    cmp_.layout_mode = layout
    cmp_.sort_mode = 'id'
    cmp_.sort_reverse = False
    cmp_.mode = 'selection'
    if len(cmp_.rois) == 0:
        cmp_.add_roi(1)

    sess = _Session(
        comparator=cmp_,
        source=source,
        root=root,
        group=group,
        dataset=dataset,
        pair=pair,
        structure=structure,
        effective_structure=_infer_effective_structure(source, root, structure, input_folder),
        reference_key=cmp_.reference_key,
        pending_point=None,
    )

    frames = _frame_choices(sess)
    methods = sorted(list(cmp_.image_files.keys()))
    preview_default = 'GT' if 'GT' in methods else (sess.reference_key if sess.reference_key in methods else methods[0])

    ref_img, grid_img, final_img, roi_table = _render_outputs(sess, preview_default)
    status = f"Loaded {len(methods)} methods, {len(frames)} frames (reference={sess.reference_key})."

    root_u, group_u, dataset_u, pair_u, structure_u = ui_update_param_relevance(sess, source, root, group, dataset, pair, structure)

    # After a successful load: keep Load clickable but make it non-primary; enable Save as primary.
    load_btn_u = gr.Button(value="Load", interactive=True, variant="secondary")
    save_btn_u = gr.Button(value="Save Outputs", interactive=True, variant="primary")

    # ROI panel state
    set_btn_u = gr.Button(value="Set ROI", variant="primary" if cmp_.mode == "selection" else "secondary")
    move_btn_u = gr.Button(value="Move ROI", variant="primary" if cmp_.mode == "position" else "secondary")
    active_roi_u = gr.Number(value=int(cmp_.active_roi) if cmp_.active_roi is not None else 0, precision=0)
    add_roi_id_u = gr.Number(value=_next_available_roi_id(cmp_.rois), precision=0)
    return (
        sess,
        gr.Dropdown(choices=frames, value=frames[0] if frames else None),
        gr.Dropdown(choices=methods, value=preview_default),
        ref_img,
        grid_img,
        final_img,
        roi_table,
        status,
        root_u,
        group_u,
        dataset_u,
        pair_u,
        structure_u,
        load_btn_u,
        save_btn_u,
        set_btn_u,
        move_btn_u,
        active_roi_u,
        add_roi_id_u,
    )


def _mode_button_updates(mode: str) -> Tuple[gr.Button, gr.Button]:
    m = (mode or "selection").lower()
    if m not in ("selection", "position"):
        m = "selection"
    set_btn_u = gr.Button(value="Set ROI", variant="primary" if m == "selection" else "secondary")
    move_btn_u = gr.Button(value="Move ROI", variant="primary" if m == "position" else "secondary")
    return set_btn_u, move_btn_u


def ui_set_mode(sess: _Session, mode: str) -> Tuple[Optional[np.ndarray], str, gr.Button, gr.Button, gr.Number]:
    if sess is None:
        set_u, move_u = _mode_button_updates("selection")
        return None, "Session not loaded yet. Click Load first.", set_u, move_u, gr.Number(value=0, precision=0)

    m = (mode or "selection").lower()
    if m not in ("selection", "position"):
        m = "selection"
    sess.comparator.mode = m
    # Cancel any partial click if user switches modes.
    sess.pending_point = None

    set_u, move_u = _mode_button_updates(m)
    active = sess.comparator.active_roi

    ref = sess.comparator.read_frame(sess.reference_key, sess.comparator.current_frame)
    ref_overlay = None
    if ref is not None:
        ref_overlay = _draw_rois_on_bgr(
            ref,
            sess.comparator.rois,
            thickness=sess.comparator.line_thickness,
            active_roi=sess.comparator.active_roi,
            cmp_=sess.comparator,
        )

    return (
        _bgr_to_rgb(ref_overlay) if ref_overlay is not None else None,
        f"Switched mode to: {m}",
        set_u,
        move_u,
        gr.Number(value=int(active) if active is not None else 0, precision=0),
    )


def ui_set_active_roi(sess: _Session, rid: Any, preview_key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], gr.Number]:
    if sess is None:
        return None, None, gr.Number(value=0, precision=0)

    try:
        rid_i = int(rid)
    except Exception:
        rid_i = 0

    if rid_i > 0 and rid_i in sess.comparator.rois:
        sess.comparator.active_roi = rid_i
    else:
        # If invalid, keep current
        rid_i = int(sess.comparator.active_roi) if sess.comparator.active_roi is not None else 0

    sess.pending_point = None
    ref = sess.comparator.read_frame(sess.reference_key, sess.comparator.current_frame)
    ref_overlay = None
    if ref is not None:
        ref_overlay = _draw_rois_on_bgr(ref, sess.comparator.rois, thickness=sess.comparator.line_thickness, active_roi=sess.comparator.active_roi, cmp_=sess.comparator)
    final = sess.comparator.build_final_layout_for_key(preview_key or sess.reference_key, sort_mode=sess.comparator.sort_mode, reverse_sort=sess.comparator.sort_reverse)
    return (
        _bgr_to_rgb(ref_overlay) if ref_overlay is not None else None,
        _bgr_to_rgb(final) if final is not None else None,
        gr.Number(value=rid_i, precision=0),
    )


def ui_apply_grid_settings(
    sess: _Session,
    columns: int,
    grid_gap: int,
    magnify: float,
) -> Tuple[Optional[np.ndarray], str]:
    if sess is None:
        return None, "Session not loaded yet. Click Load first."

    cmp_ = sess.comparator
    try:
        cmp_.columns = max(int(columns), 1)
    except Exception:
        pass
    try:
        cmp_.grid_gap = int(grid_gap)
    except Exception:
        pass
    try:
        if hasattr(cmp_, "display_scale"):
            cmp_.display_scale = float(magnify)
    except Exception:
        pass

    grid = cmp_.build_grid()
    if grid is not None:
        try:
            dscale = float(getattr(cmp_, "display_scale", 1.0) or 1.0)
            if dscale > 0 and dscale != 1.0:
                grid = cv2.resize(grid, dsize=None, fx=dscale, fy=dscale, interpolation=cv2.INTER_NEAREST)
        except Exception:
            pass
    return (_bgr_to_rgb(grid) if grid is not None else None), "Updated Per-ROI Method Grid."


def ui_apply_final_settings(
    sess: _Session,
    preview_key: str,
    layout: str,
    layout_gap: int,
    layout_border_scale: float,
    layout_bg_color: str,
) -> Tuple[Optional[np.ndarray], str]:
    if sess is None:
        return None, "Session not loaded yet. Click Load first."

    cmp_ = sess.comparator
    try:
        cmp_.layout_mode = layout
    except Exception:
        pass
    try:
        cmp_.layout_gap = int(layout_gap)
    except Exception:
        pass
    try:
        cmp_.layout_border_scale = float(layout_border_scale)
    except Exception:
        pass
    try:
        cmp_.layout_bg_color = _parse_bg_color(layout_bg_color)
    except Exception:
        pass

    final = cmp_.build_final_layout_for_key(preview_key or sess.reference_key, sort_mode=cmp_.sort_mode, reverse_sort=cmp_.sort_reverse)
    return (_bgr_to_rgb(final) if final is not None else None), "Updated Final Layout Preview."


def ui_apply_reference_settings(
    sess: _Session,
    preview_key: str,
    thickness: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if sess is None:
        return None, None, "Session not loaded yet. Click Load first."

    cmp_ = sess.comparator
    try:
        cmp_.line_thickness = max(int(thickness), 1)
    except Exception:
        pass

    ref = cmp_.read_frame(sess.reference_key, cmp_.current_frame)
    ref_overlay = None
    if ref is not None:
        ref_overlay = _draw_rois_on_bgr(ref, cmp_.rois, thickness=cmp_.line_thickness, active_roi=cmp_.active_roi, cmp_=cmp_)

    final = cmp_.build_final_layout_for_key(preview_key or sess.reference_key, sort_mode=cmp_.sort_mode, reverse_sort=cmp_.sort_reverse)
    return (
        _bgr_to_rgb(ref_overlay) if ref_overlay is not None else None,
        _bgr_to_rgb(final) if final is not None else None,
        "Updated Reference + ROIs (and Final Layout Preview).",
    )


def ui_set_preview_key(sess: _Session, preview_key: str) -> Tuple[Optional[np.ndarray], str]:
    if sess is None:
        return None, "Session not loaded yet. Click Load first."
    cmp_ = sess.comparator
    final = cmp_.build_final_layout_for_key(preview_key or sess.reference_key, sort_mode=cmp_.sort_mode, reverse_sort=cmp_.sort_reverse)
    return (_bgr_to_rgb(final) if final is not None else None), f"Preview method: {preview_key}" if preview_key else "Updated preview method."


def ui_apply_settings(
    sess: _Session,
    preview_key: str,
    columns: int,
    grid_gap: int,
    magnify: float,
    thickness: int,
    layout: str,
    layout_gap: int,
    layout_border_scale: float,
    layout_bg_color: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[List[int]], str]:
    if sess is None:
        return None, None, None, [], "Session not loaded yet. Click Load first."

    cmp_ = sess.comparator
    try:
        cmp_.columns = max(int(columns), 1)
    except Exception:
        pass
    try:
        cmp_.grid_gap = int(grid_gap)
    except Exception:
        pass
    try:
        # display_scale is used for grid magnification
        if hasattr(cmp_, "display_scale"):
            cmp_.display_scale = float(magnify)
    except Exception:
        pass
    try:
        cmp_.line_thickness = max(int(thickness), 1)
    except Exception:
        pass
    try:
        cmp_.layout_mode = layout
    except Exception:
        pass
    try:
        cmp_.layout_gap = int(layout_gap)
    except Exception:
        pass
    try:
        cmp_.layout_border_scale = float(layout_border_scale)
    except Exception:
        pass
    try:
        cmp_.layout_bg_color = _parse_bg_color(layout_bg_color)
    except Exception:
        pass

    ref_img, grid_img, final_img, roi_table = _render_outputs(sess, preview_key)
    return ref_img, grid_img, final_img, roi_table, "Applied layout/advanced settings."


def ui_set_frame(sess: _Session, frame_name: str, preview_key: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List[List[int]], str]:
    files = sess.comparator.image_files.get(sess.reference_key, [])
    if not files:
        return None, None, None, _roi_table_from_comparator(sess.comparator), "No frames loaded."
    idx = 0
    for i, p in enumerate(files):
        if os.path.basename(p) == frame_name:
            idx = i
            break
    sess.comparator.current_frame = idx
    ref_img, grid_img, final_img, roi_table = _render_outputs(sess, preview_key)
    return ref_img, grid_img, final_img, roi_table, f"Frame: {frame_name}"


def ui_update_from_table(sess: _Session, roi_table: Any, preview_key: str) -> Tuple[Any, ...]:
    if sess is None:
        return None, None, None, [], "Session not loaded yet. Click Load first.", gr.Number(value=0, precision=0), gr.Number(value=1, precision=0)

    def _to_rows(obj: Any) -> List[List[Any]]:
        if obj is None:
            return []
        if isinstance(obj, list):
            return obj
        # Gradio Dataframe may return a pandas.DataFrame depending on version.
        try:
            values = getattr(obj, "values", None)
            if values is not None:
                return values.tolist()
        except Exception:
            pass
        try:
            to_numpy = getattr(obj, "to_numpy", None)
            if callable(to_numpy):
                return to_numpy().tolist()
        except Exception:
            pass
        return []

    def _safe_int(v: Any, default: int = 0) -> int:
        if v is None:
            return default
        try:
            if isinstance(v, float) and np.isnan(v):
                return default
        except Exception:
            pass
        try:
            return int(v)
        except Exception:
            return default

    def _normalize_table(obj: Any) -> List[List[int]]:
        out: List[List[int]] = []
        for row in _to_rows(obj):
            if row is None or len(row) < 5:
                continue
            rid = _safe_int(row[0], default=0)
            if rid <= 0:
                continue
            x1 = _safe_int(row[1])
            y1 = _safe_int(row[2])
            x2 = _safe_int(row[3])
            y2 = _safe_int(row[4])
            out.append([rid, x1, y1, x2, y2])
        out.sort(key=lambda r: r[0])
        return out

    # Gradio can fire roi_table.change even when roi_table is updated programmatically
    # (e.g. Add ROI or image click handler returns a new table), which can cause a
    # second render. If the incoming table already matches the current comparator
    # state, treat this as a no-op.
    if _normalize_table(roi_table) == _normalize_table(_roi_table_from_comparator(sess.comparator)):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    _apply_roi_table_to_comparator(sess.comparator, roi_table)
    ref_img, grid_img, final_img, roi_table2 = _render_outputs(sess, preview_key)
    active = sess.comparator.active_roi
    return (
        ref_img,
        grid_img,
        final_img,
        roi_table2,
        "Updated ROIs.",
        gr.Number(value=int(active) if active is not None else 0, precision=0),
        gr.Number(value=_next_available_roi_id(sess.comparator.rois), precision=0),
    )


def ui_add_roi(sess: _Session, roi_id: int, preview_key: str) -> Tuple[Any, ...]:
    if sess is None:
        set_u, move_u = _mode_button_updates("selection")
        return None, None, None, [], "Session not loaded yet. Click Load first.", gr.Number(value=0, precision=0), set_u, move_u, gr.Number(value=1, precision=0)
    # In the CLI, add_roi(None) may re-select an existing empty ROI.
    # In Gradio, the expectation for the button is: always add a *new* ROI.
    requested = None
    try:
        requested = int(roi_id)
    except Exception:
        requested = None

    used = set(sess.comparator.rois.keys())
    if requested is None or requested <= 0:
        rid = (max(used) + 1) if used else 1
    else:
        if requested in used:
            rid = max(used) + 1
        else:
            rid = requested

    sess.comparator.add_roi(roi_id=rid)
    # Requirement: Add ROI forces selection mode and Set ROI button becomes active.
    sess.comparator.mode = "selection"
    ref_img, grid_img, final_img, roi_table = _render_outputs(sess, preview_key)
    active = sess.comparator.active_roi
    set_u, move_u = _mode_button_updates(sess.comparator.mode)
    next_id = _next_available_roi_id(sess.comparator.rois)
    return (
        ref_img,
        grid_img,
        final_img,
        roi_table,
        "Added ROI. Click twice to define its rectangle, or edit the table.",
        gr.Number(value=int(active) if active is not None else 0, precision=0),
        set_u,
        move_u,
        gr.Number(value=next_id, precision=0),
    )


def ui_clear_rois(sess: _Session, preview_key: str) -> Tuple[Any, ...]:
    if sess is None:
        set_u, move_u = _mode_button_updates("selection")
        return None, None, None, [], "Session not loaded yet. Click Load first.", gr.Number(value=0, precision=0), set_u, move_u, gr.Number(value=1, precision=0)
    sess.comparator.rois = {}
    sess.comparator.active_roi = None
    sess.comparator.selection_start = None
    sess.pending_point = None
    ref_img, grid_img, final_img, roi_table = _render_outputs(sess, preview_key)
    # Clearing resets mode back to selection for predictable next action.
    sess.comparator.mode = "selection"
    set_u, move_u = _mode_button_updates(sess.comparator.mode)
    return ref_img, grid_img, final_img, roi_table, "Cleared all ROIs.", gr.Number(value=0, precision=0), set_u, move_u, gr.Number(value=1, precision=0)


def ui_click_image(sess: _Session, evt: gr.SelectData, preview_key: str) -> Tuple[Any, ...]:
    if sess is None:
        set_u, move_u = _mode_button_updates("selection")
        return None, None, None, [], "Session not loaded yet. Click Load first.", gr.Number(value=0, precision=0), set_u, move_u, gr.Number(value=1, precision=0)
    x, y = int(evt.index[0]), int(evt.index[1])

    # Position mode: single click moves the active ROI while keeping its size.
    if getattr(sess.comparator, "mode", "selection") == "position":
        sess.pending_point = None
        rid = sess.comparator.active_roi
        if rid is None or rid not in sess.comparator.rois or sess.comparator.rois[rid].get("rect") is None:
            ref_img, grid_img, final_img, roi_table = _render_outputs(sess, preview_key)
            active = sess.comparator.active_roi
            set_u, move_u = _mode_button_updates(sess.comparator.mode)
            return (
                ref_img,
                grid_img,
                final_img,
                roi_table,
                "Position mode: no active ROI rectangle to move.",
                gr.Number(value=int(active) if active is not None else 0, precision=0),
                set_u,
                move_u,
                gr.Number(value=_next_available_roi_id(sess.comparator.rois), precision=0),
            )

        ref = sess.comparator.read_frame(sess.reference_key, sess.comparator.current_frame)
        h_img, w_img = (ref.shape[0], ref.shape[1]) if ref is not None else (None, None)
        x1, y1, x2, y2 = sess.comparator.rois[rid]["rect"]
        w = abs(int(x2) - int(x1))
        h = abs(int(y2) - int(y1))
        new_x1 = int(x - w // 2)
        new_y1 = int(y - h // 2)
        new_x2 = int(new_x1 + w)
        new_y2 = int(new_y1 + h)
        if w_img is not None and h_img is not None:
            new_x1 = max(0, min(w_img - 1, new_x1))
            new_y1 = max(0, min(h_img - 1, new_y1))
            new_x2 = max(0, min(w_img - 1, new_x2))
            new_y2 = max(0, min(h_img - 1, new_y2))
        sess.comparator.rois[rid]["rect"] = (new_x1, new_y1, new_x2, new_y2)
        ref_img, grid_img, final_img, roi_table = _render_outputs(sess, preview_key)
        active = sess.comparator.active_roi
        set_u, move_u = _mode_button_updates(sess.comparator.mode)
        return (
            ref_img,
            grid_img,
            final_img,
            roi_table,
            f"Moved ROI {rid} to center ({x}, {y}).",
            gr.Number(value=int(active) if active is not None else 0, precision=0),
            set_u,
            move_u,
            gr.Number(value=_next_available_roi_id(sess.comparator.rois), precision=0),
        )
    if sess.pending_point is None:
        sess.pending_point = (x, y)
        ref_img, grid_img, final_img, roi_table = _render_outputs(sess, preview_key)
        active = sess.comparator.active_roi
        set_u, move_u = _mode_button_updates(sess.comparator.mode)
        return (
            ref_img,
            grid_img,
            final_img,
            roi_table,
            f"Start point set at ({x}, {y}). Click again to finish ROI.",
            gr.Number(value=int(active) if active is not None else 0, precision=0),
            set_u,
            move_u,
            gr.Number(value=_next_available_roi_id(sess.comparator.rois), precision=0),
        )

    sx, sy = sess.pending_point
    sess.pending_point = None

    # Ensure there is an active ROI to write into
    if sess.comparator.active_roi is None or sess.comparator.active_roi not in sess.comparator.rois:
        sess.comparator.add_roi()

    rid = sess.comparator.active_roi
    sess.comparator.rois[rid]["rect"] = (sx, sy, x, y)

    # After a rectangle is defined (two clicks), switch to position mode
    # so the next click can move the active ROI immediately.
    sess.comparator.mode = "position"

    ref_img, grid_img, final_img, roi_table = _render_outputs(sess, preview_key)
    active = sess.comparator.active_roi
    set_u, move_u = _mode_button_updates(sess.comparator.mode)
    return (
        ref_img,
        grid_img,
        final_img,
        roi_table,
        f"ROI {rid} set to ({sx}, {sy})-({x}, {y}). Switched to position mode.",
        gr.Number(value=int(active) if active is not None else 0, precision=0),
        set_u,
        move_u,
        gr.Number(value=_next_available_roi_id(sess.comparator.rois), precision=0),
    )


def ui_save(sess: _Session, save_label: str) -> str:
    if sess is None:
        return "Session not loaded yet. Click Load first."
    label = save_label.strip() if isinstance(save_label, str) and save_label.strip() else (sess.pair if sess.source == 'external' else sess.dataset)
    sess.comparator.save(label, sess.dataset)
    out_dir = os.path.join(sess.comparator.output_folder, sess.comparator.save_session_ts or "")
    return f"Saved. Output root: {out_dir}"


def build_demo() -> gr.Blocks:
    def _supports_param(obj: Any, name: str) -> bool:
        try:
            return name in inspect.signature(obj).parameters
        except Exception:
            pass
        try:
            return name in inspect.signature(getattr(obj, "__init__")).parameters
        except Exception:
            return False

    def _mk(component_ctor: Any, info: Optional[str] = None, **kwargs):
        # Prefer Gradio's built-in hover help if available.
        if info:
            if _supports_param(component_ctor, "info"):
                kwargs["info"] = info
            elif _supports_param(component_ctor, "tooltip"):
                kwargs["tooltip"] = info
        return component_ctor(**kwargs)

    css = """
    /* Force disabled input text to render as grey (instead of keeping normal text color). */
    .gradio-container input:disabled,
    .gradio-container textarea:disabled,
    .gradio-container select:disabled {
        color: #9ca3af !important;
        -webkit-text-fill-color: #9ca3af !important;
        opacity: 1 !important;
    }
    /* Red hint text below the reference image. */
    .roi-click-hint {
        color: #ef4444;
        font-weight: 600;
    }

    /* Hide upload/replace UI for the reference preview image (read-only dataset image). */
    #ref-image input[type='file'],
    #ref-image button[aria-label*='Upload'],
    #ref-image button[aria-label*='upload'],
    #ref-image .upload-button,
    #ref-image .image-upload,
    #ref-image [data-testid='image-upload'],
    #ref-image [data-testid='upload-button'] {
        display: none !important;
    }

    /* Full-width buttons when requested. */
    .cc-fullwidth button {
        width: 100% !important;
    }

    /* Status/log box under the Load Dataset panel. */
    #cc-status {
        max-height: 8.5em;
        overflow: auto;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid rgba(0, 0, 0, 0.10);
        background: rgba(0, 0, 0, 0.03);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.9em;
        line-height: 1.35;
    }

    /* Hide ROI table completely (kept for internal syncing). */
    #roi-table {
        display: none !important;
    }
    """

    with gr.Blocks(title="Image Crop Comparator (Gradio)", css=css) as demo:
        gr.Markdown(
"""
<div style='text-align:center'>
    <h1>✨ Image Crop Comparator (Gradio)</h1>
    <p>A minimal Gradio UI for the CropComparer CLI.</p>
</div>
"""
        )

        sess_state = gr.State(value=None)

        # --- Input/Dataset Loading Panel (boxed) ---
        with gr.Group():
            with gr.Accordion("📂 Load Dataset", open=True):
                gr.Markdown("**Input**")
                with gr.Row():
                    source = _mk(
                        gr.Dropdown,
                        choices=["local", "external"],
                        value="local",
                        label="Source",
                        info="local: load from workspace root; external: expects /data/user/... and requires timeline_methods.txt",
                    )
                    root = _mk(
                        gr.Textbox,
                        value="./examples",
                        label="Root (local)",
                        info="Workspace root containing per-method folders (local mode only).",
                    )
                    structure = _mk(
                        gr.Dropdown,
                        choices=["auto", "group-dataset-pair", "group-dataset", "dataset-only", "flat", "shared"],
                        value="auto",
                        label="Structure",
                        info="Folder layout under root (auto tries multiple layouts).",
                    )

                with gr.Row():
                    group = _mk(gr.Textbox, value="SDSD-indoor+", label="Group", info="Group folder under each method (local mode).")
                    dataset = _mk(gr.Textbox, value="SDSD-indoor", label="Dataset", info="Dataset folder name; also used for saving label.")
                    pair = _mk(gr.Textbox, value="pair13", label="Pair (external)", info="Sequence name for external datasets (ignored in local mode).")

                with gr.Row():
                    load_btn = _mk(
                        gr.Button,
                        value="Load",
                        variant="primary",
                        info="Discover inputs and initialize the session.",
                        elem_classes=["cc-fullwidth"],
                    )

                with gr.Row():
                    frame_dd = _mk(gr.Dropdown, choices=[], value=None, label="Frame", info="Select which image/frame to preview.")
                    preview_key_dd = _mk(gr.Dropdown, choices=[], value=None, label="Preview Method", info="Which method to render as the final layout preview.")

            # Log/output area sits directly under Load Dataset.
            status = gr.Markdown("", elem_id="cc-status")

        # --- Reference + ROI Controls (side-by-side) ---
        with gr.Row():
            with gr.Column(scale=1):
                # Read-only preview (no upload), but still clickable via .select for ROI points.
                ref_img_kwargs: Dict[str, Any] = dict(
                    label="Reference + ROIs",
                    interactive=False, # Disable direct image edits; we handle clicks via .select
                    elem_id="ref-image",
                )
                if _supports_param(gr.Image, "show_download_button"):
                    ref_img_kwargs["show_download_button"] = False
                if _supports_param(gr.Image, "show_share_button"):
                    ref_img_kwargs["show_share_button"] = False
                if _supports_param(gr.Image, "sources"):
                    # Extra safety: prevent the component from advertising any image sources.
                    ref_img_kwargs["sources"] = []

                ref_img = _mk(
                    gr.Image,
                    info="This is a dataset preview (not an upload input). Click twice on the image to set an ROI.",
                    **ref_img_kwargs,
                )

            with gr.Column(scale=1):
                with gr.Group():
                    with gr.Accordion("🎮 ROI Controls", open=True):
                        active_roi_id = _mk(
                            gr.Number,
                            value=0,
                            precision=0,
                            label="Active ROI id",
                            interactive=True,
                            info="Change active ROI id; the preview will highlight that ROI.",
                        )
                        with gr.Row():
                            set_roi_btn = _mk(gr.Button, value="Set ROI", variant="primary", info="Selection mode: click twice to set ROI rectangle.")
                            move_roi_btn = _mk(gr.Button, value="Move ROI", variant="secondary", info="Position mode: click once to move active ROI (keeps size).")

                        add_roi_id = _mk(
                            gr.Number,
                            value=1,
                            precision=0,
                            label="Add ROI id",
                            info="Auto-advances to the next available id after adding.",
                        )
                        with gr.Row():
                            add_btn = _mk(gr.Button, value="Add ROI", variant="secondary", info="Add a new ROI entry (then click on the reference image).", elem_classes=["cc-fullwidth"])
                            clear_btn = _mk(gr.Button, value="Clear ROIs", variant="secondary", info="Remove all ROIs.", elem_classes=["cc-fullwidth"])

                gr.Markdown("<div class='roi-click-hint' style='text-align: center;'>Click twice on the image to set ROIs.</div>")

        # Keep ROI table as a hidden component (used for syncing state in callbacks).
        df_kwargs: Dict[str, Any] = dict(
            headers=["id", "x1", "y1", "x2", "y2"],
            datatype=["number"] * 5,
            elem_id="roi-table",
        )
        if _supports_param(gr.Dataframe, "visible"):
            df_kwargs["visible"] = False
        if _supports_param(gr.Dataframe, "row_count"):
            df_kwargs["row_count"] = (3, "dynamic")
        if _supports_param(gr.Dataframe, "col_count"):
            df_kwargs["col_count"] = (5, "fixed")
        if _supports_param(gr.Dataframe, "height"):
            df_kwargs["height"] = 220
        roi_table = _mk(
            gr.Dataframe,
            info="(Hidden) ROI table for internal syncing.",
            **df_kwargs,
        )

        # --- Final Layout Preview (below Reference + Controls) ---
        with gr.Row():
            grid_img = gr.Image(label="Per-ROI Method Grid")

        with gr.Row():
            with gr.Column(scale=1):
                final_img = gr.Image(label="Final Layout Preview")

            with gr.Column(scale=1):
                # Settings panels under the preview images.
                with gr.Group():
                    with gr.Accordion("🧩 Per-ROI Method Grid Settings", open=False):
                        with gr.Row():
                            columns = _mk(gr.Slider, minimum=1, maximum=12, value=6, step=1, label="Columns", info="Number of columns in the per-ROI method grid.")
                            grid_gap = _mk(gr.Slider, minimum=0, maximum=20, value=2, step=1, label="Grid Gap", info="Pixel gap between tiles in the method grid.")
                            magnify = _mk(gr.Slider, minimum=0.25, maximum=8.0, value=2.0, step=0.25, label="Magnify", info="Scale the Per-ROI Method Grid display (Gradio-only).")

                    with gr.Accordion("🖼️ Reference + ROIs Settings", open=False):
                        thickness = _mk(gr.Slider, minimum=1, maximum=12, value=2, step=1, label="ROI Thickness", info="Thickness of ROI rectangles in the reference preview.")

                    with gr.Accordion("🧷 Final Layout Preview Settings", open=True):
                        with gr.Row():
                            layout = _mk(gr.Dropdown, choices=["left", "top", "right", "bottom"], value="right", label="Layout", info="Where crops are placed relative to the base image.")
                            layout_bg_color = _mk(gr.Textbox, value="transparent", label="Layout BG Color", info="Padding color between blocks: transparent or R,G,B[,A].")
                        with gr.Row():
                            layout_gap = _mk(gr.Slider, minimum=0, maximum=50, value=10, step=1, label="Layout Gap", info="Spacing between base image and crops (and between crops).")
                            layout_border_scale = _mk(gr.Slider, minimum=0.5, maximum=6.0, value=2.0, step=0.1, label="Layout Border Scale", info="Border thickness multiplier for crop blocks in final preview.")

        # (ROIs panel moved under Controls)
        with gr.Group():
            gr.Markdown("**💾 Export**")
            with gr.Row():
                output_dir = _mk(gr.Textbox, value="./.crop_grid/", label="Output Dir", info="Export root directory. Save Outputs writes results under this folder.")
                save_label = _mk(gr.Textbox, value="", label="Save Label (optional)", info="Override save label (defaults to dataset in local mode, pair in external mode).")
            with gr.Row():
                # Full-width action button
                save_btn = _mk(
                    gr.Button,
                    value="Save Outputs",
                    variant="secondary",
                    interactive=False,
                    info="Write orig/final/crops to Output Dir.",
                )
            save_status = gr.Markdown("")

        # Wiring
        load_btn.click(
            fn=ui_load,
            inputs=[
                source,
                root,
                group,
                dataset,
                pair,
                structure,
                output_dir,
                columns,
                grid_gap,
                magnify,
                thickness,
                layout,
                layout_gap,
                layout_border_scale,
                layout_bg_color,
            ],
            outputs=[
                sess_state,
                frame_dd,
                preview_key_dd,
                ref_img,
                grid_img,
                final_img,
                roi_table,
                status,
                root,
                group,
                dataset,
                pair,
                structure,
                load_btn,
                save_btn,
                set_roi_btn,
                move_roi_btn,
                active_roi_id,
                add_roi_id,
            ],
        )

        # Gray out irrelevant inputs based on effective structure.
        demo.load(
            fn=ui_update_param_relevance,
            inputs=[sess_state, source, root, group, dataset, pair, structure],
            outputs=[root, group, dataset, pair, structure],
        )
        source.change(
            fn=ui_update_param_relevance,
            inputs=[sess_state, source, root, group, dataset, pair, structure],
            outputs=[root, group, dataset, pair, structure],
        )
        structure.change(
            fn=ui_update_param_relevance,
            inputs=[sess_state, source, root, group, dataset, pair, structure],
            outputs=[root, group, dataset, pair, structure],
        )

        # Settings updates are scoped to their preview panels.
        # - Grid settings: update grid only
        columns.release(
            fn=ui_apply_grid_settings,
            inputs=[sess_state, columns, grid_gap, magnify],
            outputs=[grid_img, status],
        )
        grid_gap.release(
            fn=ui_apply_grid_settings,
            inputs=[sess_state, columns, grid_gap, magnify],
            outputs=[grid_img, status],
        )
        magnify.release(
            fn=ui_apply_grid_settings,
            inputs=[sess_state, columns, grid_gap, magnify],
            outputs=[grid_img, status],
        )

        # - Final layout settings: update final preview only
        layout.change(
            fn=ui_apply_final_settings,
            inputs=[sess_state, preview_key_dd, layout, layout_gap, layout_border_scale, layout_bg_color],
            outputs=[final_img, status],
        )
        layout_bg_color.change(
            fn=ui_apply_final_settings,
            inputs=[sess_state, preview_key_dd, layout, layout_gap, layout_border_scale, layout_bg_color],
            outputs=[final_img, status],
        )
        layout_gap.release(
            fn=ui_apply_final_settings,
            inputs=[sess_state, preview_key_dd, layout, layout_gap, layout_border_scale, layout_bg_color],
            outputs=[final_img, status],
        )
        layout_border_scale.release(
            fn=ui_apply_final_settings,
            inputs=[sess_state, preview_key_dd, layout, layout_gap, layout_border_scale, layout_bg_color],
            outputs=[final_img, status],
        )

        # - Reference settings: update reference + final preview
        thickness.release(
            fn=ui_apply_reference_settings,
            inputs=[sess_state, preview_key_dd, thickness],
            outputs=[ref_img, final_img, status],
        )

        frame_dd.change(
            fn=ui_set_frame,
            inputs=[sess_state, frame_dd, preview_key_dd],
            outputs=[ref_img, grid_img, final_img, roi_table, status],
        )
        preview_key_dd.change(
            fn=ui_set_preview_key,
            inputs=[sess_state, preview_key_dd],
            outputs=[final_img, status],
        )

        roi_table.change(
            fn=ui_update_from_table,
            inputs=[sess_state, roi_table, preview_key_dd],
            outputs=[ref_img, grid_img, final_img, roi_table, status, active_roi_id, add_roi_id],
        )

        add_btn.click(
            fn=ui_add_roi,
            inputs=[sess_state, add_roi_id, preview_key_dd],
            outputs=[ref_img, grid_img, final_img, roi_table, status, active_roi_id, set_roi_btn, move_roi_btn, add_roi_id],
        )

        clear_btn.click(
            fn=ui_clear_rois,
            inputs=[sess_state, preview_key_dd],
            outputs=[ref_img, grid_img, final_img, roi_table, status, active_roi_id, set_roi_btn, move_roi_btn, add_roi_id],
        )

        ref_img.select(
            fn=ui_click_image,
            inputs=[sess_state, preview_key_dd],
            outputs=[ref_img, grid_img, final_img, roi_table, status, active_roi_id, set_roi_btn, move_roi_btn, add_roi_id],
        )

        active_roi_id.change(
            fn=ui_set_active_roi,
            inputs=[sess_state, active_roi_id, preview_key_dd],
            outputs=[ref_img, final_img, active_roi_id],
        )

        # Mode buttons
        set_roi_btn.click(
            fn=lambda s: ui_set_mode(s, "selection"),
            inputs=[sess_state],
            outputs=[ref_img, status, set_roi_btn, move_roi_btn, active_roi_id],
        )
        move_roi_btn.click(
            fn=lambda s: ui_set_mode(s, "position"),
            inputs=[sess_state],
            outputs=[ref_img, status, set_roi_btn, move_roi_btn, active_roi_id],
        )

        save_btn.click(
            fn=ui_save,
            inputs=[sess_state, save_label],
            outputs=[save_status],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
