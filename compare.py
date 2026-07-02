import copy
import concurrent.futures
import math
import os
import sys
import traceback

import threading
import time
import warnings
from datetime import datetime
from glob import glob

import cv2
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from utils.metrics.pyiqa_metrics import IQAs as _IQAs
    from utils.metrics.pyiqa_metrics import compute_iqa_metrics_multi_thread as _compute_iqa_metrics_multi_thread
except Exception:  # pragma: no cover - optional dependency
    print(f"Warning: IQAs not available, metric computations will be disabled")
    traceback.print_exc()
    _IQAs = None
    _compute_iqa_metrics_multi_thread = None

try:
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter, AutoMinorLocator
except Exception:  # pragma: no cover - optional dependency
    plt = None
    MaxNLocator = None
    FormatStrFormatter = None
    AutoMinorLocator = None

from typing import Union, Tuple

try:
    from natsort import natsorted as _natsorted
except Exception:  # pragma: no cover - optional dependency
    _natsorted = sorted

IMG_EXTS = ['png', 'jpg', 'jpeg', 'bmp', 'ppm']
INPUT_IDLE_UPDATE_SEC = 0.4


def is_hidden_path(path):
    if '.me' in path:
        return False
    parts = os.path.normpath(path or '').split(os.sep)
    return any(part.startswith('.') for part in parts if part not in ('', '.', '..'))


def filter_hidden(paths):
    return [p for p in paths if not is_hidden_path(p)]


sys.path.append('.')
sys.path.append('..')

try:
    from utils.logger import Logger, Color

    log = Logger()
except Exception:
    # Fallback logger if utils.logger is not available
    class _FallbackLogger:
        def __init__(self):
            pass

        def debug(self, msg):
            print(msg)

        def info(self, msg):
            print(msg)

        def success(self, msg):
            print(msg)

        def warn(self, msg):
            print(msg)

        def error(self, msg):
            print(msg)

        def note(self, msg):
            print(msg)

        def banner(self, title):
            print(title)

        def set_color_enabled(self, enabled):
            pass

        def set_level(self, level):
            pass


    log = _FallbackLogger()

try:
    from utils.io import glob_single_files, read_images_as_numpy
except Exception:
    from PIL import Image  # noqa: F401


    class PathHandler:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        @staticmethod
        def get_vanilla_path(path):
            return path

        @staticmethod
        def get_basename(path):
            return os.path.basename(path)

        @staticmethod
        def remove_extension(path):
            filename, extension = os.path.splitext(path)
            return filename

        def get_dir_removed_path(self, path):
            return os.path.relpath(path, self.dirname)


    def glob_single_files(directory, file_extensions, path_handler=PathHandler.get_vanilla_path):
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]

        file_paths = []

        directory = glob.escape(directory)

        if os.path.isdir(directory):
            for file_extension in file_extensions:
                pattern = os.path.join(directory, f"**/*.{file_extension}")
                file_paths.extend(glob.glob(pattern, recursive=True))
        else:
            for file_extension in file_extensions:
                pattern = f"{directory}*.{file_extension}"
                file_paths.extend(glob.glob(pattern))

        file_paths = _natsorted(set(os.path.normpath(p) for p in file_paths))
        file_paths = [path_handler(p) for p in file_paths]
        return file_paths


    def read_images_as_numpy(*paths, read_mode=cv2.IMREAD_UNCHANGED):
        if isinstance(paths, str):
            paths = [paths]
        images = []
        for path in paths:
            image = cv2.imread(path, read_mode)
            if image is None:
                raise FileNotFoundError(f'Failed to read image: "{path}"')
            if image.ndim == 2:
                image = np.expand_dims(image, axis=2)
            elif image.shape[2] > 3:
                image = image[:, :, :3]
            images.append(image)
        return images if len(images) > 1 else images[0]


def emphasize(text):
    # Blink emphasize; prefer logger colors in normal usage
    return f"\033[5m{text}\033[0m"


class EventDispatcher:
    """Simple event dispatcher to register and dispatch handlers."""

    def __init__(self):
        self.handlers = {}

    def register(self, key_code, handler):
        self.handlers.setdefault(key_code, []).append(handler)

    def dispatch(self, key_code, *args, **kwargs):
        handled = False
        for h in self.handlers.get(key_code, []):
            h(*args, **kwargs)
            handled = True
        return handled


class UndoManager:
    """Manage undoable actions using a stack of callables."""

    def __init__(self, capacity=200):
        self.undo_stack = []
        self.redo_stack = []
        self.capacity = capacity

    def record(self, undo_fn, redo_fn=None, desc=""):
        if undo_fn is None:
            return
        self.undo_stack.append((undo_fn, redo_fn, desc))
        if len(self.undo_stack) > self.capacity:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return False, None
        undo_fn, redo_fn, desc = self.undo_stack.pop()
        try:
            undo_fn()
        except Exception:
            return False, desc
        if redo_fn is not None:
            self.redo_stack.append((redo_fn, undo_fn, desc))
        return True, desc

    def redo(self):
        if not self.redo_stack:
            return False, None
        redo_fn, undo_fn, desc = self.redo_stack.pop()
        try:
            redo_fn()
        except Exception:
            return False, desc
        self.undo_stack.append((undo_fn, redo_fn, desc))
        return True, desc


def has_images(directory, exts=None):
    exts = exts or IMG_EXTS
    try:
        return len(filter_hidden(glob_single_files(directory, exts))) > 0
    except Exception:
        return False


def has_images_direct(directory, exts=None):
    """Check whether a directory contains image files directly (non-recursive)."""
    exts = exts or IMG_EXTS
    if not directory or not os.path.isdir(directory):
        return False
    ext_set = {str(e).lower().lstrip('.') for e in exts}
    try:
        for name in os.listdir(directory):
            if name.startswith('.'):
                continue
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lstrip('.').lower()
            if ext in ext_set:
                return True
    except Exception:
        return False
    return False


def discover_first_dataset_path(method_root, exts=None):
    """Find the first dataset-like folder with direct image files.

    Search order:
      1) <method>/<dataset>
      2) <method>/<group>/<dataset>
    """
    exts = exts or IMG_EXTS
    if not method_root or not os.path.isdir(method_root):
        return None

    try:
        level1_dirs = [
            d for d in sorted(os.listdir(method_root))
            if (not d.startswith('.')) and os.path.isdir(os.path.join(method_root, d))
        ]
    except Exception:
        return None

    for d1 in level1_dirs:
        p1 = os.path.join(method_root, d1)
        if has_images_direct(p1, exts=exts):
            return p1

    for d1 in level1_dirs:
        p1 = os.path.join(method_root, d1)
        try:
            level2_dirs = [
                d for d in sorted(os.listdir(p1))
                if (not d.startswith('.')) and os.path.isdir(os.path.join(p1, d))
            ]
        except Exception:
            continue
        for d2 in level2_dirs:
            p2 = os.path.join(p1, d2)
            if has_images_direct(p2, exts=exts):
                return p2

    return None


def resolve_group_folder(method_root, target_group):
    """Resolve a group folder allowing hyphen/underscore mismatch."""
    tg = (target_group or '').replace('-', '').replace('_', '')
    if not tg:
        return None
    try:
        for d in os.listdir(method_root):
            if d.startswith('.'):
                continue
            full = os.path.join(method_root, d)
            if not os.path.isdir(full):
                continue
            if d.replace('-', '').replace('_', '') == tg:
                return d
    except Exception:
        return None
    return None


def discover_method_path(method_root, group=None, dataset=None, pair=None, structure='auto'):
    """Return the first path that contains images for a method.

    structure choices (align with README):
      - auto: try all known layouts in order
      - group-dataset-pair: <method>/<group>/<dataset>/<pair>
      - group-dataset: <method>/<group>/<dataset>
      - dataset-only: <method>/<dataset>
      - flat: <method>/ (images directly under method)
    """
    candidates = []
    resolved_group = resolve_group_folder(method_root, group) if group else None
    group_name = resolved_group or group
    if structure == 'group-dataset-pair':
        if group_name and dataset and pair:
            candidates.append(os.path.join(method_root, group_name, dataset, pair))
    elif structure == 'group-dataset':
        if group_name and dataset:
            candidates.append(os.path.join(method_root, group_name, dataset))
    elif structure == 'dataset-only':
        if dataset:
            candidates.append(os.path.join(method_root, dataset))
    elif structure == 'flat':
        candidates.append(method_root)
    else:  # auto
        if group_name and dataset and pair:
            candidates.append(os.path.join(method_root, group_name, dataset, pair))
        if group_name and dataset:
            candidates.append(os.path.join(method_root, group_name, dataset))
        if dataset:
            candidates.append(os.path.join(method_root, dataset))
        if not group and not dataset and not pair:
            auto_dataset = discover_first_dataset_path(method_root)
            if auto_dataset:
                candidates.append(auto_dataset)
        candidates.append(method_root)

    for cand in candidates:
        if cand and os.path.isdir(cand) and has_images(cand):
            return cand
    return None


def discover_shared_folder_methods(root):
    """Handle layout where each image folder contains <method>.png/.jpg files.

    Example:
        root/
          img1/
            method1.png
            method2.png
          img2/
            method1.png
            method2.png
    Returns a mapping of method -> ordered list of files.
    """
    if not os.path.isdir(root):
        return {}
    subdirs = [d for d in os.listdir(root) if not d.startswith('.') and os.path.isdir(os.path.join(root, d))]
    subdirs = sorted(subdirs)
    if not subdirs:
        return {}

    method_names = None
    for sd in subdirs:
        cur = os.path.join(root, sd)
        files = [f for f in os.listdir(cur) if not f.startswith('.') and os.path.isfile(os.path.join(cur, f))]
        files = [f for f in files if os.path.splitext(f)[1].lstrip('.').lower() in IMG_EXTS]
        if not files:
            continue
        names = [os.path.splitext(f)[0] for f in files]
        if method_names is None:
            method_names = set(names)
        else:
            method_names |= set(names)

    if not method_names:
        return {}

    out = {}
    for method in sorted(method_names):
        paths = []
        for sd in subdirs:
            cur = os.path.join(root, sd)
            pattern = os.path.join(cur, f"{method}.*")
            matches = _natsorted([p for p in glob(pattern) if os.path.splitext(p)[1].lstrip('.').lower() in IMG_EXTS])
            paths.extend(matches)
        if paths:
            out[method] = paths
    return out


def discover_local_inputs(root, methods, group=None, dataset=None, pair=None, structure='auto'):
    """Return mapping of method -> folder (or file list) for local mode."""
    if structure == 'shared':
        return discover_shared_folder_methods(root)

    inputs = {}
    for m in methods:
        method_root = os.path.join(root, m)
        cand = discover_method_path(method_root, group=group, dataset=dataset, pair=pair, structure=structure)
        if cand:
            inputs[m] = cand
    if inputs:
        return inputs

    # fallback: shared-folder layout (methods are file names under per-image dirs)
    shared = discover_shared_folder_methods(root)
    if shared:
        return shared
    return {}


class AsyncMetricFeature:
    """Optional metric feature module, attached through event_on_init."""

    def __init__(self, host, refresh_interval_sec=0.5, max_workers=4, metric_type='psnr', threads_per_methods=16,
                 metric_runner=None, metric_result_cache=None):
        self.host = host
        # TODO: 支持任意指标的拓展接口，使用 register 编写接口以及 args 选择自定义的指标计算
        self.metric_type = str(metric_type or 'psnr').strip()
        self.metric_key = self.metric_type.split(' ')[0]
        metric_upper = self.metric_key.upper()
        self.window_psnr = f"Metric Curves ({metric_upper})"
        self.host.window_psnr = self.window_psnr
        self.metric_display_name = metric_upper
        self._threads_per_methods = max(1, int(threads_per_methods))
        self._shared_metric_runner = metric_runner
        self._metric_result_cache = metric_result_cache

        self._plot_image = None
        self._plot_image_base = None
        self._plot_line_meta = None
        self._cache_token = None
        self._dirty = True
        self._warned = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._max_workers = max_workers
        self._render_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_futures = {}
        self._pending_token = None
        self._cancel_event = threading.Event()
        self.values = {}
        self._values_n = 0
        self._values_dataset = None
        self._values_ref = None
        self._last_refresh_ts = 0.0
        self._refresh_interval_sec = float(refresh_interval_sec)
        self._legend_hitboxes = []
        self._highlight_methods = set()
        self._psnr_mouse_bound = False
        self._last_render_frame_idx = -1
        self._last_render_highlight = set()
        self._psnr_jump_armed = False
        self._completed_signal_count = 0
        self._render_future = None
        self._render_staged_request = None
        self._render_seq = 0
        self._completed_since_last_render = 0
        self._mpl_cache = None
        self._metric_runner = None
        self._enqueue_thread = None
        self._planned_tasks = None
        self._init_metric_runner()

        # Rendering is done in a worker thread on purpose; suppress this noisy warning.
        warnings.filterwarnings(
            "ignore",
            message="Starting a Matplotlib GUI outside of the main thread will likely fail.*",
            category=UserWarning,
        )

        self.host.register_event_handler('after_update_display', self.on_after_update_display)
        self.host.register_event_handler('on_rebuild_dataset', self.on_rebuild_dataset)
        self.host.register_event_handler('on_shutdown', self.on_shutdown)
        self._start_async_compute()

    def _on_psnr_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Two-click jump mode:
        # 1) click near current dashed line to arm.
        # 2) click again on plot to jump to target index.
        if self._point_in_plot_area(x, y):
            if self._psnr_jump_armed:
                target_idx = self._x_to_frame_index(x)
                self._psnr_jump_armed = False
                if target_idx is not None:
                    self._jump_to_frame_index(target_idx)
                self._dirty = True
                self.host.request_update()
                return

            marker_x = self._current_marker_x_px()
            if marker_x is not None and abs(int(x) - int(marker_x)) <= 6:
                self._psnr_jump_armed = True
                self._dirty = True
                self.host.request_update()
                log.note(f"{self.metric_display_name} jump armed: click target x on plot to jump frame")
                return

        if self._psnr_jump_armed and (not self._point_in_plot_area(x, y)):
            self._psnr_jump_armed = False
            self._dirty = True
            self.host.request_update()

        # In plot area, always prioritize curve picking to avoid legend near-hit stealing.
        if self._point_in_plot_area(x, y):
            clicked_method = self._pick_method_on_plot(x, y)
        else:
            clicked_method = self._pick_method_on_legend(x, y)
            if clicked_method is None:
                clicked_method = self._pick_method_on_plot(x, y)
        if clicked_method is None:
            if self._highlight_methods:
                self._highlight_methods.clear()
                self._dirty = True
                self.host.request_update()
            return

        shift_on = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        if shift_on:
            if clicked_method in self._highlight_methods:
                self._highlight_methods.remove(clicked_method)
            else:
                self._highlight_methods.add(clicked_method)
        else:
            if len(self._highlight_methods) == 1 and clicked_method in self._highlight_methods:
                self._highlight_methods.clear()
            else:
                self._highlight_methods = {clicked_method}

        self._dirty = True
        self.host.request_update()

    def _point_in_plot_area(self, x, y):
        meta = self._plot_line_meta or {}
        x_min = int(meta.get('x_min', 0))
        x_max = int(meta.get('x_max', -1))
        y_min = int(meta.get('y_min', 0))
        y_max = int(meta.get('y_max', -1))
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min
        return x_min <= int(x) <= x_max and y_min <= int(y) <= y_max

    def _x_to_frame_index(self, x):
        meta = self._plot_line_meta or {}
        n = int(meta.get('n', 0))
        if n <= 0:
            return None
        x_min = int(meta.get('x_min', 0))
        x_max = int(meta.get('x_max', -1))
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if x_max <= x_min:
            return 0
        ratio_x = float(int(x) - x_min) / float(max(1, x_max - x_min))
        idx = int(round(max(0.0, min(1.0, ratio_x)) * float(n - 1)))
        return max(0, min(n - 1, idx))

    def _current_marker_x_px(self):
        frame_idx = int(self.host.current_frame)
        meta = self._plot_line_meta or {}
        n = int(meta.get('n', 0))
        if n <= 0:
            return None
        x_min = int(meta.get('x_min', 0))
        x_max = int(meta.get('x_max', -1))
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if n <= 1 or x_max == x_min:
            return x_min
        idx = max(0, min(n - 1, frame_idx))
        ratio = float(idx) / float(n - 1)
        return int(round(x_min + ratio * float(x_max - x_min)))

    def _jump_to_frame_index(self, idx):
        try:
            idx = int(idx)
        except Exception:
            return
        if idx < 0:
            idx = 0

        num_frames = int(self.host.num_frames or 0)
        if num_frames > 0:
            idx = min(num_frames - 1, idx)

        self.host._set_frame(idx)

        try:
            ref_files = self.host.image_files.get(self.host.reference_key, [])
            if 0 <= idx < len(ref_files):
                name = os.path.basename(ref_files[idx])
                log.success(f"Jumped to frame #{idx + 1}: {name}")
            else:
                log.success(f"Jumped to frame #{idx + 1}")
        except Exception:
            log.success(f"Jumped to frame #{idx + 1}")

    def _pick_method_on_legend(self, x, y):
        if not self._legend_hitboxes:
            return None

        best_method = None
        best_dist2 = None
        for hb in self._legend_hitboxes:
            x1 = int(hb.get('x1', 0))
            x2 = int(hb.get('x2', -1))
            y1 = int(hb.get('y1', 0))
            y2 = int(hb.get('y2', -1))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            dx = 0
            if x < x1:
                dx = x1 - x
            elif x > x2:
                dx = x - x2

            dy = 0
            if y < y1:
                dy = y1 - y
            elif y > y2:
                dy = y - y2

            dist2 = dx * dx + dy * dy
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_method = hb.get('method')

        if best_method is None:
            return None

        # Accept exact hit, or near-hit around legend cells.
        # Use tighter vertical tolerance to prevent clicks above legend from snapping to it.
        if best_dist2 == 0:
            return best_method

        best_dist = math.sqrt(float(best_dist2)) if best_dist2 is not None else float('inf')
        if best_dist <= 14:
            best_hb = None
            # Re-find the nearest hb to inspect directional distance constraints.
            nearest_dist2 = None
            for hb in self._legend_hitboxes:
                x1 = int(hb.get('x1', 0))
                x2 = int(hb.get('x2', -1))
                y1 = int(hb.get('y1', 0))
                y2 = int(hb.get('y2', -1))
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1

                dx = 0
                if x < x1:
                    dx = x1 - x
                elif x > x2:
                    dx = x - x2

                dy = 0
                if y < y1:
                    dy = y1 - y
                elif y > y2:
                    dy = y - y2

                d2 = dx * dx + dy * dy
                if nearest_dist2 is None or d2 < nearest_dist2:
                    nearest_dist2 = d2
                    best_hb = (dx, dy)

            if best_hb is not None:
                dx, dy = best_hb
                if dy <= 6 and dx <= 20:
                    return best_method
        return None

    def _pick_method_on_plot(self, x, y):
        meta = self._plot_line_meta or {}
        n = int(meta.get('n', 0))
        if n <= 0:
            return None

        x_min = int(meta.get('x_min', 0))
        x_max = int(meta.get('x_max', -1))
        y_min = int(meta.get('y_min', 0))
        y_max = int(meta.get('y_max', -1))
        if x_max <= x_min or y_max <= y_min:
            return None
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return None

        if n <= 1:
            idx_f = 0.0
        else:
            ratio_x = float(x - x_min) / float(max(1, x_max - x_min))
            idx_f = max(0.0, min(float(n - 1), ratio_x * float(n - 1)))

        y_data_min = float(meta.get('y_data_min', 0.0))
        y_data_max = float(meta.get('y_data_max', 50.0))
        if y_data_max <= y_data_min:
            y_data_min, y_data_max = 0.0, 50.0

        def value_to_y_px(val):
            ratio_y = (float(val) - y_data_min) / float(y_data_max - y_data_min)
            ratio_y = max(0.0, min(1.0, ratio_y))
            return int(round(y_max - ratio_y * float(y_max - y_min)))

        def interp_y_at_index(series, target_idx, search_radius=6):
            if series is None or len(series) == 0:
                return None
            if n <= 1:
                v0 = float(series[0]) if len(series) > 0 else np.nan
                return value_to_y_px(v0) if np.isfinite(v0) else None

            i0 = int(math.floor(target_idx))
            i1 = int(math.ceil(target_idx))
            i0 = max(0, min(len(series) - 1, i0))
            i1 = max(0, min(len(series) - 1, i1))

            left = None
            right = None

            for d in range(search_radius + 1):
                li = i0 - d
                if li >= 0:
                    lv = float(series[li])
                    if np.isfinite(lv):
                        left = (li, lv)
                        break

            for d in range(search_radius + 1):
                ri = i1 + d
                if ri < len(series):
                    rv = float(series[ri])
                    if np.isfinite(rv):
                        right = (ri, rv)
                        break

            if left is None and right is None:
                return None
            if left is None:
                return value_to_y_px(right[1])
            if right is None:
                return value_to_y_px(left[1])

            li, lv = left
            ri, rv = right
            if ri == li:
                return value_to_y_px(lv)

            t = (target_idx - float(li)) / float(ri - li)
            t = max(0.0, min(1.0, t))
            interp_v = lv + (rv - lv) * t
            return value_to_y_px(interp_v)

        best_method = None
        best_dist = float('inf')
        for method_name, series in self.values.items():
            y_px = interp_y_at_index(series, idx_f)
            if y_px is None:
                continue
            dist = abs(int(y) - y_px)
            if dist < best_dist:
                best_dist = dist
                best_method = method_name

        # Keep picking strict to avoid accidental highlight when clicking empty region.
        if best_method is None or best_dist > 16:
            return None
        return best_method

    def _build_cache_token(self):
        items = tuple(sorted((str(k), len(v) if isinstance(v, (list, tuple)) else 0)
                             for k, v in self.host.image_files.items()))
        return (self.host.dataset, self.host.reference_key, items)

    #region ==[Computing]==
    @staticmethod
    def _normalize_for_metric(img):
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _init_metric_runner(self):
        if self._shared_metric_runner is not None:
            self._metric_runner = self._shared_metric_runner
            return
        if _IQAs is None or _compute_iqa_metrics_multi_thread is None:
            log.warn(f"Metric runner not available for '{self.metric_type}': missing dependency")
            return
        try:
            # Use the project-level metric utility for arbitrary metrics.
            # Auto-detect CUDA availability and use GPU if present
            device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                log.info(f"Metric runner using GPU (CUDA) for '{self.metric_type}'")
            self._metric_runner = _IQAs(
                self.metric_type,
                n_threads_per_metric=self._threads_per_methods,
                device=device,
                traditional=True,
            )
        except Exception as e:
            log.warn(f"Metric runner init failed for '{self.metric_type}': {e}")
            traceback.print_exc()
            self._metric_runner = None

    def _compute_one_metric(self, ref_path, cmp_path):
        cache_key = (ref_path, cmp_path)
        if self._metric_result_cache is not None:
            cached_metrics = self._metric_result_cache.get(cache_key)
            if cached_metrics is not None and self.metric_type in cached_metrics:
                return cached_metrics.get(self.metric_type, np.nan)

        ref_img = read_images_as_numpy(ref_path)
        cmp_img = read_images_as_numpy(cmp_path)
        ref_img = AsyncMetricFeature._normalize_for_metric(ref_img)
        cmp_img = AsyncMetricFeature._normalize_for_metric(cmp_img)
        if ref_img is None or cmp_img is None:
            return np.nan
        if cmp_img.shape[:2] != ref_img.shape[:2]:
            cmp_img = cv2.resize(cmp_img, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if self._metric_runner is None:
            # print(f"Warning: metric runner not available; cannot compute {self.metric_display_name}")
            if self.metric_key.lower() == 'psnr':
                return float(cv2.PSNR(cmp_img, ref_img))
            return np.nan

        if torch is None:
            return np.nan

        pred_rgb = cv2.cvtColor(cmp_img, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        pred_t = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        gt_t = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        metrics = self._metric_runner(pred_t, gt_t)
        if self._metric_result_cache is not None:
            self._metric_result_cache[cache_key] = dict(metrics)
        value = metrics.get(self.metric_type)
        if value is None:
            value = metrics.get(self.metric_key)
        if value is None:
            return np.nan
        if hasattr(value, 'item'):
            return float(value.item())
        return float(value)

    def _compute_frame_task(self, token, method_key, ref_path, cmp_path, index):
        """Compute one metric value for a single frame.

        This finer-grained task improves parallelism both across methods and
        within the same method when many frames are present.
        """
        if self._cancel_event.is_set() or token != self._build_cache_token():
            return token, method_key, index, False

        try:
            value = self._compute_one_metric(ref_path, cmp_path)
        except Exception:
            value = np.nan

        series = self.values.get(method_key)
        if series is not None and 0 <= index < len(series):
            series[index] = value

        self._signal_progress(1)
        return token, method_key, index, True

    def _signal_progress(self, count=1):
        if count <= 0:
            return
        self._completed_signal_count += int(count)

    def _on_future_done(self, _fut):
        self._completed_signal_count += 1

    @staticmethod
    def _binary_index_order(n):
        """Return indices in binary-split order: mid, left-half, right-half."""
        n = int(n or 0)
        if n <= 0:
            return []
        order = []
        stack = [(0, n - 1)]
        while stack:
            left, right = stack.pop()
            if left > right:
                continue
            mid = (left + right) // 2
            order.append(mid)
            # Push right first so left half is processed first (LIFO stack).
            if mid + 1 <= right:
                stack.append((mid + 1, right))
            if left <= mid - 1:
                stack.append((left, mid - 1))
        return order

    def _enqueue_metric_tasks(self, token, image_files_snapshot, ref_files, method_keys, method_limits, max_limit):
        planned = []
        for idx in self._binary_index_order(max_limit):
            if self._cancel_event.is_set() or token != self._pending_token:
                return
            for mk in method_keys:
                if self._cancel_event.is_set() or token != self._pending_token:
                    return
                limit = method_limits.get(mk, 0)
                if idx >= limit:
                    continue
                mk_files = image_files_snapshot.get(mk, [])
                planned.append((mk, ref_files[idx], mk_files[idx], idx))
        if self._cancel_event.is_set() or token != self._pending_token:
            return
        self._planned_tasks = planned

    def _submit_planned_metric_tasks(self):
        if self._enqueue_thread is not None and self._enqueue_thread.is_alive():
            return
        if not self._planned_tasks:
            return
        tasks = self._planned_tasks
        self._planned_tasks = None
        token = self._pending_token
        if token is None or self._cancel_event.is_set():
            return
        for mk, ref_path, cmp_path, idx in tasks:
            if self._cancel_event.is_set() or token != self._pending_token:
                return
            fut = self._executor.submit(
                self._compute_frame_task,
                token,
                mk,
                ref_path,
                cmp_path,
                idx,
            )
            self._pending_futures[(mk, idx)] = fut

    def _start_async_compute(self):
        self._cancel_pending()
        self._cancel_event = threading.Event()
        token = self._build_cache_token()

        image_files_snapshot = {
            k: list(v) if isinstance(v, (list, tuple)) else []
            for k, v in self.host.image_files.items()
        }
        ref_key_snapshot = self.host.reference_key
        dataset_snapshot = self.host.dataset
        ref_files = image_files_snapshot.get(ref_key_snapshot, [])
        n = len(ref_files)
        method_keys = [k for k in image_files_snapshot.keys() if k != ref_key_snapshot]

        # Build per-method effective image counts.
        method_limits = {
            mk: min(len(ref_files), len(image_files_snapshot.get(mk, [])))
            for mk in method_keys
        }

        # Auto-detect worker count using total image tasks, not only method count.
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        total_methods = len(method_keys)
        total_tasks = sum(method_limits.values())
        desired_workers = min(max(1, total_methods * self._threads_per_methods), self._max_workers, max(1, total_tasks), cpu_count)
        log.info(f"Desired workers for {self.metric_display_name}: {desired_workers} (methods: {total_methods}, tasks: {total_tasks}, CPU cores: {cpu_count})")

        current_workers = int(getattr(self._executor, '_max_workers', 0) or 0)
        if current_workers != desired_workers:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=desired_workers,
                thread_name_prefix="MetricWorker"
            )

        self._values_dataset = dataset_snapshot
        self._values_ref = ref_key_snapshot
        self._values_n = n
        self.values = {k: [np.nan] * n for k in method_keys}

        self._pending_token = token
        self._cache_token = token
        self._dirty = True
        self._last_refresh_ts = 0.0
        self._completed_since_last_render = 0
        self._plot_image = None
        self._plot_image_base = None
        self._plot_line_meta = None
        self._legend_hitboxes = []

        ps = {k: list(v) for k, v in self.values.items()}
        pn = int(self._values_n)
        pds = self._values_dataset
        pref = self._values_ref
        self._request_render(ps, pn, pds, pref, int(self.host.current_frame), self._highlight_methods)

        # Submit one task per frame in round-robin across methods.
        # This avoids queueing all frames of one method first, which can look
        # like single-method processing in FIFO thread pools.
        self._pending_futures = {}
        self._planned_tasks = None
        max_limit = max(method_limits.values(), default=0)
        self._enqueue_thread = threading.Thread(
            target=self._enqueue_metric_tasks,
            args=(token, image_files_snapshot, ref_files, method_keys, method_limits, max_limit),
            name="MetricEnqueueWorker",
            daemon=True,
        )
        self._enqueue_thread.start()
    #endregion

    #region ==[Render]==
    @staticmethod
    def _draw_vertical_dashed_line(img, x, y1, y2, color, thickness=1, dash_len=6, gap_len=6):
        if img is None:
            return
        h, w = img.shape[:2]
        x = max(0, min(w - 1, int(x)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h - 1, int(y2)))
        if y2 < y1:
            y1, y2 = y2, y1
        pos = y1
        while pos <= y2:
            seg_end = min(y2, pos + dash_len)
            cv2.line(img, (x, pos), (x, seg_end), color, thickness, lineType=cv2.LINE_AA)
            pos += dash_len + gap_len

    def _compose_plot_with_frame_line(self, frame_idx):
        if self._plot_image_base is None:
            return None
        out = self._plot_image_base.copy()
        meta = self._plot_line_meta or {}
        n = int(meta.get('n', 0))
        if n <= 0:
            return out

        x_min = int(meta.get('x_min', 0))
        x_max = int(meta.get('x_max', 0))
        y_min = int(meta.get('y_min', 0))
        y_max = int(meta.get('y_max', out.shape[0] - 1))
        if x_max < x_min:
            x_min, x_max = x_max, x_min

        current_idx = int(frame_idx) + 1
        current_idx = max(1, min(max(1, n), current_idx))
        if n <= 1 or x_max == x_min:
            x_px = x_min
        else:
            ratio = float(current_idx - 1) / float(n - 1)
            x_px = int(round(x_min + ratio * (x_max - x_min)))

        self._draw_vertical_dashed_line(
            out,
            x=x_px,
            y1=y_min,
            y2=y_max,
            color=(64, 64, 255),
            thickness=(3 if self._psnr_jump_armed else 1),
            dash_len=7,
            gap_len=5,
        )
        return out

    def _reset_mpl_cache(self):
        cache = self._mpl_cache
        self._mpl_cache = None
        if not cache:
            return
        fig = cache.get('fig')
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

    def _render_plot_image(self, psnr_series, n, dataset_name, reference_key, current_frame_idx, highlight_methods):
        # ---- 0) Guard clauses: rendering backend and data availability ----
        if plt is None:
            if not self._warned:
                log.warn(f"matplotlib is not available; {self.metric_display_name} curve window is disabled")
                self._warned = True
            return self._loading_canvas(), [], None

        # ---- 1) Figure/canvas baseline configuration ----
        # Render on a larger physical canvas first to avoid tiny-font collapse.
        fig_w_in = 20.0
        plot_h_in = 6.0
        render_dpi = 80
        x = np.arange(1, max(0, int(n)) + 1)
        # ---- 2) Prepare per-method series metadata (values, color, legend label) ----
        cmap = plt.get_cmap('tab20')
        other_idx = 0
        finite_values = []
        series_items = []
        for name, ys in psnr_series.items():
            y_arr = np.array(ys, dtype=float)
            finite = y_arr[np.isfinite(y_arr)]
            if finite.size > 0:
                finite_values.append(finite)

            name_l = str(name).strip().lower()
            if name_l == 'lq':
                color = (0.0, 0.0, 0.0)
            elif name_l == 'gt':
                color = (1.0, 0.0, 0.0)
            else:
                color = cmap(other_idx % 20)
                other_idx += 1
            if isinstance(color, (tuple, list, np.ndarray)) and len(color) >= 3:
                color = (float(color[0]), float(color[1]), float(color[2]))

            legend_name = str(name)
            if len(legend_name) > 18:
                legend_name = legend_name[:15] + '...'
            series_items.append((str(name), legend_name, y_arr, color))

        labels = [item[0] for item in series_items]
        if labels:
            ncol = 8
            legend_rows = int(math.ceil(len(labels) / float(ncol)))
            legend_h_in = max(1.1, 0.34 * legend_rows + 0.28)
        else:
            ncol = 1
            legend_rows = 1
            legend_h_in = 0.45

        layout_key = tuple(item[0] for item in series_items)
        cache = self._mpl_cache
        if cache is None or cache.get('layout_key') != layout_key or cache.get('ncol') != ncol:
            self._reset_mpl_cache()
            fig = plt.figure(figsize=(fig_w_in, plot_h_in + legend_h_in), dpi=render_dpi)
            gs = fig.add_gridspec(2, 1, height_ratios=[plot_h_in, legend_h_in], hspace=0.14)
            plot_ax = fig.add_subplot(gs[0, 0])
            legend_ax = fig.add_subplot(gs[1, 0])

            ds_name = dataset_name if dataset_name else "N/A"
            title_text = f"{self.metric_display_name} Curves on Dataset: {ds_name} (ref: {reference_key})"

            plot_ax.set_ylabel("PSNR (dB)" if self.metric_key.lower() == 'psnr' else self.metric_display_name)
            plot_ax.set_title(title_text, fontsize=14)
            plot_ax.tick_params(axis='x', pad=6, labelsize=10)
            plot_ax.tick_params(axis='y', labelsize=10)
            plot_ax.grid(True, linestyle='--', linewidth=0.1, alpha=1.0)
            empty_text = plot_ax.text(
                0.5,
                0.5,
                f"Waiting for {self.metric_display_name} values...",
                ha='center',
                va='center',
                fontsize=12,
                color=(0.35, 0.35, 0.35),
                transform=plot_ax.transAxes,
                visible=False,
            )
            legend_ax.axis('off')
            legend_ax.set_xlim(0.0, 1.0)
            legend_ax.set_ylim(0.0, 1.0)

            line_artists = {}
            legend_line_artists = {}
            legend_text_artists = {}
            legend_box_artists = {}
            for method_name, legend_name, _y_arr, color in series_items:
                line_obj, = plot_ax.plot([], [], linewidth=0.5, color=color, alpha=1.0, antialiased=True)
                line_artists[method_name] = line_obj

                idx = len(legend_line_artists)
                row = idx // ncol
                col = idx % ncol
                cell_w = 1.0 / float(ncol)
                cell_h = 1.0 / float(max(1, legend_rows))
                x0 = col * cell_w
                y_top = 1.0 - row * cell_h
                y_bottom = y_top - cell_h
                y_mid = (y_top + y_bottom) * 0.5

                l_obj, = legend_ax.plot(
                    [x0 + 0.06 * cell_w, x0 + 0.25 * cell_w],
                    [y_mid, y_mid],
                    color=color,
                    linewidth=1.0,
                    alpha=1.0,
                    solid_capstyle='round',
                    transform=legend_ax.transAxes,
                    clip_on=False,
                )
                txt_obj = legend_ax.text(
                    x0 + 0.30 * cell_w,
                    y_mid,
                    legend_name,
                    ha='left',
                    va='center',
                    fontsize=9,
                    fontweight='normal',
                    transform=legend_ax.transAxes,
                    clip_on=False,
                )
                legend_line_artists[method_name] = l_obj
                legend_text_artists[method_name] = txt_obj

            fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.05, hspace=0.24)
            self._mpl_cache = {
                'fig': fig,
                'plot_ax': plot_ax,
                'legend_ax': legend_ax,
                'line_artists': line_artists,
                'legend_line_artists': legend_line_artists,
                'legend_text_artists': legend_text_artists,
                'layout_key': layout_key,
                'ncol': ncol,
                'legend_rows': max(1, legend_rows),
                'title_text': title_text,
                'empty_text': empty_text,
            }
            cache = self._mpl_cache

        fig = cache['fig']
        plot_ax = cache['plot_ax']
        legend_ax = cache['legend_ax']
        line_artists = cache['line_artists']
        legend_line_artists = cache['legend_line_artists']
        legend_text_artists = cache.get('legend_text_artists', {})
        highlight_set = set(highlight_methods or [])
        has_highlight = len(highlight_set) > 0

        # ---- 3) Update dynamic plot content (data/highlight/title/ylim) ----
        for method_name, _legend_name, y_arr, _color in series_items:
            line_obj = line_artists.get(method_name)
            if line_obj is None:
                continue
            is_hl = method_name in highlight_set
            line_obj.set_data(x, y_arr)
            line_obj.set_linewidth((2.0 if is_hl else 0.2) if has_highlight else 0.5)
            line_obj.set_alpha(1.0 if (is_hl or not has_highlight) else 0.7)
            line_obj.set_zorder(6 if is_hl else 2)

            l_obj = legend_line_artists.get(method_name)
            if l_obj is not None:
                l_obj.set_linewidth(2.2 if is_hl else 1.0)

            txt_obj = legend_text_artists.get(method_name)
            if txt_obj is not None:
                txt_obj.set_fontweight('bold' if is_hl else 'normal')

        ds_name = dataset_name if dataset_name else "N/A"
        title_text = f"{self.metric_display_name} Curves on Dataset: {ds_name} (ref: {reference_key})"
        if cache.get('title_text') != title_text:
            plot_ax.set_title(title_text, fontsize=14)
            cache['title_text'] = title_text
        plot_ax.set_xlim(1, max(2, int(n)))

        if finite_values:
            y_all = np.concatenate(finite_values)
            y_min = float(np.min(y_all))
            y_max = float(np.max(y_all))
            if y_max <= y_min:
                y_min -= 1.0
                y_max += 1.0
            pad = max(0.5, (y_max - y_min) * 0.08)
            y_data_min = y_min - pad
            y_data_max = y_max + pad
            plot_ax.set_ylim(y_data_min, y_data_max)
        else:
            y_data_min = 0.0
            y_data_max = 50.0
            plot_ax.set_ylim(y_data_min, y_data_max)

        empty_text = cache.get('empty_text')
        if empty_text is not None:
            has_any_values = bool(finite_values)
            empty_text.set_visible(not has_any_values)

        # Make tick density adapt to plot resolution (larger plot -> denser ticks).
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = plot_ax.get_window_extent(renderer=renderer)
        axis_w_px = max(1.0, float(abs(bbox.x1 - bbox.x0)))
        axis_h_px = max(1.0, float(abs(bbox.y1 - bbox.y0)))

        if MaxNLocator is not None and FormatStrFormatter is not None:
            x_bins = max(3, min(24, int(round(axis_w_px / 90.0))))
            if int(n) > 0:
                x_bins = min(x_bins, max(2, int(n)))
            y_bins = max(3, min(16, int(round(axis_h_px / 60.0))))

            plot_ax.xaxis.set_major_locator(MaxNLocator(nbins=x_bins, integer=True, min_n_ticks=2))
            plot_ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            plot_ax.yaxis.set_major_locator(MaxNLocator(nbins=y_bins, min_n_ticks=3))

            if AutoMinorLocator is not None:
                x_minor = max(2, min(6, int(round(axis_w_px / 240.0))))
                y_minor = max(2, min(6, int(round(axis_h_px / 180.0))))
                plot_ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
                plot_ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))
                plot_ax.grid(True, which='minor', linestyle=':', linewidth=0.08, alpha=0.6)

        # ---- 4) Rasterize and compute hitboxes + stable plot bounds ----
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        renderer = fig.canvas.get_renderer()

        # Use axis bbox and convert display(bottom-left origin) -> image(top-left origin).
        bbox = plot_ax.get_window_extent(renderer=renderer)
        x_min = int(round(min(bbox.x0, bbox.x1)))
        x_max = int(round(max(bbox.x0, bbox.x1)))
        y_top = int(round(h - max(bbox.y0, bbox.y1)))
        y_bottom = int(round(h - min(bbox.y0, bbox.y1)))
        line_meta = {
            'n': int(n),
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_top,
            'y_max': y_bottom,
            'y_data_min': float(y_data_min),
            'y_data_max': float(y_data_max),
        }

        hitboxes = []
        rows = max(1, int(cache.get('legend_rows', 1)))
        cell_w = 1.0 / float(max(1, ncol))
        cell_h = 1.0 / float(rows)
        for i, (method_name, _legend_name, _y_arr, _color) in enumerate(series_items):
            row = i // ncol
            col = i % ncol
            x0 = col * cell_w
            y_top_ax = 1.0 - row * cell_h
            y_bottom_ax = y_top_ax - cell_h
            p1 = legend_ax.transAxes.transform((x0, y_bottom_ax))
            p2 = legend_ax.transAxes.transform((x0 + cell_w, y_top_ax))
            x1 = int(min(p1[0], p2[0]))
            x2 = int(max(p1[0], p2[0]))
            y1_disp = int(min(p1[1], p2[1]))
            y2_disp = int(max(p1[1], p2[1]))
            y1_inv = int(h - y2_disp)
            y2_inv = int(h - y1_disp)
            y_low = min(y1_inv, y2_inv, y1_disp, y2_disp)
            y_high = max(y1_inv, y2_inv, y1_disp, y2_disp)
            hitboxes.append({
                'method': method_name,
                'x1': x1,
                'x2': x2,
                'y1': y_low,
                'y2': y_high,
            })

        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = np.ascontiguousarray(rgba[:, :, :3])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr, hitboxes, line_meta

    def _submit_render_request(self, request):
        if self._render_future is None:
            fut = self._render_executor.submit(
                self._render_plot_image,
                request['ps'],
                request['pn'],
                request['pds'],
                request['pref'],
                request['frame_idx'],
                request['highlight'],
            )
            fut._psnr_request = request
            self._render_future = fut
        else:
            self._render_staged_request = request

    def _pump_render_worker(self):
        fut = self._render_future
        if fut is None or not fut.done():
            return

        try:
            image, hitboxes, line_meta = fut.result()
        except Exception as e:
            log.warn(f"{self.metric_display_name} render failed: {e}")
            image, hitboxes, line_meta = None, [], None

        request = fut._psnr_request
        self._render_future = None
        staged = self._render_staged_request
        self._render_staged_request = None

        if request is not None:
            self._plot_image_base = image
            self._plot_line_meta = line_meta
            current_frame_idx = int(self.host.current_frame)
            self._plot_image = self._compose_plot_with_frame_line(current_frame_idx)
            self._legend_hitboxes = hitboxes or []
            self._dirty = False
            self._last_render_frame_idx = current_frame_idx
            self._last_render_highlight = set(request['highlight'] or [])

        if staged is not None:
            self._submit_render_request(staged)

    def _request_render(self, ps, pn, pds, pref, frame_idx, highlight):
        highlight_payload = tuple(sorted(set(highlight or [])))
        self._render_seq += 1
        request = {
            'seq': self._render_seq,
            'ps': ps,
            'pn': pn,
            'pds': pds,
            'pref': pref,
            'frame_idx': int(frame_idx),
            'highlight': highlight_payload,
        }
        if self._render_future is None:
            fut = self._render_executor.submit(
                self._render_plot_image,
                request['ps'],
                request['pn'],
                request['pds'],
                request['pref'],
                request['frame_idx'],
                request['highlight'],
            )
            fut._psnr_request = request
            self._render_future = fut
        else:
            self._render_staged_request = request

    def _loading_canvas(self):
        canvas = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(canvas, f"Generating {self.metric_display_name} panel...", (16, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 220, 255), 2, lineType=cv2.LINE_AA)
        return canvas

    def _cancel_pending(self):
        self._cancel_event.set()
        enqueue_thread = self._enqueue_thread
        self._enqueue_thread = None
        if enqueue_thread is not None and enqueue_thread.is_alive():
            try:
                enqueue_thread.join(timeout=0.1)
            except Exception:
                pass
        self._planned_tasks = None
        try:
            for fut in list(self._pending_futures.values()):
                if fut is not None and not fut.done():
                    fut.cancel()
        except Exception:
            pass
        self._pending_futures = {}
        self._completed_signal_count = 0

    def _ensure_plot_image(self):
        token = self._build_cache_token()
        now_ts = time.time()
        current_frame_idx = int(self.host.current_frame)

        self._submit_planned_metric_tasks()

        self._pump_render_worker()

        completed_count = self._completed_signal_count
        self._completed_signal_count = 0

        for task_key, fut in list(self._pending_futures.items()):
            if not fut.done():
                continue
            try:
                done_token, done_method, done_index, done_ok = fut.result()
                if done_token != self._build_cache_token():
                    continue
                if not done_ok:
                    log.warn(
                        f"{self.metric_display_name} async task skipped/canceled for "
                        f"{done_method}[{done_index}]"
                    )
            except Exception as e:
                log.warn(f"{self.metric_display_name} async compute failed for task {task_key}: {e}")

        for task_key in [k for k, fut in self._pending_futures.items() if fut.done()]:
            self._pending_futures.pop(task_key, None)

        if (not self._pending_futures) and self._pending_token is not None:
            if self._pending_token == self._build_cache_token():
                self._cache_token = self._pending_token
            self._pending_token = None

        need_refresh = self._dirty or self._cache_token != token or self._plot_image is None
        frame_changed = (current_frame_idx != self._last_render_frame_idx)
        highlight_changed = (set(self._highlight_methods) != set(self._last_render_highlight))
        if completed_count > 0:
            self._completed_since_last_render += completed_count

        data_changed = self._completed_since_last_render > 0
        force_refresh = need_refresh or highlight_changed
        should_refresh_plot = force_refresh or data_changed

        if should_refresh_plot:
            ps = {k: list(v) for k, v in self.values.items()}
            pn = int(self._values_n)
            pds = self._values_dataset
            pref = self._values_ref
            self._request_render(ps, pn, pds, pref, current_frame_idx, self._highlight_methods)
            self._last_refresh_ts = now_ts

            if self._completed_since_last_render > 0:
                self._completed_signal_count = max(
                    0,
                    self._completed_signal_count - self._completed_since_last_render,
                )
                self._completed_since_last_render = 0
        elif frame_changed and self._plot_image_base is not None:
            self._plot_image = self._compose_plot_with_frame_line(current_frame_idx)
            self._last_render_frame_idx = current_frame_idx

    def on_before_update_display(self, _host):
        self._ensure_plot_image()

    def on_after_update_display(self, _host):
        self._ensure_plot_image()
        if not self._psnr_mouse_bound:
            cv2.namedWindow(self.window_psnr)
            cv2.setMouseCallback(self.window_psnr, self._on_psnr_mouse)
            self._psnr_mouse_bound = True
        has_render_job = self._render_future is not None
        if self._plot_image is not None:
            cv2.imshow(self.window_psnr, self._plot_image)
        elif self._pending_futures or has_render_job:
            cv2.imshow(self.window_psnr, self._loading_canvas())
    #endregion

    def update(self):
        should_update = False
        if self._completed_signal_count > 0:
            should_update = True
        if self._render_future is not None and self._render_future.done():
            should_update = True
        if should_update:
            self.host.request_update()

    def on_rebuild_dataset(self, _host):
        self._highlight_methods = set()
        self._last_render_frame_idx = -1
        self._last_render_highlight = set()
        self._psnr_jump_armed = False
        self._reset_mpl_cache()
        self._start_async_compute()

    def on_shutdown(self, _host):
        self._cancel_pending()
        if self._render_future is not None and not self._render_future.done():
            self._render_future.cancel()
        self._render_future = None
        self._render_staged_request = None
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self._render_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            if self._metric_runner is not None and hasattr(self._metric_runner, 'shutdown'):
                self._metric_runner.shutdown(wait=False)
        except Exception:
            pass
        self._psnr_jump_armed = False
        self._reset_mpl_cache()
        self._psnr_mouse_bound = False


class _FeatureBundle:
    def __init__(self, features):
        self.features = [f for f in (features or []) if f is not None]

    def update(self):
        for f in self.features:
            try:
                f.update()
            except Exception:
                pass


def install_default_metrics_feature(host, metric_types='psnr', max_workers=16, threads_per_methods=4):
    if isinstance(metric_types, str):
        metric_list = [m.strip() for m in metric_types.replace(',', ' ').split() if m.strip()]
    else:
        metric_list = [str(m).strip() for m in (metric_types or []) if str(m).strip()]
    if not metric_list:
        metric_list = ['psnr']

    log.info(f"Registered metric feature for {', '.join(metric_list)} with max_workers={max_workers} and threads_per_methods={threads_per_methods}")

    shared_metric_runner = None
    if _IQAs is not None:
        try:
            shared_metric_runner = _IQAs(
                *metric_list,
                device='cpu',
                traditional=True,
            )
        except Exception:
            traceback.print_exc()
            shared_metric_runner = None

    shared_metric_cache = {}
    setattr(host, '_metric_feature_cache', shared_metric_cache)

    features = [
        AsyncMetricFeature(
            host,
            refresh_interval_sec=0.5,
            max_workers=max_workers,
            metric_type=mt,
            threads_per_methods=threads_per_methods,
            metric_runner=shared_metric_runner,
            metric_result_cache=shared_metric_cache,
        )
        for mt in metric_list
    ]
    return _FeatureBundle(features)


class InteractiveCropComparator:
    def __init__(
            self,
            input_folders,
            output_folder,
            reference_key=None,
            columns=None,
            grid_gap=2,
            display_scale=1.0,
            full_image_scale=0.5,
            line_thickness=5,
            layout_border_scale=2.0,
            layout_gap=10,
            layout_bg_color: Union[str, Tuple] = 'transparent',
            layout_min_scale=1.0,
            layout_use_alpha: bool = False,
            compose_layout: bool = True,
            save_selection_image: bool = False,
            current_group=None,
            current_dataset=None,
            method_roots=None,
            event_on_init=None,
    ):
        self.input_folders = input_folders
        self.output_folder = output_folder
        # Auto-compute columns if not specified and method count >= 9
        if columns is None:
            num_methods = len(input_folders) if input_folders else 0
            if num_methods >= 9:
                columns = (num_methods + 1) // 2
            else:
                columns = num_methods
        self.columns = max(int(columns), 1)
        self.grid_gap = int(grid_gap)
        try:
            self.display_scale = float(display_scale)
            if self.display_scale <= 0:
                self.display_scale = 1.0
        except Exception:
            self.display_scale = 1.0
        self.display_scale_min = 0.25
        self.display_scale_max = 8.0
        try:
            self.full_image_scale = float(full_image_scale)
            if self.full_image_scale <= 0:
                self.full_image_scale = 0.5
        except Exception:
            self.full_image_scale = 0.5

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        self.method_roots = dict(method_roots or {})
        self.image_files = {}
        for name, src in self.input_folders.items():
            if isinstance(src, (list, tuple)):
                files = filter_hidden(list(src))
            else:
                files = filter_hidden(glob_single_files(src, IMG_EXTS))
            self.image_files[name] = files
            if name not in self.method_roots:
                self.method_roots[name] = self._infer_method_root(src, files)

        keys = list(self.image_files.keys())
        reference_keys = ['GT', 'input', 'reference', 'ref', 'gt', 'lq', 'hq', 'Input', 'Reference']
        if reference_key is None:
            for rk in reference_keys:
                if rk in self.image_files and len(self.image_files[rk]) > 0:
                    reference_key = rk
                    break
            if reference_key is None and keys:
                reference_key = keys[0]
        self.reference_key = reference_key

        if len(self.image_files[self.reference_key]) == 0:
            raise ValueError(f"Reference folder has no images, ref keys: {self.reference_key}, image files: {self.image_files[self.reference_key]}, input files: {self.input_folders[self.reference_key]}")

        self.num_frames = len(self.image_files[self.reference_key])
        sample = read_images_as_numpy(self.image_files[self.reference_key][0])
        if sample is None:
            raise ValueError("Failed to read reference sample image")
        self.height, self.width = sample.shape[:2]

        self.current_frame = 0

        self.window_main = "Crop Controller"
        self.window_grid = "Crop Grid"
        self.window_final = "Final Layout"
        self.window_psnr = "Metric Curves"

        self.dragging = False
        self.mode = 'idle'  # 'selection' | 'position' | 'idle'
        self.rois = {}  # dict of {id: {'rect': (x1,y1,x2,y2) or None, 'color': (b,g,r)}}
        self.active_roi = None  # active roi id
        self.selection_start = None

        self.palette = [
            (0, 0, 255),  # red
            (255, 0, 0),  # blue
            (0, 255, 0),  # green
            (0, 255, 255),  # yellow
            (255, 0, 255),  # magenta
            (255, 255, 0),  # cyan
            (255, 255, 255),  # white
            (0, 128, 255),
            (128, 255, 0),
            (255, 0, 128),
            (128, 0, 255),
            (0, 255, 128),
            (255, 128, 0),
        ]
        self.color_darken_factor = 0.9

        self.line_color = (0, 0, 255)
        self.text_color = (0, 255, 0)
        try:
            self.line_thickness = max(1, int(line_thickness))
        except Exception:
            self.line_thickness = 5
        try:
            self.layout_border_scale = max(0.5, float(layout_border_scale))
        except Exception:
            self.layout_border_scale = 2.0
        try:
            self.layout_gap = max(0, int(layout_gap))
        except Exception:
            self.layout_gap = 10
        try:
            self.layout_min_scale = max(0.01, float(layout_min_scale))
        except Exception:
            self.layout_min_scale = 1.0
        self.layout_use_alpha = bool(layout_use_alpha)
        try:
            if isinstance(layout_bg_color, str):
                lb = layout_bg_color.strip().lower()
                if lb == 'transparent':
                    layout_bg_color = (0, 0, 0, 0)
                    self.layout_use_alpha = True
                else:
                    parts = [int(x) for x in layout_bg_color.replace(' ', '').split(',') if x != '']
                    if len(parts) == 4:
                        layout_bg_color = tuple(max(0, min(255, v)) for v in parts)
                        self.layout_use_alpha = True
                    elif len(parts) == 3:
                        layout_bg_color = tuple(max(0, min(255, v)) for v in parts)
                    else:
                        raise ValueError()
            else:
                seq = tuple(layout_bg_color)
                if len(seq) == 4:
                    layout_bg_color = seq
                    self.layout_use_alpha = True
                elif len(seq) == 3:
                    layout_bg_color = seq
                else:
                    raise ValueError()
            self.layout_bg_color = layout_bg_color
        except Exception:
            self.layout_bg_color = (255, 255, 255)
            self.layout_use_alpha = False

        self.cached_images = None
        self.grid_windows = set()
        self.method_full_image_window = "Method Full Images"
        self.layout_mode = 'right'  # 'left' | 'top' | 'right' | 'bottom'
        self.sort_mode = 'position'  # 'position' | 'id'
        self.sort_reverse = False
        self.preview_key = reference_key
        self.single_crop_position = 'auto'
        self.compose_layout = bool(compose_layout)
        self.save_selection_image = bool(save_selection_image)
        self.show_all_method_images = False
        self.save_session_ts = None
        self.undo_manager = UndoManager()
        self.dispatcher = EventDispatcher()
        self._register_keybindings()
        self._pre_drag_state = None
        self._pre_drag_snapshot = None
        self.needs_update = True
        self._idle_return_mode = 'selection'
        self.preview_mask_alpha = 0.1  # fill opacity for ROI preview
        # Mouse state
        self._rbutton_down_roi_id = None
        self._rbutton_left_roi = False
        self._drag_button = None
        self._mb_down_roi_id = None
        self._mb_down_point = None
        self._last_input_activity_ts = 0.0
        self._layout_debounce_sec = float(INPUT_IDLE_UPDATE_SEC)
        self._cached_layout_signature = None
        self._cached_grid_views = {}
        self._cached_final_view = None
        self._cached_ref_frame = None
        self._cached_ref_frame_key = None
        self._cached_ref_frame_idx = -1
        self._cached_ref_frame_path = None
        # Optional event hooks for branch features (e.g. PSNR plugin)
        self._event_handlers = {}
        # dataset/group tracking
        self.group = current_group
        self.dataset = current_dataset
        # Grid display ordering / optional .srt-based sorting
        self._method_srt_stems = {}
        self._all_methods_have_srt = False
        # Derive dataset/group by reversing path parts across all inputs.
        # Rules:
        # - only 1 common tail folder: dataset=<folder>, group=None
        # - 2+ common tail folders: dataset=<1st>, group=<2nd>
        if self.group is None or self.dataset is None:
            try:
                dir_parts_list = []
                for _, src in self.input_folders.items():
                    if isinstance(src, (list, tuple)) and len(src) > 0:
                        first_path = src[0]
                        if isinstance(first_path, str) and first_path:
                            norm_dir = os.path.normpath(os.path.dirname(first_path))
                            parts = [p for p in norm_dir.split(os.sep) if p and p != '.']
                            if parts:
                                dir_parts_list.append(parts)
                        continue

                    if isinstance(src, str) and src:
                        # Prefer real image directories (already discovered from input_folders)
                        # so dataset/group inference does not collapse to method roots like "examples/A-Net".
                        files = self.image_files.get(_, [])
                        first_file = files[0] if isinstance(files, list) and len(files) > 0 else None
                        if isinstance(first_file, str) and first_file:
                            norm_dir = os.path.normpath(os.path.dirname(first_file))
                            parts = [p for p in norm_dir.split(os.sep) if p and p != '.']
                            if parts:
                                dir_parts_list.append(parts)
                                continue

                        norm = os.path.normpath(src)
                        parts = [p for p in norm.split(os.sep) if p and p != '.']
                        if parts:
                            dir_parts_list.append(parts)

                inferred_dataset = None
                inferred_group = None
                if dir_parts_list:
                    rev_lists = [list(reversed(parts)) for parts in dir_parts_list]
                    min_len = min(len(rp) for rp in rev_lists)
                    common_tail = []
                    for i in range(min_len):
                        token = rev_lists[0][i]
                        if all(rp[i] == token for rp in rev_lists[1:]):
                            common_tail.append(token)
                        else:
                            break

                    if len(common_tail) >= 1:
                        inferred_dataset = common_tail[0]
                    if len(common_tail) >= 2:
                        inferred_group = common_tail[1]
                    # Fallback rule:
                    # if the 1st tail folder differs but the 2nd is the same,
                    # use the 2nd as dataset and keep group empty.
                    if len(common_tail) == 0 and min_len >= 2:
                        second_token = rev_lists[0][1]
                        if all(rp[1] == second_token for rp in rev_lists[1:]):
                            inferred_dataset = second_token
                            inferred_group = None

                if self.dataset is None and inferred_dataset:
                    self.dataset = inferred_dataset
                if self.group is None:
                    self.group = inferred_group
            except Exception:
                pass
        log.info(f"Inferred dataset: {log.style_key(self.dataset)}, group: {log.style_key(self.group)} from input paths.")

        self._refresh_method_grid_sorting()
        self._event_on_init = event_on_init if callable(event_on_init) else None
        self._event_on_init_done = False

    def _try_runtime_event_init(self):
        """Initialize optional runtime features once, after the first frame is shown."""
        if self._event_on_init_done or self._event_on_init is None:
            return

        self._event_on_init_done = True
        try:
            feature_obj = self._event_on_init(self)
            if hasattr(feature_obj, 'update') and callable(feature_obj.update):
                self.register_event_handler(
                    'loop_tick',
                    lambda _host, _feature=feature_obj: _feature.update(),
                )
        except Exception as e:
            log.error(f"event_on_init execution failed: {e}")
            traceback.print_exc()

    def register_event_handler(self, event_name, handler):
        if not event_name or not callable(handler):
            return
        self._event_handlers.setdefault(str(event_name), []).append(handler)

    def _emit_event(self, event_name):
        handlers = self._event_handlers.get(str(event_name), [])
        for h in handlers:
            try:
                h(self)
            except Exception as e:
                log.error(f"Event handler failed for {event_name}: {e}")
                traceback.print_exc()

    def _infer_method_root(self, src, files):
        """Best-effort method root inference for rebuild_dataset support."""
        if isinstance(src, str):
            path = os.path.normpath(src)
            parts = path.split(os.sep)
            if len(parts) >= 3:
                return os.sep.join(parts[:-2])
            if len(parts) >= 2:
                return os.path.dirname(path)
        if files:
            common = os.path.commonpath(files)
            return os.path.dirname(common)
        return None

    # ---- Undo helpers ----
    def _snapshot_rois(self):
        return copy.deepcopy(self.rois), self.active_roi, self.mode

    def _restore_rois(self, snapshot):
        rois, active, mode = snapshot
        self.rois = rois
        self.active_roi = active
        self.mode = mode
        self.selection_start = None
        self.request_update()

    def _record_state_change(self, before, after, restore_fn, desc):
        self.undo_manager.record(
            lambda b=before: restore_fn(b),
            lambda a=after: restore_fn(a),
            desc,
        )

    def _record_rois_change(self, desc, before=None):
        before_snap = before or self._snapshot_rois()
        after_snap = self._snapshot_rois()
        self._record_state_change(before_snap, after_snap, self._restore_rois, desc)

    def _record_frame(self, prev_frame, new_frame):
        self.undo_manager.record(
            lambda pf=prev_frame: self._set_frame(pf),
            lambda nf=new_frame: self._set_frame(nf),
            "frame change",
        )

    def _record_layout(self, prev_layout, new_layout):
        self.undo_manager.record(
            lambda pl=prev_layout: self._set_layout(pl),
            lambda nl=new_layout: self._set_layout(nl),
            "layout change",
        )

    # ---- State setters ----
    def _refresh_current_frame_bounds(self):
        """Refresh interaction bounds from the current reference frame size."""
        try:
            img = self.read_frame(self.reference_key, self.current_frame)
            if img is not None:
                h, w = img.shape[:2]
                self.height, self.width = h, w
        except Exception:
            # Keep previous bounds if current frame cannot be read.
            pass

    def _set_frame(self, idx):
        self.current_frame = max(0, min(self.num_frames - 1, idx))
        self._refresh_current_frame_bounds()
        self.request_update()

    def _set_layout(self, mode):
        self.layout_mode = mode
        self.request_update()

    def _adjust_display_scale(self, step_count):
        if step_count == 0:
            return
        old_scale = float(self.display_scale)
        new_scale = old_scale + 0.1 * int(step_count)
        new_scale = max(self.display_scale_min, min(self.display_scale_max, new_scale))
        new_scale = round(new_scale, 1)
        if abs(new_scale - old_scale) > 1e-9:
            self.display_scale = new_scale
            log.note(f"Display scale: {self.display_scale:.2f}")
            self.request_update()

    # ---- Key binding setup ----
    def _register_keybindings(self):
        self.dispatcher.register(ord('n'), self._cmd_next_frame)
        self.dispatcher.register(ord('p'), self._cmd_prev_frame)
        self.dispatcher.register(ord(']'), self._cmd_next_frame)
        self.dispatcher.register(ord('['), self._cmd_prev_frame)
        self.dispatcher.register(9, self._cmd_toggle_all_method_images)
        self.dispatcher.register(ord('a'), self._cmd_add_roi)
        self.dispatcher.register(ord('i'), self._cmd_idle_mode)
        self.dispatcher.register(ord('r'), self._cmd_clear_rois)
        self.dispatcher.register(ord('s'), self._cmd_save)
        self.dispatcher.register(ord('z'), self._cmd_undo)
        self.dispatcher.register(ord('y'), self._cmd_redo)
        self.dispatcher.register(ord('d'), self._cmd_duplicate_roi)
        self.dispatcher.register(ord('='), lambda: self._adjust_display_scale(1))
        self.dispatcher.register(ord('+'), lambda: self._adjust_display_scale(1))
        self.dispatcher.register(ord('-'), lambda: self._adjust_display_scale(-1))
        self.dispatcher.register(ord('_'), lambda: self._adjust_display_scale(-1))
        # digits 1-9
        for d in range(1, 10):
            self.dispatcher.register(ord(str(d)), lambda rid=d: self._cmd_digit_roi(rid))
        # Shift+1..9
        shift_map = {'!': 1, '@': 2, '#': 3, '$': 4, '%': 5, '^': 6, '&': 7, '*': 8, '(': 9}
        for sym, rid in shift_map.items():
            self.dispatcher.register(ord(sym), lambda rid=rid: self._cmd_shift_digit(rid))
        # arrow keys
        self.dispatcher.register(81, lambda: self._cmd_layout('left'))
        self.dispatcher.register(82, lambda: self._cmd_layout('top'))
        self.dispatcher.register(83, lambda: self._cmd_layout('right'))
        self.dispatcher.register(84, lambda: self._cmd_layout('bottom'))
        # delete/backspace
        self.dispatcher.register(8, self._cmd_delete_roi)
        self.dispatcher.register(127, self._cmd_delete_roi)

    # ---- Commands ----
    def _cmd_next_frame(self):
        prev = self.current_frame
        if prev >= self.num_frames - 1:
            return
        new_idx = prev + 1
        self._set_frame(new_idx)
        self._record_frame(prev, new_idx)

    def _cmd_prev_frame(self):
        prev = self.current_frame
        if prev <= 0:
            return
        new_idx = prev - 1
        self._set_frame(new_idx)
        self._record_frame(prev, new_idx)

    def _cmd_add_roi(self):
        before = self._snapshot_rois()
        self.add_roi()
        self._record_rois_change("add roi", before=before)
        self.request_update()

    def _cmd_delete_roi(self):
        if self.active_roi is None or self.active_roi not in self.rois:
            return
        before = self._snapshot_rois()
        rid = self.active_roi
        self.rois.pop(rid, None)
        self.active_roi = None
        self.selection_start = None
        if not self.rois:
            self.mode = 'idle'
        self._record_rois_change(f"delete roi {rid}", before=before)
        self.request_update()

    def _cmd_clear_rois(self):
        if not self.rois:
            return
        before = self._snapshot_rois()
        self.rois = {}
        self.active_roi = None
        self.selection_start = None
        self.mode = 'selection'
        self._record_rois_change("clear all rois", before=before)
        self.add_roi()
        self.request_update()

    def _cmd_duplicate_roi(self):
        if self.active_roi is None or self.rois.get(self.active_roi, {}).get('rect') is None:
            log.warn("No active ROI to duplicate size from; create/select one first")
            return
        before = self._snapshot_rois()
        src_rect = self.rois[self.active_roi]['rect']
        new_id = self.add_roi()
        self.rois[new_id]['rect'] = src_rect
        log.success(
            f"Added ROI {log.style_num(str(new_id))} with the same size as ROI {log.style_num(str(self.active_roi))}")
        self.set_active_roi(new_id, to_selection=False)
        self._record_rois_change("duplicate roi", before=before)
        self.request_update()

    def _cmd_idle_mode(self):
        before = self._snapshot_rois()
        if self.mode != 'idle':
            self._idle_return_mode = self.mode or self._idle_return_mode
            self.mode = 'idle'
            self.active_roi = None
            self.selection_start = None
            desc = "enter idle mode"
        else:
            restore_mode = self._idle_return_mode or 'selection'
            self.mode = restore_mode
            desc = f"exit idle -> {restore_mode}"
        self._record_rois_change(desc, before=before)
        self.request_update()

    def _cmd_save(self):
        label = getattr(self, 'save_label', self.dataset)
        self.save(label, self.dataset)

    def _cmd_digit_roi(self, roi_id):
        before = self._snapshot_rois()
        if roi_id in self.rois:
            if self.active_roi == roi_id:
                self.set_active_roi(roi_id, to_selection=self.mode != 'selection')
            else:
                self.set_active_roi(roi_id, to_selection=False)
            desc = f"select roi {roi_id}"
        else:
            self.add_roi(roi_id=roi_id)
            desc = f"add roi {roi_id}"
        self._record_rois_change(desc, before=before)
        self.request_update()

    def _cmd_shift_digit(self, target_id):
        if self.active_roi is None or self.rois.get(self.active_roi, {}).get('rect') is None:
            log.warn("No active ROI to copy from; select or create one first")
            return
        before = self._snapshot_rois()
        src_rect = self.rois[self.active_roi]['rect']
        sx1, sy1, sx2, sy2 = src_rect
        sw = max(1, sx2 - sx1)
        sh = max(1, sy2 - sy1)
        if target_id not in self.rois:
            self.add_roi(roi_id=target_id)
            self.rois[target_id]['rect'] = (sx1, sy1, sx2, sy2)
            log.success(f"Duplicated ROI to {log.style_num(str(target_id))}")
        else:
            tx1, ty1, tx2, ty2 = self.rois[target_id]['rect'] or (sx1, sy1, sx2, sy2)
            tcx = (tx1 + tx2) // 2
            tcy = (ty1 + ty2) // 2
            nx1 = int(round(tcx - sw / 2))
            ny1 = int(round(tcy - sh / 2))
            nx2 = nx1 + sw
            ny2 = ny1 + sh
            self.rois[target_id]['rect'] = (nx1, ny1, nx2, ny2)
            log.success(f"Copied size to ROI {log.style_num(str(target_id))}")
        self.set_active_roi(target_id, to_selection=False)
        self._record_rois_change(f"shift duplicate {self.active_roi}->{target_id}", before=before)
        self.request_update()

    def _cmd_layout(self, mode):
        if mode == self.layout_mode:
            return
        prev = self.layout_mode
        self._set_layout(mode)
        self._record_layout(prev, mode)
        log.note(f"Layout set to {log.style_mode(self.layout_mode)}")

    def _cmd_undo(self):
        ok, desc = self.undo_manager.undo()
        if not ok:
            log.warn("Nothing to undo" if desc is None else f"Undo failed: {desc}")
        else:
            log.info(f"Undid: {desc or 'last action'}")
            self.request_update()

    def _cmd_redo(self):
        ok, desc = self.undo_manager.redo()
        if not ok:
            log.warn("Nothing to redo" if desc is None else f"Redo failed: {desc}")
        else:
            log.info(f"Redid: {desc or 'last action'}")
            self.request_update()

    def _cmd_toggle_all_method_images(self):
        self.show_all_method_images = not self.show_all_method_images
        if not self.show_all_method_images:
            log.note("Hidden full images for all methods")
        else:
            log.note("Showing full images for all methods")
        self.request_update()

    # ---- Small helpers ----
    def _set_mode(self, mode):
        self.mode = mode
        self.request_update()

    def _restore_active(self, active_id, mode):
        self.active_roi = active_id
        self.mode = mode
        self.selection_start = None
        self.request_update()

    def _restore_roi_rect(self, rid, rect):
        if rid not in self.rois:
            return
        self.rois[rid]['rect'] = rect
        self.request_update()

    def request_update(self):
        """Mark UI as dirty; main loop will repaint."""
        self.needs_update = True

    def _layout_state_signature(self):
        rois_sig = []
        for rid, r in sorted(self.rois.items()):
            rect = r.get('rect')
            if rect is None:
                rois_sig.append((rid, None))
            else:
                x1, y1, x2, y2 = self.clamp_rect(rect)
                rois_sig.append((rid, int(x1), int(y1), int(x2), int(y2)))

        return (
            str(self.mode),
            int(self.current_frame),
            str(self.preview_key),
            str(self.layout_mode),
            str(self.sort_mode),
            bool(self.sort_reverse),
            float(self.display_scale),
            float(self.preview_mask_alpha),
            tuple(rois_sig),
        )

    def _color_with_alpha(self, color):
        """Return a color matching the current canvas channel count."""
        if self.layout_use_alpha:
            if len(color) == 4:
                return tuple(color)
            return (color[0], color[1], color[2], 255)
        return tuple(color[:3])

    def _to_canvas_image(self, img):
        """Ensure image matches the current canvas channel count."""
        if img is None:
            return None
        if self.layout_use_alpha:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    def _blank_canvas(self, height, width):
        """Create a blank canvas with correct channel count and background."""
        if self.layout_use_alpha:
            return np.full((height, width, 4), self.layout_bg_color, dtype=np.uint8)
        return np.full((height, width, 3), self.layout_bg_color, dtype=np.uint8)

    def _resolve_method_keys(self):
        keys = self._ordered_method_keys_for_grid()
        if len(keys) == 0:
            keys = list(self.image_files.keys())
        return keys

    def _new_grid_canvas(self, height, width):
        """Create RGB(+alpha) buffers for grid composition with layout background."""
        if self.layout_use_alpha:
            if len(self.layout_bg_color) >= 4:
                bg = self.layout_bg_color
            else:
                bg = (
                    self.layout_bg_color[0] if len(self.layout_bg_color) >= 1 else 0,
                    self.layout_bg_color[1] if len(self.layout_bg_color) >= 2 else 0,
                    self.layout_bg_color[2] if len(self.layout_bg_color) >= 3 else 0,
                    0,
                )
            grid_rgb = np.full((height, width, 3), bg[:3], dtype=np.uint8)
            grid_alpha = np.full((height, width), bg[3], dtype=np.uint8)
            return grid_rgb, grid_alpha, bg

        bg = self.layout_bg_color[:3] if len(self.layout_bg_color) >= 3 else (0, 0, 0)
        grid_rgb = np.full((height, width, 3), bg, dtype=np.uint8)
        return grid_rgb, None, bg

    def _finalize_grid_canvas(self, grid_rgb, grid_alpha):
        if self.layout_use_alpha and grid_alpha is not None:
            return np.dstack([grid_rgb, grid_alpha])
        return grid_rgb

    @staticmethod
    def _grid_text_style(display_scale):
        safe_scale = max(display_scale, 0.01)

        text_scale = 0.35 * safe_scale
        text_thickness = max(1, int(round(0.5 * safe_scale)))
        text_y = max(8, int(round(12 * safe_scale)))
        return text_scale, text_thickness, text_y

    def _darken_color(self, color, steps):
        if steps <= 0:
            return tuple(color)
        scale = self.color_darken_factor ** steps
        bgr = [int(max(0, min(255, round(c * scale)))) for c in color[:3]]
        if len(color) == 4:
            bgr.append(color[3])
        return tuple(bgr)

    def color_for_id(self, roi_id, total_count=None):
        if not self.palette:
            base = (0, 0, 255)
        else:
            base = self.palette[(roi_id - 1) % len(self.palette)]
        try:
            n = total_count if total_count is not None else len(self.rois)
        except Exception:
            n = len(self.rois)
        steps = n // len(self.palette) if n > len(self.palette) else 0
        return self._darken_color(base, steps)

    def rebuild_dataset(self, new_dataset, new_group=None):
        # Build new input folders based on method roots
        if not self.method_roots:
            log.error("Dataset switching not supported for this layout")
            return False
        label_path = f"{new_group}/{new_dataset}" if new_group else new_dataset
        log.info(
            f"Switching dataset to {log.style_path(label_path)} "
            f"for {len(self.method_roots)} methods"
        )
        new_inputs = {}
        for name, method_root in self.method_roots.items():
            if not method_root or not os.path.isdir(method_root):
                log.warn(f"Cannot switch dataset for {log.style_path(name)} (missing method root)")
                continue
            # Handle None group for flat structures or dataset-only layouts
            if new_group:
                cand = os.path.join(method_root, name, new_group, new_dataset)
            else:
                cand = os.path.join(method_root, name, new_dataset)
            if not os.path.exists(cand):
                log.warn(f"Missing folder for {log.style_path(name)} at {log.style_path(cand)}")
                continue
            imgs = filter_hidden(glob_single_files(cand, ['png', 'jpg', 'jpeg']))
            if len(imgs) == 0:
                log.warn(f"No images in {log.style_path(cand)} for {log.style_path(name)}")
                continue
            new_inputs[name] = cand

        if not new_inputs:
            log.error(f"Switch failed: no valid inputs for {log.style_path(label_path)}")
            return False

        # Update state
        self.input_folders = new_inputs
        self.group = new_group
        self.dataset = new_dataset
        self.image_files = {name: filter_hidden(glob_single_files(path, ['png', 'jpg', 'jpeg'])) for name, path in
                            new_inputs.items()}
        self._cached_ref_frame = None
        self._cached_ref_frame_key = None
        self._cached_ref_frame_idx = -1
        self._cached_ref_frame_path = None

        # pick reference key
        keys = list(self.image_files.keys())
        if 'GT' in self.image_files:
            self.reference_key = 'GT'
        elif 'input' in self.image_files:
            self.reference_key = 'input'
        else:
            self.reference_key = keys[0]

        # ensure preview key still valid
        if self.preview_key not in self.image_files:
            self.preview_key = self.reference_key

        # reset frame and dimensions
        if len(self.image_files[self.reference_key]) == 0:
            log.error("Reference folder empty after switch")
            return False
        sample = read_images_as_numpy(self.image_files[self.reference_key][0])
        if sample is None:
            log.error("Failed to read reference sample after switch")
            return False
        self.height, self.width = sample.shape[:2]
        self.num_frames = len(self.image_files[self.reference_key])
        self.current_frame = 0

        # reset ROI state
        self.rois = {}
        self.active_roi = None
        self.selection_start = None
        self.mode = 'selection'
        # Close all existing grid windows before clearing the set
        for w in list(self.grid_windows):
            cv2.destroyWindow(w)
        self.grid_windows = set()
        self.add_roi()
        self._emit_event('on_rebuild_dataset')
        self._refresh_method_grid_sorting()
        if new_group:
            log.success(f"Switched to {log.style_path(new_group)}/{log.style_path(new_dataset)}")
        else:
            log.success(f"Switched to {log.style_path(new_dataset)}")
        return True

    # ---- Grid ordering helpers ----
    @staticmethod
    def _is_input_key(key):
        return str(key).lower() == 'input'

    @staticmethod
    def _is_gt_key(key):
        return str(key).lower() == 'gt'

    def _refresh_method_grid_sorting(self):
        """If every method folder contains a .srt file, use its stem for grid sorting."""
        self._method_srt_stems = {}

        keys = list(self.input_folders.keys()) if isinstance(self.input_folders, dict) else []
        methods = [k for k in keys if (not self._is_input_key(k)) and (not self._is_gt_key(k))]
        if not methods:
            self._all_methods_have_srt = False
            return

        def _list_srt_files(folder):
            try:
                files = [
                    f for f in os.listdir(folder)
                    if (not f.startswith('.')) and f.lower().endswith('.srt') and os.path.isfile(os.path.join(folder, f))
                ]
                return _natsorted(files)
            except Exception:
                traceback.print_exc()
                return []

        def _method_dir_from_src(method_key, src_path):
            if not isinstance(src_path, str) or not src_path:
                return None
            try:
                norm = os.path.normpath(src_path)
                parts = norm.split(os.sep)
                # Prefer the directory segment that exactly matches the method name.
                idx = None
                for i in range(len(parts) - 1, -1, -1):
                    if parts[i] == method_key:
                        idx = i
                        break
                if idx is not None:
                    cand = os.sep.join(parts[:idx + 1])
                    if os.path.isdir(cand):
                        return cand
            except Exception:
                return None
            return None

        all_have = True
        for k in methods:
            # The .srt is expected under .../<method>/ (not the dataset leaf folder)
            method_dir = None

            # 1) Try recorded method root
            try:
                mr = (self.method_roots or {}).get(k)
                if isinstance(mr, str) and os.path.isdir(mr):
                    method_dir = mr
            except Exception:
                method_dir = None

            # 2) Try deriving from the input folder path
            if method_dir is None:
                src = self.input_folders.get(k)
                method_dir = _method_dir_from_src(k, src)

            # 3) Fallback: walk up from src to find a directory containing .srt
            if method_dir is None:
                src = self.input_folders.get(k)
                if isinstance(src, str) and os.path.isdir(src):
                    cur = os.path.normpath(src)
                    for _ in range(6):
                        if os.path.isdir(cur) and _list_srt_files(cur):
                            method_dir = cur
                            break
                        parent = os.path.dirname(cur)
                        if not parent or parent == cur:
                            break
                        cur = parent

            if method_dir is None or (not os.path.isdir(method_dir)):
                all_have = False
                break

            srt_files = _list_srt_files(method_dir)
            if not srt_files:
                all_have = False
                break

            stem = os.path.splitext(srt_files[0])[0]
            self._method_srt_stems[k] = stem

        if all_have:
            log.info(f"Using `.srt`-based sorting for grid display of methods: "
                     f"{', '.join(f'{log.style_path(k)}->{v}' for k, v in self._method_srt_stems.items())}")
        else:
            missing = [k for k in methods if k not in self._method_srt_stems]
            log.info(f"Not all methods have `.srt` files. Methods missing `.srt` files: {', '.join(log.style_path(k) for k in missing)}")
        self._all_methods_have_srt = bool(all_have and len(self._method_srt_stems) == len(methods))

    def _ordered_method_keys_for_grid(self):
        """Order keys for grid tiles: Input first, GT last, optional .srt stem sorting."""
        keys = list(self.image_files.keys())
        if not keys:
            return []

        input_keys = [k for k in keys if self._is_input_key(k)]
        gt_keys = [k for k in keys if self._is_gt_key(k)]
        middle = [k for k in keys if (k not in input_keys) and (k not in gt_keys)]

        if self._all_methods_have_srt and middle:
            try:
                if all(k in self._method_srt_stems for k in middle):
                    middle = sorted(middle, key=lambda k: (self._method_srt_stems.get(k, ''), str(k)))
            except Exception:
                pass
        elif middle:
            # Fallback: when .srt ordering is unavailable, keep a stable name-based order.
            middle = _natsorted(middle)

        return input_keys + middle + gt_keys

    def jump_to_image_by_name(self, name):
        # Match by stem or filename
        files = self.image_files.get(self.reference_key, [])
        target = None
        for idx, path in enumerate(files):
            base = os.path.basename(path)
            stem = base.rsplit('.', 1)[0]
            if name == base or name == stem:
                target = idx
                break
        if target is None:
            log.warn(
                f"Image {log.style_path(name)} not found; staying on {log.style_path(os.path.basename(files[self.current_frame])) if files else 'N/A'}")
            return False
        self._set_frame(target)
        log.success(f"Jumped to image {log.style_path(os.path.basename(files[self.current_frame]))}")
        return True

    def draw_inner_border(self, img, x, y, w, h, color, thickness=None):
        H, W = img.shape[:2]
        x0 = max(0, min(W, int(x)))
        y0 = max(0, min(H, int(y)))
        x1 = max(0, min(W, int(x + w)))
        y1 = max(0, min(H, int(y + h)))
        if x1 <= x0 or y1 <= y0:
            return
        base_th = self.line_thickness
        if thickness is not None:
            base_th = thickness
        th = max(1, min(base_th, min(x1 - x0, y1 - y0) // 2))
        # top
        img[y0:y0 + th, x0:x0 + (x1 - x0)] = color
        # bottom
        img[y1 - th:y1, x0:x0 + (x1 - x0)] = color
        # left
        img[y0:y1, x0:x0 + th] = color
        # right
        img[y0:y1, x1 - th:x1] = color

    def draw_dashed_rect(self, img, pt1, pt2, color, thickness=1, dash_len=6, gap_len=6):
        x1, y1 = pt1
        x2, y2 = pt2

        # horizontal lines
        def _draw_dash_line(p1, p2):
            dist = int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])))
            if dist == 0:
                return
            vx = (p2[0] - p1[0]) / dist
            vy = (p2[1] - p1[1]) / dist
            pos = 0
            while pos < dist:
                start = pos
                end = min(dist, pos + dash_len)
                sx = int(round(p1[0] + vx * start))
                sy = int(round(p1[1] + vy * start))
                ex = int(round(p1[0] + vx * end))
                ey = int(round(p1[1] + vy * end))
                cv2.line(img, (sx, sy), (ex, ey), color, thickness)
                pos += dash_len + gap_len

        _draw_dash_line((x1, y1), (x2, y1))
        _draw_dash_line((x2, y1), (x2, y2))
        _draw_dash_line((x2, y2), (x1, y2))
        _draw_dash_line((x1, y2), (x1, y1))

    def draw_circled_label(self, img, center, text, color):
        cx, cy = int(center[0]), int(center[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(str(text), font, scale, thickness)
        radius = max(tw, th) // 2 + 8
        # circle outline
        cv2.circle(img, (cx, cy), radius, color, thickness)
        # text centered inside circle
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(img, str(text), (tx, ty), font, scale, color, thickness)

    def clamp_rect(self, rect):
        x1, y1, x2, y2 = rect
        x1 = max(0, min(self.width - 1, x1))
        y1 = max(0, min(self.height - 1, y1))
        x2 = max(0, min(self.width - 1, x2))
        y2 = max(0, min(self.height - 1, y2))
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        return (x1, y1, x2, y2)

    def add_roi(self, roi_id=None):
        # Assign an explicit id if provided; otherwise use the smallest unused id
        used_ids = set(self.rois.keys())
        if roi_id is None:
            roi_id = 1
            while roi_id in used_ids and self.rois[roi_id]['rect'] is not None:
                roi_id += 1
        if roi_id in self.rois:
            self.active_roi = roi_id
            self.mode = 'selection'
            self.selection_start = None
            return roi_id
        color = self.color_for_id(roi_id, total_count=len(self.rois) + 1)
        self.rois[roi_id] = {'rect': None, 'color': color}
        self.active_roi = roi_id
        self.mode = 'selection'
        self.selection_start = None
        return roi_id

    def set_active_roi(self, roi_id, to_selection=False):
        if roi_id not in self.rois:
            return
        self.active_roi = roi_id
        self.mode = 'selection' if to_selection else 'position'
        self.selection_start = None
        self.request_update()

    def _rois_at_point(self, x, y):
        hits = []
        for rid in sorted(self.rois.keys()):
            rect = self.rois[rid].get('rect')
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
            y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
            if x1 <= x <= x2 and y1 <= y <= y2:
                hits.append(rid)
        return hits

    def _roi_at_point(self, x, y):
        hits = self._rois_at_point(x, y)
        return hits[0] if hits else None

    def _point_in_roi(self, rid, x, y):
        rect = self.rois.get(rid, {}).get('rect')
        if rect is None:
            return False
        x1, y1, x2, y2 = rect
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        return x1 <= x <= x2 and y1 <= y <= y2

    def _square_rect(self, sx, sy, ex, ey, force_square=False):
        if not force_square:
            return (sx, sy, ex, ey)
        dx = ex - sx
        dy = ey - sy
        side = max(abs(dx), abs(dy))
        if side == 0:
            return (sx, sy, ex, ey)
        sx_sign = 1 if dx >= 0 else -1
        sy_sign = 1 if dy >= 0 else -1
        return (sx, sy, sx + sx_sign * side, sy + sy_sign * side)

    def on_mouse(self, event, x, y, flags, param):
        if event in (
                cv2.EVENT_MOUSEMOVE,
                cv2.EVENT_LBUTTONDOWN,
                cv2.EVENT_LBUTTONUP,
                cv2.EVENT_RBUTTONDOWN,
                cv2.EVENT_RBUTTONUP,
                cv2.EVENT_RBUTTONDBLCLK,
                cv2.EVENT_MBUTTONDOWN,
                cv2.EVENT_MBUTTONUP,
                cv2.EVENT_MOUSEWHEEL,
        ):
            self._last_input_activity_ts = time.time()

        if event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = int(cv2.getMouseWheelDelta(flags))
            except Exception:
                delta = (int(flags) >> 16) & 0xFFFF
                if delta >= 0x8000:
                    delta -= 0x10000

            if delta != 0:
                steps = int(delta / 120) if abs(delta) >= 120 else (1 if delta > 0 else -1)
                self._adjust_display_scale(steps)
            return

        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_RBUTTONDBLCLK:
            rois_here = self._rois_at_point(x, y)
            target = None
            if rois_here:
                if self.active_roi in rois_here:
                    idx = rois_here.index(self.active_roi)
                    target = rois_here[idx]
                else:
                    target = rois_here[0]
            else:
                self._cmd_add_roi()
            self._rbutton_down_roi_id = target
            self._rbutton_left_roi = False
            return
        elif event == cv2.EVENT_RBUTTONUP:
            if self._rbutton_down_roi_id is not None:
                rbutton_leave_roi = not self._point_in_roi(self._rbutton_down_roi_id, x, y)
                if rbutton_leave_roi:
                    rid = self._rbutton_down_roi_id
                    if rid in self.rois:
                        self.active_roi = rid
                        self._cmd_delete_roi()
                elif not self._rbutton_left_roi:
                    rois_here = self._rois_at_point(x, y)
                    if self.active_roi in rois_here:
                        idx = rois_here.index(self.active_roi)
                        target = rois_here[(idx + 1) % len(rois_here)]
                    else:
                        target = rois_here[0]
                    self._cmd_digit_roi(target)
                self._rbutton_down_roi_id = None
                self._rbutton_left_roi = False
            return
        elif event == cv2.EVENT_MBUTTONDOWN:
            hit_id = self._roi_at_point(x, y)
            self._mb_down_roi_id = hit_id
            self._mb_down_point = (x, y)
            if hit_id is not None:
                if self.active_roi == hit_id:
                    # Duplicate active ROI and drag the new one
                    self._cmd_duplicate_roi()
                    new_id = self.active_roi
                    roi = self.rois.get(new_id)
                    if roi is not None:
                        self.dragging = True
                        self._drag_button = 'middle'
                        self._pre_drag_state = (new_id, roi['rect'])
                        self._pre_drag_snapshot = self._snapshot_rois()
                        self.mode = 'position'
                        self.selection_start = None
                        self._mb_down_roi_id = new_id
                else:
                    # Copy active ROI size to this ROI (Shift+digit behavior)
                    self._cmd_shift_digit(hit_id)
            return
        elif event == cv2.EVENT_MBUTTONUP:
            if self.dragging and self._drag_button == 'middle' and self.active_roi is not None:
                roi = self.rois[self.active_roi]
                self.dragging = False
                prev_state = self._pre_drag_state
                if self.mode == 'position' and roi['rect'] is not None:
                    x1, y1, x2, y2 = roi['rect']
                    w = x2 - x1
                    h = y2 - y1
                    cx, cy = x, y
                    x1 = int(cx - w // 2)
                    y1 = int(cy - h // 2)
                    x2 = x1 + w
                    y2 = y1 + h
                    roi['rect'] = (x1, y1, x2, y2)
                if prev_state is not None:
                    rid, old_rect = prev_state
                    new_rect = roi['rect']
                    if rid == self.active_roi and old_rect != new_rect and self._pre_drag_snapshot is not None:
                        self._record_rois_change("move/resize roi", before=self._pre_drag_snapshot)
                self._pre_drag_snapshot = None
                self._drag_button = None
                self.request_update()
            else:
                self._drag_button = None
            self._mb_down_roi_id = None
            self._mb_down_point = None
            return
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._rbutton_down_roi_id is not None and not self._rbutton_left_roi:
                if not self._point_in_roi(self._rbutton_down_roi_id, x, y):
                    self._rbutton_left_roi = True
        if self.active_roi is None:
            return
        roi = self.rois[self.active_roi]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self._drag_button = 'left'
            self._pre_drag_state = (self.active_roi, roi['rect'])
            self._pre_drag_snapshot = self._snapshot_rois()
            if self.mode in ['selection', 'idle'] or roi['rect'] is None:
                self.selection_start = (x, y)
                roi['rect'] = (x, y, x, y)
                self.mode = 'selection'
            elif self.mode == 'position':
                x1, y1, x2, y2 = roi['rect']
                w = x2 - x1
                h = y2 - y1
                cx, cy = x, y
                x1 = int(cx - w // 2)
                y1 = int(cy - h // 2)
                x2 = x1 + w
                y2 = y1 + h
                roi['rect'] = (x1, y1, x2, y2)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.mode == 'selection' and self.selection_start is not None:
                sx, sy = self.selection_start
                shift_on = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
                sx2, sy2, ex2, ey2 = self._square_rect(sx, sy, x, y, force_square=shift_on)
                roi['rect'] = (sx2, sy2, ex2, ey2)
            elif self.mode == 'position' and roi['rect'] is not None:
                x1, y1, x2, y2 = roi['rect']
                w = x2 - x1
                h = y2 - y1
                cx, cy = x, y
                x1 = int(cx - w // 2)
                y1 = int(cy - h // 2)
                x2 = x1 + w
                y2 = y1 + h
                roi['rect'] = (x1, y1, x2, y2)
            self.request_update()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self._drag_button = None
            prev_state = self._pre_drag_state
            if self.mode == 'selection' and self.selection_start is not None:
                sx, sy = self.selection_start
                shift_on = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
                sx2, sy2, ex2, ey2 = self._square_rect(sx, sy, x, y, force_square=shift_on)
                roi['rect'] = (sx2, sy2, ex2, ey2)
                self.selection_start = None
                self.mode = 'position'
            elif self.mode == 'position' and roi['rect'] is not None:
                x1, y1, x2, y2 = roi['rect']
                w = x2 - x1
                h = y2 - y1
                cx, cy = x, y
                x1 = int(cx - w // 2)
                y1 = int(cy - h // 2)
                x2 = x1 + w
                y2 = y1 + h
                roi['rect'] = (x1, y1, x2, y2)
            if prev_state is not None:
                rid, old_rect = prev_state
                new_rect = roi['rect']
                if rid == self.active_roi and old_rect != new_rect and self._pre_drag_snapshot is not None:
                    self._record_rois_change("move/resize roi", before=self._pre_drag_snapshot)
            self._pre_drag_snapshot = None
            self.request_update()

    def read_frame(self, key, idx):
        files = self.image_files[key]
        if idx < 0 or idx >= len(files):
            idx = max(0, min(len(files) - 1, idx))
        path = files[idx]

        # Reuse the reference frame when key/frame/path are unchanged.
        if key == self.reference_key:
            if (
                    self._cached_ref_frame is not None
                    and self._cached_ref_frame_key == key
                    and self._cached_ref_frame_idx == idx
                    and self._cached_ref_frame_path == path
            ):
                return self._cached_ref_frame

            img = read_images_as_numpy(path)
        
            self._cached_ref_frame = img
            self._cached_ref_frame_key = key
            self._cached_ref_frame_idx = idx
            self._cached_ref_frame_path = path
        else:
            img = read_images_as_numpy(path)
        
        return img

    def build_grid_for_rect(self, rect, roi_color=None):
        x1, y1, x2, y2 = self.clamp_rect(rect)
        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)

        method_keys = self._resolve_method_keys()

        cols = self.columns
        rows = max(1, math.ceil(len(method_keys) / cols))
        gap = self.grid_gap
        grid_h = rows * roi_h + gap * (rows - 1)
        grid_w = cols * roi_w + gap * (cols - 1)
        grid_rgb, grid_alpha, bg_color = self._new_grid_canvas(grid_h, grid_w)

        # Use ROI color for labels (match the ROI selection box color)
        label_color = (0, 255, 0)
        try:
            if roi_color is not None:
                label_color = tuple(int(v) for v in roi_color[:3])
        except Exception:
            label_color = (0, 255, 0)
        # Keep text size visually stable when grid window is magnified.
        text_scale, text_thickness, text_y = self._grid_text_style(self.display_scale)

        for i, name in enumerate(method_keys):
            img = self.read_frame(name, self.current_frame)
            if img is None:
                continue
            h, w = img.shape[:2]
            rx1 = max(0, min(w - 1, x1))
            ry1 = max(0, min(h - 1, y1))
            rx2 = max(rx1 + 1, min(w, x2))
            ry2 = max(ry1 + 1, min(h, y2))
            crop = img[ry1:ry2, rx1:rx2]
            
            # Extract RGB and alpha channels from crop if needed
            crop_rgb = crop[:, :, :3] if crop.ndim == 3 and crop.shape[2] >= 3 else crop if crop.ndim == 2 else crop
            crop_alpha = crop[:, :, 3] if crop.ndim == 3 and crop.shape[2] == 4 else np.full((crop.shape[0], crop.shape[1]), 255, dtype=np.uint8)
            
            # Handle size mismatch
            if crop_rgb.shape[0] != roi_h or crop_rgb.shape[1] != roi_w:
                if self.layout_use_alpha:
                    # Pad RGB part
                    pad_rgb = np.full((roi_h, roi_w, 3), bg_color[:3], dtype=np.uint8)
                    ph = min(roi_h, crop_rgb.shape[0])
                    pw = min(roi_w, crop_rgb.shape[1])
                    if crop_rgb.ndim == 3:
                        pad_rgb[:ph, :pw] = crop_rgb[:ph, :pw]
                    crop_rgb = pad_rgb
                    
                    # Pad alpha part
                    pad_alpha = np.full((roi_h, roi_w), bg_color[3] if len(bg_color) >= 4 else 0, dtype=np.uint8)
                    pad_alpha[:ph, :pw] = crop_alpha[:ph, :pw]
                    crop_alpha = pad_alpha
                else:
                    # Pad RGB part
                    pad_rgb = np.full((roi_h, roi_w, 3), bg_color, dtype=np.uint8)
                    ph = min(roi_h, crop_rgb.shape[0])
                    pw = min(roi_w, crop_rgb.shape[1])
                    if crop_rgb.ndim == 3:
                        pad_rgb[:ph, :pw] = crop_rgb[:ph, :pw]
                    crop_rgb = pad_rgb

            row = i // cols
            col = i % cols
            rg = row * gap
            cg = col * gap
            y0 = row * roi_h + rg
            x0 = col * roi_w + cg
            
            # Place RGB crop into grid
            grid_rgb[y0:y0 + roi_h, x0:x0 + roi_w] = crop_rgb
            
            # Place alpha crop into alpha channel if exists
            if self.layout_use_alpha and grid_alpha is not None:
                grid_alpha[y0:y0 + roi_h, x0:x0 + roi_w] = crop_alpha
            
            # Draw text on RGB part
            cv2.putText(
                grid_rgb,
                name,
                (x0 + 2, y0 + text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                label_color,
                text_thickness,
                lineType=cv2.LINE_AA,
            )
        
        return self._finalize_grid_canvas(grid_rgb, grid_alpha)

    def build_grid(self):
        valid_rois = [(rid, r) for rid, r in sorted(self.rois.items()) if r['rect'] is not None]
        if len(valid_rois) == 0:
            return None
        grids = []
        gap = self.grid_gap
        for rid, r in valid_rois:
            # header = f"ROI {rid}"
            # g = self.build_grid_for_rect(r['rect'], header_text=header)
            g = self.build_grid_for_rect(r['rect'], roi_color=r.get('color'))
            if g is not None:
                grids.append(g)
        if len(grids) == 0:
            return None
        
        # Determine channel count from first grid
        grid_channels = grids[0].shape[2] if grids[0].ndim == 3 else 3
        
        total_h = sum(g.shape[0] for g in grids) + gap * (len(grids) - 1)
        total_w = max(g.shape[1] for g in grids)
        
        # Create output with same channel count as input grids
        if grid_channels == 4:
            out = np.zeros((total_h, total_w, 4), dtype=np.uint8)
        else:
            out = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        
        y = 0
        for i, g in enumerate(grids):
            h, w = g.shape[:2]
            g_channels = g.shape[2] if g.ndim == 3 else 3
            
            # Ensure grid matches output channel count
            if g_channels != grid_channels:
                if grid_channels == 4 and g_channels == 3:
                    g = np.dstack([g, np.full((g.shape[0], g.shape[1]), 255, dtype=np.uint8)])
                elif grid_channels == 3 and g_channels == 4:
                    g = g[:, :, :3]
            
            out[y:y + h, :w] = g
            y += h
            if i < len(grids) - 1:
                y += gap
        return out

    def build_full_image_grid(self):
        method_keys = self._resolve_method_keys()
        if len(method_keys) == 0:
            return None

        images = []
        labels = []
        max_h = 0
        max_w = 0
        for name in method_keys:
            img = self.read_frame(name, self.current_frame)
            if img is None:
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            images.append(img)
            labels.append(str(name))
            max_h = max(max_h, img.shape[0])
            max_w = max(max_w, img.shape[1])

        if len(images) == 0:
            return None

        cols = self.columns
        rows = max(1, math.ceil(len(images) / cols))
        gap = self.grid_gap
        grid_h = rows * max_h + gap * (rows - 1)
        grid_w = cols * max_w + gap * (cols - 1)

        grid_rgb, grid_alpha, _bg_color = self._new_grid_canvas(grid_h, grid_w)

        ada_scale = max_w / 100
        full_scale = max(0.01, self.full_image_scale) * ada_scale
        text_scale, text_thickness, text_y = self._grid_text_style(full_scale)
        label_color = (0, 255, 0)

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            y0 = row * max_h + row * gap
            x0 = col * max_w + col * gap
            h, w = img.shape[:2]
            grid_rgb[y0:y0 + h, x0:x0 + w] = img
            if self.layout_use_alpha and grid_alpha is not None:
                grid_alpha[y0:y0 + h, x0:x0 + w] = 255

            cv2.putText(
                grid_rgb,
                labels[i],
                (x0 + 2, y0 + text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                label_color,
                text_thickness,
                lineType=cv2.LINE_AA,
            )

        return self._finalize_grid_canvas(grid_rgb, grid_alpha)

    def build_final_layout_for_key(self, key, sort_mode=None, reverse_sort=False, include_outer_crop=True):
        ref = self.read_frame(key, self.current_frame)
        if ref is None:
            return None
        ref = self._to_canvas_image(ref)
        H, W = ref.shape[:2]
        valid = [(rid, r) for rid, r in sorted(self.rois.items()) if r['rect'] is not None]
        if not self.compose_layout:
            if len(valid) == 0:
                return ref.copy()
            out = ref.copy()
            for rid, r in valid:
                x1, y1, x2, y2 = self.clamp_rect(r['rect'])
                color = self._color_with_alpha(r['color'])
                cv2.rectangle(out, (x1, y1), (x2, y2), color, self.line_thickness)
            return out
        # Determine sorting strategy
        eff_sort_mode = (sort_mode or self.sort_mode).lower()
        eff_reverse = bool(reverse_sort or self.sort_reverse)
        # Order ROIs either by spatial position (default) or by id
        if len(valid) > 1:
            if eff_sort_mode == 'id':
                valid.sort(key=lambda item: item[0], reverse=eff_reverse)
            elif eff_sort_mode == 'position':  # position-based
                if self.layout_mode in ['left', 'right']:
                    valid.sort(key=lambda item: ((item[1]['rect'][1] + item[1]['rect'][3]) / 2.0), reverse=eff_reverse)
                else:
                    valid.sort(key=lambda item: ((item[1]['rect'][0] + item[1]['rect'][2]) / 2.0), reverse=eff_reverse)
            else:
                log.warn(f"Unknown sort mode: {eff_sort_mode}; defaulting to id")
                valid.sort(key=lambda item: item[0], reverse=eff_reverse)
        if len(valid) == 0:
            return None
        block_line_th = max(1, int(round(self.line_thickness * self.layout_border_scale)))

        def _norm_single_pos(raw_pos: str):
            if raw_pos is None:
                return 'auto'
            s = str(raw_pos).strip().lower()
            if not s or s == 'auto':
                return 'auto'
            mapping = {
                'outer': 'outer',
                'tl': 'tl',
                'top-left': 'tl',
                'left-top': 'tl',
                'upper-left': 'tl',
                'tr': 'tr',
                'top-right': 'tr',
                'right-top': 'tr',
                'upper-right': 'tr',
                'bl': 'bl',
                'bottom-left': 'bl',
                'left-bottom': 'bl',
                'lower-left': 'bl',
                'br': 'br',
                'bottom-right': 'br',
                'right-bottom': 'br',
                'lower-right': 'br',
            }
            return mapping.get(s, 'auto')

        pos = _norm_single_pos(self.single_crop_position) if len(valid) == 1 else 'auto'

        # Single ROI: keep the dedicated in-image placement path unless the user
        # explicitly requested outer layout, in which case we fall through to the
        # multi-ROI composition branch below with a single crop.
        if len(valid) == 1 and not (pos == 'outer' and include_outer_crop):
            rid, r0 = valid[0]
            rect = r0['rect']
            x1, y1, x2, y2 = self.clamp_rect(rect)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            ox = W - 1 - cx
            oy = H - 1 - cy
            # crop from the same image (not method grid)
            img = ref
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return ref.copy()
            gh, gw = crop.shape[:2]
            # Scale with display_scale; outer mode is allowed to expand the canvas.
            dscale = self.display_scale if hasattr(self, 'display_scale') and self.display_scale > 0 else 1.0
            scale = dscale
            if self.single_crop_position != 'outer':
                scale = min(dscale, min(W / max(gw, 1), H / max(gh, 1)))
            if scale != 1.0:
                crop = cv2.resize(crop, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                gh, gw = crop.shape[:2]
            # place at the nearest corner to mirrored center, or outside the image for outer mode
            corners = [(0, 0), (W - gw, 0), (0, H - gh), (W - gw, H - gh)]

            def dist2(ax, ay, bx, by):
                dx = ax - bx
                dy = ay - by
                return dx * dx + dy * dy

            if pos == 'outer':
                out = ref.copy()
                color = self._color_with_alpha(r0['color'])
                cv2.rectangle(out, (x1, y1), (x2, y2), color, self.line_thickness)
                return out
            if pos == 'tl':
                x0, y0 = 0, 0
            elif pos == 'tr':
                x0, y0 = W - gw, 0
            elif pos == 'bl':
                x0, y0 = 0, H - gh
            elif pos == 'br':
                x0, y0 = W - gw, H - gh
            else:
                best = min(corners, key=lambda c: dist2(c[0] + gw // 2, c[1] + gh // 2, ox, oy))
                x0, y0 = best
            out = ref.copy()
            out[y0:y0 + gh, x0:x0 + gw] = crop
            # draw ROI box on full image and crop box in final
            color = self._color_with_alpha(r0['color'])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.line_thickness)
            self.draw_inner_border(out, x0, y0, gw, gh, color, thickness=block_line_th)
            return out
        # Multiple ROIs: arrange per layout_mode
        crops = []
        roi_info = []
        for rid, r in valid:
            x1, y1, x2, y2 = self.clamp_rect(r['rect'])
            c = ref[y1:y2, x1:x2]
            if c is not None and c.size > 0:
                crops.append(c)
                roi_info.append({'id': rid, 'color': self._color_with_alpha(r['color'])})
        if len(crops) == 0:
            return ref.copy()
        mode = self.layout_mode

        # Utility: even split counts for k groups
        def _even_counts(total, k):
            base = total // k
            rem = total % k
            return [base + 1] * rem + [base] * (k - rem)

        # Evaluate column/row groupings by maximizing sum(scales) / extent
        def _best_counts_lr(crops_list):
            max_score = -1.0
            best = [len(crops_list)]
            max_h = max(c.shape[0] for c in crops_list)
            min_scale = max(self.layout_min_scale, 0.01)
            for k in range(1, len(crops_list) + 1):
                counts = _even_counts(len(crops_list), k)
                idx = 0
                col_ws = []
                col_hs = []
                scale_sum = 0.0
                valid = True
                for cnum in counts:
                    subset = crops_list[idx:idx + cnum]
                    idx += cnum
                    h0s = []
                    w0s = []
                    for g in subset:
                        gh, gw = g.shape[:2]
                        s0 = W / max(gw, 1)
                        h0 = gh * s0
                        w0 = gw * s0
                        h0s.append(h0)
                        w0s.append(w0)
                    Hp = sum(h0s)
                    if Hp <= 0:
                        valid = False
                        break
                    s_global = (H - self.layout_gap) / Hp
                    final_scales = [(W / max(subset[i].shape[1], 1)) * s_global for i in range(len(subset))]
                    if min(final_scales) < min_scale - 1e-9:
                        valid = False
                        break
                    scale_sum += sum(final_scales)
                    col_w = max(w * s_global for w in w0s)
                    col_h = sum(h * s_global for h in h0s) + self.layout_gap * (
                        len(subset) - 1 if len(subset) > 0 else 0)
                    col_ws.append(col_w)
                    col_hs.append(col_h)
                if not valid:
                    continue
                block_w_total = sum(col_ws) + self.layout_gap * (len(col_ws) - 1 if len(col_ws) > 0 else 0)
                block_h_total = max(col_hs) if col_hs else 1.0
                score = scale_sum / max(block_w_total, 1e-6)
                if score > max_score:
                    max_score = score
                    best = counts
            return best

        def _best_counts_tb(crops_list):
            max_score = -1.0
            best = [len(crops_list)]
            max_w = max(c.shape[1] for c in crops_list)
            min_scale = max(self.layout_min_scale, 0.01)
            for k in range(1, len(crops_list) + 1):
                counts = _even_counts(len(crops_list), k)
                idx = 0
                row_ws = []
                row_hs = []
                scale_sum = 0.0
                valid = True
                for cnum in counts:
                    subset = crops_list[idx:idx + cnum]
                    idx += cnum
                    w0s = []
                    h0s = []
                    for g in subset:
                        gh, gw = g.shape[:2]
                        s0 = H / max(gh, 1)
                        h0 = gh * s0
                        w0 = gw * s0
                        w0s.append(w0)
                        h0s.append(h0)
                    Wp = sum(w0s)
                    if Wp <= 0:
                        valid = False
                        break
                    s_global = (W - self.layout_gap) / Wp
                    final_scales = [(H / max(subset[i].shape[0], 1)) * s_global for i in range(len(subset))]
                    if min(final_scales) < min_scale - 1e-9:
                        valid = False
                        break
                    scale_sum += sum(final_scales)
                    row_w = sum(w * s_global for w in w0s) + self.layout_gap * (
                        len(subset) - 1 if len(subset) > 0 else 0)
                    row_h = max(h * s_global for h in h0s)
                    row_ws.append(row_w)
                    row_hs.append(row_h)
                if not valid:
                    continue
                block_w_total = max(row_ws) if row_ws else 1.0
                block_h_total = sum(row_hs) + self.layout_gap * (len(row_hs) - 1 if len(row_hs) > 0 else 0)
                score = scale_sum / max(block_h_total, 1e-6)
                if score > max_score:
                    max_score = score
                    best = counts
            return best

        if mode in ['left', 'right']:
            counts = _best_counts_lr(crops)

            col_blocks = []
            idx = 0
            inner_pad = self.layout_gap
            for cnum in counts:
                subset = crops[idx:idx + cnum]
                subset_info = roi_info[idx:idx + cnum]
                idx += cnum
                resized0 = []
                heights_scaled = []
                for g in subset:
                    gh, gw = g.shape[:2]
                    s0 = W / max(gw, 1)
                    w0 = int(round(gw * s0))
                    h0 = int(round(gh * s0))
                    img0 = cv2.resize(g, (w0, h0), interpolation=cv2.INTER_NEAREST)
                    heights_scaled.append(h0)
                    resized0.append(img0)
                Hp = sum(heights_scaled)
                s_global = (H - inner_pad * (len(resized0) - 1 if len(resized0) > 0 else 0)) / max(Hp, 1)
                resized = []
                for img0 in resized0:
                    h0, w0 = img0.shape[:2]
                    h1 = max(1, int(round(h0 * s_global)))
                    w1 = max(1, int(round(w0 * s_global)))
                    img1 = cv2.resize(img0, (w1, h1), interpolation=cv2.INTER_NEAREST)
                    resized.append(img1)
                block_w = max(img.shape[1] for img in resized)
                block_h = sum(img.shape[0] for img in resized) + inner_pad * (
                    len(resized) - 1 if len(resized) > 0 else 0)
                block = self._blank_canvas(block_h, block_w)
                y = 0
                for idx_img, img in enumerate(resized):
                    h, w = img.shape[:2]
                    x = (block_w - w) // 2
                    block[y:y + h, x:x + w] = img
                    color = subset_info[idx_img]['color'] if idx_img < len(subset_info) else (255, 255, 255)
                    self.draw_inner_border(block, x, y, w, h, color, thickness=block_line_th)
                    y += h
                    if idx_img < len(resized) - 1:
                        y += inner_pad
                col_blocks.append(block)

            pad = self.layout_gap
            block_w_total = sum(b.shape[1] for b in col_blocks) + pad * (
                len(col_blocks) - 1 if len(col_blocks) > 0 else 0)
            block_h_total = max(b.shape[0] for b in col_blocks) if col_blocks else 0
            canvas_h = max(H, block_h_total)
            canvas_w = W + block_w_total + pad
            out = self._blank_canvas(canvas_h, canvas_w)
            fy = (canvas_h - H) // 2
            if mode == 'left':
                fx = block_w_total + pad
                bx = 0
            else:  # right
                fx = 0
                bx = W + pad
            by = (canvas_h - block_h_total) // 2 if block_h_total > 0 else 0
            base = ref.copy()
            for _, r in valid:
                x1, y1, x2, y2 = self.clamp_rect(r['rect'])
                cv2.rectangle(base, (x1, y1), (x2, y2), self._color_with_alpha(r['color']), self.line_thickness)
            out[fy:fy + H, fx:fx + W] = base
            x_cursor = bx
            for b in col_blocks:
                h, w = b.shape[:2]
                y_cursor = by + (block_h_total - h) // 2
                out[y_cursor:y_cursor + h, x_cursor:x_cursor + w] = b
                x_cursor += w + pad
            return out
        else:  # 'top' or 'bottom'
            counts = _best_counts_tb(crops)

            row_blocks = []
            idx = 0
            inner_pad = self.layout_gap
            for cnum in counts:
                subset = crops[idx:idx + cnum]
                subset_info = roi_info[idx:idx + cnum]
                idx += cnum
                resized0 = []
                widths_scaled = []
                for g in subset:
                    gh, gw = g.shape[:2]
                    s0 = H / max(gh, 1)
                    h0 = int(round(gh * s0))
                    w0 = int(round(gw * s0))
                    img0 = cv2.resize(g, (w0, h0), interpolation=cv2.INTER_NEAREST)
                    widths_scaled.append(w0)
                    resized0.append(img0)
                Wp = sum(widths_scaled)
                s_global = (W - inner_pad * (len(resized0) - 1 if len(resized0) > 0 else 0)) / max(Wp, 1)
                resized = []
                for img0 in resized0:
                    h0, w0 = img0.shape[:2]
                    w1 = max(1, int(round(w0 * s_global)))
                    h1 = max(1, int(round(h0 * s_global)))
                    img1 = cv2.resize(img0, (w1, h1), interpolation=cv2.INTER_NEAREST)
                    resized.append(img1)
                block_w = sum(img.shape[1] for img in resized) + inner_pad * (
                    len(resized) - 1 if len(resized) > 0 else 0)
                block_h = max(img.shape[0] for img in resized)
                block = self._blank_canvas(block_h, block_w)
                x = 0
                for idx_img, img in enumerate(resized):
                    h, w = img.shape[:2]
                    y = (block_h - h) // 2
                    block[y:y + h, x:x + w] = img
                    color = subset_info[idx_img]['color'] if idx_img < len(subset_info) else (255, 255, 255)
                    self.draw_inner_border(block, x, y, w, h, color, thickness=block_line_th)
                    x += w
                    if idx_img < len(resized) - 1:
                        x += inner_pad
                row_blocks.append(block)

            pad = self.layout_gap
            block_w_total = max(b.shape[1] for b in row_blocks) if row_blocks else 0
            block_h_total = sum(b.shape[0] for b in row_blocks) + pad * (
                len(row_blocks) - 1 if len(row_blocks) > 0 else 0)
            canvas_w = max(W, block_w_total)
            canvas_h = H + block_h_total + pad
            out = self._blank_canvas(canvas_h, canvas_w)
            fx = (canvas_w - W) // 2
            if mode == 'top':
                fy = block_h_total + pad
                by = 0
            else:  # bottom
                fy = 0
                by = H + pad
            bx = (canvas_w - block_w_total) // 2 if block_w_total > 0 else 0
            base = ref.copy()
            for _, r in valid:
                x1, y1, x2, y2 = self.clamp_rect(r['rect'])
                cv2.rectangle(base, (x1, y1), (x2, y2), self._color_with_alpha(r['color']), self.line_thickness)
            out[fy:fy + H, fx:fx + W] = base
            y_cursor = by
            for b in row_blocks:
                h, w = b.shape[:2]
                x_cursor = bx + (block_w_total - w) // 2
                out[y_cursor:y_cursor + h, x_cursor:x_cursor + w] = b
                y_cursor += h + pad
            return out

    def update_display(self):
        ref = self.read_frame(self.reference_key, self.current_frame)
        if ref is None:
            return
        now_ts = time.time()
        layout_idle = (now_ts - float(self._last_input_activity_ts)) >= float(self._layout_debounce_sec)
        allow_secondary_update = (self.mode == 'idle') or layout_idle
        if allow_secondary_update:
            self._emit_event('before_update_display')
        
        canvas = ref.copy()
        overlay = canvas.copy()
        overlay_applied = False
        if self.mode in ['selection', 'position'] and len(self.rois) > 0:
            for rid, r in sorted(self.rois.items()):
                if r['rect'] is None:
                    continue
                x1, y1, x2, y2 = self.clamp_rect(r['rect'])
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), r['color'], thickness=-1)
                    overlay_applied = True
                if self.mode == 'selection' and self.active_roi == rid:
                    self.draw_dashed_rect(canvas, (x1, y1), (x2, y2), r['color'], thickness=self.line_thickness)
                else:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), r['color'], self.line_thickness)
                if self.active_roi == rid:
                    cx = x1 + 20
                    cy = max(20, y1 - 20)
                    self.draw_circled_label(canvas, (cx, cy), rid, r['color'])
                else:
                    cv2.putText(canvas, f"{rid}", (x1 + 3, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, r['color'],
                                2)
        if overlay_applied:
            alpha = max(0.0, min(1.0, float(self.preview_mask_alpha)))
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)
        # Top-left image name (without extension) for the interactive reference display
        try:
            ref_path = self.image_files[self.reference_key][self.current_frame]
            ref_name = os.path.basename(ref_path).rsplit('.', 1)[0]
            cv2.putText(canvas, ref_name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
        except Exception:
            pass
        cv2.imshow(self.window_main, canvas)

        def final_layout_blank():
            blank = np.zeros((min(240, self.height), min(320, self.width), 3), dtype=np.uint8)
            cv2.putText(blank, "Final layout idle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return blank

        if self.mode == 'idle':
            for w in list(self.grid_windows):
                cv2.destroyWindow(w)
            self.grid_windows.clear()
            self._cached_grid_views = {}
            self._cached_final_view = None
            self._cached_layout_signature = None

            # Final layout window (preview)
            blank = final_layout_blank()
            cv2.imshow(self.window_final, blank)
        else:
            if allow_secondary_update:
                layout_sig = self._layout_state_signature()
                layout_changed = (layout_sig != self._cached_layout_signature)
                should_rebuild_layout = (self._cached_layout_signature is None) or layout_changed

                if should_rebuild_layout:
                    rebuilt_grids = {}
                    valid_rois = [(rid, r) for rid, r in sorted(self.rois.items()) if r['rect'] is not None]
                    for rid, r in valid_rois:
                        name = f"{self.window_grid} ROI {rid}"
                        grid = self.build_grid_for_rect(r['rect'], roi_color=r.get('color'))
                        if grid is None:
                            continue
                        if self.display_scale and self.display_scale != 1.0:
                            grid = cv2.resize(
                                grid,
                                dsize=None,
                                fx=self.display_scale,
                                fy=self.display_scale,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        rebuilt_grids[name] = grid

                    preview_key = self.preview_key or self.reference_key
                    final = self.build_final_layout_for_key(
                        preview_key,
                        sort_mode=self.sort_mode,
                        reverse_sort=self.sort_reverse,
                    )
                    if final is not None and final.ndim == 3 and final.shape[2] == 4:
                        final = cv2.cvtColor(final, cv2.COLOR_BGRA2BGR)

                    self._cached_grid_views = rebuilt_grids
                    self._cached_final_view = final
                    self._cached_layout_signature = layout_sig

                needed = set(self._cached_grid_views.keys())
                for name, grid_view in self._cached_grid_views.items():
                    cv2.imshow(name, grid_view)

                # Destroy windows no longer needed
                obsolete = self.grid_windows - needed
                for w in obsolete:
                    cv2.destroyWindow(w)
                self.grid_windows = needed

                if self._cached_final_view is None:
                    blank = final_layout_blank()
                    cv2.imshow(self.window_final, blank)
                else:
                    cv2.imshow(self.window_final, self._cached_final_view)

        if self.show_all_method_images:
            full_grid = self.build_full_image_grid()
            if full_grid is None:
                blank = final_layout_blank()
                cv2.imshow(self.method_full_image_window, blank)
            else:
                display_scale = float(self.full_image_scale or 0.5)
                if display_scale <= 0:
                    display_scale = 0.5
                if display_scale != 1.0:
                    full_grid = cv2.resize(
                        full_grid,
                        dsize=None,
                        fx=display_scale,
                        fy=display_scale,
                        interpolation=cv2.INTER_NEAREST,
                    )
                if full_grid.ndim == 3 and full_grid.shape[2] == 4:
                    full_grid = cv2.cvtColor(full_grid, cv2.COLOR_BGRA2BGR)
                cv2.imshow(self.method_full_image_window, full_grid)
        else:
            try:
                cv2.destroyWindow(self.method_full_image_window)
            except Exception:
                pass
        
        if allow_secondary_update:
            self._emit_event('after_update_display')

        self.needs_update = False

    def _serialize_rois(self):
        """Return ordered ROI tuples (id, x1, y1, x2, y2)."""
        entries = []
        for rid, r in sorted(self.rois.items()):
            rect = r.get('rect')
            if rect is None:
                continue
            x1, y1, x2, y2 = self.clamp_rect(rect)
            entries.append((rid, x1, y1, x2, y2))
        return entries

    def save_rois_to_txt(self, out_path):
        entries = self._serialize_rois()
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("# roi_id x1 y1 x2 y2\n")
                for rid, x1, y1, x2, y2 in entries:
                    f.write(f"{rid} {x1} {y1} {x2} {y2}\n")
                if not entries:
                    f.write("# no roi defined\n")
            if entries:
                log.success(f"Saved ROI info to {out_path}")
            else:
                log.warn(f"Saved empty ROI info to {out_path}")
            return True
        except Exception as e:
            log.error(f"Failed to save ROI info: {e}")
            traceback.print_exc()
            return False

    def load_rois_from_txt(self, path):
        if not path:
            return False
        if not os.path.exists(path):
            log.error(f"ROI file not found: {path}")
            return False
        loaded = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.replace(',', ' ').split()
                    if len(parts) < 5:
                        continue
                    try:
                        rid = int(parts[0])
                        x1, y1, x2, y2 = [int(float(v)) for v in parts[1:5]]
                        loaded.append((rid, (x1, y1, x2, y2)))
                    except Exception:
                        continue
        except Exception as e:
            log.error(f"Failed to read ROI file {path}: {e}")
            return False

        if not loaded:
            log.warn(f"ROI file {path} has no valid entries")
            return False

        self.rois = {}
        for rid, rect in loaded:
            x1, y1, x2, y2 = self.clamp_rect(rect)
            self.rois[rid] = {
                'rect': (x1, y1, x2, y2),
                'color': self.color_for_id(rid, total_count=len(self.rois) + 1)
            }
        self.active_roi = loaded[0][0]
        self.mode = 'position'
        self.selection_start = None
        self.undo_manager.undo_stack.clear()
        self.undo_manager.redo_stack.clear()
        self.request_update()
        log.success(f"Loaded {len(loaded)} ROIs from {path}")
        return True

    def save(self, pair, dataset):
        # Prepare timestamped output directory: output/<timestamp>/<dataset>/
        if self.save_session_ts is None:
            self.save_session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.join(self.output_folder, self.save_session_ts, dataset)
        os.makedirs(base_dir, exist_ok=True)

        # Use reference image stem as the per-frame output folder name
        try:
            ref_file_for_dir = self.image_files[self.reference_key][self.current_frame]
            ref_name = os.path.basename(ref_file_for_dir).rsplit('.', 1)[0]
            frame_dir = os.path.join(base_dir, ref_name)
        except Exception:
            ref_name = ""
            frame_dir = base_dir
        os.makedirs(frame_dir, exist_ok=True)

        # Determine method keys (exclude GT/input if present)
        # method_keys = [k for k in self.image_files.keys() if k not in ['GT', 'input']]
        method_keys = [k for k in self.image_files.keys()]
        if len(method_keys) == 0:
            method_keys = list(self.image_files.keys())

        # For each method: save final layout, per-ROI crops, and original image
        for m in method_keys:
            try:
                img_file = self.image_files[m][self.current_frame]
            except Exception:
                log.warn(f"Method {m} has no image for current frame; skipping")
                continue
            img_name = os.path.basename(img_file).rsplit('.', 1)[0]
            if ref_name != img_name:
                m_suffix = f"_{img_name}"
            else:
                m_suffix = ""
            m_dir = frame_dir
            img = self.read_frame(m, self.current_frame)
            if img is None:
                log.warn(f"Skip method {m}{m_suffix}: image read failed")
                continue
            H, W = img.shape[:2]

            # Save original image
            orig_out = os.path.join(m_dir, f"orig_{m}{m_suffix}.png")
            cv2.imwrite(orig_out, img)

            # Save final layout composed for this method
            final = self.build_final_layout_for_key(m, sort_mode=self.sort_mode, reverse_sort=self.sort_reverse)
            if final is not None:
                final_out = os.path.join(m_dir, f"final_{m}{m_suffix}.png")
                cv2.imwrite(final_out, final)
            else:
                log.warn(f"Final layout for {m}{m_suffix} is empty; skipping")

            if self.save_selection_image and len([r for r in self.rois.values() if r.get('rect') is not None]) == 1:
                selection_only = self.build_final_layout_for_key(
                    m,
                    sort_mode=self.sort_mode,
                    reverse_sort=self.sort_reverse,
                    include_outer_crop=False,
                )
                if selection_only is not None:
                    selection_out = os.path.join(m_dir, f"selection_{m}{m_suffix}.png")
                    cv2.imwrite(selection_out, selection_only)

            # Save each ROI crop for this method
            for rid, r in sorted(self.rois.items()):
                rect = r.get('rect')
                if rect is None:
                    continue
                x1, y1, x2, y2 = self.clamp_rect(rect)
                # Clamp to image bounds and ensure non-empty crop
                x1 = max(0, min(W - 1, x1))
                x2 = max(0, min(W, x2))
                y1 = max(0, min(H - 1, y1))
                y2 = max(0, min(H, y2))
                if x2 <= x1 + 0 or y2 <= y1 + 0:
                    continue
                crop = img[y1:y2, x1:x2]
                
                # Apply layout_bg_color background if crop size doesn't match ROI bounds
                roi_w = x2 - x1
                roi_h = y2 - y1
                if crop.shape[0] != roi_h or crop.shape[1] != roi_w:
                    if self.layout_use_alpha:
                        bg_color = self.layout_bg_color if len(self.layout_bg_color) >= 4 else (self.layout_bg_color[0] if len(self.layout_bg_color) >= 1 else 0, self.layout_bg_color[1] if len(self.layout_bg_color) >= 2 else 0, self.layout_bg_color[2] if len(self.layout_bg_color) >= 3 else 0, 0)
                        padded = np.full((roi_h, roi_w, 4), bg_color, dtype=np.uint8)
                        if crop.ndim == 3 and crop.shape[2] == 3:
                            crop_rgba = np.dstack([crop, np.full((crop.shape[0], crop.shape[1]), 255, dtype=np.uint8)])
                        else:
                            crop_rgba = crop if crop.ndim == 3 and crop.shape[2] >= 3 else np.dstack([crop, np.full((crop.shape[0], crop.shape[1]), 255, dtype=np.uint8)])
                        ph = min(roi_h, crop_rgba.shape[0])
                        pw = min(roi_w, crop_rgba.shape[1])
                        padded[:ph, :pw] = crop_rgba[:ph, :pw]
                        crop = padded
                    else:
                        bg_color = self.layout_bg_color[:3] if len(self.layout_bg_color) >= 3 else (0, 0, 0)
                        padded = np.full((roi_h, roi_w, 3), bg_color, dtype=np.uint8)
                        ph = min(roi_h, crop.shape[0])
                        pw = min(roi_w, crop.shape[1])
                        padded[:ph, :pw] = crop[:ph, :pw]
                        crop = padded
                
                crop_out = os.path.join(m_dir, f"crop_roi{rid}_{m}{m_suffix}.png")
                cv2.imwrite(crop_out, crop)

        # Save grid images (per-ROI and all-ROIs) once per frame
        try:
            valid_rois = [(rid, r) for rid, r in sorted(self.rois.items()) if r.get('rect') is not None]
            for rid, r in valid_rois:
                grid = self.build_grid_for_rect(r['rect'], roi_color=r.get('color'))
                if grid is None:
                    continue
                grid_out = os.path.join(frame_dir, f"grid_roi{rid}.png")
                cv2.imwrite(grid_out, grid)

            grid_all = self.build_grid()
            if grid_all is not None:
                grid_all_out = os.path.join(frame_dir, "grid_all_rois.png")
                cv2.imwrite(grid_all_out, grid_all)
        except Exception as e:
            log.warn(f"Failed to save grid images: {e}")
            traceback.print_exc()
        try:
            ref_file = self.image_files[self.reference_key][self.current_frame]
            ref_stem = os.path.basename(ref_file).rsplit('.', 1)[0]
            roi_out = os.path.join(base_dir, f"roi_{ref_stem}.txt")
        except Exception:
            roi_out = os.path.join(base_dir, "roi_info.txt")
        self.save_rois_to_txt(roi_out)
        log.success(f"Saved outputs under: {base_dir}")

    def run(self, pair):
        cv2.namedWindow(self.window_main)
        cv2.setMouseCallback(self.window_main, self.on_mouse)
        self.save_label = pair

        ACTION_WIDTH = 7 + 2
        log.banner("Interactive Crop Comparator", level=1)
        log.info(f"Mouse: selection mode to draw rect; position mode to move")
        log.info(
            f"[{'Basic':^{ACTION_WIDTH}}] action: "
            + f"{log.style_key('a')} {log.style_underline('a')}dd next ROI, "
            + f"{log.style_key('1-9')} add/select ROI id; press same id to enter selection mode, "
            + f"{log.style_key('d')} {log.style_underline('d')}uplicate new ROI with active size, "
            + f"{log.style_key('Del')} delete active ROI"
        )
        log.info(
            f"[{'Status':^{ACTION_WIDTH}}] action: "
            + f"{log.style_key('[')}/{log.style_key('p')} {log.style_underline('p')}rev image, "
            + f"{log.style_key(']')}/{log.style_key('n')} {log.style_underline('n')}ext image, "
            + f"{log.style_key('s')} {log.style_underline('s')}ave outputs, "
            + f"{log.style_key('q')}/{log.style_key('Esc')} {log.style_underline('q')}uit"
        )
        log.info(
            f"[{'Advance':^{ACTION_WIDTH}}] action: "
            + f"{log.style_key('+')}/{log.style_key('=')} zoom in, "
            + f"{log.style_key('-')}/{log.style_key('_')} zoom out, "
            + f"{log.style_key('r')} clear all rois, "
            + f"{log.style_key('z')} undo, "
            + f"{log.style_key('y')} redo"
        )
        log.info(
            f"[{'Layout':^{ACTION_WIDTH}}] action: "
            + f"{log.style_key('←')} {log.style_mode('left')} (crops stack left), "
            + f"{log.style_key('↑')} {log.style_mode('top')} (crops stack above), "
            + f"{log.style_key('→')} {log.style_mode('right')} (crops stack right), "
            + f"{log.style_key('↓')} {log.style_mode('bottom')} (crops stack below)"
        )
        log.info(
            f"[{'Extra':^{ACTION_WIDTH}}] action: "
            + f"{log.style_key('Tab')} show/hide all method full-image grid, "   # Tab 需要按下两次才能执行相关操作，原因是 Tab 和窗口的某个操作冲突了，暂时没找到解决方法
            + f"{log.style_key('i')} {log.style_underline('i')}dle toggle, "
            + f"{log.style_key('Shift+1-9')} duplicate to id or copy size (see README), "
            + f"{log.style_key('Enter')} switch dataset/group, "
            + f"{log.style_key('Space')} jump to image"
        )
        log.banner("Logs", level=2)
        # Default: enter ROI 1 selection mode at startup
        if len(self.rois) == 0:
            self.add_roi()
        self.request_update()

        def _normalize_key(raw_key):
            if raw_key in (2490368, 2621440, 2424832, 2555904):  # ↑ ↓ ← →
                arrow_map = {2424832: 81, 2490368: 82, 2555904: 83, 2621440: 84}
                return arrow_map[raw_key]
            return raw_key & 0xFF if raw_key >= 0 else raw_key

        exit_keys = {ord('q'), 27}

        self._try_runtime_event_init()
        while True:
            keys = []
            first_key = cv2.waitKeyEx(1)
            if first_key >= 0:
                keys.append(_normalize_key(first_key))
                # Drain queued key events so all commands execute before one render.
                for _ in range(64):
                    k = cv2.waitKeyEx(1)
                    if k < 0:
                        break
                    keys.append(_normalize_key(k))

            if any(k in exit_keys for k in keys):
                log.info("Quitting.")
                break

            if keys:
                self._last_input_activity_ts = time.time()

            if self.mode != 'idle':
                now_ts = time.time()
                layout_idle = (now_ts - float(self._last_input_activity_ts)) >= float(self._layout_debounce_sec)
                if layout_idle:
                    if self._cached_layout_signature != self._layout_state_signature():
                        self.request_update()

            for key in keys:
                handled = self.dispatcher.dispatch(key)
                if handled:
                    continue
                if key in [13, 10]:
                    # Enter: prompt to switch dataset (optionally group/dataset)
                    try:
                        prompt = "Enter dataset (or group/dataset): "
                        text = input(prompt).strip()
                    except Exception:
                        text = ""
                    if text:
                        if '/' in text:
                            ng, nd = text.split('/', -1)
                        else:
                            ng, nd = None, text

                        if not nd:
                            log.error("Dataset is empty; aborted switch")
                        else:
                            if self.rebuild_dataset(nd, ng):
                                self.request_update()
                    else:
                        log.info("Dataset switch cancelled (empty input)")
                elif key == 32:
                    # Space: prompt to jump to image by name
                    try:
                        text = input("Enter image name to jump (stem or filename): ").strip()
                    except Exception:
                        text = ""
                    if text:
                        if self.jump_to_image_by_name(text):
                            self.request_update()
                    else:
                        log.info("Image jump cancelled (empty input)")

            # Events/plot refresh are processed after key batch, so image commands take priority.
            self._emit_event('loop_tick')
            if self.needs_update:
                self.update_display()

        self._emit_event('on_shutdown')
        cv2.destroyAllWindows()


def _parse_exclude_methods(raw):
    if not raw:
        return set()
    return {p.strip() for p in str(raw).replace(',', ' ').split() if p.strip()}


def _apply_exclude(methods, exclude_set):
    if not exclude_set:
        return methods, []
    removed = [m for m in methods if m in exclude_set]
    kept = [m for m in methods if m not in exclude_set]
    return kept, removed


if __name__ == "__main__":
    log.banner("LLIE Results - Compare Tool")

    import argparse

    parser = argparse.ArgumentParser()

    # --- Data source & paths ---
    g_data = parser.add_argument_group(
        title='Data Source & Paths',
        description='Select the data source (local/external) and related root/output paths.'
    )
    g_data.add_argument('--source', choices=['local', 'external'], default='local', type=str,
                        help='Data source: local uses the workspace structure under --root; external uses /data/user paths with --pair videos.')
    g_data.add_argument('--root', '-r', default=r'./examples/', type=str,
                        help='Workspace root containing method folders (local mode only). Example: /mnt/user/results/LLIE-results')
    g_data.add_argument('--output', '-o', default='./crop_grids/', type=str,
                        help='Root output folder. Files are saved under output/<timestamp>/<dataset>/...')

    # --- Dataset selection ---
    g_dataset = parser.add_argument_group(
        title='Dataset',
        description='Specify dataset location (group/dataset) and the video sequence (pair) for external mode.'
    )
    g_dataset.add_argument('--group', '-g', default=None, type=str,
                           help='Dataset group folder under each method (e.g., LOLv2-real+, SDSD-indoor+). Hyphens are auto-resolved across methods.')
    g_dataset.add_argument('--dataset', '-ds', default=None, type=str,
                           help='Leaf dataset folder under the group (e.g., DarkFace, DICM, LOL, SDSD-indoor).')
    g_dataset.add_argument('--pair', '-p', default=None, type=str,
                           help='Video pair/sequence name (external mode only), e.g., pair13. Ignored in local mode.')

    # --- Method discovery / filtering ---
    g_methods = parser.add_argument_group(
        title='Methods',
        description='Control how methods are discovered/matched, and optionally exclude certain methods.'
    )
    g_methods.add_argument('--structure', default='auto',
                           choices=['auto', 'group-dataset-pair', 'group-dataset', 'dataset-only', 'flat', 'shared'],
                           help='Folder structure layout: auto (default), group-dataset-pair, group-dataset, dataset-only, flat (images directly under method), or shared (image-id folders containing per-method files such as img1/methodA.png).')
    g_methods.add_argument('--exclude', '-x', default=None, type=str,
                           help='Comma/space separated method names to exclude.')

    # --- Interaction & grid visualization ---
    g_view = parser.add_argument_group(
        title='Interaction & Grid View',
        description='Interaction mode and per-ROI grid visualization settings (columns, gap, magnification).'
    )
    g_view.add_argument('--columns', '-c', default=None, type=lambda x: int(x) if x else None,
                        help='Number of columns in the per-ROI method grid view (rows are computed automatically). If not specified and there are >=9 methods, auto-computed as (num_methods+1)//2.')
    g_view.add_argument('--grid-gap', default=2, type=int,
                        help='Gap (in pixels) between tiles in per-ROI method grid windows (default: 2).')
    g_view.add_argument('--magnify', '--scale', default=2, type=float,
                        help='Display-only magnification for crop grid windows. Final preview is not globally scaled; multi-ROI ignores this.')
    g_view.add_argument('--full-image-scale', default=0.3, type=float,
                        help='Display scale for the Tab full-image grid window (default: 0.3).')

    # --- Final layout preview ---
    g_layout = parser.add_argument_group(
        title='Final Layout Preview',
        description='Final preview layout (switchable via arrow keys), preview image key, and composition/background/gap settings.'
    )
    g_layout.add_argument('--layout', default='right', type=str, choices=['left', 'top', 'right', 'bottom'],
                          help='Final layout preview mode. Use arrow keys at runtime: ← left, ↑ top, → right, ↓ bottom..')
    g_layout.add_argument('--preview', default=None, type=str,
                          help='Image key to show in Final Layout preview (default: reference key). Keys come from method names or GT/input if present.')
    g_layout.add_argument('--compose-layout', dest='compose_layout', action='store_true', default=True,
                          help='Enable side-by-side crop composition in the final layout (default: on).')
    g_layout.add_argument('--no-compose-layout', dest='compose_layout', action='store_false',
                          help='Disable composition; final layout shows only the reference image with ROI boxes.')
    g_layout.add_argument('--layout-border-scale', default=2.0, type=float,
                          help='Thickness multiplier applied only to crop block borders in final layout (default: 2.0).')
    g_layout.add_argument('--layout-gap', default=10, type=int,
                          help='Gap (in pixels) between base image and crop block, and between crops themselves (default: 10).')
    g_layout.add_argument('--layout-bg-color', default='transparent', type=str,
                          help='Padding/background color as R,G,B[,A] for final layout gaps, or "transparent" (default). e.g., 0,0,0 or 255,255,255,255.')
    g_layout.add_argument('--single-crop-position', default='auto', type=str,
                            help='When there is only 1 ROI/crop, control where that crop is placed in the final layout. '
                               'Supported: auto (default), outer, tl/tr/bl/br, top-left/top-right/bottom-left/bottom-right.')

    # --- ROI / drawing ---
    g_roi = parser.add_argument_group(
        title='ROI',
        description='ROI drawing thickness and an optional ROI preload file.'
    )
    g_roi.add_argument('--thickness', '-t', default=2, type=int,
                       help='Line thickness for ROI boxes and crop borders (default: 2; minimum: 1).')
    g_roi.add_argument('--roi-file', default=None, type=str,
                       help='Optional ROI txt file to preload; format per line: id x1 y1 x2 y2.')
    g_roi.add_argument('--save-selection-image', action='store_true',
                       help='When saving outputs, also save a selection-only image without the outer crop for single-ROI outer layouts.')

    # --- Logging ---
    g_log = parser.add_argument_group(
        title='Logging',
        description='Control log output (color on/off and verbosity level).'
    )
    g_log.add_argument('--no-color', action='store_true',
                       help='Disable ANSI colored logs (use plain text).')
    g_log.add_argument('--log-level', default='info', choices=['debug', 'info', 'warn', 'error'],
                       help='Logging level: debug|info|warn|error (default: info).')

    # --- Metrics curve panel ---
    g_metric = parser.add_argument_group(
        title='Metric Curves',
        description='Configure asynchronous metric-curve windows. Multiple metrics create multiple windows.'
    )
    g_metric.add_argument('--metrics', '-m', default='psnr', type=str,
                          help='Comma/space separated metric types (e.g., "psnr", "ssim,lpips", "niqe").')
    g_metric.add_argument('--metric-methods', default=8, type=int,
                          help='Worker threads per method (default: 8).')
    args = parser.parse_args()

    output_abs = os.path.abspath(args.output)

    pair = args.pair
    group = args.group
    dataset = args.dataset
    exclude_methods = _parse_exclude_methods(args.exclude)

    file_path = "./methods.txt"
    methods = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            methods = [line.strip() for line in f.read().strip().splitlines() if line.strip()]

    input_folder = {}
    if args.source == 'external':
        if not methods:
            raise ValueError("methods.txt is required for external source")
        methods, removed = _apply_exclude(methods, exclude_methods)
        if removed:
            log.info(f"Excluded methods: {removed}")
        if not methods:
            raise ValueError("No methods left after applying --exclude")
        input_folder = {m: f"/data/user/results/{m}/{dataset}/pred/{pair}" for m in methods}

        # Optional GT/input reference if available
        gt_path = ""
        gt_path_exist = False
        for phase in ['test', 'eval']:
            for gt in ['GT', 'high']:
                gt_path = f"/data/user/datasets/{dataset}/{phase}/{gt}/{pair}"
                if os.path.exists(gt_path):
                    gt_path_exist = True
                    break
            if gt_path_exist:
                break
        if gt_path_exist:
            input_folder['GT'] = gt_path
            input_folder['input'] = gt_path.replace('GT', 'input').replace('high', 'low')
    else:
        root = args.root
        if not methods:
            # auto-discover methods by listing directories in root
            try:
                discovered = []
                skipped = []
                for d in os.listdir(root):
                    if d.startswith('.'):
                        continue
                    full = os.path.join(root, d)
                    if not os.path.isdir(full):
                        continue

                    if os.path.commonpath([os.path.abspath(full), output_abs]) == output_abs:
                        skipped.append(d)
                        continue

                    discovered.append(d)
                methods = discovered
                if skipped:
                    log.debug(f"Skipped output folder(s) while auto-discovering methods: {skipped}")
            except Exception:
                methods = []
        log.info(f"Visualizing methods: {methods}")

        if not methods:
            raise ValueError(f"No methods found under root {args.root}; please specify methods.txt or add method folders.")

        methods, removed = _apply_exclude(methods, exclude_methods)
        if removed:
            log.info(f"Excluded methods: {removed}")
        if not methods:
            raise ValueError("No methods left after applying --exclude")

        input_folder = discover_local_inputs(root, methods, group=group, dataset=dataset, pair=pair,
                                             structure=args.structure)
        if not input_folder:
            raise ValueError(
                "No valid method inputs found. Checked layouts: "
                "<method>/<group>/<dataset>[/<pair>], <method>/<dataset>, <method>/, "
                "and shared-folder layout (image-id folders containing per-method files)."
            )

    for name, source in input_folder.items():
        if isinstance(source, (list, tuple)):
            imgs = list(source)
            if len(imgs) == 0:
                log.error(f"No images found for {name} in shared-folder layout")
                raise ValueError(f"No images found for {name}")
        else:
            if not os.path.exists(source):
                log.error(f"Folder not exist: {name} -> {source}")
                raise ValueError(f"Folder not exist: {name} -> {source}")
            # ensure there are image files
            imgs = filter_hidden(glob_single_files(source, IMG_EXTS))
            if len(imgs) == 0:
                log.error(f"Folder {source} has no images (png/jpg/jpeg)")
                raise ValueError(f"Folder {source} has no images (png/jpg/jpeg)")

    # configure logger
    log.set_color_enabled(not args.no_color)
    log.set_level(args.log_level)

    # Parse layout background color CSV (R,G,B[,A]) or "transparent"
    try:
        if isinstance(args.layout_bg_color, str) and args.layout_bg_color.strip().lower() == 'transparent':
            layout_bg_color = (0, 0, 0, 0)
        else:
            parts = [int(x) for x in str(args.layout_bg_color).replace(' ', '').split(',') if x != '']
            if len(parts) not in (3, 4):
                raise ValueError()
            layout_bg_color = tuple(max(0, min(255, v)) for v in parts)
    except Exception:
        layout_bg_color = (0, 0, 0, 0)

    comparator = InteractiveCropComparator(
        input_folder,
        output_folder=args.output,
        reference_key=None,
        columns=args.columns,
        grid_gap=args.grid_gap,
        display_scale=args.magnify,
        full_image_scale=args.full_image_scale,
        line_thickness=args.thickness,
        layout_border_scale=args.layout_border_scale,
        layout_gap=args.layout_gap,
        layout_bg_color=layout_bg_color,
        compose_layout=args.compose_layout,
        save_selection_image=args.save_selection_image,
        current_group=args.group,
        current_dataset=dataset,
        event_on_init=lambda host: install_default_metrics_feature(
            host,
            metric_types=args.metrics,
            max_workers=os.cpu_count(),
            threads_per_methods=os.cpu_count() if args.metric_methods is None else args.metric_methods,
        ),
    )
    comparator.layout_mode = args.layout
    comparator.single_crop_position = args.single_crop_position
    comparator.preview_key = args.preview or ('GT' if 'GT' in input_folder else (
        'input' if 'input' in input_folder else next(iter(input_folder.keys()))))
    # Sorting options for final layout
    comparator.sort_mode = 'id'
    comparator.sort_reverse = False
    # Optional ROI preload
    if args.roi_file:
        comparator.load_rois_from_txt(os.path.expanduser(args.roi_file))
    # Log summary of loaded data
    try:
        shared_layout = any(isinstance(v, (list, tuple)) for v in input_folder.values())
        structure_used = 'shared' if shared_layout else args.structure
        has_pair = bool(str(pair).strip()) if pair is not None else False
        per_method_counts = {
            k: len(v) if isinstance(v, (list, tuple)) else 0
            for k, v in comparator.image_files.items()
        }
        counts = list(per_method_counts.values())
        if has_pair and counts and len(set(counts)) == 1:
            count_info = f"frames per method={log.style_num(str(counts[0]))}"
        elif has_pair and counts:
            count_info = (
                f"frames per method range="
                f"{log.style_num(str(min(counts)))}~{log.style_num(str(max(counts)))}"
            )
        elif counts and len(set(counts)) == 1:
            count_info = f"images per method on dataset={log.style_num(str(counts[0]))}"
        else:
            details = ', '.join(
                f"{log.style_path(k)}:{log.style_num(str(v))}"
                for k, v in sorted(per_method_counts.items())
            ) if per_method_counts else 'N/A'
            count_info = f"images per method on dataset={details}"

        log.info(
            f"Loaded {log.style_num(str(len(comparator.image_files)))} methods "
            f"using structure={log.style_mode(structure_used)}; "
            f"reference={log.style_key(comparator.reference_key)}; "
            f"{count_info}"
        )
        if has_pair and counts and len(set(counts)) > 1:
            details = ', '.join(f"{k}:{v}" for k, v in sorted(per_method_counts.items()))
            log.warn(f"Frame counts differ across methods: {details}")
        if shared_layout:
            log.note("Detected shared layout (image-id folders containing per-method files)")
    except Exception:
        traceback.print_exc()
        pass
    # Use dataset as label for saving in local mode; pair+dataset for external
    label_pair = pair if args.source == 'external' else dataset
    comparator.run(label_pair)
