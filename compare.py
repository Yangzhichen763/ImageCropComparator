import copy
import math
import os
import sys
from datetime import datetime
from glob import glob

import cv2
import numpy as np

try:
    from natsort import natsorted as _natsorted
except Exception:  # pragma: no cover - optional dependency
    _natsorted = sorted

IMG_EXTS = ['png', 'jpg', 'jpeg']

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
    from basic.utils.io import glob_single_files
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
        for file_extension in file_extensions:
            pattern = os.path.join(directory, f"**/*.{file_extension}")
            file_paths += _natsorted(glob(pattern, recursive=True))
        file_paths = [path_handler(os.path.normpath(path)) for path in file_paths]
        return file_paths


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
        return len(glob_single_files(directory, exts)) > 0
    except Exception:
        return False


def resolve_group_folder(method_root, target_group):
    """Resolve a group folder allowing hyphen/underscore mismatch."""
    tg = (target_group or '').replace('-', '').replace('_', '')
    if not tg:
        return None
    try:
        for d in os.listdir(method_root):
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
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    subdirs = sorted(subdirs)
    if not subdirs:
        return {}

    method_names = None
    for sd in subdirs:
        cur = os.path.join(root, sd)
        files = [f for f in os.listdir(cur) if os.path.isfile(os.path.join(cur, f))]
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


class InteractiveCropComparator:
    def __init__(
            self,
            input_folders,
            output_folder,
            reference_key=None,
            columns=6,
            grid_gap=2,
            display_scale=1.0,
            line_thickness=5,
            layout_border_scale=2.0,
            layout_gap=10,
            layout_bg_color=(255, 255, 255),
            current_group=None,
            current_dataset=None,
            method_roots=None,
    ):
        self.input_folders = input_folders
        self.output_folder = output_folder
        self.columns = max(int(columns), 1)
        self.grid_gap = int(grid_gap)
        try:
            self.display_scale = float(display_scale)
            if self.display_scale <= 0:
                self.display_scale = 1.0
        except Exception:
            self.display_scale = 1.0

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        self.method_roots = dict(method_roots or {})
        self.image_files = {}
        for name, src in self.input_folders.items():
            if isinstance(src, (list, tuple)):
                files = list(src)
            else:
                files = glob_single_files(src, IMG_EXTS)
            self.image_files[name] = files
            if name not in self.method_roots:
                self.method_roots[name] = self._infer_method_root(src, files)

        keys = list(self.image_files.keys())
        if reference_key is None:
            if 'GT' in self.image_files:
                reference_key = 'GT'
            elif 'input' in self.image_files:
                reference_key = 'input'
            else:
                reference_key = keys[0]
        self.reference_key = reference_key

        if len(self.image_files[self.reference_key]) == 0:
            raise ValueError(f"Reference folder has no images: {self.input_folders[self.reference_key]}")

        self.num_frames = len(self.image_files[self.reference_key])
        sample = cv2.imread(self.image_files[self.reference_key][0])
        if sample is None:
            raise ValueError("Failed to read reference sample image")
        self.height, self.width = sample.shape[:2]

        self.current_frame = 0

        self.window_main = "Crop Controller"
        self.window_grid = "Crop Grid"
        self.window_final = "Final Layout"

        self.dragging = False
        self.mode = 'idle'  # 'selection' | 'position' | 'idle'
        self.rois = {}  # dict of {id: {'rect': (x1,y1,x2,y2) or None, 'color': (b,g,r)}}
        self.active_roi = None  # active roi id
        self.selection_start = None

        self.palette = [
            (0, 0, 255),  # red
            (0, 255, 0),  # green
            (255, 0, 0),  # blue
            (0, 255, 255),  # yellow
            (255, 0, 255),  # magenta
            (255, 255, 0),  # cyan
            (255, 255, 255),  # white
            (0, 165, 255),  # orange
        ]

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
            if isinstance(layout_bg_color, str):
                parts = [int(x) for x in layout_bg_color.replace(' ', '').split(',') if x != '']
                if len(parts) == 3:
                    layout_bg_color = tuple(max(0, min(255, v)) for v in parts)
            self.layout_bg_color = tuple(layout_bg_color) if len(layout_bg_color) == 3 else (255, 255, 255)
        except Exception:
            self.layout_bg_color = (255, 255, 255)

        self.cached_images = None
        self.grid_windows = set()
        self.layout_mode = 'right'  # 'left' | 'up' | 'right' | 'bottom'
        self.preview_key = reference_key
        self.save_session_ts = None
        self.undo_manager = UndoManager()
        self.dispatcher = EventDispatcher()
        self._register_keybindings()
        self._pre_drag_state = None
        self._pre_drag_snapshot = None
        # dataset/group tracking
        self.group = current_group
        self.dataset = current_dataset
        # Derive group/dataset if not provided
        if self.group is None or self.dataset is None:
            try:
                sample_ref = next(iter(self.image_files.values()))
                sample_path = sample_ref[0] if sample_ref else next(iter(self.input_folders.values()))
                norm = os.path.normpath(sample_path)
                parts = norm.split(os.sep)
                if len(parts) >= 2:
                    self.dataset = parts[-2] if self.dataset is None else self.dataset
                    self.group = parts[-3] if len(parts) >= 3 and self.group is None else self.group
            except Exception:
                pass

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
            try:
                common = os.path.commonpath(files)
                return os.path.dirname(common)
            except Exception:
                return None
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
        self.update_display()

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
    def _set_frame(self, idx):
        self.current_frame = max(0, min(self.num_frames - 1, idx))
        self.update_display()

    def _set_layout(self, mode):
        self.layout_mode = mode
        self.update_display()

    # ---- Key binding setup ----
    def _register_keybindings(self):
        self.dispatcher.register(ord('n'), self._cmd_next_frame)
        self.dispatcher.register(ord('p'), self._cmd_prev_frame)
        self.dispatcher.register(ord('a'), self._cmd_add_roi)
        self.dispatcher.register(ord('r'), self._cmd_idle_mode)
        self.dispatcher.register(ord('s'), self._cmd_save)
        self.dispatcher.register(ord('z'), self._cmd_undo)
        self.dispatcher.register(ord('y'), self._cmd_redo)
        self.dispatcher.register(ord('d'), self._cmd_duplicate_roi)
        # digits 1-9
        for d in range(1, 10):
            self.dispatcher.register(ord(str(d)), lambda rid=d: self._cmd_digit_roi(rid))
        # Shift+1..9
        shift_map = {'!': 1, '@': 2, '#': 3, '$': 4, '%': 5, '^': 6, '&': 7, '*': 8, '(': 9}
        for sym, rid in shift_map.items():
            self.dispatcher.register(ord(sym), lambda rid=rid: self._cmd_shift_digit(rid))
        # arrow keys
        self.dispatcher.register(81, lambda: self._cmd_layout('left'))
        self.dispatcher.register(82, lambda: self._cmd_layout('up'))
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
        self.update_display()

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
        self.update_display()

    def _cmd_duplicate_roi(self):
        if self.active_roi is None or self.rois.get(self.active_roi, {}).get('rect') is None:
            log.warn("No active ROI to duplicate size from; create/select one first")
            return
        before = self._snapshot_rois()
        src_rect = self.rois[self.active_roi]['rect']
        new_id = self.add_roi()
        self.rois[new_id]['rect'] = src_rect
        log.success(f"Added ROI {log.style_num(str(new_id))} with active size")
        self.set_active_roi(new_id, to_selection=False)
        self._record_rois_change("duplicate roi", before=before)
        self.update_display()

    def _cmd_idle_mode(self):
        before = self._snapshot_rois()
        self.mode = 'idle'
        self.active_roi = None
        self.selection_start = None
        self._record_rois_change("idle mode", before=before)
        self.update_display()

    def _cmd_save(self):
        self.save_session_ts = None  # new session per save trigger
        label = getattr(self, 'save_label', self.dataset)
        self.save(label, self.dataset)

    def _cmd_digit_roi(self, roi_id):
        before = self._snapshot_rois()
        if roi_id in self.rois:
            self.set_active_roi(roi_id, to_selection=self.mode != 'selection')
            desc = f"select roi {roi_id}"
        else:
            self.add_roi(roi_id=roi_id)
            desc = f"add roi {roi_id}"
        self._record_rois_change(desc, before=before)
        self.update_display()

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
        self.update_display()

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
            self.update_display()

    def _cmd_redo(self):
        ok, desc = self.undo_manager.redo()
        if not ok:
            log.warn("Nothing to redo" if desc is None else f"Redo failed: {desc}")
        else:
            log.info(f"Redid: {desc or 'last action'}")
            self.update_display()

    # ---- Small helpers ----
    def _set_mode(self, mode):
        self.mode = mode
        self.update_display()

    def _restore_active(self, active_id, mode):
        self.active_roi = active_id
        self.mode = mode
        self.selection_start = None
        self.update_display()

    def _restore_roi_rect(self, rid, rect):
        if rid not in self.rois:
            return
        self.rois[rid]['rect'] = rect
        self.update_display()

    def color_for_id(self, roi_id):
        if not self.palette:
            return (0, 0, 255)
        return self.palette[(roi_id - 1) % len(self.palette)]

    def rebuild_dataset(self, new_group, new_dataset):
        # Build new input folders based on method roots
        if not self.method_roots:
            log.error("Dataset switching not supported for this layout")
            return False
        log.info(
            f"Switching dataset to {log.style_path(new_group)}/{log.style_path(new_dataset)} "
            f"for {len(self.method_roots)} methods"
        )
        new_inputs = {}
        for name, method_root in self.method_roots.items():
            if not method_root or not os.path.isdir(method_root):
                log.warn(f"Cannot switch dataset for {log.style_path(name)} (missing method root)")
                continue
            cand = os.path.join(method_root, new_group, new_dataset)
            if not os.path.exists(cand):
                log.warn(f"Missing folder for {log.style_path(name)} at {log.style_path(cand)}")
                continue
            imgs = glob_single_files(cand, ['png', 'jpg', 'jpeg'])
            if len(imgs) == 0:
                log.warn(f"No images in {log.style_path(cand)} for {log.style_path(name)}")
                continue
            new_inputs[name] = cand

        if not new_inputs:
            log.error(f"Switch failed: no valid inputs for {log.style_path(new_group)}/{log.style_path(new_dataset)}")
            return False

        # Update state
        self.input_folders = new_inputs
        self.group = new_group
        self.dataset = new_dataset
        self.image_files = {name: glob_single_files(path, ['png', 'jpg', 'jpeg']) for name, path in new_inputs.items()}

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
        sample = cv2.imread(self.image_files[self.reference_key][0])
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
        self.grid_windows = set()
        self.add_roi()
        log.success(f"Switched to {log.style_path(new_group)}/{log.style_path(new_dataset)}")
        return True

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
        self.current_frame = target
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
        color = self.color_for_id(roi_id)
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

    def on_mouse(self, event, x, y, flags, param):
        if self.active_roi is None:
            return
        roi = self.rois[self.active_roi]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
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
                roi['rect'] = (sx, sy, x, y)
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
            self.update_display()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            prev_state = self._pre_drag_state
            if self.mode == 'selection' and self.selection_start is not None:
                sx, sy = self.selection_start
                roi['rect'] = (sx, sy, x, y)
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
            self.update_display()

    def read_frame(self, key, idx):
        files = self.image_files[key]
        if idx < 0 or idx >= len(files):
            idx = max(0, min(len(files) - 1, idx))
        img = cv2.imread(files[idx])
        return img

    def build_grid_for_rect(self, rect, header_text=None):
        x1, y1, x2, y2 = self.clamp_rect(rect)
        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)

        # method_keys = [k for k in self.image_files.keys() if k not in ['GT', 'input']]
        method_keys = [k for k in self.image_files.keys()]
        if len(method_keys) == 0:
            method_keys = list(self.image_files.keys())

        cols = self.columns
        rows = max(1, math.ceil(len(method_keys) / cols))
        gap = self.grid_gap
        grid_h = rows * roi_h + gap * (rows - 1)
        grid_w = cols * roi_w + gap * (cols - 1)
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

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
            if crop.shape[0] != roi_h or crop.shape[1] != roi_w:
                pad = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
                ph = min(roi_h, crop.shape[0])
                pw = min(roi_w, crop.shape[1])
                pad[:ph, :pw] = crop[:ph, :pw]
                crop = pad

            row = i // cols
            col = i % cols
            rg = row * gap
            cg = col * gap
            y0 = row * roi_h + rg
            x0 = col * roi_w + cg
            grid[y0:y0 + roi_h, x0:x0 + roi_w] = crop
            cv2.putText(grid, name, (x0 + 2, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if header_text:
            cv2.putText(grid, header_text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        return grid

    def build_grid(self):
        valid_rois = [(rid, r) for rid, r in sorted(self.rois.items()) if r['rect'] is not None]
        if len(valid_rois) == 0:
            return None
        grids = []
        gap = self.grid_gap
        for rid, r in valid_rois:
            header = f"ROI {rid}"
            g = self.build_grid_for_rect(r['rect'], header_text=header)
            if g is not None:
                grids.append(g)
        if len(grids) == 0:
            return None
        total_h = sum(g.shape[0] for g in grids) + gap * (len(grids) - 1)
        total_w = max(g.shape[1] for g in grids)
        out = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        y = 0
        for i, g in enumerate(grids):
            h, w = g.shape[:2]
            out[y:y + h, :w] = g
            y += h
            if i < len(grids) - 1:
                y += gap
        return out

    def build_final_layout_for_key(self, key):
        ref = self.read_frame(key, self.current_frame)
        if ref is None:
            return None
        H, W = ref.shape[:2]
        valid = [(rid, r) for rid, r in sorted(self.rois.items()) if r['rect'] is not None]
        # Order ROIs by their spatial position instead of id: vertical for left/right, horizontal for up/bottom
        if len(valid) > 1:
            if self.layout_mode in ['left', 'right']:
                valid.sort(key=lambda item: ((item[1]['rect'][1] + item[1]['rect'][3]) / 2.0))
            else:
                valid.sort(key=lambda item: ((item[1]['rect'][0] + item[1]['rect'][2]) / 2.0))
        if len(valid) == 0:
            return None
        block_line_th = max(1, int(round(self.line_thickness * self.layout_border_scale)))
        # Single ROI: overlay inside at the mirrored position
        if len(valid) == 1:
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
            # Scale with display_scale but clamp to fit canvas
            dscale = self.display_scale if hasattr(self, 'display_scale') and self.display_scale > 0 else 1.0
            scale = min(dscale, min(W / max(gw, 1), H / max(gh, 1)))
            if scale != 1.0:
                crop = cv2.resize(crop, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                gh, gw = crop.shape[:2]
            # place at the nearest corner to mirrored center
            corners = [(0, 0), (W - gw, 0), (0, H - gh), (W - gw, H - gh)]

            def dist2(ax, ay, bx, by):
                dx = ax - bx
                dy = ay - by
                return dx * dx + dy * dy

            best = min(corners, key=lambda c: dist2(c[0] + gw // 2, c[1] + gh // 2, ox, oy))
            x0, y0 = best
            out = ref.copy()
            out[y0:y0 + gh, x0:x0 + gw] = crop
            # draw ROI box on full image and crop box in final
            color = r0['color']
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
                roi_info.append({'id': rid, 'color': r['color']})
        if len(crops) == 0:
            return ref.copy()
        mode = self.layout_mode
        if mode in ['left', 'right']:
            # Rule: scale all crops to width W, get stacked height H', then globally scale to fit height H.
            resized0 = []
            heights_scaled = []
            for g in crops:
                gh, gw = g.shape[:2]
                s0 = W / max(gw, 1)
                w0 = int(round(gw * s0))
                h0 = int(round(gh * s0))
                img0 = cv2.resize(g, (w0, h0), interpolation=cv2.INTER_NEAREST)
                heights_scaled.append(h0)
                resized0.append(img0)
            Hp = sum(heights_scaled)
            s_global = (H - self.layout_gap) / max(Hp, 1)
            resized = []
            for img0 in resized0:
                h0, w0 = img0.shape[:2]
                h1 = max(1, int(round(h0 * s_global)))
                w1 = max(1, int(round(w0 * s_global)))
                img1 = cv2.resize(img0, (w1, h1), interpolation=cv2.INTER_NEAREST)
                resized.append(img1)
            inner_pad = self.layout_gap
            block_w = max(img.shape[1] for img in resized)
            block_h = sum(img.shape[0] for img in resized) + inner_pad * (len(resized) - 1 if len(resized) > 0 else 0)
            block = np.full((block_h, block_w, 3), self.layout_bg_color, dtype=np.uint8)
            y = 0
            for idx_img, img in enumerate(resized):
                h, w = img.shape[:2]
                x = (block_w - w) // 2
                block[y:y + h, x:x + w] = img
                color = roi_info[idx_img]['color'] if idx_img < len(roi_info) else (255, 255, 255)
                self.draw_inner_border(block, x, y, w, h, color, thickness=block_line_th)
                y += h
                if idx_img < len(resized) - 1:
                    y += inner_pad
            pad = self.layout_gap
            canvas_h = max(H, block_h)
            canvas_w = W + block_w + pad
            out = np.full((canvas_h, canvas_w, 3), self.layout_bg_color, dtype=np.uint8)
            # place full image
            fy = (canvas_h - H) // 2
            if mode == 'left':
                fx = block_w + pad
                bx = 0
            else:  # right
                fx = 0
                bx = W + pad
            by = (canvas_h - block_h) // 2
            # draw ROI boxes on full image
            base = ref.copy()
            for _, r in valid:
                x1, y1, x2, y2 = self.clamp_rect(r['rect'])
                cv2.rectangle(base, (x1, y1), (x2, y2), r['color'], self.line_thickness)
            out[fy:fy + H, fx:fx + W] = base
            out[by:by + block_h, bx:bx + block_w] = block
            return out
        else:  # 'up' or 'bottom'
            # Rule: scale all crops to height H, get stacked width W', then globally scale to fit width W.
            resized0 = []
            widths_scaled = []
            for g in crops:
                gh, gw = g.shape[:2]
                s0 = H / max(gh, 1)
                h0 = int(round(gh * s0))
                w0 = int(round(gw * s0))
                img0 = cv2.resize(g, (w0, h0), interpolation=cv2.INTER_NEAREST)
                widths_scaled.append(w0)
                resized0.append(img0)
            Wp = sum(widths_scaled)
            s_global = (W - self.layout_gap) / max(Wp, 1)
            resized = []
            for img0 in resized0:
                h0, w0 = img0.shape[:2]
                w1 = max(1, int(round(w0 * s_global)))
                h1 = max(1, int(round(h0 * s_global)))
                img1 = cv2.resize(img0, (w1, h1), interpolation=cv2.INTER_NEAREST)
                resized.append(img1)
            inner_pad = self.layout_gap
            block_w = sum(img.shape[1] for img in resized) + inner_pad * (len(resized) - 1 if len(resized) > 0 else 0)
            block_h = max(img.shape[0] for img in resized)
            block = np.full((block_h, block_w, 3), self.layout_bg_color, dtype=np.uint8)
            x = 0
            for idx_img, img in enumerate(resized):
                h, w = img.shape[:2]
                y = (block_h - h) // 2
                block[y:y + h, x:x + w] = img
                color = roi_info[idx_img]['color'] if idx_img < len(roi_info) else (255, 255, 255)
                self.draw_inner_border(block, x, y, w, h, color, thickness=block_line_th)
                x += w
                if idx_img < len(resized) - 1:
                    x += inner_pad
            pad = self.layout_gap
            canvas_w = max(W, block_w)
            canvas_h = H + block_h + pad
            out = np.full((canvas_h, canvas_w, 3), self.layout_bg_color, dtype=np.uint8)
            # place full image
            fx = (canvas_w - W) // 2
            if mode == 'up':
                fy = block_h + pad
                by = 0
            else:  # bottom
                fy = 0
                by = H + pad
            bx = (canvas_w - block_w) // 2
            base = ref.copy()
            for _, r in valid:
                x1, y1, x2, y2 = self.clamp_rect(r['rect'])
                cv2.rectangle(base, (x1, y1), (x2, y2), r['color'], self.line_thickness)
            out[fy:fy + H, fx:fx + W] = base
            out[by:by + block_h, bx:bx + block_w] = block
            return out

    def update_display(self):
        ref = self.read_frame(self.reference_key, self.current_frame)
        if ref is None:
            return
        canvas = ref.copy()
        if self.mode in ['selection', 'position'] and len(self.rois) > 0:
            for rid, r in sorted(self.rois.items()):
                if r['rect'] is None:
                    continue
                x1, y1, x2, y2 = r['rect']
                if self.mode == 'selection' and self.active_roi == rid:
                    self.draw_dashed_rect(canvas, (x1, y1), (x2, y2), r['color'], thickness=self.line_thickness)
                else:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), r['color'], self.line_thickness)
                if self.active_roi == rid:
                    cx = x1 + 20
                    cy = max(20, y1 - 20)
                    self.draw_circled_label(canvas, (cx, cy), rid, r['color'])
                else:
                    cv2.putText(canvas, f"{rid}", (x1 + 3, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, r['color'], 2)
        # Top-left image name (without extension) for the interactive reference display
        try:
            ref_path = self.image_files[self.reference_key][self.current_frame]
            ref_name = os.path.basename(ref_path).rsplit('.', 1)[0]
            cv2.putText(canvas, ref_name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
        except Exception:
            pass
        cv2.imshow(self.window_main, canvas)

        if self.mode == 'idle':
            for w in list(self.grid_windows):
                try:
                    cv2.destroyWindow(w)
                except Exception:
                    pass
            self.grid_windows.clear()
        else:
            needed = set()
            valid_rois = [(rid, r) for rid, r in sorted(self.rois.items()) if r['rect'] is not None]
            for rid, r in valid_rois:
                name = f"{self.window_grid} ROI {rid}"
                grid = self.build_grid_for_rect(r['rect'])  # , header_text=f"ROI {rid}")
                if grid is None:
                    continue
                if self.display_scale and self.display_scale != 1.0:
                    grid_view = cv2.resize(grid, dsize=None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(name, grid_view)
                else:
                    cv2.imshow(name, grid)
                needed.add(name)
            # Destroy windows no longer needed
            obsolete = self.grid_windows - needed
            for w in obsolete:
                try:
                    cv2.destroyWindow(w)
                except Exception:
                    pass
            self.grid_windows = needed

            # Final layout window (preview only shows one image)
            preview_key = self.preview_key or self.reference_key
            final = self.build_final_layout_for_key(preview_key)
            if final is None:
                blank = np.zeros((min(240, self.height), min(320, self.width), 3), dtype=np.uint8)
                cv2.putText(blank, "Final layout idle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow(self.window_final, blank)
            else:
                # Final preview image size is not affected by scale
                cv2.imshow(self.window_final, final)

    def save(self, pair, dataset):
        # Prepare timestamped output directory: output/<timestamp>/<dataset>/
        if self.save_session_ts is None:
            self.save_session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.join(self.output_folder, self.save_session_ts, dataset)
        os.makedirs(base_dir, exist_ok=True)

        # Determine method keys (exclude GT/input if present)
        # method_keys = [k for k in self.image_files.keys() if k not in ['GT', 'input']]
        method_keys = [k for k in self.image_files.keys()]
        if len(method_keys) == 0:
            method_keys = list(self.image_files.keys())

        # For each method: save final layout, per-ROI crops, and original image
        for m in method_keys:
            img_file = self.image_files[m][self.current_frame]
            img_name = os.path.basename(img_file).rsplit('.', 1)[0]
            m_dir = os.path.join(base_dir, img_name)
            os.makedirs(m_dir, exist_ok=True)
            img = self.read_frame(m, self.current_frame)
            if img is None:
                log.warn(f"Skip method {m}: image read failed")
                continue
            H, W = img.shape[:2]

            # Save original image
            orig_out = os.path.join(m_dir, f"orig_{m}.png")
            cv2.imwrite(orig_out, img)

            # Save final layout composed for this method
            final = self.build_final_layout_for_key(m)
            if final is not None:
                final_out = os.path.join(m_dir, f"final_{m}.png")
                cv2.imwrite(final_out, final)
            else:
                log.warn(f"Final layout for {m} is empty; skipping")

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
                crop_out = os.path.join(m_dir, f"crop_roi{rid}_{m}.png")
                cv2.imwrite(crop_out, crop)
        log.success(f"Saved outputs under: {base_dir}")

    def run(self, pair):
        cv2.namedWindow(self.window_main)
        cv2.setMouseCallback(self.window_main, self.on_mouse)
        self.save_label = pair

        log.banner("Interactive Crop Comparator")
        log.info(f"Mouse: selection mode to draw rect; position mode to move")
        log.info(
            "Keys: "
            + f"{log.style_key('a')} add next ROI, "
            + f"{log.style_key('1-9')} add/select ROI id; press same id to enter selection mode, "
            + f"{log.style_key('Shift+1-9')} duplicate to id or copy size (see README), "
            + f"{log.style_key('d')} add new ROI with active size, "
            + f"{log.style_key('Del')} delete active ROI"
        )
        log.info(
            "Arrow keys (switch final layout on the fly): "
            + f"{log.style_key('←')} {log.style_mode('left')} (crops stack left), "
            + f"{log.style_key('↑')} {log.style_mode('up')} (crops stack above), "
            + f"{log.style_key('→')} {log.style_mode('right')} (crops stack right), "
            + f"{log.style_key('↓')} {log.style_mode('bottom')} (crops stack below)"
        )
        log.info(
            "Other: "
            + f"{log.style_key('n')}/{log.style_key('p')} next/prev image, "
            + f"{log.style_key('s')} save outputs, "
            + f"{log.style_key('r')} idle mode, "
            + f"{log.style_key('z')} undo, "
            + f"{log.style_key('y')} redo, "
            + f"{log.style_key('Enter')} switch dataset/group, "
            + f"{log.style_key('Space')} jump to image, "
            + f"{log.style_key('q')} quit"
        )
        # Default: enter ROI 1 selection mode at startup
        if len(self.rois) == 0:
            self.add_roi()
        self.update_display()

        while True:
            key = cv2.waitKey(10) & 0xFF
            handled = self.dispatcher.dispatch(key)
            if handled:
                continue
            if key == ord('q') or key == 27:
                break
            elif key in [13, 10]:
                # Enter: prompt to switch dataset (optionally group/dataset)
                try:
                    prompt = "Enter dataset (or group/dataset): "
                    text = input(prompt).strip()
                except Exception:
                    text = ""
                if text:
                    if '/' in text:
                        ng, nd = text.split('/', 1)
                    else:
                        ng, nd = (self.group or ''), text
                    if not ng:
                        ng = self.group or ''
                    if not nd:
                        nd = self.dataset or ''
                    if not ng or not nd:
                        log.error("Group or dataset is empty; aborted switch")
                    else:
                        if self.rebuild_dataset(ng, nd):
                            self.update_display()
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
                        self.update_display()
                else:
                    log.info("Image jump cancelled (empty input)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    log.banner("LLIE Results - Compare Tool")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['local', 'external'], default='local', type=str,
                        help='Data source: local uses the workspace structure under --root; external uses /data/xr paths with --pair videos.')
    parser.add_argument('--root', '-r', default='/mnt/yzc/Results/LLIE-results', type=str,
                        help='Workspace root containing method folders (local mode only). Example: /mnt/yzc/Results/LLIE-results')
    parser.add_argument('--group', '-g', default='SDSD-indoor+', type=str,
                        help='Dataset group folder under each method (e.g., LOLv2-real+, SDSD-indoor+). Hyphens are auto-resolved across methods.')
    parser.add_argument('--dataset', '-ds', default='SDSD-indoor', type=str,
                        help='Leaf dataset folder under the group (e.g., DarkFace, DICM, LOL, SDSD-indoor).')
    parser.add_argument('--pair', '-p', default='pair13', type=str,
                        help='Video pair/sequence name (external mode only), e.g., pair13. Ignored in local mode.')
    parser.add_argument('--columns', '-c', default=6, type=int,
                        help='Number of columns in the per-ROI method grid view (rows are computed automatically).')
    parser.add_argument('--grid-gap', default=2, type=int,
                        help='Gap (in pixels) between tiles in per-ROI method grid windows (default: 2).')
    parser.add_argument('--magnify', '--scale', default=2.0, type=float,
                        help='Display-only magnification for crop grid windows. Final preview is not globally scaled; multi-ROI ignores this.')
    parser.add_argument('--layout', default='right', type=str, choices=['left', 'up', 'right', 'bottom'],
                        help='Final layout preview mode. Use arrow keys at runtime: ← left, ↑ up, → right, ↓ bottom.')
    parser.add_argument('--preview', default=None, type=str,
                        help='Image key to show in Final Layout preview (default: reference key). Keys come from method names or GT/input if present.')
    parser.add_argument('--mode', '-m', default='selection', type=str, choices=['selection', 'position', 'idle'],
                        help='Startup interaction mode: selection (draw), position (move), or idle (hide grids).')
    parser.add_argument('--output', '-o', default='./.crop_grid/', type=str,
                        help='Root output folder. Files are saved under output/<timestamp>/<dataset>/...')
    parser.add_argument('--thickness', '-t', default=2, type=int,
                        help='Line thickness for ROI boxes and crop borders (default: 2; minimum: 1).')
    parser.add_argument('--layout-border-scale', default=2.0, type=float,
                        help='Thickness multiplier applied only to crop block borders in final layout (default: 2.0).')
    parser.add_argument('--layout-gap', default=10, type=int,
                        help='Gap (in pixels) between base image and crop block, and between crops themselves (default: 10).')
    parser.add_argument('--layout-bg-color', default='255,255,255', type=str,
                        help='Padding/background color as R,G,B for gaps in final layout (default: 255,255,255 for white).')
    parser.add_argument('--structure', default='auto', choices=['auto', 'group-dataset-pair', 'group-dataset', 'dataset-only', 'flat', 'shared'],
                        help='Folder structure layout: auto (default), group-dataset-pair, group-dataset, dataset-only, flat (images directly under method), or shared (image-id folders containing per-method files such as img1/methodA.png).')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable ANSI colored logs (use plain text).')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warn', 'error'],
                        help='Logging level: debug|info|warn|error (default: info).')
    args = parser.parse_args()

    pair = args.pair
    group = args.group
    dataset = args.dataset

    file_path = "./timeline_methods.txt"
    methods = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            methods = [line.strip() for line in f.read().strip().splitlines() if line.strip()]

    input_folder = {}
    if args.source == 'external':
        if not methods:
            raise ValueError("timeline_methods.txt is required for external source")
        input_folder = {m: f"/data/xr/LLVE_results/VideoForm/{m}/{dataset}/pred/{pair}" for m in methods}

        # Optional GT/input reference if available
        gt_path = ""
        gt_path_exist = False
        for phase in ['test', 'eval']:
            for gt in ['GT', 'high']:
                gt_path = f"/data/xr/Dataset/LLVE_dataset/{dataset}_png/{phase}/{gt}/{pair}"
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
                methods = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            except Exception:
                methods = []
        try:
            log.info(f"Visualizing methods: {methods}")
        except Exception:
            pass

        input_folder = discover_local_inputs(root, methods, group=group, dataset=dataset, pair=pair, structure=args.structure)
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
            imgs = glob_single_files(source, IMG_EXTS)
            if len(imgs) == 0:
                log.error(f"Folder {source} has no images (png/jpg/jpeg)")
                raise ValueError(f"Folder {source} has no images (png/jpg/jpeg)")

    # configure logger
    try:
        log.set_color_enabled(not args.no_color)
        log.set_level(args.log_level)
    except Exception:
        pass

    # Parse layout background color CSV (R,G,B)
    try:
        parts = [int(x) for x in args.layout_bg_color.replace(' ', '').split(',') if x != '']
        if len(parts) != 3:
            raise ValueError()
        layout_bg_color = tuple(max(0, min(255, v)) for v in parts)
    except Exception:
        layout_bg_color = (255, 255, 255)

    comparator = InteractiveCropComparator(
        input_folder,
        output_folder=args.output,
        reference_key=('GT' if 'GT' in input_folder else (
            'input' if 'input' in input_folder else next(iter(input_folder.keys())))),
        columns=args.columns,
        grid_gap=args.grid_gap,
        display_scale=args.magnify,
        line_thickness=args.thickness,
        layout_border_scale=args.layout_border_scale,
        layout_gap=args.layout_gap,
        layout_bg_color=layout_bg_color,
        current_group=args.group,
        current_dataset=dataset,
    )
    comparator.mode = args.mode
    comparator.layout_mode = args.layout
    comparator.preview_key = args.preview or ('GT' if 'GT' in input_folder else (
        'input' if 'input' in input_folder else next(iter(input_folder.keys()))))
    # Log summary of loaded data
    try:
        shared_layout = any(isinstance(v, (list, tuple)) for v in input_folder.values())
        structure_used = 'shared' if shared_layout else args.structure
        log.info(
            f"Loaded {log.style_num(str(len(comparator.image_files)))} methods "
            f"using structure={log.style_mode(structure_used)}; "
            f"reference={log.style_key(comparator.reference_key)}; "
            f"frames per method≈{log.style_num(str(comparator.num_frames))}"
        )
        if shared_layout:
            log.note("Detected shared layout (image-id folders containing per-method files)")
    except Exception:
        pass
    # Use dataset as label for saving in local mode; pair+dataset for external
    label_pair = pair if args.source == 'external' else dataset
    comparator.run(label_pair)
