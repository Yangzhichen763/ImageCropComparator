# ✨ LLIE Results – Interactive Crop Comparator

An interactive, keyboard-driven tool to compare and compose crops from multiple LLIE method outputs side by side. Built with Python + OpenCV and designed for fast visual inspection, precise ROI selection, and reproducible exports. Colored logs are enabled on Linux for clear feedback.

## 🚀 Features
- **Interactive ROIs:** selection, position, and idle modes for flexible crop control.
- **Multi-ROI management:** numeric keys `1–9` select/add ROI by ID; `Shift+1–9` selects in selection mode; `a` adds the smallest unused ID.
- **Per-ROI method grids:** preview the same crop across all methods in a tiled grid.
- **Final layout preview:** compose single or multiple crops with strict layout rules (`left`, `right`, `top`, `bottom`); change at runtime via arrow keys.
  - Configurable ordering: sort by ROI position (default) or by ROI id, with optional reverse stacking. For position mode, `left/right` stack top→bottom and `top/bottom` stack left→right.
  - Configurable padding between crops and between the crops block and the base image.
  - Display overlay thickness multiplier for the final layout (default ×2).
- **Active ROI spotlight:** active ROI id is circled on the reference view for clarity.
- **Inner crop frames:** final preview uses inner borders for crops; base image retains standard ROI boxes.
- **Timestamped saving:** exports original, final preview, and each ROI crop under `output/<timestamp>/<dataset>/...`.
- **Colored logs:** ANSI color output on Linux terminals; configurable log level and color toggle.
- **Top-left overlay:** current image basename shown on the interactive window.
- **Interactive switching:** Enter switches dataset/group via prompt; Space jumps directly to an image by name.
- **Undo/redo:** `z` to undo, `y` to redo most actions (ROI add/move/delete, layout/frame changes, etc.).

## 📦 Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- natsort (`natsort`) and Pillow (`Pillow`) – used for robust path discovery when `basic.utils.io` is not available.

Install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy natsort Pillow
```

## 🗂️ Workspace Layout
Method folders sit under the workspace root (e.g., `BiFormer/`, `LLFormer/`, `FourLLIE/`, etc.). Each method contains dataset groups (e.g., `LOLv2-real+`, `SDSD-indoor+`) and then a leaf dataset (e.g., `DarkFace`, `LOL`, `SDSD-indoor`).

```
<root>/
  BiFormer/
    LOLv2-real+/
      LOL-v2-real/
  FourLLIE/
    SDSD-indoor+/
      SDSD-indoor/
  ...
  compare.py
```

i.e.
```
<root>/
  <method1>/
    <group>/
      <dataset>/
        <img1.png>
        <img2.png>
        ...
        <imgn.png>
  <method2>/
    <group>/
      <dataset>/
        <img1.png>
        <img2.png>
        ...
        <imgn.png>
```

#### Supported folder layouts (select via `--structure`)

1. `group-dataset-pair` (deep): `<root>/<method>/<group>/<dataset>/<pair>/<img>.png`
```
<root>/
  <method1>/
    <group>/
      <dataset>/
        <pair>/
          <img1.png>
          <img2.png>
          ...
          <imgn.png>
  <method2>/
    <group>/
      <dataset>/
        <pair>/
          <img1.png>
          <img2.png>
          ...
          <imgn.png>
```
2. `group-dataset` (default classic): `<root>/<method>/<group>/<dataset>/<img>.png`
```
<root>/
  <method1>/
    <group>/
      <dataset>/
        <img1.png>
        <img2.png>
        ...
        <imgn.png>
  <method2>/
    <group>/
      <dataset>/
        <img1.png>
        <img2.png>
        ...
        <imgn.png>
```
3. `dataset-only`: `<root>/<method>/<dataset>/<img>.png`
```
<root>/
  <method1>/
    <dataset>/
      <img1.png>
      <img2.png>
      ...
      <imgn.png>
  <method2>/
    <dataset>/
      <img1.png>
      <img2.png>
      ...
      <imgn.png>
```
4. `flat` (images directly under method): `<root>/<method>/<img>.png`
```
<root>/
  <method1>/
    <img1.png>
    <img2.png>
    ...
    <imgn.png>
  <method2>/
    <img1.png>
    <img2.png>
    ...
    <imgn.png>
```
5. `shared` (per-image folders contain per-method files): `<root>/<image-id>/<methodA>.png, <methodB>.png, ...`
```
<root>/
  <img1>/
    <method1.png>
    <method2.png>
    ...
    <methodn.png>
  <img2>/
    <method1.png>
    <method2.png>
    ...
    <methodn.png>
```
6. `auto` (default): tries the above in order until images are found.

## 🧭 Usage
Run locally with auto-discovery of methods:

```bash
python compare.py \
  --source local \
  --root /mnt/yzc/Results/LLIE-results \
  --group LOLv2-real+ \
  --dataset LOL-v2-real \
  --structure auto \
  --magnify 2.0 \
  --layout right \
  --output ./outputs
```

Run with external video-form datasets (requires `timeline_methods.txt`):

```bash
python compare.py \
  --source external \
  --dataset SDSD-indoor \
  --pair pair13 \
  --layout left

# Explicit structure example (flat images under each method)
python compare.py \
  --source local \
  --root /mnt/yzc/Results/LLIE-results \
  --structure flat
```

### ⚙️ CLI Options (selected)
- `--source`: `local` or `external`
- `--root`: workspace root for local mode
- `--group`: dataset group under each method (hyphens auto-resolved)
- `--dataset`: leaf dataset name
- `--pair`: sequence name for external mode
- `--columns`: grid columns for per-ROI previews
- `--magnify`/`--scale`: display-only magnification for grid windows (final preview unscaled; multi-ROI ignores this)
- `--layout`: final layout preview mode (`left|top|right|bottom`)
- `--preview`: image key used in final preview (method name or `GT`/`input` if present)
- `--mode`: startup interaction mode (`selection|position|idle`)
- `--output`: root output directory
- `--thickness`: line thickness for ROI boxes and crop borders
- `--display-thickness-mult`: multiplier for line thickness in the final layout overlays
- `--layout-padding`: padding (px) between crops and between crops and the base image in the final layout
- `--no-color`: disable ANSI colored logs
- `--log-level`: `debug|info|warn|error`

## 🎮 Interaction & Shortcuts
- Mouse: draw in selection mode; drag to reposition in position mode.
- `a`: add smallest unused ROI ID.
- `1–9`: add/select ROI by ID; pressing the active ROI id again enters selection mode.
- `Shift+1–9`: if the target ROI id does not exist, duplicate the active ROI into that id; if it exists, copy the active ROI’s size to the target ROI (center preserved).
- `d`: add a new ROI with the same size as the current active ROI (uses the smallest unused ID).
- Arrow keys: set layout (`← left`, `↑ top`, `→ right`, `↓ bottom`).
- `z` / `y`: undo / redo the last action (ROI edits, layout or frame changes, etc.).
- `Enter`: switch dataset or group/dataset (typed as `group/dataset`).
- `Space`: jump to an image by name (stem or filename); logs if missing.
- `n` / `p`: next / previous image.
- `s`: save outputs.
- `i`: idle toggle (hide/show grids).
- `r`: clear all ROIs.
- `q` or `Esc`: quit.

## 🖼️ Quick tutorial

short walkthrough showing ROI selection.

<img src="figures/basic_op_video.gif" style="width: 100%;">

Basic ROI operations:

| Select ROI                                  | Move ROI                                | Add ROI                                    |
|---------------------------------------------|-----------------------------------------|--------------------------------------------|
| ![Select ROI](figures/select_crop_area.png) | ![Move ROI](figures/move_crop_area.png) | ![Add ROI](figures/add_more_crop_area.png) |

| Reselect ROI                                    | Delete ROI                                  |
|-------------------------------------------------|---------------------------------------------|
| ![Reselect ROI](figures/reselect_crop_area.png) | ![Delete ROI](figures/delete_crop_area.png) |

More flexible layout directions:

| Layout left                                 | Layout right                                  | Layout top                               | Layout bottom                                   |
|---------------------------------------------|-----------------------------------------------|------------------------------------------|-------------------------------------------------|
| ![Layout left](figures/layout_left_img.png) | ![Layout right](figures/layout_right_img.png) | ![Layout top](figures/layout_top_img.png) | ![Layout bottom](figures/layout_bottom_img.png) |

More layout options:

| Single ROI (one)                      | Two ROIs                              | Three ROIs                                | More than three ROIs                    |
|---------------------------------------|---------------------------------------|-------------------------------------------|-----------------------------------------|
| ![Layout one](figures/layout_one.png) | ![Layout two](figures/layout_two.png) | ![Layout three](figures/layout_three.png) | ![Layout more](figures/layout_more.png) |

## 💾 Saving Structure
On save (`s`), outputs are written to:

```
<output>/<timestamp>/<dataset>/<image-basename>/
  orig_<method>.png
  final_<method>.png
  crop_roi<id>_<method>.png
```

Each method writes its own original, composed final preview, and all crops for active ROIs.

## 🎨 Colored Logs
- Linux terminals display colored status messages: info (cyan), success (green), warnings (yellow), errors (red), and notes (bright cyan).
- Banners use terminal-width separators with centered timestamps; tokens (keys/modes/paths) are colored for quick scanning.
- Toggle colors with `--no-color`; control verbosity with `--log-level`.

## 🛠️ Troubleshooting
- "Folder has no images": check dataset path and image extensions.
- "timeline_methods.txt is required": provide a method list when using `--source external`.
- Window not responsive: ensure your environment supports OpenCV GUI; on headless servers use a local machine or X-forwarding.

## 🙏 Acknowledgements
This tool is inspired by practical LLIE evaluation workflows and built for fast iteration and visual clarity.
