# ✨ Image Crop Comparator

Keyboard-driven OpenCV viewer to compare pixel level method outputs, select multiple ROIs, and export reproducible layouts. Built for fast visual inspection with transparent final composites and strict layout rules.

## 📑 Table of Contents
- [🚀 Quick Start](#quick-start)
- [🧰 Installation](#installation)
- [🗂️ Workspace Layout](#workspace-layout)
- [🧭 Usage](#usage)
- [⚙️ CLI Options](#cli-options)
- [🎮 Interaction](#interaction)
- [🧪 Workflows](#workflows)
- [💾 Saving](#saving)
- [🎨 Logs](#logs)
- [🛠️ Troubleshooting](#troubleshooting)
- [🙏 Acknowledgements](#acknowledgements)

<a id="quick-start"></a>
## 🚀 Quick Start
```bash
python compare.py \
  --source local \
  --root ./examples \
  --group LOLv2-real+ \
  --dataset LOL-v2-real \
  --layout right \
  --structure auto
```

<a id="installation"></a>
## 🧰 Installation
- Python 3.8+
- [requirements.txt](requirements.txt) pins `opencv-python==4.7.*` and `numpy==1.26.*`; install with `pip install -r requirements.txt`.
- Packages: opencv-python, numpy, natsort, Pillow

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

<a id="workspace-layout"></a>
## 🗂️ Workspace Layout
Methods live under the workspace root; each holds groups and datasets.

```
<root>/
  <method>/
    <group>/
      <dataset>/
        <images...>
```

<details open>
<summary>Supported structures (use `--structure`)</summary>

- `group-dataset-pair`: `<root>/<method>/<group>/<dataset>/<pair>/<img>.png`
- `group-dataset` (default classic): `<root>/<method>/<group>/<dataset>/<img>.png`
- `dataset-only`: `<root>/<method>/<dataset>/<img>.png`
- `flat`: `<root>/<method>/<img>.png`
- `shared`: `<root>/<image-id>/<method>.png`
- `auto` (default): tries the above in order.

</details>

<a id="usage"></a>
## 🧭 Usage
Local images with auto-discovery:

```bash
python compare.py --source local --root <root> --group <group> --dataset <dataset> --layout right
```

External (video-form) datasets require `timeline_methods.txt`:

```bash
python compare.py --source external --dataset SDSD-indoor --pair pair13 --layout left
```

<a id="cli-options"></a>
## ⚙️ CLI Options
<details open>
<summary>Core switches</summary>

- `--source`: `local` or `external`
- `--root`: workspace root (local)
- `--group`, `--dataset`, `--pair`: dataset selectors
- `--structure`: layout of files (see above)
- `--output`: output root directory

</details>

<details>
<summary>Display and layout</summary>

- `--layout`: `left|top|right|bottom`
- `--columns`: grid columns for per-ROI previews
- `--magnify` / `--scale`: display-only magnification (ignored for multi-ROI final)
- `--layout-padding`: padding between crops and base image
- `--display-thickness-mult`: overlay thickness multiplier for final layout

</details>

<details>
<summary>Interaction defaults</summary>

- `--mode`: `selection|position|idle`
- `--preview`: method key for final preview (`GT`/`input` if present)
- `--thickness`: ROI and crop border thickness

</details>

<details>
<summary>Logging</summary>

- `--log-level`: `debug|info|warn|error`
- `--no-color`: disable ANSI colors

</details>

<a id="interaction"></a>
## 🎮 Interaction
<details open>
<summary>Mouse</summary>

- Draw in selection mode; drag to reposition in position mode.
- Hold `Shift` while drawing a ROI: constrain to a square using the longer side.

Quick actions:
- Right-click on overlapping ROIs: cycle to the next higher id in that stack (wraps to smallest).
- Right-click outside any ROI: add the next ROI (same as `a`).
- Right-click down inside a ROI, release after the cursor leaves that ROI: delete it.
- Middle-button inside the active ROI: duplicate it to a new ROI, then drag the new ROI while holding.
- Middle-button inside a non-active ROI: copy the active ROI size to that ROI (Shift+digit).

</details>

<details open>
<summary>Keyboard</summary>

- `a`: add smallest unused ROI id
- `1–9`: add/select ROI by id; tapping active id enters selection mode
- `Shift+1–9`: duplicate active ROI to target id or copy its size
- `d`: add ROI with active size
- Arrow keys: set layout (`← left`, `↑ top`, `→ right`, `↓ bottom`)
- `z` / `y`: undo / redo
- `Enter`: switch dataset or `group/dataset`
- `Space`: jump to image by name
- `n` / `p`: next / previous image
- `s`: save outputs
- `i`: idle toggle (hide/show grids)
- `r`: clear all ROIs
- `q` or `Esc`: quit

</details>

<a id="workflows"></a>
## 🧪 Workflows
<details open>
<summary>Quick tutorial</summary>

Short walkthrough showing ROI selection.

<img src="figures/basic_op_video.gif" style="width: 100%;" alt="Basic operations">

</details>

<details open>
<summary>Basic ROI operations</summary>

| Select ROI                                  | Move ROI                                | Add ROI                                    |
|---------------------------------------------|-----------------------------------------|--------------------------------------------|
| ![Select ROI](figures/select_crop_area.png) | ![Move ROI](figures/move_crop_area.png) | ![Add ROI](figures/add_more_crop_area.png) |

| Reselect ROI                                    | Delete ROI                                  |
|-------------------------------------------------|---------------------------------------------|
| ![Reselect ROI](figures/reselect_crop_area.png) | ![Delete ROI](figures/delete_crop_area.png) |

</details>

<details>
<summary>Layout directions</summary>

| Layout left                                 | Layout right                                  | Layout top                               | Layout bottom                                   |
|---------------------------------------------|-----------------------------------------------|------------------------------------------|-------------------------------------------------|
| ![Layout left](figures/layout_left_img.png) | ![Layout right](figures/layout_right_img.png) | ![Layout top](figures/layout_top_img.png) | ![Layout bottom](figures/layout_bottom_img.png) |

</details>

<details>
<summary>Layout capacities</summary>

| Single ROI (one)                      | Two ROIs                              | Three ROIs                                | More than three ROIs                    |
|---------------------------------------|---------------------------------------|-------------------------------------------|-----------------------------------------|
| ![Layout one](figures/layout_one.png) | ![Layout two](figures/layout_two.png) | ![Layout three](figures/layout_three.png) | ![Layout more](figures/layout_more.png) |

</details>

<details>
<summary>Mouse-only quick tour</summary>

Optional mouse-only quick tour (all operations by mouse):
<img src="figures/mouse_op_video.gif" style="width: 100%;" alt="Mouse-only operations (optional)">

</details>

<a id="saving"></a>
## 💾 Saving
On `s`, outputs are written to:

```
<output>/<timestamp>/<dataset>/<image-basename>/
  orig_<method>.png
  final_<method>.png
  crop_roi<id>_<method>.png
```

Each method writes its own originals, composed final previews, and all crops for active ROIs.

<a id="logs"></a>
## 🎨 Logs
- Linux terminals show colored status messages: info (cyan), success (green), warnings (yellow), errors (red), notes (bright cyan).
- Toggle with `--no-color`; set verbosity via `--log-level`.

<a id="troubleshooting"></a>
## 🛠️ Troubleshooting
- "Folder has no images": check dataset path and image extensions.
- "timeline_methods.txt is required": needed for `--source external`.
- GUI not responsive: ensure OpenCV GUI support; on headless servers use a local machine or X-forwarding.

<a id="acknowledgements"></a>
## 🙏 Acknowledgements
Built for fast LLIE evaluation workflows.
