&nbsp;

<p align="center"> <img src="figures/layout_more.png" width="100%" alt="Image Crop Comparator &#40;AI generated logo&#41;"> </p>

# ✨ Image Crop Comparator 

Image Crop Comparator (ICC): A research-oriented, interactive image crop comparator for pixel-level method analysis, designed for a fast and flexible interactive workflow with multi-ROI selection, auto layout, undo/redo, side-by-side ROI comparison, and more advanced options. 

<div align="center">

[![Gradio Spaces Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20-Demo-violet)](https://huggingface.co/spaces/OrangeC4709/ImageCropComparator)

</div>

<details>
<summary>Image Crop Comparator Demo</summary>

![Image Crop Comparator Demo](figures/gradio_page.png)

</details>

## Overview 📌

### Features ✨
- ROI-based multi-image comparison viewer for image restoration and more
- Fast and flexible multi-ROI selection via keyboard and mouse shortcut, with undo/redo support
- Automatic layout arrangement (left / right / top / bottom)
- Support for multiple workspace structures
- Highly customizable CLI for future research, and a user-friendly Web interface (gradio) for improved experience

### News 🆕 

- **2025.12.30** A quick demo of our tool is available on [Hugging Face](https://huggingface.co/spaces/OrangeC4709/ImageCropComparator). 🤗
- **2025.12.30** Update Web version, a Gradio-based UI, providing better user experience for broader users. 🚀
- **2025.12.29** Update [basic tutorial](#workflows). 🧾
- **2025.12.28** Update ICC-CLI original-version, a runnable command-line (CLI) tool, including local/external sources, ROI selection, layout preview, and export. 🎈

## Contents 📑 
- Getting Started 🚀
  - [Installation](#installation)
  - [Quick Start (CLI)](#quick-start)
  - [Quick Start (Gradio)](#quick-start-gradio)
  - [Workspace Layout](#workspace-layout)
- User Guidance 🧭
  - [Tutorials](#tutorials)
  - [Interaction](#interaction)
  - [Usage](#usage)
  - [Output](#output)
- Reference ⚙️ 
  - [CLI Options](#cli-options)
  - [Logs](#logs)
  - [Troubleshooting](#troubleshooting)
- Community 🤝 
  - [Call to Action](#call-to-action)

[//]: # (- 🤝 Community)

[//]: # (  - [Contributing]&#40;#contributing&#41;)

[//]: # (  - [Acknowledgements]&#40;#acknowledgements&#41;)

[//]: # (  - [Citation]&#40;#citation&#41;)

## Getting Started 🚀

<a id="installation"></a>
### Installation 🧰
- Python 3.8+
- [requirements.txt](requirements.txt) pins `opencv-python==4.7.*` and `numpy==1.26.*`; install with `pip install -r requirements.txt`.
- Packages: opencv-python, numpy, natsort, Pillow

```bash
# Conda (recommended)
conda create -n crop-comparer python=3.10 -y  # python>=3.8 (e.g. 3.9) is also supported, and python<=3.12 is recommended
conda activate crop-comparer
pip install -r requirements.txt
```

<details>
<summary>venv alternative</summary>

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

</details>

<a id="quick-start"></a>
### Quick Start (CLI) 🚀
```bash
python compare.py
```
- Minimal two-step flow for local reviews:
  1) Copy your method results into `examples/` (or another root that matches the structures below).
  2) Run `python compare.py` and start selecting ROIs; defaults work out of the box.
- If your layout differs, set `Root`/`Structure` in the UI or CLI to point to your folders.

<a id="quick-start-gradio"></a>
### Quick Start (Gradio) 🚀
```bash
python gradio_app.py
```
- Runs the Gradio UI locally on http://127.0.0.1:7860/ by default.
- Same two-step flow: drop images into your root (default: `examples/`), launch, and compare. Set `Root`/`Structure` if you use a custom layout.

<a id="workspace-layout"></a>
### Workspace Layout 🗂️
(for default local source)
Methods live under the workspace root, and each holds datasets.

```
<root>/
  <method>/
    <dataset>/
      <images...>
```

<details>
<summary>Example</summary>

```
examples/         # <root>
  A-Net/          # <method>
    LOL-v2-real/  # <dataset>
      00698.png   # <images...>
      00775.png
    SDSD-indoor/
      00101.png
      00259.png
  B-Former/
    LOL-v2-real/
      00698.png
      00775.png
    SDSD-indoor/
      00101.png
      00259.png
  ...
```

</details>

<details open>
<summary>Supported structures (use `--structure`)</summary>

- `group-dataset-pair`: `<root>/<method>/<group>/<dataset>/<pair>/<img>.png`
- `group-dataset` (default classic): `<root>/<method>/<group>/<dataset>/<img>.png`
- `dataset-only`: `<root>/<method>/<dataset>/<img>.png`
- `flat`: `<root>/<method>/<img>.png`
- `shared`: `<root>/<image-id>/<method>.png`
- `auto` (default): tries the above in order.

</details>

<details>
<summary>Structure examples</summary>

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

3. `dataset-only`: `<root>/<method>/<dataset>/<img>.png` (example of current default for local source)
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

</details>

## User Guidance 🧭

<a id="tutorials"></a>
### Tutorials 🧪
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

| Single ROI (one)                     | Two ROIs                              | Three ROIs                                | More than three ROIs                      |
|--------------------------------------|---------------------------------------|-------------------------------------------|-------------------------------------------|
| ![Layout one](figures/layout_one.png) | ![Layout two](figures/layout_two.png) | ![Layout three](figures/layout_three.png) | ![Layout more](figures/layout_more.png)   |

</details>

<details>
<summary>Mouse-only quick tour</summary>

Optional mouse-only quick tour (all operations by mouse):
<img src="figures/mouse_op_video.gif" style="width: 100%;" alt="Mouse-only operations (optional)">

</details>


<a id="interaction"></a>
### Interaction 🎮

<details open>
<summary>Keyboard</summary>

|      Key      | Action                                                |
|:-------------:|-------------------------------------------------------|
|      `a`      | Add new ROI                                           |
|     `1–9`     | Add / select ROI by id; Tapping active id to reselect |
|  `Shift+1–9`  | Duplicate active ROI to target id or copy its size    |
|      `d`      | Add ROI with active size                              |
|   `← ↑ → ↓`   | Change layout direction                               |
|    `z / y`    | Undo / redo                                           |
|    `Enter`    | Switch dataset (`group/dataset` or `dataset`)         |
|    `Space`    | Jump to image by name                                 |
|    `n / p`    | Next / previous image                                 |
|      `s`      | Save outputs                                          |
|      `i`      | Toggle idle (hide/show grids)                         |
|      `r`      | Clear all ROIs                                        |
|   `q / Esc`   | Quit                                                  |

> Note: Arrow keys (`← ↑ → ↓`) control the layout direction, not ROI movement. 
> ROI movement is performed by mouse dragging in position mode.

</details>

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


<a id="usage"></a>
### Usage 🧭
Local images with auto-discovery:

```bash
python compare.py --source local --root <root> --group <group> --dataset <dataset> --layout right
```

<details>
<summary>External datasets</summary>

External (video-form) datasets require `methods.txt`:

```bash
python compare.py --source external --dataset SDSD-indoor --pair pair13 --layout left
```

</details>


<a id="cli-options"></a>
### CLI Options ⚙️
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


<a id="output"></a>
### Output 💾
On `s`, outputs are written to:

```
<output>/<timestamp>/<dataset>/<image-basename>/
  orig_<method>.png
  final_<method>.png
  crop_roi<id>_<method>.png
```

Each method writes its own originals, composed final previews, and all crops for active ROIs.

## Reference ⚙️

<a id="logs"></a>
### Logs 🎨
- Linux terminals show colored status messages: info (cyan), success (green), warnings (yellow), errors (red), notes (bright cyan).
- Toggle with `--no-color`; set verbosity via `--log-level`.

<a id="troubleshooting"></a>
### Troubleshooting 🛠️

**Issue:**
```
cv2.error: OpenCV(4.7.0) /io/opencv/modules/highgui/src/window.cpp:1266: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvDestroyAllWindows'
```
**Solution:**

1. uninstall `opencv-python-headless` and reinstall `opencv-python`:
    ```bash
    pip uninstall opencv-python-headless
    pip install "opencv-python==4.7.*"
    ```
2. If operation system is Linux, ensure GTK is installed:
    ```bash
    sudo apt-get install libgtk2.0-dev pkg-config
    ```

<a id="call-to-action"></a>
## Call to Action ⭐

* ⭐ If you like ICC, give it a star. It helps others discover the project and supports its growth.

[//]: # (## 🤝 Community)

[//]: # ()
[//]: # (<a id="contributing"></a>)

[//]: # (### 🤝 Contributing)

[//]: # ()
[//]: # (We welcome contributions of all kinds, including bug reports, feature requests, documentation improvements, and code contributions.)

[//]: # ()
[//]: # (<a id="acknowledgements"></a>)

[//]: # (### 🙏 Acknowledgements)

[//]: # (We thank all contributors and users who help improve this project through feedback, discussions, and code contributions.)

[//]: # (<a id="citation"></a>)

[//]: # (### 📚 Citation)

[//]: # ()
[//]: # (This tool is a byproduct of the following research work.  )

[//]: # (If you find **Image Crop Comparator &#40;ICC&#41;** useful for your research or practice, please consider citing:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@inproceedings{xu2025urwkv,)

[//]: # (  title     = {URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Restoration},)

[//]: # (  author    = {Xu, Rui and Niu, Yuzhen and Li, Yuezhou and Xu, Huangbiao and Liu, Wenxi and Chen, Yuzhong},)

[//]: # (  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition &#40;CVPR&#41;},)

[//]: # (  pages     = {21267--21276},)

[//]: # (  year      = {2025})

[//]: # (})