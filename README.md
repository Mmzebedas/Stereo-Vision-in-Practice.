# Stereo-Vision-in-Practice.
Python-based stereo computer vision project using OpenCV, SIFT, RANSAC and StereoSGBM for image matching, epipolar geometry estimation, stereo rectification, disparity mapping and interactive 3D visualization.
# Stereo Vision & Photogrammetry — Interactive Computer Vision Pipeline

**A Python/OpenCV project for matching two photographs, estimating epipolar geometry, rectifying a stereo pair, and exploring disparity through an interactive visual interface.**

**Author / portfolio presentation:** Mikel Zeberio Dasca  
**Context:** Collaborative university computer vision and photogrammetry project  
**Core technologies:** Python · OpenCV · NumPy · SIFT · Brute-Force Matching · RANSAC · StereoSGBM

> This project uses **classical computer vision and geometric estimation**—not a trained deep-learning model. It focuses on building, validating, and presenting a complete image-processing pipeline.

## Project overview

Starting from two photographs of the same scene taken from different viewpoints, the application detects distinctive image features, finds potential correspondences, rejects geometrically inconsistent matches, and estimates the relationship between the two views. It then rectifies the images and produces visualizations that help examine relative depth and stereo geometry.

Beyond the image-processing pipeline, the application includes an **interactive OpenCV viewer** with multiple visualization modes, zoom/pan, per-pixel inspection, screenshot export, and automatic HTML/TXT reports.

## Visual results

The application generates seven visualization files in `resultats_versio3_final/`:

| Output | What it demonstrates |
| --- | --- |
| `01_correspondencies_sift.png` | Feature correspondences retained after RANSAC filtering |
| `02_rectificada_esquerra.png` | Rectified left image |
| `03_rectificada_dreta.png` | Rectified right image |
| `04_anaglif.png` | Red/cyan anaglyph combining both rectified views |
| `05_diferencia_rectificades.png` | Absolute grayscale difference between rectified images |
| `06_mapa_disparitat.png` | Colorized stereo disparity visualization |
| `07_linies_epipolars.png` | Epipolar lines and matched points for geometric inspection |

If you upload the generated results alongside the code, GitHub can display a visual gallery using these links:

![SIFT feature correspondences](resultats_versio3_final/01_correspondencies_sift.png)

*Feature matches: corresponding structures in the two views are connected by lines. Only RANSAC-consistent matches are displayed.*

![Rectified stereo anaglyph](resultats_versio3_final/04_anaglif.png)

*Anaglyph: the red and cyan channels combine the rectified pair and reveal view-dependent displacement.*

![Disparity visualization](resultats_versio3_final/06_mapa_disparitat.png)

*Disparity: estimated horizontal offsets between the rectified images. The colors are for visualization; they do not represent calibrated metric depth.*

## Pipeline

```text
Two photographs
      │
      ▼
SIFT feature detection and description
      │
      ▼
Brute-force descriptor matching (L2, cross-check)
      │
      ▼
RANSAC fundamental-matrix estimation
      │
      ▼
Uncalibrated stereo rectification
      │
      ├── Rectified left/right images
      ├── Red/cyan anaglyph
      ├── Difference image
      ├── StereoSGBM disparity visualization
      └── Epipolar-line and feature-match illustrations
      │
      ▼
Interactive viewer + HTML/TXT results reports
```

### 1. Scale-aware SIFT detection

The script optionally downsizes images for faster feature extraction, then converts keypoint coordinates back to the original resolution before geometric estimation:

```python
sift = cv.SIFT_create(nfeatures=MAX_SIFT_FEATURES)
kp1_small, desc1 = sift.detectAndCompute(gray_left, None)
kp2_small, desc2 = sift.detectAndCompute(gray_right, None)

kp1 = reescalar_keypoints(kp1_small, scale_left)
kp2 = reescalar_keypoints(kp2_small, scale_right)
```

This balances computational cost with the need to rectify the original-resolution images.

### 2. Matching and geometric validation

A cross-checked brute-force matcher selects candidate pairs. RANSAC then estimates the fundamental matrix and identifies matches that agree with a common epipolar geometry:

```python
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = sorted(bf.match(desc1, desc2), key=lambda m: m.distance)
good_matches = matches[:MAX_MATCHES]

F, mask = cv.findFundamentalMat(
    pts1, pts2,
    method=cv.FM_RANSAC,
    ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
    confidence=RANSAC_CONFIDENCE,
)
```

**Why it matters:** Descriptor similarity alone does not guarantee that two points represent the same physical feature. The RANSAC inlier mask provides a geometric consistency check.

### 3. Rectification and disparity

The application estimates projective transforms for the stereo pair and uses OpenCV's semi-global block matching (`StereoSGBM`) to compute a disparity visualization:

```python
success, H1, H2 = cv.stereoRectifyUncalibrated(
    pts1_inliers, pts2_inliers, F, imgSize=(w, h)
)
rect_left = cv.warpPerspective(img_left, H1, (w, h))
rect_right = cv.warpPerspective(img_right, H2, (w, h))

stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=7,
    P1=8 * 3 * 7 * 7,
    P2=32 * 3 * 7 * 7,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY,
)
disparity = stereo.compute(gray_left, gray_right).astype(np.float32)
```

The full source includes additional matching, visualization, and filtering parameters.

## Reproduced experimental results

One reproducible run using the supplied **2,048 × 1,365 px** image pair and OpenCV **4.13.0** produced:

| Metric | Result |
| --- | ---: |
| SIFT keypoints (left) | 4,000 |
| SIFT keypoints (right) | 4,000 |
| Initial descriptor matches | 1,624 |
| Best matches considered for RANSAC | 500 |
| RANSAC inliers | **417** |
| Inlier fraction among selected matches | **83.4%** |

The original high-resolution university screenshots show **294 inliers** on a different **6,240 × 4,160 px** image run. These values are from *different input resolutions/runs* and should not be interpreted as a controlled before/after improvement.

**Interpretation:** The inlier count indicates that many tentative correspondences follow a consistent fundamental-matrix model. It does not, by itself, prove pixel-perfect rectification or calibrated depth accuracy.

## Installation

**Requirements:** Python 3.10+ recommended, a local desktop environment capable of displaying OpenCV windows, and the two input images.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the libraries:

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python numpy
```

> Use `opencv-python` (not a headless build) if you want to run the interactive viewer.

## Run the project

Keep the input images beside your Python script with these filenames (used by the original code):

```text
stereo-vision-photogrammetry/
├── README.md
├── stereo_vision.py
├── imatge1 practica1.jpg
├── imatge2 practica 1.jpg
└── resultats_versio3_final/     # created automatically
```

Run from that folder:

```bash
python stereo_vision.py
```

The application processes both images, exports the result images and reports into `resultats_versio3_final/`, then launches the interactive viewer.

**Note:** If you rename the input images or store them in an `images/` folder, change `IMG_LEFT_PATH` and `IMG_RIGHT_PATH` near the top of the script accordingly.

## Interactive viewer controls

| Key / action | Function |
| --- | --- |
| `1` / `2` | Rectified left / right image |
| `3` | Red/cyan anaglyph |
| `4` | Image difference |
| `5` | Disparity visualization |
| `C` | SIFT correspondences |
| `E` | Epipolar lines |
| `+` / `-` | Zoom in / out |
| `R` | Reset view |
| `D` | Save screenshot |
| `Q` / `Esc` | Quit |
| Left mouse drag | Pan image |
| Right mouse click | Inspect pixel information |

## Technical considerations

- **Uncalibrated geometry:** Rectification is computed from image correspondences rather than known camera intrinsics and extrinsics.
- **Depth vs. disparity:** `StereoSGBM` estimates image displacement; metric distance requires additional camera calibration and scale information.
- **Difficult scene regions:** Occlusions, repeated textures, foliage, and inconsistent lighting can create unreliable matches and disparity estimates.
- **Resolution and performance:** Feature extraction and disparity processing are performed on resized copies when necessary to control runtime and memory use.
- **Validation:** Feature-match counts are useful diagnostics, but stronger evaluation includes epipolar residuals, post-rectification vertical alignment, and disparity consistency checks.

## Repository contents

- `stereo_vision.py` — Main OpenCV processing and desktop-viewer program.
- `imatge1 practica1.jpg` and `imatge2 practica 1.jpg` — Example stereo pair.
- `resultats_versio3_final/` — Generated images and HTML/TXT report, when included.

## Engineering skills demonstrated

Python programming · OpenCV · image processing · feature detection and matching · numerical geometry · RANSAC / robust estimation · stereo vision · computational efficiency · visual debugging · interactive desktop tooling · technical reporting.

---

**Project presentation:** Mikel Zeberio Dasca  
**Academic context:** Universitat Politècnica de Catalunya (UPC)  
**Development:** Collaborative university project; teammate's identity omitted from this public portfolio presentation.
