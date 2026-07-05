# Movement Analyses: SAM 3D Body vs. Marker-Based Mocap

A university seminar project (Research in Movement Analysis) asking one question:
**can a two-camera markerless setup (Meta's SAM 3D Body model) substitute for a
marker-based Vicon mocap lab in clinical gait/movement analysis?**

Ten movement trials (walking x3, running x2, squats x2, dance, feet-movements,
random), each recorded simultaneously by a 10-camera Vicon mocap rig *and* two
ordinary RGB cameras from different angles. The pipeline runs SAM 3D Body
independently on each RGB view, aligns each to the mocap ground truth, fuses the
two views together with **zero calibration or ground-truth involved in the
fusion itself**, and then scores the fused result the same way — arriving at a
data-driven, literature-grounded verdict.

**New here? Start with [`notebooks/project_walkthrough.ipynb`](notebooks/project_walkthrough.ipynb)** —
a single executed notebook telling the whole project as a narrative, phase by
phase, importing the real code and showing the real outputs (including the
three real bugs found along the way, reproduced live on real data). GitHub
renders it directly, no setup needed to read it.

Otherwise, read [Phase 7's synthesis](output/synthesis/phase7_synthesis.json) for the
actual numbers, or skip straight to [the verdict](#the-verdict) below.

## What's in this repo, and what isn't

This repo contains all *code* and all *generated outputs* (parsed joints,
angles, metrics, figures). It does **not** contain the raw Vicon mocap files
(`.trc`/`.c3d`/etc.) or the source RGB videos — that's real human-subject
research data from a university study, and this repo is public, so the raw
recordings stay off GitHub. Everything under `output/` was derived from that
raw data by the scripts in `src/`; if you have access to the original dataset,
point `build_manifest.py` at it and every phase is reproducible from scratch
(see [Reproducing this yourself](#reproducing-this-yourself)).

## Repository layout

```
analysis/
  manifest.json          # trial -> mocap file paths + both video paths + fps + paired flag
  requirements.txt        # Phase 0/1 deps (numpy/scipy/opencv/ezc3d/...)
  constraints.txt         # pins torch to the CUDA build (see gotchas below)
  src/
    build_manifest.py     # Phase 0
    parse_mocap.py         # Phase 1 orchestrator
    mocap/                 # Phase 1 package: .trc/.vsk I/O, joint centers, joint angles
    run_sam3d.py           # Phase 2 (interactive: pick trial+view to run through SAM 3D Body)
    align.py               # shared Umeyama/Procrustes + cross-correlation primitives (Phase 3+)
    run_alignment.py       # Phase 3 orchestrator (per view -> mocap)
    compare_metrics.py     # Phase 4 (per-view MPJPE/PA-MPJPE/joint-angle metrics)
    fuse_views.py          # Phase 5 (GPA-based two-view fusion, no mocap/calibration)
    align_fused.py         # Phase 6 (fused -> mocap alignment)
    compare_metrics_fused.py  # Phase 6 metrics (reuses Phase 4's functions directly)
    synthesize.py          # Phase 7 (capstone: view1 vs view2 vs fused, verdict)
    visualize_overview.py  # Phase 1-3 checkpoint figures
    visualize_phase456.py  # Phase 4-6 checkpoint figures
    visualize_phase7.py    # Phase 7 checkpoint figures
    visualize_final.py     # Phase 8 (presentation-ready figures + report_data.json)
    audit_all.py           # full regression check across every phase
  output/                  # every phase's generated artifacts (see table below)
  third_party/sam-3d-body/ # cloned facebookresearch/sam-3d-body (gitignored, see setup)
  notebooks/
    project_walkthrough.ipynb  # the full narrative walkthrough -- start here
```

## The phases

Each phase reads the previous phase's `output/` and writes its own — every
script is safe to re-run (later stages skip work whose output already exists).

| # | Script | Reads | Writes | What it does |
|---|--------|-------|--------|---------------|
| 0 | `build_manifest.py` | raw data folders | `manifest.json` | Scans the two video folders + mocap folder, confirms fps/frame counts, flags which trials have both camera views |
| 1 | `parse_mocap.py` (+ `mocap/`) | raw `.trc`/`.vsk` | `output/mocap/*.npz` | Parses 39-marker Vicon data -> 19-joint canonical skeleton + simplified sagittal hip/knee/ankle flexion angles, calibrated against the `static` trial |
| 2 | `run_sam3d.py` | raw videos | `output/sam3d/*.npz` | Runs SAM 3D Body per-frame per-view (every 3rd frame, ~10fps effective), full-image bbox (no detector), light Savitzky-Golay smoothing |
| 3 | `run_alignment.py` (+ `align.py`) | `output/sam3d/*` | `output/aligned/*.npz` | Cross-correlation temporal sync + one whole-trial Umeyama (rotation+scale+translation) fit, per view, into mocap's frame |
| 4 | `compare_metrics.py` | `output/aligned/*` | `output/metrics/phase4_metrics.json` | Per-view MPJPE, per-frame-Procrustes PA-MPJPE, joint-angle RMSE/MAE/Pearson r/Bland-Altman vs. mocap |
| 5 | `fuse_views.py` | `output/sam3d/*` | `output/fused/*.npz` | View1<->view2 temporal sync + whole-trial rigid fit + **per-frame two-shape Generalized Procrustes Analysis**, fusing the two views with no mocap/calibration involved |
| 6 | `align_fused.py` + `compare_metrics_fused.py` | `output/fused/*` | `output/aligned_fused/*.npz`, `output/metrics/phase6_metrics.json` | The *only* place mocap touches the fused result: same alignment method as Phase 3, then the same metrics as Phase 4 |
| 7 | `synthesize.py` | Phase 4 + 6 metrics | `output/synthesis/phase7_synthesis.json` | view1 vs. view2 vs. fused: per-trial/per-joint/per-category comparison, confidence-weighting premise test, literature-grounded verdict |
| 8 | `visualize_final.py` | everything above | `output/final/*.png`, `report_data.json` | Presentation-ready joint-angle time series, pooled Bland-Altman plots, MPJPE summaries, a skeleton filmstrip |
| — | `audit_all.py` | everything | console only | Regression check: NaN patterns, angle-range sanity, per-joint alignment residuals, reflection checks, PA-MPJPE<=MPJPE invariant, GPA convergence, left/right consistency |

Three visualization scripts (`visualize_overview.py`, `visualize_phase456.py`,
`visualize_phase7.py`) produce the mid-session checkpoint figures in
`output/overview/` — useful for sanity-checking a phase right after building
it, distinct from Phase 8's polished `output/final/` pass.

## The verdict

Full numbers in [`output/synthesis/phase7_synthesis.json`](output/synthesis/phase7_synthesis.json)
and [`output/final/report_data.json`](output/final/report_data.json); the short version:

- **Fusion helps, modestly**: beats the best single view's MPJPE in 8/10 trials
  (mean +4.5%), but is a wash on PA-MPJPE and joint angles — it mainly corrects
  independent per-camera positional noise (biggest gains on distal joints:
  heels, ankles, shoulders, wrists, all +15-20%), not systematic single-cause
  bias (`hip_right`, a known ~200mm tracking offset in this dataset, barely
  moves at +0.9%, since both cameras share the same bias).
- **Not clinically usable yet, by a substantial margin**: best-case MPJPE
  (~85mm) is roughly 3.5x worse than published markerless benchmarks (~24mm),
  and even the best joint angle (`hip_flexion_left`, ~17deg RMSE) is 3-8x worse
  than the ~2-5deg considered clinically acceptable in the literature.
- **The gap looks addressable, not fundamental**: this setup used zero camera
  calibration, no person detector, 10fps-subsampled inference, and a
  deliberately simplified (non-ISB) joint-angle convention. The 1->2 camera
  improvement came from averaging out independent viewpoint noise — a 3rd/4th
  camera would likely keep helping there, but won't fix the systematic hip
  bias or the angle-convention gap without calibration and a better anatomical
  model.
- Tested (and rejected) confidence-weighted fusion: inter-view disagreement
  does **not** reliably predict true error against mocap in this dataset
  (r=-0.14 overall, ~0 within-joint) — a joint can have low disagreement while
  both cameras are equally wrong (exactly what happens at `hip_right`).

## Reproducing this yourself

You'll need the original dataset (mocap `.trc`/`.c3d`/`.vsk` files + the two
camera folders of `.mov` videos) laid out as described in `manifest.json` —
this repo doesn't include it (see above).

```powershell
# 1. Environment
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu118
.venv\Scripts\python.exe -m pip install torchvision --index-url https://download.pytorch.org/whl/cu118
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"  # confirm GPU build

# 2. SAM 3D Body model repo + gated checkpoint
git clone https://github.com/facebookresearch/sam-3d-body third_party/sam-3d-body
# accept Meta's SAM license on Hugging Face, then: hf auth login
# download facebook/sam-3d-body-vith into third_party/sam-3d-body/checkpoints/sam-3d-body-vith/
# On Windows/macOS, remove/guard the PYOPENGL_PLATFORM=egl line in
# sam_3d_body/visualization/renderer.py (EGL is Linux-only)

# 3. Point build_manifest.py at your data, then run the pipeline in order
.venv\Scripts\python.exe src\build_manifest.py
.venv\Scripts\python.exe src\parse_mocap.py
.venv\Scripts\python.exe src\run_sam3d.py          # interactive; run once per trial/view
.venv\Scripts\python.exe src\run_alignment.py
.venv\Scripts\python.exe src\compare_metrics.py
.venv\Scripts\python.exe src\fuse_views.py
.venv\Scripts\python.exe src\align_fused.py
.venv\Scripts\python.exe src\compare_metrics_fused.py
.venv\Scripts\python.exe src\synthesize.py
.venv\Scripts\python.exe src\visualize_overview.py
.venv\Scripts\python.exe src\visualize_phase456.py
.venv\Scripts\python.exe src\visualize_phase7.py
.venv\Scripts\python.exe src\visualize_final.py

# Re-run any time after touching mocap/, align.py, joint_mapping.py,
# compare_metrics.py, fuse_views.py, or align_fused.py:
.venv\Scripts\python.exe src\audit_all.py
```

### Gotchas worth knowing before you touch the code

- **No object detector is used** (`human_detector=None`) — `process_one_image`
  falls back to a full-image bounding box. Correct for this single-person,
  static-camera dataset; avoids a Detectron2 dependency that doesn't build on
  Windows without MSVC tools.
- **A plain `pip install <anything depending on torch>` can silently pull a
  CPU-only torch from PyPI** and clobber the CUDA build — always reinstall
  with `--index-url https://download.pytorch.org/whl/cu118` afterward.
- **`pred_keypoints_3d` is root-relative**; the model's true global
  translation lives in the separate `pred_cam_t` field, which must be added
  back to every joint (`joint_mapping.sam3d_canonical_joints` does this) or
  whole-body motion is silently lost.
- **`hip_right` is consistently the worst-tracked joint** across this
  dataset (~200mm body-frame offset) — investigated thoroughly and believed
  to be a real occlusion artifact (subject gripping a handrail near that hip),
  not a code bug.
- **`ankle_flexion_right`** has a genuine, unresolved wrap-around discontinuity
  in dynamic trials — flagged per-trial rather than silently trusted (see
  `output/mocap/*.npz`'s `unreliable_angles` field).
