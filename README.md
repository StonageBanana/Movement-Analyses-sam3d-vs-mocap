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
phase, importing the real code and showing the real outputs (including four of
the real bugs found along the way, reproduced live on real data — a fifth,
the pelvis-reference-frame defect described in the verdict below, was found
after the notebook was last executed and isn't in it yet). GitHub renders it
directly, no setup needed to read it. [`notebooks/per_joint_comparison.ipynb`](notebooks/per_joint_comparison.ipynb)
goes deeper on a single question: per joint, is view1/view2/fused error
dominated by noise (more cameras would help) or bias (they wouldn't) — with
all 19 joints compared across view1, view2, fused, and mocap.

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
    export_trc.py          # export aligned/fused trajectories as .trc (Vicon Nexus/Visual3D/OpenSim)
    gait_cycle.py           # gait-cycle boundary detection + %-cycle normalization (shared helper)
    visualize_gait_cycles.py           # mocap-vs-fused, classic mean+-std vs. %-gait-cycle plots
    visualize_hip_diagnostics.py       # root-cause diagnostics for the pelvis-frame bug (4 plots/trial)
    visualize_correlation_before_after.py  # per-trial before/after-fix correlation bar chart
    visualize_knee_gait_before_after.py    # knee flexion gait-cycle plot, before vs. after the fix
    visualize_all_sources_gait_cycle.py    # view1/view2/fused/mocap overlaid, %-gait-cycle, per trial
    visualize_all_sources_realtime.py      # same 4-way overlay, real-time (non-periodic trials)
    visualize_upper_body_gait_cycle.py     # %-gait-cycle position plots for the 8 upper-body joints
    visualize_correlation_matrix_after_fix.py  # all 24 signals x all 10 trials, one heatmap
    visualize_overlay.py   # renders 2D keypoints over the raw source video, per trial/view
    analyze_speed_vs_accuracy.py           # does locomotion cadence correlate with tracking error?
    test_single_frame.py   # one-off dev smoke test (single frame, no detector) -- not part of the pipeline
  output/                  # every phase's generated artifacts (see table below)
  third_party/sam-3d-body/ # cloned facebookresearch/sam-3d-body (gitignored, see setup)
  notebooks/
    project_walkthrough.ipynb   # the full narrative walkthrough -- start here
    per_joint_comparison.ipynb  # per-joint noise-vs-bias deep dive (would more cameras help?)
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
| — | `export_trc.py` | `output/aligned/*`, `output/aligned_fused/*` | `output/trc_export/*.trc` | Re-exports the aligned per-view and fused trajectories as `.trc`, so they load next to the real mocap files in Vicon Nexus/Visual3D/OpenSim |

Three visualization scripts (`visualize_overview.py`, `visualize_phase456.py`,
`visualize_phase7.py`) produce the mid-session checkpoint figures in
`output/overview/` — useful for sanity-checking a phase right after building
it, distinct from Phase 8's polished `output/final/` pass.

## Supplementary analysis & diagnostics

A second layer of scripts, built after the pelvis-frame bug (see the verdict
below) to root-cause it and then answer specific research questions the
8-phase pipeline doesn't address head-on. All read from `output/aligned_fused/`
(post Phase 6) or raw `output/sam3d/`/`output/mocap/`, so none of them require
re-running the core pipeline.

| Script | What it produces |
|---|---|
| `gait_cycle.py` | Shared module: detects gait-cycle boundaries from the broadband (all-joint mean) vertical signal and resamples any per-frame signal to `% gait cycle`. Not run directly. |
| `visualize_gait_cycles.py <trial>` | The classic clinical plot: mocap-vs-fused mean +/- 1 std curves vs. % gait cycle, for all 6 flexion angles and all 19 joint positions, plus a Pearson-r classification table (`output/gait_cycle/<trial>/`) |
| `visualize_hip_diagnostics.py <trial>` | The 4 plots that root-caused the pelvis-frame bug: hip-separation-over-time, pelvis-axis misorientation, and hip/knee flexion before vs. after frame-correcting (`output/diagnostics/<trial>/01-04_*.png`) |
| `visualize_correlation_before_after.py <trial>` | One bar chart, all 24 comparable signals, correlation with mocap before vs. after the fix (`.../05_correlation_before_after.png`) |
| `visualize_knee_gait_before_after.py <trial>` | Knee flexion gait-cycle curve, before vs. after the fix (`.../06_knee_flexion_gait_before_after.png`) |
| `visualize_all_sources_gait_cycle.py <trial>` | mocap + view1 + view2 + fused overlaid, % gait cycle, all 6 flexion angles — the full post-fix picture for periodic trials (`.../09_all_sources_gait_cycle.png`) |
| `visualize_upper_body_gait_cycle.py <trial>` | Same 4-way overlay for the 8 upper-body joints (position, not angle) (`.../10_upper_body_gait_cycle.png`) |
| `visualize_all_sources_realtime.py <trial>` | Same 4-way overlay as above but real-time instead of %-gait-cycle, for non-periodic trials (random/dance/feet-movements) (`.../11_all_sources_realtime_full_trial.png`) |
| `visualize_correlation_matrix_after_fix.py` | One heatmap, all 24 signals x all 10 trials, final post-fix correlation with mocap (`output/diagnostics/07_correlation_matrix_after_fix.png`) |
| `analyze_speed_vs_accuracy.py` | Does cadence (from mocap, not the estimate) predict tracking error, for the 5 locomotion trials? (`output/diagnostics/08_speed_vs_accuracy.png`) |
| `visualize_overlay.py <trial> <view>` | Renders the raw source video with SAM 3D Body's 2D keypoints drawn on top, for visual QA (`output/overlay/*.mp4`, gitignored — regenerate locally, not committed since a full set is ~600MB) |

Gait-cycle boundary detection (`gait_cycle.py`) is tuned for walking cadence
(`min_period=0.9s`); it needs a per-trial override for squats' slower ~2.8s
rep cycle (already applied in the scripts above via `MAX_PERIOD_OVERRIDE`)
and is **not yet reliable for the running trials**, whose faster stride can
make the detector lock onto multi-stride super-cycles — flagged here rather
than silently trusted.

## The verdict

Full numbers in [`output/synthesis/phase7_synthesis.json`](output/synthesis/phase7_synthesis.json)
and [`output/final/report_data.json`](output/final/report_data.json); the short version:

- **Fusion helps, modestly**: beats the best single view's MPJPE in 7/10 trials
  (mean +3.7%), but is close to a wash on PA-MPJPE (-0.8%) and roughly a wash
  on joint angles too — it mainly corrects independent per-camera positional
  noise (biggest gains on distal joints: shoulders, heels, wrists, ankles, all
  +10-17%), not systematic single-cause bias (`hip_right`, a known ~220mm
  tracking offset in this dataset, barely moves at +1.0%, since both cameras
  share the same bias).
- **A major joint-angle bug was found and fixed mid-project**: SAM 3D Body's
  own estimated hip-left/hip-right keypoints turned out to be compressed to
  ~44% of the true anatomical hip width and rotated ~106° off the true pelvis
  axis — a fixed, activity-independent distortion present in *every one* of
  the 10 trials (walking, running, squats, dance, etc.), and confirmed already
  present in each camera's raw, pre-alignment output — not something
  introduced by `align.py`/`fuse_views.py` (those are similarity transforms
  and provably cannot change internal shape ratios like hip-width-to-shoulder-
  width). Every hip/knee/ankle flexion angle is computed from a pelvis frame
  built from that same hip-to-hip vector, so this one defect corrupted all of
  them, even though the underlying joint positions were largely fine. Fix:
  compute the estimate's angles using mocap's own pelvis frame instead of the
  estimate's own (`mocap.angles.compute_joint_angles_from_joints`'s new
  `frame_joints` parameter). This dropped `hip_flexion_right` RMSE from 55° to
  17° and `knee_flexion_right` from 84° to 21° project-wide, and flipped their
  correlation with mocap from strongly negative to strongly positive.
- **Not clinically usable yet, by a substantial margin — but the position gap
  is now the dominant one**: best-case MPJPE (~78mm) is roughly 3.2x worse
  than published markerless benchmarks (~24mm). Joint angles, after the fix
  above, are a smaller gap: the best angle series (`hip_flexion_right`, ~17deg
  RMSE) is 3-8x worse than the ~2-5deg considered clinically acceptable in the
  literature. `ankle_flexion_right` remains separately unreliable in dynamic
  trials — a genuine, unresolved wrap-around discontinuity, unrelated to and
  unaffected by the pelvis-frame fix.
- **The gap looks addressable, not fundamental** — and the pelvis-frame fix
  above is a concrete instance of that, not just a hope: this setup used zero
  camera calibration, no person detector, 10fps-subsampled inference, and a
  deliberately simplified (non-ISB) joint-angle convention. The 1->2 camera
  improvement came from averaging out independent viewpoint noise — a 3rd/4th
  camera would likely keep helping there, but won't fix the systematic hip
  position bias or `ankle_flexion_right`'s wrap issue without calibration and
  a better anatomical model.
- Tested (and rejected) confidence-weighted fusion: inter-view disagreement
  does **not** reliably predict true error against mocap in this dataset
  (r=-0.13, p=0.07 overall) — a joint can have low disagreement while both
  cameras are equally wrong (exactly what happens at `hip_right`).

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

# Optional: .trc re-export (for Vicon Nexus/Visual3D/OpenSim) and the
# supplementary gait-cycle/diagnostic scripts (see table above; most take
# a trial name as an argument, e.g. `walking_1`):
.venv\Scripts\python.exe src\export_trc.py
.venv\Scripts\python.exe src\visualize_gait_cycles.py walking_1
.venv\Scripts\python.exe src\visualize_correlation_matrix_after_fix.py
.venv\Scripts\python.exe src\analyze_speed_vs_accuracy.py
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
- **SAM 3D Body's own hip-left/hip-right keypoints are geometrically
  distorted** (~44% of the true hip width, ~106° off the true pelvis axis) —
  a fixed, activity-independent defect confirmed in every trial and already
  present pre-alignment (not introduced by `align.py`/`fuse_views.py`, which
  are similarity transforms and can't change internal shape ratios). This
  corrupted every hip/knee/ankle flexion angle for the estimate side, since
  they're all computed relative to a pelvis frame built from that vector.
  Fixed by having `compute_joint_angles_from_joints` (new `frame_joints`
  param) project the estimate's angles onto **mocap's** pelvis frame instead
  of its own — see `CLAUDE.md`'s "Gotchas" bug #5 for the full writeup and
  before/after numbers.
- **Skeleton front/side views must use anatomically-true axes, not global
  X/Z** — the raw camera/mocap coordinate axes are arbitrarily rotated
  relative to which way the subject is actually facing, so projecting onto
  them can produce a foreshortened, oblique-looking "front" view even when
  the code is otherwise correct. `visualize_final.py`'s skeleton filmstrip
  instead computes, per frame, `left_hat = normalize(hip_left - hip_right)`
  and `anterior_hat = normalize(cross(left_hat, up))` from **mocap's own**
  joints, and projects all sources (mocap, fused) onto those shared axes.
  Relatedly, matplotlib's default axis orientation is already correct here
  (mocap's Y-up convention means large Y = head, small Y = feet) — an
  `ax.invert_yaxis()` call previously present in both `visualize_final.py`
  and `visualize_phase456.py` was flipping already-correct skeletons upside
  down; removed in both places.
- **Gait-cycle detection is tuned for walking cadence** (`min_period=0.9s`
  in `gait_cycle.py`) — squats need a per-trial period override
  (`MAX_PERIOD_OVERRIDE`) and the running trials' faster stride is not yet
  reliably segmented (the detector can lock onto a multi-stride
  super-cycle); treat any running-trial gait-cycle plot with caution.
