# chess-yolo

An attempt to detect chess moves from a phone camera using computer vision. It didn't work.

## The Goal

Build a system that watches a physical chess board through a phone camera and automatically records moves in PGN format. No expensive DGT board, just a phone and some ArUco markers.

## What I Tried

### Approach 1: YOLO Object Detection

Trained YOLOv8 to detect chess pieces. The model worked okay in isolation but failed badly in practice:

- Only detected 20-25 pieces out of 32 at any given time
- Pieces flickered in and out of detection frame-to-frame
- Confidence thresholds were a losing battle: too high meant missing pieces, too low meant false positives
- Even with temporal smoothing (requiring 30 stable frames), false moves got confirmed
- The model said I played Nh3. I didn't play Nh3.

### Approach 2: Frame Differencing

Gave up on piece detection entirely. Instead, just detect which squares changed between frames:

- Compare pixel signatures of each square against a reference frame
- If two squares changed and they match a legal move, register it
- Simpler in theory. Still didn't work.

Problems encountered:
- Finding the right similarity threshold was impossible. Too strict: missed the "from" square. Too loose: detected phantom changes everywhere.
- The grid alignment was never quite right despite trying margins of 40, 45, 50, 55, 60, 65 pixels
- When my hand blocked any ArUco marker during a move, the perspective transform went haywire and detected 60+ squares as changed
- Even with all markers visible and careful hand movement, moves wouldn't register. Did h2-h4, system saw h2-h3. Did Nf6, system saw g8-f5.

## The Setup

- Phone running IP Webcam app streaming at ~2 FPS
- Chess board with 3/4 square margin around the edge
- ArUco markers (4x4, IDs 0-3) placed at the four corners
- Camera mounted at an angle (not overhead)
- Room lighting plus phone flashlight

## What I Learned

The successful implementations I found online all have:
- Overhead camera mount (no occlusion, no perspective issues)
- Controlled, consistent lighting
- Higher frame rates
- Or they just use a $400 electronic board

I had none of these. Stacking multiple hard problems (angled perspective, variable lighting, low FPS, hand occlusion) doesn't work. Each one alone is solvable. All together, you just get frustrated.

## HPC Training

Trained on UNSW Katana HPC cluster with an NVIDIA H200 GPU (140GB VRAM). Two training runs were attempted.

### Training Setup

- **Model**: YOLOv8m (25.8M parameters)
- **Dataset**: ~1800 training images, 228 validation images from Roboflow
- **Classes**: 12 (white/black x king/queen/rook/bishop/knight/pawn)
- **Batch size**: 64
- **Image size**: 640x640
- **Hardware**: NVIDIA H200, 6 CPUs, 128GB RAM
- **Walltime requested**: 12 hours

### Results

The model converged quickly. Early stopping kicked in at epoch 138 (best model at epoch 88). Total training time: 18 minutes.

```
Final Metrics:
  Precision:    0.775
  Recall:       0.748
  mAP50:        0.750
  mAP50-95:     0.626
```

Per-class performance was reasonable:
- Best: black-bishop (mAP50 0.791), black-queen (0.768)
- Worst: black-pawn (mAP50 0.742), black-knight (0.741)

### The Problem

The metrics looked fine. 75% mAP50 seemed workable. But validation metrics don't tell the whole story.

In practice, on live video:
- Detection flickered constantly between frames
- The same piece would be detected, then not, then detected again
- Only 20-25 pieces visible at any given time instead of 32
- Confidence scores were unstable

The dataset was probably too clean. Studio lighting, overhead shots, consistent piece styles. My setup had none of that.

## Files

```
src/
  inference/
    aruco.py              # ArUco marker detection
    perspective.py        # Perspective correction
    frame_diff_detector.py # Frame differencing approach
    detector.py           # YOLO inference wrapper
    detection_smoother.py # Temporal smoothing (didn't help)
  chess_logic/
    occupancy_move_detector.py # Move detection from occupancy changes
scripts/
  live_gui_framediff.py   # Main GUI for frame diff approach
  live_gui.py             # Original YOLO-based GUI
  inference.py            # CLI inference script
```

## Running It

Don't. But if you want to:

```bash
python scripts/live_gui_framediff.py \
  --source http://<phone-ip>:8080/video \
  --margin 50 \
  --rotate180 \
  --flip
```

Press 'R' to set reference frame, then make moves. Watch them not get detected.

## What Would Actually Work

1. Mount the camera directly overhead
2. Use proper lighting
3. Get a faster video feed
4. Or just buy a DGT board

## Time Spent

Too much.

## Status

Archived. Moving on.
