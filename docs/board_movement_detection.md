# Board Movement Detection: Data Quality Filter

## Problem

The `cableholder-coarse-align` dataset had episodes where the circuit board
physically moved during recording. Since the board is supposed to be stationary
for a valid demonstration, these episodes would corrupt training — the model
would learn to expect/compensate for board motion that won't exist at test time.

The board position is not explicitly tracked in the data. We detect it
indirectly via **sudden scene changes** in the camera feed that cannot be
explained by normal robot arm motion.

## Signal: Frame-Difference Spikes

The insight: normal robot motion produces **gradual, localized** pixel changes.
A board shift produces a **sudden, global** pixel change across the entire frame
(including background regions where the arm never moves).

For each episode, we sample every Nth frame (stride=5) and compute the **Mean
Absolute Difference** (MAD) between consecutive sampled frames:

```python
diff = cv2.absdiff(prev_gray, gray)
mad = np.mean(diff)  # single float per frame pair
```

Normal frames: `MAD ≈ 2-4` (arm moves through scene).
Board shift:    `MAD ≈ 20-60` (entire scene jumps).

## Metrics per Episode

| Metric | What it measures |
|---|---|
| `median_mad` | Baseline motion level (arm moving normally) |
| `max_mad` | Worst frame-to-frame change |
| `peak_ratio` | `max_mad / median_mad` — how many × worse than normal |
| `spike_count` | Number of frames where robust z-score > 3.5 |

A healthy episode has `peak_ratio < 4`. A board shift produces `peak_ratio > 6`
and often `>10`.

## Thresholds Used

- **Definite** (remove): `peak_ratio ≥ 10` and `spike_count ≥ 1`
- **Likely** (remove): `peak_ratio 6-10` and `spike_count ≥ 1`
- **Possible** (remove): `peak_ratio 4-7` and `spike_count ≥ 1`, OR `peak_ratio ≥ 7` without spikes

## Results

### cableholder-coarse-align
227 → **199 episodes** (removed 28: [1, 2, 18, 20, 21, 22, 37, 51, 53, 57, 64,
66, 89, 96, 112, 117, 118, 150, 155, 174, 176, 195, 199, 202, 217, 220, 223,
226])

### cableholder-fine-align
227 → **214 episodes** (removed 13: [19, 21, 22, 37, 55, 64, 96, 112, 119, 132,
147, 220, 223])

### cableholder-approach
227 → **210 episodes** (removed 17: [1, 15, 20, 21, 37, 51, 66, 96, 112, 117,
155, 176, 188, 202, 217, 223, 226])

### cableholder-insert
227 → **195 episodes** (removed 32: [1, 2, 18, 19, 21, 22, 34, 37, 40, 52, 61,
66, 87, 88, 90, 96, 99, 119, 132, 151, 155, 162, 174, 176, 185, 188, 195, 199,
202, 203, 220, 223])

### Merged (approach→coarse→fine→insert)
All 4 cleaned datasets → **180 episodes** present in ALL phases (intersection).

## Files

| File | Purpose |
|---|---|
| `scripts/detect_board_movement_v3.py` | Final detection script (robust z-score + peak ratio) |
| `verify_flagged.py` | Cross-check: first-vs-last frame diff for flagged episodes |
| `extract_frames.py` | Extract sample frames for visual inspection |

## Script

```bash
# Run detection on any cableholder dataset
python scripts/detect_board_movement.py \
    --dataset yihao-brain-bot/cableholder-coarse-align \
    --snapshot <hash>
```
