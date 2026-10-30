# Crowd Detection – Abnormal Crowd Detection

Analyze camera footage and images to **count people, generate density heatmaps, track pedestrian flow, and detect abnormal gatherings**.

The core system uses FoundationVision/ByteTrack (YOLOX-X, MOT20) repository combined with a crowd density analysis module. A web application allows team members to upload videos and view processing results.

<!-- TODO: Add 2–3 screenshots from ACTUAL results to docs/, then uncomment:
![Heatmap on white background](docs/heatmap_white.png)
![Pedestrian flow](docs/flow.png)
![Web application](docs/web.png)
-->

## Features

- **Person Detection and Tracking:** Detect and track people using ByteTrack with a MOT20-trained model. The bottom-center point of each person's bounding box is used as their ground-contact position.
- **People Counting:** Count the number of people in each frame, including the number of moving individuals, their speed, and the dominant movement direction.
- **Density Heatmaps:** Generate crowd density heatmaps overlaid on the original video or displayed on a **white background**.
- **Pedestrian Flow Visualization:** Visualize movement direction and speed using regional velocity arrows, trajectory trails, and direction-based colors.
- **Abnormal Gathering Detection:** Detect potential crowd gatherings and record events, including timestamps, locations, people counts, and severity levels.
- **Web Application:** Upload videos or images, queue processing jobs, view results, and access the application using shared password authentication.

## Project Structure

```text
pipeline/
├─ ByteTrack/            # ByteTrack/YOLOX source code
├─ models/               # Model checkpoints (.pth, excluded from Git)
├─ pipeline/             # Main processing package
│  ├─ run.py             # Entry point: video input → detection → tracking → analysis → output
│  ├─ detector.py        # YOLOX initialization, preprocessing, and postprocessing
│  ├─ analyzer.py        # Density heatmaps and abnormal gathering detection
│  ├─ extras.py          # White-background heatmaps, pedestrian flow, and people statistics
│  ├─ video_io.py        # Video reading and H.264 encoding via FFmpeg
│  ├─ status.py          # Processing progress tracking (status.json)
│  └─ config.py          # Pipeline configuration
└─ webapp/
   ├─ main.py            # FastAPI: uploads, job queue, and result delivery
   └─ static/
      └─ index.html      # Web interface
```

## Installation

**Requirements:**
- Windows or Linux
- Python `3.13`
- NVIDIA GPU with CUDA support (recommended; CPU execution is supported but significantly slower)
- FFmpeg

```powershell
cd E:\Crowd_Detection
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install imageio-ffmpeg          # Alternatively, install FFmpeg and add it to PATH
```

**Model Checkpoint:**

Place `bytetrack_x_mot20.pth` in the `pipeline/models/` directory.

The model weights are available from the original FoundationVision/ByteTrack repository.

## Quick Start (Command Line)

Run the following command from the `pipeline/` directory (which contains the `pipeline/` Python package):

```powershell
cd pipeline
python -m pipeline.run --input path\to\video.mp4 --out out\demo --skip 2
```

| Parameter | Description |
|---|---|
| `--input`, `--out` | Input video path and output directory (required). |
| `--skip N` | Process one out of every N frames. The output video maintains the original playback duration by adjusting its FPS to `original_fps / N`. |
| `--device gpu\|cpu` | Device used for running the detector. |
| `--test-size H W` | Input resolution for the detector, e.g., `736 1312`. Smaller resolutions improve speed but may reduce detection accuracy for small or distant people. |
| `--conf`, `--ckpt` | Detection confidence threshold and model checkpoint path. |

## Web Application

Start the web server using:

```powershell
cd pipeline
$env:CROWD_USER = "team"
$env:CROWD_PASS = "<password>"        # Required when exposing the application externally
uvicorn webapp.main:app --host 0.0.0.0 --port 8000
```

Open the application in a browser:

`http://<server-address>:8000`

Log in and upload a video (up to 500 MB) or an image (up to 30 MB).

Each uploaded job receives a unique URL:

`http://<server-address>:8000/#job=<id>`

Processing jobs are handled sequentially using a single worker. Results older than 24 hours are automatically deleted.

### Environment Variables

| Variable | Description |
|---|---|
| `CROWD_USER`,`CROWD_PASS` | Login credentials (HTTP Basic Authentication). Leaving `CROWD_PASS` empty disables authentication. |
| `CROWD_REENCODE=1` | Re-encode output videos using FFmpeg. Disabled by default because the pipeline already produces H.264 videos. |

**Security Notice:**

HTTP Basic Authentication over plain HTTP does not encrypt credentials during transmission.
When exposing the application to the Internet, use a VPN such as Tailscale or configure HTTPS through a reverse proxy or Cloudflare Tunnel.

**Never commit passwords or other sensitive credentials to the repository.**

## How It Works

This is current system workflow:

![Workflow_Image](./system_des/human_det_thread/crowd_gathering_flow.drawio.png)

### 1. Person Detection and Tracking

The system uses **YOLOX-X + ByteTrack** to detect and track people across video frames.

Each tracked person is represented by:
- `tlwh`: Bounding box coordinates (top-left x, top-left y, width, height).
- `track_id`: Unique tracking identifier.

Bounding boxes are filtered using `min_box_area` and `aspect_ratio_thresh`, configured in `config.py`.

### 2. Crowd Density Estimation

Each detected person is represented by a ground-contact point (`f_i`), corresponding to the bottom-center of their bounding box.

The crowd density at position (`p`) is calculated using Gaussian kernels:

[`L(p) = Σ exp(-|p - f_i|² / 2σ²)`]

where:

- (`L(p)`): Estimated crowd density at position (`p`).
- (`f_i`): Ground-contact point of person (`i`).
- (`σ`): Gaussian kernel bandwidth, set to `0.04 × frame width`.

Temporal smoothing is applied using an **Exponential Moving Average (EMA)** to reduce frame-to-frame fluctuations.

### 3. Abnormal Gathering Detection

The system identifies high-density regions where:
[`L ≥ 4.0`]

These regions are tracked over consecutive frames.
A gathering event is confirmed when all the following conditions are satisfied:
- **People count:** At least 5 people are present in the region.
- **Average dwell time:** At least 3 seconds.
- **Spatial stability:** The region center moves no more than 15% of the frame width, helping distinguish stationary gatherings from groups walking together.

Detection thresholds are defined in `DEFAULTS` within `analyzer.py`.

Each confirmed gathering is recorded with its timestamp, location, people count, density, dwell time, and severity level.

### 4. Pedestrian Flow Analysis
The system estimates each person's velocity based on their displacement over a **1-second time window**.

This approach helps reduce noise caused by bounding-box jitter.

Individual velocities are aggregated into a grid-based velocity field to visualize:
- Dominant movement directions.
- Movement speed.
- Pedestrian trajectories.
- Regional flow patterns.

Flow analysis parameters are defined in `FLOW_DEFAULTS` within `extras.py`.

## Output Files

All generated results are saved in the directory specified by `--out`.

| File | Description |
|---|---|
| `output_bbox.mp4` | Video with detected bounding boxes and tracking IDs. |
| `output_heatmap.mp4` | Crowd density heatmap overlaid on the original video. |
| `output_heatmap_white.mp4` | Crowd density heatmap on a white background, with dots representing ground-contact positions. |
| `output_flow.mp4` | Pedestrian flow visualization on a white background. |
| `crowd_stats.json` | Frame-level statistics: `count`, `moving`, `speed` (px/s), and `dir`, along with an overall `summary`. |
| `gathering_log.json` | Detected gathering events, including timestamp, `center_xy`, `region_bbox`, `people_count`, `peak_density`, `avg_dwell_s`, `track_ids`, and `severity`. |
| `meta.json` | Processing duration, FPS, stage-level execution times, and GPU information. |
| `status.json` | Job status and processing progress (0–1). |

## Performance

An example benchmark was conducted with the following configuration:

| Specification | Value |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop GPU |
| Input resolution | 848 × 480 |
| Video duration | 20 seconds |
| Detector test size | 896 × 1600 |
| Frame skipping | `--skip 1` |
| Processing throughput | Approximately **1.7 FPS** |
| Detection latency | Approximately **520 ms/frame** |
| Detection time share | Approximately **88%** of total processing time |

The detection stage accounts for most of the computational cost.

To improve processing speed:

- Increase `--skip` to process fewer frames.
- Reduce `--test-size` to lower detector computation requirements.

**Trade-off:** These optimizations may reduce detection accuracy, particularly for small or distant people, and frame skipping may affect tracking and temporal analysis.

## Limitations

- **Counting accuracy:** The reported people count represents the number of displayed tracks after filtering, rather than an exact ground-truth count. Highly crowd scenes may lead to undercounting.
- **Static image limitations:** Images contain only one frame; therefore, pedestrian flow cannot be estimated, and time-dependent gathering detection generally produces no events.
- **Camera motion:** Moving cameras can distort pedestrian flow estimation because velocity is calculated in image coordinates rather than world coordinates.
- **Processing concurrency:** The web application uses a single worker to process jobs sequentially.
- **Job persistence:** Job status is stored on disk and survives application restarts. However, jobs interrupted during execution are marked as failed.

## Acknowledgments

This project builds upon the following open-source frameworks and research:

- **[ByteTrack](https://github.com/FoundationVision/ByteTrack)** — Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022. Used for multi-object tracking with the MOT20 model and configuration.
- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** — Object detection framework developed by Megvii, used as the detection backbone.
- **[FastAPI](https://github.com/fastapi/fastapi)** — Python web framework used to implement the web application and job management interface.

## License and Third-Party Software

- This project incorporates components from [ByteTrack](https://github.com/FoundationVision/ByteTrack), developed by Yifu Zhang and contributors.
- ByteTrack is distributed under the **MIT License**. The original copyright notices and license terms are included in `pipeline/ByteTrack/LICENSE`.
- The crowd density estimation, abnormal gathering detection, pedestrian flow analysis, and web application are developed as part of this project.
- The original components of Crowd Detection have not yet been assigned an open-source license. All third-party components remain subject to their respective licenses.

### Academic Citation

If you use ByteTrack in academic research, please cite the original publication:
Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. European Conference on Computer Vision (ECCV).