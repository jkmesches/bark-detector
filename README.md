# bark-detector

Outdoor acoustic event detector that runs continuously on a Linux box, records
suspected barks, classifies them with a small CNN, and publishes events to MQTT
and InfluxDB. A companion report pipeline turns the event stream into
human-readable summaries and figures.

**Out-of-fold accuracy: 95.6%** on 702 hand-labeled clips (5-fold stratified
CV). Full methodology and results are in
[`report/paper.pdf`](report/paper.pdf).

## Pipeline at a glance

```
mic ─► bandpass + RMS ─► event WAV ─► MQTT/InfluxDB
                              │
                              └─► BarkCNN (log-mel, 60k params) ─► bark?
                                       │
                                       └─► onset counter ─► est. barks/event
                                                   │
                                                   └─► daily CSV + figures
```

1. `bark_detector.py` — real-time capture service. Reads `config.toml`, opens
   the input device via PortAudio, runs a Butterworth bandpass, and emits an
   event whenever bandpassed RMS crosses `threshold_dbfs`. Each event is
   written to disk as a WAV and published to MQTT (and optionally InfluxDB).
2. `label_barks.py` — small Tk GUI for YES/NO labeling of recorded clips.
   Stores labels in an .xlsx workbook and supports keyboard-driven review.
3. `train_bark_cnn.py` — trains a compact CNN on log-mel spectrograms with
   5-fold stratified CV and writes `bark_cnn.pt` + `bark_cnn_meta.json`.
4. `tune_accuracy.py` — retrains the 5 folds, saves each as an ensemble
   member, adds test-time augmentation, and sweeps the decision threshold.
5. `bark_report.py` — scores every recording, counts onsets per clip, and
   writes `bark_report.csv` plus summary figures.
6. `report_figures.py` — rebuilds publication-quality figures from
   `bark_report.csv` alone.

## Repo contents

| File | Purpose |
| --- | --- |
| `bark_detector.py` | Continuous capture + MQTT/InfluxDB publisher |
| `label_barks.py` | Tk-based labeling GUI |
| `train_bark_cnn.py` | 5-fold CV training |
| `tune_accuracy.py` | Fold ensembling + TTA + threshold sweep |
| `bark_report.py` | Offline scoring + daily rollup |
| `report_figures.py` | Standalone figure renderer |
| `config.example.toml` | Template config — copy to `config.toml` and edit |
| `bark-detector.service` | systemd user unit template |
| `bark_cnn.pt` / `bark_cnn_meta.json` | Trained model + audio/mel hparams |
| `bark_cnn_threshold.json` | Tuned decision threshold |
| `confusion_matrix.png`, `training_report.txt` | Baseline training artifacts |
| `bark_report.csv` | Example daily rollup (hour x date) |
| `grafana-dashboard.json`, `ha-dashboard.yaml` | Optional dashboards |
| `report/paper.pdf` | Full write-up |

## Setup

```bash
git clone git@github.com:jkmesches/bark-detector.git
cd bark-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp config.example.toml config.toml
# edit config.toml: device_match, MQTT broker/creds, optional InfluxDB
```

### Run as a user service

```bash
mkdir -p ~/.config/systemd/user
cp bark-detector.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now bark-detector
journalctl --user -u bark-detector -f
```

The unit template assumes the repo lives at `~/bark-detector` with a venv at
`~/bark-detector/venv`. Adjust `WorkingDirectory` / `ExecStart` if yours lives
elsewhere.

## Retraining on your own clips

1. Run the detector for a few days with `recording.include_pre_roll = true`.
2. `python3 label_barks.py` to tag each clip YES/NO.
3. `python3 train_bark_cnn.py` for a baseline single model.
4. `python3 tune_accuracy.py` for the fold ensemble, TTA, and tuned threshold.
5. `python3 bark_report.py` to score everything in `recordings/` and emit
   `bark_report.csv`.

## Hardware

Any USB or onboard mic works, but some level of analog gain helps — the
reference setup uses an ALC897 codec with `rec_level=49` and `mic_boost=1`
(20 dB) plus +19 dB digital gain. See `config.example.toml` for the full set.

## License

No license declared. Treat as all rights reserved unless you hear otherwise.
