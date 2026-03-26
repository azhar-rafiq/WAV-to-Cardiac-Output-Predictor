# predict_co.py — Real-Time Cardiac Output Prediction from Doppler WAV

> Predict cardiac output (L/min) from a raw stereo IQ WAV file recorded by a pulmonary artery Doppler probe — no manual annotation, no post-processing, one command.

⚠️ .pkl model files are not uploaded because their size (>4GB) exceeded Github limit.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat-square&logo=lightgbm&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)
![PyWavelets](https://img.shields.io/badge/PyWavelets-5A5A5A?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

---

## What it does

`predict_co.py` is a command-line inference pipeline that takes a raw ultrasound recording and outputs a per-sample cardiac output (CO) prediction in L/min.

```
recording.wav  ──►  velocity extraction  ──►  beat detection  ──►  ML model  ──►  co_prediction.csv
                         (Hilbert)            (peak finding)      (trained)         + plot.png
```

Under the hood it reconstructs the Doppler velocity signal from the raw IQ (in-phase / quadrature) stereo WAV using the same Hilbert-based demodulation as the original acquisition UI, segments the signal into individual heartbeats, engineers physiological features per beat, and runs a trained regression model to produce continuous CO estimates at the original sampling rate.

---

## Clinical background

**Cardiac Output** is the volume of blood the heart pumps per minute (L/min). Normal resting range is 4–8 L/min. It is one of the most critical haemodynamic variables in intensive care, cardiac surgery, and anaesthesia.

**Pulmonary artery Doppler** provides direct access to right ventricular outflow — 100% of cardiac output passes through the pulmonary artery before reaching the lungs. Velocity measured here, integrated over a cardiac cycle, yields the **Velocity-Time Integral (VTI)**, which is proportional to stroke volume:

```
Stroke Volume  =  VTI  ×  Cross-sectional area of PA
Cardiac Output =  Stroke Volume  ×  Heart Rate
```

This pipeline automates that calculation from raw WAV recordings, enabling continuous non-invasive CO monitoring without requiring manual VTI tracing.

---

## Pipeline steps

| Step | What happens |
|---|---|
| WAV loading | Reads stereo IQ WAV via `scipy.io.wavfile` |
| Velocity extraction | Hilbert demodulation → instantaneous frequency → Doppler shift → velocity (m/s) |
| Signal conditioning | Wavelet denoising (PyWavelets db4, level 3) + 15 Hz lowpass Butterworth filter |
| Beat detection | Negative-peak detection on inverted velocity (`scipy.signal.find_peaks`) |
| Feature engineering | Per-beat: VTI, peak velocity, beat duration, HR + derived interaction features |
| Prediction | Scaled features → trained model → CO (L/min) at each sample |
| Export | CSV with raw + smooth velocity, CO prediction, beat features + PNG summary plot |

---

## Quickstart

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib PyWavelets

# Run prediction
python predict_co.py recording.wav --model-dir ./model_export -o results.csv
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `wav_file` | required | Path to stereo IQ WAV file |
| `--model-dir` | `./model_export` | Directory containing trained model bundle |
| `-o / --output` | `<wav_name>_co_prediction.csv` | Output CSV path |

---

## Input format

The WAV file must be:
- **Stereo** (2 channels) — channel 0 = Q (quadrature), channel 1 = I (in-phase)
- **Any sample rate** — pipeline auto-detects and adapts
- Recorded from a **pulmonary artery Doppler probe** with the signal acquisition settings:
  - Wall filter: 25 Hz highpass
  - Bandpass: 50–8000 Hz
  - Probe frequency: 8 MHz
  - Insonation angle: 30°

---

## Output files

### CSV (`*_co_prediction.csv`)

| Column | Description |
|---|---|
| `time_s` | Time from recording start (seconds) |
| `velocity_raw_m_per_s` | Raw Doppler velocity before denoising |
| `velocity_smooth_m_per_s` | Denoised velocity used for prediction |
| `predicted_CO_L_per_min` | Predicted cardiac output |
| `peak_velocity` | Max absolute velocity in current beat window |
| `vti` | Velocity-time integral for current beat (m) |
| `beat_duration_s` | Duration of current beat (RR interval, seconds) |
| `heart_rate_bpm` | Instantaneous heart rate (60 / beat_duration) |

### PNG (`*_co_prediction.png`)

Three-panel summary figure:
1. Raw vs denoised velocity with detected beat markers
2. Predicted CO over time with mean line
3. CO distribution histogram

---

## Model bundle

The model is loaded from `model_export/co_bundle.pkl` — a single pickle containing:

```python
{
  'model':        trained_sklearn_estimator,
  'scaler':       fitted_StandardScaler,        # None if no scaling
  'feature_cols': ['vti', 'peak_velocity', ...], # ordered feature list
  'model_name':   'LightGBM'                    # or RF, Ridge, XGBoost etc.
}
```

A legacy four-file layout (`co_model.pkl`, `scaler.pkl`, `feature_cols.pkl`, `metadata.pkl`) is also supported for backward compatibility.

The model was trained on PA Doppler recordings with ground-truth CO labels derived from concurrent LV pressure measurements (EDV, ESV, HR).

---

## Engineered features

| Feature | Formula | Clinical meaning |
|---|---|---|
| `vti` | ∫\|velocity\| dt per beat | Stroke volume proxy |
| `peak_velocity` | max\|velocity\| in beat | Peak flow rate |
| `beat_duration_s` | RR interval | Inverse of HR |
| `heart_rate_bpm` | 60 / beat_duration | Beats per minute |
| `vti_x_hr` | vti × HR | Approximate CO (without CSA) |
| `peak_vel_x_hr` | peak_velocity × HR | Flow × rate interaction |
| `vel_ratio` | velocity / peak_velocity | Normalised waveform position |
| `time_frac` | time_s / beat_duration | Fractional position within beat |
| `hr_squared` | HR² | Non-linear HR term |
| `vti_squared` | vti² | Non-linear VTI term |
| `peak_vel_squared` | peak_velocity² | Non-linear peak velocity term |

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | DataFrame handling and CSV export |
| `scipy` | WAV I/O, signal filtering, peak detection |
| `matplotlib` | Summary plot generation |
| `seaborn` | Statistical visualisation (training notebooks) |
| `scikit-learn` | Regression models: Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, ExtraTrees, AdaBoost, SVR, KNN, DecisionTree + preprocessing + metrics |
| `xgboost` | XGBoost regressor (optional — skipped gracefully if absent) |
| `lightgbm` | LightGBM regressor (optional — skipped gracefully if absent) |
| `tensorflow` / `keras` | GRU-based deep learning pipeline (`train_gru.py`) |
| `PyWavelets` | Wavelet denoising in inference pipeline (optional — skipped gracefully if absent) |

No deep learning framework required for inference — the model bundle uses scikit-learn compatible estimators only.

---

## Example output

```
=======================================================
  CO Prediction Pipeline
  Input: patient_001.wav
=======================================================

  [1] Loading model...           done (0.12s)  Model loaded: LightGBM  (11 features)
  [2] Reading WAV...             done (0.34s)  20000Hz, 387.2s, stereo
  [3] Extracting velocity...     done (1.21s)  7,744,000 samples
  [4] Denoising...               done (2.87s)
  [5] Detecting beats...         done (0.43s)  512 beats, ~79 bpm
  [6] Engineering features...    done (0.18s)  11 features
  [7] Predicting CO...           done (0.09s)  mean=5.14 +/- 0.38 L/min
  [8] Saving CSV...              done (3.12s)  7,744,000 rows -> patient_001_co_prediction.csv
  [9] Generating plot...         done (1.44s)  patient_001_co_prediction.png

=======================================================
  All done in 9.80s
  CO: 5.14 L/min (mean)
=======================================================
```

---

## Limitations

- Trained on single-patient data — generalisation to new patients requires retraining or fine-tuning
- CO targets during training were derived from LV pressure measurements (EDV − ESV × HR), which measure left heart output; the PA Doppler measures right heart output — these agree under normal conditions but may diverge during haemodynamic events
- Beat detection may fail on arrhythmic recordings (AF, frequent ectopics) — `WARNING: only N beats found` will be printed
- The `vel_ratio` and `time_frac` features assume stable beat morphology; heavily variable waveforms reduce prediction accuracy

---

## Repository structure

```
.
├── predict_co.py          # inference pipeline (this file)
├── model_export/
│   └── co_bundle.pkl      # trained model + scaler + feature list
├── train_gru.py           # GRU-based training script (optional deep learning path)
├── train6.py              # sklearn training pipeline
├── extract_hr.py          # HR + beat feature extraction from 0.5ms CSV
├── extract_beats.py       # beat-level dataset builder
├── smooth_co.py           # cubic spline CO smoothing
└── README.md
```

---
