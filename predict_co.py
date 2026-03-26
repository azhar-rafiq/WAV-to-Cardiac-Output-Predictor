#!/usr/bin/env python3
"""
predict_co.py - WAV doppler -> CO prediction via trained model
usage: python predict_co.py recording.wav [--model-dir ./model_export] [-o out.csv]
"""

import argparse, os, pickle, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("pywt missing, skipping wavelet denoise. pip install PyWavelets")


# -- little progress helper --

_step = 0
def step(msg):
    global _step
    _step += 1
    t = time.time()
    print(f"  [{_step}] {msg}...", end=' ', flush=True)
    return t

def done(t0, extra=''):
    elapsed = time.time() - t0
    s = f"done ({elapsed:.2f}s)"
    if extra: s += f"  {extra}"
    print(s, flush=True)


# -- wav to velocity (same as Ultrasound10.py hilbert method) --

def reproduce_ui_export(data, fs):
    wall_hz, fmin, fmax = 25.0, 50.0, 8000.0
    smooth_ms = 20.0
    c, f_tx, angle_deg = 1540.0, 8_000_000.0, 30.0

    d = data.astype(np.float32).copy()
    d = d[:, [1, 0]]  # swap IQ

    ny = fs * 0.5
    if wall_hz > 0:
        b, a = signal.butter(2, wall_hz / ny, "highpass")
        d = signal.filtfilt(b, a, d, axis=0)
    if fmax > fmin:
        b, a = signal.butter(4, [fmin / ny, fmax / ny], "bandpass")
        d = signal.filtfilt(b, a, d, axis=0)

    z = d[:, 0].astype(np.float64) + 1j * d[:, 1].astype(np.float64)

    phase = np.unwrap(np.angle(z))
    dphi = np.diff(phase, prepend=phase[0])
    fd = ((fs / (2 * np.pi)) * dphi).astype(np.float32)

    w = int(max(3, round((smooth_ms / 1000.0) * fs)))
    if w % 2 == 0: w += 1
    fd = signal.savgol_filter(fd, w, 2, mode="interp").astype(np.float32)

    sin_a = float(np.sin(np.deg2rad(angle_deg)))
    v = (fd.astype(np.float64) * c) / (2.0 * f_tx * max(sin_a, 1e-6))

    t = np.arange(len(v), dtype=np.float32) / fs
    return t, v.astype(np.float32), z


# -- signal conditioning --

def wavelet_denoise(sig):
    if not HAS_PYWT: return sig
    coeff = pywt.wavedec(sig, 'db4', level=3)
    coeff[1:] = [pywt.threshold(c, np.std(c)*0.5) for c in coeff[1:]]
    out = pywt.waverec(coeff, 'db4')
    return out[:len(sig)]

def lowpass(x, fs, cutoff=15):
    b, a = butter(3, cutoff/(fs/2), btype='low')
    return filtfilt(b, a, x)

def condition_velocity(vel, fs):
    v = wavelet_denoise(vel.astype(np.float64))
    v = lowpass(v, fs, cutoff=15)
    return v.astype(np.float32)


# -- beat detection --

def detect_beats(vel_smooth, fs, min_bpm=40, max_bpm=200):
    neg = -vel_smooth
    min_dist = int(fs * 60 / max_bpm)

    ht = np.percentile(neg, 75)
    peaks, _ = find_peaks(neg, distance=min_dist, height=ht,
                          prominence=np.std(neg)*0.3)

    if len(peaks) < 2:
        peaks, _ = find_peaks(neg, distance=min_dist,
                              height=np.percentile(neg, 50))
    if len(peaks) < 2:
        print(f"WARNING: only {len(peaks)} beats found")
    return peaks


def extract_beat_features(vel_smooth, time_s, peaks, fs):
    n = len(vel_smooth)
    nb = len(peaks)

    starts = np.zeros(nb, dtype=int)
    ends = np.zeros(nb, dtype=int)
    for i in range(nb):
        starts[i] = 0 if i == 0 else (peaks[i-1] + peaks[i]) // 2
        ends[i] = n if i == nb-1 else (peaks[i] + peaks[i+1]) // 2

    bpv = np.zeros(nb)
    bdur = np.zeros(nb)
    bvti = np.zeros(nb)
    dt = 1.0 / fs

    for i in range(nb):
        seg = vel_smooth[starts[i]:ends[i]]
        bpv[i] = np.max(np.abs(seg))
        bdur[i] = (ends[i] - starts[i]) / fs
        bvti[i] = np.trapezoid(np.abs(seg), dx=dt)

    # broadcast to sample level
    s_pv = np.zeros(n, dtype=np.float32)
    s_dur = np.zeros(n, dtype=np.float32)
    s_vti = np.zeros(n, dtype=np.float32)
    s_hr = np.zeros(n, dtype=np.float32)

    for i in range(nb):
        sl = slice(starts[i], ends[i])
        s_pv[sl] = bpv[i]
        s_dur[sl] = bdur[i]
        s_vti[sl] = bvti[i]
        s_hr[sl] = 60.0 / max(bdur[i], 0.2)

    return pd.DataFrame({
        'time_s': time_s, 'velocity_smooth': vel_smooth,
        'vti': s_vti, 'peak_velocity': s_pv,
        'beat_duration_s': s_dur, 'heart_rate_bpm': s_hr,
    })


# -- feature engineering (has to match the notebook) --

def engineer_features(df):
    df = df.copy()
    df['vti_x_hr'] = df['vti'] * df['heart_rate_bpm']
    df['peak_vel_x_hr'] = df['peak_velocity'] * df['heart_rate_bpm']
    df['vel_ratio'] = df['velocity_smooth'] / (df['peak_velocity'] + 1e-8)
    df['time_frac'] = df['time_s'] / (df['beat_duration_s'] + 1e-8)
    df['hr_squared'] = df['heart_rate_bpm'] ** 2
    df['vti_squared'] = df['vti'] ** 2
    df['peak_vel_squared'] = df['peak_velocity'] ** 2
    return df


# -- model loading --

def load_model(model_dir):
    pkl_path = os.path.join(model_dir, 'co_bundle.pkl')

    # support both new (single bundle) and old (4 separate files)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            bundle = pickle.load(f)
        model = bundle['model']
        scaler = bundle['scaler']
        feat_cols = bundle['feature_cols']
        name = bundle.get('model_name', 'unknown')
    else:
        # fallback: legacy 4-file layout
        def _load(name):
            with open(os.path.join(model_dir, name), 'rb') as f:
                return pickle.load(f)
        model = _load('co_model.pkl')
        scaler = _load('scaler.pkl')
        feat_cols = _load('feature_cols.pkl')
        meta = _load('metadata.pkl')
        name = meta.get('model_name', 'unknown')

    print(f"  Model loaded: {name}  ({len(feat_cols)} features)")
    return model, scaler, feat_cols


# -- plotting --

def save_result_plot(time_arr, vel_raw, vel_smooth, co_pred, beat_peaks, fs, png_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [2, 2, 2]})
    axes[1].sharex(axes[0])  # top two share x, histogram is independent

    # velocity raw vs smooth
    axes[0].plot(time_arr, vel_raw, color='#aaaaaa', lw=0.3, label='raw')
    axes[0].plot(time_arr, vel_smooth, color='#2196F3', lw=0.6, label='denoised')
    beat_t = time_arr[beat_peaks]
    beat_v = vel_smooth[beat_peaks]
    axes[0].scatter(beat_t, beat_v, c='red', s=12, zorder=5, label=f'beats ({len(beat_peaks)})')
    axes[0].set_ylabel('velocity (m/s)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_title('Doppler Velocity + Beat Detection')

    # predicted CO
    axes[1].plot(time_arr, co_pred, color='#E91E63', lw=0.6)
    axes[1].axhline(np.mean(co_pred), ls='--', color='gray', lw=0.8,
                    label=f'mean={np.mean(co_pred):.2f} L/min')
    axes[1].set_ylabel('CO (L/min)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_title('Predicted Cardiac Output')

    # CO distribution
    axes[2].hist(co_pred, bins=80, color='#E91E63', alpha=0.7, edgecolor='white')
    axes[2].axvline(np.mean(co_pred), ls='--', color='black', lw=1)
    axes[2].set_xlabel('CO (L/min)')
    axes[2].set_ylabel('count')
    axes[2].set_title(f'CO Distribution  (mean={np.mean(co_pred):.2f}, std={np.std(co_pred):.2f})')

    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# -- main pipeline --

def process_wav(wav_path, model_dir, output_path=None):
    global _step
    _step = 0

    if output_path is None:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        output_path = f"{base}_co_prediction.csv"
    png_path = output_path.replace('.csv', '.png')

    total_t0 = time.time()
    print(f"\n{'='*55}")
    print(f"  CO Prediction Pipeline")
    print(f"  Input: {os.path.basename(wav_path)}")
    print(f"{'='*55}\n")

    # load model
    t0 = step("Loading model")
    model, scaler, feat_cols = load_model(model_dir)
    done(t0)

    # read wav
    t0 = step("Reading WAV")
    fs, data = wavfile.read(wav_path)
    dur = len(data) / fs
    if data.ndim == 1:
        print("\n  ERROR: need stereo IQ wav"); sys.exit(1)
    done(t0, f"{fs}Hz, {dur:.1f}s, stereo")

    # velocity extraction
    t0 = step("Extracting velocity (Hilbert)")
    time_arr, vel_raw, _ = reproduce_ui_export(data, fs)
    done(t0, f"{len(vel_raw):,} samples")

    # denoise
    t0 = step("Denoising (wavelet + lowpass)")
    vel_smooth = condition_velocity(vel_raw, fs)
    done(t0)

    # beats
    t0 = step("Detecting beats")
    beat_peaks = detect_beats(vel_smooth, fs)
    avg_hr = len(beat_peaks) / dur * 60
    done(t0, f"{len(beat_peaks)} beats, ~{avg_hr:.0f} bpm")
    if len(beat_peaks) < 2:
        print("  ERROR: not enough beats"); sys.exit(1)

    # features
    t0 = step("Engineering features")
    df = extract_beat_features(vel_smooth, time_arr, beat_peaks, fs)
    df_feat = engineer_features(df)
    missing = [c for c in feat_cols if c not in df_feat.columns]
    if missing:
        print(f"\n  ERROR: missing cols {missing}"); sys.exit(1)
    done(t0, f"{len(feat_cols)} features")

    # predict
    t0 = step("Predicting CO")
    X = df_feat[feat_cols].values
    if scaler is not None: X = scaler.transform(X)
    co_pred = model.predict(X)
    done(t0, f"mean={co_pred.mean():.2f} +/- {co_pred.std():.2f} L/min")

    # save csv
    t0 = step("Saving CSV")
    out = pd.DataFrame({
        'time_s': time_arr,
        'velocity_raw_m_per_s': vel_raw,
        'velocity_smooth_m_per_s': vel_smooth,
        'predicted_CO_L_per_min': co_pred,
        'peak_velocity': df['peak_velocity'].values,
        'vti': df['vti'].values,
        'beat_duration_s': df['beat_duration_s'].values,
        'heart_rate_bpm': df['heart_rate_bpm'].values,
    })
    out.to_csv(output_path, index=False, float_format='%.6f')
    done(t0, f"{len(out):,} rows -> {output_path}")

    # save plot
    t0 = step("Generating plot")
    save_result_plot(time_arr, vel_raw, vel_smooth, co_pred, beat_peaks, fs, png_path)
    done(t0, png_path)

    total = time.time() - total_t0
    print(f"\n{'='*55}")
    print(f"  All done in {total:.2f}s")
    print(f"  CO: {co_pred.mean():.2f} L/min (mean)")
    print(f"{'='*55}\n")
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='WAV doppler -> CO prediction')
    p.add_argument('wav_file')
    p.add_argument('--model-dir', default='model_export')
    p.add_argument('-o', '--output', default=None)
    args = p.parse_args()

    if not os.path.isfile(args.wav_file):
        print(f"not found: {args.wav_file}"); sys.exit(1)
    if not os.path.isdir(args.model_dir):
        print(f"no model dir: {args.model_dir}, run notebook first"); sys.exit(1)

    process_wav(args.wav_file, args.model_dir, args.output)