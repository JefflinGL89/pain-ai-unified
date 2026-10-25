"""
Plane-Orthogonal-to-Skin (POS) Remote Photoplethysmography (rPPG) Processor
Extracts blood volume pulse (BVP), Heart Rate (BPM), Respiration Rate (BrPM),
Signal Quality Index (SQI), and Pulse Rate Variability (PRV/HRV) from skin RGB traces.
"""

from collections import deque
import numpy as np
from scipy import signal


class POSrPPGProcessor:
    def __init__(self, target_fs=30.0, window_sec=10.0, min_window_sec=4.0):
        """
        :param target_fs: Uniform sampling rate in Hz (standard webcam target).
        :param window_sec: Analysis buffer duration in seconds (recommended 10-15s for high frequency resolution).
        :param min_window_sec: Minimum seconds required before producing an initial estimation.
        """
        self.target_fs = float(target_fs)
        self.window_sec = float(window_sec)
        self.min_window_sec = float(min_window_sec)
        self.max_samples = int(self.window_sec * self.target_fs * 1.5)

        self.rgb_buffer = deque(maxlen=self.max_samples)
        self.time_buffer = deque(maxlen=self.max_samples)
        self.clean_pulse_history = deque(maxlen=int(target_fs * 4))  # For live display

        # Frequency bands (Hz)
        self.hr_low = 0.75    # 45 BPM
        self.hr_high = 2.80   # 168 BPM
        self.resp_low = 0.15  # 9 BrPM
        self.resp_high = 0.50 # 30 BrPM

        # Exponential smoothing on output BPM
        self.smoothed_bpm = None
        self.BPM_EMA_ALPHA = 0.25

    def add_sample(self, rgb_mean, timestamp):
        """
        Adds a single [R, G, B] frame sample with its high-resolution timestamp.
        """
        if rgb_mean is None or len(rgb_mean) < 3:
            return
        self.rgb_buffer.append(list(rgb_mean))
        self.time_buffer.append(float(timestamp))

    def reset(self):
        self.rgb_buffer.clear( )
        self.time_buffer.clear()
        self.clean_pulse_history.clear()
        self.smoothed_bpm = None

    def compute_metrics(self):
        """
        Computes Heart Rate (BPM), Respiration Rate (BrPM), Signal Quality Index (SQI),
        PRV/HRV (SDNN in ms), and the latest pulse wave value for visualization.
        """
        if len(self.time_buffer) < 2:
            return None

        duration = self.time_buffer[-1] - self.time_buffer[0]
        if duration < self.min_window_sec:
            progress = min(0.99, duration / self.min_window_sec)
            return {
                "bpm": None,
                "resp_rate": None,
                "sqi": 0.0,
                "hrv_sdnn": None,
                "wave_val": 0.0,
                "is_stable": False,
                "calibrating_progress": progress
            }

        # 1. Resample to strictly uniform timeline (anti-jitter)
        t_raw = np.array(self.time_buffer)
        rgb_raw = np.array(self.rgb_buffer)

        # Slice to current window_sec
        if duration > self.window_sec:
            mask = t_raw >= (t_raw[-1] - self.window_sec)
            t_raw = t_raw[mask]
            rgb_raw = rgb_raw[mask]
            duration = t_raw[-1] - t_raw[0]

        num_samples = int(duration * self.target_fs)
        if num_samples < int(self.min_window_sec * self.target_fs):
            return None

        t_uniform = np.linspace(t_raw[0], t_raw[-1], num_samples)
        traces = np.zeros((num_samples, 3))
        for c in range(3):
            traces[:, c] = np.interp(t_uniform, t_raw, rgb_raw[:, c])

        # 2. POS Projection (Wang et al., IEEE TBME 2017)
        # Normalize by temporal mean
        mean_rgb = np.mean(traces, axis=0) + 1e-6
        norm_rgb = traces / mean_rgb

        # Orthogonal chrominance planes
        # X = G - B
        # Y = G + B - 2R
        X = norm_rgb[:, 1] - norm_rgb[:, 2]
        Y = norm_rgb[:, 1] + norm_rgb[:, 2] - 2.0 * norm_rgb[:, 0]

        std_y = np.std(Y) + 1e-6
        alpha = np.std(X) / std_y
        H = X + alpha * Y

        # 3. Butterworth Zero-Phase Bandpass Filter (Cardiac band)
        nyq = 0.5 * self.target_fs
        b_filt, a_filt = signal.butter(3, [self.hr_low / nyq, self.hr_high / nyq], btype="bandpass")
        clean_pulse = signal.filtfilt(b_filt, a_filt, H)

        # 4. Zero-Padded FFT & Parabolic Peak Interpolation for High-Precision BPM
        n_fft = max(2048, int(2 ** np.ceil(np.log2(num_samples))))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.target_fs)
        fft_mag = np.abs(np.fft.rfft(clean_pulse, n=n_fft))

        valid_idx = np.where((freqs >= self.hr_low) & (freqs <= self.hr_high))[0]
        if len(valid_idx) == 0:
            return None

        band_mags = fft_mag[valid_idx]
        peak_idx_local = np.argmax(band_mags)
        peak_idx_global = valid_idx[peak_idx_local]

        # Parabolic interpolation on the peak bin for sub-bin frequency accuracy
        best_freq = freqs[peak_idx_global]
        if 0 < peak_idx_global < len(fft_mag) - 1:
            alpha_p = fft_mag[peak_idx_global - 1]
            beta_p = fft_mag[peak_idx_global]
            gamma_p = fft_mag[peak_idx_global + 1]
            denom = alpha_p - 2.0 * beta_p + gamma_p
            if abs(denom) > 1e-6:
                delta = 0.5 * (alpha_p - gamma_p) / denom
                best_freq += delta * (freqs[1] - freqs[0])

        raw_bpm = float(np.clip(best_freq * 60.0, 45.0, 165.0))

        # Exponential smoothing on BPM
        if self.smoothed_bpm is None:
            self.smoothed_bpm = raw_bpm
        else:
            self.smoothed_bpm = (self.BPM_EMA_ALPHA * raw_bpm) + ((1.0 - self.BPM_EMA_ALPHA) * self.smoothed_bpm)

        # 5. Signal Quality Index (SQI)
        # Ratio of spectral power around peak (+- 0.15 Hz) to entire cardiac band
        peak_mask = (freqs >= (best_freq - 0.15)) & (freqs <= (best_freq + 0.15))
        peak_power = np.sum(fft_mag[peak_mask] ** 2)
        total_band_power = np.sum(band_mags ** 2) + 1e-6
        sqi = float(np.clip(peak_power / total_band_power, 0.0, 1.0))

        # 6. Respiration Rate (Respiratory Sinus Arrhythmia / Baseline Wander)
        b_resp, a_resp = signal.butter(2, [self.resp_low / nyq, self.resp_high / nyq], btype="bandpass")
        resp_signal = signal.filtfilt(b_resp, a_resp, H)
        resp_fft = np.abs(np.fft.rfft(resp_signal, n=n_fft))
        resp_valid = np.where((freqs >= self.resp_low) & (freqs <= self.resp_high))[0]
        resp_rate = None
        if len(resp_valid) > 0:
            best_resp_freq = freqs[resp_valid[np.argmax(resp_fft[resp_valid])]]
            resp_rate = float(np.clip(best_resp_freq * 60.0, 9.0, 30.0))

        # 7. Inter-Beat Interval (IBI) & Pulse Rate Variability (PRV / HRV)
        min_peak_dist = int(self.target_fs * (60.0 / 165.0))  # refractory period
        peaks, _ = signal.find_peaks(clean_pulse, distance=min_peak_dist, prominence=np.std(clean_pulse) * 0.4)

        hrv_sdnn = None
        if len(peaks) >= 5 and duration >= 8.0:
            ibi_ms = np.diff(peaks) / self.target_fs * 1000.0
            valid_ibi = ibi_ms[(ibi_ms >= 360.0) & (ibi_ms <= 1333.0)]  # 45 to 165 BPM bounds
            if len(valid_ibi) >= 4:
                hrv_sdnn = float(np.clip(np.std(valid_ibi), 10.0, 130.0))

        # 8. Normalized latest waveform value for oscilloscope display
        recent_std = np.std(clean_pulse[-int(self.target_fs * 2):]) + 1e-6
        wave_val = float(np.clip(clean_pulse[-1] / (2.5 * recent_std), -1.0, 1.0))

        return {
            "bpm": float(self.smoothed_bpm),
            "raw_bpm": raw_bpm,
            "resp_rate": resp_rate,
            "sqi": sqi,
            "hrv_sdnn": hrv_sdnn,
            "wave_val": wave_val,
            "is_stable": bool(duration >= 8.0 and sqi >= 0.28),
            "calibrating_progress": 1.0
        }
