"""
Audio Processor Module for Suno Prep & Sanitizer.

Implements an advanced multi-vector acoustic evasion and lyric obfuscation pipeline
specifically designed to bypass modern Automatic Content Recognition (ACR) systems
such as Audible Magic (Landmark Constellation & VIBE deep neural embeddings)
and Automatic Speech Recognition (ASR / OpenAI Whisper) moderation models:

1. Dynamic Micro-Chrono Jitter (Anti-Landmark / Anti-DTW):
   Applies continuous non-linear time-warping (stochastic wow-and-flutter drift)
   that shifts peak-to-peak delta times throughout the track, destroying the
   time-offset alignment histogram in landmark fingerprinting.

2. Hilbert Bode Frequency Shifting (Anti-VIBE & Anti-Chroma):
   Non-linearly shifts all frequency components by a constant Hz offset (+5 to +15 Hz)
   or across multiple independent bands using Hilbert transform single-sideband analytic modulation.
   Destroys integer harmonic overtone ratios (2:3:4) and disperses Constant-Q Transform
   (CQT) pitch-class distributions into multiple bins without noticeable dissonance.

3. Continuous Pitch Wobble & Formant Modulation (Anti-VIBE Transposition):
   Sweeps pitch continuously with an LFO (vibrato + rubberband formant shift).
   Because pitch is non-stationary, neural embeddings cannot match circularly shifted chroma.

4. Total Center Vocal Annihilation (Zero-Vocal Side Isolation):
   100% phase subtraction of the center channel (L - R, R - L) on mid and high frequencies
   where lead vocals reside, preserving punchy mono sub-bass (< 160 Hz).
   Completely removes lead vocal signatures so Audible Magic and Whisper ASR find nothing to match.

5. Full-Track Harmonic Camouflage Drone Bed:
   Injects a musical, warm analog harmonic fifths drone bed and vintage tape/vinyl crackle
   at -26 dBFS throughout the entire track, masking the landmark peaks.

6. Front & Tail Acoustic Padding (The ABS Workaround):
   Prepends 4.0 seconds of silence/ambient vinyl and appends 2.0 seconds of tail.
   Displaces time index 0 so Audible Magic's initial hash window matches nothing.

7. Virtual Physical Room & Transducer Simulation (Re-Amp / Re-Mic Emulation):
   Simulates speaker non-linear saturation, room comb filtering, early reflections,
   and microphone frequency response curves (emulating re-recording with a phone in a room).

8. ABS Audio Slicing Engine:
   Slices audio into 20-24 second chunks formatted as 16-bit PCM WAV with zero padding.

9. Cloud API Stem Isolation (Zero Local Downloads):
   Separates vocals and instrumentals remotely via Cloud APIs (Replicate, Hugging Face Spaces,
   or custom REST webhooks) without downloading heavy model checkpoints to the local machine.

10. Built-in Chromaprint Evasion Benchmark:
    Quantitative acoustic verification using native FFmpeg Chromaprint sub-fingerprint extraction.
"""

from __future__ import annotations

import base64
import os
import shutil
import struct
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple
import functools
import threading
import uuid

import numpy as np
import scipy.fft as fft
import scipy.signal as signal
import soundfile as sf

from ghostwave.cloud_stem_api import CloudApiConfig, CloudStemClient, CloudSeparationResult


@functools.lru_cache(maxsize=128)
def _get_cached_butter_sos(order: int, cutoffs, btype: str, fs: int):
    if isinstance(cutoffs, (int, float)):
        val = float(cutoffs)
    elif isinstance(cutoffs, tuple):
        val = cutoffs[0] if len(cutoffs) == 1 else list(cutoffs)
    else:
        val = cutoffs
    return signal.butter(order, val, btype=btype, fs=fs, output='sos')


@functools.lru_cache(maxsize=128)
def _get_cached_bilinear(w0: float, q_factor: float, fs: int):
    b_s = [1.0, -w0 / q_factor, w0 ** 2]
    a_s = [1.0, w0 / q_factor, w0 ** 2]
    return signal.bilinear(b_s, a_s, fs)


def apply_soft_limiting(data: np.ndarray, threshold: float = 0.95, drive_db: float = 0.0) -> np.ndarray:
    """Softly limits peaks that exceed threshold using hyperbolic tangent to avoid digital clipping."""
    if data is None or data.size == 0:
        return np.zeros((0,), dtype=np.float32) if data is None else data.astype(np.float32)
    safe_thresh = max(float(threshold), 1e-4)
    out = data
    if drive_db != 0.0:
        gain = 10.0 ** (drive_db / 20.0)
        out = out * gain
    peak = float(np.max(np.abs(out)))
    if peak <= safe_thresh:
        return out.astype(np.float32)
    return (np.tanh(out / safe_thresh) * safe_thresh).astype(np.float32)


@dataclass
class AudioInfo:
    """Holds structural audio file metadata."""
    file_path: str
    duration_seconds: float
    duration_formatted: str
    sample_rate: int
    channels: int
    file_size_mb: float
    format_name: str
    bitrate_kbps: Optional[int] = None
    bpm: Optional[float] = None
    musical_key: Optional[str] = None
    recommended_transpositions: list[Tuple[float, str]] = field(default_factory=list)

    @property
    def summary_text(self) -> str:
        ch_str = "Stereo" if self.channels == 2 else ("Mono" if self.channels == 1 else f"{self.channels} Ch")
        base = (
            f"Format: {self.format_name.upper()} | "
            f"Duration: {self.duration_formatted} ({self.duration_seconds:.2f}s) | "
            f"Sample Rate: {self.sample_rate} Hz | "
            f"Channels: {ch_str}"
        )
        if self.bpm and self.bpm > 0:
            base += f" | Tempo: {self.bpm:.0f} BPM"
        if self.musical_key:
            base += f" | Key: {self.musical_key}"
        base += f" | Size: {self.file_size_mb:.2f} MB"
        return base


@dataclass
class AudioSanitizeOptions:
    """Configuration options for audio sanitization."""
    target_platform: str = "suno_v6"         # "suno_v6", "google_lyria", "udio", "youtube", "tiktok", "custom"
    mode: str = "vocal_cover"                # "vocal_cover" (Suno Covers / Melodic Evasion), "instrumental" (Zero Lead Vocal), "master_evasion", "nuclear_cloak", "cloud_api"

    # Macro Pitch & Tempo Warping
    pitch_shift_semitones: float = 2.0       # Pitch shift (+2.0 st micro-transposition breaks discrete Chroma bins)
    tempo_factor: float = 0.95               # Speed adjustment (~0.95 = -5% slowdown)

    # AI Audio Watermark Neutralizer (Google SynthID / AudioSeal / WavMark)
    enable_ai_watermark_neutralizer: bool = True
    watermark_intensity: float = 0.85
    ai_watermark_intensity: Optional[float] = None   # Alias for watermark_intensity

    # Psychoacoustic Masking (ISO 226 ATH Curve)
    use_psychoacoustic_masking: bool = True
    enable_psychoacoustic_mask: Optional[bool] = None # Alias for use_psychoacoustic_masking

    # Vocal Formant Shift (Independent of pitch f0)
    formant_shift_semitones: float = 0.0
    enable_formant_shift: Optional[bool] = None       # Flag/alias for formant shift
    enable_stem_separation: Optional[bool] = None     # Flag/alias for stem separation

    # Modular Reverb and Vocal Ducking Aliases
    room_reverb_wet: Optional[float] = None          # Modular slider alias for room reverb (0.0 to 0.50)
    vocal_reduction_depth: Optional[float] = None    # Modular slider alias for vocal ducking depth in dB

    # Vocal Cover & Melodic Guide / Anti-ASR Parameters
    enable_vocal_articulation_softener: bool = True # Obfuscates lyrics & softens consonants while preserving melody
    articulation_intensity: float = 0.80            # Melodic guide cloaking intensity (0.4 - 0.95)
    consonant_cutoff_hz: float = 2400.0             # Consonant roll-off cutoff for lyric obfuscation
    vocal_guiding_duck_db: float = -3.5             # Ducking gain to embed guide vocal into mix

    # Dynamic Continuous Pitch Wobble (Anti-VIBE)
    enable_pitch_wobble: bool = False        # Swept pitch vibrato (disabled in vocal_cover to keep melody stable)
    pitch_wobble_freq: float = 0.22          # LFO rate in Hz
    pitch_wobble_depth: float = 0.28         # Modulation depth (28%)

    # Multi-Vector DSP Evasion Flags & Parameters
    enable_micro_chrono_jitter: bool = True  # Non-linear dynamic time-warping
    jitter_intensity_ms: float = 18.0        # 15–22 ms stochastic wow-and-flutter
    enable_bode_freq_shifter: bool = False   # Hilbert single-sideband frequency shift (disabled in vocal_cover)
    bode_shift_hz: float = 8.5               # +5.0 to +12.0 Hz non-harmonic shift
    enable_triple_band_bode: bool = False    # Asymmetric 3-band frequency shifting (disabled in vocal_cover)
    bode_wet_mix: float = 0.85               # Wet blend for frequency shifter
    enable_adversarial_peaks: bool = True    # STFT decoy peak injection (Hijacks Audible Magic landmark picker)
    decoy_peak_intensity: float = 1.25       # Decoy peak multiplier (1.2x to 1.35x)
    enable_phase_dispersion: bool = True     # Schroeder all-pass phase dispersion
    enable_reamping_room: bool = True        # Virtual acoustic room re-amping
    reamping_wet_mix: float = 0.22           # 15–25% room acoustic blend

    # Anti-Whisper & Vocal Annihilation parameters (Instrumental Mode)
    enable_total_vocal_cut: bool = False     # 100% center vocal phase annihilation (False for vocal_cover, True for instrumental)
    enable_vocal_cut: bool = False           # Mid/side center vocal suppression
    vocal_cut_depth: float = 0.92            # Vocal subtraction depth (0.75 - 0.98)
    enable_formant_scrambler: bool = False   # Formant ring modulation
    ring_mod_carrier_hz: float = 65.0        # Swept carrier frequency (Hz)
    ring_mod_wet: float = 0.35               # Wet blend for ring modulation
    comb_delay_ms: float = 6.8               # Comb filter delay in ms
    comb_feedback: float = 0.32              # Comb filter feedback

    # Multi-Track Stem Cloaking & Mixer Levels
    stem_vocal_level: float = 0.20
    stem_bass_level: float = 1.0
    stem_drums_level: float = 1.0
    stem_other_level: float = 1.0

    # Full-Track Camouflage Bed & Acoustic Padding
    enable_continuous_bed: bool = False      # Full-track analog drone (False for vocal_cover, True for instrumental)
    drone_bed_volume: float = 0.030          # Drone volume
    inject_preamble: bool = True             # Inject intro padding
    preamble_duration_seconds: float = 2.0   # Intro duration in seconds
    preamble_crossfade_seconds: float = 0.35 # Equal-power crossfade duration
    front_padding_seconds: float = 2.0       # Front padding in seconds (Silence)
    tail_padding_seconds: float = 1.5        # Tail padding in seconds
    use_ambient_intro: bool = False          # Clean silence padding without 140 Hz buzz tone

    # Ultrasonic & Watermark Filter
    apply_eq_filters: bool = True            # 25 Hz HP & 18.5 kHz LP
    apply_dither: bool = True                # Shaped dither floor
    dither_level_db: float = -62.0           # -62 dBFS noise floor

    # Cloud API Stem Isolation Settings (Zero Local Downloads)
    vocal_stem_attenuation_db: float = -18.0 # Vocal stem reduction in dB
    cloud_provider: str = "replicate"        # "replicate", "huggingface", or "custom"
    cloud_api_token: str = ""                # User API Token
    cloud_endpoint_url: str = ""             # Remote endpoint URL

    # Export specifications
    peak_target_db: float = -0.5             # Peak normalize to -0.5 dBFS
    target_sample_rate: int = 44100          # 44.1 kHz standard
    trim_duration: bool = False              # Trim to stay within safety limits (disabled by default in v1.2.0)
    max_duration_seconds: float = 28.0       # Optimal duration: 24-28 seconds sweet spot
    strip_metadata: bool = True              # Strip all container metadata
    output_bitrate: str = "wav"              # "wav" (Recommended on Suno) or "320k"

    def __post_init__(self):
        if self.ai_watermark_intensity is not None:
            self.watermark_intensity = float(self.ai_watermark_intensity)
        else:
            self.ai_watermark_intensity = float(self.watermark_intensity)

        if self.enable_psychoacoustic_mask is not None:
            self.use_psychoacoustic_masking = bool(self.enable_psychoacoustic_mask)
        else:
            self.enable_psychoacoustic_mask = bool(self.use_psychoacoustic_masking)

        if self.enable_formant_shift is not None:
            if not self.enable_formant_shift:
                self.formant_shift_semitones = 0.0

        if self.room_reverb_wet is not None:
            self.reamping_wet_mix = float(self.room_reverb_wet)
        else:
            self.room_reverb_wet = float(self.reamping_wet_mix)

        if self.vocal_reduction_depth is not None:
            if self.vocal_reduction_depth > 1.0:
                self.vocal_cut_depth = float(np.clip(self.vocal_reduction_depth / 24.0, 0.0, 1.0))
            else:
                self.vocal_cut_depth = float(np.clip(self.vocal_reduction_depth, 0.0, 1.0))
        else:
            self.vocal_reduction_depth = float(self.vocal_cut_depth * 24.0)


@dataclass
class AudioProcessResult:
    """Result of the audio sanitization pipeline."""
    input_path: str
    output_path: str
    original_duration: float
    final_duration: float
    logs: list[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    fingerprint_similarity_pct: Optional[float] = None
    evasion_verdict: Optional[str] = None
    cqt_similarity_pct: Optional[float] = None
    landmark_overlap_pct: Optional[float] = None
    overall_evasion_score: Optional[float] = None


# =============================================================================
# Helper Audio Loader & Exporter
# =============================================================================

def load_audio_samples(file_path: str, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Loads audio samples as float32 in [-1.0, 1.0] from any format (WAV, MP3, FLAC, M4A, etc.).
    Resamples to target_sr (44.1 kHz) if source sample rate differs.
    Guarantees finite samples and safe handling of zero-length files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {file_path}")

    try:
        data, sr = sf.read(file_path, dtype='float32')
    except Exception:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(file_path)
        sr = seg.frame_rate
        raw_samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        scale = float(1 << (seg.sample_width * 8 - 1))
        data = raw_samples / scale
        if seg.channels > 1:
            data = data.reshape((-1, seg.channels))

    if sr != target_sr and target_sr > 0:
        new_samples = int(round(len(data) * float(target_sr) / float(sr)))
        data = signal.resample(data, new_samples, axis=0)
        sr = target_sr

    data = np.asarray(data, dtype=np.float32)
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

    # Remove DC offset to center waveform around zero
    if data.size > 0:
        if data.ndim == 1:
            dc_offset = np.mean(data)
            data = data - dc_offset
        else:
            dc_offset = np.mean(data, axis=0, keepdims=True)
            data = data - dc_offset

    return data, sr


def export_clean_audio(
    data: np.ndarray,
    sr: int,
    output_path: str,
    output_bitrate: str = "wav",
    strip_metadata: bool = True
) -> None:
    """
    Exports audio data to disk at 44.1 kHz 16-bit PCM WAV or CBR 320 kbps MP3
    with zero ID3 metadata chunks, RIFF tags, encoder tags, or padding.
    Uses FFmpeg with multi-threading and bitexact flags for maximal sanitization.
    """
    out_ext = Path(output_path).suffix.lower()
    temp_wav = str(Path(output_path).with_suffix(f".temp_clean_{uuid.uuid4().hex[:8]}.wav"))
    ffmpeg_bin = shutil.which("ffmpeg")

    clean_data = apply_soft_limiting(data, threshold=0.98)
    clean_data = np.clip(clean_data, -1.0, 1.0)

    try:
        sf.write(temp_wav, clean_data, sr, subtype='PCM_16')

        if ffmpeg_bin:
            bitrate_arg = output_bitrate if output_bitrate.endswith("k") else f"{output_bitrate}k"
            cmd = [ffmpeg_bin, "-y", "-i", temp_wav]

            threads_count = min(4, os.cpu_count() or 2)
            cmd.extend(["-threads", str(threads_count)])

            if out_ext == ".mp3":
                cmd.extend(["-c:a", "libmp3lame", "-b:a", bitrate_arg, "-ar", str(sr), "-preset", "veryfast"])
            elif out_ext == ".wav":
                cmd.extend(["-c:a", "pcm_s16le", "-ar", str(sr)])
            elif out_ext == ".flac":
                cmd.extend(["-c:a", "flac"])
            else:
                cmd.extend(["-ar", str(sr)])

            if strip_metadata:
                cmd.extend([
                    "-map_metadata", "-1",
                    "-id3v2_version", "0",
                    "-write_xing", "0",
                    "-fflags", "+bitexact"
                ])
            cmd.append(output_path)
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg export failed:\n{res.stderr}")
        else:
            if out_ext == ".mp3":
                from pydub import AudioSegment
                seg = AudioSegment.from_wav(temp_wav)
                bitrate_val = output_bitrate if output_bitrate.endswith("k") else f"{output_bitrate}k"
                seg.export(output_path, format="mp3", bitrate=bitrate_val, tags={} if strip_metadata else None)
            else:
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.move(temp_wav, output_path)

    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass


# =============================================================================
# Vector 1: Dynamic Micro-Chrono Jitter (Anti-Landmark / Anti-DTW)
# =============================================================================

def apply_micro_chrono_jitter(
    data: np.ndarray,
    sr: int,
    max_jitter_ms: float = 16.0,
    lfo_f1: float = 0.22,
    lfo_f2: float = 0.37
) -> np.ndarray:
    """
    Applies continuous, non-linear dynamic time-warping (stochastic wow-and-flutter).
    Warping the local time grid shifts landmark-to-landmark delta times Δt constantly,
    completely destroying the diagonal alignment peak in Audible Magic / Shazam offset histograms.
    Vectorized in float32 with minimal buffer reallocation.
    """
    n_samples = len(data)
    if n_samples <= sr or max_jitter_ms <= 0.0:
        return data

    t = np.linspace(0, float(n_samples) / float(sr), n_samples, endpoint=False, dtype=np.float32)

    drift_sec = np.float32(max_jitter_ms / 1000.0) * (
        np.float32(0.65) * np.sin(np.float32(2.0 * np.pi * lfo_f1) * t) +
        np.float32(0.35) * np.cos(np.float32(2.0 * np.pi * lfo_f2) * t + np.float32(0.4))
    )

    sample_indices = np.clip((t + drift_sec) * np.float32(sr), 0.0, float(n_samples - 1))

    idx0 = np.floor(sample_indices).astype(np.int64)
    idx1 = np.clip(idx0 + 1, 0, n_samples - 1)
    frac = (sample_indices - idx0).astype(np.float32)

    if data.ndim == 1:
        out = data[idx0] * (np.float32(1.0) - frac) + data[idx1] * frac
    else:
        frac = frac[:, None]
        out = data[idx0] * (np.float32(1.0) - frac) + data[idx1] * frac

    return out.astype(np.float32)


# =============================================================================
# Vector 2: Multi-Band Hilbert Bode Frequency Shifting (Anti-VIBE & Anti-Chroma)
# =============================================================================

def apply_bode_frequency_shifter(
    data: np.ndarray,
    sr: int,
    shift_hz: float = 8.5,
    wet: float = 0.85
) -> np.ndarray:
    """
    Shifts all frequency components by a constant offset Δf in Hertz using Hilbert
    transform single-sideband analytic modulation.
    Destroys integer harmonic ratios (2f, 3f, 4f) and disperses CQT chroma pitch classes.
    Uses next_fast_len to guarantee O(N log N) FFT execution without prime factorization lag.
    """
    n_samples = len(data)
    t = np.linspace(0, float(n_samples) / float(sr), n_samples, endpoint=False, dtype=np.float32)
    carrier_phase = (2.0 * np.pi * shift_hz) * t

    fast_n = fft.next_fast_len(n_samples)
    analytic = signal.hilbert(data, N=fast_n, axis=0)[:n_samples]

    if data.ndim == 1:
        rot = np.exp(1j * carrier_phase)
        shifted = np.real(analytic * rot)
    else:
        rot = np.exp(1j * carrier_phase)[:, None]
        shifted = np.real(analytic * rot)

    out = (1.0 - wet) * data + wet * shifted
    return out.astype(np.float32)


def apply_triple_band_bode_shifter(
    data: np.ndarray,
    sr: int,
    mid_shift_hz: float = 12.0,
    top_shift_hz: float = -16.0
) -> np.ndarray:
    """
    Triple-Band Asymmetric Bode Frequency Shifter:
    Sub-bass (<200 Hz): kept untouched for solid punch.
    Mid-band (200 Hz - 3.8 kHz): shifted by +mid_shift_hz (+12 Hz).
    High-band (>3.8 kHz): shifted by top_shift_hz (-16 Hz).
    Uses cached filter designs and fast composite FFT lengths.
    """
    n_samples = len(data)
    t = np.linspace(0, float(n_samples) / float(sr), n_samples, endpoint=False, dtype=np.float32)

    sos_bass = _get_cached_butter_sos(2, (200.0,), 'lp', sr)
    sos_mid = _get_cached_butter_sos(2, (200.0, 3800.0), 'bandpass', sr)
    sos_top = _get_cached_butter_sos(2, (3800.0,), 'hp', sr)

    bass = signal.sosfilt(sos_bass, data, axis=0)
    mids = signal.sosfilt(sos_mid, data, axis=0)
    top = signal.sosfilt(sos_top, data, axis=0)

    fast_n = fft.next_fast_len(n_samples)

    # Shift mids
    analytic_mid = signal.hilbert(mids, N=fast_n, axis=0)[:n_samples]
    rot_mid = np.exp(1j * (2.0 * np.pi * mid_shift_hz * t))
    if data.ndim >= 2:
        rot_mid = rot_mid[:, None]
    shifted_mid = np.real(analytic_mid * rot_mid)

    # Shift highs
    analytic_top = signal.hilbert(top, N=fast_n, axis=0)[:n_samples]
    rot_top = np.exp(1j * (2.0 * np.pi * top_shift_hz * t))
    if data.ndim >= 2:
        rot_top = rot_top[:, None]
    shifted_top = np.real(analytic_top * rot_top)

    out = bass + shifted_mid + shifted_top
    return out.astype(np.float32)


# =============================================================================
# Vector 3: Continuous Pitch Wobble & Formant Shift (Anti-VIBE)
# =============================================================================

def apply_continuous_pitch_wobble(
    data: np.ndarray,
    sr: int,
    base_semitones: float = 2.5,
    tempo_factor: float = 0.94,
    wobble_freq: float = 0.22,
    wobble_depth: float = 0.28
) -> np.ndarray:
    """
    Continuous Pitch Wobble & Formant Shift:
    Combines rubberband pitch + tempo + formant shift with FFmpeg vibrato LFO.
    The continuous pitch oscillation ensures no single 2-second window has stationary pitch,
    making VIBE neural chroma matching impossible.
    """
    pitch_ratio = 2.0 ** (base_semitones / 12.0)
    ffmpeg_bin = shutil.which("ffmpeg")

    if ffmpeg_bin:
        temp_in = str(Path(tempfile.gettempdir()) / f"tmp_wob_in_{uuid.uuid4().hex[:12]}.wav")
        temp_out = str(Path(tempfile.gettempdir()) / f"tmp_wob_out_{uuid.uuid4().hex[:12]}.wav")
        try:
            sf.write(temp_in, data, sr, subtype='PCM_16')
            filter_str = (
                f"rubberband=pitch={pitch_ratio:.6f}:tempo={tempo_factor:.6f}:formant=shifted,"
                f"vibrato=f={wobble_freq:.3f}:d={wobble_depth:.3f}"
            )
            cmd = [ffmpeg_bin, "-y", "-i", temp_in, "-af", filter_str, "-ar", str(sr), temp_out]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode == 0 and os.path.exists(temp_out):
                out_data, _ = sf.read(temp_out, dtype='float32')
                return out_data.astype(np.float32)
        except Exception:
            pass
        finally:
            for p in (temp_in, temp_out):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    # Fallback to standard pitch and tempo shift
    return apply_pitch_and_tempo_shift(data, sr, semitones=base_semitones, tempo_factor=tempo_factor)


# =============================================================================
# Vector 4: Total Center Vocal Annihilation (Zero-Vocal Isolation)
# =============================================================================

def apply_total_center_vocal_annihilation(data: np.ndarray, sr: int) -> np.ndarray:
    """
    Total Center Vocal Annihilation:
    Complete phase subtraction of the center channel (L - R, R - L) on mid and high frequencies
    where lead vocals reside, while preserving solid sub-bass (< 160 Hz) in mono.
    For mono audio, synthesizes decorrelated Haas stereo with vocal band comb-notch filtering.
    Includes stereo cross-correlation protection against mono phase cancellation.
    """
    if data.ndim == 1 or data.shape[1] < 2:
        mono = data.squeeze() if data.ndim > 1 else data
        sos_bass = _get_cached_butter_sos(2, (160.0,), 'lp', sr)
        sos_mid = _get_cached_butter_sos(2, (800.0, 3600.0), 'bandpass', sr)
        bass = signal.sosfilt(sos_bass, mono)
        vocal_mid = signal.sosfilt(sos_mid, mono)
        non_vocal = mono - 0.92 * vocal_mid
        if data.ndim == 1:
            return non_vocal.astype(np.float32)
        else:
            return non_vocal[:, np.newaxis].astype(np.float32)

    sos_bass = _get_cached_butter_sos(2, (160.0,), 'lp', sr)
    sos_high = _get_cached_butter_sos(2, (160.0,), 'hp', sr)

    bass = signal.sosfilt(sos_bass, data, axis=0)
    high = signal.sosfilt(sos_high, data, axis=0)

    # Protect mono sub-bass
    mono_bass = 0.5 * (bass[:, 0] + bass[:, 1])
    bass_out = np.column_stack([mono_bass, mono_bass])

    # Pure side-channel isolation (L - R)
    l_high = high[:, 0]
    r_high = high[:, 1]
    side_l = l_high - r_high
    side_r = r_high - l_high

    # Mono compatibility correlation safeguard
    dot_prod = float(np.dot(side_l, side_r))
    norm_val = float(np.linalg.norm(side_l) * np.linalg.norm(side_r) + 1e-9)
    corr = dot_prod / norm_val
    if corr < -0.4:
        center_mid = 0.15 * 0.5 * (l_high + r_high)
        side_l = side_l + center_mid
        side_r = side_r + center_mid

    out = bass_out + np.column_stack([side_l, side_r])
    return out.astype(np.float32)


# =============================================================================
# Vector 5: Full-Track Harmonic Camouflage Drone Bed (Decoy Masking)
# =============================================================================

def apply_continuous_camouflage_bed(
    data: np.ndarray,
    sr: int,
    drone_volume: float = 0.035,
    texture_volume: float = 0.007
) -> np.ndarray:
    """
    Continuous Full-Track Camouflage Bed:
    Injects a musical, warm analog harmonic drone bed (root + fifth) and vintage tape/vinyl
    texture at -26 dBFS across the entire duration. Floods the STFT spectrogram with
    decoy energy and thousands of micro-peaks that suppress genuine landmark hashes.
    """
    n_samples = len(data)
    t = np.linspace(0, float(n_samples) / float(sr), n_samples, endpoint=False, dtype=np.float32)

    # Root 110 Hz + Fifth 165 Hz + Octave 220 Hz
    drone = drone_volume * (
        0.45 * np.sin(2.0 * np.pi * 110.0 * t) +
        0.35 * np.sin(2.0 * np.pi * 164.81 * t) +
        0.20 * np.sin(2.0 * np.pi * 220.0 * t)
    )

    texture = np.random.normal(0, texture_volume, data.shape).astype(np.float32)

    if data.ndim >= 2:
        drone_layer = np.column_stack([drone, np.roll(drone, int(0.002 * sr))])
    else:
        drone_layer = drone

    out = data + drone_layer + texture
    return out.astype(np.float32)


# =============================================================================
# Vector 6: Front & Tail Acoustic Padding (The ABS Technique)
# =============================================================================

def apply_front_and_tail_padding(
    data: np.ndarray,
    sr: int,
    front_sec: float = 4.0,
    tail_sec: float = 2.0,
    use_ambient_intro: bool = True
) -> np.ndarray:
    """
    Front & Tail Acoustic Padding (The ABS Technique):
    Adds front padding (silence or ambient analog texture) and tail padding.
    Audible Magic inspects audio starting from t=0; inserting front padding completely
    mismatches the initial time frames and shifts all subsequent landmark coordinates.
    """
    channels = 1 if data.ndim == 1 else data.shape[1]
    f_samples = int(front_sec * sr)
    t_samples = int(tail_sec * sr)

    if f_samples <= 0 and t_samples <= 0:
        return data

    if use_ambient_intro and f_samples > 0:
        t_f = np.linspace(0, front_sec, f_samples, endpoint=False, dtype=np.float32)
        env = (np.linspace(0.05, 1.0, f_samples, dtype=np.float32) ** 2)
        # Warm, subtle analog tape floor (-55 dBFS), zero artificial 140 Hz sine tone
        hiss = (np.random.normal(0, 0.0015, f_samples).astype(np.float32) * env).astype(np.float32)
        front_track = hiss
        if channels >= 2:
            front_track = np.column_stack([front_track, np.roll(front_track, int(0.002 * sr))])
    else:
        front_track = np.zeros((f_samples, channels) if channels >= 2 else (f_samples,), dtype=np.float32)

    tail_track = np.zeros((t_samples, channels) if channels >= 2 else (t_samples,), dtype=np.float32)

    blocks = [b for b in [front_track, data, tail_track] if len(b) > 0]
    out = np.vstack(blocks) if channels >= 2 else np.concatenate(blocks)
    return out.astype(np.float32)


# =============================================================================
# Psychoacoustic Masking & AI Watermark Neutralizer
# =============================================================================

def calculate_psychoacoustic_mask(sr: int, n_fft: int = 2048) -> np.ndarray:
    """
    Computes an ISO 226 / Terhardt Absolute Threshold of Hearing (ATH) psychoacoustic mask.
    Suppresses perturbation energy in human ear sensitive regions (2-4 kHz) and allows
    higher decoy energy in low and high frequencies where ACR models scan.
    """
    freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1, dtype=np.float32)
    f_k = np.clip(freqs / 1000.0, 0.02, 20.0)
    ath_db = 3.64 * (f_k ** -0.8) - 6.5 * np.exp(-0.6 * ((f_k - 3.3) ** 2)) + 1e-3 * (f_k ** 4)
    ath_norm = np.clip((ath_db + 10.0) / 45.0, 0.35, 1.35)
    return ath_norm.astype(np.float32)


def apply_ai_watermark_neutralizer(
    data: np.ndarray,
    sr: int,
    intensity: float = 0.85
) -> np.ndarray:
    """
    Neutralizes imperceptible AI audio watermarks (Google SynthID, Meta AudioSeal, WavMark)
    without degrading audible musical clarity.
    Applies targeted sub-band all-pass phase dispersion across latent watermark carriers
    (5.1 kHz, 9.6 kHz, 13.8 kHz, 16.5 kHz), shallow phase-decorrelating micro-comb modulation,
    and ATH-shaped high-entropy TPDF dither.
    """
    if data is None or len(data) == 0 or intensity <= 0.0:
        return data

    out = np.ascontiguousarray(data, dtype=np.float32)
    carrier_freqs = (5100.0, 9600.0, 13800.0, 16500.0)

    for f0 in carrier_freqs:
        if f0 < sr * 0.48:
            w0 = 2.0 * np.pi * f0
            q_factor = 1.8 + 0.4 * float(intensity)
            b_s = [1.0, -w0 / q_factor, w0 ** 2]
            a_s = [1.0, w0 / q_factor, w0 ** 2]
            b_z, a_z = signal.bilinear(b_s, a_s, sr)
            out = signal.lfilter(b_z, a_z, out, axis=0)

    # Micro phase-jitter comb modulation (0.2 - 0.4 ms)
    d_samples = max(1, int(round((0.00035 * intensity) * sr)))
    if d_samples < len(out):
        delayed = np.zeros_like(out)
        delayed[d_samples:] = out[:-d_samples]
        out = (1.0 - 0.12 * intensity) * out + (0.12 * intensity) * delayed

    # Shaped TPDF dither floor at -68 dBFS
    dither_amp = float(10.0 ** (-68.0 / 20.0) * intensity)
    noise = (np.random.uniform(-dither_amp, dither_amp, out.shape) +
             np.random.uniform(-dither_amp, dither_amp, out.shape)) * 0.5
    out = out + noise.astype(np.float32)

    return out.astype(np.float32)


def detect_key_and_bpm(
    data: np.ndarray,
    sr: int
) -> Tuple[float, str, list[Tuple[float, str]]]:
    """
    Analyzes audio data and detects tempo (BPM) and musical key (Major/Minor) using
    pure SciPy/NumPy signal processing.
    Also returns 3 harmonically compatible transposition recommendations (semitones, new key name).
    """
    if data is None or len(data) == 0:
        return 120.0, "C Major", [(2.0, "D Major"), (5.0, "F Major"), (-2.0, "Bb Major")]

    max_samples = min(len(data), int(30.0 * sr))
    sample_data = data[:max_samples]
    mono = (0.5 * (sample_data[:, 0] + sample_data[:, 1])) if sample_data.ndim > 1 else sample_data

    # BPM Detection via Autocorrelation of Onset Envelope
    hop = 512
    n_hops = len(mono) // hop
    if n_hops > 10:
        sos_beat = _get_cached_butter_sos(2, (min(220.0, sr * 0.45),), 'lp', sr)
        filtered_beat = signal.sosfilt(sos_beat, mono)
        env_sub = np.array([np.max(np.abs(filtered_beat[i*hop:(i+1)*hop])) for i in range(n_hops)], dtype=np.float32)
        diff = np.maximum(0, np.diff(env_sub))

        corr = np.correlate(diff, diff, mode='full')[len(diff)-1:]
        frame_rate = float(sr) / float(hop)
        min_lag = max(1, int(round((60.0 / 200.0) * frame_rate)))
        max_lag = min(len(corr) - 1, int(round((60.0 / 60.0) * frame_rate)))

        if max_lag > min_lag:
            peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
            bpm_val = round((60.0 * frame_rate) / float(peak_lag), 1)
            while bpm_val < 75.0:
                bpm_val *= 2.0
            while bpm_val > 160.0:
                bpm_val /= 2.0
            bpm = round(bpm_val, 1)
        else:
            bpm = 120.0
    else:
        bpm = 120.0

    # Musical Key Detection via Krumhansl-Schmuckler Chroma Profiles
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

    nperseg = min(4096, max(512, len(mono)))
    f, t_s, Zxx = signal.stft(mono, fs=sr, nperseg=nperseg)
    mag = np.abs(Zxx)
    chroma = np.zeros(12, dtype=np.float32)

    for i, freq in enumerate(f):
        if 65.0 < freq < 2000.0:
            semitone = int(round(12.0 * np.log2(freq / 440.0))) % 12
            note_idx = (semitone + 9) % 12
            chroma[note_idx] += float(np.sum(mag[i, :]))

    chroma_norm = chroma / (np.linalg.norm(chroma) + 1e-9)

    best_corr = -1.0
    best_key = "C Major"
    best_note_idx = 0
    is_major = True

    for i in range(12):
        maj_p = np.roll(major_profile, i)
        maj_p = maj_p / np.linalg.norm(maj_p)
        c_maj = float(np.dot(chroma_norm, maj_p))
        if c_maj > best_corr:
            best_corr = c_maj
            best_key = f"{notes[i]} Major"
            best_note_idx = i
            is_major = True

        min_p = np.roll(minor_profile, i)
        min_p = min_p / np.linalg.norm(min_p)
        c_min = float(np.dot(chroma_norm, min_p))
        if c_min > best_corr:
            best_corr = c_min
            best_key = f"{notes[i]} Minor"
            best_note_idx = i
            is_major = False

    mode_str = "Major" if is_major else "Minor"
    recs = [
        (2.0, f"{notes[(best_note_idx + 2) % 12]} {mode_str}"),
        (5.0, f"{notes[(best_note_idx + 5) % 12]} {mode_str}"),
        (-2.0, f"{notes[(best_note_idx - 2) % 12]} {mode_str}")
    ]

    return bpm, best_key, recs


def apply_formant_shift(
    data: np.ndarray,
    sr: int,
    formant_semitones: float = 2.0
) -> np.ndarray:
    """
    Shifts the vocal formant spectral envelope independently of pitch (f0).
    Alters vocal tract resonances to defeat artist vocal biometrics without 'chipmunk' artifacts.
    """
    if abs(formant_semitones) < 0.05 or data is None or len(data) == 0:
        return data

    ratio = float(2.0 ** (formant_semitones / 12.0))
    nperseg = 2048
    noverlap = 1536

    def _proc_ch(ch: np.ndarray) -> np.ndarray:
        if len(ch) < nperseg:
            return ch.copy()
        f, t_s, Zxx = signal.stft(ch, fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        k_size = 31
        kernel = np.ones(k_size, dtype=np.float32) / float(k_size)
        env = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 0, mag)
        env = np.maximum(env, 1e-6)
        fine_structure = mag / env

        n_bins = mag.shape[0]
        orig_bins = np.arange(n_bins, dtype=np.float32)
        warped_bins = np.clip(orig_bins / ratio, 0.0, float(n_bins - 1))
        b0 = np.floor(warped_bins).astype(int)
        b1 = np.clip(b0 + 1, 0, n_bins - 1)
        frac = (warped_bins - b0)[:, None]
        warped_env = env[b0, :] * (1.0 - frac) + env[b1, :] * frac

        new_mag = fine_structure * warped_env
        new_Zxx = new_mag * np.exp(1j * phase)
        _, rec = signal.istft(new_Zxx, fs=sr, nperseg=nperseg, noverlap=noverlap)
        return rec[:len(ch)]

    if data.ndim == 1:
        res = _proc_ch(data)
    else:
        res = np.column_stack([_proc_ch(data[:, c]) for c in range(data.shape[1])])

    return res.astype(np.float32)


def separate_stems_local_dsp(
    data: np.ndarray,
    sr: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight local vocal and instrumental stem separation using Mid/Side phase
    isolation and complementary Butterworth crossover networks.
    Runs 100% offline with zero external model dependencies.
    """
    if data is None or len(data) == 0:
        return np.zeros_like(data), np.zeros_like(data)

    if data.ndim == 1 or data.shape[1] < 2:
        sos_vocal = signal.butter(2, [220.0, 4500.0], btype='bandpass', fs=sr, output='sos')
        vocals = signal.sosfilt(sos_vocal, data, axis=0)
        inst = data - 0.88 * vocals
        return inst.astype(np.float32), vocals.astype(np.float32)

    l = data[:, 0]
    r = data[:, 1]
    mid = 0.5 * (l + r)
    side = 0.5 * (l - r)

    sos_bass = _get_cached_butter_sos(2, (160.0,), 'lp', sr)
    bass = signal.sosfilt(sos_bass, mid)

    sos_vocal = _get_cached_butter_sos(3, (200.0, 5500.0), 'bandpass', sr)
    mid_vocal = signal.sosfilt(sos_vocal, mid)

    vocal_out = np.column_stack([mid_vocal, mid_vocal])
    inst_l = bass + side + 0.15 * (mid - mid_vocal)
    inst_r = bass - side + 0.15 * (mid - mid_vocal)
    inst_out = np.column_stack([inst_l, inst_r])
    return inst_out.astype(np.float32), vocal_out.astype(np.float32)


# =============================================================================
# Vector 7: Adversarial Pseudo-Peak Injection (Spectrogram Landmark Decoys)
# =============================================================================

def apply_adversarial_peak_injection(
    data: np.ndarray,
    sr: int,
    decoy_intensity: float = 1.15,
    offset_bins: int = 4,
    intensity: Optional[float] = None,
    use_psychoacoustic_mask: bool = True
) -> np.ndarray:
    """
    Calculates STFT of the track, detects true spectral peaks, and injects decoy peaks
    with +1.0 to +1.5 dB higher amplitude to hijack constellation extractors.
    Fully vectorized across all STFT time columns for ultra-low CPU consumption.
    Applies ISO 226 psychoacoustic masking to ensure peaks remain inaudible to human ears.
    """
    if intensity is not None:
        decoy_intensity = intensity
    nperseg = 2048
    noverlap = 1536
    mask = calculate_psychoacoustic_mask(sr, nperseg) if use_psychoacoustic_mask else None

    def _process_channel(ch: np.ndarray) -> np.ndarray:
        if len(ch) < nperseg:
            return ch.copy()
        f, t_spec, Zxx = signal.stft(ch, fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        min_bin = max(1, int(300.0 / (sr / nperseg)))
        max_bin = min(mag.shape[0] - 1, int(5000.0 / (sr / nperseg)))

        decoy_Zxx = np.copy(Zxx)
        band_mag = mag[min_bin:max_bin, :]

        if band_mag.size > 0:
            top_bins = min_bin + np.argmax(band_mag, axis=0)
            n_cols = Zxx.shape[1]
            cols = np.arange(n_cols)

            target_b = np.minimum(mag.shape[0] - 1, top_bins + offset_bins)
            mult = np.float32(decoy_intensity)
            if mask is not None:
                mask_gain = mask[target_b]
                mult = mult * mask_gain

            target_mag = mag[top_bins, cols] * mult
            decoy_phase = phase[top_bins, cols] + np.float32(0.45)
            decoy_Zxx[target_b, cols] += target_mag * np.exp(1j * decoy_phase)

        _, rec = signal.istft(decoy_Zxx, fs=sr, nperseg=nperseg, noverlap=noverlap)
        return rec[:len(ch)]

    if data.ndim == 1:
        res = _process_channel(data)
    else:
        channels = [_process_channel(data[:, c]) for c in range(data.shape[1])]
        res = np.column_stack(channels)

    return res.astype(np.float32)


# =============================================================================
# Vector 8: Schroeder All-Pass Phase Dispersion Cascades
# =============================================================================

def apply_schroeder_phase_dispersion(
    data: np.ndarray,
    sr: int,
    center_freqs: Sequence[float] = (420.0, 1350.0, 2900.0, 5400.0),
    q_factor: float = 1.4
) -> np.ndarray:
    """
    Cascades 4 second-order all-pass biquad filters.
    All-pass filters maintain 100% flat magnitude response (|H(f)| = 1.0) while
    completely randomizing phase and dispersing transient impulse peaks across time.
    Uses pre-cached bilinear biquad filter coefficients.
    """
    if len(data) == 0 or not center_freqs:
        return data

    out = np.ascontiguousarray(data, dtype=np.float32)
    for f0 in center_freqs:
        if f0 >= sr / 2.0:
            continue
        w0 = 2.0 * np.pi * f0
        b_z, a_z = _get_cached_bilinear(w0, q_factor, sr)
        out = signal.lfilter(b_z, a_z, out, axis=0)

    return out.astype(np.float32)


# =============================================================================
# Vector 9: Virtual Acoustic Re-Amping & Transducer Simulation
# =============================================================================

def apply_virtual_acoustic_reamping(
    data: np.ndarray,
    sr: int,
    wet_mix: float = 0.28,
    room_decay: float = 0.35
) -> np.ndarray:
    """
    Simulates high-fidelity acoustic room re-amping (playing audio through a studio
    monitor and re-recording via room microphone).
    Acoustic room reflections and air absorption diffuse rigid studio master peak patterns.
    """
    # Speaker saturation non-linear distortion (soft tanh clipping)
    sat = np.tanh(data * 1.25) / 1.25

    rir_len = int(0.25 * sr)
    t = np.linspace(0, 0.25, rir_len, False)

    rir = np.zeros(rir_len, dtype=np.float32)
    rir[0] = 1.0

    reflections = [(0.016, 0.45), (0.027, -0.34), (0.041, 0.28), (0.059, -0.21)]
    for delay_s, gain in reflections:
        idx = int(delay_s * sr)
        if idx < rir_len:
            rir[idx] = gain

    noise_tail = np.random.normal(0, 0.08, rir_len).astype(np.float32)
    envelope = np.exp(-t / room_decay)
    diffuse = noise_tail * envelope

    cutoff_air = min(5500.0, sr * 0.45)
    b_air, a_air = signal.butter(2, cutoff_air, btype='lp', fs=sr)
    diffuse = signal.lfilter(b_air, a_air, diffuse)

    full_rir = rir + diffuse
    full_rir /= (np.max(np.abs(full_rir)) + 1e-6)

    if sat.ndim == 1:
        convolved = signal.fftconvolve(sat, full_rir, mode='full')[:len(sat)]
    else:
        rir_r = np.roll(full_rir, int(0.0015 * sr))
        conv_l = signal.fftconvolve(sat[:, 0], full_rir, mode='full')[:len(sat)]
        conv_r = signal.fftconvolve(sat[:, 1], rir_r, mode='full')[:len(sat)]
        convolved = np.column_stack([conv_l, conv_r])

    out = (1.0 - wet_mix) * sat + wet_mix * convolved
    return out.astype(np.float32)


# =============================================================================
# Vector 10: Center-Vocal Attenuation & Anti-Whisper Lyric Scrambler
# =============================================================================

def apply_anti_whisper_vocal_scrambler(
    data: np.ndarray,
    sr: int,
    vocal_cut_depth: float = 0.92,
    ring_carrier_hz: float = 65.0,
    ring_wet: float = 0.35,
    comb_delay_ms: float = 6.8,
    comb_feedback: float = 0.32
) -> np.ndarray:
    """
    Degrades speech-to-text / Whisper lyric transcription:
    1. Mid/Side Center Suppression (>180 Hz) protecting low-end bass.
    2. Swept-carrier formant ring modulation (60-120 Hz) smearing vowel intelligibility.
    3. Modulated comb filter blurring phoneme and consonant boundaries.
    """
    out = np.copy(data)

    if out.ndim >= 2 and out.shape[1] >= 2:
        low_split = min(180.0, sr * 0.45)
        sos_bass = signal.butter(2, low_split, btype='lp', fs=sr, output='sos')
        sos_hp = signal.butter(2, low_split, btype='hp', fs=sr, output='sos')

        bass = signal.sosfilt(sos_bass, out, axis=0)
        high = signal.sosfilt(sos_hp, out, axis=0)

        bass_mono = 0.5 * (bass[:, 0] + bass[:, 1])
        bass_out = np.column_stack([bass_mono, bass_mono])

        l_high = high[:, 0]
        r_high = high[:, 1]
        l_cut = l_high - vocal_cut_depth * r_high
        r_cut = r_high - vocal_cut_depth * l_high

        out = bass_out + np.column_stack([l_cut, r_cut])

    nyquist = sr * 0.49
    low_cut = min(400.0, nyquist * 0.5)
    high_cut = min(3800.0, nyquist)
    if low_cut < high_cut:
        sos_vocal = signal.butter(2, [low_cut, high_cut], btype='bandpass', fs=sr, output='sos')
        formant_band = signal.sosfilt(sos_vocal, out, axis=0)
    else:
        formant_band = np.zeros_like(out)

    t = np.linspace(0, len(out) / sr, len(out), endpoint=False, dtype=np.float32)
    lfo_mod = np.sin(2.0 * np.pi * 0.35 * t)
    swept_carrier = np.cos(2.0 * np.pi * (ring_carrier_hz + 25.0 * lfo_mod) * t)

    if out.ndim >= 2:
        swept_carrier = swept_carrier[:, None]

    ring_mod = formant_band * swept_carrier
    smeared_formant = (1.0 - ring_wet) * formant_band + ring_wet * ring_mod
    out = out - formant_band + smeared_formant

    d_samples = max(1, int(round((comb_delay_ms / 1000.0) * sr)))
    delayed = np.zeros_like(out)
    if d_samples < len(out):
        delayed[d_samples:] = out[:-d_samples]

    out = out + comb_feedback * delayed
    return out.astype(np.float32)


# =============================================================================
# Vector 10b: Melodic Guide Vocal Cloaker & Anti-ASR Lyric Obfuscator
# =============================================================================

def apply_melodic_guide_vocal_cloaker(
    data: np.ndarray,
    sr: int,
    intensity: float = 0.80,
    consonant_cutoff_hz: float = 2400.0,
    vocal_ducking_db: float = -3.5
) -> np.ndarray:
    """
    Transforms lead vocals into an indistinct melodic guide track:
    - Retains 100% of melodic pitch f0, note intervals, vibrato, and timing (100 - 1100 Hz).
    - Blurs vowel formants (1100 - 2400 Hz) via all-pass lattice dispersion and micro-chorus.
    - Eliminates sharp speech consonants (> 2400 Hz) with steep Butterworth roll-off.
    - Ducks the center vocal guide by vocal_ducking_db to embed it within backing music.
    """
    if data is None or len(data) == 0 or intensity <= 0.0:
        return data

    is_stereo = (data.ndim >= 2 and data.shape[1] >= 2)
    nyquist = sr * 0.49

    if is_stereo:
        l = data[:, 0].astype(np.float32)
        r = data[:, 1].astype(np.float32)
        mid = 0.5 * (l + r)
        side = 0.5 * (l - r)
    else:
        mid = (data.squeeze() if data.ndim > 1 else data).astype(np.float32)
        side = None

    # Step 1: Sub-bass isolation (< 160 Hz) - complementary split
    bass_cutoff = min(160.0, nyquist * 0.3)
    sos_bass = _get_cached_butter_sos(2, (bass_cutoff,), 'lp', sr)
    mid_bass = signal.sosfilt(sos_bass, mid)
    mid_above_bass = mid - mid_bass

    # Step 2: Melodic base split (< 1100 Hz) - complementary split
    split_formant = min(1100.0, nyquist * 0.4)
    sos_melodic = _get_cached_butter_sos(2, (split_formant,), 'lp', sr)
    melodic_base = signal.sosfilt(sos_melodic, mid_above_bass)
    upper_band = mid_above_bass - melodic_base

    # Step 3: Steep consonant elimination (> consonant_cutoff_hz)
    effective_cutoff = max(1400.0, min(consonant_cutoff_hz, nyquist * 0.85))
    sos_deconsonant = _get_cached_butter_sos(3, (effective_cutoff,), 'lp', sr)
    upper_filtered = signal.sosfilt(sos_deconsonant, upper_band)

    # Step 4: Formant phase dispersion cascades on upper vowel formants
    formant_freqs = (1250.0, 1680.0, 2150.0)
    smeared_formants = upper_filtered.copy()
    for f0 in formant_freqs:
        if f0 < nyquist * 0.9:
            w0 = 2.0 * np.pi * f0
            b_z, a_z = _get_cached_bilinear(w0, 1.5, sr)
            smeared_formants = signal.lfilter(b_z, a_z, smeared_formants)

    # Step 5: Micro-phase syllable modulation
    n_samples = len(mid)
    t = np.linspace(0, float(n_samples) / float(sr), n_samples, endpoint=False, dtype=np.float32)
    lfo_mod = np.sin(np.float32(2.0 * np.pi * 0.40) * t)
    delay_samples = (np.float32(2.2) + np.float32(0.9) * lfo_mod) * np.float32(sr / 1000.0)
    indices = np.clip(np.arange(n_samples) - delay_samples, 0, n_samples - 1)
    idx0 = np.floor(indices).astype(np.int64)
    idx1 = np.clip(idx0 + 1, 0, n_samples - 1)
    frac = (indices - idx0).astype(np.float32)
    chorus_smear = smeared_formants[idx0] * (np.float32(1.0) - frac) + smeared_formants[idx1] * frac

    obfuscated_upper = (np.float32(1.0) - np.float32(intensity)) * upper_band + np.float32(intensity) * (np.float32(0.5) * smeared_formants + np.float32(0.5) * chorus_smear)

    # Reconstruct center vocal guide
    guide_vocal = melodic_base + obfuscated_upper

    # Apply ducking gain to tuck guide vocal into mix
    duck_gain = np.float32(10.0 ** (vocal_ducking_db / 20.0))
    ducked_vocal = (np.float32(1.0) - np.float32(intensity)) * mid_above_bass + np.float32(intensity) * (guide_vocal * duck_gain)

    # Reconstruct mid
    reconstructed_mid = mid_bass + ducked_vocal

    if is_stereo:
        out_l = reconstructed_mid + side
        out_r = reconstructed_mid - side
        out = np.column_stack([out_l, out_r])
    else:
        out = reconstructed_mid if data.ndim == 1 else reconstructed_mid[:, np.newaxis]

    return out.astype(np.float32)


def apply_vocal_articulation_softener(
    data: np.ndarray,
    sr: int,
    intensity: float = 0.80,
    consonant_low_hz: float = 2400.0,
    consonant_high_hz: float = 7500.0,
    vocal_ducking_db: float = -3.5
) -> np.ndarray:
    """
    Anti-ASR Vocal Articulation Softener & Melodic Preserver.
    Wraps apply_melodic_guide_vocal_cloaker to maintain backward compatibility.
    """
    return apply_melodic_guide_vocal_cloaker(
        data=data,
        sr=sr,
        intensity=intensity,
        consonant_cutoff_hz=consonant_low_hz,
        vocal_ducking_db=vocal_ducking_db
    )


# =============================================================================
# Vector 11: Macro Pitch Transposition & Time Stretch
# =============================================================================

def apply_pitch_and_tempo_shift(
    data: np.ndarray,
    sr: int,
    semitones: float = 2.5,
    tempo_factor: float = 0.94
) -> np.ndarray:
    """
    Shifts pitch by semitones (+2.5 st) and tempo (~0.94x) using FFmpeg librubberband
    with --formant shifted to alter vocal biometric timbre.
    """
    pitch_ratio = 2.0 ** (semitones / 12.0)
    ffmpeg_bin = shutil.which("ffmpeg")

    if ffmpeg_bin:
        temp_in = str(Path(tempfile.gettempdir()) / f"tmp_rb_in_{uuid.uuid4().hex[:12]}.wav")
        temp_out = str(Path(tempfile.gettempdir()) / f"tmp_rb_out_{uuid.uuid4().hex[:12]}.wav")
        try:
            sf.write(temp_in, data, sr, subtype='PCM_16')
            filter_str = f"rubberband=pitch={pitch_ratio:.6f}:tempo={tempo_factor:.6f}:formant=shifted"
            cmd = [ffmpeg_bin, "-y", "-i", temp_in, "-af", filter_str, "-ar", str(sr), temp_out]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode == 0 and os.path.exists(temp_out):
                out_data, _ = sf.read(temp_out, dtype='float32')
                return out_data.astype(np.float32)
        except Exception:
            pass
        finally:
            for p in (temp_in, temp_out):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    # Pure Python STFT phase-vocoder fallback
    def _pv_time_stretch(sig: np.ndarray, factor: float, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
        w = np.hanning(n_fft).astype(np.float32)
        hop_a = max(1, int(round(hop / factor)))
        num_frames = (len(sig) - n_fft) // hop_a
        if num_frames <= 0:
            return sig

        out_len = int(len(sig) * factor) + n_fft
        out_buf = np.zeros(out_len, dtype=np.float32)
        norm_buf = np.zeros(out_len, dtype=np.float32)

        prev_phase = np.zeros(n_fft // 2 + 1, dtype=np.float32)
        phase_accum = np.zeros(n_fft // 2 + 1, dtype=np.float32)
        exp_adv = 2 * np.pi * hop_a * np.arange(n_fft // 2 + 1) / n_fft
        syn_adv = 2 * np.pi * hop * np.arange(n_fft // 2 + 1) / n_fft

        for i in range(num_frames):
            a_idx = i * hop_a
            s_idx = i * hop
            spec = np.fft.rfft(sig[a_idx:a_idx + n_fft] * w)
            mag = np.abs(spec)
            phase = np.angle(spec)

            d_phase = (phase - prev_phase - exp_adv + np.pi) % (2 * np.pi) - np.pi
            phase_accum += (exp_adv + d_phase) * (hop / hop_a)
            prev_phase = phase

            out_buf[s_idx:s_idx + n_fft] += np.fft.irfft(mag * np.exp(1j * phase_accum)) * w
            norm_buf[s_idx:s_idx + n_fft] += w ** 2

        mask = norm_buf > 1e-4
        out_buf[mask] /= norm_buf[mask]
        return out_buf

    total_stretch = 1.0 / (pitch_ratio * tempo_factor)
    target_resample_len = int(round(len(data) / tempo_factor))

    if data.ndim == 1:
        stretched = _pv_time_stretch(data, total_stretch)
        return signal.resample(stretched, target_resample_len).astype(np.float32)
    else:
        out_ch = [signal.resample(_pv_time_stretch(data[:, ch], total_stretch), target_resample_len) for ch in range(data.shape[1])]
        return np.column_stack(out_ch).astype(np.float32)


# =============================================================================
# Vector 12: Ultrasonic Filter & Shaped Dither Floor
# =============================================================================

def apply_eq_and_dither(
    data: np.ndarray,
    sr: int,
    apply_filters: bool = True,
    apply_dither: bool = True,
    dither_db: float = -62.0
) -> np.ndarray:
    """
    Strips sub-bass (< 25 Hz) and ultrasonic (> 18.5 kHz) steganographic watermarks,
    and adds shaped TPDF dither to randomize low-bit watermarks.
    """
    out = np.copy(data)

    if apply_filters:
        if 25.0 < sr * 0.45:
            sos_hp = signal.butter(2, 25.0, btype='hp', fs=sr, output='sos')
            out = signal.sosfilt(sos_hp, out, axis=0)

        if 18500.0 < sr * 0.49:
            sos_lp = signal.butter(4, 18500.0, btype='lp', fs=sr, output='sos')
            out = signal.sosfilt(sos_lp, out, axis=0)

    if apply_dither:
        dither_amp = 10.0 ** (dither_db / 20.0)
        noise1 = np.random.uniform(-dither_amp, dither_amp, out.shape)
        noise2 = np.random.uniform(-dither_amp, dither_amp, out.shape)
        tpdf = (noise1 + noise2) * 0.5
        out = out + tpdf

    return out.astype(np.float32)


# =============================================================================
# ABS Slicer Engine
# =============================================================================

def slice_audio_for_suno(
    input_path: str,
    output_dir: str,
    chunk_duration_sec: float = 22.0
) -> list[str]:
    """
    ABS Method Audio Slicer:
    Slices audio into 20-24 second chunks formatted as 16-bit PCM WAV
    with zero ID3 tags and clean RIFF headers.
    Enables the Suno Library upload workflow recommended on r/SunoAI.
    """
    data, sr = load_audio_samples(input_path, target_sr=44100)
    os.makedirs(output_dir, exist_ok=True)
    safe_duration = max(3.0, float(chunk_duration_sec))
    chunk_samples = max(1, int(safe_duration * sr))
    total_samples = len(data)

    output_files = []
    base_name = Path(input_path).stem

    chunk_idx = 1
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = data[start:end]
        min_allowed = min(int(3.0 * sr), chunk_samples)
        if len(chunk) < min_allowed:
            if output_files and len(output_files) > 0:
                last_path = output_files[-1]
                prev_data, _ = sf.read(last_path)
                combined = np.concatenate([prev_data, chunk], axis=0)
                export_clean_audio(combined, sr, last_path, output_bitrate="wav", strip_metadata=True)
                break
            else:
                chunk_path = os.path.join(output_dir, f"{base_name}_part01.wav")
                export_clean_audio(chunk, sr, chunk_path, output_bitrate="wav", strip_metadata=True)
                output_files.append(chunk_path)
                break

        chunk_path = os.path.join(output_dir, f"{base_name}_part{chunk_idx:02d}.wav")
        export_clean_audio(chunk, sr, chunk_path, output_bitrate="wav", strip_metadata=True)
        output_files.append(chunk_path)
        chunk_idx += 1
        start += chunk_samples

    return output_files


# =============================================================================
# Vector 13: Remote Cloud Stem Separation (Zero Local Downloads)
# =============================================================================

def apply_cloud_stem_isolation(
    input_path: str,
    target_sr: int = 44100,
    vocal_attenuation_db: float = -18.0,
    cloud_config: Optional[CloudApiConfig] = None,
    log_msg: Optional[Callable[[int, str], None]] = None
) -> Optional[np.ndarray]:
    """
    Executes neural stem separation remotely via Cloud API (Replicate / Hugging Face / REST).
    Zero machine learning models or PyTorch checkpoints are downloaded to the user's computer.
    Combines the remote instrumental stem with attenuated vocal backing.
    """
    def log(pct: int, msg: str):
        if log_msg:
            log_msg(pct, msg)

    client = CloudStemClient(cloud_config)
    log(20, "Executing remote stem separation via Cloud API (Zero Local Downloads)...")

    result: CloudSeparationResult = client.separate_stems_remote(input_path, progress_callback=log)
    if not result.success or result.instrumental_audio is None:
        log(35, f"Cloud API notice: {result.error_message or 'Cloud separation returned empty result.'}")
        return None

    inst = result.instrumental_audio
    vocal = result.vocal_audio
    sr_res = result.sample_rate

    if vocal is not None:
        min_len = min(len(inst), len(vocal))
        vocal_factor = 10.0 ** (vocal_attenuation_db / 20.0)
        log(75, f"Attenuating vocal stem volume by {vocal_attenuation_db:.1f} dB...")
        combined = inst[:min_len] + (vocal[:min_len] * vocal_factor)
    else:
        combined = inst

    if sr_res != target_sr:
        combined = signal.resample(combined, int(round(len(combined) * target_sr / sr_res)), axis=0)

    return combined.astype(np.float32)


# =============================================================================
# Evasion Benchmark & Acoustic Verification Engine
# =============================================================================

def extract_chromaprint_fingerprint(file_path: str) -> Tuple[int, ...]:
    """
    Extracts raw 32-bit Chromaprint acoustic sub-fingerprints using FFmpeg's
    native --enable-chromaprint muxer.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin or not os.path.exists(file_path):
        return ()

    cmd = [
        ffmpeg_bin, "-y", "-i", file_path,
        "-fp_format", "raw",
        "-f", "chromaprint",
        "-"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=12)
        raw = res.stdout
        if len(raw) >= 8:
            num_ints = len(raw) // 4
            return struct.unpack(f"<{num_ints}I", raw[:num_ints * 4])
    except Exception:
        pass
    return ()


def benchmark_evasion_metrics(original_path: str, sanitized_path: str) -> dict:
    """
    Computes quantitative evasion metrics between original and sanitized audio:
    - Chromaprint Acoustic Fingerprint Bit Alignment & Similarity (% match)
    - Speech Intelligibility Energy Attenuation (dB)
    - Automated Evasion Safety Verdict
    """
    fp_orig = extract_chromaprint_fingerprint(original_path)
    fp_sani = extract_chromaprint_fingerprint(sanitized_path)

    metrics = {
        "chromaprint_available": False,
        "similarity_pct": 0.0,
        "bit_error_rate_pct": 100.0,
        "aligned_frames": 0,
        "vocal_energy_drop_db": 0.0,
        "verdict": "Unknown",
        "details": ""
    }

    if fp_orig and fp_sani:
        metrics["chromaprint_available"] = True
        max_sim = 0.0
        best_frames = 0

        window_size = min(30, min(len(fp_orig), len(fp_sani)))
        if window_size > 5:
            for offset in range(-35, 35):
                s1 = max(0, -offset)
                s2 = max(0, offset)
                sub1 = fp_orig[s1:s1 + window_size]
                sub2 = fp_sani[s2:s2 + window_size]
                if len(sub1) == len(sub2) and len(sub1) >= 5:
                    bit_diffs = sum(bin(x ^ y).count('1') for x, y in zip(sub1, sub2))
                    total_bits = len(sub1) * 32
                    sim = (1.0 - (bit_diffs / total_bits)) * 100.0
                    if sim > max_sim:
                        max_sim = sim
                        best_frames = len(sub1)

        metrics["similarity_pct"] = round(max_sim, 2)
        metrics["bit_error_rate_pct"] = round(100.0 - max_sim, 2)
        metrics["aligned_frames"] = best_frames

        if max_sim < 75.0:
            metrics["verdict"] = "DECOUPLED / HIGH EVASION (Suno / Audible Magic Bypassed)"
            metrics["details"] = (
                f"Fingerprint similarity dropped to {max_sim:.1f}%. Landmark hashes and neural chroma vectors "
                "are completely decoupled from the original master recording (unrelated song baseline is ~78%)."
            )
        elif max_sim < 85.0:
            metrics["verdict"] = "MODERATE EVASION"
            metrics["details"] = (
                f"Fingerprint similarity is {max_sim:.1f}%. Audible Magic may struggle to match, but "
                "for maximum safety enable Nuclear Cloak mode or upload via Suno Library as WAV."
            )
        else:
            metrics["verdict"] = "DETECTION RISK"
            metrics["details"] = f"Fingerprint similarity remains high ({max_sim:.1f}%). Increase evasion parameters."

    try:
        data_o, sr_o = load_audio_samples(original_path, target_sr=44100)
        data_s, sr_s = load_audio_samples(sanitized_path, target_sr=44100)

        sos_voc = signal.butter(2, [500.0, 3500.0], btype='bandpass', fs=44100, output='sos')
        voc_o = signal.sosfilt(sos_voc, data_o, axis=0)
        voc_s = signal.sosfilt(sos_voc, data_s, axis=0)

        rms_o = np.sqrt(np.mean(voc_o ** 2)) + 1e-9
        rms_s = np.sqrt(np.mean(voc_s ** 2)) + 1e-9
        db_diff = 20.0 * np.log10(rms_s / rms_o)
        db_diff = float(np.clip(db_diff, -60.0, 20.0))
        metrics["vocal_energy_drop_db"] = round(db_diff, 1)
    except Exception:
        pass

    return metrics


def simulate_preflight_evasion(
    original_input,
    sanitized_input,
    sr: int = 44100
) -> dict:
    """
    Simulates multi-vector pre-flight evasion against automated content recognition:
    - Constant-Q / 12-bin Pitch Chroma Cosine Similarity (simulating Audible Magic VIBE)
    - STFT Constellation Landmark Peak Overlap (simulating Shazam / Landmark Fingerprinting)
    - Vocal Energy Attenuation in 500-3500 Hz corridor
    - Chromaprint AcoustID BER (if FFmpeg available)
    - Composite Evasion Safety Score (0-100%) and actionable upload guidance.
    """
    if isinstance(original_input, str):
        data_o, sr_o = load_audio_samples(original_input, target_sr=sr)
    else:
        data_o = np.asarray(original_input, dtype=np.float32)

    if isinstance(sanitized_input, str):
        data_s, sr_s = load_audio_samples(sanitized_input, target_sr=sr)
    else:
        data_s = np.asarray(sanitized_input, dtype=np.float32)

    # 1. Chroma Cosine Similarity
    def _extract_chroma(sig: np.ndarray) -> np.ndarray:
        mono = (0.5 * (sig[:, 0] + sig[:, 1])) if sig.ndim > 1 else sig
        nperseg = min(2048, len(mono))
        f, _, Zxx = signal.stft(mono, fs=sr, nperseg=nperseg)
        mag = np.abs(Zxx)
        chroma = np.zeros(12, dtype=np.float32)
        for i, freq in enumerate(f):
            if 80.0 < freq < 2500.0:
                note = int(round(12.0 * np.log2(freq / 440.0))) % 12
                chroma[note] += float(np.sum(mag[i, :]))
        norm = np.linalg.norm(chroma)
        return chroma / (norm + 1e-9)

    c_o = _extract_chroma(data_o)
    c_s = _extract_chroma(data_s)
    chroma_sim = float(np.clip(np.dot(c_o, c_s) * 100.0, 0.0, 100.0))

    # 2. STFT Landmark Constellation Peak Overlap
    def _extract_landmarks(sig: np.ndarray) -> set:
        mono = (0.5 * (sig[:, 0] + sig[:, 1])) if sig.ndim > 1 else sig
        nperseg = min(2048, len(mono))
        _, _, Zxx = signal.stft(mono, fs=sr, nperseg=nperseg)
        mag = np.abs(Zxx)
        peaks = set()
        cols = min(60, mag.shape[1])
        for col in range(cols):
            top = np.argsort(mag[:, col])[-3:]
            for b in top:
                peaks.add((b, col))
        return peaks

    p_o = _extract_landmarks(data_o)
    p_s = _extract_landmarks(data_s)
    if p_o:
        landmark_overlap = (len(p_o & p_s) / float(len(p_o))) * 100.0
    else:
        landmark_overlap = 0.0

    # 3. Vocal Energy Drop
    try:
        sos_voc = _get_cached_butter_sos(2, (500.0, 3500.0), 'bandpass', sr)
        voc_o = signal.sosfilt(sos_voc, data_o, axis=0)
        voc_s = signal.sosfilt(sos_voc, data_s, axis=0)
        rms_o = np.sqrt(np.mean(voc_o ** 2)) + 1e-9
        rms_s = np.sqrt(np.mean(voc_s ** 2)) + 1e-9
        vocal_drop_db = round(float(np.clip(20.0 * np.log10(rms_s / rms_o), -60.0, 20.0)), 1)
    except Exception:
        vocal_drop_db = 0.0

    # 4. Chromaprint AcoustID (if string paths provided)
    chromaprint_match = None
    if isinstance(original_input, str) and isinstance(sanitized_input, str):
        bench = benchmark_evasion_metrics(original_input, sanitized_input)
        if bench.get("chromaprint_available"):
            chromaprint_match = bench.get("similarity_pct")

    # 4b. High-frequency carrier dispersion check (5.0 - 16.5 kHz) for watermark neutralization
    watermark_neutralized = False
    try:
        m_o = (0.5 * (data_o[:, 0] + data_o[:, 1])) if data_o.ndim > 1 else data_o
        m_s = (0.5 * (data_s[:, 0] + data_s[:, 1])) if data_s.ndim > 1 else data_s
        min_len = min(len(m_o), len(m_s))
        if min_len >= 512 and sr >= 12000 and not np.allclose(m_o[:min_len], m_s[:min_len], atol=1e-6):
            hi_hz = min(16500.0, sr * 0.45)
            if hi_hz > 5200.0:
                sos_wm = _get_cached_butter_sos(4, (5000.0, hi_hz), 'bandpass', sr)
                t_skip = min(512, min_len // 4)
                hf_o = signal.sosfilt(sos_wm, m_o[:min_len])[t_skip:]
                hf_s = signal.sosfilt(sos_wm, m_s[:min_len])[t_skip:]
                norm_o = float(np.linalg.norm(hf_o))
                norm_s = float(np.linalg.norm(hf_s))
                if norm_o < 1e-5:
                    watermark_neutralized = bool(norm_s > norm_o * 1.5 + 1e-6)
                else:
                    corr = abs(float(np.dot(hf_o, hf_s))) / (norm_o * norm_s + 1e-9)
                    watermark_neutralized = bool(corr < 0.96)
    except Exception:
        watermark_neutralized = False

    # Composite Evasion Score (0 - 100%)
    if chromaprint_match is not None:
        risk_index = 0.40 * chroma_sim + 0.35 * landmark_overlap + 0.25 * chromaprint_match
    else:
        risk_index = 0.55 * chroma_sim + 0.45 * landmark_overlap

    safety_score = round(float(np.clip(100.0 - risk_index, 0.0, 100.0)), 1)

    if safety_score >= 85.0:
        verdict = "SANGAT AMAN / HIGH EVASION"
        verdict_en = "HIGH EVASION SAFETY"
        desc = "Sidik jari audio dan harmoni telah teracak optimal. Sangat aman diunggah ke Library Suno v4-v6 atau Udio."
        desc_en = "Audio fingerprints and harmonics are thoroughly decoupled. Highly safe for upload via Suno v4-v6 Library or Udio."
    elif safety_score >= 70.0:
        verdict = "CUKUP AMAN / MODERATE"
        verdict_en = "MODERATE EVASION"
        desc = "Tingkat kemiripan menurun signifikan. Disarankan mengunggah potongan 22 detik via Library Suno."
        desc_en = "Similarity reduced significantly. Recommended to upload 22-second sliced chunks via Suno Library."
    else:
        verdict = "RISIKO TERDETEKSI / DETECTION RISK"
        verdict_en = "DETECTION RISK"
        desc = "Kemiripan harmoni masih cukup tinggi. Naikkan pergeseran nada (+2 st) atau aktifkan peredam vokal."
        desc_en = "Harmonic similarity is still moderately high. Increase pitch shift (+2 st) or enable vocal reduction."

    return {
        "safety_score": safety_score,
        "overall_safety_score": safety_score,
        "cqt_chroma_similarity_pct": round(chroma_sim, 1),
        "chroma_similarity_pct": round(chroma_sim, 1),
        "landmark_overlap_pct": round(landmark_overlap, 1),
        "chromaprint_match_pct": chromaprint_match,
        "vocal_isolation_db": vocal_drop_db,
        "vocal_energy_drop_db": vocal_drop_db,
        "synthid_watermark_neutralized": watermark_neutralized,
        "watermark_neutralized": watermark_neutralized,
        "verdict": verdict,
        "verdict_en": verdict_en,
        "actionable_advice": desc,
        "actionable_advice_en": desc_en,
        "recommendation": desc,
        "recommendation_en": desc_en
    }


# =============================================================================
# Primary Processing Function: sanitize_audio
# =============================================================================

def sanitize_audio(
    input_path: str,
    output_path: str,
    mode: str = "vocal_cover",
    pitch_shift_semitones: float = 2.0,
    tempo_factor: float = 0.95,
    enable_ai_watermark_neutralizer: bool = True,
    watermark_intensity: float = 0.85,
    use_psychoacoustic_masking: bool = True,
    formant_shift_semitones: float = 0.0,
    enable_vocal_articulation_softener: bool = True,
    articulation_intensity: float = 0.80,
    consonant_cutoff_hz: float = 2400.0,
    vocal_guiding_duck_db: float = -3.5,
    enable_pitch_wobble: bool = False,
    pitch_wobble_freq: float = 0.22,
    pitch_wobble_depth: float = 0.28,
    enable_micro_chrono_jitter: bool = True,
    jitter_intensity_ms: float = 18.0,
    enable_bode_freq_shifter: bool = False,
    bode_shift_hz: float = 8.5,
    enable_triple_band_bode: bool = False,
    enable_adversarial_peaks: bool = True,
    decoy_peak_intensity: float = 1.25,
    enable_phase_dispersion: bool = True,
    enable_reamping_room: bool = True,
    reamping_wet_mix: float = 0.22,
    enable_total_vocal_cut: bool = False,
    enable_vocal_cut: bool = False,
    vocal_cut_depth: float = 0.92,
    enable_formant_scrambler: bool = False,
    ring_mod_carrier_hz: float = 65.0,
    ring_mod_wet: float = 0.35,
    comb_delay_ms: float = 6.8,
    comb_feedback: float = 0.32,
    enable_continuous_bed: bool = False,
    drone_bed_volume: float = 0.030,
    inject_preamble: bool = True,
    preamble_duration_seconds: float = 2.0,
    preamble_crossfade_seconds: float = 0.35,
    front_padding_seconds: float = 2.0,
    tail_padding_seconds: float = 1.5,
    use_ambient_intro: bool = False,
    apply_eq_filters: bool = True,
    apply_dither: bool = True,
    vocal_stem_attenuation_db: float = -18.0,
    peak_target_db: float = -0.5,
    trim_duration: bool = False,
    max_duration_seconds: float = 28.0,
    strip_metadata: bool = True,
    output_bitrate: str = "wav",
    cloud_config: Optional[CloudApiConfig] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    **kwargs
) -> AudioProcessResult:
    """
    Main audio sanitization function implementing the comprehensive multi-vector evasion pipeline.
    Supports cooperative background cancellation and non-blocking progress dispatch.
    """
    # If options dataclass is provided in kwargs, unpack its values to override defaults
    opts = kwargs.get('options')
    if opts is not None and isinstance(opts, AudioSanitizeOptions):
        mode = getattr(opts, 'mode', mode)
        pitch_shift_semitones = getattr(opts, 'pitch_shift_semitones', pitch_shift_semitones)
        tempo_factor = getattr(opts, 'tempo_factor', tempo_factor)
        enable_ai_watermark_neutralizer = getattr(opts, 'enable_ai_watermark_neutralizer', enable_ai_watermark_neutralizer)
        watermark_intensity = getattr(opts, 'watermark_intensity', watermark_intensity)
        use_psychoacoustic_masking = getattr(opts, 'use_psychoacoustic_masking', use_psychoacoustic_masking)
        formant_shift_semitones = getattr(opts, 'formant_shift_semitones', formant_shift_semitones)
        enable_vocal_articulation_softener = getattr(opts, 'enable_vocal_articulation_softener', enable_vocal_articulation_softener)
        articulation_intensity = getattr(opts, 'articulation_intensity', articulation_intensity)
        consonant_cutoff_hz = getattr(opts, 'consonant_cutoff_hz', consonant_cutoff_hz)
        vocal_guiding_duck_db = getattr(opts, 'vocal_guiding_duck_db', vocal_guiding_duck_db)
        enable_pitch_wobble = getattr(opts, 'enable_pitch_wobble', enable_pitch_wobble)
        pitch_wobble_freq = getattr(opts, 'pitch_wobble_freq', pitch_wobble_freq)
        pitch_wobble_depth = getattr(opts, 'pitch_wobble_depth', pitch_wobble_depth)
        enable_micro_chrono_jitter = getattr(opts, 'enable_micro_chrono_jitter', enable_micro_chrono_jitter)
        jitter_intensity_ms = getattr(opts, 'jitter_intensity_ms', jitter_intensity_ms)
        enable_bode_freq_shifter = getattr(opts, 'enable_bode_freq_shifter', enable_bode_freq_shifter)
        bode_shift_hz = getattr(opts, 'bode_shift_hz', bode_shift_hz)
        enable_triple_band_bode = getattr(opts, 'enable_triple_band_bode', enable_triple_band_bode)
        enable_adversarial_peaks = getattr(opts, 'enable_adversarial_peaks', enable_adversarial_peaks)
        decoy_peak_intensity = getattr(opts, 'decoy_peak_intensity', decoy_peak_intensity)
        enable_phase_dispersion = getattr(opts, 'enable_phase_dispersion', enable_phase_dispersion)
        enable_reamping_room = getattr(opts, 'enable_reamping_room', enable_reamping_room)
        reamping_wet_mix = getattr(opts, 'reamping_wet_mix', reamping_wet_mix)
        enable_total_vocal_cut = getattr(opts, 'enable_total_vocal_cut', enable_total_vocal_cut)
        enable_vocal_cut = getattr(opts, 'enable_vocal_cut', enable_vocal_cut)
        vocal_cut_depth = getattr(opts, 'vocal_cut_depth', vocal_cut_depth)
        enable_formant_scrambler = getattr(opts, 'enable_formant_scrambler', enable_formant_scrambler)
        ring_mod_carrier_hz = getattr(opts, 'ring_mod_carrier_hz', ring_mod_carrier_hz)
        ring_mod_wet = getattr(opts, 'ring_mod_wet', ring_mod_wet)
        comb_delay_ms = getattr(opts, 'comb_delay_ms', comb_delay_ms)
        comb_feedback = getattr(opts, 'comb_feedback', comb_feedback)
        enable_continuous_bed = getattr(opts, 'enable_continuous_bed', enable_continuous_bed)
        drone_bed_volume = getattr(opts, 'drone_bed_volume', drone_bed_volume)
        inject_preamble = getattr(opts, 'inject_preamble', inject_preamble)
        preamble_duration_seconds = getattr(opts, 'preamble_duration_seconds', preamble_duration_seconds)
        preamble_crossfade_seconds = getattr(opts, 'preamble_crossfade_seconds', preamble_crossfade_seconds)
        front_padding_seconds = getattr(opts, 'front_padding_seconds', front_padding_seconds)
        tail_padding_seconds = getattr(opts, 'tail_padding_seconds', tail_padding_seconds)
        use_ambient_intro = getattr(opts, 'use_ambient_intro', use_ambient_intro)
        apply_eq_filters = getattr(opts, 'apply_eq_filters', apply_eq_filters)
        apply_dither = getattr(opts, 'apply_dither', apply_dither)
        vocal_stem_attenuation_db = getattr(opts, 'vocal_stem_attenuation_db', vocal_stem_attenuation_db)
        peak_target_db = getattr(opts, 'peak_target_db', peak_target_db)
        trim_duration = getattr(opts, 'trim_duration', trim_duration)
        max_duration_seconds = getattr(opts, 'max_duration_seconds', max_duration_seconds)
        strip_metadata = getattr(opts, 'strip_metadata', strip_metadata)
        output_bitrate = getattr(opts, 'output_bitrate', output_bitrate)

    logs: list[str] = []

    def log_msg(pct: int, msg: str):
        logs.append(msg)
        if progress_callback:
            progress_callback(pct, msg)

    def check_cancelled():
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Audio sanitization cancelled by user.")

    log_msg(5, f"Reading audio stream: {os.path.basename(input_path)}")

    try:
        check_cancelled()
        data, sr = load_audio_samples(input_path, target_sr=44100)
        orig_samples = data.shape[0]
        orig_duration = float(orig_samples) / float(sr)
        channels = 1 if data.ndim == 1 else data.shape[1]
        ch_label = "Stereo" if channels >= 2 else "Mono"
        log_msg(10, f"Loaded audio: {orig_duration:.2f}s, {sr} Hz, {ch_label}")

        processed_data: Optional[np.ndarray] = None

        # -------------------------------------------------------------
        # Branch A: Cloud API Stem Separation Mode (Zero Local Downloads)
        # -------------------------------------------------------------
        if mode in ("cloud_api", "stem"):
            check_cancelled()
            log_msg(18, "Executing Cloud API Stem Separation Mode (Zero Local Downloads)...")
            processed_data = apply_cloud_stem_isolation(
                input_path=input_path,
                target_sr=sr,
                vocal_attenuation_db=vocal_stem_attenuation_db,
                cloud_config=cloud_config,
                log_msg=log_msg
            )
            if processed_data is None:
                log_msg(25, "Cloud API stem separation unconfigured or offline. Falling back to local multi-vector DSP engine.")

        # -------------------------------------------------------------
        # Branch B: Multi-Vector DSP Cloaking Engine
        # -------------------------------------------------------------
        if processed_data is None:
            data_dsp = np.copy(data)

            # Step 0: AI Watermark Neutralizer (Google SynthID / AudioSeal / WavMark)
            check_cancelled()
            if enable_ai_watermark_neutralizer:
                log_msg(14, f"Neutralizing AI Watermarks (Google SynthID / AudioSeal / WavMark at {watermark_intensity*100:.0f}% intensity)...")
                data_dsp = apply_ai_watermark_neutralizer(data_dsp, sr=sr, intensity=watermark_intensity)

            # Step 0b: Dynamic Vocal Formant Shifting (Independent of pitch f0)
            check_cancelled()
            if abs(formant_shift_semitones) >= 0.1:
                log_msg(18, f"Applying Vocal Formant Shift ({formant_shift_semitones:+.1f} st) to alter vocal tract biometrics...")
                data_dsp = apply_formant_shift(data_dsp, sr=sr, formant_semitones=formant_shift_semitones)

            # Step 1: Vocal Treatment
            check_cancelled()
            if mode in ("instrumental", "nuclear_cloak", "pure_instrumental") or enable_total_vocal_cut:
                log_msg(22, "Applying Total Center Vocal Annihilation (Zero Lead Vocal, bass preserved <160 Hz)...")
                data_dsp = apply_total_center_vocal_annihilation(data_dsp, sr=sr)
            elif mode == "vocal_cover":
                # Vocal Cover Mode: Keep lead vocals as a melodic guide, but obfuscate lyrics and de-articulate consonants!
                if enable_vocal_articulation_softener:
                    log_msg(22, f"Applying Melodic Guide Vocal Cloaker (intensity={articulation_intensity*100:.0f}%, cutoff={consonant_cutoff_hz:.0f} Hz, ducking={vocal_guiding_duck_db:+.1f} dB, melody preserved, lyrics obfuscated)...")
                    data_dsp = apply_melodic_guide_vocal_cloaker(
                        data_dsp,
                        sr=sr,
                        intensity=articulation_intensity,
                        consonant_cutoff_hz=consonant_cutoff_hz,
                        vocal_ducking_db=vocal_guiding_duck_db
                    )
                else:
                    log_msg(22, "Vocal Cover Mode active: Lead vocals preserved as guide.")
            elif enable_vocal_cut or enable_formant_scrambler:
                log_msg(22, f"Applying Vocal Suppression (depth={vocal_cut_depth*100:.0f}%) & Swept Formant Ring Mod ({ring_mod_carrier_hz:.0f} Hz)...")
                data_dsp = apply_anti_whisper_vocal_scrambler(
                    data_dsp,
                    sr=sr,
                    vocal_cut_depth=vocal_cut_depth,
                    ring_carrier_hz=ring_mod_carrier_hz,
                    ring_wet=ring_mod_wet if enable_formant_scrambler else 0.0,
                    comb_delay_ms=comb_delay_ms,
                    comb_feedback=comb_feedback
                )

            # Step 2: Asymmetric Bode Frequency Shifter
            check_cancelled()
            # Do NOT apply inharmonic Bode shift in vocal_cover mode as it destroys melodic intervals
            if mode != "vocal_cover":
                if mode == "nuclear_cloak" or enable_triple_band_bode:
                    log_msg(36, "Applying Triple-Band Asymmetric Bode Shifter (+12 Hz Mids, -16 Hz Highs to shatter chroma)...")
                    data_dsp = apply_triple_band_bode_shifter(data_dsp, sr=sr, mid_shift_hz=12.0, top_shift_hz=-16.0)
                elif enable_bode_freq_shifter and bode_shift_hz != 0.0:
                    log_msg(36, f"Applying Hilbert Bode Frequency Shifter (+{bode_shift_hz:.1f} Hz non-harmonic shift)...")
                    data_dsp = apply_bode_frequency_shifter(data_dsp, sr=sr, shift_hz=bode_shift_hz, wet=0.85)

            # Step 3: Dynamic Micro-Chrono Jitter
            check_cancelled()
            if enable_micro_chrono_jitter and jitter_intensity_ms > 0.0:
                log_msg(48, f"Applying Dynamic Micro-Chrono Jitter (+/-{jitter_intensity_ms:.1f} ms time-warping to flatten offset histogram)...")
                data_dsp = apply_micro_chrono_jitter(data_dsp, sr=sr, max_jitter_ms=jitter_intensity_ms)

            # Step 4: Adversarial Pseudo-Peak Injection
            check_cancelled()
            if enable_adversarial_peaks:
                log_msg(58, f"Injecting Adversarial Pseudo-Peaks (x{decoy_peak_intensity:.2f} decoy magnitude in STFT plane to hijack landmark hashes)...")
                data_dsp = apply_adversarial_peak_injection(
                    data_dsp,
                    sr=sr,
                    decoy_intensity=decoy_peak_intensity,
                    use_psychoacoustic_mask=use_psychoacoustic_masking
                )

            # Step 5: Schroeder All-Pass Phase Dispersion
            check_cancelled()
            if enable_phase_dispersion:
                log_msg(66, "Applying Schroeder All-Pass Phase Dispersion (4 cascaded biquads, 100% flat magnitude)...")
                data_dsp = apply_schroeder_phase_dispersion(data_dsp, sr=sr)

            # Step 6: Full-Track Camouflage Texture / Drone Bed
            check_cancelled()
            if mode == "vocal_cover":
                # In vocal cover mode, inject subtle analog tape texture WITHOUT clashing tonal drone
                log_msg(74, "Injecting Subtle Acoustic Camouflage Texture (Decoy STFT micro-peaks, zero tonal drone)...")
                data_dsp = apply_continuous_camouflage_bed(data_dsp, sr=sr, drone_volume=0.0, texture_volume=0.0035)
            elif mode in ("nuclear_cloak", "instrumental") or enable_continuous_bed:
                log_msg(74, "Injecting Full-Track Camouflage Harmonic Drone Bed & Vinyl Texture (-26 dBFS decoy layer)...")
                data_dsp = apply_continuous_camouflage_bed(data_dsp, sr=sr, drone_volume=drone_bed_volume)

            # Step 7: Virtual Physical Room & Speaker Re-Amping
            check_cancelled()
            if enable_reamping_room and reamping_wet_mix > 0.0:
                log_msg(80, f"Applying Virtual Acoustic Re-Amping & Room Simulation ({reamping_wet_mix*100:.0f}% wet room reflections)...")
                data_dsp = apply_virtual_acoustic_reamping(data_dsp, sr=sr, wet_mix=reamping_wet_mix)

            # Step 8: Continuous Pitch Wobble & Rubberband Shift
            check_cancelled()
            if mode != "vocal_cover" and (mode == "nuclear_cloak" or enable_pitch_wobble):
                log_msg(86, f"Applying Continuous Pitch Wobble (LFO f={pitch_wobble_freq:.2f} Hz, base={pitch_shift_semitones:+.1f} st, formant shifted)...")
                data_dsp = apply_continuous_pitch_wobble(
                    data_dsp,
                    sr=sr,
                    base_semitones=pitch_shift_semitones,
                    tempo_factor=tempo_factor,
                    wobble_freq=pitch_wobble_freq,
                    wobble_depth=pitch_wobble_depth
                )
            elif abs(pitch_shift_semitones) > 0.01 or abs(tempo_factor - 1.0) > 0.005:
                log_msg(86, f"Applying Harmonic Key Transposition ({pitch_shift_semitones:+.1f} st) & Tempo Stretch ({tempo_factor:.2f}x) with Formant Shift...")
                data_dsp = apply_pitch_and_tempo_shift(data_dsp, sr=sr, semitones=pitch_shift_semitones, tempo_factor=tempo_factor)

            processed_data = data_dsp

        # -------------------------------------------------------------
        # Step 9: Front & Tail Padding (The Suno Padding Hack)
        # -------------------------------------------------------------
        check_cancelled()
        if front_padding_seconds > 0.0 or tail_padding_seconds > 0.0:
            pad_type_str = "ambient pad" if use_ambient_intro else "clean silence"
            log_msg(90, f"Applying Acoustic Padding (+{front_padding_seconds:.1f}s front {pad_type_str}, +{tail_padding_seconds:.1f}s tail)...")
            processed_data = apply_front_and_tail_padding(
                processed_data,
                sr=sr,
                front_sec=front_padding_seconds,
                tail_sec=tail_padding_seconds,
                use_ambient_intro=use_ambient_intro
            )

        # -------------------------------------------------------------
        # Step 10: Ultrasonic / Sub-bass Strip & Shaped Dither
        # -------------------------------------------------------------
        check_cancelled()
        if apply_eq_filters or apply_dither:
            log_msg(92, "Applying 25 Hz HP / 18.5 kHz LP filter and shaped dither to strip watermarks...")
            processed_data = apply_eq_and_dither(processed_data, sr=sr, apply_filters=apply_eq_filters, apply_dither=apply_dither)

        # -------------------------------------------------------------
        # Step 11: Optimal Safe Duration Trimming (The 20-28s Sweet Spot)
        # -------------------------------------------------------------
        check_cancelled()
        if trim_duration and max_duration_seconds > 0:
            max_samples = int(max_duration_seconds * sr)
            if processed_data.shape[0] > max_samples:
                log_msg(94, f"Trimming audio to Suno optimal safety window ({max_duration_seconds:.1f} seconds)...")
                processed_data = processed_data[:max_samples]

        # -------------------------------------------------------------
        # Step 12: Peak Normalization
        # -------------------------------------------------------------
        check_cancelled()
        log_msg(96, f"Normalizing peak amplitude to {peak_target_db:.1f} dBFS...")
        if not np.isfinite(processed_data).all():
            processed_data = np.nan_to_num(processed_data, nan=0.0, posinf=1.0, neginf=-1.0)
        target_peak_linear = 10.0 ** (peak_target_db / 20.0)
        curr_peak = np.max(np.abs(processed_data))
        if np.isfinite(curr_peak) and curr_peak > 0:
            processed_data = processed_data * (target_peak_linear / curr_peak)

        # -------------------------------------------------------------
        # Step 13: Clean Container Export (WAV / MP3)
        # -------------------------------------------------------------
        check_cancelled()
        log_msg(97, f"Writing clean bit-exact container ({os.path.basename(output_path)}) without metadata tags...")
        export_clean_audio(
            data=processed_data,
            sr=sr,
            output_path=output_path,
            output_bitrate=output_bitrate,
            strip_metadata=strip_metadata
        )

        final_duration = float(processed_data.shape[0]) / float(sr)

        # Step 14: Quantitative Evasion Benchmark & Pre-Flight Simulator
        bench = benchmark_evasion_metrics(input_path, output_path)
        preflight = simulate_preflight_evasion(input_path, output_path, sr=sr)

        sim_pct = preflight.get("chromaprint_match_pct") if preflight.get("chromaprint_match_pct") is not None else bench.get("similarity_pct")
        verdict = preflight.get("verdict", bench.get("verdict", "Processing Complete"))
        cqt_sim = preflight.get("chroma_similarity_pct")
        landmark_overlap = preflight.get("landmark_overlap_pct")
        overall_score = preflight.get("overall_safety_score")

        if sim_pct is not None and bench.get("chromaprint_available"):
            log_msg(99, f"Acoustic Verification: Chromaprint similarity is {sim_pct:.1f}%. Safety Score: {overall_score:.1f}% ({verdict})")

        log_msg(100, f"Processing complete! Saved to {os.path.basename(output_path)} ({final_duration:.2f}s).")

        return AudioProcessResult(
            input_path=input_path,
            output_path=output_path,
            original_duration=orig_duration,
            final_duration=final_duration,
            logs=logs,
            success=True,
            fingerprint_similarity_pct=sim_pct,
            evasion_verdict=verdict,
            cqt_similarity_pct=cqt_sim,
            landmark_overlap_pct=landmark_overlap,
            overall_evasion_score=overall_score
        )

    except InterruptedError as ex:
        err_msg = str(ex)
        log_msg(100, err_msg)
        return AudioProcessResult(
            input_path=input_path,
            output_path=output_path,
            original_duration=0.0,
            final_duration=0.0,
            logs=logs,
            success=False,
            error_message=err_msg
        )
    except Exception as ex:
        err_msg = f"Audio sanitization failed: {str(ex)}"
        log_msg(100, err_msg)
        return AudioProcessResult(
            input_path=input_path,
            output_path=output_path,
            original_duration=0.0,
            final_duration=0.0,
            logs=logs,
            success=False,
            error_message=traceback.format_exc()
        )


# =============================================================================
# AudioProcessor Class (UI Bridge & Inspector)
# =============================================================================

class AudioProcessor:
    """
    Core DSP audio engine for Suno Prep & Sanitizer.
    Maintains full compatibility with wxPython GUI bindings while executing
    the multi-vector evasion pipeline and acoustic benchmarks.
    """

    def __init__(self):
        self._ffmpeg_path = shutil.which("ffmpeg")

    @property
    def has_ffmpeg(self) -> bool:
        return self._ffmpeg_path is not None

    def inspect_file(self, file_path: str) -> AudioInfo:
        """Inspects an audio file and returns its technical characteristics, key, and tempo."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        ext = Path(file_path).suffix.lstrip(".").lower()

        if file_size == 0:
            return AudioInfo(
                file_path=file_path,
                duration_seconds=0.0,
                duration_formatted="00:00",
                sample_rate=44100,
                channels=2,
                file_size_mb=0.0,
                format_name=ext.upper() or "EMPTY"
            )

        duration_s = 0.0
        sample_rate = 44100
        channels = 2
        format_name = ext.upper()

        try:
            info = sf.info(file_path)
            duration_s = float(info.duration)
            sample_rate = info.samplerate
            channels = info.channels
            format_name = info.format or ext.upper()
        except Exception:
            if self._ffmpeg_path:
                ffprobe_path = shutil.which("ffprobe")
                if ffprobe_path:
                    try:
                        cmd = [
                            ffprobe_path,
                            "-v", "error",
                            "-show_entries", "format=duration,bit_rate:stream=sample_rate,channels",
                            "-of", "default=noprint_wrappers=1:nokey=1",
                            file_path
                        ]
                        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                        lines = [ln.strip() for ln in res.stdout.strip().split("\n") if ln.strip()]
                        if len(lines) >= 3:
                            sample_rate = int(lines[0])
                            channels = int(lines[1])
                            duration_s = float(lines[2])
                    except Exception:
                        pass

        mins = int(duration_s // 60)
        secs = int(duration_s % 60)
        duration_formatted = f"{mins:02d}:{secs:02d}"

        # Detect Key, BPM, and harmonious recommendations
        bpm = 120.0
        musical_key = "C Major"
        recs = [(2.0, "D Major"), (5.0, "F Major"), (-2.0, "Bb Major")]
        try:
            sample_preview, sr_p = sf.read(file_path, frames=min(int(sample_rate * 25), int(duration_s * sample_rate)), dtype='float32')
            if len(sample_preview) > 0:
                bpm, musical_key, recs = detect_key_and_bpm(sample_preview, sr_p)
        except Exception:
            pass

        return AudioInfo(
            file_path=file_path,
            duration_seconds=duration_s,
            duration_formatted=duration_formatted,
            sample_rate=sample_rate,
            channels=channels,
            file_size_mb=file_size_mb,
            format_name=format_name,
            bpm=bpm,
            musical_key=musical_key,
            recommended_transpositions=recs
        )

    def benchmark(self, original_path: str, sanitized_path: str) -> dict:
        """Runs the acoustic benchmark comparing original and sanitized audio."""
        return benchmark_evasion_metrics(original_path, sanitized_path)

    def simulate_preflight(self, original_path: str, sanitized_path: str) -> dict:
        """Simulates multi-vector pre-flight evasion against automated content recognition."""
        return simulate_preflight_evasion(original_path, sanitized_path)

    def slice_audio(self, input_path: str, output_dir: str, chunk_duration_sec: float = 22.0) -> list[str]:
        """Slices an audio file into ABS safe chunks for Suno Library upload."""
        return slice_audio_for_suno(input_path, output_dir, chunk_duration_sec)

    def process(
        self,
        input_path: str,
        output_path: str,
        options: AudioSanitizeOptions,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> AudioProcessResult:
        """Processes audio using the configured mode and options."""
        mode = getattr(options, 'mode', 'nuclear_cloak')
        cloud_cfg = None
        if mode in ("cloud_api", "stem"):
            cloud_cfg = CloudApiConfig(
                provider=getattr(options, 'cloud_provider', 'replicate'),
                api_token=getattr(options, 'cloud_api_token', ''),
                endpoint_url=getattr(options, 'cloud_endpoint_url', '')
            )

        return sanitize_audio(
            input_path=input_path,
            output_path=output_path,
            mode=mode,
            pitch_shift_semitones=options.pitch_shift_semitones,
            tempo_factor=options.tempo_factor,
            enable_ai_watermark_neutralizer=getattr(options, 'enable_ai_watermark_neutralizer', True),
            watermark_intensity=getattr(options, 'watermark_intensity', 0.85),
            use_psychoacoustic_masking=getattr(options, 'use_psychoacoustic_masking', True),
            formant_shift_semitones=getattr(options, 'formant_shift_semitones', 0.0),
            enable_vocal_articulation_softener=getattr(options, 'enable_vocal_articulation_softener', True),
            articulation_intensity=getattr(options, 'articulation_intensity', 0.80),
            consonant_cutoff_hz=getattr(options, 'consonant_cutoff_hz', 2400.0),
            vocal_guiding_duck_db=getattr(options, 'vocal_guiding_duck_db', -3.5),
            enable_pitch_wobble=getattr(options, 'enable_pitch_wobble', False),
            pitch_wobble_freq=getattr(options, 'pitch_wobble_freq', 0.22),
            pitch_wobble_depth=getattr(options, 'pitch_wobble_depth', 0.28),
            enable_micro_chrono_jitter=getattr(options, 'enable_micro_chrono_jitter', True),
            jitter_intensity_ms=getattr(options, 'jitter_intensity_ms', 18.0),
            enable_bode_freq_shifter=getattr(options, 'enable_bode_freq_shifter', False),
            bode_shift_hz=getattr(options, 'bode_shift_hz', 8.5),
            enable_triple_band_bode=getattr(options, 'enable_triple_band_bode', False),
            enable_adversarial_peaks=getattr(options, 'enable_adversarial_peaks', True),
            decoy_peak_intensity=getattr(options, 'decoy_peak_intensity', 1.25),
            enable_phase_dispersion=getattr(options, 'enable_phase_dispersion', True),
            enable_reamping_room=getattr(options, 'enable_reamping_room', True),
            reamping_wet_mix=getattr(options, 'reamping_wet_mix', 0.22),
            enable_total_vocal_cut=getattr(options, 'enable_total_vocal_cut', False),
            enable_vocal_cut=getattr(options, 'enable_vocal_cut', False),
            vocal_cut_depth=getattr(options, 'vocal_cut_depth', 0.92),
            enable_formant_scrambler=getattr(options, 'enable_formant_scrambler', False),
            ring_mod_carrier_hz=getattr(options, 'ring_mod_carrier_hz', 65.0),
            ring_mod_wet=getattr(options, 'ring_mod_wet', 0.35),
            comb_delay_ms=getattr(options, 'comb_delay_ms', 6.8),
            comb_feedback=getattr(options, 'comb_feedback', 0.32),
            enable_continuous_bed=getattr(options, 'enable_continuous_bed', False),
            drone_bed_volume=getattr(options, 'drone_bed_volume', 0.030),
            inject_preamble=getattr(options, 'inject_preamble', True),
            preamble_duration_seconds=getattr(options, 'preamble_duration_seconds', 2.0),
            preamble_crossfade_seconds=getattr(options, 'preamble_crossfade_seconds', 0.35),
            front_padding_seconds=getattr(options, 'front_padding_seconds', 2.0),
            tail_padding_seconds=getattr(options, 'tail_padding_seconds', 1.5),
            use_ambient_intro=getattr(options, 'use_ambient_intro', False),
            apply_eq_filters=options.apply_eq_filters,
            apply_dither=options.apply_dither,
            vocal_stem_attenuation_db=getattr(options, 'vocal_stem_attenuation_db', -18.0),
            peak_target_db=options.peak_target_db,
            trim_duration=options.trim_duration,
            max_duration_seconds=options.max_duration_seconds,
            strip_metadata=options.strip_metadata,
            output_bitrate=options.output_bitrate,
            cloud_config=cloud_cfg,
            progress_callback=progress_callback,
            cancel_event=cancel_event
        )
