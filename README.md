# GhostWave Studio

[![CI](https://github.com/muhamadalfian20892/GhostWave-Studio/actions/workflows/ci.yml/badge.svg)](https://github.com/muhamadalfian20892/GhostWave-Studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Version](https://img.shields.io/badge/Version-1.7.0-046C4E)

GhostWave Studio is a screen-reader-first desktop application and CLI tool for audio signal processing, acoustic fingerprint testing, and lyrics preparation. Built in Python with wxPython Phoenix, SciPy, NumPy, soundfile, sounddevice, FFmpeg, and cryptography, it runs completely offline for local DSP while keeping optional cloud stem separation and lyrics rewriting behind remote APIs so users never have to download multi-gigabyte PyTorch weights.

I built GhostWave Studio so blind, low-vision, and sighted musicians can inspect how automated content recognition systems (Audible Magic, CQT chroma embeddings, neural audio watermarks, and Whisper speech filters) evaluate uploaded tracks, test countermeasures quantitatively, and prepare vocal covers or backing tracks without fighting inaccessible web tools.

## Why Accessibility Comes First

Most audio workstations and AI music utilities depend heavily on unlabeled canvas widgets, mouse-only sliders, and visual meters with no text fallback. GhostWave Studio treats accessibility as a primary engineering requirement:

- Native MSAA/IAccessible labels via `wx.Accessible` on every input field, compound `SpinCtrlDouble`, checkbox, choice box, and button so NVDA, JAWS, and Windows Narrator announce exact control names in English or Bahasa Indonesia.
- Real-time bilingual UI switching (`English` and `Bahasa Indonesia`) stored in an encrypted local `.sn` vault and applied without restarting.
- Scrollable panels (`ScrolledPanel`) and two-row action controls so high-DPI scaling and low-vision zoom never clip controls.
- Accessible waveform summary readout beneath the dual-waveform canvas reporting peak amplitude, RMS dB levels, and mapped acoustic delta peaks as plain text.
- High-contrast WCAG AA/AAA status banners in the Pre-Flight Evasion Simulator (`#046C4E`, `#92400E`, `#B91C1C` with `#FFFFFF` text) plus read-only scrollable dialogs for release notes (`F2`), keyboard shortcuts (`F1`), upload protocol (`Ctrl+H`), and support tickets (`Ctrl+T`).

## Core Features

### Audio DSP & Pre-Flight Analysis
- Tuned target platform presets for Suno v4-v6 Cover Mode, Google Lyria 3.5 (watermark neutralizer only), Udio 1.5/2.0, YouTube Content ID, TikTok/Reels, and Custom.
- Automatic musical key and BPM estimation on file load, paired with one-click harmonious transposition suggestions (`+2`, `+5`, or `-2` semitones).
- Modular DSP pipeline covering pitch shift (`-12` to `+12` st), independent vocal formant shift (`-6` to `+6` st), tempo factor (`0.85` to `1.15`), studio room re-amping, center vocal ducking or cut, AI watermark neutralization (SynthID/AudioSeal/WavMark phase dispersion), Terhardt ATH psychoacoustic decoy peak masking, micro-chrono jitter, and Schroeder all-pass phase dispersion.
- Dual-waveform visualizer plotting original vs sanitized signals with real envelope-delta peak markers, plus inline A/B/Delta audio audition buttons inside the main window.
- Pre-Flight Evasion Simulator (`Alt+F`) and Chromaprint Audit (`Alt+E`) measuring CQT chroma cosine similarity, STFT landmark constellation overlap, high-frequency carrier dispersion, vocal band attenuation, and Chromaprint BER before uploading.
- Batch Queue and background Folder Watcher (`Ctrl+Shift+B`) for processing whole directories with optional 22-second WAV slicing.

### Lyrics Studio
- Multilingual phonotactic syllabifier that splits English, Indonesian, Spanish, French, and other Latin-script lyrics into singable hyphenated syllables while leaving inline tags like `[Chorus]` untouched.
- Rhyme-preserving semantic cloaking (protects end-of-line rhyming words), phonetic spelling variation, hybrid ad-lib breaking, sound-alike substitution, and optional Cloud LLM rewriting (Groq, OpenRouter, OpenAI, or custom endpoints).
- Single-pass regex celebrity and profanity filtering, Unicode normalization, structure tag standardization, Suno/Udio style prompt generation, and 4-gram overlap auditing.

### Security, Updates & Support
- Encrypted local `.sn` vault storing API tokens, DSP settings, support ticket keys, and custom blacklists with Fernet (`AES-128-CBC` + `HMAC-SHA256`), `PBKDF2-HMAC-SHA256` key derivation (100,000 iterations), and `.bak` rotation.
- GitHub Releases update checker with offline fallback to root `changelog.txt` and inline release note reading (`F2`).
- Built-in Support Ticket Center (`Ctrl+T`) for submitting bug reports or feature requests and reading developer replies via a private Cloudflare Worker relay.

## Installation and Quick Start

### Option 1: Windows Installer or Portable Archive
Download the latest release from [GitHub Releases](https://github.com/muhamadalfian20892/GhostWave-Studio/releases):
- `GhostWaveStudio-v1.7.0-Setup.exe`: Installs desktop and CLI binaries, Start Menu shortcuts, optional `PATH` integration for `ghostwave-cli`, and uninstaller.
- `GhostWaveStudio-v1.7.0-Portable.zip`: Extract anywhere and run `GhostWaveStudio.exe` or `ghostwave-cli.exe` with no Python installation required.

### Option 2: Run from Python Source
Requires Python 3.10+ and FFmpeg in your system `PATH` (with `libchromaprint` enabled if you want Chromaprint CLI audits):

```bash
pip install -r requirements.txt
python main.py
```

### CLI Usage Examples

```bash
# Process an audio file with default preset to 16-bit PCM WAV
python main.py -i input.mp3 -o cloaked.wav --format wav --preset max

# Run Pre-Flight Evasion simulation comparing original and processed audio
python main.py -i input.mp3 -o cloaked.wav --preflight

# Slice a track into 22-second WAV segments
python main.py -i input.mp3 --slice --slice-duration 22.0 -o ./suno_chunks

# Batch process a directory or watch an inbox folder
python main.py --batch-dir ./raw_tracks -o ./processed_tracks --format wav
python main.py --watch-input ./inbox --watch-output ./outbox

# Sanitize lyrics with the multilingual phonetic scrambler
python main.py --lyrics lyrics.txt --lyrics-out cloaked.txt --lyrics-mode scramble
```

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Alt+1` / `Alt+2` / `Ctrl+Tab` | Switch between Audio Sanitizer and Lyrics Sanitizer tabs |
| `Alt+B` / `Ctrl+O` | Browse for an audio file |
| `Alt+P` / `Ctrl+P` | Process and export audio |
| `Alt+F` | Open Pre-Flight AI Evasion Simulator |
| `Alt+E` / `Ctrl+E` | Open Chromaprint Audio Similarity Audit |
| `Ctrl+Shift+B` | Open Batch Processing & Folder Watcher |
| `Ctrl+U` | Open ABS Audio Slicer |
| `Ctrl+B` | Edit Celebrity Blacklist |
| `Ctrl+Shift+S` / `Ctrl+Shift+O` | Export or import encrypted `.sn` profile |
| `Ctrl+T` | Open Support Ticket Center |
| `F1` / `F2` / `Shift+F1` | Accessibility Guide / Changelog / Suno Upload Guide |
| `Escape` | Close active dialog |

## Project Structure

```text
GhostWave-Studio/
├── main.py                  # Unified CLI and GUI entry point
├── ghostwave/               # Core application package
│   ├── version.py           # Single source of truth for APP_VERSION
│   ├── audio_processor.py   # Vectorized SciPy/NumPy DSP and pre-flight simulator
│   ├── lyrics_processor.py  # Multilingual syllabifier, cloaker, and style prompt builder
│   ├── batch_processor.py   # Batch queue manager and background folder watcher
│   ├── cloud_stem_api.py    # Optional remote stem separation (Replicate/HF/REST)
│   ├── config_manager.py    # Fernet encrypted .sn vault and .bak recovery
│   ├── gui.py               # Accessible wxPython Phoenix desktop interface
│   ├── i18n.py              # Bilingual runtime localization (EN / ID)
│   ├── support_client.py    # Cloudflare Worker support ticket client
│   └── updater.py           # GitHub Releases updater and changelog cache
├── locales/                 # Translation catalogs (en.json, id.json)
├── docs/                    # Plain-text accessible user guides (EN / ID)
├── packaging/               # PyInstaller spec and Inno Setup installer script
├── assets/                  # Application icon
├── tests/                   # Automated unit test suite
└── changelog.txt            # Release history (kept at root for in-app updater)
```

## Testing

Run the full test suite (77 unit tests covering DSP modules, Pre-Flight simulator, `.sn` encryption, lyrics syllabification, batch/folder watcher, CLI guards, i18n parity, contrast compliance, and NVDA accessible names):

```bash
python -m unittest discover -s tests -v
```

## Building Standalone Binaries

```bash
pyinstaller packaging/ghostwave.spec --noconfirm --clean
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
```

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup, accessibility rules, and pull request steps, and see [changelog.txt](changelog.txt) for full version history.

## Research and Educational Disclaimer

GhostWave Studio is an engineering and accessibility research project exploring acoustic fingerprinting robustness, neural audio embeddings, and automated speech recognition filters. Users are responsible for ensuring that their use complies with applicable laws and platform terms of service.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).

Copyright (c) 2026 Muhamad Alfian / Technokers Lab.
