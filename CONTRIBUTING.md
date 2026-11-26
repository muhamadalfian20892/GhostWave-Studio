# Contributing to GhostWave Studio

Thank you for your interest in contributing to GhostWave Studio. This document outlines the technical standards, testing workflow, and pull request steps for the project.

## Technical Scope and Project Mission

GhostWave Studio is an accessibility-focused audio and lyrics research toolkit built to study acoustic fingerprinting, neural audio embeddings, watermark dispersion, and speech recognition filters. Contributions should prioritize signal quality, screen-reader accessibility, test coverage, and local configuration security.

## Development Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/muhamadalfian20892/GhostWave-Studio.git
   cd GhostWave-Studio
   ```

2. Create a Python virtual environment (Python 3.10 or higher):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install FFmpeg:
   Make sure `ffmpeg` and `ffprobe` are available in your system `PATH`. For Chromaprint audit checks, FFmpeg must include `libchromaprint`.

## Coding Standards and Architecture

- Keep DSP operations in `ghostwave/audio_processor.py` vectorized in NumPy and SciPy. Avoid adding local PyTorch weights; remote stem separation belongs behind REST endpoints in `ghostwave/cloud_stem_api.py`.
- Keep original song meaning and singable syllable cadence intact in `ghostwave/lyrics_processor.py` while disrupting exact n-gram matches.
- Never commit plaintext API keys or personal data. All configuration persistence in `ghostwave/config_manager.py` must route through the `.sn` container using Fernet encryption with PBKDF2 key derivation.
- Give every interactive control in `ghostwave/gui.py` (buttons, inputs, checkboxes, spin controls, choices) an explicit accessible name via `make_accessible_control` or `.SetName()`, helpful tooltips, keyboard navigation, and WCAG AA contrast (`>= 4.5:1` normal text, `>= 3.0:1` large text).
- Keep 100% key parity between `locales/en.json` and `locales/id.json` with zero unescaped `\n` literals or hardcoded UI strings.

## Testing Requirements

Every contribution must pass the full unit test suite before opening a pull request:

```bash
python -m unittest discover -s tests -v
```

When adding DSP filters, lyrics transformations, or UI dialogs, add corresponding tests in `tests/test_app.py` covering edge cases, accessibility names, and roundtrip serialization.

## Building Executables and Installer

```bash
pyinstaller packaging/ghostwave.spec --noconfirm --clean
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
```

## Pull Request Guidelines

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Keep changes focused, write clear commit messages, and run the full test suite.
3. Open a pull request against `main` summarizing what changed and how you verified it.
