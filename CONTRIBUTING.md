# Contributing to GhostWave Studio

Thank you for your interest in contributing to GhostWave Studio. This document outlines the technical standards, testing workflow, and submission guidelines for the project.

---

## Technical Scope and Project Mission

GhostWave Studio is an audio research project designed to study acoustic fingerprinting systems, neural audio embeddings, and automated speech recognition ingestion filters. Contributions should focus on signal processing fidelity, accessibility standards, test coverage, and security of user configuration data.

---

## Development Environment Setup

1. Clone the repository:
   ```bash
   git clone git@github-personal:muhamadalfian20892/GhostWave-Studio.git
   cd GhostWave-Studio
   ```

2. Create a clean Python virtual environment (Python 3.10 or higher):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install FFmpeg:
   Ensure FFmpeg and ffprobe are available in your system PATH. For Chromaprint audit capabilities, FFmpeg must be built with libchromaprint enabled.

---

## Coding Standards and Architecture

* **Signal Processing (audio_processor.py):**
  Keep DSP operations vectorized through NumPy and SciPy. Avoid introducing local PyTorch checkpoint requirements. Heavy machine learning tasks such as stem separation should use remote REST endpoints or external APIs.

* **Lyrics Processing (lyrics_processor.py):**
  The primary cloaking strategy must preserve original lyrics tokens while perturbing orthography, syllable meter, and cadence endings. Ensure changes do not break singable phrasing.

* **Security and Storage (config_manager.py):**
  Never commit plaintext API keys or credentials. All configuration persistence must route through the encrypted .sn container using Fernet encryption with PBKDF2 key derivation.

* **Accessibility (gui.py):**
  Maintain accessibility standards for all wxPython controls. Every interactive element (buttons, text fields, checkboxes, sliders) requires explicit accessible names via `.SetName()` and helpful tooltips. Maintain keyboard navigation order with standard mnemonic accelerators.

---

## Testing Requirements

Every contribution must pass the full test suite before submission. Run tests using:

```bash
python -m unittest test_app.py
```

When adding new DSP filters or lyrics transforms, include corresponding unit test cases in `test_app.py` covering edge cases, empty inputs, and roundtrip serialization.

---

## Building Executables and Installer

To verify binary packaging locally:

```bash
# 1. Compile standalone binaries
pyinstaller ghostwave.spec --noconfirm --clean

# 2. Compile Windows installer
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

---

## Pull Request Guidelines

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write clean, readable code with descriptive commit messages.
3. Run all unit tests and verify clean exit status.
4. Open a pull request against the `main` branch with a concise explanation of changes made and verification steps performed.
