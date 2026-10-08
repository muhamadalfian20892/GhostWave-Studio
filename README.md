# GhostWave Studio v1.0: Stealth Audio Cloak and Fingerprint Evasion

GhostWave Studio is an accessibility-focused desktop application and command-line tool written in Python using wxPython Phoenix, SciPy, FFmpeg, and cryptography. I put this together to study how automated audio fingerprinting systems, specifically Audible Magic ACR, VIBE neural embeddings, and Whisper speech recognition, inspect audio uploads on platforms like Suno.

The project implements digital signal processing techniques designed to disrupt landmark peak hashes and neural chroma representations while keeping the audio usable for prompting or continuation. All user preferences, remote API tokens, DSP parameters, and custom blacklists are stored inside an encrypted .sn binary container file so secrets stay protected locally. Audio processing runs through local DSP algorithms or optional remote cloud APIs for stem isolation, which avoids downloading multi-gigabyte PyTorch checkpoints to the local machine.

---

### Installation and Distribution Options

GhostWave Studio is available as a compiled Windows installer, a portable zip archive, or directly from source.

#### Option 1: Windows Installer (Recommended)
Run the setup wizard located at:
`dist/installer/GhostWaveStudio-v1.0-Setup.exe`
- Installs to the user application directory
- Creates Desktop and Start Menu shortcuts
- Provides an optional checkbox to add `ghostwave-cli` to your user PATH
- Includes an uninstaller in Windows Settings and Control Panel

#### Option 2: Portable Archive (No Installation Required)
Extract `dist/GhostWaveStudio-v1.0-Portable.zip` to any folder:
- Run `GhostWaveStudio.exe` to launch the graphical interface
- Run `ghostwave-cli.exe` from PowerShell or Command Prompt for batch workflows
- Runs self-contained without needing Python on the host machine

#### Option 3: Run from Python Source
Requires Python 3.10+ and a local FFmpeg installation:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the desktop interface
python gui.py
# or:
python main.py

# 3. Process an audio file via CLI
python main.py -i input.mp3 -o cloaked.wav --format wav --preset max

# 4. Split an audio track into 22-second segments for library upload
python main.py -i input.mp3 --slice --slice-duration 22.0

# 5. Cloak a lyrics file using the acoustic spelling scrambler
python main.py --lyrics lyrics.txt --lyrics-out cloaked.txt --lyrics-mode scramble

# 6. Run the test suite (23 unit tests)
python -m unittest test_app.py
```

#### Rebuilding the Executables and Installer from Source

```bash
# Compile standalone binaries using PyInstaller
pyinstaller ghostwave.spec --noconfirm --clean

# Compile Windows setup installer using Inno Setup 6
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

---

## Development Changelog and Release Notes

New on 10/08/2026:

	Added native multilingual support to the Acoustic Spelling Scrambler for Indonesian, Spanish, French, and other Latin-script languages. The earlier dictionary was English-oriented, which caused non-English text to pass through without orthographic changes.

	Implemented the Universal Multilingual Phonotactic Syllabifier in lyrics_processor.py. The algorithm parses multi-syllable tokens into musical singing syllables using hyphens according to phonotactic boundary rules and language digraphs (ng, ny, kh, sy, ch, sh, th, ph, ll, rr, qu). In tests with Indonesian song lyrics, tokens like "menghapus jejakmu" become "meng-ha-pus je-jak-mu" and "kuasai diriku" become "ku-a-sa-i di-ri-ku". Suno synthesizes these with natural vocal delivery, while 4-gram database matching drops to 0.0%.

	Added over 200 Indonesian singing terms to the phonetic dictionary, including singable elisions such as "t'lah", "s'dah", "s'per-ti", "s'la-lu", "s'mu-a", and "kar'na".

	Added standalone binary distribution and Inno Setup installer packaging. GhostWave Studio can be installed via dist/installer/GhostWaveStudio-v1.0-Setup.exe or run portably from dist/GhostWaveStudio-v1.0-Portable.zip without requiring a local Python installation.

	Built a dual-binary PyInstaller specification (ghostwave.spec) that bundles both the accessible graphical application (GhostWaveStudio.exe) and the command-line utility (ghostwave-cli.exe) with shared runtime libraries.

	Generated a multi-resolution application icon (ghostwave.ico) featuring an acoustic waveform motif, embedded across both executables, the installer, and desktop shortcuts.

	Configured the Inno Setup 6 compiler script (installer.iss) with LZMA2 solid compression, modern wizard styling, automatic uninstaller registration, and an optional task to register ghostwave-cli in the user environment PATH.

	Added vocal vibrato glide markers (~) to cadence endings. Appending tildes to line endings breaks sentence matchers and regex filters while prompting Suno's vocal model to add melodic inflection.

	Added rule-based morphological scramblers that convert "-ing" endings into "[stem]-in'", past-tense "-ed" suffixes into "[stem]'d", and "-ght" clusters into "-ite".

	Updated syllable counting logic to parse hyphenated syllables directly, ensuring exact metric measurements across all languages.

	Maintained optional alternative modes for testing, including Stealth Hybrid, Phonetic Disguise, Semantic Cadence, and remote Cloud LLM rewriting.

	Added an automatic Lyrics Evasion Audit tool that calculates 4-gram sequence overlap, token similarity, and syllable cadence accuracy in the interface and CLI.

	Added CLI options (--lyrics, --lyrics-out, --lyrics-mode, --no-adlibs) for automated batch lyrics processing.

	Added unit tests in test_app.py covering Indonesian and Spanish lyrics cloaking, bringing the total suite to 23 unit tests. All tests pass.

New on 10/07/2026:

	Renamed the project to GhostWave Studio v1.0. The old temporary filenames were getting confusing across the codebase.

	Added the encrypted .sn configuration container format. The application now saves API credentials, blacklist entries, and DSP preferences using Fernet (AES-128-CBC with HMAC-SHA256) with keys derived through PBKDF2-HMAC-SHA256 across 100,000 iterations.

	The .sn container file uses a 16-byte magic header (GHOSTWAVE_SN_V1\x00). Opening the file in a plain text editor shows only binary ciphertext.

	Added HMAC integrity verification on load. If any byte in the configuration file is modified externally, the loader detects the signature mismatch and cleanly rejects the file.

	Added an automatic migration routine for older setups. Existing JSON configurations are imported into the user's default .sn vault at startup.

	Added Ctrl+Shift+S to export profiles and Ctrl+Shift+O to import them in the File menu.

	The CLI now accepts --config-sn to specify a custom vault path.

	Added 4 unit tests covering header validation, ciphertext verification, roundtrip serialization, and corrupted file handling. All 21 unit tests pass cleanly.

New on 10/05/2026:

	Investigated the upload rejection error from Suno regarding matching recordings. Testing confirmed that simple semitone pitch shifts and static vocal cuts remain detectable by Audible Magic's VIBE embeddings and landmark hashers.

	Added an asymmetric Hilbert Bode frequency shifter. Shifting mid frequencies by +12 Hz and high frequencies by -16 Hz via single-sideband analytic modulation alters harmonic spacing, which disrupts Constant-Q Transform chroma bins used in neural cover detection.

	Added micro-chrono jitter using dual out-of-phase low-frequency modulators to apply continuous non-linear time warping of ±16ms. This modulates the time deltas between spectral peaks to flatten time-offset histograms.

	Added a slow pitch wobble LFO at 0.22 Hz and 28% depth to prevent stationary chroma vectors from forming.

	Added an STFT pseudo-peak generator that places small decoy energy peaks at offset coordinates to draw peak extraction away from original landmarks.

	Added 4-stage Schroeder all-pass phase dispersion filters at 420 Hz, 1350 Hz, 2900 Hz, and 5400 Hz. These shift signal phase while maintaining flat unit gain across frequencies.

	Added an acoustic room impulse response convolution step to simulate physical speaker playback in a small studio room.

New on 10/03/2026:

	Added the audio slicer utility, accessible via Alt+S, Ctrl+U, or the --slice flag in the CLI.

	Tests show Audible Magic requires approximately 5 to 8 contiguous seconds of matching peak patterns. Dividing tracks into 20 to 24 second chunks disrupts long-term correlation windows.

	The slicer exports 16-bit PCM WAV files at 48 kHz with clean headers, omitting ID3 tags and encoder padding.

	Adjusted the center vocal cancellation crossover. Audio below 160 Hz is preserved in mono to keep low-end bass intact, while frequencies above 160 Hz undergo center subtraction.

	Added a low-level background drone and vinyl texture at -26 dBFS to introduce additional spectral energy across the timeline.

	Added 4.0 seconds of front padding and 2.0 seconds of tail padding to displace initial time offsets.

New on 10/01/2026:

	Added remote stem separation support via cloud REST endpoints (Replicate and Hugging Face). This keeps the local environment free from heavy PyTorch dependencies and checkpoint files.

	Added the Chromaprint audit dialog (Alt+E or Ctrl+E). It calculates bitwise Hamming distances and sliding-window cross-correlation against the original file to report estimated evasion rates and vocal band attenuation.

	Added the lyrics moderation tab with a configurable artist blacklist to catch names that could trigger Whisper transcription filters.

New on 09/28/2026:

	Updated accessibility across wxPython dialogs. Controls, sliders, and input fields now include explicit accessible names for screen readers such as NVDA, JAWS, and Windows Narrator.

	Added keyboard shortcuts across primary operations, including Alt+O for browsing, Alt+C for cloaking, Alt+P for playback, Alt+S for the slicer, Alt+E for audits, and Shift+F1 for guidelines.

	Fixed an issue where pressing Escape during playback preview left the audio playback thread running in the background. Windows now stop active playback on close.

	Implemented atomic file writes for exports and configuration updates to prevent truncated files if an operation is canceled.

	FFmpeg temporary files are now unlinked immediately after conversion completes or on failure.

---

## Operational Findings for Suno Uploads

When uploading audio to Suno, uploads are evaluated by automated filters. If you encounter the recording match notice, the following procedures address the primary detection vectors:

1. Export as 16-bit PCM WAV. MP3 files include encoder delay padding and container metadata that fingerprinting systems inspect during initial ingestion.
2. Upload through the Library page rather than the Create page. The Library interface processes files through background workers with different validation constraints. After processing, tracks can be extended via the track menu.
3. Keep segments between 18 and 24 seconds. Shorter segments limit the window Audible Magic has to identify continuous landmark sequences.
4. If an upload was recently rejected, clear site cookies or use a private browsing window to reset session tracking.
5. Apply center vocal reduction or cloud stem separation, and check lyrics for artist names before submission.

---

## Graphical User Interface Reference

Start the interface by running `python gui.py` or `python main.py`.

* Preset Selector:
  - Maximum Evasion: Activates all DSP vectors, sets output to 16-bit WAV, and configures the 24-second target duration.
  - Balanced Quality: Uses moderate micro-pitch jitter and mild phase dispersion for higher acoustic clarity.
  - Vocal Reduction Only: Applies center-channel cancellation without time warping or pitch shifting.
* ABS Audio Slicer (Alt+S or Ctrl+U): Slices tracks into 20 to 24 second WAV segments with configurable overlap.
* Acoustic Evasion Audit (Alt+E or Ctrl+E): Computes Chromaprint cross-correlation, bitwise Hamming distance, and vocal attenuation against the source file.
* Profile Export and Import (Ctrl+Shift+S and Ctrl+Shift+O): Saves or loads encrypted .sn configuration files.
* Cloud API Vault (Alt+A or Ctrl+K): Manages Replicate and Hugging Face tokens for remote stem processing.
* Guidelines Reference (Shift+F1 or Ctrl+H): Opens the operational upload reference window.

---

## Command-Line Interface Reference

The command-line interface provides the same processing capabilities for scripts and terminal use:

```bash
# Maximum evasion preset with WAV output
python main.py -i song.mp3 -o cloaked.wav --format wav --preset max

# Slice a track into 22-second segments
python main.py -i song.mp3 --slice --slice-duration 22.0 -o ./sliced_chunks

# Use a custom encrypted profile vault
python main.py -i song.mp3 -o cloaked.wav --config-sn "custom_profile.sn"

# Balanced preset with MP3 output
python main.py -i song.mp3 -o cloaked.mp3 --format mp3 --preset balanced

# Compare two existing audio files with Chromaprint
python main.py -i original.mp3 -o cloaked.wav --audit-only

# Cloak a lyrics file using the acoustic spelling scrambler (preserves words)
python main.py --lyrics song_lyrics.txt --lyrics-out cloaked_lyrics.txt --lyrics-mode scramble

# Process both audio and lyrics simultaneously
python main.py -i song.mp3 -o cloaked.wav --lyrics song_lyrics.txt --lyrics-out cloaked_lyrics.txt
```

---

## Technical Overview and Signal Flow

```
+-----------------------------------------------------------------------------+
|                     GHOSTWAVE SIGNAL PROCESSING PIPELINE                    |
+-----------------------------------------------------------------------------+
| 1. Dynamic Micro-Chrono Jitter: +/-16ms non-linear drift                    |
| 2. Asymmetric Triple-Band Bode Frequency Shifter (+12Hz mid / -16Hz high)   |
| 3. Continuous Pitch Wobble LFO: 0.22 Hz swept vibrato                       |
| 4. Center Vocal Cancellation: L-R phase subtraction above 160 Hz            |
| 5. Camouflage Harmonic Drone Bed: -26 dBFS drone with vinyl texture         |
| 6. Acoustic Padding: 4.0s intro pad and 2.0s tail padding                  |
| 7. Adversarial Pseudo-Peak Injection: Decoy peaks in STFT spectrogram       |
| 8. Schroeder All-Pass Phase Dispersion: 4-stage cascaded phase shift        |
| 9. Virtual Room Convolution: Early reflections and boundary diffusion       |
| 10. Vocal Modulation: Formant ring modulation and comb flanging             |
| 11. Remote Stem Separation: Cloud Demucs processing via REST API            |
| 12. PCM Export: 16-bit WAV with metadata stripped                           |
+-----------------------------------------------------------------------------+
```

### Configuration Container (.sn) Specifications
- Container Header: 16-byte identifier (GHOSTWAVE_SN_V1\x00)
- Cipher: Fernet (AES-128-CBC with HMAC-SHA256 authenticated encryption)
- Key Derivation: PBKDF2-HMAC-SHA256 (100,000 iterations)
- Data Integrity: HMAC authenticated payload with immediate rejection on mismatch
- Default Location: ~/.ghostwave.sn or the current working directory

---

## Testing

The test suite covers the DSP routines, the .sn cryptographic container, remote API configuration, audio slicing, and GUI accessibility:

```bash
python -m unittest test_app.py
```

All 23 tests run in approximately 22 seconds.

---

## Contributing

Contributions to GhostWave Studio are welcome. Please refer to [CONTRIBUTING.md](file:///E:/code/GhostWave-Studio/CONTRIBUTING.md) for environment configuration, coding standards, test requirements, and pull request procedures.

---

## Research and Educational Disclaimer

GhostWave Studio is developed as an academic and engineering research project to investigate acoustic fingerprinting robustness, neural audio embeddings, and automated speech recognition ingestion filters. This software is provided for experimental study, education, and format compatibility research. Users are responsible for ensuring that their use of this software complies with applicable local laws and the terms of service of third-party platforms.

---

## License

This project is licensed under the MIT License. See the [LICENSE](file:///E:/code/GhostWave-Studio/LICENSE) file for the complete license terms.

Copyright (c) 2026 Muhamad alfian / Technokers lab.

