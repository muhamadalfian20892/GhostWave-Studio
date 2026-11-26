GHOSTWAVE STUDIO (VERSION 1.7.0)
USER GUIDE AND GETTING STARTED MANUAL

Original Website: http://technokerslab.blogspot.com/

Welcome to GhostWave Studio! This guide walks you through preparing audio tracks
and lyrics using an interface built from the ground up for screen readers
(NVDA, JAWS, Windows Narrator) and full keyboard navigation. You do not need
programming knowledge or heavy machine learning downloads to use any feature.


WHAT THIS PROGRAM DOES

When you upload audio to AI music platforms, automated scanners inspect your
file for matching spectral landmarks, melody chroma curves, embedded watermarks,
and vocal speech patterns.

GhostWave Studio applies local digital signal processing in pure SciPy and NumPy
to decouple those fingerprint patterns while keeping musical phrasing intact for
cover prompting or backing tracks. All API tokens, preferences, and custom
blacklists stay inside an encrypted local profile vault (.sn) on your computer.


LANGUAGE SUPPORT AND REALTIME SWITCHING

GhostWave Studio supports both English and Bahasa Indonesia.

On first launch, a language dialog lets you choose your preferred language. You
can switch languages anytime from the Language menu in the menu bar. Every
label, tooltip, menu item, and status announcement updates right away without
restarting the program.


SUPPORT TICKETS AND GETTING HELP

If you run into a bug, an unsupported audio file, or have a feature idea, you
can send a support ticket directly inside the application:

1. Press Ctrl+T or select Help > Support Tickets in the menu bar.
2. Pick a category: Bug Report, Feature Request, Question and Support, or Other.
3. Enter a subject and clear description of the issue.
4. Keep the system diagnostics checkbox checked so OS and app version details
   are included automatically.
5. Click Submit Ticket.

Open the "My Tickets" tab anytime to read developer replies and send follow-up
messages. You can also email hafiyanajah@gmail.com for questions or feedback.


QUICK START: PROCESSING YOUR AUDIO

Step 1: Open your audio file.
Click Browse Audio File or press Alt+B (or Ctrl+O). Pick a WAV, MP3, FLAC,
M4A, OGG, or AAC file, or drag and drop a file directly into the window.
GhostWave Studio automatically inspects duration, sample rate, musical key, and
BPM, and offers a one-click harmonious pitch recommendation.

Step 2: Choose a target platform and settings.
Pick a preset platform at the top of the Audio tab:
- Suno v4-v6 Cover Mode: Preserves singing melody while decoupling Audible
  Magic landmarks and CQT chroma.
- Google Lyria 3.5 / SynthID Cleaner: Runs only AI watermark neutralization and
  strips metadata while keeping original audio pitch and timing untouched.
- Udio 1.5/2.0 Mode: Tuned for Udio context windows and stem separation.
- YouTube Content ID Evasion: Stronger pitch, formant, and room diffusion.
- TikTok / Reels Mask: Lightweight adjustments for short-form uploads.
- Custom / Manual: Full manual control over all sliders.

Adjust the modular sliders (Pitch Shift, Formant Shift, Tempo Factor, Studio
Room Reverb, Center Vocal Reduction) or expand Show Advanced Settings for
fine-grained DSP toggles.

Step 3: Process, audition, and verify.
Click Process Audio (or press Alt+P) and save as 16-bit PCM WAV.
Once finished, you can:
- Click Play Original, Play Sanitized, or Play Delta to audition results inside
  the window, or read the accessible waveform summary below the canvas.
- Press Alt+F to open the Pre-Flight AI Evasion Simulator for a safety score
  and advice before uploading.
- Press Alt+E to run the Chromaprint similarity audit.


BATCH PROCESSING AND FOLDER WATCHER (CTRL+SHIFT+B)

To process many tracks at once, press Ctrl+Shift+B:
- Batch Queue tab: Add multiple files or an entire folder, optionally enable
  automatic 22-second WAV slicing, and click Start Batch Queue.
- Folder Watcher tab: Pick an Input Folder and Output Folder and click Start
  Folder Watcher to process newly added tracks automatically in the background.


CLEANING AND FORMATTING YOUR LYRICS (CTRL+TAB)

Switch to the Lyrics Sanitizer tab with Alt+2 or Ctrl+Tab:

1. Paste your lyrics into the input box (or use Generate Suno / Udio Style Prompt
   to build genre tags and song sections from your track's detected BPM and key).
2. Choose a mode: Phonetic Spelling Variation keeps your words singable across
   English, Indonesian, Spanish, and other Latin-script languages while breaking
   4-gram database matches.
3. Keep "Preserve Rhyme Scheme and Cadence" checked to protect end-line rhymes.
4. Click Sanitize Lyrics (Alt+S), inspect the 4-gram overlap audit, and click
   Copy to Clipboard (Alt+C).


SUNO UPLOAD TIPS

1. Export as 16-bit PCM WAV rather than MP3 so encoder padding and ID3 tags are
   never left behind.
2. Upload via Suno Library > Upload Audio rather than the Create page.
3. Keep uploaded clips between 20 and 24 seconds using the Audio Slicer (Alt+S
   or Ctrl+U) to avoid long-horizon fingerprint matches.
4. If an upload is flagged, clear Suno cookies or switch to a private browser
   window before retrying.


KEYBOARD SHORTCUTS FOR SCREEN READERS

Press F1 anytime to open the Accessibility Guide dialog. Press Escape to close
any active dialog.

Tab / Shift+Tab: Move forward and backward through controls.
Alt+1 / Alt+2 / Ctrl+Tab: Switch between Audio and Lyrics tabs.
Alt+B / Ctrl+O: Browse for an audio file.
Alt+P / Ctrl+P: Process and export audio.
Alt+F: Open Pre-Flight AI Evasion Simulator.
Alt+E / Ctrl+E: Run Chromaprint audio similarity audit.
Ctrl+Shift+B: Open Batch Processing and Folder Watcher.
Ctrl+U: Open ABS Audio Slicer from any tab.
Alt+A / Ctrl+K: Open Cloud API stem settings.
Ctrl+B: Open Celebrity Blacklist editor.
Ctrl+T: Open Support Ticket Center.
Ctrl+H or Shift+F1: Open Suno Upload Guide.
Ctrl+Shift+S / Ctrl+Shift+O: Export or import settings profile (.sn).
F1: Open Accessibility Guide.
F2: Open Release Notes / Changelog viewer.


AUTOMATIC UPDATES

GhostWave Studio can check GitHub Releases automatically on startup or via
Help > Check for Updates. You can read release notes inline in the update dialog
and download the latest installer or portable archive directly.


OPEN SOURCE AND SOURCE CODE

GhostWave Studio is open source software under the MIT License. You can inspect
the source code, report bugs, or contribute at:
https://github.com/muhamadalfian20892/GhostWave-Studio
