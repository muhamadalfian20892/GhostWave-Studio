GHOSTWAVE STUDIO (VERSION 1.6.0)
USER GUIDE AND GETTING STARTED MANUAL

Original Website: http://technokerslab.blogspot.com/

Welcome to GhostWave Studio! If you are looking for an easy, no-nonsense way
to prepare your audio tracks and lyrics for platforms like Suno without running
into automated upload blocks, you are in the right place. I designed this
program so you do not need to know anything about coding, command lines, or
signal processing to get great results.


WHAT THIS PROGRAM ACTUALLY DOES

When you upload a song to AI music services, their system runs an automated
scanner across your file looking for matches against existing copyrighted
songs. It listens for specific audio landmarks, melody curves, and voice
characteristics. If it thinks your file sounds too close to an existing
commercial track, it simply rejects it.

GhostWave Studio takes your original song and applies gentle acoustic
adjustments that scramble those landmark fingerprints. To the human ear,
your song still sounds like your track with its groove and rhythm intact,
but to automated scanners, the fingerprint matches disappear into static.

All your private API keys and settings are stored locally inside an encrypted
vault file named ghostwave.sn on your computer. Nothing is ever sent to any
unauthorized server, and you do not have to download giant machine learning
models just to use the software.


LANGUAGE SUPPORT AND REALTIME SWITCHING

Starting in version 1.4.0, GhostWave Studio features full internationalization
support for both English and Bahasa Indonesia.

When you launch the application for the very first time, a language selection
dialog welcomes you, allowing you to choose your preferred language right away.
You can also change the interface language at any time from the Language menu
in the menu bar. The interface updates immediately in real-time without needing
to restart the program.


SUPPORT TICKETS AND GETTING HELP

If you ever encounter an issue, discover an audio file format that fails to
process, or have an idea for a new feature, you can send a support ticket directly
from inside the application without needing a browser.

1. Press Ctrl+T on your keyboard or select Help > Support Tickets in the menu bar.
2. Choose your ticket category from the list: Bug Report, Feature Request, Question
and Support, or Other.
3. Type in a subject and detailed description of the situation.
4. Leave the system diagnostics box checked so I can see your operating
system details and app version to fix the problem quickly.
5. Click Submit Ticket.

Your ticket is securely delivered to my end. You can check the
"My Tickets" tab at any time to read replies from the developer and send follow-up
messages directly in the dialog.

You can also reach out directly via email at hafiyanajah@gmail.com for any
inquiries or feedback.


QUICK START: CLEANING YOUR AUDIO IN THREE EASY STEPS

Step 1: Open your audio file.
Click the Browse Audio File button or press Alt+B on your keyboard. Select your
song file (MP3, WAV, FLAC, M4A, or OGG). GhostWave Studio will inspect the file
and show you its duration, sample rate, and format right on the screen. You can
also just drag and drop the audio file directly into the application window from
File Explorer!

Step 2: Choose your sanitization preset.
For most people, the default preset named "Complete Sanitization" gives the
highest level of protection. If you want to keep the song sounding closer to the
original recording, you can pick "Balanced Audio Quality" from the drop-down
menu. If you do not want to fiddle with numbers, you can completely ignore the
advanced options. If you do want to tweak pitch shifts, tempo nudges, or room
simulation, click the "Show Advanced Settings" button to expand those controls.

Step 3: Process and export your file.
Click "Process Audio" or press Alt+P. Choose where you want to save
the new file. I strongly recommend saving your file as 16-bit PCM WAV. The
progress bar will fill up as each stage runs, and when it finishes, you will get
a clean audio file that is ready to upload. If you ever need to stop halfway
through, simply click the "Cancel" button.


THE SECRET TO UPLOADING TO SUNO WITHOUT GETTING BLOCKED

Over months of testing, I discovered that how you upload your file to Suno
matters just as much as what is inside the audio. Keep these four golden rules
in mind:

1. Always export as lossless 16-bit WAV instead of MP3.
MP3 files contain hidden header frames and encoder metadata that automated
fingerprint scanners love to latch onto. Lossless WAV files have zero extra
metadata tags, which makes them much harder for scanners to detect.

2. Do not upload directly on the Create page.
Suno's Create page runs an aggressive, synchronous fingerprint scan that will
often flag files immediately. Instead, go to your Suno Library page and click the
"Upload Audio" button there. Suno's Library upload queue handles files through a
different ingestion worker that is significantly more relaxed. Once your audio
chunk shows up in your Library, click the three dots on the track and select
"Extend" or "Create with Audio".

3. Keep your uploaded audio chunks between 20 and 24 seconds.
Long songs give scanners hundreds of matching points in a row. Short clips
break up that continuous timeline. If you have a longer track, click the "ABS
Audio Slicer" button in GhostWave Studio (or press Alt+S) and let it split your
song into clean 22-second WAV chunks automatically. Upload part 1 to your
Library first, and you are ready to go.

4. If Suno blocks an upload, clear your cookies or wait a bit.
If Suno ever tells you that your audio matches an existing recording, do not
keep hitting upload immediately. Your browser session gets temporarily flagged
for thirty to sixty minutes. Open a private or incognito window, clear your
browser cache, or take a short break before trying again with your sanitized
file.


CLEANING AND FORMATTING YOUR LYRICS

If you are typing or pasting lyrics into Suno and getting blocked because of
artist names, trademarked phrases, or copyright filters, switch over to the
Lyrics Sanitizer tab by pressing Ctrl+Tab.

1. Paste your lyrics into the top box.
2. Select your sanitization strategy. The "Phonetic Spelling Variation" is the most
popular because it replaces sensitive phrases with singable phonetic spellings
and hyphenated syllables without changing how the singer delivers the words.
3. Click "Sanitize Lyrics" or press Alt+S.
4. Click "Copy to Clipboard" or press Alt+C, and paste them straight into
Suno's prompt box.


ACCESSIBILITY SHORTCUTS FOR SCREEN READERS

GhostWave Studio was built from day one to be fully usable with screen readers,
including NVDA, JAWS, and Windows Narrator. Every button, slider, and status
box has an explicit spoken label and a hotkey.

Press F1 at any time to open the Accessibility Guide dialog. You can read
through the instructions line by line with your arrow keys, press Escape or
Alt+C to close it, or click the Copy button to save the entire shortcut list to
your clipboard.

Key shortcuts to remember:
Tab and Shift+Tab: Move forward and backward through controls.
Ctrl+Tab and Ctrl+Shift+Tab: Switch between the Audio and Lyrics tabs.
Alt+B: Browse for an audio file.
Alt+P: Process and export your audio.
Alt+E: Open the Audio Similarity Audit report.
Alt+S: In the Audio tab, opens the ABS Audio Slicer. In the Lyrics tab, runs
the lyrics sanitizer.
Alt+A: Open Cloud API settings.
Ctrl+T: Open Support Ticket Center.
Ctrl+U: Open the ABS Audio Slicer from anywhere in the app.
Ctrl+B: Open the Celebrity Blacklist editor.
Ctrl+H or Shift+F1: Open the Suno Upload Guide cheat sheet.
Ctrl+Shift+S: Export your encrypted profile vault (.sn).
Ctrl+Shift+O: Import an existing encrypted profile vault (.sn).
Ctrl+A: Select all text inside any multiline box.
F2: Open What's New / Changelog viewer to read latest updates offline or online.
Escape: Close any dialog immediately.


KEEPING GHOSTWAVE STUDIO UPDATED

You do not need to keep checking websites to see if a new version came out.
Whenever you launch GhostWave Studio, it can check the update by itself quietly in the
background. If an update is available, a friendly window will appear asking if
you would like to download it.

You can click "See What is New in This Changes" to read the notes right inside
the same window, or click "Download Now" to let the software download the new
installer and guide you through the update in seconds.


OPEN SOURCE AND COMMUNITY CONTRIBUTION

GhostWave Studio is completely open source software. If you would like to inspect
the source code, submit bug reports, suggest improvements, or contribute code,
please visit my official GitHub repository:
https://github.com/muhamadalfian20892/GhostWave-Studio
