"""
Main Application Entry Point for GhostWave Studio.

Supports both Accessible Desktop GUI (wxPython) and Command-Line Interface (CLI)
for automated audio sanitization, slicing, and acoustic similarity audits.
All user settings and API credentials are saved in the settings profile (.sn).
"""

from __future__ import annotations

import argparse
import os
import sys

from ghostwave.audio_processor import AudioProcessor, AudioSanitizeOptions, benchmark_evasion_metrics
from ghostwave.config_manager import GhostWaveConfig, DEFAULT_SN_FILENAME
from ghostwave.version import APP_VERSION


def print_banner():
    print("=" * 64)
    print(f"GhostWave Studio v{APP_VERSION} - Audio & Lyrics Sanitizer")
    print("Settings Profile (.sn) Active | Suno Preparation Engine")
    print("=" * 64)


def _build_cli_options(args: argparse.Namespace, vault: GhostWaveConfig) -> AudioSanitizeOptions:
    """Builds AudioSanitizeOptions from parsed CLI arguments and the .sn vault."""
    return AudioSanitizeOptions(
        mode="cloud_api" if args.preset == "cloud" else "master_evasion",
        pitch_shift_semitones=args.pitch,
        tempo_factor=args.tempo,
        formant_shift_semitones=getattr(args, "formant", 2.0),
        enable_ai_watermark_neutralizer=getattr(args, "watermark_clean", True),
        use_psychoacoustic_masking=not getattr(args, "no_psycho", False),
        enable_micro_chrono_jitter=not args.no_jitter,
        enable_bode_freq_shifter=not args.no_bode,
        bode_shift_hz=args.bode_hz,
        enable_adversarial_peaks=not args.no_decoy,
        enable_phase_dispersion=not args.no_allpass,
        enable_reamping_room=not args.no_room,
        enable_vocal_cut=not args.no_vocal_cut,
        enable_formant_scrambler=not args.no_vocal_cut,
        inject_preamble=not args.no_preamble,
        trim_duration=bool(args.trim and not args.no_trim),
        max_duration_seconds=args.duration,
        strip_metadata=True,
        output_bitrate="320k" if args.format == "mp3" else "wav",
        cloud_provider=vault.cloud_provider,
        cloud_api_token=vault.cloud_api_token,
        cloud_endpoint_url=vault.cloud_endpoint_url,
    )


def run_cli(args: argparse.Namespace) -> int:
    """Executes audio sanitization and prints the acoustic evasion audit from CLI."""
    if not getattr(args, "no_banner", False):
        print_banner()

    # Load configuration from .sn profile
    vault = GhostWaveConfig.load(args.config_sn) if hasattr(args, "config_sn") and args.config_sn else GhostWaveConfig.load()

    processor = AudioProcessor()
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        return 1

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem = os.path.splitext(input_path)[0]
        ext = ".mp3" if args.format == "mp3" else ".wav"
        output_path = f"{stem}_ghostwave_ready{ext}"

    if args.slice:
        out_dir = os.path.abspath(args.output) if args.output else os.path.join(os.path.dirname(input_path), "ghostwave_chunks")
        print(f"\n[ABS Slicer] Slicing {os.path.basename(input_path)} into {args.slice_duration:.1f}s WAV chunks...")
        chunks = processor.slice_audio(input_path, out_dir, chunk_duration_sec=args.slice_duration)
        print(f"[ABS Slicer] Successfully created {len(chunks)} chunks in: {out_dir}")
        for i, c in enumerate(chunks, 1):
            print(f"  Part {i:02d}: {os.path.basename(c)}")
        print("\nSuno Upload Protocol: Upload chunks to Suno Library -> Upload Audio, then Create with Audio.")
        return 0

    if args.audit_only:
        print(f"\n[Acoustic Audit] Auditing {os.path.basename(input_path)} against {os.path.basename(output_path)}...")
        metrics = benchmark_evasion_metrics(input_path, output_path)
        print("--------------------------------------------------")
        print(f"Similarity Score: {metrics.get('similarity_pct', 0.0):.1f}%")
        print(f"Bit Error Rate:   {metrics.get('bit_error_rate_pct', 0.0):.1f}%")
        print(f"Vocal Band Drop:  {metrics.get('vocal_energy_drop_db', 0.0):.1f} dB")
        print(f"Verdict:          {metrics.get('verdict', 'Unknown')}")
        print(f"Assessment:       {metrics.get('details', '')}")
        print("--------------------------------------------------")
        return 0

    if getattr(args, "preflight", False):
        if not os.path.exists(output_path):
            print(f"Error: Pre-flight audit compares an original file (-i) against a processed file (-o). File not found: {output_path}")
            return 1
        print(f"\n[Pre-Flight Evasion Simulator] Auditing {os.path.basename(input_path)} against {os.path.basename(output_path)}...")
        report = processor.simulate_preflight(input_path, output_path)
        print("=" * 55)
        print(f"SAFETY SCORE:             {report.get('safety_score', 0)}% ({report.get('verdict_en', report.get('verdict', 'Unknown'))})")
        print(f"CQT Chroma Similarity:    {report.get('cqt_chroma_similarity_pct', 0.0):.1f}% (Threshold: < 35%)")
        print(f"Landmark Overlap:         {report.get('landmark_overlap_pct', 0.0):.1f}% (Threshold: < 15%)")
        print(f"SynthID Watermark Clean:  {report.get('synthid_watermark_neutralized', False)}")
        print(f"Center Vocal Attenuation: {report.get('vocal_isolation_db', 0.0):.1f} dB")
        print("-" * 55)
        print(f"Actionable Advice: {report.get('actionable_advice_en', report.get('actionable_advice', ''))}")
        print("=" * 55)
        return 0

    print(f"\n[GhostWave Studio] Source Audio: {os.path.basename(input_path)}")
    print(f"[GhostWave Studio] Preset: {args.preset.upper()}")
    print(f"[GhostWave Studio] Settings Profile: (.sn) Loaded")

    options = _build_cli_options(args, vault)

    def cli_progress(pct: int, msg: str):
        print(f"[{pct:3d}%] {msg}")

    result = processor.process(input_path, output_path, options, progress_callback=cli_progress)

    if result.success:
        print("\n[SUCCESS] Audio sanitization completed successfully!")
        print(f"Sanitized Audio Saved: {result.output_path}")
        print(f"Final Duration: {result.final_duration:.2f} seconds")

        metrics = benchmark_evasion_metrics(input_path, output_path)
        if metrics.get("chromaprint_available"):
            print("\n=== QUANTITATIVE ACOUSTIC AUDIT ===")
            print(f"Chromaprint Match:  {metrics.get('similarity_pct', 0.0):.1f}% (Target: < 35%)")
            print(f"Bit Error Rate:     {metrics.get('bit_error_rate_pct', 0.0):.1f}%")
            print(f"Vocal Band Drop:    {metrics.get('vocal_energy_drop_db', 0.0):.1f} dB")
            print(f"Safety Verdict:     {metrics.get('verdict')}")
            print(f"Assessment:         {metrics.get('details')}")
            print("====================================\n")
        return 0
    else:
        print(f"\n[ERROR] Audio processing failed: {result.error_message}")
        return 1


def run_lyrics_cli(args: argparse.Namespace) -> int:
    """Executes lyrics sanitization and formatting from CLI."""
    lyrics_path = os.path.abspath(args.lyrics)
    if not os.path.exists(lyrics_path):
        print(f"Error: Lyrics file does not exist: {lyrics_path}")
        return 1

    with open(lyrics_path, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    from ghostwave.lyrics_processor import LyricsProcessor
    vault = GhostWaveConfig.load(args.config_sn) if hasattr(args, "config_sn") and args.config_sn else GhostWaveConfig.load()
    processor = LyricsProcessor()

    result = processor.sanitize(
        text=raw_text,
        cloak_lyrics=True,
        cloak_mode=args.lyrics_mode,
        add_vibrato_glides=True,
        preserve_syllables=True,
        break_ngrams=not args.no_adlibs,
        cloud_token=vault.lyrics_cloud_api_token,
        cloud_provider=vault.lyrics_cloud_provider,
        cloud_model=vault.lyrics_cloud_model,
        cloud_endpoint=vault.lyrics_cloud_endpoint_url
    )

    print("\n" + "=" * 50)
    print("GHOSTWAVE LYRICS SANITIZATION REPORT")
    print("=" * 50)
    if result.audit:
        aud = result.audit
        print(f"Verdict:                 {aud.evasion_verdict}")
        print(f"4-Gram Sequence Overlap: {aud.ngram_overlap_pct:.1f}%")
        print(f"Token Similarity:        {aud.token_similarity_pct:.1f}%")
        print(f"Syllable Cadence Match:  {aud.syllable_accuracy_pct:.1f}%")
        print(f"Original Words:          {aud.original_words} -> Sanitized: {aud.cloaked_words}")

    print("\nApplied Transformations:")
    for chg in result.changes:
        print(f"  * {chg}")

    if args.lyrics_out:
        out_path = os.path.abspath(args.lyrics_out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.sanitized_text)
        print(f"\nSanitized lyrics written to: {out_path}")
    else:
        print("\nSanitized Lyrics Output:")
        print("-" * 50)
        print(result.sanitized_text)
        print("-" * 50)

    return 0


def run_gui():
    """Initializes wx application loop and displays GhostWave Studio."""
    from ghostwave.gui import main as gui_main
    gui_main()


def main():
    parser = argparse.ArgumentParser(
        description=f"GhostWave Studio v{APP_VERSION} - Audio and Lyrics Sanitizer",
        add_help=True
    )
    parser.add_argument("-v", "--version", action="version", version=f"GhostWave Studio v{APP_VERSION}")
    parser.add_argument("-i", "--input", help="Source audio file to sanitize")
    parser.add_argument("-o", "--output", help="Destination audio file path")
    parser.add_argument("--format", choices=["mp3", "wav"], default="wav", help="Output container format (default: wav)")
    parser.add_argument("--preset", choices=["max", "balanced", "cloud"], default="max", help="Sanitization preset (default: max)")
    parser.add_argument("--pitch", type=float, default=2.5, help="Pitch shift in semitones (default: 2.5)")
    parser.add_argument("--tempo", type=float, default=0.94, help="Tempo factor (default: 0.94)")
    parser.add_argument("--bode-hz", type=float, default=8.5, help="Frequency shift in Hz (default: 8.5)")
    parser.add_argument("--duration", type=float, default=24.0, help="Max duration in seconds (default: 24.0)")
    parser.add_argument("--config-sn", help="Path to GhostWave settings profile (.sn)")
    parser.add_argument("--no-jitter", action="store_true", help="Disable timing variation jitter")
    parser.add_argument("--no-bode", action="store_true", help="Disable Bode frequency shifter")
    parser.add_argument("--no-decoy", action="store_true", help="Disable decoy harmonic injection")
    parser.add_argument("--no-allpass", action="store_true", help="Disable phase dispersion filter")
    parser.add_argument("--no-room", action="store_true", help="Disable room acoustic simulation")
    parser.add_argument("--no-vocal-cut", action="store_true", help="Disable vocal suppression")
    parser.add_argument("--no-preamble", action="store_true", help="Disable intro preamble injection")
    parser.add_argument("--trim", action="store_true", help="Trim audio to safe duration for Suno clips (disabled by default)")
    parser.add_argument("--no-trim", action="store_true", help="Explicitly disable duration trimming")
    parser.add_argument("--audit-only", action="store_true", help="Only run Chromaprint audit on existing files")
    parser.add_argument("--preflight", action="store_true", help="Run pre-flight multi-vector evasion simulation")
    parser.add_argument("--watermark-clean", action="store_true", default=True, help="Neutralize neural AI audio watermarks (Google SynthID / AudioSeal)")
    parser.add_argument("--formant", type=float, default=2.0, help="Formant shift in semitones (default: 2.0)")
    parser.add_argument("--no-psycho", action="store_true", help="Disable psychoacoustic threshold masking")
    parser.add_argument("--slice", action="store_true", help="Slice audio into safe chunks for Suno Library upload")
    parser.add_argument("--slice-duration", type=float, default=22.0, help="Chunk duration in seconds for --slice (default: 22.0)")
    parser.add_argument("--watch-input", help="Directory path to monitor with automatic Folder Watcher")
    parser.add_argument("--watch-output", help="Directory path where sanitized tracks are saved by Folder Watcher")
    parser.add_argument("--batch-dir", help="Directory containing audio tracks to batch sanitize")
    parser.add_argument("--lyrics", help="Path to text file containing lyrics to sanitize")
    parser.add_argument("--lyrics-out", help="Destination path for sanitized lyrics output file")
    parser.add_argument("--lyrics-mode", choices=["scramble", "hybrid", "phonetic", "semantic", "cloud"], default="scramble", help="Lyrics sanitization strategy (default: scramble)")
    parser.add_argument("--no-adlibs", action="store_true", help="Disable rhythmic ad-libs in lyrics sanitization")
    parser.add_argument("--check-update", action="store_true", help="Check for application updates")
    parser.add_argument("--gui", action="store_true", help="Explicitly launch graphical desktop user interface")
    parser.add_argument("--no-banner", action="store_true", help="Suppress startup banner output in CLI mode")

    args = parser.parse_args()

    if args.watch_input and args.watch_output:
        from ghostwave.batch_processor import FolderWatcher
        vault = GhostWaveConfig.load(args.config_sn) if args.config_sn else GhostWaveConfig.load()
        cli_opts = _build_cli_options(args, vault)
        print(f"Starting Folder Watcher daemon on '{args.watch_input}' -> '{args.watch_output}'...")
        watcher = FolderWatcher(
            args.watch_input,
            args.watch_output,
            options=cli_opts,
            audio_processor=AudioProcessor(),
        )
        watcher.start()
        print("Folder Watcher is running in background. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            watcher.stop()
            print("Watcher stopped.")
        sys.exit(0)

    if args.batch_dir:
        from ghostwave.batch_processor import BatchProcessor, BatchTask
        vault = GhostWaveConfig.load(args.config_sn) if args.config_sn else GhostWaveConfig.load()
        cli_opts = _build_cli_options(args, vault)
        b_proc = BatchProcessor(AudioProcessor())
        exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        out_ext = ".mp3" if args.format == "mp3" else ".wav"
        out_dir = os.path.abspath(args.output) if args.output else None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tasks = []
        for e in sorted(os.scandir(args.batch_dir), key=lambda x: x.name.lower()):
            if e.is_file() and os.path.splitext(e.name)[1].lower() in exts:
                stem = os.path.splitext(e.name)[0] if out_dir else os.path.splitext(e.path)[0]
                out_p = os.path.join(out_dir, f"{stem}_ghostwave_ready{out_ext}") if out_dir else f"{stem}_ghostwave_ready{out_ext}"
                tasks.append(BatchTask(e.path, out_p))
        print(f"Batch processing {len(tasks)} files in '{args.batch_dir}'...")
        b_proc.process_queue(tasks, cli_opts, progress_callback=lambda idx, tot, pct, msg: print(f"[{idx+1}/{tot}] {pct}% - {msg}"))
        print("Batch processing completed.")
        sys.exit(0)

    if args.check_update:
        from ghostwave.updater import check_for_updates
        print(f"Checking for updates (current version: {APP_VERSION})...")
        info = check_for_updates()
        if info.has_update:
            print(f"Update available: v{info.latest_version}")
            if info.download_url:
                print(f"Download URL: {info.download_url}")
            if info.release_notes:
                print("\nWhat is new:")
                print(info.release_notes)
        else:
            if info.error_message:
                print(f"Could not check for updates: {info.error_message}")
            else:
                print(f"GhostWave Studio is up to date (v{APP_VERSION}).")
        sys.exit(0)

    # If --lyrics is provided without audio input, process lyrics only
    if args.lyrics and not args.input and not args.gui:
        sys.exit(run_lyrics_cli(args))

    # If --input is provided, run CLI mode. Otherwise, launch GUI.
    if args.input and not args.gui:
        if args.lyrics:
            run_lyrics_cli(args)
        sys.exit(run_cli(args))
    else:
        run_gui()


if __name__ == "__main__":
    main()
