"""
Main Application Entry Point for GhostWave Studio v1.0.

Supports both Accessible Desktop GUI (wxPython) and Command-Line Interface (CLI)
for automated batch audio cloaking, ABS slicing, and Chromaprint acoustic evasion audits.
All user settings and API credentials are encrypted inside the proprietary .sn vault.
"""

from __future__ import annotations

import argparse
import os
import sys

from audio_processor import AudioProcessor, AudioSanitizeOptions, benchmark_evasion_metrics
from config_manager import GhostWaveConfig, DEFAULT_SN_FILENAME


def print_banner():
    print("=" * 64)
    print("GhostWave Studio v1.0 - Next-Gen Stealth Audio Cloaking Engine")
    print("Encrypted Vault (.sn) Active | Multi-Vector Acoustic Evasion")
    print("=" * 64)


def run_cli(args: argparse.Namespace) -> int:
    """Executes audio sanitization and prints the acoustic evasion audit from CLI."""
    print_banner()

    # Load configuration from encrypted .sn vault
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

    print(f"\n[GhostWave Studio] Source Audio: {os.path.basename(input_path)}")
    print(f"[GhostWave Studio] Cloaking Preset: {args.preset.upper()}")
    print(f"[GhostWave Studio] Profile Vault: Encrypted (.sn) Loaded")

    options = AudioSanitizeOptions(
        mode="cloud_api" if args.preset == "cloud" else "master_evasion",
        pitch_shift_semitones=args.pitch,
        tempo_factor=args.tempo,
        enable_micro_chrono_jitter=not args.no_jitter,
        enable_bode_freq_shifter=not args.no_bode,
        bode_shift_hz=args.bode_hz,
        enable_adversarial_peaks=not args.no_decoy,
        enable_phase_dispersion=not args.no_allpass,
        enable_reamping_room=not args.no_room,
        enable_vocal_cut=not args.no_vocal_cut,
        enable_formant_scrambler=not args.no_vocal_cut,
        inject_preamble=not args.no_preamble,
        trim_duration=not args.no_trim,
        max_duration_seconds=args.duration,
        strip_metadata=True,
        output_bitrate="320k" if args.format == "mp3" else "wav",
        cloud_provider=vault.cloud_provider,
        cloud_api_token=vault.cloud_api_token,
        cloud_endpoint_url=vault.cloud_endpoint_url
    )

    def cli_progress(pct: int, msg: str):
        print(f"[{pct:3d}%] {msg}")

    result = processor.process(input_path, output_path, options, progress_callback=cli_progress)

    if result.success:
        print("\n[SUCCESS] Audio cloaking completed successfully!")
        print(f"Cloaked Audio Saved: {result.output_path}")
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
        print(f"\n[ERROR] Audio cloaking failed: {result.error_message}")
        return 1


def run_gui():
    """Initializes wx application loop and displays GhostWave Studio v1.0."""
    import wx
    from gui import GhostWaveFrame

    app = wx.App(False)
    frame = GhostWaveFrame()
    frame.Show()
    app.MainLoop()


def main():
    parser = argparse.ArgumentParser(
        description="GhostWave Studio v1.0 - Next-Gen Stealth Audio Cloaking Engine & Acoustic Shield",
        add_help=True
    )
    parser.add_argument("-i", "--input", help="Source audio file to cloak and sanitize")
    parser.add_argument("-o", "--output", help="Destination audio file path")
    parser.add_argument("--format", choices=["mp3", "wav"], default="wav", help="Output container format (default: wav)")
    parser.add_argument("--preset", choices=["max", "balanced", "cloud"], default="max", help="Evasion preset (default: max)")
    parser.add_argument("--pitch", type=float, default=2.5, help="Pitch shift in semitones (default: 2.5)")
    parser.add_argument("--tempo", type=float, default=0.94, help="Tempo factor (default: 0.94)")
    parser.add_argument("--bode-hz", type=float, default=8.5, help="Hilbert Bode frequency shift in Hz (default: 8.5)")
    parser.add_argument("--duration", type=float, default=24.0, help="Max duration in seconds (default: 24.0)")
    parser.add_argument("--config-sn", help="Path to encrypted GhostWave profile vault (.sn)")
    parser.add_argument("--no-jitter", action="store_true", help="Disable micro-chrono jitter")
    parser.add_argument("--no-bode", action="store_true", help="Disable Bode frequency shifter")
    parser.add_argument("--no-decoy", action="store_true", help="Disable STFT decoy peak injection")
    parser.add_argument("--no-allpass", action="store_true", help="Disable Schroeder all-pass dispersion")
    parser.add_argument("--no-room", action="store_true", help="Disable virtual acoustic re-amping")
    parser.add_argument("--no-vocal-cut", action="store_true", help="Disable vocal suppression")
    parser.add_argument("--no-preamble", action="store_true", help="Disable front-end preamble injection")
    parser.add_argument("--no-trim", action="store_true", help="Disable duration trimming")
    parser.add_argument("--audit-only", action="store_true", help="Only run Chromaprint audit on existing files")
    parser.add_argument("--slice", action="store_true", help="Slice audio into ABS safe chunks for Suno Library upload")
    parser.add_argument("--slice-duration", type=float, default=22.0, help="Chunk duration in seconds for --slice (default: 22.0)")
    parser.add_argument("--gui", action="store_true", help="Explicitly launch graphical desktop user interface")

    args = parser.parse_args()

    # If --input is provided, run CLI mode. Otherwise, launch GUI.
    if args.input and not args.gui:
        sys.exit(run_cli(args))
    else:
        run_gui()


if __name__ == "__main__":
    main()
