"""
Automated Unit Tests for GhostWave Studio v1.0.

Tests DSP acoustic cloaking algorithms, ABS audio slicer, proprietary .sn
encrypted profile vault, lyrics moderation and cloaking engine, and accessibility GUI.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ghostwave.config_manager import GhostWaveConfig, MAGIC_HEADER
from ghostwave.lyrics_processor import (
    LyricsProcessor,
    count_syllables,
    count_line_syllables,
    audit_lyrics,
    LyricsCloaker
)


class TestGhostWaveEncryptedVault(unittest.TestCase):
    """Verifies cryptographic security and tamper-resistance of the .sn vault."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.vault_path = os.path.join(self.test_dir, "test_profile.sn")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_magic_header_present(self):
        cfg = GhostWaveConfig(cloud_api_token="secret_test_token_12345")
        raw = cfg.to_encrypted_bytes()
        self.assertTrue(raw.startswith(MAGIC_HEADER))
        self.assertEqual(len(MAGIC_HEADER), 16)

    def test_no_plaintext_leakage(self):
        token = "secret_replicate_token_abcdef123456"
        cfg = GhostWaveConfig(cloud_api_token=token)
        raw = cfg.to_encrypted_bytes()
        self.assertNotIn(token.encode("utf-8"), raw)
        self.assertNotIn(b"replicate", raw[16:])

    def test_save_and_load_roundtrip(self):
        cfg = GhostWaveConfig(
            cloud_api_token="my_token_999",
            pitch_shift_semitones=3.2,
            export_format="wav",
            lyrics_cloak_mode="hybrid"
        )
        saved = cfg.save(self.vault_path)
        self.assertTrue(saved)
        self.assertTrue(os.path.exists(self.vault_path))

        loaded = GhostWaveConfig.load(self.vault_path)
        self.assertEqual(loaded.cloud_api_token, "my_token_999")
        self.assertAlmostEqual(loaded.pitch_shift_semitones, 3.2)
        self.assertEqual(loaded.export_format, "wav")
        self.assertEqual(loaded.lyrics_cloak_mode, "hybrid")

    def test_tampered_file_rejected(self):
        cfg = GhostWaveConfig(cloud_api_token="valid_secret_key")
        raw = bytearray(cfg.to_encrypted_bytes())
        # Flip a bit in the encrypted payload
        raw[25] ^= 0xFF
        with self.assertRaises(ValueError):
            GhostWaveConfig.from_encrypted_bytes(bytes(raw))

    def test_invalid_header_rejected(self):
        bad_bytes = b"INVALID_HEADER_X" + b"random_payload"
        with self.assertRaises(ValueError):
            GhostWaveConfig.from_encrypted_bytes(bad_bytes)


class TestLyricsProcessorAndCloaking(unittest.TestCase):
    """Verifies moderation filtering, syllable counting, and copyright cloaking."""

    def setUp(self):
        self.processor = LyricsProcessor()

    def test_syllable_counter(self):
        self.assertEqual(count_syllables("yesterday"), 3)
        self.assertEqual(count_syllables("trouble"), 2)
        self.assertEqual(count_syllables("small"), 1)
        self.assertEqual(count_syllables("town"), 1)
        self.assertEqual(count_syllables("girl"), 1)
        self.assertEqual(count_syllables("living"), 2)
        self.assertEqual(count_syllables("lonely"), 2)
        self.assertEqual(count_syllables("world"), 1)
        self.assertEqual(count_syllables("midnight"), 2)
        self.assertEqual(count_syllables("train"), 1)
        self.assertEqual(count_syllables("fantasy"), 3)
        self.assertEqual(count_syllables("reality"), 4)

    def test_line_syllables(self):
        line = "Just a small town girl"
        self.assertEqual(count_line_syllables(line), 5)
        tag_line = "[Verse 1] Just a small town girl"
        self.assertEqual(count_line_syllables(tag_line), 0)

    def test_celebrity_filter(self):
        raw = "I saw Taylor Swift and Drake walking down the street."
        sanitized, changes = self.processor.filter_celebrities(raw)
        self.assertNotIn("Taylor Swift", sanitized)
        self.assertNotIn("Drake", sanitized)
        self.assertIn("[Artist]", sanitized)
        self.assertGreaterEqual(len(changes), 2)

    def test_profanity_filter(self):
        raw = "Fuck that bullshit and that shit."
        sanitized, changes = self.processor.filter_profanities(raw)
        self.assertNotIn("fuck", sanitized.lower())
        self.assertNotIn("bullshit", sanitized.lower())
        self.assertNotIn("shit", sanitized.lower())
        self.assertGreaterEqual(len(changes), 3)

    def test_unicode_normalization(self):
        raw = "Beyoncé said “Don’t wait—live your life…”"
        cleaned, changes = self.processor.normalize_unicode(raw)
        self.assertNotIn("Beyoncé", cleaned)
        self.assertIn("Beyonce", cleaned)
        self.assertNotIn("“", cleaned)
        self.assertNotIn("”", cleaned)
        self.assertNotIn("—", cleaned)

    def test_structural_tag_formatting(self):
        raw = "[[Chorus]]\n(Verse 1)\nDrop:"
        formatted, changes = self.processor.format_structural_tags(raw)
        self.assertIn("[Chorus]", formatted)
        self.assertIn("[Verse 1]", formatted)
        self.assertIn("[Drop]", formatted)

    def test_lyrics_cloaking_hybrid(self):
        song = (
            "[Verse 1]\n"
            "Just a small town girl, living in a lonely world\n"
            "She took the midnight train going anywhere\n"
            "[Chorus]\n"
            "Don't stop believing, hold on to that feeling\n"
            "Streetlights, people, living just to find emotion\n"
        )
        res = self.processor.sanitize(
            text=song,
            cloak_lyrics=True,
            cloak_mode="hybrid",
            preserve_syllables=True,
            break_ngrams=True
        )

        self.assertIsNotNone(res.audit)
        aud = res.audit
        self.assertLess(aud.ngram_overlap_pct, 15.0)
        self.assertLess(aud.token_similarity_pct, 45.0)
        self.assertGreaterEqual(aud.syllable_accuracy_pct, 75.0)
        self.assertIn("EXCELLENT", aud.evasion_verdict.upper())

    def test_lyrics_cloaking_phonetic(self):
        song = "Is this the real life? Is this just fantasy?\nCaught in a landslide, no escape from reality"
        res = self.processor.sanitize(
            text=song,
            cloak_lyrics=True,
            cloak_mode="phonetic",
            preserve_syllables=True,
            break_ngrams=False
        )
        out = res.sanitized_text
        self.assertIn("reel", out)
        self.assertIn("fan-ta-see", out)
        self.assertIn("land-slyde", out)
        self.assertIn("re-al-i-tee", out)

    def test_lyrics_scramble_preserves_words(self):
        song = (
            "[Verse 1]\n"
            "Just a small town girl, living in a lonely world\n"
            "She took the midnight train going anywhere\n"
        )
        res = self.processor.sanitize(
            text=song,
            cloak_lyrics=True,
            cloak_mode="scramble",
            preserve_syllables=True,
            break_ngrams=False
        )
        out = res.sanitized_text
        # Verifies words are disguised phonetically, not replaced by synonyms
        self.assertIn("smal", out)
        self.assertIn("toun", out)
        self.assertIn("gurl", out)
        self.assertIn("liv-in'", out)
        self.assertIn("lone-ly", out)
        self.assertIn("werld", out)
        self.assertIn("mid-nite", out)
        self.assertIn("trayn", out)
        self.assertIn("go-in'", out)

    def test_lyrics_multilingual_indonesian(self):
        song = (
            "[Verse 1]\n"
            "Kuasai diriku, tenanglah hatiku\n"
            "Jangan pernah goyah, berjalanlah bersamaku\n"
            "[Chorus]\n"
            "Biar ku menepi, menghapus jejakmu\n"
        )
        res = self.processor.sanitize(
            text=song,
            cloak_lyrics=True,
            cloak_mode="scramble",
            preserve_syllables=True,
            break_ngrams=False
        )
        out = res.sanitized_text
        self.assertIn("Ku-a-sa-i", out)
        self.assertIn("di-ri-ku", out)
        self.assertIn("te-nang-lah", out)
        self.assertIn("ha-ti-ku", out)
        self.assertIn("meng-ha-pus", out)
        self.assertIn("je-jak-mu", out)
        self.assertLess(res.audit.ngram_overlap_pct, 15.0)

    def test_lyrics_multilingual_spanish(self):
        song = (
            "Despacito quiero respirar tu cuello despacito\n"
            "Deja que te diga cosas al oido\n"
        )
        res = self.processor.sanitize(
            text=song,
            cloak_lyrics=True,
            cloak_mode="scramble",
            preserve_syllables=True,
            break_ngrams=False
        )
        out = res.sanitized_text
        self.assertIn("Des-pa-ci-to", out)
        self.assertIn("res-pi-rar", out)
        self.assertLess(res.audit.ngram_overlap_pct, 15.0)

    def test_audit_function(self):
        orig = "one two three four five six seven eight nine ten"
        cloaked = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        aud = audit_lyrics(orig, cloaked)
        self.assertEqual(aud.ngram_overlap_pct, 0.0)
        self.assertEqual(aud.token_similarity_pct, 0.0)
        self.assertIn("EXCELLENT", aud.evasion_verdict)


class TestAudioProcessorModules(unittest.TestCase):
    """Verifies core audio processing routines and the ABS audio slicer."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sr = 48000
        t = np.linspace(0, 4.0, int(self.sr * 4.0), endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 440 * t)
        self.sample_audio = np.vstack([left, right]).T
        self.wav_path = os.path.join(self.test_dir, "test_input.wav")
        sf.write(self.wav_path, self.sample_audio, self.sr)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_slicer_creates_chunks(self):
        from ghostwave.audio_processor import AudioProcessor
        processor = AudioProcessor()
        long_t = np.linspace(0, 25.0, int(self.sr * 25.0), endpoint=False)
        long_audio = np.vstack([0.3 * np.sin(2 * np.pi * 220 * long_t)] * 2).T
        long_path = os.path.join(self.test_dir, "long_song.wav")
        sf.write(long_path, long_audio, self.sr)

        out_dir = os.path.join(self.test_dir, "chunks")
        chunks = processor.slice_audio(long_path, out_dir, chunk_duration_sec=10.0)
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertTrue(os.path.exists(c))
            data, sr = sf.read(c)
            self.assertIn(sr, (44100, 48000))
            self.assertGreater(len(data), 0)

    def test_micro_chrono_jitter(self):
        from ghostwave.audio_processor import apply_micro_chrono_jitter
        jittered = apply_micro_chrono_jitter(self.sample_audio, self.sr, max_jitter_ms=16.0)
        self.assertEqual(jittered.shape, self.sample_audio.shape)
        self.assertFalse(np.array_equal(jittered, self.sample_audio))

    def test_bode_frequency_shifter(self):
        from ghostwave.audio_processor import apply_bode_frequency_shifter
        shifted = apply_bode_frequency_shifter(self.sample_audio, self.sr, shift_hz=12.0)
        self.assertEqual(shifted.shape, self.sample_audio.shape)
        self.assertFalse(np.array_equal(shifted, self.sample_audio))

    def test_center_vocal_cut_bass_retention(self):
        from ghostwave.audio_processor import apply_total_center_vocal_annihilation
        cut = apply_total_center_vocal_annihilation(self.sample_audio, self.sr)
        self.assertEqual(cut.shape, self.sample_audio.shape)

    def test_schroeder_phase_dispersion(self):
        from ghostwave.audio_processor import apply_schroeder_phase_dispersion
        dispersed = apply_schroeder_phase_dispersion(self.sample_audio, self.sr)
        self.assertEqual(dispersed.shape, self.sample_audio.shape)

    def test_harmonic_drone_bed(self):
        from ghostwave.audio_processor import apply_continuous_camouflage_bed
        drone = apply_continuous_camouflage_bed(self.sample_audio, self.sr)
        self.assertEqual(drone.shape, self.sample_audio.shape)


class TestGhostWaveUpdater(unittest.TestCase):
    """Verifies update checking, changelog parsing, and screen reader accessible dialogs."""

    def test_normalize_version(self):
        from ghostwave.updater import normalize_version
        self.assertEqual(normalize_version("1.0.0"), (1, 0, 0))
        self.assertEqual(normalize_version("v1.2.0"), (1, 2, 0))
        self.assertEqual(normalize_version("1.0"), (1, 0, 0))
        self.assertEqual(normalize_version("v2.1.3-beta"), (2, 1, 3))

    def test_compare_versions(self):
        from ghostwave.updater import compare_versions
        # Remote newer -> 1
        self.assertEqual(compare_versions("1.0.0", "1.2.0"), 1)
        self.assertEqual(compare_versions("1.0.0", "v1.0.1"), 1)
        self.assertEqual(compare_versions("1.0.0", "2.0.0"), 1)
        # Equal -> 0
        self.assertEqual(compare_versions("1.0.0", "1.0.0"), 0)
        self.assertEqual(compare_versions("1.0.0", "v1.0.0"), 0)
        # Remote older -> -1
        self.assertEqual(compare_versions("1.2.0", "1.0.0"), -1)
        self.assertEqual(compare_versions("2.0.0", "1.9.9"), -1)

    def test_parse_changelog_text(self):
        from ghostwave.updater import parse_changelog_text
        sample_cl = (
            "Version: 1.2.0\n"
            "Release Date: 2026-10-09\n"
            "Installer: https://github.com/muhamadalfian20892/GhostWave-Studio/releases/download/v1.2.0/GhostWaveStudio-v1.2.0-Setup.exe\n"
            "Portable: https://github.com/muhamadalfian20892/GhostWave-Studio/releases/download/v1.2.0/GhostWaveStudio-v1.2.0-Portable.zip\n\n"
            "What is new:\n"
            "* Added in-app updater\n"
            "* Added changelog viewer\n"
        )
        parsed = parse_changelog_text(sample_cl)
        self.assertEqual(parsed["version"], "1.2.0")
        self.assertEqual(parsed["date"], "2026-10-09")
        self.assertIn("GhostWaveStudio-v1.2.0-Setup.exe", parsed["installer_url"])
        self.assertIn("GhostWaveStudio-v1.2.0-Portable.zip", parsed["portable_url"])
        self.assertIn("Added in-app updater", parsed["notes"])

    def test_update_info_model(self):
        from ghostwave.updater import UpdateInfo
        info = UpdateInfo(
            has_update=True,
            current_version="1.0.0",
            latest_version="1.2.0",
            release_notes="New features added.",
            download_url="https://github.com/test.exe"
        )
        self.assertTrue(info.has_update)
        self.assertEqual(info.latest_version, "1.2.0")
        self.assertEqual(info.current_version, "1.0.0")

    def test_update_dialog_accessibility_and_toggle(self):
        import wx
        from ghostwave.updater import UpdateInfo, UpdateDialog

        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        info = UpdateInfo(
            has_update=True,
            current_version="1.0.0",
            latest_version="1.2.0",
            release_notes="* Tested item 1\n* Tested item 2",
            download_url="https://github.com/example/installer.exe"
        )

        dlg = UpdateDialog(None, info)
        try:
            # Check prompt label accessibility
            self.assertEqual(dlg.prompt_label.GetName(), "Update Question Prompt")
            self.assertIn("1.2.0", dlg.prompt_label.GetLabel())

            # Check context label accessibility
            self.assertEqual(dlg.context_label.GetName(), "Current Installed Version Information")
            self.assertIn("1.0.0", dlg.context_label.GetLabel())

            # Check changelog text and header labels
            self.assertEqual(dlg.changelog_header_label.GetName(), "Changelog and Release Notes Label")
            self.assertEqual(dlg.changelog_ctrl.GetName(), "What's new Text Area")
            self.assertTrue(dlg.changelog_ctrl.IsEditable() == False)

            # Check action buttons accessibility
            self.assertEqual(dlg.see_new_btn.GetName(), "See What's New in This Changes Button")
            self.assertEqual(dlg.later_btn.GetName(), "Download Later Button")
            self.assertEqual(dlg.download_btn.GetName(), "Download Now Button")

            # Check toggle changelog visibility in the same window
            self.assertFalse(dlg.changelog_visible)
            self.assertFalse(dlg.changelog_panel.IsShown())

            # Click / trigger toggle
            dlg.on_toggle_changelog(None)
            self.assertTrue(dlg.changelog_visible)
            self.assertTrue(dlg.changelog_panel.IsShown())
            self.assertEqual(dlg.see_new_btn.GetLabel(), "&Hide What's New")
            self.assertEqual(dlg.see_new_btn.GetName(), "Hide What's New Button")

            # Both Download Later and Download Now remain available
            self.assertTrue(dlg.later_btn.IsShown())
            self.assertTrue(dlg.download_btn.IsShown())

            # Click again to collapse
            dlg.on_toggle_changelog(None)
            self.assertFalse(dlg.changelog_visible)
            self.assertFalse(dlg.changelog_panel.IsShown())
            self.assertEqual(dlg.see_new_btn.GetLabel(), "&See What's New in This Changes")
            self.assertEqual(dlg.see_new_btn.GetName(), "See What's New in This Changes Button")

        finally:
            dlg.Destroy()

    def test_update_download_dialog_accessibility(self):
        import wx
        from ghostwave.updater import UpdateDownloadDialog

        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        dlg = UpdateDownloadDialog(None, "https://example.com/file.exe", "file.exe", "1.2.0")
        try:
            self.assertEqual(dlg.status_label.GetName(), "Download Status Description")
            self.assertEqual(dlg.gauge.GetName(), "Download Progress Meter")
            self.assertEqual(dlg.progress_label.GetName(), "Download Progress Details Label")
            self.assertEqual(dlg.cancel_btn.GetName(), "Cancel Download Button")
        finally:
            dlg.Destroy()


class TestVersion120UpgradesAndPersonas(unittest.TestCase):
    """
    Validates v1.2.0 features, optimizations, and behaviors across
    standard user, developer, and constrained hardware ('lagger') personas.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        from ghostwave.i18n import set_language
        set_language("en")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_wav(self, duration_sec: float = 1.0, sr: int = 44100) -> str:
        path = os.path.join(self.test_dir, f"test_{int(duration_sec*1000)}.wav")
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
        signal_l = 0.5 * np.sin(2 * np.pi * 440 * t)
        signal_r = 0.5 * np.cos(2 * np.pi * 554.37 * t)
        data = np.column_stack([signal_l, signal_r]).astype(np.float32)
        sf.write(path, data, sr, subtype="PCM_16")
        return path

    # Perspective 1: Standard User Persona
    def test_trimmer_disabled_by_default(self):
        """Ensures trimmer is unchecked and disabled by default across options and configuration."""
        from ghostwave.audio_processor import AudioSanitizeOptions
        from ghostwave.config_manager import GhostWaveConfig

        opts = AudioSanitizeOptions()
        self.assertFalse(opts.trim_duration)

        cfg = GhostWaveConfig()
        self.assertFalse(cfg.trim_duration)

    def test_gui_advanced_settings_toggle_and_trimmer_default(self):
        """Verifies collapsible advanced settings toggle and trimmer unchecked state in UI."""
        import wx
        from ghostwave.gui import AudioSanitizerPanel
        from ghostwave.audio_processor import AudioProcessor
        from ghostwave.i18n import set_language, tr

        set_language("en")

        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        frame = wx.Frame(None)
        processor = AudioProcessor()
        panel = AudioSanitizerPanel(frame, processor, lambda msg: None)
        try:
            # 1. Trimmer checkbox is unchecked by default
            self.assertFalse(panel.trim_chk.IsChecked())
            self.assertFalse(panel.trim_spin.IsEnabled())

            # 2. Advanced panel is collapsed by default to avoid clutter
            self.assertFalse(panel.adv_panel.IsShown())
            self.assertEqual(panel.toggle_adv_btn.GetLabel(), tr("AUDIO_SHOW_ADVANCED_BTN"))
            self.assertEqual(panel.toggle_adv_btn.GetName(), "Show Advanced Settings Button")

            # 3. Toggling button expands advanced panel
            panel.on_toggle_advanced(None)
            self.assertTrue(panel.adv_panel.IsShown())
            self.assertEqual(panel.toggle_adv_btn.GetLabel(), tr("AUDIO_HIDE_ADVANCED_BTN"))

            # 4. Toggling again collapses it
            panel.on_toggle_advanced(None)
            self.assertFalse(panel.adv_panel.IsShown())
            self.assertEqual(panel.toggle_adv_btn.GetLabel(), tr("AUDIO_SHOW_ADVANCED_BTN"))

            # 5. Switching presets keeps trimmer unchecked
            for preset_idx in range(6):
                panel.preset_choice.SetSelection(preset_idx)
                panel.on_preset_changed(None)
                self.assertFalse(panel.trim_chk.IsChecked())
                self.assertFalse(panel.trim_spin.IsEnabled())
        finally:
            frame.Destroy()

    def test_gui_lyrics_sanitizer_panel_and_main_frame_instantiation(self):
        """Verifies LyricsSanitizerPanel and GhostWaveFrame full UI construction without errors."""
        import wx
        from ghostwave.gui import LyricsSanitizerPanel, GhostWaveFrame
        from ghostwave.lyrics_processor import LyricsProcessor

        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        frame = wx.Frame(None)
        processor = LyricsProcessor()
        panel = LyricsSanitizerPanel(frame, processor, lambda msg: None)
        try:
            self.assertIsNotNone(panel.mode_lbl)
            self.assertEqual(panel.mode_lbl.GetName(), "Cloaking Strategy Label")
            self.assertIsNotNone(panel.changes_label)
            self.assertEqual(panel.changes_label.GetName(), "Changes Made Label")
            self.assertTrue(panel.cloak_chk.IsChecked())
        finally:
            frame.Destroy()

        main_win = GhostWaveFrame()
        try:
            self.assertIsNotNone(main_win.audio_tab)
            self.assertIsNotNone(main_win.lyrics_tab)
            self.assertIsNotNone(main_win.notebook)
        finally:
            main_win.Destroy()


    # Perspective 2: Developer Persona (DSP vectorization, caching, benchmarks)
    def test_dsp_filter_caching(self):
        """Validates that filter design functions leverage LRU caching."""
        from ghostwave.audio_processor import _get_cached_butter_sos, _get_cached_bilinear

        # Butterworth caching
        sos1 = _get_cached_butter_sos(2, 25.0, "hp", 44100)
        sos2 = _get_cached_butter_sos(2, 25.0, "hp", 44100)
        self.assertIs(sos1, sos2)

        # Bilinear biquad caching
        b1, a1 = _get_cached_bilinear(500.0, 0.707, 44100)
        b2, a2 = _get_cached_bilinear(500.0, 0.707, 44100)
        self.assertIs(b1, b2)
        self.assertIs(a1, a2)

    def test_fast_fft_padding_in_bode_shifter(self):
        """Verifies Hilbert Bode frequency shifter with arbitrary non-power-of-2 lengths."""
        from ghostwave.audio_processor import apply_bode_frequency_shifter, apply_triple_band_bode_shifter

        sr = 44100
        samples = 44101
        data = np.random.uniform(-0.5, 0.5, (samples, 2)).astype(np.float32)

        shifted_single = apply_bode_frequency_shifter(data, sr=sr, shift_hz=8.5)
        self.assertEqual(shifted_single.shape, data.shape)
        self.assertTrue(np.all(np.isfinite(shifted_single)))

        shifted_multi = apply_triple_band_bode_shifter(data, sr=sr)
        self.assertEqual(shifted_multi.shape, data.shape)
        self.assertTrue(np.all(np.isfinite(shifted_multi)))

    def test_adversarial_peaks_vectorized(self):
        """Validates vectorized STFT adversarial peak injection produces finite decoy landmarks."""
        from ghostwave.audio_processor import apply_adversarial_peak_injection

        sr = 44100
        data = np.random.uniform(-0.4, 0.4, (22050, 2)).astype(np.float32)
        injected = apply_adversarial_peak_injection(data, sr=sr, intensity=1.15)
        self.assertEqual(injected.shape, data.shape)
        self.assertTrue(np.all(np.isfinite(injected)))

    def test_single_pass_regex_blacklist(self):
        """Validates single-pass regex compilation correctly masks names and profanities."""
        from ghostwave.lyrics_processor import LyricsProcessor

        lp = LyricsProcessor()
        text = "Taylor Swift and Drake met Eminem in Hollywood."
        filtered, changes = lp.filter_celebrities(text)
        self.assertNotIn("Taylor Swift", filtered)
        self.assertNotIn("Drake", filtered)
        self.assertNotIn("Eminem", filtered)
        self.assertGreaterEqual(len(changes), 3)

    # Perspective 3: Constrained / Lagging Environment Persona
    def test_cooperative_cancellation(self):
        """Verifies DSP processing cleanly aborts when cancellation signal is asserted."""
        import threading
        from ghostwave.audio_processor import AudioProcessor, AudioSanitizeOptions

        input_wav = self._create_synthetic_wav(duration_sec=3.0)
        output_wav = os.path.join(self.test_dir, "cancelled_out.wav")

        processor = AudioProcessor()
        cancel_event = threading.Event()
        cancel_event.set()

        opts = AudioSanitizeOptions()
        result = processor.process(input_wav, output_wav, opts, cancel_event=cancel_event)
        self.assertFalse(result.success)
        self.assertIn("cancelled", (result.error_message or "").lower())
        self.assertFalse(os.path.exists(output_wav))

    def test_soft_limiter_prevents_clipping(self):
        """Verifies soft limiter bounds extreme amplitudes within [-1.0, 1.0]."""
        from ghostwave.audio_processor import apply_soft_limiting

        extreme_signal = np.array([[2.5, -3.8], [5.0, -10.0]], dtype=np.float32)
        limited = apply_soft_limiting(extreme_signal, drive_db=3.0)
        self.assertTrue(np.all(limited <= 1.0))
        self.assertTrue(np.all(limited >= -1.0))

    def test_zero_byte_file_protection(self):
        """Verifies zero-byte files are handled gracefully without exceptions."""
        from ghostwave.audio_processor import AudioProcessor

        empty_file = os.path.join(self.test_dir, "empty.wav")
        open(empty_file, "w").close()

        processor = AudioProcessor()
        info = processor.inspect_file(empty_file)
        self.assertEqual(info.duration_seconds, 0.0)
        self.assertEqual(info.duration_formatted, "00:00")

    def test_dc_offset_removal(self):
        """Verifies load_audio_samples removes DC offset from audio signals."""
        from ghostwave.audio_processor import load_audio_samples
        test_wav = os.path.join(self.test_dir, "dc_test.wav")
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        signal_with_dc = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32) * 0.3 + 0.5
        sf.write(test_wav, signal_with_dc, sr)

        loaded, loaded_sr = load_audio_samples(test_wav, target_sr=sr)
        self.assertEqual(loaded_sr, sr)
        mean_offset = float(np.mean(loaded))
        self.assertAlmostEqual(mean_offset, 0.0, places=4)

    def test_zero_jitter_and_dispersion_guards(self):
        """Verifies zero jitter and empty dispersion center freqs return untouched buffers."""
        from ghostwave.audio_processor import apply_micro_chrono_jitter, apply_schroeder_phase_dispersion
        sr = 44100
        data = np.ones((1000, 2), dtype=np.float32) * 0.5
        out_jitter = apply_micro_chrono_jitter(data, sr=sr, max_jitter_ms=0.0)
        np.testing.assert_array_equal(out_jitter, data)

        out_disp = apply_schroeder_phase_dispersion(data, sr=sr, center_freqs=())
        np.testing.assert_array_equal(out_disp, data)

    def test_syllabify_lines_batch(self):
        """Verifies batch syllabification preserves tags and splits lyrics words."""
        from ghostwave.lyrics_processor import syllabify_lines
        input_lyrics = "[Verse 1]\nMalam sunyi bintang bersinar"
        result = syllabify_lines(input_lyrics)
        lines = result.splitlines()
        self.assertEqual(lines[0], "[Verse 1]")
        self.assertIn("Ma-lam", lines[1])
        self.assertIn("su-nyi", lines[1])
        self.assertIn("bin-tang", lines[1])

    def test_changelog_has_no_download_urls(self):
        """Verifies changelog.txt contains strictly zero direct installer/portable download links."""
        changelog_path = os.path.join(ROOT_DIR, "changelog.txt")
        self.assertTrue(os.path.exists(changelog_path))
        with open(changelog_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("Installer:", content)
        self.assertNotIn("Portable:", content)
        self.assertNotIn(".exe", content)
        self.assertNotIn(".zip", content)

    def test_readme_txt_pure_plain_text(self):
        """Verifies README.txt contains zero markdown syntax, backticks, em dashes, or emojis."""
        readme_path = os.path.join(ROOT_DIR, "docs", "README.txt")
        self.assertTrue(os.path.exists(readme_path))
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertGreater(len(content), 100)
        self.assertNotIn("```", content)
        self.assertNotIn("`", content)
        self.assertNotIn("##", content)
        self.assertNotIn("**", content)
        self.assertNotIn("—", content)
        non_ascii = [ch for ch in content if ord(ch) > 127]
        self.assertEqual(len(non_ascii), 0, f"Found non-ASCII characters: {set(non_ascii)}")
        self.assertIn("http://technokerslab.blogspot.com/", content)
        self.assertIn("hafiyanajah@gmail.com", content)
        self.assertIn("https://github.com/muhamadalfian20892/GhostWave-Studio", content)

    def test_readme_id_txt_pure_plain_text(self):
        """Verifies README_ID.txt contains zero markdown syntax, backticks, em dashes, or emojis."""
        readme_path = os.path.join(ROOT_DIR, "docs", "README_ID.txt")
        self.assertTrue(os.path.exists(readme_path))
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertGreater(len(content), 100)
        self.assertNotIn("```", content)
        self.assertNotIn("`", content)
        self.assertNotIn("##", content)
        self.assertNotIn("**", content)
        self.assertNotIn("—", content)
        non_ascii = [ch for ch in content if ord(ch) > 127]
        self.assertEqual(len(non_ascii), 0, f"Found non-ASCII characters: {set(non_ascii)}")
        self.assertIn("http://technokerslab.blogspot.com/", content)
        self.assertIn("hafiyanajah@gmail.com", content)
        self.assertIn("https://github.com/muhamadalfian20892/GhostWave-Studio", content)

    def test_i18n_catalogs_and_switching(self):
        """Verifies language catalogs, fallback behavior, and key parity between en and id."""
        from ghostwave.i18n import init_translations, set_language, get_language, tr, _load_catalog
        en_cat = _load_catalog("en")
        id_cat = _load_catalog("id")
        self.assertGreater(len(en_cat), 20)
        self.assertGreater(len(id_cat), 20)
        # Verify key parity
        missing_in_id = set(en_cat.keys()) - set(id_cat.keys())
        self.assertEqual(len(missing_in_id), 0, f"Keys missing in id.json: {missing_in_id}")

        init_translations("en")
        self.assertEqual(get_language(), "en")
        self.assertEqual(tr("MENU_FILE"), "&File")

        set_language("id")
        self.assertEqual(get_language(), "id")
        self.assertEqual(tr("MENU_FILE"), "&Berkas")
        # Test variable formatting
        stat_msg = tr("LYRICS_STATS_FORMAT", chars=10, words=2, lines=1)
        self.assertIn("10", stat_msg)
        self.assertIn("Karakter:", stat_msg)

        # Reset back to en
        set_language("en")

    def test_support_client_diagnostics(self):
        """Verifies ticket key generation and system diagnostics collection."""
        from ghostwave.support_client import generate_ticket_key, get_system_diagnostics
        key1 = generate_ticket_key()
        key2 = generate_ticket_key()
        self.assertEqual(len(key1), 32)
        self.assertNotEqual(key1, key2)

        diag = get_system_diagnostics("1.4.0")
        self.assertIn("OS:", diag)
        self.assertIn("App Version: 1.4.0", diag)

    def test_config_manager_v1_6_2(self):
        """Verifies v1.6.2/v1.7.0 configuration fields, default presets, and ticket persistence."""
        from ghostwave.config_manager import GhostWaveConfig
        cfg = GhostWaveConfig()
        self.assertIn(cfg.app_version, ["1.6.2", "1.7.0"])
        self.assertEqual(cfg.language, "en")
        self.assertEqual(cfg.default_preset, "Complete Sanitization (Recommended)")
        self.assertTrue(cfg.first_run)
        self.assertEqual(len(cfg.user_tickets), 0)

        cfg.add_user_ticket({"ticket_id": 42, "title": "Test Ticket", "status": "open"})
        self.assertEqual(len(cfg.user_tickets), 1)
        self.assertEqual(cfg.user_tickets[0]["ticket_id"], 42)

    def test_changelog_caching_and_clean_title(self):
        """Verifies changelog caching and clean title without extra tagline."""
        import wx
        from ghostwave.updater import get_cached_changelog_text
        from ghostwave.gui import GhostWaveFrame, ChangelogDialog

        notes = get_cached_changelog_text()
        self.assertIn("GhostWave Studio", notes)

        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        frame = GhostWaveFrame()
        try:
            self.assertIn(frame.GetTitle(), ["GhostWave Studio v1.6.2", "GhostWave Studio v1.7.0"])
            self.assertIsNotNone(frame.changelog_item)
        finally:
            frame.Destroy()

        dlg = ChangelogDialog()
        try:
            self.assertTrue(dlg.text_ctrl.IsEditable() is False)
            self.assertIsNotNone(dlg.copy_btn)
            self.assertIsNotNone(dlg.refresh_btn)
            self.assertIsNotNone(dlg.close_btn)
        finally:
            dlg.Destroy()

    def test_accessibility_guide_and_about_dialogs_structure(self):
        """Verifies AccessibilityGuideDialog and AboutDialog are structured with read-only text and close buttons."""
        import wx
        from ghostwave.gui import AccessibilityGuideDialog, AboutDialog
        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        dlg = AccessibilityGuideDialog()
        self.assertIn("GhostWave Studio", dlg.text_ctrl.GetValue())
        self.assertIn("http://technokerslab.blogspot.com/", dlg.text_ctrl.GetValue())
        self.assertIn("README.txt", dlg.text_ctrl.GetValue())
        self.assertNotIn("github.com", dlg.text_ctrl.GetValue().lower())
        self.assertTrue(dlg.text_ctrl.IsEditable() is False)
        self.assertIsNotNone(dlg.close_btn)
        self.assertIsNotNone(dlg.copy_btn)
        dlg.Destroy()

        about_dlg = AboutDialog()
        self.assertIn("GhostWave Studio", about_dlg.text_ctrl.GetValue())
        self.assertIn("http://technokerslab.blogspot.com/", about_dlg.text_ctrl.GetValue())
        self.assertIn("README.txt", about_dlg.text_ctrl.GetValue())
        self.assertNotIn("github.com", about_dlg.text_ctrl.GetValue().lower())
        self.assertTrue(about_dlg.text_ctrl.IsEditable() is False)
        self.assertIsNotNone(about_dlg.close_btn)
        about_dlg.Destroy()

    def test_v1_6_1_dialog_and_tooltip_translations(self):
        """Verifies Indonesian localization for Suno protocol, About, and dynamic tooltips."""
        import wx
        from ghostwave.i18n import init_translations, set_language, tr
        from ghostwave.gui import SunoCheatSheetDialog, AboutDialog, AudioSanitizerPanel
        from ghostwave.audio_processor import AudioProcessor

        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        init_translations("id")
        set_language("id")

        frame = wx.Frame(None)
        processor = AudioProcessor()
        try:
            suno_dlg = SunoCheatSheetDialog(frame)
            try:
                val = suno_dlg.text_ctrl.GetValue()
                self.assertIn("PROTOKOL UNGGAH AUDIO KE SUNO AI", val)
                self.assertIn("SELALU EKSPOR SEBAGAI WAV", val)
            finally:
                suno_dlg.Destroy()

            about_dlg = AboutDialog(frame)
            try:
                val = about_dlg.text_ctrl.GetValue()
                self.assertIn("Aplikasi desktop yang mudah diakses", val)
                self.assertIn("Profil Pengaturan", val)
            finally:
                about_dlg.Destroy()

            audio_panel = AudioSanitizerPanel(frame, processor, lambda msg: None)
            try:
                audio_panel.retranslate_ui()
                tip = audio_panel.browse_btn.GetToolTipText()
                self.assertEqual(tip, tr("TIP_AUDIO_BROWSE_BTN"))
            finally:
                audio_panel.Destroy()
        finally:
            frame.Destroy()
            set_language("en")

    def test_v1_6_2_bugfixes(self):
        """Verifies v1.6.2 DSP safeguards, international lyrics preservation, and unescaped newlines."""
        import numpy as np
        from ghostwave.i18n import init_translations, set_language, tr
        from ghostwave.audio_processor import (
            apply_virtual_acoustic_reamping,
            apply_adversarial_peak_injection,
            apply_eq_and_dither,
            slice_audio_for_suno,
            export_clean_audio
        )
        from ghostwave.lyrics_processor import LyricsProcessor, syllabify_lines, LyricsCloaker

        # 1. Test unescaped newlines in translations
        init_translations("id")
        set_language("id")
        cloud_info = tr("CLOUD_API_INFO")
        self.assertNotIn(r"\n", cloud_info)
        self.assertIn("\n", cloud_info)
        set_language("en")

        # 2. Test re-amping with 8 kHz sample rate (Nyquist guard)
        sig_8k = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 4000)).astype(np.float32)
        reamped = apply_virtual_acoustic_reamping(sig_8k, sr=8000)
        self.assertEqual(len(reamped), len(sig_8k))

        # 3. Test peak injection on short audio (< nperseg)
        short_sig = np.sin(2 * np.pi * 440 * np.linspace(0, 0.02, 882)).astype(np.float32)
        injected = apply_adversarial_peak_injection(short_sig, sr=44100)
        self.assertEqual(len(injected), len(short_sig))

        # 4. Test EQ and dither with low sample rate (16 kHz)
        sig_16k = np.sin(2 * np.pi * 440 * np.linspace(0, 0.2, 3200)).astype(np.float32)
        dithered = apply_eq_and_dither(sig_16k, sr=16000)
        self.assertEqual(len(dithered), len(sig_16k))

        # 5. Test international lyrics preservation in normalize_unicode
        processor = LyricsProcessor()
        japanese_text = "こんにちは世界 (Hello World)"
        norm_jp, _ = processor.normalize_unicode(japanese_text)
        self.assertIn("こんにちは世界", norm_jp)

        # 6. Test structural tags formatting with parentheses, colons, and attached numbers
        tags_raw = "[Guitar Solo (Fast)]\n[verse1]\n[Verse: Acoustic]"
        fmt_tags, _ = processor.format_structural_tags(tags_raw)
        self.assertIn("[Guitar Solo (Fast)]", fmt_tags)
        self.assertIn("[Verse 1]", fmt_tags)
        self.assertIn("[Verse: Acoustic]", fmt_tags)

        # 7. Test syllabify_lines protecting inline structural tags
        mixed_line = "Singing a song [Chorus] in the rain"
        syll_res = syllabify_lines(mixed_line)
        self.assertIn("[Chorus]", syll_res)

        # 8. Test quote spacing in cloaker
        quoted_text = '"Hello" world'
        cloaked, _ = LyricsCloaker.cloak(quoted_text, mode="scramble", add_vibrato_glides=False)
        self.assertIn('" ', cloaked)

    def test_vocal_cover_and_articulation_softener(self):
        """Verifies Vocal Cover Mode preserves melody and pitch while softening consonant articulation."""
        from ghostwave.audio_processor import (
            apply_vocal_articulation_softener,
            apply_front_and_tail_padding,
            sanitize_audio,
            AudioSanitizeOptions
        )

        # 1. Test consonant articulation softening
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        vowel_tone = 0.4 * np.sin(2 * np.pi * 440 * t)
        consonant_spike = 0.5 * np.sin(2 * np.pi * 5000 * t) * (np.sin(2 * np.pi * 8 * t) > 0.8).astype(np.float32)
        test_audio = np.column_stack([vowel_tone + consonant_spike, vowel_tone + consonant_spike]).astype(np.float32)

        softened = apply_vocal_articulation_softener(test_audio, sr, intensity=0.75)
        self.assertEqual(softened.shape, test_audio.shape)
        self.assertTrue(np.isfinite(softened).all())
        self.assertLessEqual(np.max(np.abs(softened)), np.max(np.abs(test_audio)) + 1e-3)

        # 2. Test clean silence padding without 140 Hz buzz tone
        padded_silence = apply_front_and_tail_padding(test_audio, sr, front_sec=2.0, tail_sec=1.0, use_ambient_intro=False)
        front_samples = int(2.0 * sr)
        self.assertEqual(float(np.max(np.abs(padded_silence[:front_samples]))), 0.0)

        # 3. Test apply_melodic_guide_vocal_cloaker preserves melody while de-articulating consonants
        from ghostwave.audio_processor import apply_melodic_guide_vocal_cloaker
        cloaked_guide = apply_melodic_guide_vocal_cloaker(test_audio, sr, intensity=0.85, consonant_cutoff_hz=2400.0, vocal_ducking_db=-3.0)
        self.assertEqual(cloaked_guide.shape, test_audio.shape)
        self.assertTrue(np.isfinite(cloaked_guide).all())

        # 4. Test sanitize_audio in vocal_cover mode preserves vocal melody energy
        in_wav = self._create_synthetic_wav(duration_sec=3.0, sr=sr)
        out_cover = os.path.join(self.test_dir, "out_cover.wav")
        res_cover = sanitize_audio(in_wav, out_cover, mode="vocal_cover", front_padding_seconds=0.0, tail_padding_seconds=0.0)
        self.assertTrue(res_cover.success)
        self.assertTrue(os.path.exists(out_cover))
        data_cov, _ = sf.read(out_cover)
        rms_cover = np.sqrt(np.mean(data_cov ** 2))
        self.assertGreater(rms_cover, 0.05)


class TestNewModularFeatures(unittest.TestCase):
    """
    Tests modern 2026 features:
    - Key & BPM detection with harmonious recommendations
    - AI Watermark Neutralizer (Google SynthID / AudioSeal)
    - Psychoacoustic adversarial masking
    - Formant shift
    - Local DSP stem separation
    - Pre-flight evasion simulation
    - Batch processor & Folder watcher
    - Suno style prompt generator
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sr = 44100
        t = np.linspace(0, 2.5, int(self.sr * 2.5), endpoint=False, dtype=np.float32)
        beat = (np.sin(2 * np.pi * 2.0 * t) > 0.8).astype(np.float32)
        tone = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * beat
        self.audio = np.column_stack([tone, tone]).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_key_and_bpm_detection(self):
        from ghostwave.audio_processor import detect_key_and_bpm
        bpm, key, recs = detect_key_and_bpm(self.audio, self.sr)
        self.assertIsInstance(bpm, float)
        self.assertGreater(bpm, 0)
        self.assertIsInstance(key, str)
        self.assertGreater(len(key), 0)
        self.assertIsInstance(recs, list)
        self.assertGreaterEqual(len(recs), 1)
        for shift_val, target_name in recs:
            self.assertIsInstance(shift_val, float)
            self.assertIsInstance(target_name, str)

    def test_ai_watermark_neutralizer(self):
        from ghostwave.audio_processor import apply_ai_watermark_neutralizer
        cleaned = apply_ai_watermark_neutralizer(self.audio, self.sr, intensity=0.85)
        self.assertEqual(cleaned.shape, self.audio.shape)
        self.assertTrue(np.isfinite(cleaned).all())
        rms_in = np.sqrt(np.mean(self.audio ** 2))
        rms_out = np.sqrt(np.mean(cleaned ** 2))
        self.assertAlmostEqual(rms_in, rms_out, delta=0.20)

    def test_psychoacoustic_masking(self):
        from ghostwave.audio_processor import calculate_psychoacoustic_mask, apply_adversarial_peak_injection
        mask = calculate_psychoacoustic_mask(self.sr, n_fft=2048)
        self.assertEqual(len(mask), 1025)
        self.assertTrue(np.isfinite(mask).all())
        self.assertTrue((mask > 0).all())

        injected = apply_adversarial_peak_injection(self.audio, self.sr, use_psychoacoustic_mask=True)
        self.assertEqual(injected.shape, self.audio.shape)
        self.assertTrue(np.isfinite(injected).all())

    def test_formant_shift(self):
        from ghostwave.audio_processor import apply_formant_shift
        shifted = apply_formant_shift(self.audio, self.sr, formant_semitones=2.0)
        self.assertEqual(shifted.shape, self.audio.shape)
        self.assertTrue(np.isfinite(shifted).all())
        self.assertLessEqual(np.max(np.abs(shifted)), 1.5)

    def test_stem_separation_dsp(self):
        from ghostwave.audio_processor import separate_stems_local_dsp
        vocals, instrumental = separate_stems_local_dsp(self.audio, self.sr)
        self.assertEqual(vocals.shape, self.audio.shape)
        self.assertEqual(instrumental.shape, self.audio.shape)
        self.assertTrue(np.isfinite(vocals).all())
        self.assertTrue(np.isfinite(instrumental).all())

    def test_preflight_evasion_simulator(self):
        from ghostwave.audio_processor import simulate_preflight_evasion, sanitize_audio, AudioSanitizeOptions
        in_path = os.path.join(self.test_dir, "preflight_in.wav")
        out_path = os.path.join(self.test_dir, "preflight_out.wav")
        sf.write(in_path, self.audio, self.sr)

        opts = AudioSanitizeOptions(
            pitch_shift_semitones=2.0,
            tempo_factor=0.940,
            enable_ai_watermark_neutralizer=True
        )
        res = sanitize_audio(in_path, out_path, options=opts)
        self.assertTrue(res.success)

        report = simulate_preflight_evasion(in_path, out_path, sr=self.sr)
        self.assertIn("safety_score", report)
        self.assertGreaterEqual(report["safety_score"], 0)
        self.assertLessEqual(report["safety_score"], 100)
        self.assertIn("verdict", report)
        self.assertIn("cqt_chroma_similarity_pct", report)
        self.assertIn("landmark_overlap_pct", report)
        self.assertIn("actionable_advice", report)

    def test_batch_processor_and_folder_watcher(self):
        from ghostwave.batch_processor import BatchProcessor, BatchTask, FolderWatcher
        from ghostwave.audio_processor import AudioProcessor, AudioSanitizeOptions

        p1 = os.path.join(self.test_dir, "song1.wav")
        p2 = os.path.join(self.test_dir, "song2.wav")
        out1 = os.path.join(self.test_dir, "song1_out.wav")
        out2 = os.path.join(self.test_dir, "song2_out.wav")
        sf.write(p1, self.audio, self.sr)
        sf.write(p2, self.audio, self.sr)

        tasks = [BatchTask(p1, out1), BatchTask(p2, out2)]
        bproc = BatchProcessor(AudioProcessor())
        results = bproc.process_queue(tasks, AudioSanitizeOptions())
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].success)
        self.assertTrue(results[1].success)
        self.assertTrue(os.path.exists(out1))
        self.assertTrue(os.path.exists(out2))

        watch_in = os.path.join(self.test_dir, "watch_in")
        watch_out = os.path.join(self.test_dir, "watch_out")
        os.makedirs(watch_in, exist_ok=True)
        os.makedirs(watch_out, exist_ok=True)

        watcher = FolderWatcher(watch_in, watch_out, audio_processor=AudioProcessor(), poll_interval_sec=0.1)
        self.assertFalse(watcher.is_running)
        watcher.start()
        self.assertTrue(watcher.is_running)
        watcher.stop()
        self.assertFalse(watcher.is_running)

        # Legacy positional AudioProcessor argument remains supported
        legacy_watcher = FolderWatcher(watch_in, watch_out, AudioProcessor(), poll_interval_sec=0.1)
        self.assertIsNotNone(legacy_watcher.audio_processor)

    def test_suno_prompt_generator(self):
        from ghostwave.lyrics_processor import generate_suno_style_prompt
        prompt = generate_suno_style_prompt(bpm=128, key="D Minor", mood="Energetic", user_genre="Synthwave")
        self.assertIsInstance(prompt, str)
        self.assertIn("Synthwave", prompt)
        self.assertIn("128 BPM", prompt)
        self.assertIn("[Verse 1]", prompt)
        self.assertIn("Chorus", prompt)


class TestGitHubCIChecker(unittest.TestCase):
    """
    Automated GitHub CI checker suite verifying:
    - AudioSanitizeOptions kwargs synchronization and backward compatibility
    - Verification of NVDA accessible labels on all input/edit/spin controls
    - Verification that no Indonesian leak occurs when app is set to English
    - Full localization key parity between en.json and id.json
    - Presets integrity and valid AudioSanitizeOptions generation
    - End-to-end audio processing with modular 2026 flags
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sr = 44100
        t = np.linspace(0, 1.5, int(self.sr * 1.5), endpoint=False, dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)
        self.audio = np.column_stack([tone, tone]).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ci_audio_sanitize_options_kwargs_and_sync(self):
        """Verifies room_reverb_wet and vocal_reduction_depth work seamlessly without TypeError."""
        from ghostwave.audio_processor import AudioSanitizeOptions

        # Direct instantiation with modern modular keywords
        opts = AudioSanitizeOptions(
            room_reverb_wet=0.22,
            vocal_reduction_depth=14.0,
            enable_ai_watermark_neutralizer=True,
            formant_shift_semitones=1.5,
            enable_formant_shift=True,
            enable_stem_separation=False
        )
        self.assertEqual(opts.room_reverb_wet, 0.22)
        self.assertEqual(opts.reamping_wet_mix, 0.22)
        self.assertEqual(opts.vocal_reduction_depth, 14.0)
        self.assertAlmostEqual(opts.vocal_cut_depth, 14.0 / 24.0, places=3)

        # Reverse instantiation with legacy reamping_wet_mix and vocal_cut_depth
        opts_legacy = AudioSanitizeOptions(
            reamping_wet_mix=0.35,
            vocal_cut_depth=0.5
        )
        self.assertEqual(opts_legacy.reamping_wet_mix, 0.35)
        self.assertEqual(opts_legacy.room_reverb_wet, 0.35)
        self.assertEqual(opts_legacy.vocal_cut_depth, 0.5)
        self.assertEqual(opts_legacy.vocal_reduction_depth, 12.0)

    def test_ci_locale_parity_and_cleanliness(self):
        """Verifies 100% key parity between en.json and id.json without empty or unparsed strings."""
        import json
        locales_dir = os.path.join(ROOT_DIR, "locales")
        en_path = os.path.join(locales_dir, "en.json")
        id_path = os.path.join(locales_dir, "id.json")

        self.assertTrue(os.path.exists(en_path))
        self.assertTrue(os.path.exists(id_path))

        with open(en_path, "r", encoding="utf-8") as f:
            en_dict = json.load(f)
        with open(id_path, "r", encoding="utf-8") as f:
            id_dict = json.load(f)

        en_keys = set(en_dict.keys())
        id_keys = set(id_dict.keys())

        missing_in_id = en_keys - id_keys
        missing_in_en = id_keys - en_keys

        self.assertEqual(missing_in_id, set(), f"Keys missing in id.json: {missing_in_id}")
        self.assertEqual(missing_in_en, set(), f"Keys missing in en.json: {missing_in_en}")

        # Check for unescaped literal \\n in values
        for k, v in en_dict.items():
            if isinstance(v, str):
                self.assertNotIn(r"\n", v, f"Literal \\n found in en.json key: {k}")
                self.assertTrue(len(v.strip()) > 0, f"Empty string in en.json key: {k}")
        for k, v in id_dict.items():
            if isinstance(v, str):
                self.assertNotIn(r"\n", v, f"Literal \\n found in id.json key: {k}")
                self.assertTrue(len(v.strip()) > 0, f"Empty string in id.json key: {k}")

    def test_ci_english_mode_no_indonesian_leak(self):
        """Verifies that in English mode, UI elements do not display Indonesian words."""
        import wx
        from ghostwave.i18n import init_translations, set_language
        from ghostwave.gui import GhostWaveFrame
        from ghostwave.audio_processor import simulate_preflight_evasion

        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        init_translations("en")
        set_language("en")

        frame = GhostWaveFrame()
        try:
            frame.retranslate_ui()
            audio_tab = frame.audio_tab

            # Verify options build cleanly
            opts = audio_tab._get_current_options()
            self.assertIsNotNone(opts)

            # Indonesian word blacklist to detect accidental leakage in English UI
            id_leak_words = [
                "Masukkan", "Berkas Audio", "Pengaturan Lanjutan", "Pratinjau",
                "Deteksi", "Gagal", "Selesai", "Lirik", "Riwayat",
                "Pusat Bantuan", "Lanjutkan"
            ]

            # Check audio tab labels
            controls_to_check = [
                audio_tab.file_label, audio_tab.preset_label,
                audio_tab.pitch_lbl, audio_tab.formant_lbl,
                audio_tab.tempo_lbl, audio_tab.reverb_lbl,
                audio_tab.vocal_duck_lbl, audio_tab.watermark_chk,
                audio_tab.psycho_chk
            ]
            for ctrl in controls_to_check:
                lbl = ctrl.GetLabel()
                for bad_word in id_leak_words:
                    self.assertNotIn(bad_word, lbl, f"Indonesian leak '{bad_word}' found in label: '{lbl}'")

            # Check preflight evasion simulator report in EN mode
            in_wav = os.path.join(self.test_dir, "test_preflight_in.wav")
            out_wav = os.path.join(self.test_dir, "test_preflight_out.wav")
            sf.write(in_wav, self.audio, self.sr)
            sf.write(out_wav, self.audio, self.sr)

            report = simulate_preflight_evasion(in_wav, out_wav, sr=self.sr)
            verdict_en = report.get("verdict_en", "")
            self.assertNotIn("Tinggi", verdict_en)
            self.assertNotIn("Sedang", verdict_en)
            self.assertNotIn("Rendah", verdict_en)
            self.assertIn(verdict_en, ["HIGH EVASION SAFETY", "MODERATE EVASION", "DETECTION RISK"])
        finally:
            frame.Destroy()

    def test_ci_nvda_accessible_control_labels(self):
        """Verifies that all input editboxes and spin controls have accessible names for NVDA."""
        import wx
        from ghostwave.i18n import init_translations, set_language
        from ghostwave.gui import (
            GhostWaveFrame, PreFlightEvasionDialog, BatchProcessingDialog,
            SunoPromptDialog, AbsAudioSlicerDialog, CloudApiDialog,
            LyricsCloudDialog, BlacklistDialog, SupportTicketDialog
        )

        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        init_translations("en")
        set_language("en")

        frame = GhostWaveFrame()
        try:
            audio_tab = frame.audio_tab
            lyrics_tab = frame.lyrics_tab

            # 1. Audio tab controls
            edit_and_spin_controls = [
                audio_tab.file_path_ctrl,
                audio_tab.pitch_spin,
                audio_tab.formant_spin,
                audio_tab.tempo_spin,
                audio_tab.reverb_spin,
                audio_tab.vocal_duck_spin,
                audio_tab.trim_spin
            ]

            for ctrl in edit_and_spin_controls:
                name = ctrl.GetName()
                self.assertTrue(len(name.strip()) > 0, f"Control {ctrl} has empty GetName()")
                acc = ctrl.GetAccessible()
                self.assertIsNotNone(acc, f"Control {ctrl} missing wx.Accessible")
                code, acc_name = acc.GetName(0)
                self.assertEqual(code, wx.ACC_OK)
                self.assertTrue(len(acc_name.strip()) > 0, f"Control {ctrl} has empty accName")

                # Verify composite children (SpinCtrl inner TextCtrl and SpinButton) have accessible names for screen readers
                for child in ctrl.GetChildren():
                    c_name = child.GetName()
                    self.assertTrue(len(c_name.strip()) > 0, f"Child {child} of {ctrl} has empty GetName()")

            # 2. Lyrics tab controls
            for ctrl in [lyrics_tab.input_text_ctrl, lyrics_tab.output_text_ctrl, lyrics_tab.changes_ctrl]:
                acc = ctrl.GetAccessible()
                self.assertIsNotNone(acc)
                code, acc_name = acc.GetName(0)
                self.assertEqual(code, wx.ACC_OK)
                self.assertTrue(len(acc_name.strip()) > 0)

            # 3. Dialog controls
            dlg_batch = BatchProcessingDialog(frame, frame.audio_processor, lambda: frame.audio_tab._get_current_options())
            try:
                for ctrl in [dlg_batch.watch_in_ctrl, dlg_batch.watch_out_ctrl]:
                    acc = ctrl.GetAccessible()
                    self.assertIsNotNone(acc)
                    code, acc_name = acc.GetName(0)
                    self.assertEqual(code, wx.ACC_OK)
                    self.assertTrue(len(acc_name.strip()) > 0)
            finally:
                dlg_batch.Destroy()

            dlg_suno = SunoPromptDialog(frame, lyrics_tab.input_text_ctrl)
            try:
                for ctrl in [dlg_suno.bpm_spin, dlg_suno.key_ctrl, dlg_suno.prompt_ctrl, dlg_suno.tags_ctrl]:
                    acc = ctrl.GetAccessible()
                    self.assertIsNotNone(acc)
                    code, acc_name = acc.GetName(0)
                    self.assertEqual(code, wx.ACC_OK)
                    self.assertTrue(len(acc_name.strip()) > 0)
            finally:
                dlg_suno.Destroy()

            dlg_slicer = AbsAudioSlicerDialog(frame, frame.audio_processor)
            try:
                for ctrl in [dlg_slicer.file_path_ctrl, dlg_slicer.out_dir_ctrl, dlg_slicer.chunk_spin, dlg_slicer.log_ctrl]:
                    acc = ctrl.GetAccessible()
                    self.assertIsNotNone(acc)
                    code, acc_name = acc.GetName(0)
                    self.assertEqual(code, wx.ACC_OK)
                    self.assertTrue(len(acc_name.strip()) > 0)
            finally:
                dlg_slicer.Destroy()

            dlg_api = CloudApiDialog(frame)
            try:
                for ctrl in [dlg_api.token_ctrl, dlg_api.url_ctrl, dlg_api.timeout_spin]:
                    acc = ctrl.GetAccessible()
                    self.assertIsNotNone(acc)
                    code, acc_name = acc.GetName(0)
                    self.assertEqual(code, wx.ACC_OK)
                    self.assertTrue(len(acc_name.strip()) > 0)
            finally:
                dlg_api.Destroy()

            dlg_lyrics_api = LyricsCloudDialog(frame)
            try:
                for ctrl in [dlg_lyrics_api.token_ctrl, dlg_lyrics_api.model_ctrl, dlg_lyrics_api.url_ctrl]:
                    acc = ctrl.GetAccessible()
                    self.assertIsNotNone(acc)
                    code, acc_name = acc.GetName(0)
                    self.assertEqual(code, wx.ACC_OK)
                    self.assertTrue(len(acc_name.strip()) > 0)
            finally:
                dlg_lyrics_api.Destroy()

            dlg_bl = BlacklistDialog(frame, frame.lyrics_processor)
            try:
                acc = dlg_bl.text_ctrl.GetAccessible()
                self.assertIsNotNone(acc)
                code, acc_name = acc.GetName(0)
                self.assertEqual(code, wx.ACC_OK)
                self.assertTrue(len(acc_name.strip()) > 0)
            finally:
                dlg_bl.Destroy()

            dlg_supp = SupportTicketDialog(frame)
            try:
                for ctrl in [dlg_supp.title_ctrl, dlg_supp.desc_ctrl, dlg_supp.reply_ctrl]:
                    acc = ctrl.GetAccessible()
                    self.assertIsNotNone(acc)
                    code, acc_name = acc.GetName(0)
                    self.assertEqual(code, wx.ACC_OK)
                    self.assertTrue(len(acc_name.strip()) > 0)
            finally:
                dlg_supp.Destroy()

        finally:
            frame.Destroy()

    def test_ci_modular_audio_sanitization_end_to_end(self):
        """Verifies end-to-end processing with all modular 2026 flags enabled."""
        from ghostwave.audio_processor import sanitize_audio, AudioSanitizeOptions

        in_wav = os.path.join(self.test_dir, "input_modular.wav")
        out_wav = os.path.join(self.test_dir, "output_modular.wav")
        sf.write(in_wav, self.audio, self.sr)

        opts = AudioSanitizeOptions(
            pitch_shift_semitones=1.0,
            tempo_factor=1.02,
            enable_ai_watermark_neutralizer=True,
            ai_watermark_intensity=0.85,
            enable_formant_shift=True,
            formant_shift_semitones=1.5,
            enable_psychoacoustic_mask=True,
            room_reverb_wet=0.20,
            vocal_reduction_depth=10.0,
            front_padding_seconds=0.2,
            tail_padding_seconds=0.2
        )

        res = sanitize_audio(in_wav, out_wav, options=opts)
        self.assertTrue(res.success, f"Modular sanitization failed: {res.error_message}")
        self.assertTrue(os.path.exists(out_wav))

        out_data, out_sr = sf.read(out_wav)
        self.assertEqual(out_sr, self.sr)
        self.assertTrue(np.isfinite(out_data).all())
        rms = np.sqrt(np.mean(out_data ** 2))
        self.assertGreater(rms, 0.01)

    def test_ci_google_lyria_synthid_only_mode(self):
        """Verifies that selecting Google Lyria enables ONLY SynthID neutralizer and preserves pure audio."""
        import wx
        from ghostwave.gui import GhostWaveFrame
        from ghostwave.audio_processor import sanitize_audio

        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        frame = GhostWaveFrame()
        try:
            audio_tab = frame.audio_tab
            # Select platform index 1: Google Lyria 3.5 / SynthID Cleaner
            audio_tab.platform_choice.SetSelection(1)
            audio_tab.on_platform_changed()

            # Verify GUI controls state
            self.assertEqual(audio_tab.pitch_spin.GetValue(), 0.0)
            self.assertEqual(audio_tab.tempo_spin.GetValue(), 1.000)
            self.assertEqual(audio_tab.formant_spin.GetValue(), 0.0)
            self.assertEqual(audio_tab.reverb_spin.GetValue(), 0.0)
            self.assertEqual(audio_tab.vocal_duck_spin.GetValue(), 0.0)
            self.assertTrue(audio_tab.watermark_chk.IsChecked())
            self.assertFalse(audio_tab.psycho_chk.IsChecked())

            opts = audio_tab._get_current_options()
            self.assertEqual(opts.target_platform, "google_lyria")
            self.assertTrue(opts.enable_ai_watermark_neutralizer)
            self.assertEqual(opts.front_padding_seconds, 0.0)
            self.assertEqual(opts.tail_padding_seconds, 0.0)
            self.assertEqual(opts.pitch_shift_semitones, 0.0)
            self.assertEqual(opts.tempo_factor, 1.000)
            self.assertFalse(opts.enable_micro_chrono_jitter)
            self.assertFalse(opts.enable_adversarial_peaks)
            self.assertFalse(opts.enable_phase_dispersion)
            self.assertFalse(opts.enable_reamping_room)

            # Test end-to-end processing with Lyria options: audio length and melody are identical
            in_wav = os.path.join(self.test_dir, "input_lyria.wav")
            out_wav = os.path.join(self.test_dir, "output_lyria.wav")
            sf.write(in_wav, self.audio, self.sr)

            res = sanitize_audio(in_wav, out_wav, options=opts)
            self.assertTrue(res.success)
            out_data, out_sr = sf.read(out_wav)
            self.assertEqual(out_sr, self.sr)
            # Duration must be identical (no front/tail padding added)
            self.assertEqual(len(out_data), len(self.audio))
        finally:
            frame.Destroy()

    def test_ci_audio_preview_audition_player(self):
        """Verifies that original, sanitized, delta playback and stop execute reliably."""
        import wx
        from ghostwave.gui import GhostWaveFrame

        app = wx.App.Get()
        if not app:
            app = wx.App(False)

        frame = GhostWaveFrame()
        try:
            audio_tab = frame.audio_tab
            in_wav = os.path.join(self.test_dir, "test_prev_in.wav")
            out_wav = os.path.join(self.test_dir, "test_prev_out.wav")
            sf.write(in_wav, self.audio, self.sr)
            sf.write(out_wav, self.audio * 0.95, self.sr)

            audio_tab.load_file(in_wav)
            audio_tab.last_output_path = out_wav
            audio_tab.play_sani_btn.Enable(True)
            audio_tab.play_delta_btn.Enable(True)

            # 1. Play original
            audio_tab.on_play_original(None)
            # 2. Play sanitized
            audio_tab.on_play_sanitized(None)
            # 3. Play delta
            audio_tab.on_play_delta(None)
            # 4. Stop playback
            audio_tab.on_stop_playback(None)

            # 5. Play direct data array
            audio_tab.play_audio_data(self.audio[:1000], self.sr, status_text="Test Data")
            audio_tab.on_stop_playback(None)
        finally:
            frame.Destroy()

    def test_recommended_transposition_apply_no_crash(self):
        """Verifies load_file unpacks (float, key_name) and on_apply_rec_pitch applies float pitch cleanly."""
        import wx
        from ghostwave.gui import GhostWaveFrame

        app = wx.App.Get() or wx.App(False)
        frame = GhostWaveFrame()
        try:
            in_wav = os.path.join(self.test_dir, "rec_test.wav")
            sf.write(in_wav, self.audio, self.sr)
            frame.audio_tab.load_file(in_wav)
            self.assertIsInstance(frame.audio_tab._recommended_pitch, float)
            frame.audio_tab.on_apply_rec_pitch(None)
            self.assertAlmostEqual(
                frame.audio_tab.pitch_spin.GetValue(),
                frame.audio_tab._recommended_pitch,
                places=1,
            )
        finally:
            frame.Destroy()

    def test_all_used_i18n_keys_exist_and_no_broken_tabs(self):
        """Verifies every tr()/_t() key referenced in source exists in en.json and id.json without escaped tabs."""
        import json
        import pathlib
        import re

        en = json.loads(pathlib.Path(ROOT_DIR, "locales", "en.json").read_text(encoding="utf-8"))
        id_ = json.loads(pathlib.Path(ROOT_DIR, "locales", "id.json").read_text(encoding="utf-8"))

        for k, v in list(en.items()) + list(id_.items()):
            self.assertNotIn("\\t", v, f"Broken escaped tab in locale key {k}")

        used_keys = set()
        py_files = [pathlib.Path(ROOT_DIR, "main.py")] + sorted(pathlib.Path(ROOT_DIR, "ghostwave").glob("*.py"))
        for p in py_files:
            txt = p.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r'(?:\btr|\b_t)\(\s*["\']([A-Z0-9_]+)["\']', txt):
                used_keys.add(m.group(1))

        self.assertEqual(sorted(used_keys - set(en)), [])
        self.assertEqual(sorted(used_keys - set(id_)), [])

    def test_preserve_rhymes_behavior(self):
        """Verifies preserve_rhymes=True protects line-ending rhymes in semantic mode."""
        from ghostwave.lyrics_processor import LyricsCloaker
        lyrics = "I walk alone in a lonely town\nLeaving all my trouble in this world"
        kept, changes = LyricsCloaker.cloak(
            lyrics, mode="semantic", add_vibrato_glides=False, preserve_rhymes=True
        )
        lines_kept = kept.splitlines()
        self.assertTrue(lines_kept[0].lower().endswith("town"))
        self.assertTrue(lines_kept[1].lower().endswith("world"))
        self.assertTrue(any("Rhyme Preservation" in c for c in changes))

        free, _ = LyricsCloaker.cloak(
            lyrics, mode="semantic", add_vibrato_glides=False, preserve_rhymes=False
        )
        lines_free = free.splitlines()
        self.assertFalse(lines_free[0].lower().endswith("town"))
        self.assertFalse(lines_free[1].lower().endswith("world"))

    def test_preflight_honest_watermark_and_verdict(self):
        """Verifies identical inputs report watermark_neutralized=False and watermark-cleaned audio reports True."""
        from ghostwave.audio_processor import simulate_preflight_evasion, apply_ai_watermark_neutralizer

        t = np.linspace(0, 1.5, int(self.sr * 1.5), endpoint=False, dtype=np.float32)
        sig = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.08 * np.sin(2 * np.pi * 6500 * t) + 0.05 * np.sin(2 * np.pi * 11000 * t)).astype(np.float32)
        rich_audio = np.column_stack([sig, sig])

        identical = simulate_preflight_evasion(rich_audio, rich_audio, sr=self.sr)
        self.assertFalse(identical["synthid_watermark_neutralized"])
        self.assertEqual(identical["verdict_en"], "DETECTION RISK")

        cleaned = apply_ai_watermark_neutralizer(rich_audio, self.sr, intensity=0.85)
        wm_report = simulate_preflight_evasion(rich_audio, cleaned, sr=self.sr)
        self.assertTrue(wm_report["synthid_watermark_neutralized"])

    def test_visualizer_real_peaks_and_accessible_summary(self):
        """Verifies AudioVisualizerPanel computes real delta peaks and updates its screen-reader summary."""
        import wx
        from ghostwave.gui import AudioVisualizerPanel

        app = wx.App.Get() or wx.App(False)
        frame = wx.Frame(None)
        try:
            vis = AudioVisualizerPanel(frame)
            self.assertIn("No audio loaded", vis.get_accessible_summary())
            vis.set_original_audio(self.audio)
            self.assertIn("Original waveform loaded", vis.get_accessible_summary())
            vis.set_sanitized_audio(self.audio * 0.5)
            self.assertGreater(len(vis.decoy_peaks), 0)
            self.assertIn("Comparison ready", vis.get_accessible_summary())
            self.assertEqual(vis.GetName(), vis.get_accessible_summary())
        finally:
            frame.Destroy()

    def test_platform_switch_from_lyria_restores_dsp(self):
        """Verifies switching Suno -> Google Lyria -> Suno restores core DSP flags."""
        import wx
        from ghostwave.gui import GhostWaveFrame

        app = wx.App.Get() or wx.App(False)
        frame = GhostWaveFrame()
        try:
            tab = frame.audio_tab
            tab.platform_choice.SetSelection(1)
            tab.on_platform_changed()
            self.assertFalse(tab.jitter_chk.IsChecked())
            self.assertFalse(tab.decoy_chk.IsChecked())

            tab.platform_choice.SetSelection(0)
            tab.on_platform_changed()
            self.assertTrue(tab.jitter_chk.IsChecked())
            self.assertTrue(tab.decoy_chk.IsChecked())
            self.assertTrue(tab.allpass_chk.IsChecked())
            self.assertTrue(tab.reamping_chk.IsChecked())
        finally:
            frame.Destroy()

    def test_preflight_banner_wcag_aa_contrast(self):
        """Verifies all three Pre-Flight status banner colors meet WCAG AA (>= 4.5:1) with white text."""
        def rel_lum(r: int, g: int, b: int) -> float:
            def lin(c: float) -> float:
                c /= 255.0
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

        for rgb in ((4, 108, 78), (146, 64, 14), (185, 28, 28)):
            ratio = (1.0 + 0.05) / (rel_lum(*rgb) + 0.05)
            self.assertGreaterEqual(ratio, 4.5, f"Banner RGB {rgb} failed WCAG AA: {ratio:.2f}")

    def test_cli_batch_dir_output_and_preflight_guard(self):
        """Verifies CLI --batch-dir writes into -o folder and --preflight handles missing -o gracefully."""
        import subprocess
        import sys as _sys

        in_dir = os.path.join(self.test_dir, "cli_in")
        out_dir = os.path.join(self.test_dir, "cli_out")
        os.makedirs(in_dir, exist_ok=True)
        wav_in = os.path.join(in_dir, "track.wav")
        sf.write(wav_in, self.audio, self.sr)

        res_batch = subprocess.run(
            [_sys.executable, os.path.join(ROOT_DIR, "main.py"), "--batch-dir", in_dir, "-o", out_dir, "--no-banner"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(res_batch.returncode, 0, res_batch.stderr)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "track_ghostwave_ready.wav")))

        missing_out = os.path.join(self.test_dir, "does_not_exist.wav")
        res_pre = subprocess.run(
            [_sys.executable, os.path.join(ROOT_DIR, "main.py"), "-i", wav_in, "-o", missing_out, "--preflight", "--no-banner"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(res_pre.returncode, 1)
        self.assertNotIn("Traceback", res_pre.stderr)

    def test_batch_dialog_close_stops_watcher(self):
        """Verifies BatchProcessingDialog cleanup stops an active FolderWatcher."""
        import wx
        from ghostwave.gui import BatchProcessingDialog
        from ghostwave.audio_processor import AudioProcessor, AudioSanitizeOptions

        app = wx.App.Get() or wx.App(False)
        frame = wx.Frame(None)
        dlg = BatchProcessingDialog(frame, AudioProcessor(), lambda: AudioSanitizeOptions())
        try:
            w_in = os.path.join(self.test_dir, "win")
            w_out = os.path.join(self.test_dir, "wout")
            os.makedirs(w_in, exist_ok=True)
            os.makedirs(w_out, exist_ok=True)
            dlg.watch_in_ctrl.SetValue(w_in)
            dlg.watch_out_ctrl.SetValue(w_out)
            dlg.on_toggle_watcher(None)
            self.assertIsNotNone(dlg.folder_watcher)
            self.assertTrue(dlg.folder_watcher.is_running)
            dlg._cleanup_workers()
            self.assertIsNone(dlg.folder_watcher)
        finally:
            dlg.Destroy()
            frame.Destroy()


if __name__ == "__main__":
    unittest.main()

