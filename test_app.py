"""
Automated Unit Tests for GhostWave Studio v1.0.

Tests DSP acoustic cloaking algorithms, ABS audio slicer, proprietary .sn
encrypted profile vault, lyrics moderation and cloaking engine, and accessibility GUI.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import numpy as np
import soundfile as sf

from config_manager import GhostWaveConfig, MAGIC_HEADER
from lyrics_processor import (
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
        from audio_processor import AudioProcessor
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
        from audio_processor import apply_micro_chrono_jitter
        jittered = apply_micro_chrono_jitter(self.sample_audio, self.sr, max_jitter_ms=16.0)
        self.assertEqual(jittered.shape, self.sample_audio.shape)
        self.assertFalse(np.array_equal(jittered, self.sample_audio))

    def test_bode_frequency_shifter(self):
        from audio_processor import apply_bode_frequency_shifter
        shifted = apply_bode_frequency_shifter(self.sample_audio, self.sr, shift_hz=12.0)
        self.assertEqual(shifted.shape, self.sample_audio.shape)
        self.assertFalse(np.array_equal(shifted, self.sample_audio))

    def test_center_vocal_cut_bass_retention(self):
        from audio_processor import apply_total_center_vocal_annihilation
        cut = apply_total_center_vocal_annihilation(self.sample_audio, self.sr)
        self.assertEqual(cut.shape, self.sample_audio.shape)

    def test_schroeder_phase_dispersion(self):
        from audio_processor import apply_schroeder_phase_dispersion
        dispersed = apply_schroeder_phase_dispersion(self.sample_audio, self.sr)
        self.assertEqual(dispersed.shape, self.sample_audio.shape)

    def test_harmonic_drone_bed(self):
        from audio_processor import apply_continuous_camouflage_bed
        drone = apply_continuous_camouflage_bed(self.sample_audio, self.sr)
        self.assertEqual(drone.shape, self.sample_audio.shape)


class TestGhostWaveUpdater(unittest.TestCase):
    """Verifies update checking, changelog parsing, and screen reader accessible dialogs."""

    def test_normalize_version(self):
        from updater import normalize_version
        self.assertEqual(normalize_version("1.0.0"), (1, 0, 0))
        self.assertEqual(normalize_version("v1.2.0"), (1, 2, 0))
        self.assertEqual(normalize_version("1.0"), (1, 0, 0))
        self.assertEqual(normalize_version("v2.1.3-beta"), (2, 1, 3))

    def test_compare_versions(self):
        from updater import compare_versions
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
        from updater import parse_changelog_text
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
        from updater import UpdateInfo
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
        from updater import UpdateInfo, UpdateDialog

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
            self.assertEqual(dlg.changelog_ctrl.GetName(), "Changelog Text Area")
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
        from updater import UpdateDownloadDialog

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
        from audio_processor import AudioSanitizeOptions
        from config_manager import GhostWaveConfig

        opts = AudioSanitizeOptions()
        self.assertFalse(opts.trim_duration)

        cfg = GhostWaveConfig()
        self.assertFalse(cfg.trim_duration)

    def test_gui_advanced_settings_toggle_and_trimmer_default(self):
        """Verifies collapsible advanced settings toggle and trimmer unchecked state in UI."""
        import wx
        from gui import AudioSanitizerPanel
        from audio_processor import AudioProcessor

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
            self.assertEqual(panel.toggle_adv_btn.GetLabel(), "&Show Advanced Settings")
            self.assertEqual(panel.toggle_adv_btn.GetName(), "Show Advanced Settings Button")

            # 3. Toggling button expands advanced panel
            panel.on_toggle_advanced(None)
            self.assertTrue(panel.adv_panel.IsShown())
            self.assertEqual(panel.toggle_adv_btn.GetLabel(), "&Hide Advanced Settings")

            # 4. Toggling again collapses it
            panel.on_toggle_advanced(None)
            self.assertFalse(panel.adv_panel.IsShown())
            self.assertEqual(panel.toggle_adv_btn.GetLabel(), "&Show Advanced Settings")

            # 5. Switching presets keeps trimmer unchecked
            for preset_idx in range(6):
                panel.preset_choice.SetSelection(preset_idx)
                panel.on_preset_changed(None)
                self.assertFalse(panel.trim_chk.IsChecked())
                self.assertFalse(panel.trim_spin.IsEnabled())
        finally:
            frame.Destroy()

    # Perspective 2: Developer Persona (DSP vectorization, caching, benchmarks)
    def test_dsp_filter_caching(self):
        """Validates that filter design functions leverage LRU caching."""
        from audio_processor import _get_cached_butter_sos, _get_cached_bilinear

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
        from audio_processor import apply_bode_frequency_shifter, apply_triple_band_bode_shifter

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
        from audio_processor import apply_adversarial_peak_injection

        sr = 44100
        data = np.random.uniform(-0.4, 0.4, (22050, 2)).astype(np.float32)
        injected = apply_adversarial_peak_injection(data, sr=sr, intensity=1.15)
        self.assertEqual(injected.shape, data.shape)
        self.assertTrue(np.all(np.isfinite(injected)))

    def test_single_pass_regex_blacklist(self):
        """Validates single-pass regex compilation correctly masks names and profanities."""
        from lyrics_processor import LyricsProcessor

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
        from audio_processor import AudioProcessor, AudioSanitizeOptions

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
        from audio_processor import apply_soft_limiting

        extreme_signal = np.array([[2.5, -3.8], [5.0, -10.0]], dtype=np.float32)
        limited = apply_soft_limiting(extreme_signal, drive_db=3.0)
        self.assertTrue(np.all(limited <= 1.0))
        self.assertTrue(np.all(limited >= -1.0))

    def test_zero_byte_file_protection(self):
        """Verifies zero-byte files are handled gracefully without exceptions."""
        from audio_processor import AudioProcessor

        empty_file = os.path.join(self.test_dir, "empty.wav")
        open(empty_file, "w").close()

        processor = AudioProcessor()
        info = processor.inspect_file(empty_file)
        self.assertEqual(info.duration_seconds, 0.0)
        self.assertEqual(info.duration_formatted, "00:00")


if __name__ == "__main__":
    unittest.main()

