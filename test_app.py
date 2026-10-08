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


if __name__ == "__main__":
    unittest.main()
