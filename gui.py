"""
Accessible Desktop GUI Module for Suno Prep & Sanitizer.

Built using wxPython Phoenix with an Accessibility-First approach:
- Explicit accessible names (.SetName) on all input fields, buttons, sliders, and checkboxes.
- Logical keyboard navigation with wx.TAB_TRAVERSAL and standard mnemonic accelerators.
- Native multiline wx.TextCtrl controls with full screen reader compatibility.
- Live status updates via wx.StatusBar, system accessibility bells, and focused modal feedback dialogs.
- Dual-tab notebook: Audio Sanitizer (Multi-Vector Acoustic Evasion) and Lyrics Sanitizer (Moderation & Normalization).
- Cloud API Stem Separation configuration dialog (Zero Local Model Downloads).
- Quantitative Chromaprint Evasion Safety Audit verification dialog.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import wx

from audio_processor import (
    AudioProcessor,
    AudioSanitizeOptions,
    AudioProcessResult,
    benchmark_evasion_metrics
)
from cloud_stem_api import CloudApiConfig, CloudStemClient
from config_manager import GhostWaveConfig, DEFAULT_SN_FILENAME, resolve_config_path
from lyrics_processor import LyricsProcessor, LyricsSanitizeResult, DEFAULT_CELEBRITY_BLACKLIST
from updater import (
    APP_VERSION,
    UpdateInfo,
    UpdateDialog,
    UpdateDownloadDialog,
    check_for_updates,
    check_updates_background,
    run_update_flow
)
from i18n import tr, _t, get_language, set_language, get_supported_languages
import support_client


class BlacklistDialog(wx.Dialog):
    """
    Accessible Dialog to view and edit the custom celebrity and artist blacklist.
    """

    def __init__(self, parent: wx.Window, lyrics_processor: LyricsProcessor):
        super().__init__(
            parent,
            title=f"Edit Celebrity Blacklist - GhostWave Studio v{APP_VERSION}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(520, 560)
        )
        self.SetName("Edit Celebrity Blacklist Dialog")
        self.processor = lyrics_processor

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Celebrity Blacklist Editor Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Instructions Label
        instr_label = wx.StaticText(
            panel,
            label="Edit the list of celebrity and artist names to strip (one name per line):"
        )
        instr_label.SetName("Blacklist Instructions")
        main_sizer.Add(instr_label, 0, wx.ALL, 10)

        # Blacklist Text Editor
        self.text_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_DONTWRAP,
            name="Celebrity Blacklist Text Area"
        )
        self.text_ctrl.SetToolTip("Enter celebrity and artist names, one per line.")
        initial_text = "\n".join(sorted(self.processor.celebrity_blacklist, key=str.lower))
        self.text_ctrl.SetValue(initial_text)
        main_sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Bottom Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.reset_btn = wx.Button(panel, label="&Reset to Defaults")
        self.reset_btn.SetName("Reset to Defaults Button")
        self.reset_btn.SetToolTip("Restore the default list of 100+ global artists and celebrities.")
        self.reset_btn.Bind(wx.EVT_BUTTON, self.on_reset)
        btn_sizer.Add(self.reset_btn, 0, wx.RIGHT, 10)

        btn_sizer.AddStretchSpacer()

        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, label="&Cancel")
        self.cancel_btn.SetName("Cancel Button")
        btn_sizer.Add(self.cancel_btn, 0, wx.RIGHT, 10)

        self.save_btn = wx.Button(panel, wx.ID_OK, label="&Save Changes")
        self.save_btn.SetName("Save Changes Button")
        self.save_btn.Bind(wx.EVT_BUTTON, self.on_save)
        self.save_btn.SetDefault()
        btn_sizer.Add(self.save_btn, 0)

        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        panel.SetSizer(main_sizer)

        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dialog_sizer)
        self.CentreOnParent()

    def on_reset(self, event: wx.CommandEvent):
        confirm = wx.MessageBox(
            "Are you sure you want to restore the default celebrity list?",
            "Confirm Reset",
            wx.YES_NO | wx.ICON_QUESTION,
            self
        )
        if confirm == wx.YES:
            default_text = "\n".join(sorted(DEFAULT_CELEBRITY_BLACKLIST, key=str.lower))
            self.text_ctrl.SetValue(default_text)
            self.text_ctrl.SetFocus()

    def on_save(self, event: wx.CommandEvent):
        raw_text = self.text_ctrl.GetValue()
        names = [line.strip() for line in raw_text.split("\n") if line.strip()]
        if not names:
            wx.MessageBox(
                "Blacklist cannot be completely empty.",
                "Validation Error",
                wx.OK | wx.ICON_WARNING,
                self
            )
            return

        success = self.processor.save_blacklist(names)
        if success:
            self.EndModal(wx.ID_OK)
        else:
            wx.MessageBox(
                "Failed to save blacklist to configuration file.",
                "File Save Error",
                wx.OK | wx.ICON_ERROR,
                self
            )


class CloudApiDialog(wx.Dialog):
    """
    Accessible Dialog to configure Cloud API credentials for remote stem separation.
    Zero local model weights are ever downloaded.
    """

    def __init__(self, parent: wx.Window):
        super().__init__(
            parent,
            title=f"Cloud API Settings (Encrypted .sn Vault) - GhostWave Studio v{APP_VERSION}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(560, 440)
        )
        self.SetName("Cloud API Settings Dialog")
        self.config = CloudApiConfig.load()

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Cloud API Settings Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        info_label = wx.StaticText(
            panel,
            label="Configure Cloud API credentials to separate vocals and instrumentals remotely.\n"
                  "All tokens are securely encrypted in the GhostWave .sn vault.\n"
                  "Zero machine learning models or weights are downloaded to this device."
        )
        info_label.SetName("Cloud API Information Label")
        main_sizer.Add(info_label, 0, wx.ALL, 12)

        grid = wx.FlexGridSizer(cols=2, vgap=12, hgap=12)
        grid.AddGrowableCol(1, 1)

        # Provider Choice
        prov_label = wx.StaticText(panel, label="Cloud Provider:")
        prov_label.SetName("Cloud Provider Label")
        grid.Add(prov_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.provider_choice = wx.Choice(
            panel,
            choices=["Replicate (Demucs API)", "Hugging Face Spaces / Inference API", "Custom REST Endpoint"],
            name="Cloud Provider Selection"
        )
        cur_prov = self.config.provider.lower()
        if "hug" in cur_prov:
            self.provider_choice.SetSelection(1)
        elif "cust" in cur_prov:
            self.provider_choice.SetSelection(2)
        else:
            self.provider_choice.SetSelection(0)
        grid.Add(self.provider_choice, 0, wx.EXPAND)

        # API Token
        token_label = wx.StaticText(panel, label="API Key / Token:")
        token_label.SetName("API Token Label")
        grid.Add(token_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.token_ctrl = wx.TextCtrl(
            panel,
            value=self.config.api_token,
            style=wx.TE_PASSWORD,
            name="Cloud API Token Input"
        )
        self.token_ctrl.SetToolTip("Enter your Replicate or Hugging Face API token.")
        grid.Add(self.token_ctrl, 0, wx.EXPAND)

        # Endpoint URL
        url_label = wx.StaticText(panel, label="Endpoint URL (Optional):")
        url_label.SetName("Endpoint URL Label")
        grid.Add(url_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.url_ctrl = wx.TextCtrl(
            panel,
            value=self.config.endpoint_url,
            name="Cloud Endpoint URL Input"
        )
        self.url_ctrl.SetToolTip("Custom HF Space or REST webhook URL (leave blank for defaults).")
        grid.Add(self.url_ctrl, 0, wx.EXPAND)

        # Timeout Spin
        timeout_label = wx.StaticText(panel, label="Remote Timeout (Seconds):")
        timeout_label.SetName("Remote Timeout Label")
        grid.Add(timeout_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.timeout_spin = wx.SpinCtrl(
            panel,
            value=str(self.config.timeout_seconds),
            min=30,
            max=600,
            name="Remote Job Timeout in Seconds"
        )
        grid.Add(self.timeout_spin, 0, wx.EXPAND)

        main_sizer.Add(grid, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        # Bottom Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, label="&Cancel", name="Cancel Cloud Settings Button")
        btn_sizer.Add(self.cancel_btn, 0, wx.RIGHT, 10)

        self.save_btn = wx.Button(panel, wx.ID_OK, label="&Save Settings", name="Save Cloud Settings Button")
        self.save_btn.Bind(wx.EVT_BUTTON, self.on_save)
        self.save_btn.SetDefault()
        btn_sizer.Add(self.save_btn, 0)

        main_sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 12)
        panel.SetSizer(main_sizer)

        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dialog_sizer)
        self.CentreOnParent()

    def on_save(self, event: wx.CommandEvent):
        sel = self.provider_choice.GetSelection()
        prov = "replicate" if sel == 0 else ("huggingface" if sel == 1 else "custom")
        self.config.provider = prov
        self.config.api_token = self.token_ctrl.GetValue().strip()
        self.config.endpoint_url = self.url_ctrl.GetValue().strip()
        self.config.timeout_seconds = self.timeout_spin.GetValue()
        self.config.save()
        self.EndModal(wx.ID_OK)


class LyricsCloudDialog(wx.Dialog):
    """
    Accessible dialog for configuring remote LLM API credentials for lyrics cloaking.
    """

    def __init__(self, parent: wx.Window):
        super().__init__(
            parent,
            title=f"Lyrics Cloud LLM Settings - GhostWave Studio v{APP_VERSION}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(560, 420)
        )
        self.SetName("Lyrics Cloud LLM Settings Dialog")
        from config_manager import GhostWaveConfig
        self.config = GhostWaveConfig.load()

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Lyrics Cloud Settings Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        info_label = wx.StaticText(
            panel,
            label="Configure remote LLM credentials for automated lyrics cloaking.\n"
                  "API keys are encrypted inside the GhostWave .sn vault.\n"
                  "Zero models or checkpoint files are downloaded to this device."
        )
        info_label.SetName("Lyrics Cloud Info Label")
        main_sizer.Add(info_label, 0, wx.ALL, 12)

        grid = wx.FlexGridSizer(cols=2, vgap=12, hgap=12)
        grid.AddGrowableCol(1, 1)

        prov_label = wx.StaticText(panel, label="LLM Provider:")
        prov_label.SetName("LLM Provider Label")
        grid.Add(prov_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.provider_choice = wx.Choice(
            panel,
            choices=["Groq (Fastest & Free Tier)", "OpenRouter (Multi-model)", "OpenAI (GPT-4o mini)", "Custom OpenAI-Compatible"],
            name="LLM Provider Selection"
        )
        p = self.config.lyrics_cloud_provider.lower()
        if "groq" in p:
            self.provider_choice.SetSelection(0)
        elif "openrouter" in p:
            self.provider_choice.SetSelection(1)
        elif "openai" in p:
            self.provider_choice.SetSelection(2)
        else:
            self.provider_choice.SetSelection(3)
        grid.Add(self.provider_choice, 0, wx.EXPAND)

        tok_label = wx.StaticText(panel, label="API Key / Token:")
        tok_label.SetName("Lyrics API Token Label")
        grid.Add(tok_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.token_ctrl = wx.TextCtrl(
            panel,
            value=self.config.lyrics_cloud_api_token,
            style=wx.TE_PASSWORD,
            name="Lyrics API Token Input"
        )
        self.token_ctrl.SetToolTip("Enter API token for remote lyrics rewriting.")
        grid.Add(self.token_ctrl, 0, wx.EXPAND)

        mod_label = wx.StaticText(panel, label="Model Identifier:")
        mod_label.SetName("Model Identifier Label")
        grid.Add(mod_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.model_ctrl = wx.TextCtrl(
            panel,
            value=self.config.lyrics_cloud_model or "llama-3.3-70b-versatile",
            name="Model Identifier Input"
        )
        self.model_ctrl.SetToolTip("Target LLM model name (e.g. llama-3.3-70b-versatile, gpt-4o-mini).")
        grid.Add(self.model_ctrl, 0, wx.EXPAND)

        url_label = wx.StaticText(panel, label="Endpoint URL (Optional):")
        url_label.SetName("Lyrics Endpoint URL Label")
        grid.Add(url_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.url_ctrl = wx.TextCtrl(
            panel,
            value=self.config.lyrics_cloud_endpoint_url,
            name="Lyrics Endpoint URL Input"
        )
        self.url_ctrl.SetToolTip("Custom base URL (leave blank for provider default).")
        grid.Add(self.url_ctrl, 0, wx.EXPAND)

        main_sizer.Add(grid, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, label="&Cancel", name="Cancel Lyrics Cloud Button")
        btn_sizer.Add(self.cancel_btn, 0, wx.RIGHT, 10)

        self.save_btn = wx.Button(panel, wx.ID_OK, label="&Save Settings", name="Save Lyrics Cloud Button")
        self.save_btn.Bind(wx.EVT_BUTTON, self.on_save)
        self.save_btn.SetDefault()
        btn_sizer.Add(self.save_btn, 0)
        main_sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 12)

        panel.SetSizer(main_sizer)
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dlg_sizer)
        self.CentreOnParent()

    def on_save(self, event: wx.CommandEvent):
        sel = self.provider_choice.GetSelection()
        prov_map = ["groq", "openrouter", "openai", "custom"]
        self.config.lyrics_cloud_provider = prov_map[sel] if sel < len(prov_map) else "groq"
        self.config.lyrics_cloud_api_token = self.token_ctrl.GetValue().strip()
        self.config.lyrics_cloud_model = self.model_ctrl.GetValue().strip()
        self.config.lyrics_cloud_endpoint_url = self.url_ctrl.GetValue().strip()
        self.config.save()
        self.EndModal(wx.ID_OK)


class EvasionAuditDialog(wx.Dialog):
    """
    Accessible Dialog presenting quantitative acoustic evasion benchmark results.
    """

    def __init__(self, parent: wx.Window, metrics: dict, orig_file: str, sani_file: str):
        super().__init__(
            parent,
            title=f"Acoustic Evasion Safety Audit - GhostWave Studio v{APP_VERSION}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(580, 500)
        )
        self.SetName("Acoustic Evasion Safety Audit Dialog")

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Acoustic Audit Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        title_lbl = wx.StaticText(panel, label="Quantitative Acoustic Fingerprint & ASR Audit:")
        title_lbl.SetName("Audit Title Label")
        font = title_lbl.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title_lbl.SetFont(font)
        main_sizer.Add(title_lbl, 0, wx.ALL, 10)

        report_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="Audit Details Report Text"
        )

        sim = metrics.get("similarity_pct", 0.0)
        verdict = metrics.get("verdict", "Unknown")
        ber = metrics.get("bit_error_rate_pct", 0.0)
        drop_db = metrics.get("vocal_energy_drop_db", 0.0)
        frames = metrics.get("aligned_frames", 0)

        lines = [
            f"Original File: {os.path.basename(orig_file)}",
            f"Sanitized File: {os.path.basename(sani_file)}",
            "--------------------------------------------------",
            f"Evasion Safety Rating: {verdict.upper()}",
            "--------------------------------------------------",
            f"• Acoustic Fingerprint (Chromaprint) Similarity: {sim:.1f}%",
            f"• Bit Error / Hash Decorrelation Rate: {ber:.1f}%",
            f"• Aligned Sub-fingerprint Frame Window: {frames} frames",
            f"• Vocal Intelligibility Band Attenuation: {drop_db:.1f} dB",
            "",
            "DETAILED TECHNICAL ASSESSMENT:",
            f"{metrics.get('details', '')}",
            "",
            "RECOMMENDATIONS FOR SUNO UPLOAD:",
            "1. If similarity is under 35%, landmark hashes are decoupled and will pass Audible Magic.",
            "2. If lyrics were spoken/sung, ensure vocal ducking or Cloud Stem separation was applied.",
            "3. Exported files have all ID3 chunks, RIFF tags, and metadata completely stripped."
        ]
        report_ctrl.SetValue("\n".join(lines))
        main_sizer.Add(report_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        ok_btn = wx.Button(panel, wx.ID_OK, label="&Close Audit", name="Close Audit Button")
        ok_btn.SetDefault()
        main_sizer.Add(ok_btn, 0, wx.ALIGN_RIGHT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(main_sizer)
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dlg_sizer)
        self.CentreOnParent()


class SunoCheatSheetDialog(wx.Dialog):
    """
    Accessible Dialog providing the operational protocol and cheat sheet
    for successfully uploading audio to Suno AI without triggering blocks.
    """

    def __init__(self, parent: wx.Window):
        super().__init__(
            parent,
            title=f"GhostWave Stealth Protocol & Suno Cheat Sheet - v{APP_VERSION}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(640, 560)
        )
        self.SetName("Suno AI Anti-Block Protocol Dialog")

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Cheat Sheet Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        title_lbl = wx.StaticText(panel, label="Battle-Tested Suno Upload Protocol (Audible Magic & VIBE Evasion):")
        title_lbl.SetName("Protocol Title Label")
        font = title_lbl.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title_lbl.SetFont(font)
        main_sizer.Add(title_lbl, 0, wx.ALL, 10)

        cheat_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="Suno Protocol Text"
        )

        protocol_lines = [
            "==================================================================",
            "SUNO AI ANTI-BLOCK UPLOAD PROTOCOL (STUDY & RESEARCH REFERENCE)",
            "Original Website: http://technokerslab.blogspot.com/",
            "==================================================================",
            "",
            "1. ALWAYS EXPORT AS LOSSLESS 16-BIT PCM WAV (NOT MP3):",
            "   • MP3 encoding inserts frame headers, encoder delay, and padding bits",
            "     that align with standard MP3 hash buckets.",
            "   • 16-bit PCM WAV contains clean, raw linear samples with zero metadata,",
            "     making landmark extraction less prone to container-fingerprint leaks.",
            "",
            "2. THE ABS (ANNOYING BUT SUCCESSFUL) LIBRARY UPLOAD ROUTE:",
            "   • CRITICAL: DO NOT upload audio directly inside the 'Create' / Studio page!",
            "     Suno's Create page runs an aggressive, synchronous Audible Magic check.",
            "   • INSTEAD: Navigate to 'Library' -> click 'Upload Audio' button.",
            "   • Upload your sanitized WAV chunk there first. Suno's library ingestion",
            "     uses a separate background worker queue with significantly relaxed rules.",
            "   • Once uploaded to Library, click the '...' menu on the track and choose",
            "     'Extend' or 'Create with Audio'.",
            "",
            "3. KEEP DURATION BETWEEN 18 AND 25 SECONDS (SWEET SPOT):",
            "   • Full songs provide hundreds of contiguous landmark constellations.",
            "   • Audible Magic requires at least 5-8 contiguous seconds of constellation",
            "     matches to produce a high-confidence match.",
            "   • Slicing your audio into 20-24s chunks destroys long-horizon alignment",
            "     and severely degrades Suno's VIBE neural embedding model.",
            "   • Use the built-in 'ABS Audio Slicer' tool to split tracks automatically.",
            "",
            "4. CLEAR BROWSER SESSION OR USE INCOGNITO IF RECENTLY BLOCKED:",
            "   • If Suno displayed 'This audio matches an existing recording', your",
            "     account session and IP have an elevated risk flag for 30-60 minutes.",
            "   • Open an Incognito / Private Window or clear Suno cookies and cache",
            "     before uploading the sanitized file.",
            "",
            "5. VOCAL ANNIHILATION & LYRICS MODERATION:",
            "   • Lead vocals contain the most prominent biometric and melodic markers.",
            "   • Always use 'Zero-Match Nuclear Cloak' (or Cloud API Demucs isolation)",
            "     to eradicate the center vocal channel.",
            "   • If pasting lyrics into Suno, use the 'Lyrics Sanitizer' tab to strip",
            "     celebrity and artist names to avoid prompt filter rejections.",
            "",
            "6. HARMONIC DRONE & PREAMBLE MASKING:",
            "   • The 'Continuous Camouflage Bed' injects an analog fifths drone at -26 dBFS",
            "     which populates the STFT spectrogram with thousands of decoy energy peaks,",
            "     preventing Audible Magic's peak pickers from indexing original audio.",
            "==================================================================",
            "Open Source Notice:",
            "GhostWave Studio is open source software. To contribute or inspect source code,",
            "visit: https://github.com/muhamadalfian20892/GhostWave-Studio",
            "=================================================================="
        ]
        cheat_ctrl.SetValue("\n".join(protocol_lines))
        main_sizer.Add(cheat_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        ok_btn = wx.Button(panel, wx.ID_OK, label="&Close Protocol", name="Close Protocol Button")
        ok_btn.SetDefault()
        main_sizer.Add(ok_btn, 0, wx.ALIGN_RIGHT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(main_sizer)
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dlg_sizer)
        self.CentreOnParent()


class AbsAudioSlicerDialog(wx.Dialog):
    """
    Accessible Dialog to slice audio tracks into safe 20-24s 16-bit PCM WAV chunks
    for Suno Library upload (The ABS Technique).
    """

    def __init__(self, parent: wx.Window, audio_processor: AudioProcessor, initial_file: Optional[str] = None):
        super().__init__(
            parent,
            title=f"ABS Audio Slicer - GhostWave Studio v{APP_VERSION}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(580, 520)
        )
        self.SetName("ABS Audio Slicer Dialog")
        self.processor = audio_processor
        self.initial_file = initial_file
        self.is_slicing = False

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("ABS Audio Slicer Panel")
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        info_lbl = wx.StaticText(
            panel,
            label="Slices audio into clean 20-24s 16-bit PCM WAV chunks with zero padding or metadata.\n"
                  "Uploading short chunks to Suno Library prevents Audible Magic long-horizon matches."
        )
        info_lbl.SetName("ABS Slicer Description Label")
        main_sizer.Add(info_lbl, 0, wx.ALL, 10)

        # Source File Picker
        file_box = wx.StaticBox(panel, label="Source Audio File")
        file_box.SetName("Slicer Source File Group")
        fb_sizer = wx.StaticBoxSizer(file_box, wx.HORIZONTAL)

        self.file_path_ctrl = wx.TextCtrl(panel, style=wx.TE_READONLY, name="Slicer Audio File Path")
        self.file_path_ctrl.SetToolTip("Path to audio file to slice.")
        if initial_file and os.path.exists(initial_file):
            self.file_path_ctrl.SetValue(initial_file)
        else:
            self.file_path_ctrl.SetValue("No file selected.")
        fb_sizer.Add(self.file_path_ctrl, 1, wx.EXPAND | wx.RIGHT, 8)

        self.browse_btn = wx.Button(panel, label="&Browse...", name="Browse Slicer Audio File Button")
        self.browse_btn.SetToolTip("Select source audio file to slice.")
        self.browse_btn.Bind(wx.EVT_BUTTON, self.on_browse)
        fb_sizer.Add(self.browse_btn, 0)
        main_sizer.Add(fb_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Chunk Settings Grid
        grid = wx.FlexGridSizer(cols=2, vgap=10, hgap=12)
        grid.AddGrowableCol(1, 1)

        dur_lbl = wx.StaticText(panel, label="Chunk Duration (Seconds):")
        dur_lbl.SetName("Chunk Duration Label")
        grid.Add(dur_lbl, 0, wx.ALIGN_CENTER_VERTICAL)

        self.chunk_spin = wx.SpinCtrlDouble(
            panel,
            value="22.0",
            min=10.0,
            max=60.0,
            inc=1.0,
            name="Chunk Duration Seconds Input"
        )
        self.chunk_spin.SetDigits(1)
        self.chunk_spin.SetToolTip("Optimal safe chunk duration for Suno Library is 20.0 to 24.0 seconds.")
        grid.Add(self.chunk_spin, 0, wx.EXPAND)

        out_dir_lbl = wx.StaticText(panel, label="Output Directory:")
        out_dir_lbl.SetName("Output Directory Label")
        grid.Add(out_dir_lbl, 0, wx.ALIGN_CENTER_VERTICAL)

        out_dir_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.out_dir_ctrl = wx.TextCtrl(panel, name="Slicer Output Directory Path")
        self.out_dir_ctrl.SetToolTip("Directory where chunk files will be saved.")
        if initial_file and os.path.exists(initial_file):
            default_dir = os.path.join(os.path.dirname(initial_file), "suno_chunks")
        else:
            default_dir = os.path.join(os.getcwd(), "suno_chunks")
        self.out_dir_ctrl.SetValue(default_dir)
        out_dir_sizer.Add(self.out_dir_ctrl, 1, wx.EXPAND | wx.RIGHT, 8)

        self.browse_dir_btn = wx.Button(panel, label="Bro&wse Folder...", name="Browse Slicer Output Directory Button")
        self.browse_dir_btn.SetToolTip("Choose destination directory for slices.")
        self.browse_dir_btn.Bind(wx.EVT_BUTTON, self.on_browse_dir)
        out_dir_sizer.Add(self.browse_dir_btn, 0)
        grid.Add(out_dir_sizer, 1, wx.EXPAND)

        main_sizer.Add(grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Slice Action Button & Gauge
        action_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.slice_btn = wx.Button(panel, label="&Generate Suno Chunks", name="Generate Suno Chunks Button")
        self.slice_btn.SetToolTip("Split audio into lossless WAV chunks.")
        self.slice_btn.Bind(wx.EVT_BUTTON, self.on_slice)
        action_sizer.Add(self.slice_btn, 0, wx.RIGHT, 10)

        self.gauge = wx.Gauge(panel, range=100, style=wx.GA_HORIZONTAL | wx.GA_SMOOTH, name="Slicing Progress")
        action_sizer.Add(self.gauge, 1, wx.ALIGN_CENTER_VERTICAL)
        main_sizer.Add(action_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Output Log TextCtrl
        log_lbl = wx.StaticText(panel, label="Slicer Generation Log:")
        log_lbl.SetName("Slicer Generation Log Label")
        main_sizer.Add(log_lbl, 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)

        self.log_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="Slicer Generation Log"
        )
        self.log_ctrl.SetToolTip("Log of sliced audio chunks.")
        main_sizer.Add(self.log_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Close button
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.close_btn = wx.Button(panel, wx.ID_CANCEL, label="&Close", name="Close Slicer Dialog Button")
        btn_sizer.Add(self.close_btn, 0)
        main_sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(main_sizer)
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dlg_sizer)
        self.CentreOnParent()

    def on_browse(self, event: wx.CommandEvent):
        wildcard = (
            "Supported Audio Files (*.wav;*.mp3;*.flac;*.m4a;*.ogg)|*.wav;*.mp3;*.flac;*.m4a;*.ogg|"
            "All Files (*.*)|*.*"
        )
        with wx.FileDialog(self, "Select Audio File to Slice", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                self.file_path_ctrl.SetValue(path)
                default_dir = os.path.join(os.path.dirname(path), "suno_chunks")
                self.out_dir_ctrl.SetValue(default_dir)

    def on_browse_dir(self, event: wx.CommandEvent):
        with wx.DirDialog(self, "Select Destination Folder for Chunks", defaultPath=self.out_dir_ctrl.GetValue()) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.out_dir_ctrl.SetValue(dlg.GetPath())

    def on_slice(self, event: wx.CommandEvent):
        file_path = self.file_path_ctrl.GetValue().strip()
        if not os.path.exists(file_path):
            wx.MessageBox("Please select a valid source audio file first.", "File Not Found", wx.OK | wx.ICON_WARNING, self)
            return

        out_dir = self.out_dir_ctrl.GetValue().strip()
        os.makedirs(out_dir, exist_ok=True)
        chunk_dur = float(self.chunk_spin.GetValue())

        self.slice_btn.Enable(False)
        self.browse_btn.Enable(False)
        self.gauge.SetValue(20)
        self.log_ctrl.Clear()
        self.log_ctrl.AppendText(f"Slicing '{os.path.basename(file_path)}' into {chunk_dur:.1f}s WAV chunks...\n")

        def worker():
            try:
                chunks = self.processor.slice_audio(file_path, out_dir, chunk_duration_sec=chunk_dur)
                def done():
                    self.gauge.SetValue(100)
                    self.slice_btn.Enable(True)
                    self.browse_btn.Enable(True)
                    self.log_ctrl.AppendText(f"\nSuccessfully created {len(chunks)} chunk(s) in:\n{out_dir}\n\n")
                    for i, c in enumerate(chunks, 1):
                        self.log_ctrl.AppendText(f"Part {i:02d}: {os.path.basename(c)}\n")
                    self.log_ctrl.AppendText("\nUPLOAD INSTRUCTION:\nGo to Suno Library -> Upload Audio -> Upload Part 01 first.\n")
                    wx.Bell()
                    wx.MessageBox(
                        f"Successfully generated {len(chunks)} chunks!\nSaved to: {out_dir}\n\nUpload via Suno Library -> Upload Audio.",
                        "Chunks Created",
                        wx.OK | wx.ICON_INFORMATION,
                        self
                    )
                wx.CallAfter(done)
            except Exception as ex:
                def fail():
                    self.gauge.SetValue(0)
                    self.slice_btn.Enable(True)
                    self.browse_btn.Enable(True)
                    self.log_ctrl.AppendText(f"\nError slicing audio: {str(ex)}\n")
                    wx.MessageBox(f"Slicing failed: {str(ex)}", "Error", wx.OK | wx.ICON_ERROR, self)
                wx.CallAfter(fail)

        threading.Thread(target=worker, daemon=True).start()


class AudioFileDropTarget(wx.FileDropTarget):
    """
    Drag and drop target allowing users to drag audio files directly
    from File Explorer into the Audio Sanitizer tab.
    """

    def __init__(self, panel: AudioSanitizerPanel):
        super().__init__()
        self.panel = panel

    def OnDropFiles(self, x: int, y: int, filenames: list[str]) -> bool:
        if not filenames:
            return False
        first_file = filenames[0]
        valid_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
        ext = os.path.splitext(first_file)[1].lower()
        if ext in valid_exts:
            self.panel.load_file(first_file)
            return True
        else:
            wx.Bell()
            wx.MessageBox(
                f"Unsupported file format: {ext}\nPlease drop a WAV, MP3, FLAC, M4A, OGG, or AAC file.",
                "Unsupported Audio Format",
                wx.OK | wx.ICON_WARNING,
                self.panel
            )
            return False


class LanguageSelectionDialog(wx.Dialog):
    """
    Accessible dialog prompting users to choose their preferred interface language.
    Shown automatically on first startup and accessible from Settings or Language menu.
    """

    def __init__(self, parent: Optional[wx.Window] = None, current_lang: str = "en"):
        super().__init__(
            parent,
            title=tr("LANG_SELECT_TITLE"),
            style=wx.DEFAULT_DIALOG_STYLE,
            size=(460, 270)
        )
        self.SetName("Language Selection Dialog")
        self.current_lang = current_lang

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Language Selection Panel")
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        prompt_label = wx.StaticText(
            panel,
            label=tr("LANG_SELECT_PROMPT")
        )
        prompt_label.SetName("Language Prompt Label")
        main_sizer.Add(prompt_label, 0, wx.ALL, 12)

        languages = [("en", "English"), ("id", "Bahasa Indonesia")]
        choices = [f"{name} ({code.upper()})" for code, name in languages]
        self.lang_codes = [code for code, _ in languages]

        initial_sel = 0
        if current_lang in self.lang_codes:
            initial_sel = self.lang_codes.index(current_lang)

        self.radio_box = wx.RadioBox(
            panel,
            label=tr("MENU_LANGUAGE"),
            choices=choices,
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
            name="Language Selection Radio Group"
        )
        self.radio_box.SetSelection(initial_sel)
        main_sizer.Add(self.radio_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddStretchSpacer()

        self.save_btn = wx.Button(
            panel,
            wx.ID_OK,
            label=tr("LANG_SELECT_BTN"),
            name="Save Language Selection Button"
        )
        self.save_btn.SetDefault()
        btn_sizer.Add(self.save_btn, 0, wx.RIGHT, 6)

        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        panel.SetSizer(main_sizer)

        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dialog_sizer)
        self.CentreOnParent()

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_hook)

    def on_key_hook(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()

    def get_selected_language(self) -> str:
        idx = self.radio_box.GetSelection()
        if 0 <= idx < len(self.lang_codes):
            return self.lang_codes[idx]
        return "en"


class SupportTicketDialog(wx.Dialog):
    """
    Accessible Support Center Dialog for GhostWave Studio v1.4.0.
    Enables users to submit support tickets, report bugs, request features,
    and review ticket discussion threads directly within the application.
    """

    def __init__(self, parent: Optional[wx.Window] = None):
        super().__init__(
            parent,
            title=f"{tr('TICKET_DIALOG_TITLE')} - GhostWave Studio v{APP_VERSION}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(720, 620)
        )
        self.SetName("Support Center Dialog")
        self.config = GhostWaveConfig.load()
        self.active_ticket: Optional[dict] = None

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Support Center Panel")
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        self.notebook = wx.Notebook(panel, style=wx.NB_TOP | wx.TAB_TRAVERSAL)
        self.notebook.SetName("Support Center Tabs")

        # Tab 1: Submit New Ticket
        self.new_ticket_page = wx.Panel(self.notebook, style=wx.TAB_TRAVERSAL)
        self._build_new_ticket_tab(self.new_ticket_page)
        self.notebook.AddPage(self.new_ticket_page, tr("TICKET_TAB_NEW"), select=True)

        # Tab 2: My Tickets & Discussion Thread
        self.my_tickets_page = wx.Panel(self.notebook, style=wx.TAB_TRAVERSAL)
        self._build_my_tickets_tab(self.my_tickets_page)
        self.notebook.AddPage(self.my_tickets_page, tr("TICKET_TAB_LIST"), select=False)

        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 10)

        # Bottom Close Button
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddStretchSpacer()
        self.close_btn = wx.Button(
            panel,
            wx.ID_CANCEL,
            label=tr("TICKET_CLOSE_BTN"),
            name="Close Support Dialog Button"
        )
        btn_sizer.Add(self.close_btn, 0)
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(main_sizer)
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dlg_sizer)
        self.CentreOnParent()

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_hook)
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_page_changed)

    def on_key_hook(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()

    def on_page_changed(self, event: wx.BookCtrlEvent):
        if event.GetSelection() == 1:
            self._populate_ticket_list()
        event.Skip()

    def _build_new_ticket_tab(self, page: wx.Panel):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Category Row
        cat_sizer = wx.BoxSizer(wx.HORIZONTAL)
        cat_lbl = wx.StaticText(page, label=tr("TICKET_CATEGORY_LABEL"))
        cat_lbl.SetName("Ticket Category Label")
        cat_sizer.Add(cat_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.category_choice = wx.Choice(
            page,
            choices=[
                tr("TICKET_CAT_BUG"),
                tr("TICKET_CAT_FEATURE"),
                tr("TICKET_CAT_SUPPORT"),
                tr("TICKET_CAT_OTHER")
            ],
            name="Ticket Category Selection"
        )
        self.category_choice.SetSelection(0)
        self.category_choice.SetToolTip("Select the category that best describes your inquiry.")
        cat_sizer.Add(self.category_choice, 1, wx.EXPAND)
        sizer.Add(cat_sizer, 0, wx.EXPAND | wx.ALL, 8)

        # Title Row
        title_lbl = wx.StaticText(page, label=tr("TICKET_TITLE_LABEL"))
        title_lbl.SetName("Ticket Title Label")
        sizer.Add(title_lbl, 0, wx.LEFT | wx.RIGHT | wx.TOP, 8)

        self.title_ctrl = wx.TextCtrl(page, name="Ticket Title Input")
        self.title_ctrl.SetToolTip("Short summary of the issue or request.")
        sizer.Add(self.title_ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Description Row
        desc_lbl = wx.StaticText(page, label=tr("TICKET_DESC_LABEL"))
        desc_lbl.SetName("Ticket Description Label")
        sizer.Add(desc_lbl, 0, wx.LEFT | wx.RIGHT | wx.TOP, 8)

        self.desc_ctrl = wx.TextCtrl(page, style=wx.TE_MULTILINE, name="Ticket Description Input")
        self.desc_ctrl.SetToolTip("Provide detailed steps to reproduce the problem or explain your inquiry.")
        sizer.Add(self.desc_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Attach Diagnostics Checkbox
        self.sysinfo_chk = wx.CheckBox(
            page,
            label=tr("TICKET_ATTACH_SYSINFO"),
            name="Attach System Diagnostics Checkbox"
        )
        self.sysinfo_chk.SetValue(True)
        sizer.Add(self.sysinfo_chk, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Status Label
        self.submit_status_lbl = wx.StaticText(page, label="", name="Submit Status Label")
        sizer.Add(self.submit_status_lbl, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # Submit Button
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddStretchSpacer()
        self.submit_btn = wx.Button(page, label=tr("TICKET_SUBMIT_BTN"), name="Submit Ticket Button")
        self.submit_btn.SetDefault()
        self.submit_btn.Bind(wx.EVT_BUTTON, self.on_submit_ticket)
        btn_sizer.Add(self.submit_btn, 0)
        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        page.SetSizer(sizer)

    def _build_my_tickets_tab(self, page: wx.Panel):
        sizer = wx.BoxSizer(wx.VERTICAL)
        split_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left Column: Ticket List
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        list_lbl = wx.StaticText(page, label=tr("TICKET_TAB_LIST"))
        left_sizer.Add(list_lbl, 0, wx.BOTTOM, 4)

        self.ticket_listbox = wx.ListBox(page, style=wx.LB_SINGLE, name="User Tickets List")
        self.ticket_listbox.Bind(wx.EVT_LISTBOX, self.on_ticket_selected)
        left_sizer.Add(self.ticket_listbox, 1, wx.EXPAND)

        refresh_btn = wx.Button(page, label=tr("TICKET_REFRESH_BTN"), name="Refresh Ticket List Button")
        refresh_btn.Bind(wx.EVT_BUTTON, lambda e: self._refresh_ticket_thread())
        left_sizer.Add(refresh_btn, 0, wx.TOP | wx.EXPAND, 6)

        split_sizer.Add(left_sizer, 1, wx.EXPAND | wx.RIGHT, 10)

        # Right Column: Conversation Thread & Reply
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        self.ticket_status_lbl = wx.StaticText(page, label=tr("TICKET_SELECT_PROMPT"), name="Ticket Status Header")
        right_sizer.Add(self.ticket_status_lbl, 0, wx.BOTTOM, 4)

        self.thread_ctrl = wx.TextCtrl(
            page,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="Ticket Discussion Thread Text"
        )
        right_sizer.Add(self.thread_ctrl, 1, wx.EXPAND | wx.BOTTOM, 6)

        reply_lbl = wx.StaticText(page, label=tr("TICKET_REPLY_LABEL"))
        right_sizer.Add(reply_lbl, 0, wx.BOTTOM, 2)

        reply_box = wx.BoxSizer(wx.HORIZONTAL)
        self.reply_ctrl = wx.TextCtrl(page, name="Ticket Reply Input Box")
        reply_box.Add(self.reply_ctrl, 1, wx.EXPAND | wx.RIGHT, 6)

        self.reply_btn = wx.Button(page, label=tr("TICKET_REPLY_BTN"), name="Send Reply Button")
        self.reply_btn.Bind(wx.EVT_BUTTON, self.on_send_reply)
        reply_box.Add(self.reply_btn, 0)
        right_sizer.Add(reply_box, 0, wx.EXPAND)

        split_sizer.Add(right_sizer, 2, wx.EXPAND)
        sizer.Add(split_sizer, 1, wx.EXPAND | wx.ALL, 8)
        page.SetSizer(sizer)

        self._populate_ticket_list()

    def _populate_ticket_list(self):
        self.ticket_listbox.Clear()
        tickets = getattr(self.config, "user_tickets", []) or []
        for t in tickets:
            t_id = t.get("ticket_id", "?")
            t_title = t.get("title", "Untitled")
            t_status = t.get("status", "open").upper()
            self.ticket_listbox.Append(f"#{t_id} [{t_status}] {t_title}", t)
        if tickets:
            self.ticket_listbox.SetSelection(0)
            self._display_selected_ticket()
        else:
            self.ticket_status_lbl.SetLabel(tr("TICKET_NO_TICKETS"))
            self.thread_ctrl.SetValue("")

    def on_ticket_selected(self, event: wx.CommandEvent):
        self._display_selected_ticket()

    def _display_selected_ticket(self):
        sel = self.ticket_listbox.GetSelection()
        if sel == wx.NOT_FOUND:
            return
        ticket_data = self.ticket_listbox.GetClientData(sel)
        if not ticket_data:
            return
        self.active_ticket = ticket_data
        t_id = ticket_data.get("ticket_id")
        t_title = ticket_data.get("title")
        t_status = ticket_data.get("status", "open")
        status_label = tr("TICKET_STATUS_OPEN") if t_status == "open" else tr("TICKET_STATUS_CLOSED")
        self.ticket_status_lbl.SetLabel(f"#{t_id} - {t_title} ({status_label})")
        self._refresh_ticket_thread()

    def _refresh_ticket_thread(self):
        if not self.active_ticket:
            return
        ticket_id = self.active_ticket.get("ticket_id")
        ticket_key = self.active_ticket.get("ticket_key")
        self.thread_ctrl.SetValue("Loading replies from support server...")

        def worker():
            success, data, err = support_client.fetch_ticket(ticket_id, ticket_key)

            def done():
                if success:
                    status = data.get("status", "open")
                    self.active_ticket["status"] = status
                    self.config.save()
                    status_label = tr("TICKET_STATUS_OPEN") if status == "open" else tr("TICKET_STATUS_CLOSED")
                    self.ticket_status_lbl.SetLabel(
                        f"#{ticket_id} - {self.active_ticket.get('title')} ({status_label})"
                    )

                    lines = []
                    comments = data.get("comments", [])
                    if not comments:
                        lines.append("No replies yet. Your ticket is currently awaiting developer review.")
                    else:
                        for c in comments:
                            author = "Developer (Muhamad Alfian)" if c.get("is_admin") else "You"
                            time_str = c.get("created_at", "")[:19].replace("T", " ")
                            lines.append(f"[{author}] {time_str}")
                            lines.append(c.get("body", "").strip())
                            lines.append("-" * 48)
                    self.thread_ctrl.SetValue("\n\n".join(lines))
                else:
                    self.thread_ctrl.SetValue(f"Unable to fetch replies: {err}")

            wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    def on_submit_ticket(self, event: wx.CommandEvent):
        title = self.title_ctrl.GetValue().strip()
        desc = self.desc_ctrl.GetValue().strip()
        if not title or not desc:
            wx.Bell()
            wx.MessageBox(
                "Please enter both a title and description for your ticket.",
                "Missing Fields",
                wx.OK | wx.ICON_WARNING,
                self
            )
            return

        cat_idx = self.category_choice.GetSelection()
        cat_map = ["Bug Report", "Feature Request", "Question & Support", "Other"]
        category = cat_map[cat_idx] if cat_idx < len(cat_map) else "Support"

        self.submit_btn.Enable(False)
        self.submit_status_lbl.SetLabel(tr("TICKET_SUBMITTING"))

        def worker():
            success, res, err = support_client.create_ticket(
                category=category,
                title=title,
                description=desc,
                client_version=APP_VERSION,
                attach_sys_info=self.sysinfo_chk.IsChecked()
            )

            def done():
                self.submit_btn.Enable(True)
                if success:
                    ticket_record = {
                        "ticket_id": res.get("ticket_id"),
                        "ticket_key": res.get("ticket_key"),
                        "title": title,
                        "category": category,
                        "status": res.get("status", "open"),
                        "created_at": res.get("created_at", "")
                    }
                    self.config.add_user_ticket(ticket_record)
                    self.config.save()

                    self.title_ctrl.SetValue("")
                    self.desc_ctrl.SetValue("")
                    self.submit_status_lbl.SetLabel("")
                    wx.Bell()
                    wx.MessageBox(
                        tr("TICKET_SUBMIT_SUCCESS", id=ticket_record["ticket_id"]),
                        "Ticket Submitted",
                        wx.OK | wx.ICON_INFORMATION,
                        self
                    )
                    self._populate_ticket_list()
                    self.notebook.SetSelection(1)
                else:
                    self.submit_status_lbl.SetLabel(tr("TICKET_SUBMIT_FAILED", error=err))
                    wx.MessageBox(
                        tr("TICKET_SUBMIT_FAILED", error=err),
                        "Submission Error",
                        wx.OK | wx.ICON_ERROR,
                        self
                    )

            wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    def on_send_reply(self, event: wx.CommandEvent):
        if not self.active_ticket:
            return
        reply_msg = self.reply_ctrl.GetValue().strip()
        if not reply_msg:
            return

        ticket_id = self.active_ticket.get("ticket_id")
        ticket_key = self.active_ticket.get("ticket_key")
        self.reply_btn.Enable(False)

        def worker():
            success, res, err = support_client.reply_to_ticket(ticket_id, ticket_key, reply_msg)

            def done():
                self.reply_btn.Enable(True)
                if success:
                    self.reply_ctrl.SetValue("")
                    self._refresh_ticket_thread()
                else:
                    wx.MessageBox(tr("TICKET_REPLY_FAILED", error=err), "Error", wx.OK | wx.ICON_ERROR, self)

            wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()


class AccessibilityGuideDialog(wx.Dialog):
    """
    Accessible dialog displaying screen reader navigation guidelines,
    shortcut keys, and software accessibility architecture.
    Features a read-only scrollable text area, clipboard copy, and escape key dismissal.
    """

    def __init__(self, parent: Optional[wx.Window] = None):
        super().__init__(
            parent,
            title=f"Screen Reader Accessibility Guide - GhostWave Studio v{APP_VERSION}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(640, 560)
        )
        self.SetName("Screen Reader Accessibility Guide Dialog")

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Accessibility Guide Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        header_label = wx.StaticText(
            panel,
            label="GhostWave Studio Keyboard Navigation and Screen Reader Shortcuts:"
        )
        header_label.SetName("Accessibility Guide Header Label")
        main_sizer.Add(header_label, 0, wx.ALL, 10)

        guide_content = (
            f"GhostWave Studio v{APP_VERSION}\n"
            "Screen Reader Accessibility and Keyboard Navigation Manual\n\n"
            "Original Website: http://technokerslab.blogspot.com/\n\n"
            "Overview:\n"
            "GhostWave Studio is designed with an accessibility-first architecture.\n"
            "All controls provide explicit MSAA and UI Automation accessible names.\n"
            "Focus order strictly follows visual layout and logical workflow.\n\n"
            "General Navigation:\n"
            "  Tab: Move focus forward to the next interactive control.\n"
            "  Shift+Tab: Move focus backward to the previous interactive control.\n"
            "  Ctrl+Tab: Switch forward between Audio Sanitizer and Lyrics Sanitizer tabs.\n"
            "  Ctrl+Shift+Tab: Switch backward between tabs.\n"
            "  Escape: Dismiss active modal dialogs and popup sheets.\n"
            "  F1: Open this accessibility guide.\n"
            "  Ctrl+T: Open Support Ticket center.\n"
            "  Ctrl+H or Shift+F1: Open the Suno Stealth Protocol cheat sheet.\n\n"
            "Audio Sanitizer Tab (Alt+1 or Ctrl+Tab):\n"
            "  Alt+B: Browse for input audio file.\n"
            "  Alt+P: Process and export sanitized audio.\n"
            "  Alt+E: Run quantitative Chromaprint fingerprint audit.\n"
            "  Alt+S: Toggle advanced settings panel or open ABS Audio Slicer.\n"
            "  Alt+A: Open Cloud API settings dialog.\n"
            "  Ctrl+U: Launch ABS Audio Slicer from any tab.\n"
            "  Ctrl+Shift+S: Export encrypted configuration vault (.sn).\n"
            "  Ctrl+Shift+O: Import encrypted configuration vault (.sn).\n\n"
            "Lyrics Sanitizer Tab (Alt+2 or Ctrl+Tab):\n"
            "  Alt+S: Sanitize and cloak lyrics.\n"
            "  Alt+C: Copy cloaked lyrics output to system clipboard.\n"
            "  Alt+L: Clear lyrics input and output fields.\n"
            "  Alt+M: Load sample copyright test lyrics.\n"
            "  Ctrl+B: Open celebrity and artist blacklist editor.\n"
            "  Ctrl+L: Open Cloud LLM settings dialog.\n\n"
            "Screen Reader Compatibility:\n"
            "Tested with NVDA 2023+, JAWS 2023+, and Windows Narrator.\n"
            "All status updates are announced via status bar text and system audio cues.\n\n"
            "Support and Inquiries:\n"
            "For assistance, questions, or bug reports, email hafiyanajah@gmail.com or submit an in-app support ticket (Ctrl+T).\n\n"
            "Open Source Notice:\n"
            "GhostWave Studio is open source software. If you would like to contribute or view the repository, visit: https://github.com/muhamadalfian20892/GhostWave-Studio"
        )

        self.text_ctrl = wx.TextCtrl(
            panel,
            value=guide_content,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="Accessibility Guide Read-Only Text Area"
        )
        self.text_ctrl.SetToolTip("Read-only accessibility and shortcut instructions.")
        main_sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.copy_btn = wx.Button(
            panel,
            label="&Copy Guide to Clipboard",
            name="Copy Guide to Clipboard Button"
        )
        self.copy_btn.SetToolTip("Copy the full accessibility guide text to clipboard.")
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        btn_sizer.Add(self.copy_btn, 0, wx.RIGHT, 10)

        btn_sizer.AddStretchSpacer()

        self.close_btn = wx.Button(
            panel,
            wx.ID_CANCEL,
            label="&Close Guide",
            name="Close Accessibility Guide Button"
        )
        self.close_btn.SetToolTip("Close this accessibility guide dialog (Hotkey: Escape).")
        self.close_btn.SetDefault()
        btn_sizer.Add(self.close_btn, 0)

        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        panel.SetSizer(main_sizer)

        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dialog_sizer)
        self.CentreOnParent()

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_hook)

    def on_copy(self, event: wx.CommandEvent):
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(self.text_ctrl.GetValue()))
            wx.TheClipboard.Close()
            wx.Bell()

    def on_key_hook(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()


class AboutDialog(wx.Dialog):
    """
    Accessible About dialog with scrollable read-only information,
    architecture summary, and close button.
    """

    def __init__(self, parent: Optional[wx.Window] = None):
        super().__init__(
            parent,
            title=f"About GhostWave Studio v{APP_VERSION}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(620, 540)
        )
        self.SetName("About GhostWave Studio Dialog")

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("About GhostWave Studio Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        header_label = wx.StaticText(
            panel,
            label=f"GhostWave Studio v{APP_VERSION} Software Information:"
        )
        header_label.SetName("About Header Label")
        main_sizer.Add(header_label, 0, wx.ALL, 10)

        about_text = (
            f"GhostWave Studio v{APP_VERSION}\n\n"
            "Acoustic Stealth and Audio Cloaking Engine for Music Generation Platforms.\n"
            "Transforms source audio and lyrics into acoustic structures that bypass\n"
            "automated acoustic fingerprinting (Audible Magic, VIBE) and speech transcription filters.\n\n"
            "Original Website:\n"
            "  http://technokerslab.blogspot.com/\n\n"
            "Encrypted Profile Vault (.sn):\n"
            "  Hardware-keyed AES-256 binary container protecting API keys, presets, and paths.\n"
            "  Zero plaintext credential exposure on disk.\n"
            "  Automatic backup rotation with .bak failover recovery.\n\n"
            "Multi-Vector Evasion Architecture:\n"
            "  Dynamic Micro-Chrono Jitter: Non-linear temporal warp disrupting landmark histograms.\n"
            "  Hilbert Bode Frequency Shifter: Asymmetric single-sideband frequency translation.\n"
            "  Adversarial Pseudo-Peak Injection: High-energy decoy spectral coordinates.\n"
            "  Schroeder All-Pass Dispersion: Multi-stage phase scrambler preserving flat frequency response.\n"
            "  Virtual Acoustic Re-Amping: Early reflections and room boundary diffusion.\n"
            "  Anti-Whisper Scrambler: Center vocal notch and formant ring modulation.\n"
            "  Front-End Preamble Camouflage: Analog noise preamble resetting fingerprint alignment.\n"
            "  Cloud API Stem Isolation: Remote stem separation with zero local model weights.\n"
            "  Chromaprint Audit: Quantitative evasion verification.\n\n"
            "Accessibility:\n"
            "  MSAA and UI Automation compliant controls, keyboard navigation, and screen reader feedback.\n\n"
            "Support and Inquiries:\n"
            "  Contact: hafiyanajah@gmail.com or submit an in-app support ticket (Ctrl+T).\n\n"
            "Open Source Contribution:\n"
            "  GhostWave Studio is open source software. To contribute or inspect source code, visit: https://github.com/muhamadalfian20892/GhostWave-Studio"
        )

        self.text_ctrl = wx.TextCtrl(
            panel,
            value=about_text,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="About Software Read-Only Text Area"
        )
        self.text_ctrl.SetToolTip("Software specifications and architecture overview.")
        main_sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddStretchSpacer()

        self.close_btn = wx.Button(
            panel,
            wx.ID_CANCEL,
            label="&Close About",
            name="Close About Dialog Button"
        )
        self.close_btn.SetToolTip("Close this dialog (Hotkey: Escape).")
        self.close_btn.SetDefault()
        btn_sizer.Add(self.close_btn, 0)

        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        panel.SetSizer(main_sizer)

        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dialog_sizer)
        self.CentreOnParent()

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_hook)

    def on_key_hook(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()


class AudioSanitizerPanel(wx.Panel):
    """
    Accessible Tab for configuring and processing audio files for Suno AI
    using the multi-vector evasion DSP and Cloud API pipeline.
    """

    def __init__(self, parent: wx.Window, audio_processor: AudioProcessor, status_callback):
        super().__init__(parent, style=wx.TAB_TRAVERSAL)
        self.SetName("Audio Sanitizer Panel")
        self.processor = audio_processor
        self.set_status = status_callback
        self.selected_file_path: Optional[str] = None
        self.last_output_path: Optional[str] = None
        self.is_processing = False
        self.cancel_event: Optional[threading.Event] = None

        self._build_ui()
        self.SetDropTarget(AudioFileDropTarget(self))

    def _build_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # -------------------------------------------------------------
        # Section 1: Source Audio Picker
        # -------------------------------------------------------------
        file_box = wx.StaticBox(self, label="1. Source Audio File")
        file_box.SetName("Source Audio File Group")
        file_box_sizer = wx.StaticBoxSizer(file_box, wx.VERTICAL)

        picker_sizer = wx.BoxSizer(wx.HORIZONTAL)

        file_label = wx.StaticText(self, label="Selected File:")
        file_label.SetName("Selected Audio File Label")
        picker_sizer.Add(file_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.file_path_ctrl = wx.TextCtrl(
            self,
            style=wx.TE_READONLY,
            name="Selected Audio File Path"
        )
        self.file_path_ctrl.SetToolTip("Full file path of the chosen audio file.")
        self.file_path_ctrl.SetValue("No audio file selected.")
        picker_sizer.Add(self.file_path_ctrl, 1, wx.EXPAND | wx.RIGHT, 10)

        self.browse_btn = wx.Button(self, label="&Browse Audio File...", name="Browse Audio File Button")
        self.browse_btn.SetToolTip("Open file browser to pick an audio file (Hot key: Alt+B).")
        self.browse_btn.Bind(wx.EVT_BUTTON, self.on_browse)
        picker_sizer.Add(self.browse_btn, 0, wx.ALIGN_CENTER_VERTICAL)

        file_box_sizer.Add(picker_sizer, 0, wx.EXPAND | wx.ALL, 8)

        self.file_info_text = wx.StaticText(
            self,
            label="Audio Specifications: None (Select a file to inspect)",
            name="Audio Specifications Information"
        )
        file_box_sizer.Add(self.file_info_text, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        main_sizer.Add(file_box_sizer, 0, wx.EXPAND | wx.ALL, 8)

        # -------------------------------------------------------------
        # Section 2: Sanitization Parameters & Multi-Vector DSP
        # -------------------------------------------------------------
        param_box = wx.StaticBox(self, label="2. Sanitization Parameters (Acoustic Evasion)")
        param_box.SetName("Sanitization Parameters Group")
        param_box_sizer = wx.StaticBoxSizer(param_box, wx.VERTICAL)

        # Preset Choice + Cloud Settings Button
        preset_sizer = wx.BoxSizer(wx.HORIZONTAL)
        preset_label = wx.StaticText(self, label="Evasion Preset:")
        preset_label.SetName("Evasion Preset Label")
        preset_sizer.Add(preset_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.preset_choice = wx.Choice(
            self,
            choices=[
                "Zero-Match Nuclear Cloak (Ultra Evasion - Recommended)",
                "Balanced Quality & Evasion",
                "Acoustic Re-Amping & Room Simulation",
                "Bode Frequency Shifter & Anti-Chroma",
                "Anti-Whisper / Vocal Scrambler Only",
                "Cloud API Stem Isolation (Zero Local Downloads)"
            ],
            name="Evasion Preset Selection"
        )
        self.preset_choice.SetSelection(0)
        self.preset_choice.SetToolTip("Select a tuned multi-vector evasion profile.")
        self.preset_choice.Bind(wx.EVT_CHOICE, self.on_preset_changed)
        preset_sizer.Add(self.preset_choice, 1, wx.EXPAND | wx.RIGHT, 10)

        self.cloud_btn = wx.Button(self, label="Cloud &API Settings...", name="Cloud API Settings Button")
        self.cloud_btn.SetToolTip("Configure remote stem separation credentials (Hotkey: Alt+A).")
        self.cloud_btn.Bind(wx.EVT_BUTTON, self.on_cloud_settings)
        preset_sizer.Add(self.cloud_btn, 0, wx.ALIGN_CENTER_VERTICAL)

        param_box_sizer.Add(preset_sizer, 0, wx.EXPAND | wx.ALL, 6)

        # Basic Format & Advanced Toggle Row
        format_sizer = wx.BoxSizer(wx.HORIZONTAL)
        format_label = wx.StaticText(self, label="Export Format & Bitrate:")
        format_label.SetName("Export Format Label")
        format_sizer.Add(format_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.format_choice = wx.Choice(
            self,
            choices=[
                "WAV (Uncompressed Lossless 16-bit PCM - Recommended for Suno)",
                "MP3 (320 kbps Constant Bitrate)",
                "MP3 (192 kbps Constant Bitrate)"
            ],
            name="Export Format and Bitrate"
        )
        self.format_choice.SetSelection(0)
        format_sizer.Add(self.format_choice, 1, wx.EXPAND | wx.RIGHT, 10)

        self.toggle_adv_btn = wx.Button(
            self,
            label="&Show Advanced Settings",
            name="Show Advanced Settings Button"
        )
        self.toggle_adv_btn.SetToolTip("Show or hide advanced DSP tuning sliders, evasion checkboxes, and trimmer (Hotkey: Alt+S in this tab).")
        self.toggle_adv_btn.Bind(wx.EVT_BUTTON, self.on_toggle_advanced)
        format_sizer.Add(self.toggle_adv_btn, 0, wx.ALIGN_CENTER_VERTICAL)

        param_box_sizer.Add(format_sizer, 0, wx.EXPAND | wx.ALL, 6)

        # Advanced Settings Collapsible Panel
        self.adv_panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        self.adv_panel.SetName("Advanced DSP Settings Panel")
        adv_sizer = wx.BoxSizer(wx.VERTICAL)

        grid_sizer = wx.FlexGridSizer(cols=2, vgap=6, hgap=15)
        grid_sizer.AddGrowableCol(1, 1)

        # Macro Key Transposition / Pitch Shift
        pitch_label = wx.StaticText(self.adv_panel, label="Key Transposition (Semitones):")
        pitch_label.SetName("Micro Pitch Shift Label")
        grid_sizer.Add(pitch_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.pitch_spin = wx.SpinCtrlDouble(
            self.adv_panel,
            value="2.5",
            min=-12.0,
            max=12.0,
            inc=0.5,
            name="Micro Pitch Shift (Semitones/Cents)"
        )
        self.pitch_spin.SetDigits(1)
        self.pitch_spin.SetToolTip("Shift pitch by macro interval (e.g. +2.5 st) to disrupt neural melody and chord embeddings.")
        grid_sizer.Add(self.pitch_spin, 0, wx.EXPAND)

        # Tempo Adjustment
        tempo_label = wx.StaticText(self.adv_panel, label="Tempo Shift Factor:")
        tempo_label.SetName("Tempo Shift Factor Label")
        grid_sizer.Add(tempo_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.tempo_spin = wx.SpinCtrlDouble(
            self.adv_panel,
            value="0.940",
            min=0.850,
            max=1.150,
            inc=0.010,
            name="Tempo Shift Factor"
        )
        self.tempo_spin.SetDigits(3)
        self.tempo_spin.SetToolTip("Modify playback tempo (e.g. 0.940 = -6%) to disrupt fingerprint time scales.")
        grid_sizer.Add(self.tempo_spin, 0, wx.EXPAND)

        adv_sizer.Add(grid_sizer, 0, wx.EXPAND | wx.ALL, 4)

        # Multi-Vector Evasion Checkboxes
        chk_sizer = wx.FlexGridSizer(cols=2, vgap=4, hgap=12)
        chk_sizer.AddGrowableCol(0, 1)
        chk_sizer.AddGrowableCol(1, 1)

        self.jitter_chk = wx.CheckBox(
            self.adv_panel,
            label="Dynamic Micro-Chrono Jitter (Anti-Landmark / Anti-DTW)",
            name="Dynamic Micro-Chrono Jitter Checkbox"
        )
        self.jitter_chk.SetValue(True)
        self.jitter_chk.SetToolTip("Applies stochastic time-warping (+/-16 ms) to flatten offset alignment histograms.")
        chk_sizer.Add(self.jitter_chk, 0, wx.ALL, 2)

        self.bode_chk = wx.CheckBox(
            self.adv_panel,
            label="Hilbert Bode Frequency Shifter (+8.5 Hz Anti-Chroma)",
            name="Hilbert Bode Frequency Shifter Checkbox"
        )
        self.bode_chk.SetValue(True)
        self.bode_chk.SetToolTip("Decouples harmonic overtones and breaks CQT chroma pitch classes.")
        chk_sizer.Add(self.bode_chk, 0, wx.ALL, 2)

        self.decoy_chk = wx.CheckBox(
            self.adv_panel,
            label="Adversarial Pseudo-Peak Injection (STFT Decoy Landmarks)",
            name="Adversarial Pseudo-Peak Injection Checkbox"
        )
        self.decoy_chk.SetValue(True)
        self.decoy_chk.SetToolTip("Injects psychoacoustically placed decoy peaks into STFT bins to hijack landmark extractors.")
        chk_sizer.Add(self.decoy_chk, 0, wx.ALL, 2)

        self.allpass_chk = wx.CheckBox(
            self.adv_panel,
            label="Schroeder All-Pass Phase Dispersion (Flat Frequency Mag)",
            name="Schroeder All-Pass Phase Dispersion Checkbox"
        )
        self.allpass_chk.SetValue(True)
        self.allpass_chk.SetToolTip("Scrambles phase and disperses transients without altering frequency magnitude.")
        chk_sizer.Add(self.allpass_chk, 0, wx.ALL, 2)

        self.reamping_chk = wx.CheckBox(
            self.adv_panel,
            label="Virtual Acoustic Re-Amping (Studio Room Simulation)",
            name="Virtual Acoustic Re-Amping Checkbox"
        )
        self.reamping_chk.SetValue(True)
        self.reamping_chk.SetToolTip("Convolves with room reflections to simulate re-recording through a monitor speaker.")
        chk_sizer.Add(self.reamping_chk, 0, wx.ALL, 2)

        self.vocal_chk = wx.CheckBox(
            self.adv_panel,
            label="Center Vocal Suppression & Anti-Whisper Scrambler",
            name="Center Vocal Suppression Checkbox"
        )
        self.vocal_chk.SetValue(True)
        self.vocal_chk.SetToolTip("Mid/side vocal cancellation, swept formant ring modulation, and phoneme blurring.")
        chk_sizer.Add(self.vocal_chk, 0, wx.ALL, 2)

        self.preamble_chk = wx.CheckBox(
            self.adv_panel,
            label="Front-End Preamble Camouflage (3.5s Analog Synth Intro)",
            name="Front-End Preamble Camouflage Checkbox"
        )
        self.preamble_chk.SetValue(True)
        self.preamble_chk.SetToolTip("Prepend warm analog intro pad crossfaded into the track to displace frame 0.")
        chk_sizer.Add(self.preamble_chk, 0, wx.ALL, 2)

        self.filter_chk = wx.CheckBox(
            self.adv_panel,
            label="Apply 25 Hz High-Pass & 18 kHz Low-Pass filter to strip fingerprint metadata",
            name="Apply 25 Hz High-Pass and 18 kHz Low-Pass filter to strip fingerprint metadata"
        )
        self.filter_chk.SetValue(True)
        chk_sizer.Add(self.filter_chk, 0, wx.ALL, 2)

        self.dither_chk = wx.CheckBox(
            self.adv_panel,
            label="Add ultra-low floor dither (-65 dB) to break spectral hash",
            name="Add ultra-low floor dither (-65 dB) to break spectral hash"
        )
        self.dither_chk.SetValue(True)
        chk_sizer.Add(self.dither_chk, 0, wx.ALL, 2)

        self.metadata_chk = wx.CheckBox(
            self.adv_panel,
            label="Automatically strip all ID3 tags, artist tags, and album art from output",
            name="Automatically strip all ID3 tags, artist tags, and album art from output"
        )
        self.metadata_chk.SetValue(True)
        chk_sizer.Add(self.metadata_chk, 0, wx.ALL, 2)

        adv_sizer.Add(chk_sizer, 0, wx.EXPAND | wx.ALL, 4)

        # Trimmer sizer (Default unchecked for short clip)
        trim_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.trim_chk = wx.CheckBox(
            self.adv_panel,
            label="Trim audio to maximum duration for Suno free tier safety:",
            name="Trim audio to maximum duration for Suno free tier safety"
        )
        self.trim_chk.SetValue(False)
        self.trim_chk.Bind(wx.EVT_CHECKBOX, self.on_toggle_trim)
        trim_sizer.Add(self.trim_chk, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        self.trim_spin = wx.SpinCtrlDouble(
            self.adv_panel,
            value="24.0",
            min=5.0,
            max=600.0,
            inc=1.0,
            name="Maximum Audio Duration in Seconds"
        )
        self.trim_spin.SetDigits(1)
        self.trim_spin.Enable(False)
        trim_sizer.Add(self.trim_spin, 0, wx.ALIGN_CENTER_VERTICAL)

        seconds_label = wx.StaticText(self.adv_panel, label="seconds")
        seconds_label.SetName("Seconds Unit Label")
        trim_sizer.Add(seconds_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)

        adv_sizer.Add(trim_sizer, 0, wx.ALL, 4)

        self.adv_panel.SetSizer(adv_sizer)
        self.adv_panel.Show(False)
        param_box_sizer.Add(self.adv_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        main_sizer.Add(param_box_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # -------------------------------------------------------------
        # Section 3: Action Buttons & Progress
        # -------------------------------------------------------------
        action_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.process_btn = wx.Button(
            self,
            label="&Process and Export Audio",
            name="Process and Export Audio Button"
        )
        self.process_btn.SetToolTip("Process the audio file with selected parameters and save output (Hotkey: Alt+P).")
        self.process_btn.Bind(wx.EVT_BUTTON, self.on_process)
        action_sizer.Add(self.process_btn, 0, wx.RIGHT, 8)

        self.cancel_btn = wx.Button(
            self,
            label="&Cancel Process",
            name="Cancel Audio Processing Button"
        )
        self.cancel_btn.SetToolTip("Cancel ongoing audio sanitization immediately.")
        self.cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel_processing)
        self.cancel_btn.Enable(False)
        action_sizer.Add(self.cancel_btn, 0, wx.RIGHT, 8)

        self.audit_btn = wx.Button(
            self,
            label="Verify &Evasion Safety (Audit)...",
            name="Verify Evasion Safety Button"
        )
        self.audit_btn.SetToolTip("Run quantitative Chromaprint fingerprint audit (Hotkey: Alt+E).")
        self.audit_btn.Bind(wx.EVT_BUTTON, self.on_audit)
        self.audit_btn.Enable(False)
        action_sizer.Add(self.audit_btn, 0, wx.RIGHT, 8)

        self.slicer_btn = wx.Button(
            self,
            label="ABS Audio &Slicer...",
            name="ABS Audio Slicer Button"
        )
        self.slicer_btn.SetToolTip("Slice audio into safe 20-24s WAV chunks for Suno Library (Hotkey: Alt+S in this tab).")
        self.slicer_btn.Bind(wx.EVT_BUTTON, self.on_slicer)
        action_sizer.Add(self.slicer_btn, 0, wx.RIGHT, 8)

        self.protocol_btn = wx.Button(
            self,
            label="Suno &Protocol...",
            name="Suno Upload Protocol Button"
        )
        self.protocol_btn.SetToolTip("View battle-tested Suno upload guidelines and anti-block rules.")
        self.protocol_btn.Bind(wx.EVT_BUTTON, self.on_protocol)
        action_sizer.Add(self.protocol_btn, 0, wx.RIGHT, 10)

        self.gauge = wx.Gauge(
            self,
            range=100,
            style=wx.GA_HORIZONTAL | wx.GA_SMOOTH,
            name="Processing Progress"
        )
        action_sizer.Add(self.gauge, 1, wx.ALIGN_CENTER_VERTICAL)

        main_sizer.Add(action_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # -------------------------------------------------------------
        # Section 4: Processing Log Area
        # -------------------------------------------------------------
        log_label = wx.StaticText(self, label="Audio Processing Log:")
        log_label.SetName("Audio Processing Log Label")
        main_sizer.Add(log_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 8)

        self.log_ctrl = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="Audio Processing Log"
        )
        self.log_ctrl.SetToolTip("Log detailing all DSP modifications, acoustic evasion steps, and export metrics.")
        main_sizer.Add(self.log_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.SetSizer(main_sizer)

    def on_toggle_advanced(self, event: wx.CommandEvent):
        show_adv = not self.adv_panel.IsShown()
        self.adv_panel.Show(show_adv)
        if show_adv:
            self.toggle_adv_btn.SetLabel("&Hide Advanced Settings")
            self.toggle_adv_btn.SetToolTip("Collapse advanced DSP tuning controls.")
        else:
            self.toggle_adv_btn.SetLabel("&Show Advanced Settings")
            self.toggle_adv_btn.SetToolTip("Expand advanced DSP tuning controls and evasion flags.")
        self.Layout()
        parent = self.GetTopLevelParent()
        if parent:
            parent.Layout()
            parent.Refresh()

    def on_cancel_processing(self, event: wx.CommandEvent):
        if self.is_processing and self.cancel_event:
            self.cancel_event.set()
            self.cancel_btn.Enable(False)
            self.set_status("Cancelling audio processing...")
            self.log_ctrl.AppendText("\nCancellation signal sent to worker thread...\n")

    def on_preset_changed(self, event: wx.CommandEvent):
        idx = self.preset_choice.GetSelection()
        if idx == 0:  # Zero-Match Nuclear Cloak (Ultra Evasion)
            self.pitch_spin.SetValue(2.5)
            self.tempo_spin.SetValue(0.940)
            self.format_choice.SetSelection(0)  # WAV
            self.trim_chk.SetValue(False)
            self.trim_spin.SetValue(24.0)
            self.trim_spin.Enable(False)
            self.jitter_chk.SetValue(True)
            self.bode_chk.SetValue(True)
            self.decoy_chk.SetValue(True)
            self.allpass_chk.SetValue(True)
            self.reamping_chk.SetValue(True)
            self.vocal_chk.SetValue(True)
            self.preamble_chk.SetValue(True)
        elif idx == 1:  # Balanced Quality & Evasion
            self.pitch_spin.SetValue(1.5)
            self.tempo_spin.SetValue(0.970)
            self.format_choice.SetSelection(0)  # WAV
            self.trim_chk.SetValue(False)
            self.trim_spin.SetValue(28.0)
            self.trim_spin.Enable(False)
            self.jitter_chk.SetValue(True)
            self.bode_chk.SetValue(True)
            self.decoy_chk.SetValue(False)
            self.allpass_chk.SetValue(True)
            self.reamping_chk.SetValue(True)
            self.vocal_chk.SetValue(True)
            self.preamble_chk.SetValue(False)
        elif idx == 2:  # Acoustic Re-Amping & Room Simulation
            self.pitch_spin.SetValue(1.0)
            self.tempo_spin.SetValue(0.980)
            self.trim_chk.SetValue(False)
            self.trim_spin.Enable(False)
            self.jitter_chk.SetValue(False)
            self.bode_chk.SetValue(False)
            self.decoy_chk.SetValue(False)
            self.allpass_chk.SetValue(True)
            self.reamping_chk.SetValue(True)
            self.vocal_chk.SetValue(True)
            self.preamble_chk.SetValue(True)
        elif idx == 3:  # Bode Frequency Shifter & Anti-Chroma
            self.pitch_spin.SetValue(2.0)
            self.tempo_spin.SetValue(0.950)
            self.trim_chk.SetValue(False)
            self.trim_spin.Enable(False)
            self.jitter_chk.SetValue(True)
            self.bode_chk.SetValue(True)
            self.decoy_chk.SetValue(True)
            self.allpass_chk.SetValue(True)
            self.reamping_chk.SetValue(False)
            self.vocal_chk.SetValue(True)
            self.preamble_chk.SetValue(False)
        elif idx == 4:  # Anti-Whisper / Vocal Scrambler Only
            self.pitch_spin.SetValue(0.0)
            self.tempo_spin.SetValue(1.000)
            self.trim_chk.SetValue(False)
            self.trim_spin.Enable(False)
            self.jitter_chk.SetValue(False)
            self.bode_chk.SetValue(False)
            self.decoy_chk.SetValue(False)
            self.allpass_chk.SetValue(False)
            self.reamping_chk.SetValue(False)
            self.vocal_chk.SetValue(True)
            self.preamble_chk.SetValue(False)
        elif idx == 5:  # Cloud API Stem Isolation
            self.pitch_spin.SetValue(2.0)
            self.tempo_spin.SetValue(0.950)
            self.format_choice.SetSelection(0)  # WAV
            self.trim_chk.SetValue(False)
            self.trim_spin.Enable(False)
            self.jitter_chk.SetValue(True)
            self.bode_chk.SetValue(True)
            self.decoy_chk.SetValue(False)
            self.allpass_chk.SetValue(True)
            self.reamping_chk.SetValue(True)
            self.vocal_chk.SetValue(False)
            self.preamble_chk.SetValue(True)

    def on_cloud_settings(self, event: wx.CommandEvent):
        dlg = CloudApiDialog(self)
        dlg.ShowModal()
        dlg.Destroy()

    def on_audit(self, event: wx.CommandEvent):
        if not self.selected_file_path or not self.last_output_path:
            wx.MessageBox("Please process an audio file first to generate the audit report.", "No File", wx.OK | wx.ICON_INFORMATION, self)
            return

        metrics = benchmark_evasion_metrics(self.selected_file_path, self.last_output_path)
        dlg = EvasionAuditDialog(self, metrics, self.selected_file_path, self.last_output_path)
        dlg.ShowModal()
        dlg.Destroy()

    def on_slicer(self, event: wx.CommandEvent):
        dlg = AbsAudioSlicerDialog(self, self.processor, initial_file=self.selected_file_path)
        dlg.ShowModal()
        dlg.Destroy()

    def on_protocol(self, event: wx.CommandEvent):
        dlg = SunoCheatSheetDialog(self)
        dlg.ShowModal()
        dlg.Destroy()

    def on_toggle_trim(self, event: wx.CommandEvent):
        self.trim_spin.Enable(self.trim_chk.IsChecked())

    def on_browse(self, event: wx.CommandEvent):
        wildcard = (
            "Supported Audio Files (*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac)|*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac|"
            "WAV Audio (*.wav)|*.wav|"
            "MP3 Audio (*.mp3)|*.mp3|"
            "FLAC Audio (*.flac)|*.flac|"
            "M4A / AAC Audio (*.m4a;*.aac)|*.m4a;*.aac|"
            "All Files (*.*)|*.*"
        )
        with wx.FileDialog(
            self,
            message="Choose an audio file to sanitize",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            chosen_path = file_dialog.GetPath()
            self.load_file(chosen_path)

    def load_file(self, file_path: str):
        self.selected_file_path = file_path
        self.file_path_ctrl.SetValue(file_path)
        try:
            info = self.processor.inspect_file(file_path)
            self.file_info_text.SetLabel(f"Audio Specifications: {info.summary_text}")
            self.log_ctrl.AppendText(f"Loaded source file: {os.path.basename(file_path)}\n{info.summary_text}\n\n")
            self.set_status(f"Loaded file: {os.path.basename(file_path)} ({info.duration_formatted})")
            top = self.GetTopLevelParent()
            if top and hasattr(top, "SetTitle"):
                top.SetTitle(f"GhostWave Studio v{APP_VERSION} - [{os.path.basename(file_path)}]")
        except Exception as ex:
            self.file_info_text.SetLabel(f"Audio Specifications: Error reading file ({str(ex)})")
            self.set_status("Error loading audio file.")

    def on_process(self, event: wx.CommandEvent):
        if not self.selected_file_path or not os.path.exists(self.selected_file_path):
            wx.Bell()
            wx.MessageBox(
                "Please select a valid source audio file first.",
                "No Audio File Selected",
                wx.OK | wx.ICON_WARNING,
                self
            )
            self.browse_btn.SetFocus()
            return

        if self.is_processing:
            return

        input_p = Path(self.selected_file_path)
        stem = input_p.stem
        fmt_idx = self.format_choice.GetSelection()

        if fmt_idx == 0:
            default_ext = ".wav"
            wildcard = "WAV Audio (*.wav)|*.wav"
            bitrate_opt = "wav"
        elif fmt_idx == 1:
            default_ext = ".mp3"
            wildcard = "MP3 Audio (*.mp3)|*.mp3"
            bitrate_opt = "320k"
        else:
            default_ext = ".mp3"
            wildcard = "MP3 Audio (*.mp3)|*.mp3"
            bitrate_opt = "192k"

        default_out_name = f"{stem}_suno_ready{default_ext}"

        with wx.FileDialog(
            self,
            message="Save Sanitized Audio File",
            defaultDir=str(input_p.parent),
            defaultFile=default_out_name,
            wildcard=wildcard,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as save_dialog:
            if save_dialog.ShowModal() == wx.ID_CANCEL:
                return
            output_path = save_dialog.GetPath()

        preset_idx = self.preset_choice.GetSelection()
        mode_val = "cloud_api" if preset_idx == 5 else "master_evasion"

        options = AudioSanitizeOptions(
            mode=mode_val,
            pitch_shift_semitones=float(self.pitch_spin.GetValue()),
            tempo_factor=float(self.tempo_spin.GetValue()),
            enable_micro_chrono_jitter=self.jitter_chk.IsChecked(),
            enable_bode_freq_shifter=self.bode_chk.IsChecked(),
            enable_adversarial_peaks=self.decoy_chk.IsChecked(),
            enable_phase_dispersion=self.allpass_chk.IsChecked(),
            enable_reamping_room=self.reamping_chk.IsChecked(),
            enable_vocal_cut=self.vocal_chk.IsChecked(),
            enable_formant_scrambler=self.vocal_chk.IsChecked(),
            inject_preamble=self.preamble_chk.IsChecked(),
            apply_eq_filters=self.filter_chk.IsChecked(),
            apply_dither=self.dither_chk.IsChecked(),
            trim_duration=self.trim_chk.IsChecked(),
            max_duration_seconds=float(self.trim_spin.GetValue()),
            strip_metadata=self.metadata_chk.IsChecked(),
            output_bitrate=bitrate_opt
        )

        self._start_processing_thread(self.selected_file_path, output_path, options)

    def _start_processing_thread(self, input_path: str, output_path: str, options: AudioSanitizeOptions):
        self.cancel_event = threading.Event()
        self.is_processing = True
        self.process_btn.Enable(False)
        self.cancel_btn.Enable(True)
        self.browse_btn.Enable(False)
        self.audit_btn.Enable(False)
        self.gauge.SetValue(0)
        self.log_ctrl.Clear()
        self.log_ctrl.AppendText(f"Starting multi-vector sanitization for '{os.path.basename(input_path)}'...\n")
        self.set_status("Processing audio file... Please wait.")

        def worker():
            last_update = [0.0]
            last_pct = [-1]

            def progress_cb(pct: int, msg: str):
                now = time.time()
                # Throttle progress callbacks to prevent event loop queue flooding on constrained CPUs
                if pct == 100 or pct == 0 or (now - last_update[0] >= 0.04 and abs(pct - last_pct[0]) >= 2):
                    last_update[0] = now
                    last_pct[0] = pct
                    wx.CallAfter(self._update_progress, pct, msg)

            result = self.processor.process(
                input_path,
                output_path,
                options,
                progress_callback=progress_cb,
                cancel_event=self.cancel_event
            )
            wx.CallAfter(self._on_processing_finished, result)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _update_progress(self, percent: int, message: str):
        self.gauge.SetValue(percent)
        self.log_ctrl.AppendText(f"[{percent}%] {message}\n")
        self.set_status(f"[{percent}%] {message}")

    def _on_processing_finished(self, result: AudioProcessResult):
        self.is_processing = False
        self.cancel_btn.Enable(False)
        self.process_btn.Enable(True)
        self.browse_btn.Enable(True)
        self.gauge.SetValue(100 if result.success else 0)

        wx.Bell()

        if result.success:
            self.last_output_path = result.output_path
            self.audit_btn.Enable(True)
            self.set_status(f"Export complete: {os.path.basename(result.output_path)}")

            audit_txt = ""
            if result.fingerprint_similarity_pct is not None:
                audit_txt = (
                    f"\n\nAcoustic Verification:\n"
                    f"Chromaprint Similarity: {result.fingerprint_similarity_pct:.1f}%\n"
                    f"Verdict: {result.evasion_verdict or 'Ready for Suno'}"
                )

            wx.MessageBox(
                f"Audio successfully processed and sanitized for Suno AI!\n\n"
                f"Saved to:\n{result.output_path}\n\n"
                f"Duration: {result.final_duration:.2f} seconds\n"
                f"Acoustic fingerprints decoupled and all metadata stripped.{audit_txt}",
                "Processing Complete",
                wx.OK | wx.ICON_INFORMATION,
                self
            )
        elif result.error_message and "cancelled" in result.error_message.lower():
            self.set_status("Audio processing cancelled by user.")
            self.log_ctrl.AppendText("\nProcess was cancelled by the user. Output discarded.\n")
            wx.MessageBox(
                "Audio processing operation was cancelled.",
                "Processing Cancelled",
                wx.OK | wx.ICON_INFORMATION,
                self
            )
        else:
            self.set_status("Audio processing encountered an error.")
            wx.MessageBox(
                f"Audio processing failed:\n\n{result.error_message or 'Unknown error.'}",
                "Processing Error",
                wx.OK | wx.ICON_ERROR,
                self
            )
        self.process_btn.SetFocus()

    def retranslate_ui(self):
        """Dynamically retranslates UI labels and tooltips in the Audio Sanitizer tab."""
        self.browse_btn.SetLabel(f"&{tr('AUDIO_SELECT_BTN')}")
        self.process_btn.SetLabel(tr("AUDIO_PROCESS_BTN"))
        if self.adv_panel.IsShown():
            self.toggle_adv_btn.SetLabel(tr("AUDIO_HIDE_ADVANCED_BTN"))
        else:
            self.toggle_adv_btn.SetLabel(tr("AUDIO_SHOW_ADVANCED_BTN"))

        self.jitter_chk.SetLabel(tr("AUDIO_CHRONO_JITTER"))
        self.bode_chk.SetLabel(tr("AUDIO_BODE_SHIFT"))
        self.decoy_chk.SetLabel(tr("AUDIO_ADVERSARIAL_PEAKS"))
        self.allpass_chk.SetLabel(tr("AUDIO_PHASE_DISPERSION"))
        self.reamping_chk.SetLabel(tr("AUDIO_REAMP_ROOM"))
        self.vocal_chk.SetLabel(tr("AUDIO_VOCAL_CUT"))
        self.filter_chk.SetLabel(tr("AUDIO_EQ_FILTERS"))
        self.dither_chk.SetLabel(tr("AUDIO_DITHER"))
        self.trim_chk.SetLabel(tr("AUDIO_TRIM_CLIP"))
        self.metadata_chk.SetLabel(tr("AUDIO_STRIP_METADATA"))


class LyricsSanitizerPanel(wx.Panel):
    """
    Accessible Tab for sanitizing, moderating, and formatting lyrics for Suno AI.
    """

    def __init__(self, parent: wx.Window, lyrics_processor: LyricsProcessor, status_callback):
        super().__init__(parent, style=wx.TAB_TRAVERSAL)
        self.SetName("Lyrics Sanitizer Panel")
        self.processor = lyrics_processor
        self.set_status = status_callback

        self._build_ui()

    def _build_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Section 1: Original Lyrics Input
        in_header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        in_label = wx.StaticText(self, label="1. Original Lyrics Input:")
        in_label.SetName("Original Lyrics Input Label")
        in_header_sizer.Add(in_label, 0, wx.ALIGN_CENTER_VERTICAL)
        in_header_sizer.AddStretchSpacer()

        self.input_stats_label = wx.StaticText(
            self,
            label="Characters: 0 | Words: 0 | Lines: 0",
            name="Input Lyrics Statistics"
        )
        in_header_sizer.Add(self.input_stats_label, 0, wx.ALIGN_CENTER_VERTICAL)
        main_sizer.Add(in_header_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)

        self.input_text_ctrl = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE,
            size=(-1, 130),
            name="Original Lyrics Input"
        )
        self.input_text_ctrl.SetToolTip("Type or paste your original lyrics here.")
        self.input_text_ctrl.Bind(wx.EVT_TEXT, self.on_input_text_changed)
        main_sizer.Add(self.input_text_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Section 2: Moderation Filters & Structural Standards
        rules_box = wx.StaticBox(self, label="2. Moderation Filters & Structural Standards")
        rules_box.SetName("Moderation Rules Group")
        rules_box_sizer = wx.StaticBoxSizer(rules_box, wx.VERTICAL)

        celeb_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.celeb_chk = wx.CheckBox(
            self,
            label="Strip known artist and celebrity names",
            name="Strip known artist and celebrity names"
        )
        self.celeb_chk.SetValue(True)
        self.celeb_chk.SetToolTip("Replaces names of copyright artists and celebrities with generic tags.")
        celeb_sizer.Add(self.celeb_chk, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 15)

        self.edit_blacklist_btn = wx.Button(
            self,
            label="Edit &Blacklist...",
            name="Edit Celebrity Blacklist Button"
        )
        self.edit_blacklist_btn.SetToolTip("View and edit the blacklist of celebrity and artist names.")
        self.edit_blacklist_btn.Bind(wx.EVT_BUTTON, self.on_edit_blacklist)
        celeb_sizer.Add(self.edit_blacklist_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        rules_box_sizer.Add(celeb_sizer, 0, wx.ALL, 4)

        self.profanity_chk = wx.CheckBox(
            self,
            label="Filter profanity and flagged slurs",
            name="Filter profanity and flagged slurs"
        )
        self.profanity_chk.SetValue(True)
        self.profanity_chk.SetToolTip("Replaces explicit language and slurs with musical, safe homophones.")
        rules_box_sizer.Add(self.profanity_chk, 0, wx.ALL, 4)

        self.unicode_chk = wx.CheckBox(
            self,
            label="Normalize non-standard characters and diacritics to clean UTF-8",
            name="Normalize non-standard characters and diacritics to clean UTF-8"
        )
        self.unicode_chk.SetValue(True)
        self.unicode_chk.SetToolTip("Flattens accents, replaces smart typographic quotes and dashes, and strips non-printable characters.")
        rules_box_sizer.Add(self.unicode_chk, 0, wx.ALL, 4)

        self.tags_chk = wx.CheckBox(
            self,
            label="Auto-format Suno structural tags ([Verse], [Chorus], [Drop])",
            name="Auto-format Suno structural tags"
        )
        self.tags_chk.SetValue(True)
        self.tags_chk.SetToolTip("Standardizes cues like [Verse], [Chorus], [Drop], [Bridge], [Outro] and fixes nested brackets.")
        rules_box_sizer.Add(self.tags_chk, 0, wx.ALL, 4)

        self.length_chk = wx.CheckBox(
            self,
            label="Character & line length limiter",
            name="Character and line length limiter"
        )
        self.length_chk.SetValue(True)
        self.length_chk.SetToolTip("Audits line lengths against the recommended 80-char ceiling and 3,000 token limit.")
        rules_box_sizer.Add(self.length_chk, 0, wx.ALL, 4)

        main_sizer.Add(rules_box_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Section 3: Copyright Evasion & Lyrics Cloaking Engine
        cloak_box = wx.StaticBox(self, label="3. Copyright Evasion & Lyrics Cloaking Engine")
        cloak_box.SetName("Lyrics Cloaking Group")
        cloak_box_sizer = wx.StaticBoxSizer(cloak_box, wx.VERTICAL)

        self.cloak_chk = wx.CheckBox(
            self,
            label="Enable copyright evasion cloaking (breaks database n-gram detection)",
            name="Enable copyright evasion cloaking"
        )
        self.cloak_chk.SetValue(True)
        self.cloak_chk.SetToolTip("Transforms lyrics using acoustic and cadence disguises so copyright database checks pass.")
        cloak_box_sizer.Add(self.cloak_chk, 0, wx.ALL, 4)

        mode_sizer = wx.BoxSizer(wx.HORIZONTAL)
        mode_lbl = wx.StaticText(self, label="Cloaking Strategy:")
        mode_lbl.SetName("Cloaking Strategy Label")
        mode_sizer.Add(mode_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.cloak_mode_choice = wx.Choice(
            self,
            choices=[
                "Acoustic Spelling Scrambler (Preserves Original Lyrics) [Recommended]",
                "Stealth Hybrid (Acoustic Disguise + Musical Ad-libs)",
                "Phonetic Disguise (Acoustic Homophones Only)",
                "Semantic Cadence (Syllable-Preserved Synonyms)",
                "Cloud API Rewriter (Remote LLM)"
            ],
            name="Lyrics Cloaking Mode Selection"
        )
        self.cloak_mode_choice.SetSelection(0)
        self.cloak_mode_choice.SetToolTip("Select cloaking strategy. Scrambler mode disrupts orthography while preserving exact song words and vocal flow.")
        mode_sizer.Add(self.cloak_mode_choice, 1, wx.EXPAND | wx.RIGHT, 10)

        self.cloud_llm_btn = wx.Button(
            self,
            label="Cloud &LLM Settings...",
            name="Lyrics Cloud LLM Settings Button"
        )
        self.cloud_llm_btn.SetToolTip("Configure Groq, OpenRouter, or OpenAI API credentials in the .sn vault.")
        self.cloud_llm_btn.Bind(wx.EVT_BUTTON, self.on_cloud_llm_settings)
        mode_sizer.Add(self.cloud_llm_btn, 0)
        cloak_box_sizer.Add(mode_sizer, 0, wx.EXPAND | wx.ALL, 4)

        toggles_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.vibrato_chk = wx.CheckBox(
            self,
            label="Add vocal vibrato glides (~) to phrase cadences",
            name="Add vocal vibrato glides to phrase cadences"
        )
        self.vibrato_chk.SetValue(True)
        self.vibrato_chk.SetToolTip("Appends singing vibrato glides (~) to phrase endings to disrupt regex matchers while signaling singing inflection to Suno.")
        toggles_sizer.Add(self.vibrato_chk, 0, wx.RIGHT, 15)

        self.syllable_chk = wx.CheckBox(
            self,
            label="Preserve musical syllable meter per line",
            name="Preserve musical syllable meter per line"
        )
        self.syllable_chk.SetValue(True)
        self.syllable_chk.SetToolTip("Ensures every rewritten line maintains the original syllable count so it fits the audio backing track.")
        toggles_sizer.Add(self.syllable_chk, 0, wx.RIGHT, 15)

        self.ngram_chk = wx.CheckBox(
            self,
            label="Inject rhythmic backing ad-libs (yeah, oh)",
            name="Inject rhythmic backing ad-libs"
        )
        self.ngram_chk.SetValue(False)
        self.ngram_chk.SetToolTip("Injects musical ad-libs to break sequence matchers without disrupting vocals.")
        toggles_sizer.Add(self.ngram_chk, 0)
        cloak_box_sizer.Add(toggles_sizer, 0, wx.ALL, 4)

        main_sizer.Add(cloak_box_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Section 4: Action Buttons
        action_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.sanitize_btn = wx.Button(
            self,
            label="&Sanitize & Cloak Lyrics",
            name="Sanitize and Cloak Lyrics Button"
        )
        self.sanitize_btn.SetToolTip("Run moderation filters and copyright evasion cloaking (Hotkey: Alt+S).")
        self.sanitize_btn.Bind(wx.EVT_BUTTON, self.on_sanitize)
        self.sanitize_btn.SetDefault()
        action_sizer.Add(self.sanitize_btn, 0, wx.RIGHT, 10)

        self.copy_btn = wx.Button(
            self,
            label="&Copy Cloaked Lyrics to Clipboard",
            name="Copy Cloaked Lyrics to Clipboard Button"
        )
        self.copy_btn.SetToolTip("Copy the cloaked output lyrics to system clipboard (Hotkey: Alt+C).")
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        action_sizer.Add(self.copy_btn, 0, wx.RIGHT, 10)

        self.sample_btn = wx.Button(
            self,
            label="Load &Sample Lyrics",
            name="Load Sample Lyrics Button"
        )
        self.sample_btn.SetToolTip("Insert sample test lyrics containing copyright lines, artist names, and tags.")
        self.sample_btn.Bind(wx.EVT_BUTTON, self.on_load_sample)
        action_sizer.Add(self.sample_btn, 0, wx.RIGHT, 10)

        self.clear_btn = wx.Button(
            self,
            label="C&lear",
            name="Clear All Lyrics Button"
        )
        self.clear_btn.SetToolTip("Clear input, output, and logs.")
        self.clear_btn.Bind(wx.EVT_BUTTON, self.on_clear)
        action_sizer.Add(self.clear_btn, 0)

        main_sizer.Add(action_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Section 5: Sanitized Output & Changes Made Log
        out_splitter = wx.BoxSizer(wx.VERTICAL)

        out_label = wx.StaticText(self, label="4. Sanitized & Cloaked Lyrics Output:")
        out_label.SetName("Sanitized Lyrics Output Label")
        out_splitter.Add(out_label, 0, wx.BOTTOM, 4)

        self.output_text_ctrl = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE,
            size=(-1, 130),
            name="Sanitized Lyrics Output"
        )
        self.output_text_ctrl.SetToolTip("Final cloaked lyrics ready to paste into Suno AI.")
        out_splitter.Add(self.output_text_ctrl, 1, wx.EXPAND | wx.BOTTOM, 8)

        changes_label = wx.StaticText(self, label="Changes Made & Copyright Evasion Audit:")
        changes_label.SetName("Changes Made Label")
        out_splitter.Add(changes_label, 0, wx.BOTTOM, 4)

        self.changes_ctrl = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            size=(-1, 120),
            name="Changes Made"
        )
        self.changes_ctrl.SetToolTip("Audit report of copyright evasion metrics, token similarity, and applied changes.")
        out_splitter.Add(self.changes_ctrl, 1, wx.EXPAND)

        main_sizer.Add(out_splitter, 2, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.SetSizer(main_sizer)

    def on_edit_blacklist(self, event: wx.CommandEvent):
        dlg = BlacklistDialog(self, self.processor)
        dlg.ShowModal()
        dlg.Destroy()
        self.set_status(f"Celebrity blacklist updated ({len(self.processor.celebrity_blacklist)} entries).")

    def on_cloud_llm_settings(self, event: wx.CommandEvent):
        dlg = LyricsCloudDialog(self)
        if dlg.ShowModal() == wx.ID_OK:
            self.set_status("Lyrics Cloud LLM settings saved to .sn vault.")
        dlg.Destroy()

    def on_load_sample(self, event: wx.CommandEvent):
        sample_lyrics = (
            "[Verse 1]\n"
            "Just a small town girl, living in a lonely world\n"
            "She took the midnight train going anywhere\n"
            "I saw Taylor Swift and Drake chilling at the late night spot\n"
            "Giving everything we got, forget the bullshit around\n"
            "[Chorus]\n"
            "Don't stop believing, hold on to that feeling\n"
            "Streetlights, people, living just to find emotion\n"
            "[Outro]\n"
            "Fade out into silence."
        )
        self.input_text_ctrl.SetValue(sample_lyrics)
        self.set_status("Loaded sample copyright lyrics into input.")
        self.sanitize_btn.SetFocus()

    def on_input_text_changed(self, event: Optional[wx.CommandEvent] = None):
        val = self.input_text_ctrl.GetValue()
        chars = len(val)
        words = len(val.split())
        lines = len([line for line in val.splitlines() if line.strip()])
        self.input_stats_label.SetLabel(tr("LYRICS_STATS_FORMAT", chars=chars, words=words, lines=lines))
        if event:
            event.Skip()

    def on_clear(self, event: wx.CommandEvent):
        self.input_text_ctrl.Clear()
        self.output_text_ctrl.Clear()
        self.changes_ctrl.Clear()
        self.input_stats_label.SetLabel(tr("LYRICS_STATS_FORMAT", chars=0, words=0, lines=0))
        self.set_status("Cleared lyrics fields.")
        self.input_text_ctrl.SetFocus()

    def on_sanitize(self, event: wx.CommandEvent):
        raw_text = self.input_text_ctrl.GetValue()
        if not raw_text.strip():
            wx.Bell()
            wx.MessageBox(
                "Please enter or paste some lyrics into the input area.",
                "Empty Lyrics Input",
                wx.OK | wx.ICON_WARNING,
                self
            )
            self.input_text_ctrl.SetFocus()
            return

        mode_idx = self.cloak_mode_choice.GetSelection()
        mode_map = ["scramble", "hybrid", "phonetic", "semantic", "cloud"]
        selected_mode = mode_map[mode_idx] if mode_idx < len(mode_map) else "scramble"

        from config_manager import GhostWaveConfig
        sn_cfg = GhostWaveConfig.load()

        result: LyricsSanitizeResult = self.processor.sanitize(
            text=raw_text,
            strip_celebrities=self.celeb_chk.IsChecked(),
            filter_profanity=self.profanity_chk.IsChecked(),
            normalize_unicode=self.unicode_chk.IsChecked(),
            format_tags=self.tags_chk.IsChecked(),
            check_limits=self.length_chk.IsChecked(),
            cloak_lyrics=self.cloak_chk.IsChecked(),
            cloak_mode=selected_mode,
            add_vibrato_glides=self.vibrato_chk.IsChecked(),
            preserve_syllables=self.syllable_chk.IsChecked(),
            break_ngrams=self.ngram_chk.IsChecked(),
            cloud_token=sn_cfg.lyrics_cloud_api_token,
            cloud_provider=sn_cfg.lyrics_cloud_provider,
            cloud_model=sn_cfg.lyrics_cloud_model,
            cloud_endpoint=sn_cfg.lyrics_cloud_endpoint_url
        )

        self.output_text_ctrl.SetValue(result.sanitized_text)

        log_sections = []
        if result.audit:
            aud = result.audit
            log_sections.append("=== COPYRIGHT EVASION AUDIT ===")
            log_sections.append(f"Verdict: {aud.evasion_verdict}")
            log_sections.append(f"4-Gram Sequence Overlap: {aud.ngram_overlap_pct:.1f}%")
            log_sections.append(f"Token Similarity: {aud.token_similarity_pct:.1f}%")
            log_sections.append(f"Syllable Cadence Match: {aud.syllable_accuracy_pct:.1f}%")
            log_sections.append(f"Original Words: {aud.original_words} -> Cloaked Words: {aud.cloaked_words}\n")

        log_sections.append("=== MODERATION & TRANSFORMATION AUDIT ===")
        for chg in result.changes:
            log_sections.append(f"* {chg}")

        if result.warnings:
            log_sections.append("\n=== POTENTIAL ISSUES & WARNINGS ===")
            for warn in result.warnings:
                log_sections.append(f"Warning: {warn}")

        if result.stats:
            log_sections.append("\n=== LYRICS METRICS ===")
            st = result.stats
            log_sections.append(
                f"Characters: {st.get('total_characters', 0)} | "
                f"Lines: {st.get('total_lines', 0)} | "
                f"Words: {st.get('word_count', 0)} | "
                f"Estimated Singing Duration: {st.get('estimated_duration', '0s')}"
            )

        self.changes_ctrl.SetValue("\n".join(log_sections))

        change_count = len(result.changes)
        warning_count = len(result.warnings)
        status_msg = f"Lyrics cloaked and sanitized! {change_count} changes applied, {warning_count} warnings."
        self.set_status(status_msg)

        wx.Bell()
        self.output_text_ctrl.SetFocus()

    def on_copy(self, event: wx.CommandEvent):
        sanitized_text = self.output_text_ctrl.GetValue()
        if not sanitized_text.strip():
            wx.Bell()
            wx.MessageBox(
                "No sanitized lyrics to copy. Please run 'Sanitize Lyrics' first.",
                "Nothing to Copy",
                wx.OK | wx.ICON_INFORMATION,
                self
            )
            return

        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(sanitized_text))
            wx.TheClipboard.Close()
            self.set_status("Sanitized lyrics copied to clipboard!")
            wx.Bell()
            wx.MessageBox(
                "Sanitized lyrics copied to system clipboard successfully!",
                "Copied",
                wx.OK | wx.ICON_INFORMATION,
                self
            )
        else:
            wx.MessageBox(
                "Unable to open system clipboard.",
                "Clipboard Error",
                wx.OK | wx.ICON_ERROR,
                self
            )

    def retranslate_ui(self):
        """Dynamically retranslates UI labels in the Lyrics Sanitizer tab."""
        self.sanitize_btn.SetLabel(tr("LYRICS_PROCESS_BTN"))
        self.copy_btn.SetLabel(tr("LYRICS_COPY_BTN"))
        self.on_input_text_changed(None)


class GhostWaveFrame(wx.Frame):
    """
    Main Application Window for GhostWave Studio - Next-Gen Stealth Audio Cloak.
    """

    def __init__(self):
        super().__init__(
            parent=None,
            id=wx.ID_ANY,
            title=f"GhostWave Studio v{APP_VERSION} - Next-Gen Stealth Audio Cloak",
            size=(900, 760),
            style=wx.DEFAULT_FRAME_STYLE
        )
        self.SetName("GhostWave Studio Main Window")
        self.SetMinSize((760, 620))

        # Core Engines
        self.audio_processor = AudioProcessor()
        self.lyrics_processor = LyricsProcessor()

        # Build Status Bar for accessible live announcements
        self.statusbar = self.CreateStatusBar(1, wx.STB_DEFAULT_STYLE)
        self.statusbar.SetName("Application Status Bar")
        self.set_status_text(f"Ready. GhostWave Studio v{APP_VERSION} initialized with encrypted profile vault (.sn).")

        # Build Menu Bar
        self._build_menu()

        # Main Panel
        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Main Container Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Tabbed Notebook
        self.notebook = wx.Notebook(panel, style=wx.NB_TOP | wx.TAB_TRAVERSAL)
        self.notebook.SetName("Main Navigation Tabs")

        # Tab 1: Audio Sanitizer
        self.audio_tab = AudioSanitizerPanel(self.notebook, self.audio_processor, self.set_status_text)
        self.notebook.AddPage(self.audio_tab, tr("TAB_AUDIO"), select=True)

        # Tab 2: Lyrics Sanitizer
        self.lyrics_tab = LyricsSanitizerPanel(self.notebook, self.lyrics_processor, self.set_status_text)
        self.notebook.AddPage(self.lyrics_tab, tr("TAB_LYRICS"), select=False)

        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 6)
        panel.SetSizer(main_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(frame_sizer)

        self.Centre()

        # Background update check if enabled in encrypted config vault
        try:
            cfg = GhostWaveConfig.load()
            if getattr(cfg, "check_updates_on_startup", True):
                wx.CallLater(1500, self._check_updates_startup)
        except Exception:
            pass

    def _build_menu(self):
        self.menu_bar = wx.MenuBar()

        # File Menu
        self.file_menu = wx.Menu()
        self.open_item = self.file_menu.Append(wx.ID_OPEN, "&Browse Audio File...\tCtrl+O", "Select an audio file for processing")
        self.file_menu.AppendSeparator()
        self.export_profile_item = self.file_menu.Append(
            wx.ID_ANY,
            "&Export Encrypted Profile (.sn)...\tCtrl+Shift+S",
            "Export all settings and API tokens into an encrypted .sn vault"
        )
        self.import_profile_item = self.file_menu.Append(
            wx.ID_ANY,
            "&Import Encrypted Profile (.sn)...\tCtrl+Shift+O",
            "Load and decrypt configuration from a .sn vault"
        )
        self.file_menu.AppendSeparator()
        self.exit_item = self.file_menu.Append(wx.ID_EXIT, tr("MENU_EXIT"), "Exit GhostWave Studio")
        self.menu_bar.Append(self.file_menu, tr("MENU_FILE"))

        # Edit Menu
        self.edit_menu = wx.Menu()
        self.blacklist_item = self.edit_menu.Append(wx.ID_ANY, "Edit Celebrity &Blacklist...\tCtrl+B", "Open the celebrity blacklist editor")
        self.copy_item = self.edit_menu.Append(wx.ID_COPY, "&Copy Sanitized Lyrics\tCtrl+Shift+C", "Copy sanitized lyrics to clipboard")
        self.menu_bar.Append(self.edit_menu, "&Edit")

        # Actions Menu
        self.action_menu = wx.Menu()
        self.process_audio_item = self.action_menu.Append(wx.ID_ANY, "&Process Audio\tCtrl+P", "Execute audio sanitization")
        self.audit_item = self.action_menu.Append(wx.ID_ANY, "Run &Evasion Audit...\tCtrl+E", "Run Chromaprint acoustic audit")
        self.slicer_item = self.action_menu.Append(wx.ID_ANY, "ABS Audio &Slicer...\tCtrl+U", "Slice audio into safe 20-24s WAV chunks")
        self.protocol_item = self.action_menu.Append(wx.ID_ANY, "GhostWave Stealth &Protocol...\tCtrl+H", "View Suno upload protocol and cheat sheet")
        self.cloud_item = self.action_menu.Append(wx.ID_ANY, "Cloud &API Settings (Vault)...\tCtrl+K", "Configure Cloud stem separation credentials")
        self.action_menu.AppendSeparator()
        self.sanitize_lyrics_item = self.action_menu.Append(wx.ID_ANY, "&Sanitize Lyrics\tCtrl+S", "Execute lyrics sanitization")
        self.menu_bar.Append(self.action_menu, "&Actions")

        # Language Menu
        self.lang_menu = wx.Menu()
        self.lang_en_item = self.lang_menu.AppendRadioItem(wx.ID_ANY, tr("MENU_LANG_EN"), "Switch interface language to English")
        self.lang_id_item = self.lang_menu.AppendRadioItem(wx.ID_ANY, tr("MENU_LANG_ID"), "Ganti bahasa antarmuka ke Bahasa Indonesia")
        current_lang = get_language()
        if current_lang == "id":
            self.lang_id_item.Check(True)
        else:
            self.lang_en_item.Check(True)
        self.menu_bar.Append(self.lang_menu, tr("MENU_LANGUAGE"))

        # Help Menu
        self.help_menu = wx.Menu()
        self.protocol_help_item = self.help_menu.Append(wx.ID_ANY, "Stealth &Upload Protocol && Cheat Sheet...\tShift+F1", "View battle-tested Suno upload cheat sheet")
        self.ticket_item = self.help_menu.Append(wx.ID_ANY, tr("MENU_SUPPORT_TICKETS"), "Open support center and submit tickets")
        self.check_update_item = self.help_menu.Append(wx.ID_ANY, tr("MENU_CHECK_UPDATES"), "Check GitHub for latest release and changelog")
        self.a11y_item = self.help_menu.Append(wx.ID_HELP, tr("MENU_ACCESSIBILITY_GUIDE"), "View screen reader accessibility shortcuts")
        self.about_item = self.help_menu.Append(wx.ID_ABOUT, f"{tr('MENU_ABOUT')} v{APP_VERSION}", "About this application")
        self.menu_bar.Append(self.help_menu, tr("MENU_HELP"))

        self.SetMenuBar(self.menu_bar)

        # Bind Menu Events
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_browse(e), self.open_item)
        self.Bind(wx.EVT_MENU, self.on_export_profile, self.export_profile_item)
        self.Bind(wx.EVT_MENU, self.on_import_profile, self.import_profile_item)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(True), self.exit_item)
        self.Bind(wx.EVT_MENU, lambda e: self.lyrics_tab.on_edit_blacklist(e), self.blacklist_item)
        self.Bind(wx.EVT_MENU, lambda e: self.lyrics_tab.on_copy(e), self.copy_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_process(e), self.process_audio_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_audit(e), self.audit_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_slicer(e), self.slicer_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_protocol(e), self.protocol_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_protocol(e), self.protocol_help_item)
        self.Bind(wx.EVT_MENU, lambda e: self.on_change_language("en"), self.lang_en_item)
        self.Bind(wx.EVT_MENU, lambda e: self.on_change_language("id"), self.lang_id_item)
        self.Bind(wx.EVT_MENU, self.on_support_tickets, self.ticket_item)
        self.Bind(wx.EVT_MENU, self.on_check_updates_menu, self.check_update_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_cloud_settings(e), self.cloud_item)
        self.Bind(wx.EVT_MENU, lambda e: self.lyrics_tab.on_sanitize(e), self.sanitize_lyrics_item)
        self.Bind(wx.EVT_MENU, self.on_accessibility_guide, self.a11y_item)
        self.Bind(wx.EVT_MENU, self.on_about, self.about_item)

    def on_support_tickets(self, event: wx.CommandEvent):
        dlg = SupportTicketDialog(self)
        dlg.ShowModal()
        dlg.Destroy()

    def on_change_language(self, lang_code: str):
        set_language(lang_code)
        cfg = GhostWaveConfig.load()
        cfg.language = lang_code
        cfg.save()
        self.retranslate_ui()
        wx.Bell()
        self.set_status_text(f"Language changed to: {lang_code.upper()}")

    def retranslate_ui(self):
        self.SetTitle(f"{tr('APP_TITLE')} v{APP_VERSION} - {tr('APP_SUBTITLE')}")
        self.menu_bar.SetMenuLabel(0, tr("MENU_FILE"))
        self.menu_bar.SetMenuLabel(1, "&Edit")
        self.menu_bar.SetMenuLabel(2, "&Actions")
        self.menu_bar.SetMenuLabel(3, tr("MENU_LANGUAGE"))
        self.menu_bar.SetMenuLabel(4, tr("MENU_HELP"))

        self.exit_item.SetItemLabel(tr("MENU_EXIT"))
        self.lang_en_item.SetItemLabel(tr("MENU_LANG_EN"))
        self.lang_id_item.SetItemLabel(tr("MENU_LANG_ID"))
        self.a11y_item.SetItemLabel(tr("MENU_ACCESSIBILITY_GUIDE"))
        self.about_item.SetItemLabel(f"{tr('MENU_ABOUT')} v{APP_VERSION}")
        self.check_update_item.SetItemLabel(tr("MENU_CHECK_UPDATES"))
        self.ticket_item.SetItemLabel(tr("MENU_SUPPORT_TICKETS"))

        self.notebook.SetPageText(0, tr("TAB_AUDIO"))
        self.notebook.SetPageText(1, tr("TAB_LYRICS"))

        if hasattr(self.audio_tab, "retranslate_ui"):
            self.audio_tab.retranslate_ui()
        if hasattr(self.lyrics_tab, "retranslate_ui"):
            self.lyrics_tab.retranslate_ui()

    def on_export_profile(self, event: wx.CommandEvent):
        """Exports all configurations to a user-specified encrypted .sn file."""
        with wx.FileDialog(
            self,
            message="Export Encrypted GhostWave Profile (.sn)",
            defaultFile=DEFAULT_SN_FILENAME,
            wildcard="GhostWave Encrypted Profile (*.sn)|*.sn|All Files (*.*)|*.*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                cfg = GhostWaveConfig.load()
                # Synchronize current UI controls
                cfg.default_preset = self.audio_tab.preset_choice.GetStringSelection()
                cfg.pitch_shift_semitones = float(self.audio_tab.pitch_spin.GetValue())
                cfg.tempo_factor = float(self.audio_tab.tempo_spin.GetValue())
                cfg.max_duration_seconds = float(self.audio_tab.trim_spin.GetValue())
                fmt_idx = self.audio_tab.format_choice.GetSelection()
                cfg.export_format = "wav" if fmt_idx == 0 else ("320k" if fmt_idx == 1 else "192k")
                saved_path = cfg.save(path)
                self.set_status_text(f"Encrypted profile saved: {os.path.basename(saved_path)}")
                wx.Bell()
                wx.MessageBox(
                    f"Profile encrypted and saved successfully!\n\n"
                    f"File: {saved_path}\n"
                    f"Format: Encrypted Binary Vault (.sn)\n\n"
                    f"Protected by military-grade AES-128-CBC & HMAC-SHA256.\n"
                    f"Only GhostWave Studio can decrypt and read this file.",
                    "Profile Exported",
                    wx.OK | wx.ICON_INFORMATION,
                    self
                )

    def on_import_profile(self, event: wx.CommandEvent):
        """Imports and decrypts a user-specified .sn configuration profile."""
        with wx.FileDialog(
            self,
            message="Import Encrypted GhostWave Profile (.sn)",
            wildcard="GhostWave Encrypted Profile (*.sn)|*.sn|All Files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                try:
                    with open(path, "rb") as f:
                        raw = f.read()
                    cfg = GhostWaveConfig.from_encrypted_bytes(raw)
                    cfg.save()  # Store into default active vault
                    # Synchronize UI
                    self.audio_tab.pitch_spin.SetValue(cfg.pitch_shift_semitones)
                    self.audio_tab.tempo_spin.SetValue(cfg.tempo_factor)
                    self.audio_tab.trim_spin.SetValue(cfg.max_duration_seconds)
                    fmt_idx = 0 if cfg.export_format == "wav" else (1 if cfg.export_format == "320k" else 2)
                    self.audio_tab.format_choice.SetSelection(fmt_idx)
                    self.set_status_text(f"Decrypted and loaded profile: {os.path.basename(path)}")
                    wx.Bell()
                    wx.MessageBox(
                        f"Profile decrypted and applied successfully!\n\n"
                        f"Source File: {path}\n"
                        f"App Version: {cfg.app_version}\n"
                        f"Cloud Provider: {cfg.cloud_provider}\n"
                        f"Preset: {cfg.default_preset}",
                        "Profile Imported",
                        wx.OK | wx.ICON_INFORMATION,
                        self
                    )
                except Exception as ex:
                    wx.MessageBox(
                        f"Failed to decrypt and load profile:\n\n{str(ex)}",
                        "Profile Decryption Error",
                        wx.OK | wx.ICON_ERROR,
                        self
                    )

    def set_status_text(self, text: str):
        """Thread-safe status bar announcement."""
        def _update():
            if self.statusbar:
                self.statusbar.SetStatusText(text)
        if wx.IsMainThread():
            _update()
        else:
            wx.CallAfter(_update)

    def _check_updates_startup(self):
        """Silently queries GitHub for updates in the background on startup."""
        check_updates_background(self, silent=True, current_version=APP_VERSION)

    def on_check_updates_menu(self, event: wx.CommandEvent):
        """Manually checks for application updates from GitHub."""
        self.set_status_text("Checking GitHub for GhostWave Studio updates...")

        def _finish(info):
            self.set_status_text("Ready.")

        check_updates_background(self, silent=False, current_version=APP_VERSION, on_finish=_finish)

    def on_accessibility_guide(self, event: wx.CommandEvent):
        dlg = AccessibilityGuideDialog(self)
        dlg.ShowModal()
        dlg.Destroy()

    def on_about(self, event: wx.CommandEvent):
        dlg = AboutDialog(self)
        dlg.ShowModal()
        dlg.Destroy()


# Backward-compatible alias
SunoSanitizerFrame = GhostWaveFrame


def main():
    app = wx.App(False)
    frame = GhostWaveFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
