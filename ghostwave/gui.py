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
import tempfile
from pathlib import Path
from typing import Optional

try:
    import winsound
except ImportError:
    winsound = None

try:
    import sounddevice as sd
except Exception:
    sd = None

import numpy as np
import soundfile as sf
import wx
import wx.lib.scrolledpanel as scrolled

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostwave.audio_processor import (
    AudioProcessor,
    AudioSanitizeOptions,
    AudioProcessResult,
    benchmark_evasion_metrics,
    load_audio_samples
)
from ghostwave.batch_processor import BatchProcessor, BatchTask, FolderWatcher
from ghostwave.cloud_stem_api import CloudApiConfig, CloudStemClient
from ghostwave.config_manager import GhostWaveConfig, DEFAULT_SN_FILENAME, resolve_config_path
from ghostwave.lyrics_processor import LyricsProcessor, LyricsSanitizeResult, DEFAULT_CELEBRITY_BLACKLIST, generate_suno_style_prompt
from ghostwave.updater import (
    UpdateInfo,
    UpdateDialog,
    UpdateDownloadDialog,
    check_for_updates,
    check_updates_background,
    run_update_flow,
    get_cached_changelog_text,
    fetch_and_save_latest_changelog
)
from ghostwave.version import APP_VERSION
from ghostwave.i18n import tr, _t, get_language, set_language, get_supported_languages, init_translations
from ghostwave import support_client


class AccessibleControlName(wx.Accessible):
    """
    Ensures NVDA, JAWS, and Windows Narrator reliably announce the control label
    via native MSAA/IAccessible get_accName.
    """
    def __init__(self, win: wx.Window, name_supplier):
        super().__init__(win)
        self._supplier = name_supplier

    def GetName(self, childId: int):
        try:
            val = self._supplier() if callable(self._supplier) else str(self._supplier)
            return (wx.ACC_OK, str(val))
        except Exception:
            return (wx.ACC_NOT_IMPLEMENTED, "")


def make_accessible_control(ctrl: wx.Window, name_or_fn, tooltip_or_fn=None):
    """
    Binds accessible name and tooltip across a control and all child controls (e.g. SpinCtrl edit boxes).
    Guarantees screen readers speak the correct label on focus in the active language.
    """
    if not ctrl:
        return

    def get_name():
        try:
            return name_or_fn() if callable(name_or_fn) else str(name_or_fn)
        except Exception:
            return ""

    init_name = get_name()
    ctrl.SetName(init_name)
    try:
        acc = AccessibleControlName(ctrl, get_name)
        ctrl._acc_obj = acc
        ctrl.SetAccessible(acc)
    except Exception:
        pass

    tip = ""
    if tooltip_or_fn:
        try:
            tip = tooltip_or_fn() if callable(tooltip_or_fn) else str(tooltip_or_fn)
            ctrl.SetToolTip(tip)
        except Exception:
            pass

    # For compound controls such as wx.SpinCtrl or wx.SpinCtrlDouble (TextCtrl and SpinButton):
    for child in ctrl.GetChildren():
        child.SetName(init_name)
        try:
            child_acc = AccessibleControlName(child, get_name)
            child._acc_obj = child_acc
            child.SetAccessible(child_acc)
        except Exception:
            pass
        if tip:
            try:
                child.SetToolTip(tip)
            except Exception:
                pass


class BlacklistDialog(wx.Dialog):
    """
    Accessible Dialog to view and edit the custom celebrity and artist blacklist.
    """

    def __init__(self, parent: wx.Window, lyrics_processor: LyricsProcessor):
        super().__init__(
            parent,
            title=tr("BLACKLIST_TITLE"),
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
            label=tr("BLACKLIST_INSTR")
        )
        instr_label.SetName("Blacklist Instructions")
        main_sizer.Add(instr_label, 0, wx.ALL, 10)

        # Blacklist Text Editor
        self.text_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_DONTWRAP,
            name="Celebrity Blacklist Text Area"
        )
        make_accessible_control(self.text_ctrl, lambda: tr("BLACKLIST_TITLE"), lambda: tr("BLACKLIST_TIP_TEXT"))
        initial_text = "\n".join(sorted(self.processor.celebrity_blacklist, key=str.lower))
        self.text_ctrl.SetValue(initial_text)
        main_sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Bottom Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.reset_btn = wx.Button(panel, label=tr("BLACKLIST_BTN_RESET"))
        self.reset_btn.SetName("Reset to Defaults Button")
        self.reset_btn.SetToolTip(tr("BLACKLIST_TIP_RESET"))
        self.reset_btn.Bind(wx.EVT_BUTTON, self.on_reset)
        btn_sizer.Add(self.reset_btn, 0, wx.RIGHT, 10)

        btn_sizer.AddStretchSpacer()

        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, label=tr("CANCEL_BTN"))
        self.cancel_btn.SetName("Cancel Button")
        btn_sizer.Add(self.cancel_btn, 0, wx.RIGHT, 10)

        self.save_btn = wx.Button(panel, wx.ID_OK, label=tr("BLACKLIST_BTN_SAVE"))
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
            tr("BLACKLIST_CONFIRM_RESET_MSG"),
            tr("BLACKLIST_CONFIRM_RESET_TITLE"),
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
                tr("BLACKLIST_EMPTY_ERR_MSG"),
                tr("BLACKLIST_VALIDATION_ERR_TITLE"),
                wx.OK | wx.ICON_WARNING,
                self
            )
            return

        success = self.processor.save_blacklist(names)
        if success:
            self.EndModal(wx.ID_OK)
        else:
            wx.MessageBox(
                tr("BLACKLIST_SAVE_ERR_MSG"),
                tr("BLACKLIST_SAVE_ERR_TITLE"),
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
            title=tr("CLOUD_API_TITLE"),
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
            label=tr("CLOUD_API_INFO")
        )
        info_label.SetName("Cloud API Information Label")
        main_sizer.Add(info_label, 0, wx.ALL, 12)

        grid = wx.FlexGridSizer(cols=2, vgap=12, hgap=12)
        grid.AddGrowableCol(1, 1)

        # Provider Choice
        prov_label = wx.StaticText(panel, label=tr("CLOUD_API_PROVIDER_LBL"))
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
        token_label = wx.StaticText(panel, label=tr("CLOUD_API_TOKEN_LBL"))
        token_label.SetName("API Token Label")
        grid.Add(token_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.token_ctrl = wx.TextCtrl(
            panel,
            value=self.config.api_token,
            style=wx.TE_PASSWORD,
            name="Cloud API Token Input"
        )
        make_accessible_control(self.token_ctrl, lambda: tr("CLOUD_API_TOKEN_LBL"), lambda: tr("CLOUD_API_TIP_TOKEN"))
        grid.Add(self.token_ctrl, 0, wx.EXPAND)

        # Endpoint URL
        url_label = wx.StaticText(panel, label=tr("CLOUD_API_URL_LBL"))
        url_label.SetName("Endpoint URL Label")
        grid.Add(url_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.url_ctrl = wx.TextCtrl(
            panel,
            value=self.config.endpoint_url,
            name="Cloud Endpoint URL Input"
        )
        make_accessible_control(self.url_ctrl, lambda: tr("CLOUD_API_URL_LBL"), lambda: tr("CLOUD_API_TIP_URL"))
        grid.Add(self.url_ctrl, 0, wx.EXPAND)

        # Timeout Spin
        timeout_label = wx.StaticText(panel, label=tr("CLOUD_API_TIMEOUT_LBL"))
        timeout_label.SetName("Remote Timeout Label")
        grid.Add(timeout_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.timeout_spin = wx.SpinCtrl(
            panel,
            value=str(self.config.timeout_seconds),
            min=30,
            max=600,
            name="Remote Job Timeout in Seconds"
        )
        make_accessible_control(self.timeout_spin, lambda: tr("CLOUD_API_TIMEOUT_LBL"))
        grid.Add(self.timeout_spin, 0, wx.EXPAND)

        main_sizer.Add(grid, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        # Bottom Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, label=tr("CANCEL_BTN"), name="Cancel Cloud Settings Button")
        btn_sizer.Add(self.cancel_btn, 0, wx.RIGHT, 10)

        self.save_btn = wx.Button(panel, wx.ID_OK, label=tr("CLOUD_API_BTN_SAVE"), name="Save Cloud Settings Button")
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
            title=tr("LYRICS_CLOUD_TITLE"),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(560, 420)
        )
        self.SetName("Lyrics Cloud LLM Settings Dialog")
        self.config = GhostWaveConfig.load()

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Lyrics Cloud Settings Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        info_label = wx.StaticText(
            panel,
            label=tr("LYRICS_CLOUD_INFO")
        )
        info_label.SetName("Lyrics Cloud Info Label")
        main_sizer.Add(info_label, 0, wx.ALL, 12)

        grid = wx.FlexGridSizer(cols=2, vgap=12, hgap=12)
        grid.AddGrowableCol(1, 1)

        prov_label = wx.StaticText(panel, label=tr("LYRICS_CLOUD_PROVIDER_LBL"))
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

        tok_label = wx.StaticText(panel, label=tr("LYRICS_CLOUD_TOKEN_LBL"))
        tok_label.SetName("Lyrics API Token Label")
        grid.Add(tok_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.token_ctrl = wx.TextCtrl(
            panel,
            value=self.config.lyrics_cloud_api_token,
            style=wx.TE_PASSWORD,
            name="Lyrics API Token Input"
        )
        make_accessible_control(self.token_ctrl, lambda: tr("LYRICS_CLOUD_TOKEN_LBL"), lambda: tr("LYRICS_CLOUD_TIP_TOKEN"))
        grid.Add(self.token_ctrl, 0, wx.EXPAND)

        mod_label = wx.StaticText(panel, label=tr("LYRICS_CLOUD_MODEL_LBL"))
        mod_label.SetName("Model Identifier Label")
        grid.Add(mod_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.model_ctrl = wx.TextCtrl(
            panel,
            value=self.config.lyrics_cloud_model or "llama-3.3-70b-versatile",
            name="Model Identifier Input"
        )
        make_accessible_control(self.model_ctrl, lambda: tr("LYRICS_CLOUD_MODEL_LBL"), lambda: tr("LYRICS_CLOUD_TIP_MODEL"))
        grid.Add(self.model_ctrl, 0, wx.EXPAND)

        url_label = wx.StaticText(panel, label=tr("LYRICS_CLOUD_URL_LBL"))
        url_label.SetName("Lyrics Endpoint URL Label")
        grid.Add(url_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.url_ctrl = wx.TextCtrl(
            panel,
            value=self.config.lyrics_cloud_endpoint_url,
            name="Lyrics Endpoint URL Input"
        )
        make_accessible_control(self.url_ctrl, lambda: tr("LYRICS_CLOUD_URL_LBL"), lambda: tr("LYRICS_CLOUD_TIP_URL"))
        grid.Add(self.url_ctrl, 0, wx.EXPAND)

        main_sizer.Add(grid, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, label=tr("CANCEL_BTN"), name="Cancel Lyrics Cloud Button")
        btn_sizer.Add(self.cancel_btn, 0, wx.RIGHT, 10)

        self.save_btn = wx.Button(panel, wx.ID_OK, label=tr("LYRICS_CLOUD_BTN_SAVE"), name="Save Lyrics Cloud Button")
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
            title=tr("AUDIT_DIALOG_TITLE"),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(580, 500)
        )
        self.SetName("Acoustic Evasion Safety Audit Dialog")

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Acoustic Audit Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        title_lbl = wx.StaticText(panel, label=tr("AUDIT_HEADER_LBL"))
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
        make_accessible_control(
            report_ctrl,
            lambda: tr("AUDIT_REPORT_NAME"),
            lambda: tr("AUDIT_REPORT_NAME")
        )

        sim = metrics.get("similarity_pct", 0.0)
        verdict = metrics.get("verdict", "Unknown")
        ber = metrics.get("bit_error_rate_pct", 0.0)
        drop_db = metrics.get("vocal_energy_drop_db", 0.0)
        frames = metrics.get("aligned_frames", 0)

        if get_language() == "id":
            lines = [
                f"Berkas Asli: {os.path.basename(orig_file)}",
                f"Berkas Hasil Sanitasi: {os.path.basename(sani_file)}",
                "--------------------------------------------------",
                f"Tingkat Keamanan Evasion: {verdict.upper()}",
                "--------------------------------------------------",
                f"• Kemiripan Sidik Jari Akustik (Chromaprint): {sim:.1f}%",
                f"• Tingkat Bit Error / Hash Decorrelation: {ber:.1f}%",
                f"• Frame Sub-fingerprint Terpadu: {frames} frame",
                f"• Pelemahan Pita Intelligibility Vokal: {drop_db:.1f} dB",
                "",
                "PENILAIAN TEKNIS RINCI:",
                f"{metrics.get('details', '')}",
                "",
                "REKOMENDASI UNGGAH SUNO:",
                "1. Jika kemiripan di bawah 35%, hash penanda telah terurai dan akan lolos Audible Magic.",
                "2. Jika audio berisi vokal, pastikan peredaman vokal atau isolasi stem Cloud telah diterapkan.",
                "3. Berkas ekspor telah dibersihkan dari seluruh tag ID3, chunk RIFF, dan metadata."
            ]
        else:
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
        self.report_ctrl = report_ctrl
        main_sizer.Add(report_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.copy_btn = wx.Button(panel, label=tr("AUDIO_BTN_COPY_AUDIT"), name="Copy Audit Report Button")
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        btn_sizer.Add(self.copy_btn, 0, wx.RIGHT, 8)
        btn_sizer.AddStretchSpacer()

        ok_btn = wx.Button(panel, wx.ID_OK, label=tr("AUDIT_BTN_CLOSE"), name="Close Audit Button")
        ok_btn.SetDefault()
        btn_sizer.Add(ok_btn, 0)
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(main_sizer)
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dlg_sizer)
        self.CentreOnParent()
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_hook)

    def on_copy(self, event: wx.CommandEvent):
        if wx.TheClipboard.Open():
            try:
                wx.TheClipboard.SetData(wx.TextDataObject(self.report_ctrl.GetValue()))
                wx.Bell()
            finally:
                wx.TheClipboard.Close()

    def on_key_hook(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()


class AudioVisualizerPanel(wx.Panel):
    """
    Accessible Dual-Waveform & STFT Decoy Peak Visualizer.
    Renders original audio vs sanitized audio and real envelope-delta peaks
    with a live screen-reader metric summary.
    """

    def __init__(self, parent: wx.Window, size=(-1, 105)):
        super().__init__(parent, size=size, style=wx.FULL_REPAINT_ON_RESIZE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.orig_wave: Optional[np.ndarray] = None
        self.sani_wave: Optional[np.ndarray] = None
        self.decoy_peaks: list[tuple[float, float]] = []
        self.orig_peak: float = 0.0
        self.orig_rms_db: float = -96.0
        self.sani_rms_db: float = -96.0
        self.SetName(self.get_accessible_summary())

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)

    def get_accessible_summary(self) -> str:
        """Returns a localized text summary of the loaded waveforms for screen readers."""
        if self.orig_wave is None and self.sani_wave is None:
            return tr("AUDIO_VIS_SUMMARY_EMPTY")
        if self.sani_wave is None:
            return tr("AUDIO_VIS_SUMMARY_ORIG", peak=self.orig_peak, rms_db=self.orig_rms_db)
        return tr(
            "AUDIO_VIS_SUMMARY_BOTH",
            orig_db=self.orig_rms_db,
            sani_db=self.sani_rms_db,
            peaks=len(self.decoy_peaks),
        )

    def _downsample_envelope(self, mono: np.ndarray, pts: int = 240) -> np.ndarray:
        if len(mono) >= pts:
            step = len(mono) // pts
            env = np.abs(mono[:step * pts].reshape(pts, step)).max(axis=1)
        elif len(mono) > 0:
            x_old = np.linspace(0.0, 1.0, len(mono))
            x_new = np.linspace(0.0, 1.0, pts)
            env = np.interp(x_new, x_old, np.abs(mono)).astype(np.float32)
        else:
            env = np.zeros(pts, dtype=np.float32)
        mx = float(np.max(env))
        if mx > 1e-6:
            env = env / mx
        return env.astype(np.float32)

    def on_size(self, event: wx.SizeEvent):
        self.Refresh()
        event.Skip()

    def set_original_audio(self, audio_input):
        try:
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                data, _ = load_audio_samples(audio_input, target_sr=44100)
            elif isinstance(audio_input, np.ndarray):
                data = audio_input
            else:
                return
            mono = np.mean(data, axis=1) if data.ndim > 1 else data
            self.orig_peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
            rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
            self.orig_rms_db = max(-96.0, float(20.0 * np.log10(rms + 1e-9)))
            self.orig_wave = self._downsample_envelope(mono, pts=240)
            self.sani_wave = None
            self.decoy_peaks = []
            self.SetName(self.get_accessible_summary())
            self.Refresh()
        except Exception:
            pass

    def set_sanitized_audio(self, audio_input):
        try:
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                data, _ = load_audio_samples(audio_input, target_sr=44100)
            elif isinstance(audio_input, np.ndarray):
                data = audio_input
            else:
                return
            mono = np.mean(data, axis=1) if data.ndim > 1 else data
            rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
            self.sani_rms_db = max(-96.0, float(20.0 * np.log10(rms + 1e-9)))
            self.sani_wave = self._downsample_envelope(mono, pts=240)

            # Derive decoy peak markers from real envelope delta between sanitized and original audio
            pts = len(self.sani_wave)
            if self.orig_wave is not None and len(self.orig_wave) == pts:
                delta_env = np.abs(self.sani_wave - self.orig_wave)
            else:
                delta_env = self.sani_wave
            order = np.argsort(delta_env)[::-1]
            selected: list[int] = []
            for idx in order:
                idx_i = int(idx)
                if 4 <= idx_i <= pts - 5 and all(abs(idx_i - s) >= 8 for s in selected):
                    selected.append(idx_i)
                    if len(selected) >= 14:
                        break
            self.decoy_peaks = [
                (
                    float(idx_i) / float(max(1, pts - 1)),
                    0.5 - (0.38 * float(self.sani_wave[idx_i])) if (k % 2 == 0) else 0.5 + (0.38 * float(self.sani_wave[idx_i]))
                )
                for k, idx_i in enumerate(sorted(selected))
            ]
            self.SetName(self.get_accessible_summary())
            self.Refresh()
        except Exception:
            pass

    def clear(self):
        self.orig_wave = None
        self.sani_wave = None
        self.decoy_peaks = []
        self.orig_peak = 0.0
        self.orig_rms_db = -96.0
        self.sani_rms_db = -96.0
        self.SetName(self.get_accessible_summary())
        self.Refresh()

    def on_paint(self, event: wx.PaintEvent):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetClientSize()
        if w < 10 or h < 10:
            return

        dc.SetBackground(wx.Brush(wx.Colour(20, 24, 30)))
        dc.Clear()

        # Grid lines
        dc.SetPen(wx.Pen(wx.Colour(34, 40, 50), 1, wx.PENSTYLE_SOLID))
        mid_y = h // 2
        dc.DrawLine(0, mid_y, w, mid_y)
        for i in range(1, 5):
            x = int(w * i / 5)
            dc.DrawLine(x, 0, x, h)

        if self.orig_wave is None and self.sani_wave is None:
            dc.SetTextForeground(wx.Colour(195, 205, 220))
            font = dc.GetFont()
            font.SetPointSize(9)
            dc.SetFont(font)
            msg = tr("VISUALIZER_PROMPT_EMPTY")
            tw, th = dc.GetTextExtent(msg)
            dc.DrawText(msg, max(10, (w - tw) // 2), max(5, (h - th) // 2))
            return

        # Original waveform in accessible blue
        if self.orig_wave is not None and len(self.orig_wave) > 1:
            dc.SetPen(wx.Pen(wx.Colour(96, 165, 250), 1))
            n = len(self.orig_wave)
            pts = []
            for i, val in enumerate(self.orig_wave):
                x = int((i / (n - 1)) * w)
                amp = int(val * (h * 0.40))
                pts.append((x, mid_y - amp))
            for i in range(len(pts) - 1):
                dc.DrawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                dc.DrawLine(pts[i][0], mid_y + (mid_y - pts[i][1]), pts[i + 1][0], mid_y + (mid_y - pts[i + 1][1]))

        # Sanitized waveform in accessible emerald green
        if self.sani_wave is not None and len(self.sani_wave) > 1:
            dc.SetPen(wx.Pen(wx.Colour(52, 211, 153), 2))
            n = len(self.sani_wave)
            pts = []
            for i, val in enumerate(self.sani_wave):
                x = int((i / (n - 1)) * w)
                amp = int(val * (h * 0.40))
                pts.append((x, mid_y - amp))
            for i in range(len(pts) - 1):
                dc.DrawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                dc.DrawLine(pts[i][0], mid_y + (mid_y - pts[i][1]), pts[i + 1][0], mid_y + (mid_y - pts[i + 1][1]))

        # Real delta peaks (amber dots)
        if self.decoy_peaks:
            dc.SetBrush(wx.Brush(wx.Colour(251, 191, 36)))
            dc.SetPen(wx.Pen(wx.Colour(255, 215, 0), 1))
            for x_norm, y_norm in self.decoy_peaks:
                px = int(x_norm * w)
                py = int(y_norm * h)
                dc.DrawCircle(px, py, 3)

        # Legend
        dc.SetTextForeground(wx.Colour(220, 228, 240))
        font = dc.GetFont()
        font.SetPointSize(8)
        dc.SetFont(font)
        legend = tr("VISUALIZER_LEGEND")
        dc.DrawText(legend, 10, 6)


class PreFlightEvasionDialog(wx.Dialog):
    """
    Accessible Pre-Flight Evasion Simulator Dialog.
    Audits CQT Chroma similarity, landmark constellation overlap,
    SynthID / AudioSeal watermark neutralization, and overall safety rating.
    """

    def __init__(self, parent: wx.Window, orig_file: str, sani_file: str, processor: AudioProcessor):
        super().__init__(
            parent,
            title=tr("PREFLIGHT_DIALOG_TITLE"),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(600, 520)
        )
        self.SetName("Pre-Flight Evasion Simulator Dialog")
        self.orig_file = orig_file
        self.sani_file = sani_file

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        if isinstance(processor, dict):
            self.processor = None
            self.report = processor
        else:
            if processor is None:
                self.processor = AudioProcessor()
            else:
                self.processor = processor
            self.report = self.processor.simulate_preflight(orig_file, sani_file)
        score = self.report.get("safety_score", 85)
        is_id = (get_language() == "id")
        verdict = self.report.get("verdict" if is_id else "verdict_en", self.report.get("verdict", "Lolos Aman" if is_id else "High Evasion"))

        # WCAG AA/AAA verified background fills paired with #FFFFFF text
        badge_pnl = wx.Panel(panel)
        badge_sizer = wx.BoxSizer(wx.VERTICAL)
        badge_color = wx.Colour(4, 108, 78) if score >= 75 else (wx.Colour(146, 64, 14) if score >= 50 else wx.Colour(185, 28, 28))
        badge_pnl.SetBackgroundColour(badge_color)

        score_text = (
            f"  SKOR KEAMANAN LOLOS AI: {score}% ({verdict.upper()})  "
            if is_id
            else f"  AI EVASION SAFETY SCORE: {score}% ({verdict.upper()})  "
        )
        badge_label = wx.StaticText(badge_pnl, label=score_text)
        badge_label.SetForegroundColour(wx.Colour(255, 255, 255))
        font = badge_label.GetFont()
        font.SetPointSize(11)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        badge_label.SetFont(font)
        badge_sizer.Add(badge_label, 0, wx.ALL | wx.ALIGN_CENTER, 8)
        badge_pnl.SetSizer(badge_sizer)
        main_sizer.Add(badge_pnl, 0, wx.EXPAND | wx.ALL, 10)

        self.report_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="Pre-Flight Evasion Report Text"
        )
        make_accessible_control(self.report_ctrl, lambda: tr("PREFLIGHT_DIALOG_TITLE"))
        cqt_sim = self.report.get("cqt_chroma_similarity_pct", 0.0)
        lm_sim = self.report.get("landmark_overlap_pct", 0.0)
        wm_status = self.report.get("synthid_watermark_neutralized", False)
        voc_drop = self.report.get("vocal_isolation_db", 0.0)
        advice = self.report.get("actionable_advice" if is_id else "actionable_advice_en", self.report.get("actionable_advice", ""))

        if score >= 75:
            user_note = (
                "Komponen sidik jari rasi frekuensi dan harmoni sudah teracak baik sehingga siap diuji unggah."
                if is_id
                else "Constellation peaks and harmonic profiles are well decoupled for upload testing."
            )
        elif score >= 50:
            user_note = (
                "Kemiripan menurun sebagian. Disarankan memotong audio menjadi klip 22 detik lewat Audio Slicer."
                if is_id
                else "Similarity is partially reduced. Slicing into 22-second clips via Audio Slicer is recommended."
            )
        else:
            user_note = (
                "Skor keamanan masih rendah. Naikkan pergeseran nada atau aktifkan mode Instrumental sebelum mengunggah."
                if is_id
                else "Safety score is still low. Increase pitch shift or enable stronger vocal reduction before uploading."
            )

        lines = [
            f"Berkas Asli: {os.path.basename(orig_file)}" if is_id else f"Original File: {os.path.basename(orig_file)}",
            f"Berkas Hasil: {os.path.basename(sani_file)}" if is_id else f"Sanitized File: {os.path.basename(sani_file)}",
            "=" * 50,
            "HASIL AUDIT MULTI-VEKTOR SISTEM DETEKSI:" if is_id else "MULTI-VECTOR DETECTION SYSTEM AUDIT:",
            f"• Kemiripan Melodi CQT Chroma: {cqt_sim:.1f}% (Batas Aman: < 35%)" if is_id else f"• CQT Chroma Melody Similarity: {cqt_sim:.1f}% (Target: < 35%)",
            f"• Tumpang Tindih Landmark Akustik: {lm_sim:.1f}% (Batas Aman: < 15%)" if is_id else f"• Acoustic Landmark Overlap: {lm_sim:.1f}% (Target: < 15%)",
            f"• Watermark AI (SynthID / AudioSeal): {'Dinetralkan & Terpencar' if wm_status else 'Belum Terdeteksi Perubahan'}" if is_id else f"• AI Watermark (SynthID / AudioSeal): {'Neutralized & Diffused' if wm_status else 'No Phase Dispersion Detected'}",
            f"• Pelemahan Karakter Vokal Tengah: {voc_drop:.1f} dB" if is_id else f"• Center Vocal Attenuation: {voc_drop:.1f} dB",
            "",
            "KESIMPULAN & SARAN TINDAKAN:" if is_id else "VERDICT & ACTIONABLE ADVICE:",
            f"{advice}",
            "",
            "CATATAN PENGGUNA:" if is_id else "USER NOTE:",
            user_note,
        ]
        self.report_ctrl.SetValue("\n".join(lines))
        main_sizer.Add(self.report_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.copy_btn = wx.Button(panel, label=tr("PREFLIGHT_BTN_COPY"), name="Copy Pre-Flight Report Button")
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        btn_sizer.Add(self.copy_btn, 0, wx.RIGHT, 8)
        btn_sizer.AddStretchSpacer()

        ok_btn = wx.Button(panel, wx.ID_OK, label=tr("AUDIT_BTN_CLOSE"), name="Close Preflight Dialog Button")
        ok_btn.SetDefault()
        btn_sizer.Add(ok_btn, 0)
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(main_sizer)
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dlg_sizer)
        self.CentreOnParent()
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_hook)

    def on_copy(self, event: wx.CommandEvent):
        if wx.TheClipboard.Open():
            try:
                wx.TheClipboard.SetData(wx.TextDataObject(self.report_ctrl.GetValue()))
                wx.Bell()
            finally:
                wx.TheClipboard.Close()

    def on_key_hook(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()


class BatchProcessingDialog(wx.Dialog):
    """
    Accessible Batch Processing and Folder Watcher Automator Dialog.
    Processes audio queues and monitors incoming folders for hands-free workflow.
    """

    def __init__(self, parent: wx.Window, audio_processor: AudioProcessor, get_options_fn):
        super().__init__(
            parent,
            title=tr("BATCH_DIALOG_TITLE"),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(700, 560)
        )
        self.SetName("Batch Processing and Folder Watcher Dialog")
        self.processor = audio_processor
        self.get_options_fn = get_options_fn
        self.batch_processor = BatchProcessor(audio_processor)
        self.folder_watcher: Optional[FolderWatcher] = None
        self.tasks: list[BatchTask] = []
        self.is_processing = False

        self._build_ui()
        self.Bind(wx.EVT_CLOSE, self.on_close_dialog)
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_hook)

    def _cleanup_workers(self):
        if self.is_processing:
            self.batch_processor.cancel()
        if self.folder_watcher and self.folder_watcher.is_running:
            self.folder_watcher.stop()
            self.folder_watcher = None

    def on_close_dialog(self, event: wx.CloseEvent):
        self._cleanup_workers()
        event.Skip()

    def on_key_hook(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self._cleanup_workers()
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()

    def _build_ui(self):
        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        notebook = wx.Notebook(panel, style=wx.NB_TOP | wx.TAB_TRAVERSAL)

        # ---------------- Tab 1: Queue Manager ----------------
        queue_pnl = wx.Panel(notebook)
        q_sizer = wx.BoxSizer(wx.VERTICAL)

        tb_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.add_files_btn = wx.Button(queue_pnl, label=tr("BATCH_BTN_ADD_FILES"), name="Add Files to Batch")
        self.add_files_btn.Bind(wx.EVT_BUTTON, self.on_add_files)
        tb_sizer.Add(self.add_files_btn, 0, wx.RIGHT, 6)

        self.add_folder_btn = wx.Button(queue_pnl, label=tr("BATCH_BTN_ADD_FOLDER"), name="Add Folder to Batch")
        self.add_folder_btn.Bind(wx.EVT_BUTTON, self.on_add_folder)
        tb_sizer.Add(self.add_folder_btn, 0, wx.RIGHT, 6)

        self.remove_btn = wx.Button(queue_pnl, label=tr("BATCH_BTN_REMOVE"), name="Remove Selected Item")
        self.remove_btn.Bind(wx.EVT_BUTTON, self.on_remove_item)
        tb_sizer.Add(self.remove_btn, 0, wx.RIGHT, 6)

        self.clear_btn = wx.Button(queue_pnl, label=tr("BATCH_BTN_CLEAR"), name="Clear All Items")
        self.clear_btn.Bind(wx.EVT_BUTTON, self.on_clear_items)
        tb_sizer.Add(self.clear_btn, 0)
        q_sizer.Add(tb_sizer, 0, wx.EXPAND | wx.ALL, 8)

        self.list_ctrl = wx.ListCtrl(queue_pnl, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.list_ctrl.SetName("Batch Processing File Queue List")
        self.list_ctrl.InsertColumn(0, tr("BATCH_COL_NO"), width=45)
        self.list_ctrl.InsertColumn(1, tr("BATCH_COL_FILENAME"), width=340)
        self.list_ctrl.InsertColumn(2, tr("BATCH_COL_STATUS"), width=120)
        self.list_ctrl.InsertColumn(3, tr("BATCH_COL_PROGRESS"), width=90)
        q_sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        opt_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.auto_slice_chk = wx.CheckBox(queue_pnl, label=tr("BATCH_CHK_AUTO_SLICE"))
        self.auto_slice_chk.SetValue(False)
        opt_sizer.Add(self.auto_slice_chk, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        self.slice_sec_spin = wx.SpinCtrlDouble(queue_pnl, value="24.0", min=5.0, max=60.0, inc=1.0)
        self.slice_sec_spin.SetDigits(1)
        make_accessible_control(self.slice_sec_spin, lambda: tr("SLICER_CHUNK_LBL"), lambda: tr("SLICER_TIP_CHUNK"))
        opt_sizer.Add(self.slice_sec_spin, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        opt_sizer.Add(wx.StaticText(queue_pnl, label=tr("AUDIO_UNIT_SECONDS")), 0, wx.ALIGN_CENTER_VERTICAL)
        q_sizer.Add(opt_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        act_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(queue_pnl, label=tr("BATCH_BTN_START"), name="Start Batch Processing")
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start_batch)
        act_sizer.Add(self.start_btn, 0, wx.RIGHT, 8)

        self.cancel_btn = wx.Button(queue_pnl, label=tr("BATCH_BTN_CANCEL"), name="Cancel Batch Processing")
        self.cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel_batch)
        self.cancel_btn.Enable(False)
        act_sizer.Add(self.cancel_btn, 0, wx.RIGHT, 10)

        self.gauge = wx.Gauge(queue_pnl, range=100)
        self.gauge.SetName("Batch Queue Overall Progress Gauge")
        act_sizer.Add(self.gauge, 1, wx.ALIGN_CENTER_VERTICAL)
        q_sizer.Add(act_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        queue_pnl.SetSizer(q_sizer)
        notebook.AddPage(queue_pnl, tr("BATCH_QUEUE_TAB"), select=True)

        # ---------------- Tab 2: Folder Watcher ----------------
        watch_pnl = wx.Panel(notebook)
        w_sizer = wx.BoxSizer(wx.VERTICAL)

        info_lbl = wx.StaticText(
            watch_pnl,
            label=tr("BATCH_WATCHER_INFO")
        )
        w_sizer.Add(info_lbl, 0, wx.ALL, 10)

        in_sizer = wx.BoxSizer(wx.HORIZONTAL)
        in_sizer.Add(wx.StaticText(watch_pnl, label=tr("BATCH_WATCH_IN_LABEL")), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.watch_in_ctrl = wx.TextCtrl(watch_pnl, style=wx.TE_READONLY)
        make_accessible_control(self.watch_in_ctrl, lambda: tr("BATCH_WATCH_IN_LABEL"))
        in_sizer.Add(self.watch_in_ctrl, 1, wx.EXPAND | wx.RIGHT, 8)
        self.pick_in_btn = wx.Button(watch_pnl, label=tr("BATCH_BTN_BROWSE"), name="Browse Watch Input Directory")
        self.pick_in_btn.Bind(wx.EVT_BUTTON, self.on_pick_watch_in)
        in_sizer.Add(self.pick_in_btn, 0)
        w_sizer.Add(in_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        out_sizer = wx.BoxSizer(wx.HORIZONTAL)
        out_sizer.Add(wx.StaticText(watch_pnl, label=tr("BATCH_WATCH_OUT_LABEL")), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.watch_out_ctrl = wx.TextCtrl(watch_pnl, style=wx.TE_READONLY)
        make_accessible_control(self.watch_out_ctrl, lambda: tr("BATCH_WATCH_OUT_LABEL"))
        out_sizer.Add(self.watch_out_ctrl, 1, wx.EXPAND | wx.RIGHT, 8)
        self.pick_out_btn = wx.Button(watch_pnl, label=tr("BATCH_BTN_BROWSE"), name="Browse Watch Output Directory")
        self.pick_out_btn.Bind(wx.EVT_BUTTON, self.on_pick_watch_out)
        out_sizer.Add(self.pick_out_btn, 0)
        w_sizer.Add(out_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        w_act_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.toggle_watch_btn = wx.Button(watch_pnl, label=tr("BATCH_BTN_START_WATCH"), name="Toggle Folder Watcher Button")
        self.toggle_watch_btn.Bind(wx.EVT_BUTTON, self.on_toggle_watcher)
        w_act_sizer.Add(self.toggle_watch_btn, 0, wx.RIGHT, 10)

        self.watcher_status_lbl = wx.StaticText(watch_pnl, label=tr("BATCH_WATCHER_STATUS_IDLE"))
        w_act_sizer.Add(self.watcher_status_lbl, 0, wx.ALIGN_CENTER_VERTICAL)
        w_sizer.Add(w_act_sizer, 0, wx.EXPAND | wx.ALL, 10)

        watch_pnl.SetSizer(w_sizer)
        notebook.AddPage(watch_pnl, tr("BATCH_WATCHER_TAB"), select=False)

        main_sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 8)

        close_btn = wx.Button(panel, wx.ID_CANCEL, label=tr("AUDIT_BTN_CLOSE"), name="Close Batch Dialog Button")
        main_sizer.Add(close_btn, 0, wx.ALIGN_RIGHT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(main_sizer)
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dlg_sizer)
        self.CentreOnParent()

    def on_add_files(self, event: wx.CommandEvent):
        wildcard = "Audio Files (*.wav;*.mp3;*.flac;*.m4a;*.ogg)|*.wav;*.mp3;*.flac;*.m4a;*.ogg|All Files (*.*)|*.*"
        with wx.FileDialog(self, tr("AUDIO_DIALOG_TITLE"), wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                paths = dlg.GetPaths()
                for p in paths:
                    self._add_task_path(p)

    def on_add_folder(self, event: wx.CommandEvent):
        with wx.DirDialog(self, tr("BATCH_PICK_FOLDER_TITLE"), style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                d = dlg.GetPath()
                exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
                for entry in sorted(os.scandir(d), key=lambda e: e.name.lower()):
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in exts:
                        self._add_task_path(entry.path)

    def _add_task_path(self, in_path: str):
        stem, _ = os.path.splitext(in_path)
        options = self.get_options_fn()
        out_ext = ".mp3" if getattr(options, "output_bitrate", "wav") in ("320k", "192k") else ".wav"
        out_path = f"{stem}_ghostwave_ready{out_ext}"
        task = BatchTask(in_path, out_path)
        self.tasks.append(task)
        idx = self.list_ctrl.GetItemCount()
        self.list_ctrl.InsertItem(idx, str(idx + 1))
        self.list_ctrl.SetItem(idx, 1, os.path.basename(in_path))
        self.list_ctrl.SetItem(idx, 2, tr("STATUS_BATCH_WAITING"))
        self.list_ctrl.SetItem(idx, 3, "0%")

    def on_remove_item(self, event: wx.CommandEvent):
        sel = self.list_ctrl.GetFirstSelected()
        if sel != -1 and sel < len(self.tasks):
            self.tasks.pop(sel)
            self.list_ctrl.DeleteItem(sel)
            for i in range(self.list_ctrl.GetItemCount()):
                self.list_ctrl.SetItem(i, 0, str(i + 1))

    def on_clear_items(self, event: wx.CommandEvent):
        if not self.is_processing:
            self.tasks.clear()
            self.list_ctrl.DeleteAllItems()

    def on_start_batch(self, event: wx.CommandEvent):
        if not self.tasks or self.is_processing:
            return
        self.is_processing = True
        self.start_btn.Enable(False)
        self.cancel_btn.Enable(True)
        self.add_files_btn.Enable(False)
        self.add_folder_btn.Enable(False)
        self.remove_btn.Enable(False)
        self.clear_btn.Enable(False)

        options = self.get_options_fn()
        out_ext = ".mp3" if getattr(options, "output_bitrate", "wav") in ("320k", "192k") else ".wav"
        for task in self.tasks:
            stem, _ = os.path.splitext(task.input_path)
            task.output_path = f"{stem}_ghostwave_ready{out_ext}"

        def worker():
            def cb(idx, total, pct, msg):
                if wx.GetApp():
                    wx.CallAfter(self._update_task_ui, idx, total, pct, msg)

            self.batch_processor.process_queue(
                self.tasks,
                options,
                progress_callback=cb,
                auto_slice=self.auto_slice_chk.IsChecked(),
                slice_duration_sec=float(self.slice_sec_spin.GetValue())
            )
            if wx.GetApp():
                wx.CallAfter(self._batch_finished)

        threading.Thread(target=worker, daemon=True).start()

    def _update_task_ui(self, idx: int, total: int, pct: int, msg: str):
        try:
            if not self or self.IsBeingDeleted():
                return
            if 0 <= idx < self.list_ctrl.GetItemCount():
                self.list_ctrl.SetItem(idx, 2, msg)
                self.list_ctrl.SetItem(idx, 3, f"{pct}%")
            overall = int(((idx + (pct / 100.0)) / max(1, total)) * 100)
            self.gauge.SetValue(min(100, overall))
        except RuntimeError:
            pass

    def _batch_finished(self):
        try:
            if not self or self.IsBeingDeleted():
                return
            self.is_processing = False
            self.start_btn.Enable(True)
            self.cancel_btn.Enable(False)
            self.add_files_btn.Enable(True)
            self.add_folder_btn.Enable(True)
            self.remove_btn.Enable(True)
            self.clear_btn.Enable(True)
            self.gauge.SetValue(100)
            wx.Bell()
            wx.MessageBox(tr("BATCH_MSG_FINISHED"), tr("BATCH_TITLE_FINISHED"), wx.OK | wx.ICON_INFORMATION, self)
        except RuntimeError:
            pass

    def on_cancel_batch(self, event: wx.CommandEvent):
        if self.is_processing:
            self.batch_processor.cancel()
            self.cancel_btn.Enable(False)

    def on_pick_watch_in(self, event: wx.CommandEvent):
        with wx.DirDialog(self, tr("BATCH_WATCH_IN_LABEL"), style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.watch_in_ctrl.SetValue(dlg.GetPath())

    def on_pick_watch_out(self, event: wx.CommandEvent):
        with wx.DirDialog(self, tr("BATCH_WATCH_OUT_LABEL"), style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.watch_out_ctrl.SetValue(dlg.GetPath())

    def on_toggle_watcher(self, event: wx.CommandEvent):
        if self.folder_watcher and self.folder_watcher.is_running:
            self.folder_watcher.stop()
            self.folder_watcher = None
            self.toggle_watch_btn.SetLabel(tr("BATCH_BTN_START_WATCH"))
            self.watcher_status_lbl.SetLabel(tr("BATCH_WATCHER_STATUS_IDLE"))
            return

        in_d = self.watch_in_ctrl.GetValue().strip()
        out_d = self.watch_out_ctrl.GetValue().strip()
        if not in_d or not os.path.exists(in_d) or not out_d:
            wx.MessageBox(tr("BATCH_MSG_PICK_DIRS"), tr("AUDIT_WARN_TITLE"), wx.OK | wx.ICON_WARNING, self)
            return

        options = self.get_options_fn()
        self.folder_watcher = FolderWatcher(
            input_dir=in_d,
            output_dir=out_d,
            audio_processor=self.processor,
            options=options,
            poll_interval_sec=2.0
        )
        self.folder_watcher.start()
        self.toggle_watch_btn.SetLabel(tr("BATCH_BTN_STOP_WATCH"))
        self.watcher_status_lbl.SetLabel(tr("BATCH_WATCHER_STATUS_RUNNING", in_dir=os.path.basename(in_d)))


class SunoPromptDialog(wx.Dialog):
    """
    Accessible Dialog to generate optimized Suno v4-v6 & Udio meta-prompts
    and structural section tags.
    """

    def __init__(self, parent: wx.Window, target_text_ctrl: wx.TextCtrl, detected_bpm=None, detected_key=None):
        super().__init__(
            parent,
            title=tr("SUNO_PROMPT_DIALOG_TITLE"),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(580, 520)
        )
        self.SetName("Suno Style Prompt Generator Dialog")
        self.target_text_ctrl = target_text_ctrl

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        grid = wx.FlexGridSizer(cols=2, vgap=8, hgap=12)
        grid.AddGrowableCol(1, 1)

        grid.Add(wx.StaticText(panel, label=tr("PROMPT_GENRE_LABEL")), 0, wx.ALIGN_CENTER_VERTICAL)
        self.genre_choice = wx.Choice(
            panel,
            choices=[
                "Pop", "Electronic / EDM", "Rock / Alternative", "R&B / Soul",
                "Hip Hop / Trap", "Lo-Fi Beats", "Acoustic / Folk", "Synthwave 80s",
                "Cinematic Orchestral", "Indie Rock"
            ],
            name="Genre Choice"
        )
        self.genre_choice.SetSelection(0)
        self.genre_choice.Bind(wx.EVT_CHOICE, self.on_update_preview)
        grid.Add(self.genre_choice, 1, wx.EXPAND)

        grid.Add(wx.StaticText(panel, label=tr("PROMPT_MOOD_LABEL")), 0, wx.ALIGN_CENTER_VERTICAL)
        self.mood_choice = wx.Choice(
            panel,
            choices=[
                "Energetic & Uplifting", "Melancholic & Emotional", "Chill & Relaxed",
                "Dark & Cinematic", "Euphoric & Festival", "Dreamy & Nostalgic"
            ],
            name="Mood Choice"
        )
        self.mood_choice.SetSelection(0)
        self.mood_choice.Bind(wx.EVT_CHOICE, self.on_update_preview)
        grid.Add(self.mood_choice, 1, wx.EXPAND)

        grid.Add(wx.StaticText(panel, label=tr("PROMPT_TEMPO_LABEL")), 0, wx.ALIGN_CENTER_VERTICAL)
        bpm_val = str(int(round(detected_bpm))) if detected_bpm and detected_bpm > 0 else "124"
        self.bpm_spin = wx.SpinCtrl(panel, value=bpm_val, min=40, max=240)
        make_accessible_control(self.bpm_spin, lambda: tr("PROMPT_TEMPO_LABEL"))
        self.bpm_spin.Bind(wx.EVT_SPINCTRL, self.on_update_preview)
        grid.Add(self.bpm_spin, 0)

        grid.Add(wx.StaticText(panel, label=tr("PROMPT_KEY_LABEL")), 0, wx.ALIGN_CENTER_VERTICAL)
        self.key_ctrl = wx.TextCtrl(panel, value=detected_key or "C Major")
        make_accessible_control(self.key_ctrl, lambda: tr("PROMPT_KEY_LABEL"))
        self.key_ctrl.Bind(wx.EVT_TEXT, self.on_update_preview)
        grid.Add(self.key_ctrl, 1, wx.EXPAND)

        main_sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 10)

        prompt_box = wx.StaticBox(panel, label=tr("PROMPT_RESULT_GROUP"))
        p_sizer = wx.StaticBoxSizer(prompt_box, wx.VERTICAL)
        self.prompt_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 65))
        make_accessible_control(self.prompt_ctrl, lambda: tr("PROMPT_RESULT_GROUP"))
        p_sizer.Add(self.prompt_ctrl, 1, wx.EXPAND | wx.ALL, 6)
        main_sizer.Add(p_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        tags_box = wx.StaticBox(panel, label=tr("PROMPT_TAGS_GROUP"))
        t_sizer = wx.StaticBoxSizer(tags_box, wx.VERTICAL)
        self.tags_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE, size=(-1, 110))
        make_accessible_control(self.tags_ctrl, lambda: tr("PROMPT_TAGS_GROUP"))
        t_sizer.Add(self.tags_ctrl, 1, wx.EXPAND | wx.ALL, 6)
        main_sizer.Add(t_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.insert_btn = wx.Button(panel, label=tr("PROMPT_BTN_INSERT"), name="Insert to Lyrics Button")
        self.insert_btn.Bind(wx.EVT_BUTTON, self.on_insert)
        self.insert_btn.SetDefault()
        btn_sizer.Add(self.insert_btn, 0, wx.RIGHT, 8)

        self.copy_btn = wx.Button(panel, label=tr("PROMPT_BTN_COPY"), name="Copy Prompt Button")
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        btn_sizer.Add(self.copy_btn, 0, wx.RIGHT, 8)

        btn_sizer.AddStretchSpacer()
        close_btn = wx.Button(panel, wx.ID_CANCEL, label=tr("PROMPT_BTN_CLOSE"), name="Close Prompt Generator Dialog")
        btn_sizer.Add(close_btn, 0)
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(main_sizer)
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dlg_sizer)
        self.CentreOnParent()
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_hook)

        self.on_update_preview(None)

    def on_key_hook(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()

    def on_update_preview(self, event):
        genre = self.genre_choice.GetStringSelection()
        mood = self.mood_choice.GetStringSelection().split("&")[0].strip()
        bpm = self.bpm_spin.GetValue()
        key = self.key_ctrl.GetValue().strip()

        p_text = generate_suno_style_prompt(bpm=bpm, key=key, mood=mood, user_genre=genre)
        parts = p_text.split("\n\n", 1)
        style = parts[0] if parts else p_text
        struct = parts[1] if len(parts) > 1 else ""
        self.prompt_ctrl.SetValue(style)
        self.tags_ctrl.SetValue(struct)

    def on_insert(self, event):
        style = self.prompt_ctrl.GetValue()
        struct = self.tags_ctrl.GetValue()
        header = f"// Style Prompt: {style}\n\n"
        existing = self.target_text_ctrl.GetValue()
        if existing.strip():
            self.target_text_ctrl.SetValue(header + existing)
        else:
            self.target_text_ctrl.SetValue(header + struct + "\n")
        self.EndModal(wx.ID_OK)

    def on_copy(self, event):
        if wx.TheClipboard.Open():
            try:
                wx.TheClipboard.SetData(wx.TextDataObject(self.prompt_ctrl.GetValue()))
                wx.Bell()
            finally:
                wx.TheClipboard.Close()
            wx.MessageBox(tr("PROMPT_COPY_SUCCESS"), tr("LYRICS_COPIED_TITLE"), wx.OK | wx.ICON_INFORMATION, self)


class SunoCheatSheetDialog(wx.Dialog):
    """
    Accessible Dialog providing the operational protocol and cheat sheet
    for successfully uploading audio to Suno AI without triggering blocks.
    """

    def __init__(self, parent: wx.Window):
        super().__init__(
            parent,
            title=tr("SUNO_GUIDE_TITLE"),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(640, 560)
        )
        self.SetName("Suno AI Anti-Block Protocol Dialog")

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Cheat Sheet Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        title_lbl = wx.StaticText(panel, label=tr("SUNO_GUIDE_HEADER_LBL"))
        title_lbl.SetName("Protocol Title Label")
        font = title_lbl.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title_lbl.SetFont(font)
        main_sizer.Add(title_lbl, 0, wx.ALL, 10)

        self.text_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="Suno Protocol Text"
        )
        make_accessible_control(
            self.text_ctrl,
            lambda: tr("SUNO_PROTOCOL_NAME"),
            lambda: tr("SUNO_PROTOCOL_NAME")
        )
        cheat_ctrl = self.text_ctrl

        if get_language() == "id":
            protocol_lines = [
                "==================================================================",
                "PROTOKOL UNGGAH AUDIO KE SUNO AI (REFERENSI RISET DAN PENGUJIAN)",
                "Situs Web Asli: http://technokerslab.blogspot.com/",
                "==================================================================",
                "",
                "1. SELALU EKSPOR SEBAGAI WAV PCM 16-BIT LOSSLESS (BUKAN MP3):",
                "   • Pengodean MP3 menyisipkan header frame, penundaan encoder, dan bit padding",
                "     yang sejajar dengan hash standar MP3.",
                "   • WAV PCM 16-bit berisi sampel linear murni tanpa metadata sama sekali,",
                "     sehingga ekstraksi landmark tidak terdeteksi oleh pemindai kontainer.",
                "",
                "2. RUTE UNGGAH PUSTAKA ABS (LEBIH AMAN):",
                "   • PENTING: JANGAN mengunggah audio langsung di halaman 'Create' / Studio!",
                "     Halaman Create Suno menjalankan pemeriksaan sinkron Audible Magic secara ketat.",
                "   • SEBALIKNYA: Buka tab 'Library' -> klik tombol 'Upload Audio'.",
                "   • Unggah potongan WAV hasil sanitasi Anda di sana terlebih dahulu. Pemrosesan",
                "     di Library menggunakan antrean latar belakang dengan aturan yang jauh lebih fleksibel.",
                "   • Setelah terunggah di Library, klik menu '...' pada trek tersebut lalu pilih",
                "     'Extend' atau 'Create with Audio'.",
                "",
                "3. PERTAHANKAN DURASI ANTARA 18 HINGGA 25 DETIK (RENTANG TERBAIK):",
                "   • Lagu lengkap menyediakan ratusan konstelasi penanda berurutan.",
                "   • Audible Magic membutuhkan setidaknya 5-8 detik kecocokan konstelasi berurutan",
                "     untuk menghasilkan deteksi dengan tingkat keyakinan tinggi.",
                "   • Memotong audio menjadi klip 20-24 detik merusak keselarasan jangka panjang",
                "     dan menyulitkan model embedding saraf VIBE milik Suno.",
                "   • Gunakan fitur bawaan 'Audio Slicer' untuk memotong trek secara otomatis.",
                "",
                "4. BERSIHKAN SESI BROWSER ATAU GUNAKAN MODE PENYAMARAN JIKA BARU DIBLOKIR:",
                "   • Jika Suno menampilkan 'This audio matches an existing recording', sesi akun",
                "     dan IP Anda akan ditandai berisiko tinggi selama 30-60 menit.",
                "   • Buka Jendela Penyamaran (Incognito / Private Window) atau hapus cookie dan cache Suno",
                "     sebelum mengunggah berkas yang telah disanitasi.",
                "",
                "5. PEREDAMAN VOKAL DAN MODERASI LIRIK:",
                "   • Vokal utama memiliki penanda biometrik dan melodi yang paling mudah dikenali.",
                "   • Gunakan preset sanitasi vokal atau isolasi stem Cloud untuk meredam kanal vokal tengah.",
                "   • Jika menempelkan lirik ke Suno, gunakan tab 'Lyrics Sanitizer' untuk menghapus",
                "     nama selebritas dan artis agar tidak ditolak oleh filter teks prompt.",
                "",
                "6. VARIASI HARMONIK DAN JEDA AWAL:",
                "   • Filter variasi harmonik mengisi spektrogram dengan titik energi acak",
                "     yang mengalihkan perhatian sistem pemindai dari audio asli.",
                "==================================================================",
                "Pemberitahuan Sumber Terbuka:",
                "GhostWave Studio adalah perangkat lunak sumber terbuka. Untuk kode sumber",
                "dan kontribusi, silakan lihat README_ID.txt.",
                "=================================================================="
            ]
        else:
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
                "   • Use the built-in 'Audio Slicer' tool to split tracks automatically.",
                "",
                "4. CLEAR BROWSER SESSION OR USE INCOGNITO IF RECENTLY BLOCKED:",
                "   • If Suno displayed 'This audio matches an existing recording', your",
                "     account session and IP have an elevated risk flag for 30-60 minutes.",
                "   • Open an Incognito / Private Window or clear Suno cookies and cache",
                "     before uploading the sanitized file.",
                "",
                "5. VOCAL REDUCTION & LYRICS MODERATION:",
                "   • Lead vocals contain the most prominent biometric and melodic markers.",
                "   • Always use vocal sanitization or Cloud stem isolation",
                "     to attenuate the center vocal channel.",
                "   • If pasting lyrics into Suno, use the 'Lyrics Sanitizer' tab to strip",
                "     celebrity and artist names to avoid prompt filter rejections.",
                "",
                "6. HARMONIC VARIATION & PREAMBLE MASKING:",
                "   • Harmonic variation populates the spectrogram with subtle energy variations,",
                "     preventing fingerprint matchers from indexing original audio.",
                "==================================================================",
                "Open Source Notice:",
                "GhostWave Studio is open source software. For source code and contributions,",
                "please see README.txt.",
                "=================================================================="
            ]
        cheat_ctrl.SetValue("\n".join(protocol_lines))
        main_sizer.Add(cheat_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        ok_btn = wx.Button(panel, wx.ID_OK, label=tr("SUNO_GUIDE_BTN_CLOSE"), name="Close Protocol Button")
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
            title=tr("SLICER_TITLE"),
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
            label=tr("SLICER_INFO")
        )
        info_lbl.SetName("ABS Slicer Description Label")
        main_sizer.Add(info_lbl, 0, wx.ALL, 10)

        # Source File Picker
        file_box = wx.StaticBox(panel, label=tr("SLICER_GROUP_SOURCE"))
        file_box.SetName("Slicer Source File Group")
        fb_sizer = wx.StaticBoxSizer(file_box, wx.HORIZONTAL)

        self.file_path_ctrl = wx.TextCtrl(panel, style=wx.TE_READONLY, name="Slicer Audio File Path")
        make_accessible_control(self.file_path_ctrl, lambda: tr("SLICER_GROUP_SOURCE"), lambda: tr("SLICER_TIP_PATH"))
        if initial_file and os.path.exists(initial_file):
            self.file_path_ctrl.SetValue(initial_file)
        else:
            self.file_path_ctrl.SetValue(tr("AUDIO_NO_FILE_SELECTED"))
        fb_sizer.Add(self.file_path_ctrl, 1, wx.EXPAND | wx.RIGHT, 8)

        self.browse_btn = wx.Button(panel, label=tr("SLICER_BTN_BROWSE"), name="Browse Slicer Audio File Button")
        self.browse_btn.SetToolTip(tr("SLICER_TIP_BROWSE"))
        self.browse_btn.Bind(wx.EVT_BUTTON, self.on_browse)
        fb_sizer.Add(self.browse_btn, 0)
        main_sizer.Add(fb_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Chunk Settings Grid
        grid = wx.FlexGridSizer(cols=2, vgap=10, hgap=12)
        grid.AddGrowableCol(1, 1)

        dur_lbl = wx.StaticText(panel, label=tr("SLICER_CHUNK_LBL"))
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
        make_accessible_control(self.chunk_spin, lambda: tr("SLICER_CHUNK_LBL"), lambda: tr("SLICER_TIP_CHUNK"))
        grid.Add(self.chunk_spin, 0, wx.EXPAND)

        out_dir_lbl = wx.StaticText(panel, label=tr("SLICER_OUT_DIR_LBL"))
        out_dir_lbl.SetName("Output Directory Label")
        grid.Add(out_dir_lbl, 0, wx.ALIGN_CENTER_VERTICAL)

        out_dir_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.out_dir_ctrl = wx.TextCtrl(panel, name="Slicer Output Directory Path")
        make_accessible_control(self.out_dir_ctrl, lambda: tr("SLICER_OUT_DIR_LBL"), lambda: tr("SLICER_TIP_OUT_DIR"))
        self.out_dir_ctrl.SetToolTip(tr("SLICER_TIP_OUT_DIR"))
        if initial_file and os.path.exists(initial_file):
            default_dir = os.path.join(os.path.dirname(initial_file), "suno_chunks")
        else:
            default_dir = os.path.join(os.getcwd(), "suno_chunks")
        self.out_dir_ctrl.SetValue(default_dir)
        out_dir_sizer.Add(self.out_dir_ctrl, 1, wx.EXPAND | wx.RIGHT, 8)

        self.browse_dir_btn = wx.Button(panel, label=tr("SLICER_BTN_BROWSE_DIR"), name="Browse Slicer Output Directory Button")
        self.browse_dir_btn.SetToolTip(tr("SLICER_TIP_BROWSE_DIR"))
        self.browse_dir_btn.Bind(wx.EVT_BUTTON, self.on_browse_dir)
        out_dir_sizer.Add(self.browse_dir_btn, 0)
        grid.Add(out_dir_sizer, 1, wx.EXPAND)

        main_sizer.Add(grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Slice Action Button & Gauge
        action_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.slice_btn = wx.Button(panel, label=tr("SLICER_BTN_START"), name="Generate Suno Chunks Button")
        self.slice_btn.SetToolTip(tr("SLICER_TIP_START"))
        self.slice_btn.Bind(wx.EVT_BUTTON, self.on_slice)
        action_sizer.Add(self.slice_btn, 0, wx.RIGHT, 10)

        self.gauge = wx.Gauge(panel, range=100, style=wx.GA_HORIZONTAL | wx.GA_SMOOTH, name="Slicing Progress")
        action_sizer.Add(self.gauge, 1, wx.ALIGN_CENTER_VERTICAL)
        main_sizer.Add(action_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Output Log TextCtrl
        log_lbl = wx.StaticText(panel, label=tr("SLICER_TIP_LOG"))
        log_lbl.SetName("Slicer Generation Log Label")
        main_sizer.Add(log_lbl, 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)

        self.log_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="Slicer Generation Log"
        )
        self.log_ctrl.SetToolTip(tr("SLICER_TIP_LOG"))
        make_accessible_control(
            self.log_ctrl,
            lambda: tr("SLICER_TIP_LOG"),
            lambda: tr("SLICER_TIP_LOG")
        )
        main_sizer.Add(self.log_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Close button
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.close_btn = wx.Button(panel, wx.ID_CANCEL, label=tr("CLOSE_BTN"), name="Close Slicer Dialog Button")
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
        with wx.FileDialog(self, tr("SLICER_BROWSE_FILE_TITLE"), wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                self.file_path_ctrl.SetValue(path)
                default_dir = os.path.join(os.path.dirname(path), "suno_chunks")
                self.out_dir_ctrl.SetValue(default_dir)

    def on_browse_dir(self, event: wx.CommandEvent):
        with wx.DirDialog(self, tr("SLICER_BROWSE_DIR_TITLE"), defaultPath=self.out_dir_ctrl.GetValue()) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.out_dir_ctrl.SetValue(dlg.GetPath())

    def on_slice(self, event: wx.CommandEvent):
        file_path = self.file_path_ctrl.GetValue().strip()
        if not os.path.exists(file_path):
            wx.MessageBox(tr("SLICER_ERR_NO_FILE"), tr("SLICER_ERR_TITLE"), wx.OK | wx.ICON_WARNING, self)
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
                    if len(chunks) == 0:
                        wx.MessageBox(
                            tr("SLICER_NO_CHUNKS_MSG"),
                            tr("SLICER_NO_CHUNKS_TITLE"),
                            wx.OK | wx.ICON_WARNING,
                            self
                        )
                    else:
                        wx.MessageBox(
                            tr("SLICER_SUCCESS_MSG", count=len(chunks), path=out_dir),
                            tr("SLICER_SUCCESS_TITLE"),
                            wx.OK | wx.ICON_INFORMATION,
                            self
                        )
                if wx.GetApp():
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
                tr("DROP_UNSUPPORTED_MSG", ext=ext),
                tr("DROP_UNSUPPORTED_TITLE"),
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
            title=tr("TICKET_DIALOG_TITLE"),
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
        self.category_choice.SetToolTip(tr("TICKET_TIP_CATEGORY"))
        cat_sizer.Add(self.category_choice, 1, wx.EXPAND)
        sizer.Add(cat_sizer, 0, wx.EXPAND | wx.ALL, 8)

        # Title Row
        title_lbl = wx.StaticText(page, label=tr("TICKET_TITLE_LABEL"))
        title_lbl.SetName("Ticket Title Label")
        sizer.Add(title_lbl, 0, wx.LEFT | wx.RIGHT | wx.TOP, 8)

        self.title_ctrl = wx.TextCtrl(page, name="Ticket Title Input")
        make_accessible_control(self.title_ctrl, lambda: tr("TICKET_TITLE_LABEL"), lambda: tr("TICKET_TIP_TITLE"))
        sizer.Add(self.title_ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Description Row
        desc_lbl = wx.StaticText(page, label=tr("TICKET_DESC_LABEL"))
        desc_lbl.SetName("Ticket Description Label")
        sizer.Add(desc_lbl, 0, wx.LEFT | wx.RIGHT | wx.TOP, 8)

        self.desc_ctrl = wx.TextCtrl(page, style=wx.TE_MULTILINE, name="Ticket Description Input")
        make_accessible_control(self.desc_ctrl, lambda: tr("TICKET_DESC_LABEL"), lambda: tr("TICKET_TIP_DESC"))
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
        make_accessible_control(self.thread_ctrl, lambda: tr("TICKET_SELECT_PROMPT"))
        right_sizer.Add(self.thread_ctrl, 1, wx.EXPAND | wx.BOTTOM, 6)

        reply_lbl = wx.StaticText(page, label=tr("TICKET_REPLY_LABEL"))
        right_sizer.Add(reply_lbl, 0, wx.BOTTOM, 2)

        reply_box = wx.BoxSizer(wx.HORIZONTAL)
        self.reply_ctrl = wx.TextCtrl(page, name="Ticket Reply Input Box")
        make_accessible_control(self.reply_ctrl, lambda: tr("TICKET_REPLY_LABEL"))
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
            t_status_raw = t.get("status", "open").lower()
            t_status_str = tr("TICKET_STATUS_OPEN") if t_status_raw == "open" else tr("TICKET_STATUS_CLOSED")
            self.ticket_listbox.Append(f"#{t_id} [{t_status_str.upper()}] {t_title}", t)
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
        self._render_ticket_view(loading=True)
        self._refresh_ticket_thread()

    def _render_ticket_view(self, comments: Optional[list] = None, loading: bool = False, err_msg: Optional[str] = None):
        if not self.active_ticket:
            return
        t_id = self.active_ticket.get("ticket_id", "?")
        t_title = self.active_ticket.get("title", "Untitled")
        t_cat = self.active_ticket.get("category", "Support")
        t_status = self.active_ticket.get("status", "open")
        t_created = self.active_ticket.get("created_at", "")[:19].replace("T", " ")
        t_desc = (self.active_ticket.get("description") or "").strip() or "(Original description not locally cached)"

        status_text = tr("TICKET_STATUS_OPEN") if t_status == "open" else tr("TICKET_STATUS_CLOSED")
        is_closed = (t_status == "closed")

        self.reply_ctrl.Enable(not is_closed)
        self.reply_btn.Enable(not is_closed)
        if is_closed:
            self.reply_ctrl.SetToolTip(tr("TICKET_CLOSED_TOOLTIP"))
            self.ticket_status_lbl.SetLabel(f"#{t_id} - {t_title} ({status_text}) - {tr('TICKET_CLOSED_NOTICE')}")
        else:
            self.reply_ctrl.SetToolTip(tr("TICKET_TIP_REPLY"))
            self.ticket_status_lbl.SetLabel(f"#{t_id} - {t_title} ({status_text})")

        lines = [
            tr("TICKET_HEADER_INFO"),
            tr("TICKET_INFO_ID", id=t_id),
            tr("TICKET_INFO_TITLE", title=t_title),
            tr("TICKET_INFO_CATEGORY", category=t_cat),
            tr("TICKET_INFO_STATUS", status=status_text)
        ]
        if t_created:
            lines.append(tr("TICKET_INFO_CREATED", created_at=t_created))
        lines.append(tr("TICKET_INFO_DESC", description=t_desc))
        lines.append("=" * 48)
        lines.append("")
        lines.append(tr("TICKET_THREAD_HEADER"))

        if is_closed:
            lines.append(f"[{tr('TICKET_CLOSED_NOTICE')}]")
            lines.append("")

        if loading:
            lines.append(tr("TICKET_LOADING_REPLIES"))
        elif err_msg:
            lines.append(tr("TICKET_FETCH_ERR", error=err_msg))
        elif comments is not None:
            if not comments:
                lines.append(tr("TICKET_NO_REPLIES"))
            else:
                for c in comments:
                    author = tr("TICKET_AUTHOR_STAFF") if c.get("is_admin") else tr("TICKET_AUTHOR_YOU")
                    time_str = c.get("created_at", "")[:19].replace("T", " ")
                    lines.append(f"[{author}] {time_str}")
                    lines.append(c.get("body", "").strip())
                    lines.append("-" * 48)

        self.thread_ctrl.SetValue("\n".join(lines))

    def _refresh_ticket_thread(self):
        if not self.active_ticket:
            return
        ticket_id = self.active_ticket.get("ticket_id")
        ticket_key = self.active_ticket.get("ticket_key")

        def worker():
            try:
                success, data, err = support_client.fetch_ticket(ticket_id, ticket_key)
            except Exception as e:
                success, data, err = False, {}, str(e)

            def done():
                if success:
                    status = data.get("status", "open")
                    self.active_ticket["status"] = status
                    if not self.active_ticket.get("description") and data.get("description"):
                        self.active_ticket["description"] = data.get("description")
                    self.config.save()
                    comments = data.get("comments", [])
                    self._render_ticket_view(comments=comments)
                else:
                    self._render_ticket_view(err_msg=err)

            if wx.GetApp():
                wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    def on_submit_ticket(self, event: wx.CommandEvent):
        title = self.title_ctrl.GetValue().strip()
        desc = self.desc_ctrl.GetValue().strip()
        if not title or not desc:
            wx.Bell()
            wx.MessageBox(
                tr("TICKET_ERR_MISSING_FIELDS"),
                tr("TICKET_ERR_MISSING_TITLE"),
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
                        "description": desc,
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

            if wx.GetApp():
                wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    def on_send_reply(self, event: wx.CommandEvent):
        if not self.active_ticket:
            return
        if self.active_ticket.get("status") == "closed":
            wx.Bell()
            wx.MessageBox(
                tr("TICKET_CLOSED_NOTICE"),
                "Ticket Closed",
                wx.OK | wx.ICON_WARNING,
                self
            )
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

            if wx.GetApp():
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
            title=tr("ACCESSIBILITY_TITLE"),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(640, 560)
        )
        self.SetName("Screen Reader Accessibility Guide Dialog")

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Accessibility Guide Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        header_label = wx.StaticText(
            panel,
            label=tr("ACCESSIBILITY_HEADER")
        )
        header_label.SetName("Accessibility Guide Header Label")
        main_sizer.Add(header_label, 0, wx.ALL, 10)

        if get_language() == "id":
            guide_content = (
                f"GhostWave Studio v{APP_VERSION}\n"
                "Panduan Aksesibilitas Pembaca Layar dan Navigasi Keyboard\n\n"
                "Situs Web Asli: http://technokerslab.blogspot.com/\n\n"
                "Ikhtisar:\n"
                "GhostWave Studio dirancang dengan arsitektur yang mengutamakan aksesibilitas.\n"
                "Semua kontrol menyediakan nama aksesibel MSAA dan UI Automation yang jelas.\n"
                "Urutan fokus mengikuti tata letak visual dan alur kerja logis.\n\n"
                "Navigasi Umum:\n"
                "  Tab: Berpindah maju ke kontrol interaktif berikutnya.\n"
                "  Shift+Tab: Berpindah mundur ke kontrol interaktif sebelumnya.\n"
                "  Ctrl+Tab: Berpindah antara tab Audio Sanitizer dan Lyrics Sanitizer.\n"
                "  Ctrl+Shift+Tab: Berpindah mundur antar tab.\n"
                "  Escape: Menutup jendela dialog dan panel popup yang aktif.\n"
                "  F1: Membuka panduan aksesibilitas ini.\n"
                "  F2: Melihat Catatan Rilis dan Changelog.\n"
                "  Ctrl+T: Membuka Pusat Tiket Dukungan.\n"
                "  Ctrl+H atau Shift+F1: Membuka Panduan dan Tips Unggah Suno.\n\n"
                "Tab Audio Sanitizer (Alt+1 atau Ctrl+Tab):\n"
                "  Alt+B: Jelajahi berkas audio masukan.\n"
                "  Alt+P: Proses dan ekspor audio hasil sanitasi.\n"
                "  Alt+F: Buka Simulator Lolos Deteksi AI (Pre-Flight).\n"
                "  Alt+E: Jalankan audit kemiripan audio.\n"
                "  Alt+S: Tampilkan/sembunyikan pengaturan lanjutan atau buka Pemotong Audio.\n"
                "  Alt+A: Buka pengaturan API Cloud.\n"
                "  Ctrl+Shift+B: Buka Antrean Batch dan Pemantau Folder Otomatis.\n"
                "  Ctrl+U: Buka Pemotong Audio dari tab mana saja.\n"
                "  Ctrl+Shift+S: Ekspor profil pengaturan (.sn).\n"
                "  Ctrl+Shift+O: Impor profil pengaturan (.sn).\n\n"
                "Tab Lyrics Sanitizer (Alt+2 atau Ctrl+Tab):\n"
                "  Alt+S: Proses sanitasi lirik.\n"
                "  Alt+C: Salin teks lirik hasil sanitasi ke papan klip.\n"
                "  Alt+L: Bersihkan teks masukan dan keluaran lirik.\n"
                "  Alt+M: Muat contoh lirik uji coba.\n"
                "  Ctrl+B: Buka editor daftar hitam nama artis dan selebriti.\n"
                "  Ctrl+L: Buka pengaturan AI Cloud lirik.\n\n"
                "Kompatibilitas Pembaca Layar:\n"
                "Telah diuji dengan NVDA 2023+, JAWS 2023+, dan Windows Narrator.\n"
                "Semua pembaruan status disampaikan melalui teks bilah status dan sinyal audio sistem.\n\n"
                "Dukungan dan Pertanyaan:\n"
                "Untuk bantuan, pertanyaan, atau laporan masalah, kirim email ke hafiyanajah@gmail.com atau buat tiket dukungan (Ctrl+T).\n\n"
                "Pemberitahuan Sumber Terbuka:\n"
                "GhostWave Studio adalah perangkat lunak sumber terbuka. Untuk kode sumber dan kontribusi, silakan lihat README_ID.txt."
            )
        else:
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
                "  F2: View What's New and Changelog.\n"
                "  Ctrl+T: Open Support Ticket center.\n"
                "  Ctrl+H or Shift+F1: Open the Suno Upload Guide and Tips.\n\n"
                "Audio Sanitizer Tab (Alt+1 or Ctrl+Tab):\n"
                "  Alt+B: Browse for input audio file.\n"
                "  Alt+P: Process and export sanitized audio.\n"
                "  Alt+F: Open Pre-Flight AI Evasion Simulator.\n"
                "  Alt+E: Run audio similarity audit.\n"
                "  Alt+S: Toggle advanced settings panel or open Audio Slicer.\n"
                "  Alt+A: Open Cloud API settings dialog.\n"
                "  Ctrl+Shift+B: Open Batch Queue and Folder Watcher dialog.\n"
                "  Ctrl+U: Launch Audio Slicer from any tab.\n"
                "  Ctrl+Shift+S: Export configuration profile (.sn).\n"
                "  Ctrl+Shift+O: Import configuration profile (.sn).\n\n"
                "Lyrics Sanitizer Tab (Alt+2 or Ctrl+Tab):\n"
                "  Alt+S: Sanitize lyrics.\n"
                "  Alt+C: Copy sanitized lyrics output to system clipboard.\n"
                "  Alt+L: Clear lyrics input and output fields.\n"
                "  Alt+M: Load sample lyrics.\n"
                "  Ctrl+B: Open celebrity and artist blacklist editor.\n"
                "  Ctrl+L: Open Cloud LLM settings dialog.\n\n"
                "Screen Reader Compatibility:\n"
                "Tested with NVDA 2023+, JAWS 2023+, and Windows Narrator.\n"
                "All status updates are announced via status bar text and system audio cues.\n\n"
                "Support and Inquiries:\n"
                "For assistance, questions, or bug reports, email hafiyanajah@gmail.com or submit an in-app support ticket (Ctrl+T).\n\n"
                "Open Source Notice:\n"
                "GhostWave Studio is open source software. For source code and contributions, please see README.txt."
            )

        self.text_ctrl = wx.TextCtrl(
            panel,
            value=guide_content,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="Accessibility Guide Read-Only Text Area"
        )
        self.text_ctrl.SetToolTip(tr("ACCESSIBILITY_TIP_TEXT"))
        make_accessible_control(
            self.text_ctrl,
            lambda: tr("ACCESSIBILITY_TITLE"),
            lambda: tr("ACCESSIBILITY_TIP_TEXT")
        )
        main_sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.copy_btn = wx.Button(
            panel,
            label=tr("ACCESSIBILITY_BTN_COPY"),
            name="Copy Guide to Clipboard Button"
        )
        self.copy_btn.SetToolTip(tr("ACCESSIBILITY_TIP_COPY"))
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        btn_sizer.Add(self.copy_btn, 0, wx.RIGHT, 10)

        btn_sizer.AddStretchSpacer()

        self.close_btn = wx.Button(
            panel,
            wx.ID_CANCEL,
            label=tr("ACCESSIBILITY_BTN_CLOSE"),
            name="Close Accessibility Guide Button"
        )
        self.close_btn.SetToolTip(tr("ACCESSIBILITY_TIP_CLOSE"))
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
            try:
                wx.TheClipboard.SetData(wx.TextDataObject(self.text_ctrl.GetValue()))
                wx.Bell()
            finally:
                wx.TheClipboard.Close()

    def on_key_hook(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()


class ChangelogDialog(wx.Dialog):
    """
    Accessible dialog displaying What's New and Changelog.
    Reads local cache immediately, and fetches fresh release notes from GitHub in background.
    """

    def __init__(self, parent: Optional[wx.Window] = None):
        super().__init__(
            parent,
            title=tr("CHANGELOG_DIALOG_TITLE"),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(640, 560)
        )
        self.SetName("What's New Dialog")

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("What's New Dialog Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        self.header_label = wx.StaticText(
            panel,
            label=tr("CHANGELOG_HEADER")
        )
        self.header_label.SetName("What's New Header Label")
        main_sizer.Add(self.header_label, 0, wx.ALL, 10)

        initial_text = get_cached_changelog_text()

        self.text_ctrl = wx.TextCtrl(
            panel,
            value=initial_text,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_THEME,
            name="Changelog and Release Notes Text Area"
        )
        self.text_ctrl.SetToolTip(tr("CHANGELOG_TIP_TEXT"))
        make_accessible_control(
            self.text_ctrl,
            lambda: tr("CHANGELOG_TITLE"),
            lambda: tr("CHANGELOG_TIP_TEXT")
        )
        main_sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.refresh_btn = wx.Button(
            panel,
            label=tr("CHANGELOG_REFRESH_BTN"),
            name="Refresh Changelog Button"
        )
        self.refresh_btn.SetToolTip(tr("CHANGELOG_TIP_REFRESH"))
        self.refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh)
        btn_sizer.Add(self.refresh_btn, 0, wx.RIGHT, 10)

        self.copy_btn = wx.Button(
            panel,
            label=f"&{tr('COPY_BTN')}",
            name="Copy Changelog to Clipboard Button"
        )
        self.copy_btn.SetToolTip(tr("CHANGELOG_TIP_COPY"))
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        btn_sizer.Add(self.copy_btn, 0, wx.RIGHT, 10)

        btn_sizer.AddStretchSpacer()

        self.close_btn = wx.Button(
            panel,
            wx.ID_CANCEL,
            label=f"&{tr('CLOSE_BTN')}",
            name="Close What's New Dialog Button"
        )
        self.close_btn.SetToolTip(tr("CHANGELOG_TIP_CLOSE"))
        self.close_btn.SetDefault()
        btn_sizer.Add(self.close_btn, 0)

        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        panel.SetSizer(main_sizer)

        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dialog_sizer)
        self.CentreOnParent()

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key)

        import threading
        threading.Thread(target=self._fetch_remote_changelog, daemon=True).start()

    def _fetch_remote_changelog(self):
        fresh = fetch_and_save_latest_changelog(timeout=4.0)
        if fresh and wx.App.Get():
            try:
                wx.CallAfter(self._update_text, fresh)
            except Exception:
                pass

    def _update_text(self, text: str):
        try:
            if self and hasattr(self, "text_ctrl") and self.text_ctrl:
                self.text_ctrl.SetValue(text)
        except Exception:
            pass

    def on_refresh(self, event: wx.CommandEvent):
        self.refresh_btn.Enable(False)
        self.refresh_btn.SetLabel(tr("CHANGELOG_UPDATING_BTN"))

        import threading
        def _worker():
            fresh = fetch_and_save_latest_changelog(timeout=5.0)
            def _done():
                try:
                    if self and hasattr(self, "refresh_btn") and self.refresh_btn:
                        self.refresh_btn.Enable(True)
                        self.refresh_btn.SetLabel(tr("CHANGELOG_REFRESH_BTN"))
                        if fresh and hasattr(self, "text_ctrl") and self.text_ctrl:
                            self.text_ctrl.SetValue(fresh)
                            wx.Bell()
                except Exception:
                    pass
            if wx.App.Get():
                try:
                    wx.CallAfter(_done)
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    def on_copy(self, event: wx.CommandEvent):
        content = self.text_ctrl.GetValue()
        if wx.TheClipboard.Open():
            try:
                wx.TheClipboard.SetData(wx.TextDataObject(content))
                wx.Bell()
            finally:
                wx.TheClipboard.Close()

    def on_key(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()


class AboutDialog(wx.Dialog):
    """
    Accessible About dialog with scrollable read-only information,
    features overview, and close button.
    """

    def __init__(self, parent: Optional[wx.Window] = None):
        super().__init__(
            parent,
            title=f"{tr('ABOUT_TITLE')} v{APP_VERSION}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(620, 540)
        )
        self.SetName("About GhostWave Studio Dialog")

        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("About GhostWave Studio Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        header_label = wx.StaticText(
            panel,
            label=f"{tr('ABOUT_HEADER')} v{APP_VERSION}:"
        )
        header_label.SetName("About Header Label")
        main_sizer.Add(header_label, 0, wx.ALL, 10)

        if get_language() == "id":
            about_text = (
                f"GhostWave Studio v{APP_VERSION}\n\n"
                "Aplikasi desktop yang mudah diakses untuk sanitasi audio dan lirik.\n"
                "Mempersiapkan trek audio dan lirik untuk platform seperti Suno\n"
                "dengan menerapkan filter pemrosesan audio dan pemformatan teks.\n\n"
                "Situs Web Asli:\n"
                "  http://technokerslab.blogspot.com/\n\n"
                "Fitur:\n"
                "  - Audio Sanitizer: Preset platform target, deteksi nada dasar dan BPM, pergeseran nada, formant, tempo, penetral watermark AI, dan peredaman vokal.\n"
                "  - Pre-Flight Simulator & Audit: Simulasi skor keamanan multi-vektor, visualizer gelombang, dan pemutar uji dengar A/B/Delta.\n"
                "  - Pemrosesan Batch & Pemantau Folder: Antrean banyak berkas dan pemrosesan otomatis di latar belakang.\n"
                "  - Lyrics Sanitizer: Filter kata kasar, filter nama selebriti, pembersihan Unicode, pemenggal suku kata, dan pembuat prompt gaya musik.\n"
                "  - Audio Slicer: Memotong trek audio menjadi potongan WAV yang siap diunggah ke pustaka.\n"
                "  - Tiket Dukungan: Kirim tiket dan pantau balasan pengembang langsung dari aplikasi.\n"
                "  - Profil Pengaturan: Disimpan secara aman di komputer lokal dalam format .sn.\n"
                "  - Aksesibilitas: Kompatibilitas penuh dengan pembaca layar dan pintasan keyboard.\n\n"
                "Dukungan dan Pertanyaan:\n"
                "  Kontak: hafiyanajah@gmail.com atau kirim tiket dukungan dalam aplikasi (Ctrl+T).\n\n"
                "Pemberitahuan Sumber Terbuka:\n"
                "  GhostWave Studio adalah perangkat lunak sumber terbuka. Untuk kode sumber dan panduan kontribusi, silakan lihat README_ID.txt."
            )
        else:
            about_text = (
                f"GhostWave Studio v{APP_VERSION}\n\n"
                "An accessible desktop application for audio and lyrics sanitization.\n"
                "Prepares audio tracks and lyrics for platforms like Suno\n"
                "by applying audio processing filters and text formatting.\n\n"
                "Original Website:\n"
                "  http://technokerslab.blogspot.com/\n\n"
                "Features:\n"
                "  - Audio Sanitizer: Target platform presets, key and BPM detection, pitch, formant, tempo, AI watermark neutralizer, and vocal reduction.\n"
                "  - Pre-Flight Simulator & Audit: Multi-vector safety audit, waveform visualizer, and real-time A/B/Delta audition player.\n"
                "  - Batch & Folder Watcher: Multi-file processing queue and automatic background folder watching.\n"
                "  - Lyrics Sanitizer: Profanity filter, celebrity name filter, Unicode cleanup, multilingual syllabifier, and style prompt generator.\n"
                "  - Audio Slicer: Split audio tracks into WAV chunks suitable for library uploads.\n"
                "  - In-App Support: Submit tickets and track developer replies directly within the app.\n"
                "  - Settings Profiles: Stored locally in .sn format.\n"
                "  - Accessibility: Full screen reader compatibility and keyboard shortcuts.\n\n"
                "Support and Inquiries:\n"
                "  Contact: hafiyanajah@gmail.com or submit an in-app support ticket (Ctrl+T).\n\n"
                "Open Source Notice:\n"
                "  GhostWave Studio is open source software. For source code and contributions, please see README.txt."
            )

        self.text_ctrl = wx.TextCtrl(
            panel,
            value=about_text,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            name="About Software Read-Only Text Area"
        )
        self.text_ctrl.SetToolTip(tr("ABOUT_TIP_TEXT"))
        make_accessible_control(
            self.text_ctrl,
            lambda: tr("ABOUT_TITLE"),
            lambda: tr("ABOUT_TIP_TEXT")
        )
        main_sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddStretchSpacer()

        self.close_btn = wx.Button(
            panel,
            wx.ID_CANCEL,
            label=tr("ABOUT_BTN_CLOSE"),
            name="Close About Dialog Button"
        )
        self.close_btn.SetToolTip(tr("ABOUT_TIP_CLOSE"))
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


class AudioSanitizerPanel(scrolled.ScrolledPanel):
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
        if hasattr(self, "on_mode_changed"):
            self.on_mode_changed(None)

    def _build_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # -------------------------------------------------------------
        # Section 1: Source Audio Picker
        # -------------------------------------------------------------
        self.file_box = wx.StaticBox(self, label=tr("AUDIO_GROUP_SOURCE"))
        self.file_box.SetName("Source Audio File Group")
        file_box_sizer = wx.StaticBoxSizer(self.file_box, wx.VERTICAL)

        picker_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.file_label = wx.StaticText(self, label=tr("AUDIO_LABEL_SELECTED_FILE"))
        self.file_label.SetName("Selected Audio File Label")
        picker_sizer.Add(self.file_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.file_path_ctrl = wx.TextCtrl(
            self,
            style=wx.TE_READONLY,
            name="Selected Audio File Path"
        )
        make_accessible_control(self.file_path_ctrl, lambda: tr("AUDIO_LABEL_SELECTED_FILE"), lambda: tr("TIP_AUDIO_FILE_PATH"))
        self.file_path_ctrl.SetValue(tr("AUDIO_NO_FILE_SELECTED"))
        picker_sizer.Add(self.file_path_ctrl, 1, wx.EXPAND | wx.RIGHT, 10)

        self.browse_btn = wx.Button(self, label=f"&{tr('AUDIO_SELECT_BTN')}", name="Browse Audio File Button")
        self.browse_btn.SetToolTip(tr("TIP_AUDIO_BROWSE_BTN"))
        self.browse_btn.Bind(wx.EVT_BUTTON, self.on_browse)
        picker_sizer.Add(self.browse_btn, 0, wx.ALIGN_CENTER_VERTICAL)

        file_box_sizer.Add(picker_sizer, 0, wx.EXPAND | wx.ALL, 8)

        self.file_info_text = wx.StaticText(
            self,
            label=tr("AUDIO_SPECS_NONE"),
            name="Audio Specifications Information"
        )
        file_box_sizer.Add(self.file_info_text, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Key & BPM Detection & Recommendation Row
        rec_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.key_bpm_label = wx.StaticText(
            self,
            label=tr("AUDIO_KEY_BPM_NONE"),
            name="Key and BPM Detection Label"
        )
        rec_sizer.Add(self.key_bpm_label, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.apply_rec_btn = wx.Button(
            self,
            label=tr("AUDIO_BTN_APPLY_REC", rec=""),
            name="Apply Key Recommendation Button"
        )
        self.apply_rec_btn.SetToolTip(tr("AUDIO_BTN_APPLY_REC", rec="Key"))
        self.apply_rec_btn.Enable(False)
        self.apply_rec_btn.Bind(wx.EVT_BUTTON, self.on_apply_rec_pitch)
        rec_sizer.Add(self.apply_rec_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        file_box_sizer.Add(rec_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        main_sizer.Add(file_box_sizer, 0, wx.EXPAND | wx.ALL, 8)

        # -------------------------------------------------------------
        # Section 2: Sanitization Parameters & Multi-Vector DSP
        # -------------------------------------------------------------
        self.param_box = wx.StaticBox(self, label=tr("AUDIO_GROUP_PARAMS"))
        self.param_box.SetName("Sanitization Parameters Group")
        param_box_sizer = wx.StaticBoxSizer(self.param_box, wx.VERTICAL)

        # Target Platform Selection
        platform_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.platform_label = wx.StaticText(self, label=tr("AUDIO_LABEL_PLATFORM"))
        self.platform_label.SetName("Platform Target Label")
        platform_sizer.Add(self.platform_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.platform_choice = wx.Choice(
            self,
            choices=[
                tr("AUDIO_PLATFORM_SUNO"),
                tr("AUDIO_PLATFORM_LYRIA"),
                tr("AUDIO_PLATFORM_UDIO"),
                tr("AUDIO_PLATFORM_YOUTUBE"),
                tr("AUDIO_PLATFORM_TIKTOK"),
                tr("AUDIO_PLATFORM_CUSTOM")
            ],
            name="Platform Target Selection"
        )
        self.platform_choice.SetSelection(0)
        self.platform_choice.Bind(wx.EVT_CHOICE, self.on_platform_changed)
        platform_sizer.Add(self.platform_choice, 1, wx.EXPAND | wx.RIGHT, 10)
        param_box_sizer.Add(platform_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 6)

        # Workflow Mode Selection (Vocal Cover vs Instrumental)
        mode_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.mode_label = wx.StaticText(self, label=tr("AUDIO_LABEL_MODE"))
        self.mode_label.SetName("Workflow Mode Label")
        mode_sizer.Add(self.mode_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.mode_choice = wx.Choice(
            self,
            choices=[
                tr("AUDIO_MODE_VOCAL_COVER"),
                tr("AUDIO_MODE_INSTRUMENTAL")
            ],
            name="Workflow Mode Selection"
        )
        self.mode_choice.SetSelection(0)
        self.mode_choice.SetToolTip(tr("TIP_AUDIO_MODE"))
        self.mode_choice.Bind(wx.EVT_CHOICE, self.on_mode_changed)
        mode_sizer.Add(self.mode_choice, 1, wx.EXPAND | wx.RIGHT, 10)
        param_box_sizer.Add(mode_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 6)

        # Modular Controls Box (User-adjustable sliders with clear defaults)
        self.modular_box = wx.StaticBox(self, label=tr("AUDIO_MODULAR_BOX"))
        self.modular_box.SetName("Modular Controls Group")
        mod_box_sizer = wx.StaticBoxSizer(self.modular_box, wx.VERTICAL)

        sliders_grid = wx.FlexGridSizer(cols=4, vgap=6, hgap=12)
        sliders_grid.AddGrowableCol(1, 1)
        sliders_grid.AddGrowableCol(3, 1)

        # Pitch Shift Slider
        self.pitch_lbl = wx.StaticText(self, label=tr("AUDIO_SLIDER_PITCH"))
        self.pitch_lbl.SetName("Pitch Shift Label")
        sliders_grid.Add(self.pitch_lbl, 0, wx.ALIGN_CENTER_VERTICAL)
        self.pitch_spin = wx.SpinCtrlDouble(self, value="2.0", min=-12.0, max=12.0, inc=0.5, name="Pitch Shift Semitones")
        self.pitch_spin.SetDigits(1)
        make_accessible_control(self.pitch_spin, lambda: tr("AUDIO_SLIDER_PITCH"), lambda: tr("TIP_AUDIO_PITCH"))
        sliders_grid.Add(self.pitch_spin, 1, wx.EXPAND)

        # Formant Shift Slider
        self.formant_lbl = wx.StaticText(self, label=tr("AUDIO_SLIDER_FORMANT"))
        self.formant_lbl.SetName("Formant Shift Label")
        sliders_grid.Add(self.formant_lbl, 0, wx.ALIGN_CENTER_VERTICAL)
        self.formant_spin = wx.SpinCtrlDouble(self, value="2.0", min=-6.0, max=6.0, inc=0.5, name="Formant Shift Semitones")
        self.formant_spin.SetDigits(1)
        make_accessible_control(self.formant_spin, lambda: tr("AUDIO_SLIDER_FORMANT"), lambda: tr("TIP_AUDIO_FORMANT"))
        sliders_grid.Add(self.formant_spin, 1, wx.EXPAND)

        # Tempo Slider
        self.tempo_lbl = wx.StaticText(self, label=tr("AUDIO_SLIDER_TEMPO"))
        self.tempo_lbl.SetName("Tempo Shift Label")
        sliders_grid.Add(self.tempo_lbl, 0, wx.ALIGN_CENTER_VERTICAL)
        self.tempo_spin = wx.SpinCtrlDouble(self, value="0.940", min=0.850, max=1.150, inc=0.010, name="Tempo Factor")
        self.tempo_spin.SetDigits(3)
        make_accessible_control(self.tempo_spin, lambda: tr("AUDIO_SLIDER_TEMPO"), lambda: tr("TIP_AUDIO_TEMPO"))
        sliders_grid.Add(self.tempo_spin, 1, wx.EXPAND)

        # Reverb Wet Slider
        self.reverb_lbl = wx.StaticText(self, label=tr("AUDIO_SLIDER_REVERB"))
        self.reverb_lbl.SetName("Reverb Wet Label")
        sliders_grid.Add(self.reverb_lbl, 0, wx.ALIGN_CENTER_VERTICAL)
        self.reverb_spin = wx.SpinCtrlDouble(self, value="18.0", min=0.0, max=50.0, inc=1.0, name="Reverb Wet Percent")
        self.reverb_spin.SetDigits(0)
        make_accessible_control(self.reverb_spin, lambda: tr("AUDIO_SLIDER_REVERB"), lambda: tr("TIP_AUDIO_REVERB"))
        sliders_grid.Add(self.reverb_spin, 1, wx.EXPAND)

        # Vocal Ducking Slider
        self.vocal_duck_lbl = wx.StaticText(self, label=tr("AUDIO_SLIDER_VOCAL"))
        self.vocal_duck_lbl.SetName("Vocal Ducking Label")
        sliders_grid.Add(self.vocal_duck_lbl, 0, wx.ALIGN_CENTER_VERTICAL)
        self.vocal_duck_spin = wx.SpinCtrlDouble(self, value="12.0", min=0.0, max=24.0, inc=1.0, name="Vocal Ducking dB")
        self.vocal_duck_spin.SetDigits(1)
        make_accessible_control(self.vocal_duck_spin, lambda: tr("AUDIO_SLIDER_VOCAL"), lambda: tr("TIP_AUDIO_VOCAL_DUCK"))
        sliders_grid.Add(self.vocal_duck_spin, 1, wx.EXPAND)

        mod_box_sizer.Add(sliders_grid, 0, wx.EXPAND | wx.ALL, 6)

        # Modular Checkboxes
        mod_chk_sizer = wx.FlexGridSizer(cols=2, vgap=4, hgap=12)
        mod_chk_sizer.AddGrowableCol(0, 1)
        mod_chk_sizer.AddGrowableCol(1, 1)

        self.watermark_chk = wx.CheckBox(self, label=tr("AUDIO_CHK_WATERMARK"), name="AI Watermark Neutralizer Checkbox")
        self.watermark_chk.SetValue(True)
        self.watermark_chk.SetToolTip(tr("TIP_AUDIO_WATERMARK"))
        mod_chk_sizer.Add(self.watermark_chk, 0, wx.ALL, 2)

        self.psycho_chk = wx.CheckBox(self, label=tr("AUDIO_CHK_PSYCHO"), name="Psychoacoustic Masking Checkbox")
        self.psycho_chk.SetValue(True)
        self.psycho_chk.SetToolTip(tr("TIP_AUDIO_PSYCHO"))
        mod_chk_sizer.Add(self.psycho_chk, 0, wx.ALL, 2)

        self.articulation_chk = wx.CheckBox(self, label=tr("AUDIO_ARTICULATION_SOFTENER"), name="Vocal Articulation Softener Checkbox")
        self.articulation_chk.SetValue(True)
        self.articulation_chk.SetToolTip(tr("TIP_AUDIO_ARTICULATION"))
        mod_chk_sizer.Add(self.articulation_chk, 0, wx.ALL, 2)

        self.vocal_chk = wx.CheckBox(self, label=tr("AUDIO_VOCAL_CUT"), name="Center Vocal Suppression Checkbox")
        self.vocal_chk.SetValue(False)
        self.vocal_chk.SetToolTip(tr("TIP_AUDIO_VOCAL"))
        mod_chk_sizer.Add(self.vocal_chk, 0, wx.ALL, 2)

        mod_box_sizer.Add(mod_chk_sizer, 0, wx.EXPAND | wx.ALL, 6)
        param_box_sizer.Add(mod_box_sizer, 0, wx.EXPAND | wx.ALL, 6)

        # Preset Choice + Cloud Settings Button
        preset_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.preset_label = wx.StaticText(self, label=tr("AUDIO_LABEL_PRESET"))
        self.preset_label.SetName("Sanitization Preset Label")
        preset_sizer.Add(self.preset_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.preset_choice = wx.Choice(
            self,
            choices=[
                "Complete Sanitization (Recommended)",
                "Balanced Audio Quality",
                "Room Acoustic Simulation",
                "Frequency & Phase Shift",
                "Vocal Reduction Only",
                "Cloud API Stem Separation"
            ],
            name="Sanitization Preset Selection"
        )
        self.preset_choice.SetSelection(0)
        self.preset_choice.SetToolTip(tr("TIP_AUDIO_PRESET"))
        self.preset_choice.Bind(wx.EVT_CHOICE, self.on_preset_changed)
        preset_sizer.Add(self.preset_choice, 1, wx.EXPAND | wx.RIGHT, 10)

        self.cloud_btn = wx.Button(self, label=tr("AUDIO_BTN_CLOUD_SETTINGS"), name="Cloud API Settings Button")
        self.cloud_btn.SetToolTip(tr("TIP_AUDIO_CLOUD_BTN"))
        self.cloud_btn.Bind(wx.EVT_BUTTON, self.on_cloud_settings)
        preset_sizer.Add(self.cloud_btn, 0, wx.ALIGN_CENTER_VERTICAL)

        param_box_sizer.Add(preset_sizer, 0, wx.EXPAND | wx.ALL, 6)

        # Basic Format & Advanced Toggle Row
        format_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.format_label = wx.StaticText(self, label=tr("AUDIO_LABEL_FORMAT"))
        self.format_label.SetName("Export Format Label")
        format_sizer.Add(self.format_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

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
            label=tr("AUDIO_SHOW_ADVANCED_BTN"),
            name="Show Advanced Settings Button"
        )
        self.toggle_adv_btn.SetToolTip(tr("TIP_AUDIO_ADV_SHOW"))
        self.toggle_adv_btn.Bind(wx.EVT_BUTTON, self.on_toggle_advanced)
        format_sizer.Add(self.toggle_adv_btn, 0, wx.ALIGN_CENTER_VERTICAL)

        param_box_sizer.Add(format_sizer, 0, wx.EXPAND | wx.ALL, 6)

        # Advanced Settings Collapsible Panel
        self.adv_panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        self.adv_panel.SetName("Advanced DSP Settings Panel")
        adv_sizer = wx.BoxSizer(wx.VERTICAL)

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
        self.jitter_chk.SetToolTip(tr("TIP_AUDIO_JITTER"))
        chk_sizer.Add(self.jitter_chk, 0, wx.ALL, 2)

        self.bode_chk = wx.CheckBox(
            self.adv_panel,
            label="Hilbert Bode Frequency Shifter (+8.5 Hz Anti-Chroma)",
            name="Hilbert Bode Frequency Shifter Checkbox"
        )
        self.bode_chk.SetValue(True)
        self.bode_chk.SetToolTip(tr("TIP_AUDIO_BODE"))
        chk_sizer.Add(self.bode_chk, 0, wx.ALL, 2)

        self.decoy_chk = wx.CheckBox(
            self.adv_panel,
            label="Adversarial Pseudo-Peak Injection (STFT Decoy Landmarks)",
            name="Adversarial Pseudo-Peak Injection Checkbox"
        )
        self.decoy_chk.SetValue(True)
        self.decoy_chk.SetToolTip(tr("TIP_AUDIO_DECOY"))
        chk_sizer.Add(self.decoy_chk, 0, wx.ALL, 2)

        self.allpass_chk = wx.CheckBox(
            self.adv_panel,
            label="Schroeder All-Pass Phase Dispersion (Flat Frequency Mag)",
            name="Schroeder All-Pass Phase Dispersion Checkbox"
        )
        self.allpass_chk.SetValue(True)
        self.allpass_chk.SetToolTip(tr("TIP_AUDIO_ALLPASS"))
        chk_sizer.Add(self.allpass_chk, 0, wx.ALL, 2)

        self.reamping_chk = wx.CheckBox(
            self.adv_panel,
            label="Virtual Acoustic Re-Amping (Studio Room Simulation)",
            name="Virtual Acoustic Re-Amping Checkbox"
        )
        self.reamping_chk.SetValue(True)
        self.reamping_chk.SetToolTip(tr("TIP_AUDIO_REAMPING"))
        chk_sizer.Add(self.reamping_chk, 0, wx.ALL, 2)

        self.preamble_chk = wx.CheckBox(
            self.adv_panel,
            label="Front-End Preamble Camouflage (3.5s Analog Synth Intro)",
            name="Front-End Preamble Camouflage Checkbox"
        )
        self.preamble_chk.SetValue(True)
        self.preamble_chk.SetToolTip(tr("TIP_AUDIO_PREAMBLE"))
        chk_sizer.Add(self.preamble_chk, 0, wx.ALL, 2)

        self.filter_chk = wx.CheckBox(
            self.adv_panel,
            label="Apply 25 Hz High-Pass & 18 kHz Low-Pass filter to strip fingerprint metadata",
            name="Apply 25 Hz High-Pass and 18 kHz Low-Pass filter to strip fingerprint metadata"
        )
        self.filter_chk.SetValue(True)
        self.filter_chk.SetToolTip(tr("TIP_AUDIO_FILTERS"))
        chk_sizer.Add(self.filter_chk, 0, wx.ALL, 2)

        self.dither_chk = wx.CheckBox(
            self.adv_panel,
            label="Add ultra-low floor dither (-65 dB) to break spectral hash",
            name="Add ultra-low floor dither (-65 dB) to break spectral hash"
        )
        self.dither_chk.SetValue(True)
        self.dither_chk.SetToolTip(tr("TIP_AUDIO_DITHER"))
        chk_sizer.Add(self.dither_chk, 0, wx.ALL, 2)

        self.metadata_chk = wx.CheckBox(
            self.adv_panel,
            label="Automatically strip all ID3 tags, artist tags, and album art from output",
            name="Automatically strip all ID3 tags, artist tags, and album art from output"
        )
        self.metadata_chk.SetValue(True)
        self.metadata_chk.SetToolTip(tr("TIP_AUDIO_METADATA"))
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
        self.trim_chk.SetToolTip(tr("TIP_AUDIO_TRIM_CHK"))
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
        make_accessible_control(self.trim_spin, lambda: tr("AUDIO_TRIM_CLIP"), lambda: tr("TIP_AUDIO_TRIM_SPIN"))
        self.trim_spin.Enable(False)
        trim_sizer.Add(self.trim_spin, 0, wx.ALIGN_CENTER_VERTICAL)

        self.seconds_label = wx.StaticText(self.adv_panel, label=tr("AUDIO_UNIT_SECONDS"))
        self.seconds_label.SetName("Seconds Unit Label")
        trim_sizer.Add(self.seconds_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)

        adv_sizer.Add(trim_sizer, 0, wx.ALL, 4)

        self.adv_panel.SetSizer(adv_sizer)
        self.adv_panel.Show(False)
        param_box_sizer.Add(self.adv_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        main_sizer.Add(param_box_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # -------------------------------------------------------------
        # Section 3: Action Buttons & Progress (2 balanced rows for 760px+)
        # -------------------------------------------------------------
        action_sizer = wx.BoxSizer(wx.VERTICAL)
        action_row1 = wx.BoxSizer(wx.HORIZONTAL)

        self.process_btn = wx.Button(
            self,
            label=tr("AUDIO_PROCESS_BTN"),
            name="Process and Export Audio Button"
        )
        self.process_btn.SetToolTip(tr("TIP_AUDIO_PROCESS_BTN"))
        self.process_btn.Bind(wx.EVT_BUTTON, self.on_process)
        action_row1.Add(self.process_btn, 0, wx.RIGHT, 6)

        self.cancel_btn = wx.Button(
            self,
            label=tr("AUDIO_CANCEL_BTN"),
            name="Cancel Audio Processing Button"
        )
        self.cancel_btn.SetToolTip(tr("TIP_AUDIO_CANCEL_BTN"))
        self.cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel_processing)
        self.cancel_btn.Enable(False)
        action_row1.Add(self.cancel_btn, 0, wx.RIGHT, 6)

        self.preflight_btn = wx.Button(
            self,
            label=tr("AUDIO_BTN_PREFLIGHT"),
            name="Pre-Flight Evasion Simulator Button"
        )
        self.preflight_btn.SetToolTip(tr("TIP_PREFLIGHT_BTN"))
        self.preflight_btn.Bind(wx.EVT_BUTTON, self.on_preflight)
        self.preflight_btn.Enable(False)
        action_row1.Add(self.preflight_btn, 0, wx.RIGHT, 8)

        self.gauge = wx.Gauge(
            self,
            range=100,
            style=wx.GA_HORIZONTAL | wx.GA_SMOOTH,
            name="Processing Progress"
        )
        action_row1.Add(self.gauge, 1, wx.ALIGN_CENTER_VERTICAL)
        action_sizer.Add(action_row1, 0, wx.EXPAND | wx.BOTTOM, 6)

        action_row2 = wx.BoxSizer(wx.HORIZONTAL)

        self.batch_btn = wx.Button(
            self,
            label=tr("AUDIO_BTN_BATCH"),
            name="Batch Processing and Folder Watcher Button"
        )
        self.batch_btn.SetToolTip(tr("TIP_BATCH_BTN"))
        self.batch_btn.Bind(wx.EVT_BUTTON, self.on_batch)
        action_row2.Add(self.batch_btn, 0, wx.RIGHT, 6)

        self.audit_btn = wx.Button(
            self,
            label=tr("AUDIO_BTN_RUN_AUDIT"),
            name="Verify Evasion Safety Button"
        )
        self.audit_btn.SetToolTip(tr("TIP_AUDIO_AUDIT_BTN"))
        self.audit_btn.Bind(wx.EVT_BUTTON, self.on_audit)
        self.audit_btn.Enable(False)
        action_row2.Add(self.audit_btn, 0, wx.RIGHT, 6)

        self.slicer_btn = wx.Button(
            self,
            label=tr("AUDIO_BTN_SLICER"),
            name="ABS Audio Slicer Button"
        )
        self.slicer_btn.SetToolTip(tr("TIP_AUDIO_SLICER_BTN"))
        self.slicer_btn.Bind(wx.EVT_BUTTON, self.on_slicer)
        action_row2.Add(self.slicer_btn, 0, wx.RIGHT, 6)

        self.protocol_btn = wx.Button(
            self,
            label=tr("AUDIO_BTN_PROTOCOL"),
            name="Suno Upload Protocol Button"
        )
        self.protocol_btn.SetToolTip(tr("TIP_AUDIO_PROTOCOL_BTN"))
        self.protocol_btn.Bind(wx.EVT_BUTTON, self.on_protocol)
        action_row2.Add(self.protocol_btn, 0)
        action_sizer.Add(action_row2, 0, wx.EXPAND)

        main_sizer.Add(action_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # -------------------------------------------------------------
        # Section 3B: Visualizer & A/B/Delta Audition Player
        # -------------------------------------------------------------
        self.vis_box = wx.StaticBox(self, label=tr("AUDIO_VIS_GROUP"))
        self.vis_box.SetName("Waveform Visualizer and Audition Player Group")
        vis_box_sizer = wx.StaticBoxSizer(self.vis_box, wx.VERTICAL)

        self.visualizer = AudioVisualizerPanel(self, size=(-1, 105))
        vis_box_sizer.Add(self.visualizer, 0, wx.EXPAND | wx.ALL, 4)

        self.vis_summary_lbl = wx.TextCtrl(
            self,
            value=self.visualizer.get_accessible_summary(),
            style=wx.TE_READONLY | wx.BORDER_NONE,
            name="Waveform Accessibility Summary"
        )
        make_accessible_control(self.vis_summary_lbl, lambda: tr("AUDIO_VIS_GROUP"))
        vis_box_sizer.Add(self.vis_summary_lbl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        player_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.play_orig_btn = wx.Button(self, label=tr("AUDIO_BTN_PLAY_ORIG"), name="Play Original Audio Button")
        self.play_orig_btn.SetToolTip(tr("AUDIO_BTN_PLAY_ORIG"))
        self.play_orig_btn.Enable(False)
        self.play_orig_btn.Bind(wx.EVT_BUTTON, self.on_play_original)
        player_sizer.Add(self.play_orig_btn, 0, wx.RIGHT, 6)

        self.play_sani_btn = wx.Button(self, label=tr("AUDIO_BTN_PLAY_SANI"), name="Play Sanitized Audio Button")
        self.play_sani_btn.SetToolTip(tr("AUDIO_BTN_PLAY_SANI"))
        self.play_sani_btn.Enable(False)
        self.play_sani_btn.Bind(wx.EVT_BUTTON, self.on_play_sanitized)
        player_sizer.Add(self.play_sani_btn, 0, wx.RIGHT, 6)

        self.play_delta_btn = wx.Button(self, label=tr("AUDIO_BTN_PLAY_DELTA"), name="Play Delta Audio Button")
        self.play_delta_btn.SetToolTip(tr("AUDIO_BTN_PLAY_DELTA"))
        self.play_delta_btn.Enable(False)
        self.play_delta_btn.Bind(wx.EVT_BUTTON, self.on_play_delta)
        player_sizer.Add(self.play_delta_btn, 0, wx.RIGHT, 6)

        self.stop_play_btn = wx.Button(self, label=tr("AUDIO_BTN_STOP_PLAY"), name="Stop Audio Playback Button")
        self.stop_play_btn.SetToolTip(tr("AUDIO_BTN_STOP_PLAY"))
        self.stop_play_btn.Bind(wx.EVT_BUTTON, self.on_stop_playback)
        player_sizer.Add(self.stop_play_btn, 0)

        vis_box_sizer.Add(player_sizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        main_sizer.Add(vis_box_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # -------------------------------------------------------------
        # Section 4: Processing Log Area
        # -------------------------------------------------------------
        self.log_label = wx.StaticText(self, label=tr("AUDIO_GROUP_LOGS"))
        self.log_label.SetName("Audio Processing Log Label")
        main_sizer.Add(self.log_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 8)

        self.log_ctrl = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            size=(-1, 95),
            name="Audio Processing Log"
        )
        make_accessible_control(self.log_ctrl, lambda: tr("AUDIO_GROUP_LOGS"), lambda: tr("TIP_AUDIO_LOG"))
        main_sizer.Add(self.log_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.SetSizer(main_sizer)
        self.SetupScrolling(scroll_x=False, scroll_y=True)

    def on_toggle_advanced(self, event: wx.CommandEvent):
        show_adv = not self.adv_panel.IsShown()
        self.adv_panel.Show(show_adv)
        if show_adv:
            self.toggle_adv_btn.SetLabel(tr("AUDIO_HIDE_ADVANCED_BTN"))
            self.toggle_adv_btn.SetToolTip(tr("TIP_AUDIO_ADV_HIDE"))
        else:
            self.toggle_adv_btn.SetLabel(tr("AUDIO_SHOW_ADVANCED_BTN"))
            self.toggle_adv_btn.SetToolTip(tr("TIP_AUDIO_ADV_SHOW"))
        self.Layout()
        self.SetupScrolling(scroll_x=False, scroll_y=True)
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

    def _restore_core_dsp_flags(self, is_vocal_cover: bool, bode_enabled: bool):
        for attr, val in (
            ("jitter_chk", True),
            ("bode_chk", bode_enabled),
            ("decoy_chk", True),
            ("allpass_chk", True),
            ("reamping_chk", True),
            ("preamble_chk", not is_vocal_cover),
            ("filter_chk", True),
            ("dither_chk", True),
            ("metadata_chk", True),
        ):
            if hasattr(self, attr):
                getattr(self, attr).SetValue(val)

    def on_platform_changed(self, event: Optional[wx.CommandEvent] = None):
        idx = self.platform_choice.GetSelection() if hasattr(self, "platform_choice") else 0
        is_vocal_cover = (self.mode_choice.GetSelection() == 0) if hasattr(self, "mode_choice") else True
        if idx == 0:  # Suno v4-v6 Cover Mode
            self.pitch_spin.SetValue(2.0 if is_vocal_cover else 2.5)
            self.formant_spin.SetValue(2.0)
            self.tempo_spin.SetValue(0.940)
            self.reverb_spin.SetValue(18.0)
            self.vocal_duck_spin.SetValue(12.0)
            self.watermark_chk.SetValue(True)
            self.psycho_chk.SetValue(True)
            self.articulation_chk.SetValue(is_vocal_cover)
            self.vocal_chk.SetValue(not is_vocal_cover)
            self._restore_core_dsp_flags(is_vocal_cover, bode_enabled=not is_vocal_cover)
            self.set_status(tr("STATUS_PLATFORM_SUNO"))
        elif idx == 1:  # Google Lyria 3.5 / SynthID Cleaner (Watermark Neutralizer Only)
            self.pitch_spin.SetValue(0.0)
            self.formant_spin.SetValue(0.0)
            self.tempo_spin.SetValue(1.000)
            self.reverb_spin.SetValue(0.0)
            self.vocal_duck_spin.SetValue(0.0)
            self.watermark_chk.SetValue(True)
            self.psycho_chk.SetValue(False)
            for attr in (
                "articulation_chk", "vocal_chk", "jitter_chk", "bode_chk",
                "decoy_chk", "allpass_chk", "reamping_chk", "preamble_chk",
                "filter_chk", "dither_chk", "trim_chk",
            ):
                if hasattr(self, attr):
                    getattr(self, attr).SetValue(False)
            if hasattr(self, "metadata_chk"):
                self.metadata_chk.SetValue(True)
            self.set_status(tr("STATUS_PLATFORM_LYRIA"))
        elif idx == 2:  # Udio Mode
            self.pitch_spin.SetValue(2.5)
            self.formant_spin.SetValue(2.0)
            self.tempo_spin.SetValue(0.920)
            self.reverb_spin.SetValue(20.0)
            self.vocal_duck_spin.SetValue(14.0)
            self.watermark_chk.SetValue(True)
            self.psycho_chk.SetValue(True)
            self.articulation_chk.SetValue(is_vocal_cover)
            self.vocal_chk.SetValue(not is_vocal_cover)
            self._restore_core_dsp_flags(is_vocal_cover, bode_enabled=not is_vocal_cover)
            self.set_status(tr("STATUS_PLATFORM_UDIO"))
        elif idx == 3:  # YouTube Content ID
            self.pitch_spin.SetValue(3.0)
            self.formant_spin.SetValue(2.5)
            self.tempo_spin.SetValue(0.910)
            self.reverb_spin.SetValue(22.0)
            self.vocal_duck_spin.SetValue(15.0)
            self.watermark_chk.SetValue(True)
            self.psycho_chk.SetValue(True)
            self.articulation_chk.SetValue(is_vocal_cover)
            self.vocal_chk.SetValue(not is_vocal_cover)
            self._restore_core_dsp_flags(is_vocal_cover, bode_enabled=True)
            self.set_status(tr("STATUS_PLATFORM_YOUTUBE"))
        elif idx == 4:  # TikTok / Reels
            self.pitch_spin.SetValue(2.2)
            self.formant_spin.SetValue(1.8)
            self.tempo_spin.SetValue(0.950)
            self.reverb_spin.SetValue(15.0)
            self.vocal_duck_spin.SetValue(10.0)
            self.watermark_chk.SetValue(True)
            self.psycho_chk.SetValue(True)
            self.articulation_chk.SetValue(is_vocal_cover)
            self.vocal_chk.SetValue(not is_vocal_cover)
            self._restore_core_dsp_flags(is_vocal_cover, bode_enabled=not is_vocal_cover)
            self.set_status(tr("STATUS_PLATFORM_TIKTOK"))
        else:  # Custom
            self.set_status(tr("STATUS_PLATFORM_CUSTOM"))

    def on_mode_changed(self, event: Optional[wx.CommandEvent] = None):
        mode_idx = self.mode_choice.GetSelection() if hasattr(self, "mode_choice") else 0
        if mode_idx == 0:  # Vocal Cover Mode
            self.pitch_spin.SetValue(2.0)
            self.formant_spin.SetValue(2.0)
            self.tempo_spin.SetValue(0.940)
            self.vocal_chk.SetValue(False)
            if hasattr(self, "articulation_chk"):
                self.articulation_chk.SetValue(True)
            if hasattr(self, "jitter_chk"):
                self.jitter_chk.SetValue(True)
            if hasattr(self, "bode_chk"):
                self.bode_chk.SetValue(False)
            if hasattr(self, "decoy_chk"):
                self.decoy_chk.SetValue(True)
            if hasattr(self, "allpass_chk"):
                self.allpass_chk.SetValue(True)
            if hasattr(self, "reamping_chk"):
                self.reamping_chk.SetValue(True)
            if hasattr(self, "preamble_chk"):
                self.preamble_chk.SetValue(False)
            self.set_status(tr("STATUS_PRESET_COVER"))
        else:  # Instrumental Mode
            self.pitch_spin.SetValue(2.5)
            self.formant_spin.SetValue(2.0)
            self.tempo_spin.SetValue(0.940)
            self.vocal_chk.SetValue(True)
            if hasattr(self, "articulation_chk"):
                self.articulation_chk.SetValue(False)
            if hasattr(self, "jitter_chk"):
                self.jitter_chk.SetValue(True)
            if hasattr(self, "bode_chk"):
                self.bode_chk.SetValue(True)
            if hasattr(self, "decoy_chk"):
                self.decoy_chk.SetValue(True)
            if hasattr(self, "allpass_chk"):
                self.allpass_chk.SetValue(True)
            if hasattr(self, "reamping_chk"):
                self.reamping_chk.SetValue(True)
            if hasattr(self, "preamble_chk"):
                self.preamble_chk.SetValue(True)
            self.set_status(tr("STATUS_MODE_INSTRUMENTAL"))

    def on_preset_changed(self, event: wx.CommandEvent):
        idx = self.preset_choice.GetSelection()
        is_vocal_cover = (self.mode_choice.GetSelection() == 0) if hasattr(self, "mode_choice") else False
        if idx == 0:  # Complete Sanitization
            self.pitch_spin.SetValue(2.0 if is_vocal_cover else 2.5)
            self.formant_spin.SetValue(2.0)
            self.tempo_spin.SetValue(0.940)
            self.reverb_spin.SetValue(18.0)
            self.vocal_duck_spin.SetValue(12.0)
            self.watermark_chk.SetValue(True)
            self.psycho_chk.SetValue(True)
            self.format_choice.SetSelection(0)
            self.trim_chk.SetValue(False)
            self.jitter_chk.SetValue(True)
            self.bode_chk.SetValue(not is_vocal_cover)
            self.decoy_chk.SetValue(True)
            self.allpass_chk.SetValue(True)
            self.reamping_chk.SetValue(True)
            self.vocal_chk.SetValue(not is_vocal_cover)
            self.articulation_chk.SetValue(is_vocal_cover)
            self.preamble_chk.SetValue(not is_vocal_cover)
        elif idx == 1:  # Balanced Quality
            self.pitch_spin.SetValue(1.5)
            self.formant_spin.SetValue(1.0)
            self.tempo_spin.SetValue(0.970)
            self.reverb_spin.SetValue(12.0)
            self.vocal_duck_spin.SetValue(8.0)
            self.watermark_chk.SetValue(True)
            self.psycho_chk.SetValue(True)
            self.format_choice.SetSelection(0)
            self.trim_chk.SetValue(False)
            self.jitter_chk.SetValue(True)
            self.bode_chk.SetValue(not is_vocal_cover)
            self.decoy_chk.SetValue(False)
            self.allpass_chk.SetValue(True)
            self.reamping_chk.SetValue(True)
            self.vocal_chk.SetValue(not is_vocal_cover)
            self.articulation_chk.SetValue(is_vocal_cover)
            self.preamble_chk.SetValue(False)
        elif idx == 2:  # Room Simulation
            self.pitch_spin.SetValue(1.0)
            self.formant_spin.SetValue(1.0)
            self.tempo_spin.SetValue(0.980)
            self.reverb_spin.SetValue(25.0)
            self.vocal_duck_spin.SetValue(6.0)
            self.jitter_chk.SetValue(False)
            self.bode_chk.SetValue(False)
            self.decoy_chk.SetValue(False)
            self.allpass_chk.SetValue(True)
            self.reamping_chk.SetValue(True)
            self.vocal_chk.SetValue(True)
            self.preamble_chk.SetValue(True)
        elif idx == 3:  # Frequency & Phase Shift
            self.pitch_spin.SetValue(2.0)
            self.formant_spin.SetValue(2.0)
            self.tempo_spin.SetValue(0.950)
            self.reverb_spin.SetValue(10.0)
            self.jitter_chk.SetValue(True)
            self.bode_chk.SetValue(True)
            self.decoy_chk.SetValue(True)
            self.allpass_chk.SetValue(True)
            self.reamping_chk.SetValue(False)
            self.vocal_chk.SetValue(True)
            self.preamble_chk.SetValue(False)
        elif idx == 4:  # Vocal Reduction Only
            self.pitch_spin.SetValue(0.0)
            self.formant_spin.SetValue(0.0)
            self.tempo_spin.SetValue(1.000)
            self.reverb_spin.SetValue(0.0)
            self.vocal_duck_spin.SetValue(18.0)
            self.jitter_chk.SetValue(False)
            self.bode_chk.SetValue(False)
            self.decoy_chk.SetValue(False)
            self.allpass_chk.SetValue(False)
            self.reamping_chk.SetValue(False)
            self.vocal_chk.SetValue(True)
            self.preamble_chk.SetValue(False)
        elif idx == 5:  # Cloud API Stem Isolation
            self.pitch_spin.SetValue(2.0)
            self.formant_spin.SetValue(2.0)
            self.tempo_spin.SetValue(0.950)
            self.reverb_spin.SetValue(15.0)
            self.vocal_duck_spin.SetValue(14.0)
            self.format_choice.SetSelection(0)
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
            wx.MessageBox(
                tr("AUDIO_AUDIT_NEED_PROCESS_MSG"),
                tr("AUDIO_AUDIT_NEED_PROCESS_TITLE"),
                wx.OK | wx.ICON_INFORMATION,
                self
            )
            return

        metrics = benchmark_evasion_metrics(self.selected_file_path, self.last_output_path)
        dlg = EvasionAuditDialog(self, metrics, self.selected_file_path, self.last_output_path)
        dlg.ShowModal()
        dlg.Destroy()

    def on_preflight(self, event: wx.CommandEvent):
        if not self.selected_file_path or not self.last_output_path:
            is_id = (get_language() == "id")
            msg = "Silakan proses audio terlebih dahulu untuk menjalankan simulasi lolos AI." if is_id else "Please process an audio file first to run the AI evasion simulation."
            title = "Belum Ada Berkas Hasil" if is_id else "No Processed Audio File"
            wx.MessageBox(
                msg,
                title,
                wx.OK | wx.ICON_INFORMATION,
                self
            )
            return
        dlg = PreFlightEvasionDialog(self, self.selected_file_path, self.last_output_path, self.processor)
        dlg.ShowModal()
        dlg.Destroy()

    def on_batch(self, event: wx.CommandEvent):
        dlg = BatchProcessingDialog(self, self.processor, self._get_current_options)
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
            message=tr("AUDIO_BROWSE_TITLE"),
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
        self.last_output_path = None
        if hasattr(self, "play_sani_btn"):
            self.play_sani_btn.Enable(False)
        if hasattr(self, "play_delta_btn"):
            self.play_delta_btn.Enable(False)
        if hasattr(self, "audit_btn"):
            self.audit_btn.Enable(False)
        if hasattr(self, "preflight_btn"):
            self.preflight_btn.Enable(False)
        try:
            info = self.processor.inspect_file(file_path)
            self.file_info_text.SetLabel(f"Audio Specifications: {info.summary_text}")
            self.log_ctrl.AppendText(f"Loaded source file: {os.path.basename(file_path)}\n{info.summary_text}\n\n")
            self.set_status(f"Loaded file: {os.path.basename(file_path)} ({info.duration_formatted})")

            # Key & BPM detection & Recommendation
            if hasattr(self, "key_bpm_label"):
                bpm_str = f"{info.bpm:.0f}" if info.bpm else "N/A"
                key_str = info.musical_key or "N/A"
                self.key_bpm_label.SetLabel(tr("AUDIO_KEY_BPM_FORMAT", key=key_str, bpm=bpm_str))
            if hasattr(self, "apply_rec_btn"):
                if info.recommended_transpositions:
                    rec_val, rec_key = info.recommended_transpositions[0]
                    self._recommended_pitch = float(rec_val)
                    self._recommended_key = str(rec_key)
                    rec_display = f"{self._recommended_pitch:+.1f} st ({self._recommended_key})"
                    self.apply_rec_btn.SetLabel(tr("AUDIO_BTN_APPLY_REC", rec=rec_display))
                    self.apply_rec_btn.Enable(True)
                else:
                    self._recommended_pitch = None
                    self._recommended_key = None
                    self.apply_rec_btn.Enable(False)

            # Update visualizer & audition
            if hasattr(self, "visualizer"):
                self.visualizer.set_original_audio(file_path)
                if hasattr(self, "vis_summary_lbl"):
                    self.vis_summary_lbl.SetValue(self.visualizer.get_accessible_summary())
            if hasattr(self, "play_orig_btn"):
                self.play_orig_btn.Enable(True)

            top = self.GetTopLevelParent()
            if top and hasattr(top, "SetTitle"):
                top.SetTitle(f"GhostWave Studio v{APP_VERSION} - [{os.path.basename(file_path)}]")
        except Exception as ex:
            self.file_info_text.SetLabel(f"Audio Specifications: Error reading file ({str(ex)})")
            self.set_status("Error loading audio file.")

    def on_apply_rec_pitch(self, event: wx.CommandEvent):
        if getattr(self, "_recommended_pitch", None) is not None and hasattr(self, "pitch_spin"):
            self.pitch_spin.SetValue(float(self._recommended_pitch))
            msg = f"Applying recommended pitch: {self._recommended_pitch:+.1f} st" if get_language() == "en" else f"Menerapkan transposisi nada rekomendasi: {self._recommended_pitch:+.1f} semitone"
            self.set_status(msg)

    def on_play_original(self, event: Optional[wx.CommandEvent] = None):
        if not self.selected_file_path or not os.path.exists(self.selected_file_path):
            return
        fname = os.path.basename(self.selected_file_path)
        self.play_audio_file(
            self.selected_file_path,
            status_text=tr("AUDIO_BTN_PLAY_ORIG_STATUS", file=fname)
        )

    def on_play_sanitized(self, event: Optional[wx.CommandEvent] = None):
        if not self.last_output_path or not os.path.exists(self.last_output_path):
            return
        fname = os.path.basename(self.last_output_path)
        self.play_audio_file(
            self.last_output_path,
            status_text=tr("AUDIO_BTN_PLAY_SANI_STATUS", file=fname)
        )

    def on_play_delta(self, event: Optional[wx.CommandEvent] = None):
        if not self.selected_file_path or not self.last_output_path:
            return
        try:
            d1, sr1 = load_audio_samples(self.selected_file_path, target_sr=44100)
            d2, sr2 = load_audio_samples(self.last_output_path, target_sr=44100)
            if d1.ndim == 1:
                d1 = np.column_stack([d1, d1])
            if d2.ndim == 1:
                d2 = np.column_stack([d2, d2])
            min_len = min(len(d1), len(d2), int(sr1 * 30))
            delta = (d2[:min_len] - d1[:min_len]) * 2.5
            delta = np.clip(delta, -0.98, 0.98)
            fname = os.path.basename(self.selected_file_path)
            self.play_audio_data(
                delta.astype(np.float32),
                sr=sr1,
                status_text=tr("AUDIO_STATUS_DELTA_PLAYING", file=fname)
            )
        except Exception as ex:
            self.log_ctrl.AppendText(f"Could not generate Delta preview: {ex}\n")
            self.set_status(f"Error previewing Delta: {ex}")

    def on_stop_playback(self, event: Optional[wx.CommandEvent] = None):
        if sd:
            try:
                sd.stop()
            except Exception:
                pass
        if winsound:
            try:
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        self.set_status(tr("AUDIO_STATUS_STOPPED"))

    def play_audio_file(self, path: str, status_text: Optional[str] = None):
        try:
            data, sr = load_audio_samples(path, target_sr=44100)
            max_s = min(len(data), int(sr * 30))
            self.play_audio_data(
                data[:max_s],
                sr=sr,
                status_text=status_text or tr("AUDIO_STATUS_PLAYING", file=os.path.basename(path))
            )
        except Exception as ex:
            self._fallback_play_file(path, status_text or tr("AUDIO_STATUS_PLAYING", file=os.path.basename(path)))

    def play_audio_data(self, data: np.ndarray, sr: int = 44100, status_text: str = ""):
        self.on_stop_playback()
        if data is None or len(data) == 0:
            return

        mx = float(np.max(np.abs(data)))
        if mx > 0.98:
            data = (data / mx) * 0.95
        data = np.ascontiguousarray(data, dtype=np.float32)

        played_successfully = False
        if sd:
            try:
                sd.play(data, sr)
                played_successfully = True
            except Exception:
                played_successfully = False

        if not played_successfully:
            try:
                temp_wav = os.path.join(tempfile.gettempdir(), "ghostwave_audition_temp.wav")
                sf.write(temp_wav, data, sr, subtype='PCM_16')
                if winsound:
                    winsound.PlaySound(temp_wav, winsound.SND_ASYNC | winsound.SND_FILENAME)
                    played_successfully = True
            except Exception as ex:
                self.set_status(tr("AUDIO_STATUS_PLAY_INFO", msg=str(ex)))
                return

        if status_text:
            self.set_status(status_text)

    def _fallback_play_file(self, path: str, status_text: str = ""):
        if not winsound:
            self.set_status(status_text)
            return
        try:
            winsound.PlaySound(None, 0)
            target = path
            if not path.lower().endswith(".wav"):
                d, sr = load_audio_samples(path, target_sr=44100)
                temp_wav = os.path.join(tempfile.gettempdir(), "ghostwave_audition_temp.wav")
                sf.write(temp_wav, d[:int(sr * 15)], sr, subtype='PCM_16')
                target = temp_wav
            winsound.PlaySound(target, winsound.SND_ASYNC | winsound.SND_FILENAME)
            if status_text:
                self.set_status(status_text)
        except Exception as ex:
            self.set_status(tr("AUDIO_STATUS_PLAY_INFO", msg=str(ex)))

    def _get_current_options(self) -> AudioSanitizeOptions:
        platform_keys = ["suno_v6", "google_lyria", "udio", "youtube", "tiktok", "custom"]
        p_idx = self.platform_choice.GetSelection() if hasattr(self, "platform_choice") else 0
        plat_code = platform_keys[p_idx] if 0 <= p_idx < len(platform_keys) else "suno_v6"
        is_lyria = (plat_code == "google_lyria")

        is_vocal_cover = (self.mode_choice.GetSelection() == 0) if hasattr(self, "mode_choice") else True
        mode_val = "vocal_cover" if is_vocal_cover else "instrumental"
        if hasattr(self, "preset_choice") and self.preset_choice.GetSelection() == 5:
            mode_val = "cloud_api"

        fmt_idx = self.format_choice.GetSelection() if hasattr(self, "format_choice") else 0
        bitrate_opt = "wav" if fmt_idx == 0 else ("320k" if fmt_idx == 1 else "192k")

        return AudioSanitizeOptions(
            mode=mode_val,
            target_platform=plat_code,
            pitch_shift_semitones=0.0 if is_lyria else (float(self.pitch_spin.GetValue()) if hasattr(self, "pitch_spin") else 2.0),
            tempo_factor=1.000 if is_lyria else (float(self.tempo_spin.GetValue()) if hasattr(self, "tempo_spin") else 0.940),
            formant_shift_semitones=0.0 if is_lyria else (float(self.formant_spin.GetValue()) if hasattr(self, "formant_spin") else 2.0),
            enable_ai_watermark_neutralizer=self.watermark_chk.IsChecked() if hasattr(self, "watermark_chk") else True,
            use_psychoacoustic_masking=False if is_lyria else (self.psycho_chk.IsChecked() if hasattr(self, "psycho_chk") else True),
            room_reverb_wet=0.0 if is_lyria else ((float(self.reverb_spin.GetValue()) / 100.0) if hasattr(self, "reverb_spin") else 0.18),
            vocal_reduction_depth=0.0 if is_lyria else (float(self.vocal_duck_spin.GetValue()) if hasattr(self, "vocal_duck_spin") else 12.0),
            enable_vocal_articulation_softener=False if is_lyria else (self.articulation_chk.IsChecked() if hasattr(self, "articulation_chk") else is_vocal_cover),
            articulation_intensity=0.80,
            consonant_cutoff_hz=2400.0,
            vocal_guiding_duck_db=-3.5,
            enable_micro_chrono_jitter=False if is_lyria else (self.jitter_chk.IsChecked() if hasattr(self, "jitter_chk") else True),
            enable_bode_freq_shifter=False if is_lyria else (self.bode_chk.IsChecked() if hasattr(self, "bode_chk") else True),
            enable_triple_band_bode=False if is_lyria else (self.bode_chk.IsChecked() and not is_vocal_cover if hasattr(self, "bode_chk") else False),
            enable_adversarial_peaks=False if is_lyria else (self.decoy_chk.IsChecked() if hasattr(self, "decoy_chk") else True),
            enable_phase_dispersion=False if is_lyria else (self.allpass_chk.IsChecked() if hasattr(self, "allpass_chk") else True),
            enable_reamping_room=False if is_lyria else (self.reamping_chk.IsChecked() if hasattr(self, "reamping_chk") else True),
            enable_total_vocal_cut=False if is_lyria else ((not is_vocal_cover) and (self.vocal_chk.IsChecked() if hasattr(self, "vocal_chk") else False)),
            enable_vocal_cut=False if is_lyria else ((self.vocal_chk.IsChecked() if hasattr(self, "vocal_chk") else False) if not is_vocal_cover else False),
            enable_formant_scrambler=False if is_lyria else ((self.vocal_chk.IsChecked() if hasattr(self, "vocal_chk") else False) if not is_vocal_cover else False),
            enable_continuous_bed=False if is_lyria else ((not is_vocal_cover) and (self.preamble_chk.IsChecked() if hasattr(self, "preamble_chk") else False)),
            inject_preamble=False if is_lyria else ((self.preamble_chk.IsChecked() if hasattr(self, "preamble_chk") else False) if not is_vocal_cover else False),
            use_ambient_intro=is_vocal_cover and not is_lyria,
            front_padding_seconds=0.0 if is_lyria else (1.0 if is_vocal_cover else 2.0),
            tail_padding_seconds=0.0 if is_lyria else (1.0 if is_vocal_cover else 1.5),
            apply_eq_filters=False if is_lyria else (self.filter_chk.IsChecked() if hasattr(self, "filter_chk") else True),
            apply_dither=False if is_lyria else (self.dither_chk.IsChecked() if hasattr(self, "dither_chk") else True),
            trim_duration=False if is_lyria else (self.trim_chk.IsChecked() if hasattr(self, "trim_chk") else False),
            max_duration_seconds=float(self.trim_spin.GetValue()) if hasattr(self, "trim_spin") else 24.0,
            strip_metadata=self.metadata_chk.IsChecked() if hasattr(self, "metadata_chk") else True,
            output_bitrate=bitrate_opt
        )

    def on_process(self, event: wx.CommandEvent):
        if not self.selected_file_path or not os.path.exists(self.selected_file_path):
            wx.Bell()
            wx.MessageBox(
                tr("AUDIO_ERR_NO_FILE_MSG"),
                tr("AUDIO_ERR_NO_FILE_TITLE"),
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
        elif fmt_idx == 1:
            default_ext = ".mp3"
            wildcard = "MP3 Audio (*.mp3)|*.mp3"
        else:
            default_ext = ".mp3"
            wildcard = "MP3 Audio (*.mp3)|*.mp3"

        default_out_name = f"{stem}_suno_ready{default_ext}"

        with wx.FileDialog(
            self,
            message=tr("AUDIO_SAVE_TITLE"),
            defaultDir=str(input_p.parent),
            defaultFile=default_out_name,
            wildcard=wildcard,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as save_dialog:
            if save_dialog.ShowModal() == wx.ID_CANCEL:
                return
            output_path = save_dialog.GetPath()

        options = self._get_current_options()
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
            if hasattr(self, "preflight_btn"):
                self.preflight_btn.Enable(True)
            if hasattr(self, "play_sani_btn"):
                self.play_sani_btn.Enable(True)
            if hasattr(self, "play_delta_btn"):
                self.play_delta_btn.Enable(True)
            if hasattr(self, "visualizer"):
                self.visualizer.set_sanitized_audio(result.output_path)
                if hasattr(self, "vis_summary_lbl"):
                    self.vis_summary_lbl.SetValue(self.visualizer.get_accessible_summary())
            self.set_status(f"Export complete: {os.path.basename(result.output_path)}")

            audit_txt = ""
            if result.fingerprint_similarity_pct is not None:
                audit_txt = (
                    f"\n\nAcoustic Verification:\n"
                    f"Chromaprint Similarity: {result.fingerprint_similarity_pct:.1f}%\n"
                    f"Verdict: {result.evasion_verdict or 'Ready for Suno'}"
                )

            wx.MessageBox(
                tr("AUDIO_COMPLETE_MSG", path=result.output_path, duration=result.final_duration, audit=audit_txt),
                tr("AUDIO_COMPLETE_TITLE"),
                wx.OK | wx.ICON_INFORMATION,
                self
            )
        elif result.error_message and "cancelled" in result.error_message.lower():
            self.set_status("Audio processing cancelled by user.")
            self.log_ctrl.AppendText("\nProcess was cancelled by the user. Output discarded.\n")
            wx.MessageBox(
                tr("AUDIO_CANCEL_MSG"),
                tr("AUDIO_CANCEL_TITLE"),
                wx.OK | wx.ICON_INFORMATION,
                self
            )
        else:
            self.set_status("Audio processing encountered an error.")
            wx.MessageBox(
                tr("AUDIO_FAIL_MSG", error=result.error_message or 'Unknown error.'),
                tr("AUDIO_FAIL_TITLE"),
                wx.OK | wx.ICON_ERROR,
                self
            )
        self.process_btn.SetFocus()

    def retranslate_ui(self):
        """Dynamically retranslates UI labels and tooltips in the Audio Sanitizer tab."""
        if hasattr(self, "file_box"):
            self.file_box.SetLabel(tr("AUDIO_GROUP_SOURCE"))
        if hasattr(self, "file_label"):
            self.file_label.SetLabel(tr("AUDIO_LABEL_SELECTED_FILE"))
        if hasattr(self, "browse_btn"):
            self.browse_btn.SetLabel(f"&{tr('AUDIO_SELECT_BTN')}")
        if hasattr(self, "file_path_ctrl") and (not self.selected_file_path or self.file_path_ctrl.GetValue() in ("No audio file selected.", "Belum ada berkas audio yang dipilih.")):
            self.file_path_ctrl.SetValue(tr("AUDIO_NO_FILE_SELECTED"))
        if hasattr(self, "file_info_text") and not self.selected_file_path:
            self.file_info_text.SetLabel(tr("AUDIO_SPECS_NONE"))

        if hasattr(self, "param_box"):
            self.param_box.SetLabel(tr("AUDIO_GROUP_PARAMS"))
        if hasattr(self, "mode_label"):
            self.mode_label.SetLabel(tr("AUDIO_LABEL_MODE"))
        if hasattr(self, "mode_choice"):
            cur_mode = self.mode_choice.GetSelection()
            self.mode_choice.Clear()
            self.mode_choice.Append(tr("AUDIO_MODE_VOCAL_COVER"))
            self.mode_choice.Append(tr("AUDIO_MODE_INSTRUMENTAL"))
            self.mode_choice.SetSelection(cur_mode if cur_mode in (0, 1) else 0)
            self.mode_choice.SetToolTip(tr("TIP_AUDIO_MODE"))
        if hasattr(self, "articulation_chk"):
            self.articulation_chk.SetLabel(tr("AUDIO_ARTICULATION_SOFTENER"))
            self.articulation_chk.SetToolTip(tr("TIP_AUDIO_ARTICULATION"))
        if hasattr(self, "preset_label"):
            self.preset_label.SetLabel(tr("AUDIO_LABEL_PRESET"))
        if hasattr(self, "cloud_btn"):
            self.cloud_btn.SetLabel(tr("AUDIO_BTN_CLOUD_SETTINGS"))
        if hasattr(self, "format_label"):
            self.format_label.SetLabel(tr("AUDIO_LABEL_FORMAT"))

        if hasattr(self, "toggle_adv_btn"):
            if self.adv_panel.IsShown():
                self.toggle_adv_btn.SetLabel(tr("AUDIO_HIDE_ADVANCED_BTN"))
            else:
                self.toggle_adv_btn.SetLabel(tr("AUDIO_SHOW_ADVANCED_BTN"))

        if hasattr(self, "jitter_chk"):
            self.jitter_chk.SetLabel(tr("AUDIO_CHRONO_JITTER"))
        if hasattr(self, "bode_chk"):
            self.bode_chk.SetLabel(tr("AUDIO_BODE_SHIFT"))
        if hasattr(self, "decoy_chk"):
            self.decoy_chk.SetLabel(tr("AUDIO_ADVERSARIAL_PEAKS"))
        if hasattr(self, "allpass_chk"):
            self.allpass_chk.SetLabel(tr("AUDIO_PHASE_DISPERSION"))
        if hasattr(self, "reamping_chk"):
            self.reamping_chk.SetLabel(tr("AUDIO_REAMP_ROOM"))
        if hasattr(self, "vocal_chk"):
            self.vocal_chk.SetLabel(tr("AUDIO_VOCAL_CUT"))
        if hasattr(self, "preamble_chk"):
            self.preamble_chk.SetLabel(tr("AUDIO_INJECT_PREAMBLE"))
        if hasattr(self, "filter_chk"):
            self.filter_chk.SetLabel(tr("AUDIO_EQ_FILTERS"))
        if hasattr(self, "dither_chk"):
            self.dither_chk.SetLabel(tr("AUDIO_DITHER"))
        if hasattr(self, "metadata_chk"):
            self.metadata_chk.SetLabel(tr("AUDIO_STRIP_METADATA"))
        if hasattr(self, "trim_chk"):
            self.trim_chk.SetLabel(tr("AUDIO_TRIM_CLIP"))

        if hasattr(self, "process_btn"):
            self.process_btn.SetLabel(tr("AUDIO_PROCESS_BTN"))
        if hasattr(self, "cancel_btn"):
            self.cancel_btn.SetLabel(tr("AUDIO_CANCEL_BTN"))
        if hasattr(self, "audit_btn"):
            self.audit_btn.SetLabel(tr("AUDIO_BTN_RUN_AUDIT"))

        # Tooltips Retranslation
        if hasattr(self, "file_path_ctrl"):
            self.file_path_ctrl.SetToolTip(tr("TIP_AUDIO_FILE_PATH"))
        if hasattr(self, "browse_btn"):
            self.browse_btn.SetToolTip(tr("TIP_AUDIO_BROWSE_BTN"))
        if hasattr(self, "preset_choice"):
            self.preset_choice.SetToolTip(tr("TIP_AUDIO_PRESET"))
        if hasattr(self, "cloud_btn"):
            self.cloud_btn.SetToolTip(tr("TIP_AUDIO_CLOUD_BTN"))
        if hasattr(self, "toggle_adv_btn"):
            if hasattr(self, "adv_panel") and self.adv_panel.IsShown():
                self.toggle_adv_btn.SetToolTip(tr("TIP_AUDIO_ADV_HIDE"))
            else:
                self.toggle_adv_btn.SetToolTip(tr("TIP_AUDIO_ADV_SHOW"))
        if hasattr(self, "pitch_spin"):
            self.pitch_spin.SetToolTip(tr("TIP_AUDIO_PITCH"))
        if hasattr(self, "tempo_spin"):
            self.tempo_spin.SetToolTip(tr("TIP_AUDIO_TEMPO"))
        if hasattr(self, "jitter_chk"):
            self.jitter_chk.SetToolTip(tr("TIP_AUDIO_JITTER"))
        if hasattr(self, "bode_chk"):
            self.bode_chk.SetToolTip(tr("TIP_AUDIO_BODE"))
        if hasattr(self, "decoy_chk"):
            self.decoy_chk.SetToolTip(tr("TIP_AUDIO_DECOY"))
        if hasattr(self, "allpass_chk"):
            self.allpass_chk.SetToolTip(tr("TIP_AUDIO_ALLPASS"))
        if hasattr(self, "reamping_chk"):
            self.reamping_chk.SetToolTip(tr("TIP_AUDIO_REAMPING"))
        if hasattr(self, "vocal_chk"):
            self.vocal_chk.SetToolTip(tr("TIP_AUDIO_VOCAL"))
        if hasattr(self, "preamble_chk"):
            self.preamble_chk.SetToolTip(tr("TIP_AUDIO_PREAMBLE"))
        if hasattr(self, "filter_chk"):
            self.filter_chk.SetToolTip(tr("TIP_AUDIO_FILTERS"))
        if hasattr(self, "dither_chk"):
            self.dither_chk.SetToolTip(tr("TIP_AUDIO_DITHER"))
        if hasattr(self, "metadata_chk"):
            self.metadata_chk.SetToolTip(tr("TIP_AUDIO_METADATA"))
        if hasattr(self, "trim_chk"):
            self.trim_chk.SetToolTip(tr("TIP_AUDIO_TRIM_CHK"))
        if hasattr(self, "trim_spin"):
            self.trim_spin.SetToolTip(tr("TIP_AUDIO_TRIM_SPIN"))
        if hasattr(self, "process_btn"):
            self.process_btn.SetToolTip(tr("TIP_AUDIO_PROCESS_BTN"))
        if hasattr(self, "cancel_btn"):
            self.cancel_btn.SetToolTip(tr("TIP_AUDIO_CANCEL_BTN"))
        if hasattr(self, "audit_btn"):
            self.audit_btn.SetToolTip(tr("TIP_AUDIO_AUDIT_BTN"))
        if hasattr(self, "slicer_btn"):
            self.slicer_btn.SetToolTip(tr("TIP_AUDIO_SLICER_BTN"))
        if hasattr(self, "protocol_btn"):
            self.protocol_btn.SetToolTip(tr("TIP_AUDIO_PROTOCOL_BTN"))
        if hasattr(self, "log_ctrl"):
            self.log_ctrl.SetToolTip(tr("TIP_AUDIO_LOG"))

        if hasattr(self, "platform_label"):
            self.platform_label.SetLabel(tr("AUDIO_LABEL_PLATFORM"))
        if hasattr(self, "platform_choice"):
            cur_p = self.platform_choice.GetSelection()
            self.platform_choice.Clear()
            self.platform_choice.Append(tr("AUDIO_PLATFORM_SUNO"))
            self.platform_choice.Append(tr("AUDIO_PLATFORM_LYRIA"))
            self.platform_choice.Append(tr("AUDIO_PLATFORM_UDIO"))
            self.platform_choice.Append(tr("AUDIO_PLATFORM_YOUTUBE"))
            self.platform_choice.Append(tr("AUDIO_PLATFORM_TIKTOK"))
            self.platform_choice.Append(tr("AUDIO_PLATFORM_CUSTOM"))
            self.platform_choice.SetSelection(cur_p if 0 <= cur_p < 6 else 0)

        if hasattr(self, "modular_box"):
            self.modular_box.SetLabel(tr("AUDIO_MODULAR_BOX"))
        if hasattr(self, "pitch_lbl"):
            self.pitch_lbl.SetLabel(tr("AUDIO_SLIDER_PITCH"))
        if hasattr(self, "formant_lbl"):
            self.formant_lbl.SetLabel(tr("AUDIO_SLIDER_FORMANT"))
        if hasattr(self, "tempo_lbl"):
            self.tempo_lbl.SetLabel(tr("AUDIO_SLIDER_TEMPO"))
        if hasattr(self, "reverb_lbl"):
            self.reverb_lbl.SetLabel(tr("AUDIO_SLIDER_REVERB"))
        if hasattr(self, "vocal_duck_lbl"):
            self.vocal_duck_lbl.SetLabel(tr("AUDIO_SLIDER_VOCAL"))

        if hasattr(self, "pitch_spin"):
            make_accessible_control(self.pitch_spin, lambda: tr("AUDIO_SLIDER_PITCH"), lambda: tr("TIP_AUDIO_PITCH"))
        if hasattr(self, "formant_spin"):
            make_accessible_control(self.formant_spin, lambda: tr("AUDIO_SLIDER_FORMANT"), lambda: tr("TIP_AUDIO_FORMANT"))
        if hasattr(self, "tempo_spin"):
            make_accessible_control(self.tempo_spin, lambda: tr("AUDIO_SLIDER_TEMPO"), lambda: tr("TIP_AUDIO_TEMPO"))
        if hasattr(self, "reverb_spin"):
            make_accessible_control(self.reverb_spin, lambda: tr("AUDIO_SLIDER_REVERB"), lambda: tr("TIP_AUDIO_REVERB"))
        if hasattr(self, "vocal_duck_spin"):
            make_accessible_control(self.vocal_duck_spin, lambda: tr("AUDIO_SLIDER_VOCAL"), lambda: tr("TIP_AUDIO_VOCAL_DUCK"))

        if hasattr(self, "watermark_chk"):
            self.watermark_chk.SetLabel(tr("AUDIO_CHK_WATERMARK"))
            self.watermark_chk.SetToolTip(tr("TIP_AUDIO_WATERMARK"))
        if hasattr(self, "psycho_chk"):
            self.psycho_chk.SetLabel(tr("AUDIO_CHK_PSYCHO"))
            self.psycho_chk.SetToolTip(tr("TIP_AUDIO_PSYCHO"))

        if hasattr(self, "preflight_btn"):
            self.preflight_btn.SetLabel(tr("AUDIO_BTN_PREFLIGHT"))
            self.preflight_btn.SetToolTip(tr("TIP_PREFLIGHT_BTN"))
        if hasattr(self, "batch_btn"):
            self.batch_btn.SetLabel(tr("AUDIO_BTN_BATCH"))
            self.batch_btn.SetToolTip(tr("TIP_BATCH_BTN"))
        if hasattr(self, "slicer_btn"):
            self.slicer_btn.SetLabel(tr("AUDIO_BTN_SLICER"))
        if hasattr(self, "protocol_btn"):
            self.protocol_btn.SetLabel(tr("AUDIO_BTN_PROTOCOL"))
        if hasattr(self, "seconds_label"):
            self.seconds_label.SetLabel(tr("AUDIO_UNIT_SECONDS"))
        if hasattr(self, "log_label"):
            self.log_label.SetLabel(tr("AUDIO_GROUP_LOGS"))

        if hasattr(self, "vis_box"):
            self.vis_box.SetLabel(tr("AUDIO_VIS_GROUP"))
        if hasattr(self, "vis_summary_lbl") and hasattr(self, "visualizer"):
            summary = self.visualizer.get_accessible_summary()
            self.vis_summary_lbl.SetValue(summary)
            self.visualizer.SetName(summary)

        if hasattr(self, "play_orig_btn"):
            self.play_orig_btn.SetLabel(tr("AUDIO_BTN_PLAY_ORIG"))
            self.play_orig_btn.SetToolTip(tr("AUDIO_BTN_PLAY_ORIG"))
        if hasattr(self, "play_sani_btn"):
            self.play_sani_btn.SetLabel(tr("AUDIO_BTN_PLAY_SANI"))
            self.play_sani_btn.SetToolTip(tr("AUDIO_BTN_PLAY_SANI"))
        if hasattr(self, "play_delta_btn"):
            self.play_delta_btn.SetLabel(tr("AUDIO_BTN_PLAY_DELTA"))
            self.play_delta_btn.SetToolTip(tr("AUDIO_BTN_PLAY_DELTA"))
        if hasattr(self, "stop_play_btn"):
            self.stop_play_btn.SetLabel(tr("AUDIO_BTN_STOP_PLAY"))
            self.stop_play_btn.SetToolTip(tr("AUDIO_BTN_STOP_PLAY"))

        if hasattr(self, "key_bpm_label") and not self.selected_file_path:
            self.key_bpm_label.SetLabel(tr("AUDIO_KEY_BPM_NONE"))
        if hasattr(self, "apply_rec_btn") and getattr(self, "_recommended_pitch", None) is not None:
            rec_display = f"{self._recommended_pitch:+.1f} st ({getattr(self, '_recommended_key', '')})"
            self.apply_rec_btn.SetLabel(tr("AUDIO_BTN_APPLY_REC", rec=rec_display))
        if hasattr(self, "visualizer"):
            self.visualizer.Refresh()


class LyricsSanitizerPanel(scrolled.ScrolledPanel):
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
        self.in_label = wx.StaticText(self, label=tr("LYRICS_INPUT_LABEL"))
        self.in_label.SetName("Original Lyrics Input Label")
        in_header_sizer.Add(self.in_label, 0, wx.ALIGN_CENTER_VERTICAL)
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
        make_accessible_control(self.input_text_ctrl, lambda: tr("LYRICS_INPUT_LABEL"), lambda: tr("TIP_LYRICS_INPUT"))
        self.input_text_ctrl.Bind(wx.EVT_TEXT, self.on_input_text_changed)
        main_sizer.Add(self.input_text_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Section 2: Moderation Filters & Structural Standards
        self.rules_box = wx.StaticBox(self, label=tr("LYRICS_GROUP_INPUT"))
        self.rules_box.SetName("Moderation Rules Group")
        rules_box_sizer = wx.StaticBoxSizer(self.rules_box, wx.VERTICAL)

        celeb_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.celeb_chk = wx.CheckBox(
            self,
            label=tr("LYRICS_CHK_CELEB"),
            name="Strip known artist and celebrity names"
        )
        self.celeb_chk.SetValue(True)
        self.celeb_chk.SetToolTip(tr("TIP_LYRICS_CELEB"))
        celeb_sizer.Add(self.celeb_chk, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 15)

        self.edit_blacklist_btn = wx.Button(
            self,
            label=tr("LYRICS_BTN_EDIT_BLACKLIST"),
            name="Edit Celebrity Blacklist Button"
        )
        self.edit_blacklist_btn.SetToolTip(tr("TIP_LYRICS_BLACKLIST_BTN"))
        self.edit_blacklist_btn.Bind(wx.EVT_BUTTON, self.on_edit_blacklist)
        celeb_sizer.Add(self.edit_blacklist_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        rules_box_sizer.Add(celeb_sizer, 0, wx.ALL, 4)

        self.profanity_chk = wx.CheckBox(
            self,
            label=tr("LYRICS_CHK_PROFANITY"),
            name="Filter profanity and flagged slurs"
        )
        self.profanity_chk.SetValue(True)
        self.profanity_chk.SetToolTip(tr("TIP_LYRICS_PROFANITY"))
        rules_box_sizer.Add(self.profanity_chk, 0, wx.ALL, 4)

        self.unicode_chk = wx.CheckBox(
            self,
            label=tr("LYRICS_CHK_UNICODE"),
            name="Normalize non-standard characters and diacritics to clean UTF-8"
        )
        self.unicode_chk.SetValue(True)
        self.unicode_chk.SetToolTip(tr("TIP_LYRICS_UNICODE"))
        rules_box_sizer.Add(self.unicode_chk, 0, wx.ALL, 4)

        self.tags_chk = wx.CheckBox(
            self,
            label=tr("LYRICS_CHK_TAGS"),
            name="Auto-format Suno structural tags"
        )
        self.tags_chk.SetValue(True)
        self.tags_chk.SetToolTip(tr("TIP_LYRICS_TAGS"))
        rules_box_sizer.Add(self.tags_chk, 0, wx.ALL, 4)

        self.length_chk = wx.CheckBox(
            self,
            label=tr("LYRICS_CHK_LIMITS"),
            name="Character and line length limiter"
        )
        self.length_chk.SetValue(True)
        self.length_chk.SetToolTip(tr("TIP_LYRICS_LENGTH"))
        rules_box_sizer.Add(self.length_chk, 0, wx.ALL, 4)

        main_sizer.Add(rules_box_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Section 3: Copyright Evasion & Lyrics Cloaking Engine
        self.cloak_box = wx.StaticBox(self, label=tr("LYRICS_GROUP_OUTPUT"))
        self.cloak_box.SetName("Lyrics Cloaking Group")
        cloak_box_sizer = wx.StaticBoxSizer(self.cloak_box, wx.VERTICAL)

        self.cloak_chk = wx.CheckBox(
            self,
            label=tr("LYRICS_CHK_CLOAK"),
            name="Enable lyrics variation and spelling filters"
        )
        self.cloak_chk.SetValue(True)
        self.cloak_chk.SetToolTip(tr("TIP_LYRICS_CLOAK"))
        cloak_box_sizer.Add(self.cloak_chk, 0, wx.ALL, 4)

        mode_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.mode_lbl = wx.StaticText(self, label=tr("LYRICS_MODE_LABEL"))
        self.mode_lbl.SetName("Cloaking Strategy Label")
        mode_sizer.Add(self.mode_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.cloak_mode_choice = wx.Choice(
            self,
            choices=[
                "Phonetic Spelling Variation (Preserves Lyrics) [Recommended]",
                "Hybrid Variation (Spelling + Ad-libs)",
                "Sound-Alike Substitution",
                "Synonym Replacement (Preserves Syllables)",
                "Cloud AI Rewriter"
            ],
            name="Lyrics Sanitization Mode Selection"
        )
        self.cloak_mode_choice.SetSelection(0)
        self.cloak_mode_choice.SetToolTip(tr("TIP_LYRICS_MODE"))
        mode_sizer.Add(self.cloak_mode_choice, 1, wx.EXPAND | wx.RIGHT, 10)

        self.cloud_llm_btn = wx.Button(
            self,
            label=tr("LYRICS_BTN_CLOUD_LLM"),
            name="Lyrics Cloud LLM Settings Button"
        )
        self.cloud_llm_btn.SetToolTip(tr("TIP_LYRICS_CLOUD_BTN"))
        self.cloud_llm_btn.Bind(wx.EVT_BUTTON, self.on_cloud_llm_settings)
        mode_sizer.Add(self.cloud_llm_btn, 0)
        cloak_box_sizer.Add(mode_sizer, 0, wx.EXPAND | wx.ALL, 4)

        toggles_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.vibrato_chk = wx.CheckBox(
            self,
            label=tr("LYRICS_ADD_VIBRATO"),
            name="Add vocal vibrato glides to phrase cadences"
        )
        self.vibrato_chk.SetValue(True)
        self.vibrato_chk.SetToolTip(tr("TIP_LYRICS_VIBRATO"))
        toggles_sizer.Add(self.vibrato_chk, 0, wx.RIGHT, 15)

        self.syllable_chk = wx.CheckBox(
            self,
            label=tr("LYRICS_PRESERVE_SYLLABLES"),
            name="Preserve musical syllable meter per line"
        )
        self.syllable_chk.SetValue(True)
        self.syllable_chk.SetToolTip(tr("TIP_LYRICS_SYLLABLE"))
        toggles_sizer.Add(self.syllable_chk, 0, wx.RIGHT, 15)

        self.rhyme_chk = wx.CheckBox(
            self,
            label=tr("LYRICS_PRESERVE_RHYMES"),
            name="Preserve Rhyme Scheme and Cadence"
        )
        self.rhyme_chk.SetValue(True)
        self.rhyme_chk.SetToolTip(tr("TIP_LYRICS_RHYME"))
        toggles_sizer.Add(self.rhyme_chk, 0, wx.RIGHT, 15)

        self.ngram_chk = wx.CheckBox(
            self,
            label=tr("LYRICS_BREAK_NGRAMS"),
            name="Inject rhythmic backing ad-libs"
        )
        self.ngram_chk.SetValue(False)
        self.ngram_chk.SetToolTip(tr("TIP_LYRICS_NGRAM"))
        toggles_sizer.Add(self.ngram_chk, 0)
        cloak_box_sizer.Add(toggles_sizer, 0, wx.ALL, 4)

        main_sizer.Add(cloak_box_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Section 4: Action Buttons
        action_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.sanitize_btn = wx.Button(
            self,
            label=tr("LYRICS_PROCESS_BTN"),
            name="Sanitize Lyrics Button"
        )
        self.sanitize_btn.SetToolTip(tr("TIP_LYRICS_PROCESS_BTN"))
        self.sanitize_btn.Bind(wx.EVT_BUTTON, self.on_sanitize)
        self.sanitize_btn.SetDefault()
        action_sizer.Add(self.sanitize_btn, 0, wx.RIGHT, 10)

        self.prompt_btn = wx.Button(
            self,
            label=tr("LYRICS_BTN_SUNO_PROMPT"),
            name="Generate Suno Style Prompt Button"
        )
        self.prompt_btn.SetToolTip(tr("TIP_LYRICS_PROMPT_BTN"))
        self.prompt_btn.Bind(wx.EVT_BUTTON, self.on_prompt_gen)
        action_sizer.Add(self.prompt_btn, 0, wx.RIGHT, 10)

        self.copy_btn = wx.Button(
            self,
            label=tr("LYRICS_COPY_BTN"),
            name="Copy Sanitized Lyrics to Clipboard Button"
        )
        self.copy_btn.SetToolTip(tr("TIP_LYRICS_COPY_BTN"))
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        action_sizer.Add(self.copy_btn, 0, wx.RIGHT, 10)

        self.sample_btn = wx.Button(
            self,
            label=tr("LYRICS_BTN_LOAD_SAMPLE"),
            name="Load Sample Lyrics Button"
        )
        self.sample_btn.SetToolTip(tr("TIP_LYRICS_SAMPLE_BTN"))
        self.sample_btn.Bind(wx.EVT_BUTTON, self.on_load_sample)
        action_sizer.Add(self.sample_btn, 0, wx.RIGHT, 10)

        self.clear_btn = wx.Button(
            self,
            label=tr("LYRICS_BTN_CLEAR"),
            name="Clear All Lyrics Button"
        )
        self.clear_btn.SetToolTip(tr("TIP_LYRICS_CLEAR_BTN"))
        self.clear_btn.Bind(wx.EVT_BUTTON, self.on_clear)
        action_sizer.Add(self.clear_btn, 0)

        main_sizer.Add(action_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Section 5: Sanitized Output & Changes Made Log
        out_splitter = wx.BoxSizer(wx.VERTICAL)

        self.out_label = wx.StaticText(self, label=tr("LYRICS_OUTPUT_LABEL"))
        self.out_label.SetName("Sanitized Lyrics Output Label")
        out_splitter.Add(self.out_label, 0, wx.BOTTOM, 4)

        self.output_text_ctrl = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE,
            size=(-1, 130),
            name="Sanitized Lyrics Output"
        )
        make_accessible_control(self.output_text_ctrl, lambda: tr("LYRICS_OUTPUT_LABEL"), lambda: tr("TIP_LYRICS_OUTPUT"))
        out_splitter.Add(self.output_text_ctrl, 1, wx.EXPAND | wx.BOTTOM, 8)

        self.changes_label = wx.StaticText(self, label=tr("LYRICS_GROUP_AUDIT"))
        self.changes_label.SetName("Changes Made Label")
        out_splitter.Add(self.changes_label, 0, wx.BOTTOM, 4)

        self.changes_ctrl = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            size=(-1, 120),
            name="Changes Made"
        )
        make_accessible_control(self.changes_ctrl, lambda: tr("LYRICS_GROUP_AUDIT"), lambda: tr("TIP_LYRICS_CHANGES"))
        out_splitter.Add(self.changes_ctrl, 1, wx.EXPAND)

        main_sizer.Add(out_splitter, 2, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.SetSizer(main_sizer)
        self.SetupScrolling(scroll_x=False, scroll_y=True)

    def on_edit_blacklist(self, event: wx.CommandEvent):
        dlg = BlacklistDialog(self, self.processor)
        dlg.ShowModal()
        dlg.Destroy()
        self.set_status(f"Celebrity blacklist updated ({len(self.processor.celebrity_blacklist)} entries).")

    def on_prompt_gen(self, event: wx.CommandEvent):
        bpm = None
        key = None
        top = self.GetTopLevelParent()
        if top and hasattr(top, "audio_tab") and hasattr(top.audio_tab, "processor") and top.audio_tab.selected_file_path:
            try:
                info = top.audio_tab.processor.inspect_file(top.audio_tab.selected_file_path)
                bpm = info.bpm
                key = info.musical_key
            except Exception:
                pass
        dlg = SunoPromptDialog(self, self.input_text_ctrl, detected_bpm=bpm, detected_key=key)
        dlg.ShowModal()
        dlg.Destroy()

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
        self.set_status(tr("LYRICS_CLEAR_STATUS"))
        self.input_text_ctrl.SetFocus()

    def on_sanitize(self, event: wx.CommandEvent):
        raw_text = self.input_text_ctrl.GetValue()
        if not raw_text.strip():
            wx.Bell()
            wx.MessageBox(
                tr("LYRICS_NO_INPUT_MSG"),
                tr("LYRICS_NO_INPUT_TITLE"),
                wx.OK | wx.ICON_WARNING,
                self
            )
            self.input_text_ctrl.SetFocus()
            return

        mode_idx = self.cloak_mode_choice.GetSelection()
        mode_map = ["scramble", "hybrid", "phonetic", "semantic", "cloud"]
        selected_mode = mode_map[mode_idx] if mode_idx < len(mode_map) else "scramble"

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
            preserve_rhymes=self.rhyme_chk.IsChecked() if hasattr(self, "rhyme_chk") else True,
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
            log_sections.append(tr("AUDIT_HEADER_EVASION"))
            log_sections.append(tr("AUDIT_VERDICT_LABEL", verdict=aud.evasion_verdict))
            log_sections.append(tr("AUDIT_NGRAM_OVERLAP", pct=aud.ngram_overlap_pct))
            log_sections.append(tr("AUDIT_TOKEN_SIMILARITY", pct=aud.token_similarity_pct))
            log_sections.append(tr("AUDIT_SYLLABLE_CADENCE", pct=aud.syllable_accuracy_pct))
            log_sections.append(tr("AUDIT_WORD_COUNT_CHANGE", orig=aud.original_words, cloaked=aud.cloaked_words) + "\n")

        log_sections.append(tr("AUDIT_HEADER_TRANSFORM"))
        for chg in result.changes:
            log_sections.append(f"* {chg}")

        if result.warnings:
            log_sections.append(f"\n{tr('AUDIT_HEADER_WARNINGS')}")
            for warn in result.warnings:
                log_sections.append(f"Warning: {warn}")

        if result.stats:
            log_sections.append(f"\n{tr('AUDIT_HEADER_METRICS')}")
            st = result.stats
            log_sections.append(
                tr(
                    "AUDIT_METRICS_FORMAT",
                    chars=st.get("total_characters", 0),
                    lines=st.get("total_lines", 0),
                    words=st.get("word_count", 0),
                    duration=st.get("estimated_duration", "0s")
                )
            )

        self.changes_ctrl.SetValue("\n".join(log_sections))

        change_count = len(result.changes)
        warning_count = len(result.warnings)
        status_msg = tr("LYRICS_STATUS_DONE", changes=change_count, warnings=warning_count)
        self.set_status(status_msg)

        wx.Bell()
        self.output_text_ctrl.SetFocus()

    def on_copy(self, event: wx.CommandEvent):
        sanitized_text = self.output_text_ctrl.GetValue()
        if not sanitized_text.strip():
            wx.Bell()
            wx.MessageBox(
                tr("LYRICS_NO_COPY_MSG"),
                tr("LYRICS_NO_COPY_TITLE"),
                wx.OK | wx.ICON_INFORMATION,
                self
            )
            return

        try:
            if wx.TheClipboard.Open():
                try:
                    wx.TheClipboard.SetData(wx.TextDataObject(sanitized_text))
                    self.set_status(tr("LYRICS_COPIED_MSG"))
                    wx.Bell()
                    wx.MessageBox(
                        tr("LYRICS_COPIED_MSG"),
                        tr("LYRICS_COPIED_TITLE"),
                        wx.OK | wx.ICON_INFORMATION,
                        self
                    )
                finally:
                    wx.TheClipboard.Close()
            else:
                wx.MessageBox(
                    tr("CLIPBOARD_ERR_MSG"),
                    tr("CLIPBOARD_ERR_TITLE"),
                    wx.OK | wx.ICON_ERROR,
                    self
                )
        except Exception as e:
            wx.MessageBox(f"Clipboard error: {e}", tr("CLIPBOARD_ERR_TITLE"), wx.OK | wx.ICON_ERROR, self)

    def retranslate_ui(self):
        """Dynamically retranslates UI labels in the Lyrics Sanitizer tab."""
        if hasattr(self, "in_label"):
            self.in_label.SetLabel(tr("LYRICS_INPUT_LABEL"))
        if hasattr(self, "rules_box"):
            self.rules_box.SetLabel(tr("LYRICS_GROUP_INPUT"))
        if hasattr(self, "sample_btn"):
            self.sample_btn.SetLabel(tr("LYRICS_BTN_LOAD_SAMPLE"))
        if hasattr(self, "clear_btn"):
            self.clear_btn.SetLabel(tr("LYRICS_BTN_CLEAR"))

        if hasattr(self, "celeb_chk"):
            self.celeb_chk.SetLabel(tr("LYRICS_CHK_CELEB"))
        if hasattr(self, "edit_blacklist_btn"):
            self.edit_blacklist_btn.SetLabel(tr("LYRICS_BTN_EDIT_BLACKLIST"))
        if hasattr(self, "profanity_chk"):
            self.profanity_chk.SetLabel(tr("LYRICS_CHK_PROFANITY"))
        if hasattr(self, "unicode_chk"):
            self.unicode_chk.SetLabel(tr("LYRICS_CHK_UNICODE"))
        if hasattr(self, "tags_chk"):
            self.tags_chk.SetLabel(tr("LYRICS_CHK_TAGS"))
        if hasattr(self, "length_chk"):
            self.length_chk.SetLabel(tr("LYRICS_CHK_LIMITS"))

        if hasattr(self, "cloak_box"):
            self.cloak_box.SetLabel(tr("LYRICS_GROUP_OUTPUT"))
        if hasattr(self, "cloak_chk"):
            self.cloak_chk.SetLabel(tr("LYRICS_CHK_CLOAK"))
        if hasattr(self, "mode_lbl"):
            self.mode_lbl.SetLabel(tr("LYRICS_MODE_LABEL"))
        if hasattr(self, "cloud_llm_btn"):
            self.cloud_llm_btn.SetLabel(tr("LYRICS_BTN_CLOUD_LLM"))

        if hasattr(self, "vibrato_chk"):
            self.vibrato_chk.SetLabel(tr("LYRICS_ADD_VIBRATO"))
        if hasattr(self, "syllable_chk"):
            self.syllable_chk.SetLabel(tr("LYRICS_PRESERVE_SYLLABLES"))
        if hasattr(self, "ngram_chk"):
            self.ngram_chk.SetLabel(tr("LYRICS_BREAK_NGRAMS"))

        if hasattr(self, "out_label"):
            self.out_label.SetLabel(tr("LYRICS_OUTPUT_LABEL"))
        if hasattr(self, "changes_label"):
            self.changes_label.SetLabel(tr("LYRICS_GROUP_AUDIT"))

        if hasattr(self, "sanitize_btn"):
            self.sanitize_btn.SetLabel(tr("LYRICS_PROCESS_BTN"))
        if hasattr(self, "copy_btn"):
            self.copy_btn.SetLabel(tr("LYRICS_COPY_BTN"))

        # Tooltips Retranslation
        if hasattr(self, "input_text_ctrl"):
            self.input_text_ctrl.SetToolTip(tr("TIP_LYRICS_INPUT"))
        if hasattr(self, "celeb_chk"):
            self.celeb_chk.SetToolTip(tr("TIP_LYRICS_CELEB"))
        if hasattr(self, "edit_blacklist_btn"):
            self.edit_blacklist_btn.SetToolTip(tr("TIP_LYRICS_BLACKLIST_BTN"))
        if hasattr(self, "profanity_chk"):
            self.profanity_chk.SetToolTip(tr("TIP_LYRICS_PROFANITY"))
        if hasattr(self, "unicode_chk"):
            self.unicode_chk.SetToolTip(tr("TIP_LYRICS_UNICODE"))
        if hasattr(self, "tags_chk"):
            self.tags_chk.SetToolTip(tr("TIP_LYRICS_TAGS"))
        if hasattr(self, "length_chk"):
            self.length_chk.SetToolTip(tr("TIP_LYRICS_LENGTH"))
        if hasattr(self, "cloak_chk"):
            self.cloak_chk.SetToolTip(tr("TIP_LYRICS_CLOAK"))
        if hasattr(self, "cloak_mode_choice"):
            self.cloak_mode_choice.SetToolTip(tr("TIP_LYRICS_MODE"))
        if hasattr(self, "cloud_llm_btn"):
            self.cloud_llm_btn.SetToolTip(tr("TIP_LYRICS_CLOUD_BTN"))
        if hasattr(self, "vibrato_chk"):
            self.vibrato_chk.SetToolTip(tr("TIP_LYRICS_VIBRATO"))
        if hasattr(self, "syllable_chk"):
            self.syllable_chk.SetToolTip(tr("TIP_LYRICS_SYLLABLE"))
        if hasattr(self, "ngram_chk"):
            self.ngram_chk.SetToolTip(tr("TIP_LYRICS_NGRAM"))
        if hasattr(self, "sanitize_btn"):
            self.sanitize_btn.SetToolTip(tr("TIP_LYRICS_PROCESS_BTN"))
        if hasattr(self, "copy_btn"):
            self.copy_btn.SetToolTip(tr("TIP_LYRICS_COPY_BTN"))
        if hasattr(self, "sample_btn"):
            self.sample_btn.SetToolTip(tr("TIP_LYRICS_SAMPLE_BTN"))
        if hasattr(self, "clear_btn"):
            self.clear_btn.SetToolTip(tr("TIP_LYRICS_CLEAR_BTN"))
        if hasattr(self, "output_text_ctrl"):
            self.output_text_ctrl.SetToolTip(tr("TIP_LYRICS_OUTPUT"))
        if hasattr(self, "changes_ctrl"):
            self.changes_ctrl.SetToolTip(tr("TIP_LYRICS_CHANGES"))
        if hasattr(self, "rhyme_chk"):
            self.rhyme_chk.SetLabel(tr("LYRICS_PRESERVE_RHYMES"))
            self.rhyme_chk.SetToolTip(tr("TIP_LYRICS_RHYME"))
        if hasattr(self, "prompt_btn"):
            self.prompt_btn.SetLabel(tr("LYRICS_BTN_SUNO_PROMPT"))
            self.prompt_btn.SetToolTip(tr("TIP_LYRICS_PROMPT_BTN"))

        self.on_input_text_changed(None)


class GhostWaveFrame(wx.Frame):
    """
    Main Application Window for GhostWave Studio.
    """

    def __init__(self):
        super().__init__(
            parent=None,
            id=wx.ID_ANY,
            title=f"GhostWave Studio v{APP_VERSION}",
            size=(900, 760),
            style=wx.DEFAULT_FRAME_STYLE
        )
        self.SetName("GhostWave Studio Main Window")
        self.SetMinSize((760, 620))

        # Core Engines
        self.audio_processor = AudioProcessor()
        self.lyrics_processor = LyricsProcessor()

        # Load active configuration and apply language early
        self.config = GhostWaveConfig.load()
        set_language(getattr(self.config, "language", "en"))

        # Build Status Bar for accessible live announcements
        self.statusbar = self.CreateStatusBar(1, wx.STB_DEFAULT_STYLE)
        self.statusbar.SetName("Application Status Bar")
        self.set_status_text(f"Ready. GhostWave Studio v{APP_VERSION} initialized.")

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

        # Bind close and session end handlers to persist active configuration
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_END_SESSION, self.on_close)
        self.Bind(wx.EVT_QUERY_END_SESSION, self.on_close)

        # Apply active language translations to all menus, labels, and tabs immediately
        self.retranslate_ui()

        # Background update check if enabled in encrypted config vault
        try:
            cfg = GhostWaveConfig.load()
            if getattr(cfg, "check_updates_on_startup", True):
                wx.CallLater(1500, self._check_updates_startup)
        except Exception:
            pass

        # Background fetch and cache changelog so it's always available offline/online
        threading.Thread(target=fetch_and_save_latest_changelog, daemon=True).start()

    def on_close(self, event: wx.CloseEvent):
        """Safely saves active language and UI state into encrypted vault on exit."""
        try:
            if hasattr(self, "audio_tab") and hasattr(self.audio_tab, "on_stop_playback"):
                self.audio_tab.on_stop_playback()
        except Exception:
            pass

        try:
            cfg = GhostWaveConfig.load()
            cfg.language = get_language()
            cfg.first_run = False
            if hasattr(self, "audio_tab") and hasattr(self.audio_tab, "preset_choice"):
                cfg.default_preset = self.audio_tab.preset_choice.GetStringSelection()
                fmt_idx = self.audio_tab.format_choice.GetSelection()
                cfg.export_format = "wav" if fmt_idx == 0 else ("320k" if fmt_idx == 1 else "192k")
            cfg.save()
        except Exception:
            pass
        event.Skip()

    def _build_menu(self):
        self.menu_bar = wx.MenuBar()

        # File Menu
        self.file_menu = wx.Menu()
        self.open_item = self.file_menu.Append(wx.ID_OPEN, tr("MENU_BROWSE_AUDIO"), tr("HELP_BROWSE_AUDIO"))
        self.file_menu.AppendSeparator()
        self.export_profile_item = self.file_menu.Append(
            wx.ID_ANY,
            tr("MENU_EXPORT_PROFILE"),
            tr("HELP_EXPORT_PROFILE")
        )
        self.import_profile_item = self.file_menu.Append(
            wx.ID_ANY,
            tr("MENU_IMPORT_PROFILE"),
            tr("HELP_IMPORT_PROFILE")
        )
        self.file_menu.AppendSeparator()
        self.exit_item = self.file_menu.Append(wx.ID_EXIT, tr("MENU_EXIT"), tr("HELP_EXIT"))
        self.menu_bar.Append(self.file_menu, tr("MENU_FILE"))

        # Edit Menu
        self.edit_menu = wx.Menu()
        self.blacklist_item = self.edit_menu.Append(wx.ID_ANY, tr("MENU_BLACKLIST"), tr("HELP_BLACKLIST"))
        self.copy_item = self.edit_menu.Append(wx.ID_COPY, tr("MENU_COPY_LYRICS"), tr("HELP_COPY_LYRICS"))
        self.menu_bar.Append(self.edit_menu, tr("MENU_EDIT"))

        # Actions Menu
        self.action_menu = wx.Menu()
        self.process_audio_item = self.action_menu.Append(wx.ID_ANY, tr("MENU_PROCESS_AUDIO"), tr("HELP_PROCESS_AUDIO"))
        self.preflight_item = self.action_menu.Append(wx.ID_ANY, tr("MENU_PREFLIGHT"), tr("HELP_PREFLIGHT"))
        self.batch_item = self.action_menu.Append(wx.ID_ANY, f"{tr('AUDIO_BTN_BATCH')}\tCtrl+Shift+B", tr("HELP_BATCH"))
        self.audit_item = self.action_menu.Append(wx.ID_ANY, tr("MENU_AUDIT"), tr("HELP_AUDIT"))
        self.slicer_item = self.action_menu.Append(wx.ID_ANY, tr("MENU_SLICER"), tr("HELP_SLICER"))
        self.protocol_item = self.action_menu.Append(wx.ID_ANY, tr("MENU_PROTOCOL"), tr("HELP_PROTOCOL"))
        self.cloud_item = self.action_menu.Append(wx.ID_ANY, tr("MENU_CLOUD_SETTINGS"), tr("HELP_CLOUD_SETTINGS"))
        self.action_menu.AppendSeparator()
        self.sanitize_lyrics_item = self.action_menu.Append(wx.ID_ANY, tr("MENU_SANITIZE_LYRICS"), tr("HELP_SANITIZE_LYRICS"))
        self.prompt_item = self.action_menu.Append(wx.ID_ANY, tr("LYRICS_BTN_SUNO_PROMPT"), tr("HELP_PROMPT_GEN"))
        self.menu_bar.Append(self.action_menu, tr("MENU_ACTIONS"))

        # Language Menu
        self.lang_menu = wx.Menu()
        self.lang_en_item = self.lang_menu.AppendRadioItem(wx.ID_ANY, tr("MENU_LANG_EN"), tr("HELP_LANG_EN"))
        self.lang_id_item = self.lang_menu.AppendRadioItem(wx.ID_ANY, tr("MENU_LANG_ID"), tr("HELP_LANG_ID"))
        current_lang = get_language()
        if current_lang == "id":
            self.lang_id_item.Check(True)
        else:
            self.lang_en_item.Check(True)
        self.menu_bar.Append(self.lang_menu, tr("MENU_LANGUAGE"))

        # Help Menu
        self.help_menu = wx.Menu()
        self.changelog_item = self.help_menu.Append(wx.ID_ANY, tr("MENU_CHANGELOG"), tr("HELP_CHANGELOG"))
        self.protocol_help_item = self.help_menu.Append(wx.ID_ANY, tr("MENU_PROTOCOL_HELP"), tr("HELP_PROTOCOL_HELP"))
        self.ticket_item = self.help_menu.Append(wx.ID_ANY, tr("MENU_SUPPORT_TICKETS"), tr("HELP_SUPPORT_TICKETS"))
        self.check_update_item = self.help_menu.Append(wx.ID_ANY, tr("MENU_CHECK_UPDATES"), tr("HELP_CHECK_UPDATES"))
        self.a11y_item = self.help_menu.Append(wx.ID_HELP, tr("MENU_ACCESSIBILITY_GUIDE"), tr("HELP_ACCESSIBILITY_GUIDE"))
        self.about_item = self.help_menu.Append(wx.ID_ABOUT, f"{tr('MENU_ABOUT')} v{APP_VERSION}", tr("HELP_ABOUT"))
        self.menu_bar.Append(self.help_menu, tr("MENU_HELP"))

        self.SetMenuBar(self.menu_bar)

        self.tab_audio_id = wx.NewIdRef()
        self.tab_lyrics_id = wx.NewIdRef()
        accel_entries = [
            wx.AcceleratorEntry(wx.ACCEL_NORMAL, wx.WXK_F2, self.changelog_item.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_ALT, ord('F'), self.preflight_item.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord('B'), self.batch_item.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_ALT, ord('1'), self.tab_audio_id),
            wx.AcceleratorEntry(wx.ACCEL_ALT, ord('2'), self.tab_lyrics_id),
        ]
        self.SetAcceleratorTable(wx.AcceleratorTable(accel_entries))
        self.Bind(wx.EVT_MENU, lambda e: self.notebook.SetSelection(0), id=self.tab_audio_id)
        self.Bind(wx.EVT_MENU, lambda e: self.notebook.SetSelection(1), id=self.tab_lyrics_id)

        # Bind Menu Events
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_browse(e), self.open_item)
        self.Bind(wx.EVT_MENU, self.on_export_profile, self.export_profile_item)
        self.Bind(wx.EVT_MENU, self.on_import_profile, self.import_profile_item)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(True), self.exit_item)
        self.Bind(wx.EVT_MENU, lambda e: self.lyrics_tab.on_edit_blacklist(e), self.blacklist_item)
        self.Bind(wx.EVT_MENU, lambda e: self.lyrics_tab.on_copy(e), self.copy_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_process(e), self.process_audio_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_preflight(e), self.preflight_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_batch(e), self.batch_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_audit(e), self.audit_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_slicer(e), self.slicer_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_protocol(e), self.protocol_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_protocol(e), self.protocol_help_item)
        self.Bind(wx.EVT_MENU, lambda e: self.on_change_language("en"), self.lang_en_item)
        self.Bind(wx.EVT_MENU, lambda e: self.on_change_language("id"), self.lang_id_item)
        self.Bind(wx.EVT_MENU, self.on_view_changelog, self.changelog_item)
        self.Bind(wx.EVT_MENU, self.on_support_tickets, self.ticket_item)
        self.Bind(wx.EVT_MENU, self.on_check_updates_menu, self.check_update_item)
        self.Bind(wx.EVT_MENU, lambda e: self.audio_tab.on_cloud_settings(e), self.cloud_item)
        self.Bind(wx.EVT_MENU, lambda e: self.lyrics_tab.on_sanitize(e), self.sanitize_lyrics_item)
        self.Bind(wx.EVT_MENU, lambda e: self.lyrics_tab.on_prompt_gen(e), self.prompt_item)
        self.Bind(wx.EVT_MENU, self.on_accessibility_guide, self.a11y_item)
        self.Bind(wx.EVT_MENU, self.on_about, self.about_item)

    def on_view_changelog(self, event: Optional[wx.CommandEvent] = None):
        """Displays the What's New / Changelog dialog."""
        dlg = ChangelogDialog(self)
        dlg.ShowModal()
        dlg.Destroy()

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
        self.set_status_text(tr("STATUS_LANG_CHANGED", lang=lang_code.upper()))

    def retranslate_ui(self):
        if hasattr(self, "audio_tab") and getattr(self.audio_tab, "selected_file_path", None):
            fn = os.path.basename(self.audio_tab.selected_file_path)
            self.SetTitle(f"GhostWave Studio v{APP_VERSION} - [{fn}]")
        else:
            self.SetTitle(f"GhostWave Studio v{APP_VERSION}")

        self.menu_bar.SetMenuLabel(0, tr("MENU_FILE"))
        self.menu_bar.SetMenuLabel(1, tr("MENU_EDIT"))
        self.menu_bar.SetMenuLabel(2, tr("MENU_ACTIONS"))
        self.menu_bar.SetMenuLabel(3, tr("MENU_LANGUAGE"))
        self.menu_bar.SetMenuLabel(4, tr("MENU_HELP"))

        self.open_item.SetItemLabel(tr("MENU_BROWSE_AUDIO"))
        self.open_item.SetHelp(tr("HELP_BROWSE_AUDIO"))
        self.export_profile_item.SetItemLabel(tr("MENU_EXPORT_PROFILE"))
        self.export_profile_item.SetHelp(tr("HELP_EXPORT_PROFILE"))
        self.import_profile_item.SetItemLabel(tr("MENU_IMPORT_PROFILE"))
        self.import_profile_item.SetHelp(tr("HELP_IMPORT_PROFILE"))
        self.exit_item.SetItemLabel(tr("MENU_EXIT"))
        self.exit_item.SetHelp(tr("HELP_EXIT"))

        self.blacklist_item.SetItemLabel(tr("MENU_BLACKLIST"))
        self.blacklist_item.SetHelp(tr("HELP_BLACKLIST"))
        self.copy_item.SetItemLabel(tr("MENU_COPY_LYRICS"))
        self.copy_item.SetHelp(tr("HELP_COPY_LYRICS"))

        self.process_audio_item.SetItemLabel(tr("MENU_PROCESS_AUDIO"))
        self.process_audio_item.SetHelp(tr("HELP_PROCESS_AUDIO"))
        if hasattr(self, "preflight_item"):
            self.preflight_item.SetItemLabel(tr("MENU_PREFLIGHT"))
            self.preflight_item.SetHelp(tr("HELP_PREFLIGHT"))
        if hasattr(self, "batch_item"):
            self.batch_item.SetItemLabel(f"{tr('AUDIO_BTN_BATCH')}\tCtrl+Shift+B")
            self.batch_item.SetHelp(tr("HELP_BATCH"))
        self.audit_item.SetItemLabel(tr("MENU_AUDIT"))
        self.audit_item.SetHelp(tr("HELP_AUDIT"))
        self.slicer_item.SetItemLabel(tr("MENU_SLICER"))
        self.slicer_item.SetHelp(tr("HELP_SLICER"))
        self.protocol_item.SetItemLabel(tr("MENU_PROTOCOL"))
        self.protocol_item.SetHelp(tr("HELP_PROTOCOL"))
        self.cloud_item.SetItemLabel(tr("MENU_CLOUD_SETTINGS"))
        self.cloud_item.SetHelp(tr("HELP_CLOUD_SETTINGS"))
        self.sanitize_lyrics_item.SetItemLabel(tr("MENU_SANITIZE_LYRICS"))
        self.sanitize_lyrics_item.SetHelp(tr("HELP_SANITIZE_LYRICS"))
        if hasattr(self, "prompt_item"):
            self.prompt_item.SetItemLabel(tr("LYRICS_BTN_SUNO_PROMPT"))
            self.prompt_item.SetHelp(tr("HELP_PROMPT_GEN"))

        self.lang_en_item.SetItemLabel(tr("MENU_LANG_EN"))
        self.lang_en_item.SetHelp(tr("HELP_LANG_EN"))
        self.lang_id_item.SetItemLabel(tr("MENU_LANG_ID"))
        self.lang_id_item.SetHelp(tr("HELP_LANG_ID"))
        current_lang = get_language()
        if current_lang == "id":
            self.lang_id_item.Check(True)
        else:
            self.lang_en_item.Check(True)

        if hasattr(self, "changelog_item"):
            self.changelog_item.SetItemLabel(tr("MENU_CHANGELOG"))
            self.changelog_item.SetHelp(tr("HELP_CHANGELOG"))
        self.protocol_help_item.SetItemLabel(tr("MENU_PROTOCOL_HELP"))
        self.protocol_help_item.SetHelp(tr("HELP_PROTOCOL_HELP"))
        self.ticket_item.SetItemLabel(tr("MENU_SUPPORT_TICKETS"))
        self.ticket_item.SetHelp(tr("HELP_SUPPORT_TICKETS"))
        self.check_update_item.SetItemLabel(tr("MENU_CHECK_UPDATES"))
        self.check_update_item.SetHelp(tr("HELP_CHECK_UPDATES"))
        self.a11y_item.SetItemLabel(tr("MENU_ACCESSIBILITY_GUIDE"))
        self.a11y_item.SetHelp(tr("HELP_ACCESSIBILITY_GUIDE"))
        self.about_item.SetItemLabel(f"{tr('MENU_ABOUT')} v{APP_VERSION}")
        self.about_item.SetHelp(tr("HELP_ABOUT"))

        self.notebook.SetPageText(0, tr("TAB_AUDIO"))
        self.notebook.SetPageText(1, tr("TAB_LYRICS"))

        if hasattr(self.audio_tab, "retranslate_ui"):
            self.audio_tab.retranslate_ui()
        if hasattr(self.lyrics_tab, "retranslate_ui"):
            self.lyrics_tab.retranslate_ui()

    def on_export_profile(self, event: wx.CommandEvent):
        """Exports all configurations to a user-specified .sn profile file."""
        with wx.FileDialog(
            self,
            message=tr("PROFILE_EXPORT_TITLE"),
            defaultFile=DEFAULT_SN_FILENAME,
            wildcard=tr("PROFILE_WILDCARD"),
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                cfg = GhostWaveConfig.load()
                # Synchronize current UI controls
                cfg.default_preset = self.audio_tab.preset_choice.GetStringSelection()
                platform_keys = ["suno_v6", "google_lyria", "udio", "youtube", "tiktok", "custom"]
                p_idx = self.audio_tab.platform_choice.GetSelection()
                cfg.target_platform = platform_keys[p_idx] if 0 <= p_idx < len(platform_keys) else "suno_v6"
                cfg.pitch_shift_semitones = float(self.audio_tab.pitch_spin.GetValue())
                cfg.formant_shift_semitones = float(self.audio_tab.formant_spin.GetValue())
                cfg.tempo_factor = float(self.audio_tab.tempo_spin.GetValue())
                cfg.room_reverb_wet = float(self.audio_tab.reverb_spin.GetValue()) / 100.0
                cfg.vocal_reduction_depth = float(self.audio_tab.vocal_duck_spin.GetValue())
                cfg.enable_ai_watermark_neutralizer = bool(self.audio_tab.watermark_chk.IsChecked())
                cfg.use_psychoacoustic_masking = bool(self.audio_tab.psycho_chk.IsChecked())
                cfg.max_duration_seconds = float(self.audio_tab.trim_spin.GetValue())
                fmt_idx = self.audio_tab.format_choice.GetSelection()
                cfg.export_format = "wav" if fmt_idx == 0 else ("320k" if fmt_idx == 1 else "192k")
                saved_path = cfg.save(path)
                fn = os.path.basename(saved_path)
                self.set_status_text(tr("PROFILE_EXPORT_STATUS", name=fn))
                wx.Bell()
                wx.MessageBox(
                    tr("PROFILE_EXPORT_SUCCESS_MSG", path=saved_path),
                    tr("PROFILE_EXPORT_SUCCESS_TITLE"),
                    wx.OK | wx.ICON_INFORMATION,
                    self
                )

    def on_import_profile(self, event: wx.CommandEvent):
        """Imports a user-specified .sn configuration profile."""
        with wx.FileDialog(
            self,
            message=tr("PROFILE_IMPORT_TITLE"),
            wildcard=tr("PROFILE_WILDCARD"),
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
                    platform_keys = ["suno_v6", "google_lyria", "udio", "youtube", "tiktok", "custom"]
                    if cfg.target_platform in platform_keys:
                        self.audio_tab.platform_choice.SetSelection(platform_keys.index(cfg.target_platform))
                    self.audio_tab.pitch_spin.SetValue(cfg.pitch_shift_semitones)
                    self.audio_tab.formant_spin.SetValue(cfg.formant_shift_semitones)
                    self.audio_tab.tempo_spin.SetValue(cfg.tempo_factor)
                    self.audio_tab.reverb_spin.SetValue(round(cfg.room_reverb_wet * 100.0, 1))
                    self.audio_tab.vocal_duck_spin.SetValue(cfg.vocal_reduction_depth)
                    self.audio_tab.watermark_chk.SetValue(bool(cfg.enable_ai_watermark_neutralizer))
                    self.audio_tab.psycho_chk.SetValue(bool(cfg.use_psychoacoustic_masking))
                    self.audio_tab.trim_spin.SetValue(cfg.max_duration_seconds)
                    fmt_idx = 0 if cfg.export_format == "wav" else (1 if cfg.export_format == "320k" else 2)
                    self.audio_tab.format_choice.SetSelection(fmt_idx)
                    fn = os.path.basename(path)
                    self.set_status_text(tr("PROFILE_IMPORT_STATUS", name=fn))
                    wx.Bell()
                    wx.MessageBox(
                        tr("PROFILE_IMPORT_SUCCESS_MSG", path=path, version=cfg.app_version, preset=cfg.default_preset),
                        tr("PROFILE_IMPORT_SUCCESS_TITLE"),
                        wx.OK | wx.ICON_INFORMATION,
                        self
                    )
                except Exception as ex:
                    wx.MessageBox(
                        tr("PROFILE_IMPORT_ERR_MSG", error=str(ex)),
                        tr("PROFILE_IMPORT_ERR_TITLE"),
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
        """Silently queries for updates in the background on startup."""
        check_updates_background(self, silent=True, current_version=APP_VERSION)

    def on_check_updates_menu(self, event: wx.CommandEvent):
        """Manually checks for application updates."""
        self.set_status_text("Checking for application updates...")

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

    try:
        cfg = GhostWaveConfig.load()
    except Exception:
        cfg = GhostWaveConfig()

    saved_lang = getattr(cfg, "language", "en")
    init_translations(saved_lang)

    # First run check: prompt user to pick preferred language
    if getattr(cfg, "first_run", False):
        dlg = LanguageSelectionDialog(None, current_lang=saved_lang)
        if dlg.ShowModal() == wx.ID_OK:
            selected_lang = dlg.get_selected_language()
            cfg.language = selected_lang
            cfg.first_run = False
            cfg.save()
            set_language(selected_lang)
        else:
            cfg.first_run = False
            cfg.save()
        dlg.Destroy()

    frame = GhostWaveFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
