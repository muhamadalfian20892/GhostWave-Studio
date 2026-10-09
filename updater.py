"""
GitHub-Driven Automatic Update Engine and Changelog Viewer for GhostWave Studio.

Provides non-blocking update checks against the official GitHub repository,
accessible update notifications, inline changelog viewing in the same window,
and direct streaming downloads for installer and portable packages.
Designed with an Accessibility-First approach: all controls have explicit
accessible names, mnemonics, and screen reader labels.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import wx


APP_VERSION = "1.2.0"
GITHUB_REPO = "muhamadalfian20892/GhostWave-Studio"
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RAW_CHANGELOG = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/changelog.txt"
DEFAULT_USER_AGENT = "GhostWave-Studio-Updater/1.2"


def normalize_version(v_str: str) -> tuple[int, ...]:
    """
    Normalizes semantic version strings into comparable integer tuples.
    Handles prefixes such as 'v1.2.0', suffixes, and variable component counts.
    """
    clean = v_str.strip().lstrip("vV")
    base = re.split(r"[-+]", clean)[0]
    parts: list[int] = []
    for item in base.split("."):
        digits = re.sub(r"\D", "", item)
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def compare_versions(local_ver: str, remote_ver: str) -> int:
    """
    Compares two version strings.
    Returns:
       1 if remote_ver is newer than local_ver (update available)
       0 if versions are equal
      -1 if remote_ver is older than local_ver
    """
    loc = normalize_version(local_ver)
    rem = normalize_version(remote_ver)
    if rem > loc:
        return 1
    if rem < loc:
        return -1
    return 0


@dataclass
class UpdateInfo:
    """Encapsulates parsed release metadata and download coordinates."""
    has_update: bool = False
    current_version: str = APP_VERSION
    latest_version: str = APP_VERSION
    release_name: str = ""
    release_notes: str = ""
    html_url: str = ""
    download_url: str = ""
    asset_name: str = ""
    asset_size: int = 0
    error_message: Optional[str] = None


def parse_changelog_text(raw_text: str) -> dict[str, str]:
    """
    Parses version headers and change items from changelog.txt.
    Extracts version, release date, download link, and release notes.
    """
    lines = [line.rstrip() for line in raw_text.splitlines()]
    version = ""
    date = ""
    installer_url = ""
    portable_url = ""
    notes_lines: list[str] = []
    in_notes = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("version:"):
            version = stripped.split(":", 1)[1].strip()
        elif lower.startswith("date:") or lower.startswith("release date:"):
            date = stripped.split(":", 1)[1].strip()
        elif lower.startswith("installer:"):
            installer_url = stripped.split(":", 1)[1].strip()
        elif lower.startswith("portable:"):
            portable_url = stripped.split(":", 1)[1].strip()
        elif lower.startswith("download:"):
            if not installer_url:
                installer_url = stripped.split(":", 1)[1].strip()
        elif (
            lower.startswith("what's new:")
            or lower.startswith("what is new:")
            or lower.startswith("changelog:")
            or lower.startswith("changes:")
        ):
            in_notes = True
            notes_lines.append(stripped)
        elif in_notes:
            notes_lines.append(line)

    notes = "\n".join(notes_lines).strip()
    if not notes:
        notes = raw_text.strip()

    return {
        "version": version,
        "date": date,
        "installer_url": installer_url,
        "portable_url": portable_url,
        "notes": notes,
    }


def fetch_raw_changelog(timeout: float = 6.0) -> Optional[dict[str, str]]:
    """Fetches and parses changelog.txt directly from the GitHub repository."""
    try:
        req = urllib.request.Request(
            GITHUB_RAW_CHANGELOG,
            headers={"User-Agent": DEFAULT_USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_text = resp.read().decode("utf-8", errors="replace")
            return parse_changelog_text(raw_text)
    except Exception:
        return None


def check_for_updates(
    current_version: str = APP_VERSION,
    timeout: float = 6.0
) -> UpdateInfo:
    """
    Checks GitHub for newer releases.
    Queries GitHub Releases API first; falls back to raw changelog.txt if needed.
    """
    info = UpdateInfo(current_version=current_version)

    # 1. Attempt GitHub Releases API lookup
    api_success = False
    try:
        req = urllib.request.Request(
            GITHUB_API_LATEST,
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "application/vnd.github.v3+json",
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tag_name = data.get("tag_name", "").strip()
            clean_tag = tag_name.lstrip("vV")
            info.latest_version = clean_tag
            info.release_name = data.get("name", f"GhostWave Studio {tag_name}")
            info.release_notes = data.get("body", "").strip()
            info.html_url = data.get("html_url", f"https://github.com/{GITHUB_REPO}/releases")

            # Discover downloadable setup binary or portable archive
            assets = data.get("assets", [])
            for asset in assets:
                name = asset.get("name", "")
                download_url = asset.get("browser_download_url", "")
                if name.lower().endswith(".exe"):
                    info.asset_name = name
                    info.download_url = download_url
                    info.asset_size = asset.get("size", 0)
                    break
                if name.lower().endswith(".zip") and not info.download_url:
                    info.asset_name = name
                    info.download_url = download_url
                    info.asset_size = asset.get("size", 0)

            # If release body is sparse, supplement with raw changelog.txt
            if len(info.release_notes) < 30 or "commits" in info.release_notes:
                raw_cl = fetch_raw_changelog(timeout=timeout)
                if raw_cl and raw_cl.get("notes"):
                    info.release_notes = raw_cl["notes"]

            cmp = compare_versions(current_version, clean_tag)
            info.has_update = (cmp > 0)
            api_success = True
    except urllib.error.HTTPError as http_err:
        # HTTP 403 or 404 might indicate rate limiting or pending release
        info.error_message = f"GitHub API error: {http_err.code} {http_err.reason}"
    except Exception as exc:
        info.error_message = f"Connection failed: {str(exc)}"

    # 2. Fallback to raw changelog.txt if API call failed
    if not api_success:
        raw_cl = fetch_raw_changelog(timeout=timeout)
        if raw_cl and raw_cl.get("version"):
            rem_ver = raw_cl["version"]
            info.latest_version = rem_ver
            info.release_name = f"GhostWave Studio v{rem_ver}"
            info.release_notes = raw_cl.get("notes", "")
            info.download_url = raw_cl.get("installer_url", "")
            info.asset_name = os.path.basename(info.download_url) if info.download_url else ""
            info.html_url = f"https://github.com/{GITHUB_REPO}/releases"
            cmp = compare_versions(current_version, rem_ver)
            info.has_update = (cmp > 0)
            info.error_message = None

    return info


class UpdateDialog(wx.Dialog):
    """
    Accessible dialog informing the user of an available application update.
    Features:
    - Clear prompt statement with version numbers.
    - Expandable inline changelog viewer in the same window.
    - Labeled controls throughout for complete screen reader accessibility.
    - Persistent 'Download Later' and 'Download Now' action buttons.
    """

    def __init__(self, parent: Optional[wx.Window], update_info: UpdateInfo):
        super().__init__(
            parent,
            title="Software Update - GhostWave Studio",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(560, 240)
        )
        self.SetName("Software Update Dialog")
        self.info = update_info
        self.changelog_visible = False

        self._build_ui()
        self.CentreOnParent()

    def _build_ui(self):
        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Software Update Dialog Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Header Question Prompt
        prompt_text = f"Version {self.info.latest_version} is available. Download this update?"
        self.prompt_label = wx.StaticText(panel, label=prompt_text)
        self.prompt_label.SetName("Update Question Prompt")
        font = self.prompt_label.GetFont()
        font.SetPointSize(font.GetPointSize() + 2)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.prompt_label.SetFont(font)
        main_sizer.Add(self.prompt_label, 0, wx.ALL, 12)

        # Version Context Label
        context_text = f"Currently installed version: {self.info.current_version}"
        self.context_label = wx.StaticText(panel, label=context_text)
        self.context_label.SetName("Current Installed Version Information")
        main_sizer.Add(self.context_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        # Inline Changelog Panel (Initially Hidden)
        self.changelog_panel = wx.Panel(panel, style=wx.TAB_TRAVERSAL)
        self.changelog_panel.SetName("Changelog Container Panel")
        changelog_sizer = wx.BoxSizer(wx.VERTICAL)

        self.changelog_header_label = wx.StaticText(
            self.changelog_panel,
            label="&Changelog and release notes:"
        )
        self.changelog_header_label.SetName("Changelog and Release Notes Label")
        changelog_sizer.Add(self.changelog_header_label, 0, wx.BOTTOM, 6)

        initial_notes = self.info.release_notes if self.info.release_notes else "No detailed changelog provided."
        self.changelog_ctrl = wx.TextCtrl(
            self.changelog_panel,
            value=initial_notes,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP | wx.BORDER_THEME,
            name="Changelog Text Area"
        )
        self.changelog_ctrl.SetName("Changelog Text Area")
        self.changelog_ctrl.SetToolTip("Read-only view of changes and release notes for this update.")
        changelog_sizer.Add(self.changelog_ctrl, 1, wx.EXPAND)

        self.changelog_panel.SetSizer(changelog_sizer)
        self.changelog_panel.Hide()
        main_sizer.Add(self.changelog_panel, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        # Action Buttons Sizer
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.see_new_btn = wx.Button(panel, label="&See What's New in This Changes")
        self.see_new_btn.SetName("See What's New in This Changes Button")
        self.see_new_btn.SetToolTip("View detailed changes and release notes for this update.")
        self.see_new_btn.Bind(wx.EVT_BUTTON, self.on_toggle_changelog)
        btn_sizer.Add(self.see_new_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        btn_sizer.AddStretchSpacer()

        self.later_btn = wx.Button(panel, wx.ID_CANCEL, label="Download &Later")
        self.later_btn.SetName("Download Later Button")
        self.later_btn.SetToolTip("Postpone downloading this update.")
        btn_sizer.Add(self.later_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        self.download_btn = wx.Button(panel, wx.ID_OK, label="&Download Now")
        self.download_btn.SetName("Download Now Button")
        self.download_btn.SetToolTip("Download and install this update now.")
        self.download_btn.SetDefault()
        self.download_btn.Bind(wx.EVT_BUTTON, self.on_download_now)
        btn_sizer.Add(self.download_btn, 0, wx.ALIGN_CENTER_VERTICAL)

        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        panel.SetSizer(main_sizer)

        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dialog_sizer)

    def on_toggle_changelog(self, event: wx.CommandEvent):
        """Toggles inline changelog display within the current window."""
        if not self.changelog_visible:
            self.changelog_visible = True
            self.changelog_panel.Show(True)
            self.see_new_btn.SetLabel("&Hide What's New")
            self.see_new_btn.SetName("Hide What's New Button")
            self.SetSize((580, 520))
            self.Layout()
            self.changelog_ctrl.SetFocus()
        else:
            self.changelog_visible = False
            self.changelog_panel.Show(False)
            self.see_new_btn.SetLabel("&See What's New in This Changes")
            self.see_new_btn.SetName("See What's New in This Changes Button")
            self.SetSize((560, 240))
            self.Layout()
            self.see_new_btn.SetFocus()

    def on_download_now(self, event: wx.CommandEvent):
        """Initiates download process or opens release page."""
        self.EndModal(wx.ID_OK)


class UpdateDownloadDialog(wx.Dialog):
    """
    Accessible progress dialog for downloading release packages from GitHub.
    Runs chunked network transfers on a background worker thread.
    """

    def __init__(self, parent: Optional[wx.Window], download_url: str, filename: str, version: str):
        super().__init__(
            parent,
            title="Downloading GhostWave Studio Update",
            style=wx.DEFAULT_DIALOG_STYLE,
            size=(500, 200)
        )
        self.SetName("Download Update Progress Dialog")
        self.download_url = download_url
        self.filename = filename if filename else f"GhostWaveStudio-v{version}-Setup.exe"
        self.version = version

        self.cancelled = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.downloaded_path: Optional[str] = None
        self.error_message: Optional[str] = None

        self._build_ui()
        self.CentreOnParent()

        self.Bind(wx.EVT_CLOSE, self.on_close)
        wx.CallAfter(self.start_download)

    def _build_ui(self):
        panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
        panel.SetName("Download Progress Panel")

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        self.status_label = wx.StaticText(
            panel,
            label=f"Downloading GhostWave Studio v{self.version} from GitHub..."
        )
        self.status_label.SetName("Download Status Description")
        main_sizer.Add(self.status_label, 0, wx.ALL, 12)

        self.gauge = wx.Gauge(panel, range=100, size=(-1, 24))
        self.gauge.SetName("Download Progress Meter")
        self.gauge.SetToolTip("Visual gauge displaying download completion percentage.")
        main_sizer.Add(self.gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 12)

        self.progress_label = wx.StaticText(
            panel,
            label="Connecting to GitHub release servers..."
        )
        self.progress_label.SetName("Download Progress Details Label")
        main_sizer.Add(self.progress_label, 0, wx.ALL, 12)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddStretchSpacer()

        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, label="&Cancel")
        self.cancel_btn.SetName("Cancel Download Button")
        self.cancel_btn.SetToolTip("Cancel the active download.")
        self.cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)
        btn_sizer.Add(self.cancel_btn, 0)

        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        panel.SetSizer(main_sizer)

        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(dialog_sizer)

    def start_download(self):
        self.worker_thread = threading.Thread(target=self._download_worker, daemon=True)
        self.worker_thread.start()

    def _download_worker(self):
        # Determine destination folder (Downloads folder if available, else temp)
        downloads_dir = Path.home() / "Downloads"
        if not downloads_dir.exists():
            downloads_dir = Path(os.environ.get("TEMP", "."))

        target_file = downloads_dir / self.filename
        part_file = downloads_dir / f"{self.filename}.part"

        try:
            req = urllib.request.Request(
                self.download_url,
                headers={"User-Agent": DEFAULT_USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=30.0) as response:
                total_size_header = response.headers.get("Content-Length")
                total_size = int(total_size_header) if total_size_header else 0

                downloaded = 0
                chunk_size = 65536

                with open(part_file, "wb") as f_out:
                    while True:
                        if self.cancelled.is_set():
                            break
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        downloaded += len(chunk)

                        pct = int((downloaded / total_size) * 100) if total_size > 0 else 0
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)

                        def _update_ui(p=pct, d=mb_done, t=mb_total, tot=total_size):
                            if self and self.gauge:
                                self.gauge.SetValue(min(100, p))
                                if tot > 0:
                                    self.progress_label.SetLabel(f"Downloaded {d:.1f} MB of {t:.1f} MB ({p}%)")
                                else:
                                    self.progress_label.SetLabel(f"Downloaded {d:.1f} MB")

                        wx.CallAfter(_update_ui)

            if self.cancelled.is_set():
                if part_file.exists():
                    part_file.unlink(missing_ok=True)
                return

            if target_file.exists():
                target_file.unlink(missing_ok=True)
            part_file.rename(target_file)
            self.downloaded_path = str(target_file)

            wx.CallAfter(self._on_download_complete)

        except Exception as exc:
            if part_file.exists():
                part_file.unlink(missing_ok=True)
            self.error_message = str(exc)
            wx.CallAfter(self._on_download_failed)

    def _on_download_complete(self):
        self.status_label.SetLabel("Download completed successfully!")
        self.progress_label.SetLabel("File saved to disk.")
        self.gauge.SetValue(100)
        self.EndModal(wx.ID_OK)

    def _on_download_failed(self):
        err = self.error_message or "Unknown download error."
        wx.MessageBox(
            f"Failed to download update from GitHub:\n\n{err}",
            "Download Error",
            wx.OK | wx.ICON_ERROR,
            self
        )
        self.EndModal(wx.ID_CANCEL)

    def on_cancel(self, event: wx.CommandEvent):
        self.cancelled.set()
        self.EndModal(wx.ID_CANCEL)

    def on_close(self, event: wx.CloseEvent):
        self.cancelled.set()
        event.Skip()


def run_update_flow(parent: Optional[wx.Window], info: UpdateInfo):
    """
    Executes the interactive update user experience:
    1. Shows UpdateDialog.
    2. If user clicks Download Now, either initiates direct download or opens GitHub.
    3. Offers to launch the downloaded installer package upon completion.
    """
    dlg = UpdateDialog(parent, info)
    res = dlg.ShowModal()
    dlg.Destroy()

    if res == wx.ID_OK:
        if info.download_url:
            dl_dlg = UpdateDownloadDialog(parent, info.download_url, info.asset_name, info.latest_version)
            dl_res = dl_dlg.ShowModal()
            saved_path = dl_dlg.downloaded_path
            dl_dlg.Destroy()

            if dl_res == wx.ID_OK and saved_path and os.path.exists(saved_path):
                # Prompt to launch installer
                msg = (
                    f"Update package downloaded successfully!\n\n"
                    f"File: {saved_path}\n\n"
                    f"Do you want to run the installer now?\n"
                    f"(GhostWave Studio will close to allow installation.)"
                )
                launch_dlg = wx.MessageDialog(
                    parent,
                    msg,
                    "Run Update Installer",
                    wx.YES_NO | wx.ICON_QUESTION
                )
                if launch_dlg.ShowModal() == wx.ID_YES:
                    launch_dlg.Destroy()
                    try:
                        if sys.platform == "win32":
                            os.startfile(saved_path)
                        else:
                            import subprocess
                            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", saved_path])
                        if parent:
                            parent.Close(force=True)
                    except Exception as ex:
                        wx.MessageBox(
                            f"Could not automatically launch installer:\n\n{str(ex)}",
                            "Launch Error",
                            wx.OK | wx.ICON_ERROR,
                            parent
                        )
                else:
                    launch_dlg.Destroy()
        else:
            # Fallback to browser release page
            target_url = info.html_url or f"https://github.com/{GITHUB_REPO}/releases"
            webbrowser.open(target_url)


def check_updates_background(
    parent: Optional[wx.Window],
    silent: bool = True,
    current_version: str = APP_VERSION,
    on_finish: Optional[Callable[[UpdateInfo], None]] = None
):
    """
    Checks for updates on a background thread.
    If silent is True, notifications appear only when an update is available.
    If silent is False (manual check), reports current status when up to date.
    """
    def _worker():
        info = check_for_updates(current_version=current_version)

        def _handle_result():
            if on_finish:
                on_finish(info)
            if info.has_update:
                run_update_flow(parent, info)
            elif not silent:
                if info.error_message:
                    wx.MessageBox(
                        f"Could not connect to GitHub to check for updates.\n\n"
                        f"Error: {info.error_message}\n\n"
                        f"Please check your internet connection.",
                        "Update Check Failed",
                        wx.OK | wx.ICON_WARNING,
                        parent
                    )
                else:
                    wx.MessageBox(
                        f"You are running the latest version of GhostWave Studio (v{current_version}).\n\n"
                        f"No updates are currently available on GitHub.",
                        "GhostWave Studio Up to Date",
                        wx.OK | wx.ICON_INFORMATION,
                        parent
                    )

        wx.CallAfter(_handle_result)

    threading.Thread(target=_worker, daemon=True).start()
