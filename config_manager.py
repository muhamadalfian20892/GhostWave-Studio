"""
GhostWave Studio v1.0 - Encrypted Configuration & Profile Vault (.sn)

Provides military-grade encrypted persistence for all application settings,
Cloud API tokens, DSP presets, and user moderation blacklists.
Saved exclusively as an encrypted binary profile container (.sn).
Only GhostWave Studio can decrypt and parse this configuration.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# File format signature: 16-byte magic identifier
MAGIC_HEADER = b"GHOSTWAVE_SN_V1\x00"

# Application internal key derivation constants
_APP_SALT = b"GhostWave_Studio_v1.0_Acoustic_Cloak_Vault_Salt_2026"
_APP_SECRET = b"GhostWave_Core_Internal_Master_Cipher_Key_v1.0_Protected"

# Default paths
DEFAULT_SN_FILENAME = "ghostwave.sn"
DEFAULT_USER_SN_PATH = os.path.join(os.path.expanduser("~"), f".{DEFAULT_SN_FILENAME}")
LOCAL_SN_PATH = os.path.join(os.getcwd(), DEFAULT_SN_FILENAME)


def _get_fernet_cipher() -> Fernet:
    """Derives an application-specific Fernet key using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_APP_SALT,
        iterations=100_000
    )
    derived_key = base64.urlsafe_b64encode(kdf.derive(_APP_SECRET))
    return Fernet(derived_key)


@dataclass
class GhostWaveConfig:
    """
    Centralized configuration data model for GhostWave Studio v1.0.
    Stores all sensitive API tokens, DSP engine parameters, and user preferences.
    """
    # Metadata
    app_name: str = "GhostWave Studio"
    app_version: str = "1.6.0"
    check_updates_on_startup: bool = True
    language: str = "en"
    first_run: bool = True
    user_tickets: list[dict[str, Any]] = field(default_factory=list)

    # Cloud Stem API Credentials (Encrypted)
    cloud_provider: str = "replicate"             # "replicate", "huggingface", or "custom"
    cloud_api_token: str = ""                     # Secret API Token
    cloud_endpoint_url: str = ""                  # Custom endpoint URL or HF Space
    cloud_model_version: str = "cjwbw/demucs"     # Demucs model version
    cloud_timeout_seconds: int = 180              # Remote job timeout in seconds

    # Audio DSP & Sanitization Preferences
    default_preset: str = "Complete Sanitization (Recommended)"
    export_format: str = "wav"                    # "wav", "320k", "192k"
    pitch_shift_semitones: float = 2.5
    tempo_factor: float = 0.940
    max_duration_seconds: float = 24.0
    slicer_chunk_duration: float = 22.0
    bode_shift_hz: float = 8.5

    # DSP Flags
    enable_micro_chrono_jitter: bool = True
    enable_bode_freq_shifter: bool = True
    enable_adversarial_peaks: bool = True
    enable_phase_dispersion: bool = True
    enable_reamping_room: bool = True
    enable_vocal_cut: bool = True
    enable_formant_scrambler: bool = True
    inject_preamble: bool = True
    apply_eq_filters: bool = True
    apply_dither: bool = True
    trim_duration: bool = False
    strip_metadata: bool = True

    # Lyrics Moderation, Cloaking & Custom Blacklist
    custom_celebrity_blacklist: list[str] = field(default_factory=list)
    filter_profanity: bool = True
    strip_celebrities: bool = True
    normalize_unicode: bool = True
    format_tags: bool = True
    check_limits: bool = True
    enable_lyrics_cloaking: bool = True
    lyrics_cloak_mode: str = "scramble"           # "scramble", "syllable", "phonetic", "semantic", "cloud"
    lyrics_preserve_syllables: bool = True
    lyrics_add_vibrato_glides: bool = True
    lyrics_break_ngrams: bool = False
    lyrics_adlib_frequency: float = 0.5           # frequency of adlib insertions (0.0 to 1.0)
    lyrics_cloud_provider: str = "groq"           # "groq", "openrouter", "openai", "custom"
    lyrics_cloud_api_token: str = ""              # LLM API Token
    lyrics_cloud_endpoint_url: str = ""           # Custom endpoint or base URL
    lyrics_cloud_model: str = "llama-3.3-70b-versatile"

    def to_dict(self) -> dict[str, Any]:
        """Converts configuration dataclass to dictionary."""
        return asdict(self)

    def add_user_ticket(self, ticket: dict[str, Any]) -> None:
        """Appends a new support ticket record to the user's local ticket vault."""
        if not hasattr(self, "user_tickets") or self.user_tickets is None:
            self.user_tickets = []
        # Prepend so newest is first
        self.user_tickets.insert(0, ticket)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GhostWaveConfig":
        """Reconstructs configuration dataclass from dictionary with safe fallbacks."""
        cfg = cls()
        for key, val in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    def to_encrypted_bytes(self) -> bytes:
        """
        Serializes configuration to JSON and encrypts using AES-128-CBC + HMAC-SHA256 (Fernet).
        Prepends the proprietary 16-byte magic identifier.
        """
        cipher = _get_fernet_cipher()
        json_bytes = json.dumps(self.to_dict(), ensure_ascii=False, indent=None).encode("utf-8")
        encrypted_payload = cipher.encrypt(json_bytes)
        return MAGIC_HEADER + encrypted_payload

    @classmethod
    def from_encrypted_bytes(cls, raw_bytes: bytes) -> "GhostWaveConfig":
        """
        Validates magic header, decrypts payload, and parses JSON into a GhostWaveConfig instance.
        Raises ValueError if file signature is invalid or decryption fails.
        """
        if not raw_bytes.startswith(MAGIC_HEADER):
            raise ValueError("Invalid .sn file format: Missing GhostWave Studio cryptographic signature.")

        cipher = _get_fernet_cipher()
        encrypted_payload = raw_bytes[len(MAGIC_HEADER):]

        try:
            decrypted_json_bytes = cipher.decrypt(encrypted_payload)
        except InvalidToken as ex:
            raise ValueError("Decryption failed: Profile has been tampered with or corrupted.") from ex

        data = json.loads(decrypted_json_bytes.decode("utf-8"))
        return cls.from_dict(data)

    def save(self, file_path: Optional[str] = None) -> str:
        """
        Encrypts and writes the configuration to a .sn file.
        Creates a .bak backup copy of the prior valid vault before replacing it.
        Returns the absolute path to the written file.
        """
        target_path = resolve_config_path(file_path, for_writing=True)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        raw_bytes = self.to_encrypted_bytes()

        # If previous target exists, preserve a backup
        if os.path.exists(target_path):
            try:
                shutil.copy2(target_path, f"{target_path}.bak")
            except Exception:
                pass

        temp_target = f"{target_path}.tmp"
        with open(temp_target, "wb") as f:
            f.write(raw_bytes)
            f.flush()
            os.fsync(f.fileno())

        shutil.move(temp_target, target_path)
        return target_path

    @classmethod
    def load(cls, file_path: Optional[str] = None) -> "GhostWaveConfig":
        """
        Locates and decrypts the .sn file.
        Recovers from .bak backup file if primary file reading encounters corruption.
        If no .sn file is found, checks for legacy JSON configurations and migrates them automatically.
        """
        target_path = resolve_config_path(file_path, for_writing=False)
        if os.path.exists(target_path):
            try:
                with open(target_path, "rb") as f:
                    raw_bytes = f.read()
                return cls.from_encrypted_bytes(raw_bytes)
            except Exception:
                # If primary vault is corrupted, attempt backup recovery
                bak_path = f"{target_path}.bak"
                if os.path.exists(bak_path):
                    try:
                        with open(bak_path, "rb") as bf:
                            bak_bytes = bf.read()
                        return cls.from_encrypted_bytes(bak_bytes)
                    except Exception:
                        pass
                pass

        # Try automatic migration from legacy unencrypted configs
        cfg = cls._migrate_legacy_configs()
        # Save newly migrated config into encrypted .sn vault
        try:
            cfg.save(target_path)
        except Exception:
            pass
        return cfg

    @classmethod
    def _migrate_legacy_configs(cls) -> "GhostWaveConfig":
        """Migrates unencrypted legacy JSON files into a fresh GhostWaveConfig."""
        cfg = cls()

        # Legacy 1: ~/.suno_sanitizer_cloud_config.json
        legacy_cloud = os.path.join(os.path.expanduser("~"), ".suno_sanitizer_cloud_config.json")
        if os.path.exists(legacy_cloud):
            try:
                with open(legacy_cloud, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                cfg.cloud_provider = cdata.get("provider", "replicate")
                cfg.cloud_api_token = cdata.get("api_token", "")
                cfg.cloud_endpoint_url = cdata.get("endpoint_url", "")
                cfg.cloud_timeout_seconds = int(cdata.get("timeout_seconds", 180))
            except Exception:
                pass

        # Legacy 2: celebrity_blacklist.json
        app_dir = Path(__file__).resolve().parent
        legacy_bl = app_dir / "celebrity_blacklist.json"
        if legacy_bl.exists():
            try:
                with open(legacy_bl, "r", encoding="utf-8") as f:
                    bdata = json.load(f)
                if isinstance(bdata, list):
                    cfg.custom_celebrity_blacklist = bdata
            except Exception:
                pass

        return cfg


def resolve_config_path(explicit_path: Optional[str] = None, for_writing: bool = False) -> str:
    """
    Resolves the active .sn configuration path.
    Priority:
    1. Explicitly provided path.
    2. Local directory 'ghostwave.sn' if it exists.
    3. User home directory '~/.ghostwave.sn'.
    """
    if explicit_path:
        p = os.path.abspath(explicit_path)
        if not p.endswith(".sn"):
            p = f"{p}.sn"
        return p

    if os.path.exists(LOCAL_SN_PATH):
        return LOCAL_SN_PATH

    return DEFAULT_USER_SN_PATH
