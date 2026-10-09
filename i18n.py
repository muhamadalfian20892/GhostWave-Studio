"""
Internationalization (I18N) and Dynamic Localization Engine for GhostWave Studio.

Provides real-time runtime language switching, fallback localization catalog,
and accessible UI string resolution for English and Bahasa Indonesia.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_LANGUAGES: List[Tuple[str, str]] = [
    ("en", "English"),
    ("id", "Bahasa Indonesia")
]

DEFAULT_LANGUAGE = "en"
_CURRENT_LANGUAGE = DEFAULT_LANGUAGE
_LOADED_CATALOGS: Dict[str, Dict[str, str]] = {}


def _get_locales_dir() -> Path:
    """Resolves the directory containing language catalog JSON files."""
    if hasattr(sys, "_MEIPASS"):
        # Running inside PyInstaller bundled bundle
        base = Path(sys._MEIPASS)
    else:
        # Running directly from source tree
        base = Path(__file__).resolve().parent

    locales_path = base / "locales"
    if locales_path.is_dir():
        return locales_path
    
    # Fallback to local directory
    return Path(os.getcwd()) / "locales"


def _load_catalog(lang_code: str) -> Dict[str, str]:
    """Loads a language catalog from its JSON file with error handling."""
    locales_dir = _get_locales_dir()
    catalog_file = locales_dir / f"{lang_code}.json"
    
    if catalog_file.is_file():
        try:
            with open(catalog_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass

    return {}


def init_translations(default_lang: str = "en") -> None:
    """Initializes in-memory translation catalogs."""
    global _CURRENT_LANGUAGE, _LOADED_CATALOGS
    _CURRENT_LANGUAGE = default_lang if default_lang in dict(SUPPORTED_LANGUAGES) else DEFAULT_LANGUAGE
    
    for code, _ in SUPPORTED_LANGUAGES:
        _LOADED_CATALOGS[code] = _load_catalog(code)


def set_language(lang_code: str) -> None:
    """Switches the active application language in real-time."""
    global _CURRENT_LANGUAGE
    if lang_code in dict(SUPPORTED_LANGUAGES):
        _CURRENT_LANGUAGE = lang_code
        if lang_code not in _LOADED_CATALOGS or not _LOADED_CATALOGS[lang_code]:
            _LOADED_CATALOGS[lang_code] = _load_catalog(lang_code)


def get_language() -> str:
    """Returns the currently active language code."""
    return _CURRENT_LANGUAGE


def get_supported_languages() -> List[Tuple[str, str]]:
    """Returns list of supported language code and name tuples."""
    return list(SUPPORTED_LANGUAGES)


def tr(key: str, **kwargs: Any) -> str:
    """
    Translates a key into the active language with English fallback and variable formatting.
    """
    global _CURRENT_LANGUAGE, _LOADED_CATALOGS

    # Ensure current language catalog is loaded
    if _CURRENT_LANGUAGE not in _LOADED_CATALOGS:
        _LOADED_CATALOGS[_CURRENT_LANGUAGE] = _load_catalog(_CURRENT_LANGUAGE)

    catalog = _LOADED_CATALOGS.get(_CURRENT_LANGUAGE, {})
    text = catalog.get(key)

    # Fallback to English if key missing in current language
    if text is None:
        if "en" not in _LOADED_CATALOGS:
            _LOADED_CATALOGS["en"] = _load_catalog("en")
        text = _LOADED_CATALOGS.get("en", {}).get(key, key)

    if kwargs and text:
        try:
            return text.format(**kwargs)
        except Exception:
            return text

    return text or key


_t = tr
