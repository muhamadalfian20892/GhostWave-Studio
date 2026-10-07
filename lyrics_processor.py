"""
Lyrics Processor Module for Suno Prep & Sanitizer.

Provides moderation compliance, celebrity name stripping, profanity filtering
with rhythmic homophones/mild alternatives, Unicode normalization, Suno
structural tag standardization, and line/token length validation.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Default comprehensive celebrity and artist blacklist known to trigger Suno/Udio moderation
DEFAULT_CELEBRITY_BLACKLIST: list[str] = [
    # Top Pop / Hip-hop / R&B
    "Taylor Swift", "Ariana Grande", "Drake", "Eminem", "Kanye West", "Kanye", "Ye",
    "Beyoncé", "Beyonce", "Billie Eilish", "Ed Sheeran", "Justin Bieber", "Post Malone",
    "Travis Scott", "Kendrick Lamar", "Rihanna", "Michael Jackson", "The Weeknd",
    "Bruno Mars", "Lady Gaga", "Dua Lipa", "Olivia Rodrigo", "Selena Gomez",
    "Cardi B", "Nicki Minaj", "Megan Thee Stallion", "Doja Cat", "Ice Spice",
    "Snoop Dogg", "Dr. Dre", "Jay-Z", "Jay Z", "Lil Wayne", "Lil Nas X", "Future",
    "2Pac", "Tupac", "Tupac Shakur", "Notorious B.I.G.", "Biggie", "Biggie Smalls",
    "50 Cent", "Chris Brown", "Usher", "The Kid LAROI", "Jack Harlow", "Machine Gun Kelly",
    "Playboi Carti", "XXXTentacion", "Juice WRLD", "Mac Miller", "A$AP Rocky",
    "Tyler, The Creator", "J. Cole", "Logic", "Macklemore", "Pitbull",
    # Rock / Legends
    "Elvis Presley", "Elvis", "Beatles", "The Beatles", "John Lennon", "Paul McCartney",
    "Freddie Mercury", "Queen", "Kurt Cobain", "Nirvana", "Bob Dylan", "David Bowie",
    "Mick Jagger", "Rolling Stones", "The Rolling Stones", "Led Zeppelin", "Jimmy Page",
    "Robert Plant", "Jimi Hendrix", "Prince", "Madonna", "Stevie Wonder", "Elton John",
    "Frank Sinatra", "Johnny Cash", "Bob Marley", "Metallica", "AC/DC", "Pink Floyd",
    "Guns N' Roses", "Axl Rose", "Slash", "Ozzy Osbourne", "Black Sabbath", "Linkin Park",
    "Chester Bennington", "Green Day", "Billie Joe Armstrong", "Red Hot Chili Peppers",
    "Coldplay", "Chris Martin", "U2", "Bono", "Radiohead", "Thom Yorke",
    # Modern Alt / Indie / EDM
    "Lana Del Rey", "Hozier", "Charli XCX", "Chappell Roan", "Sabrina Carpenter",
    "Harry Styles", "Zayn Malik", "Adele", "Sam Smith", "Sia", "Shawn Mendes",
    "Avicii", "Calvin Harris", "David Guetta", "Skrillex", "Marshmello", "Martin Garrix",
    "Daft Punk", "Deadmau5", "The Chainsmokers", "Kygo",
    # Country / Global
    "Morgan Wallen", "Luke Combs", "Zach Bryan", "Dolly Parton", "Taylor",
    "Bad Bunny", "J Balvin", "Daddy Yankee", "Rosalía", "Rosalia", "Peso Pluma",
    "BTS", "Blackpink", "Jungkook", "Lisa", "Jennie", "J-Hope", "Suga"
]

# Dictionary of profanities and slurs with musical, safe homophones and mild alternatives
PROFANITY_HOMOPHONES: dict[str, str] = {
    # Severe Slurs & Hate Speech (Neutralized / Redacted cleanly)
    "nigger": "brother",
    "niggers": "brothers",
    "nigga": "homie",
    "niggas": "homies",
    "faggot": "fool",
    "faggots": "fools",
    "fag": "poser",
    "dyke": "rebel",
    "kike": "stranger",
    "chink": "rival",
    "spic": "shadow",
    "retard": "rookie",
    "retarded": "reckless",

    # F-word variations
    "motherfucker": "troublemaker",
    "motherfuckers": "troublemakers",
    "motherfuckin": "crazy wild",
    "motherfucking": "crazy wild",
    "fuck": "freak",
    "fucks": "freaks",
    "fucking": "freaking",
    "fuckin": "freakin",
    "fuckin'": "freakin'",
    "fucked": "messed",
    "fucker": "trickster",
    "fuckers": "tricksters",

    # S-word variations
    "bullshit": "nonsense",
    "horseshit": "nonsense",
    "dipshit": "hothead",
    "shit": "spit",
    "shits": "spits",
    "shitty": "gritty",
    "shitting": "spitting",

    # B-word variations
    "bitch": "witch",
    "bitches": "witches",
    "bitching": "grumbling",
    "bitchin": "grumblin",
    "bitchin'": "grumblin'",
    "bitchy": "snappy",

    # A-word variations
    "badass": "fearless",
    "dumbass": "foolish",
    "jackass": "wild card",
    "smartass": "clever",
    "asshole": "jerk",
    "assholes": "jerks",
    "half-ass": "halfway",
    "half-assed": "halfway",
    "ass": "back",
    "asses": "backs",

    # D-word / C-word / Gender variations
    "goddamn": "gosh darn",
    "goddamnit": "gosh darn it",
    "goddamned": "darned",
    "damn": "darn",
    "damned": "darned",
    "dammit": "darn it",
    "hell": "heck",
    "crap": "scrap",
    "dick": "stick",
    "dicks": "sticks",
    "cock": "rock",
    "cocks": "rocks",
    "pussy": "softie",
    "pussies": "softies",
    "cunt": "foe",
    "cunts": "foes",
    "whore": "chaser",
    "whores": "chasers",
    "slut": "player",
    "sluts": "players",
    "bastard": "rascal",
    "bastards": "rascals",
    "piss": "steam",
    "pissed": "steamed",
    "pissing": "steaming",
    "slutty": "flashy",
    "hoe": "crew",
    "hoes": "crew",
    "thot": "stranger",
    "thots": "strangers"
}

# Recognized Suno Structural Tag keywords
RECOGNIZED_SUNO_TAGS: list[str] = [
    "Verse", "Chorus", "Pre-Chorus", "Post-Chorus", "Hook", "Bridge",
    "Intro", "Outro", "Drop", "Beat Drop", "Build-up", "Instrumental",
    "Guitar Solo", "Synth Solo", "Piano Solo", "Solo", "Breakdown",
    "Interlude", "Refrain", "Fade Out", "Spoken Word", "Ending"
]


@dataclass
class LyricsSanitizeResult:
    """Stores the outcome of the lyrics sanitization process."""
    original_text: str
    sanitized_text: str
    changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class LyricsProcessor:
    """
    Sanitizes song lyrics for Suno AI compatibility and moderation compliance.
    """

    def __init__(self, custom_blacklist_path: Optional[str] = None):
        self.blacklist_path = custom_blacklist_path or self._get_default_blacklist_path()
        self.celebrity_blacklist = self.load_blacklist()

    def _get_default_blacklist_path(self) -> str:
        app_dir = Path(os.environ.get("APPDATA", ".")) / "SunoPrepSanitizer"
        app_dir.mkdir(parents=True, exist_ok=True)
        return str(app_dir / "celebrity_blacklist.json")

    def load_blacklist(self) -> list[str]:
        """Loads blacklist from encrypted .sn vault, JSON file, or defaults."""
        try:
            from config_manager import GhostWaveConfig
            sn_cfg = GhostWaveConfig.load()
            if sn_cfg.custom_celebrity_blacklist:
                return sorted(list(set(sn_cfg.custom_celebrity_blacklist)), key=lambda x: (-len(x), x.lower()))
        except Exception:
            pass

        if os.path.exists(self.blacklist_path):
            try:
                with open(self.blacklist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return sorted(list(set(data)), key=lambda x: (-len(x), x.lower()))
            except Exception:
                pass
        return sorted(list(set(DEFAULT_CELEBRITY_BLACKLIST)), key=lambda x: (-len(x), x.lower()))

    def save_blacklist(self, names: list[str]) -> bool:
        """Saves custom blacklist to encrypted .sn vault and local cache."""
        try:
            cleaned = sorted(list({name.strip() for name in names if name.strip()}),
                             key=lambda x: (-len(x), x.lower()))
            self.celebrity_blacklist = cleaned

            try:
                from config_manager import GhostWaveConfig
                sn_cfg = GhostWaveConfig.load()
                sn_cfg.custom_celebrity_blacklist = cleaned
                sn_cfg.save()
            except Exception:
                pass

            with open(self.blacklist_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    @staticmethod
    def match_case(original: str, replacement: str) -> str:
        """Preserves title case, uppercase, or lowercase of the matched token."""
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.capitalize()
        if original.islower():
            return replacement.lower()
        return replacement

    def normalize_unicode(self, text: str) -> Tuple[str, list[str]]:
        """
        Normalizes unicode characters, removes diacritics, and cleans fancy quotes/dashes.
        """
        changes: list[str] = []
        original = text

        # Replace fancy typographic characters first
        char_replacements = {
            "’": "'", "‘": "'", "‚": "'", "‛": "'", "`": "'",
            "“": '"', "”": '"', "„": '"', "‟": '"',
            "—": "-", "–": "-", "―": "-", "−": "-",
            "…": "...", "•": "*", "·": "*",
            "\u00A0": " ", "\u200B": "", "\u200C": "", "\u200D": "", "\uFEFF": ""
        }
        replaced_chars = 0
        for char, repl in char_replacements.items():
            if char in text:
                count = text.count(char)
                text = text.replace(char, repl)
                replaced_chars += count

        # Strip accents / diacritics (e.g. Beyoncé -> Beyonce)
        # We decompose and drop non-spacing marks (category 'Mn')
        decomposed = unicodedata.normalize('NFKD', text)
        cleaned_chars = []
        stripped_accents = 0
        for c in decomposed:
            if unicodedata.category(c) == 'Mn':
                stripped_accents += 1
            else:
                cleaned_chars.append(c)

        text = "".join(cleaned_chars)
        # Standardize remaining unicode
        text = unicodedata.normalize('NFC', text)

        # Remove unprintable control characters except newline and tab
        text = re.sub(r'[^\x20-\x7E\r\n\t]', '', text)

        if text != original:
            detail = []
            if replaced_chars > 0:
                detail.append(f"{replaced_chars} smart punctuation marks")
            if stripped_accents > 0:
                detail.append(f"{stripped_accents} accent marks")
            changes.append(f"Normalized Unicode: converted {' and '.join(detail) if detail else 'non-standard characters'} to clean standard UTF-8.")

        return text, changes

    def filter_celebrities(self, text: str, replacement_token: str = "[Artist]") -> Tuple[str, list[str]]:
        """
        Strips known artist and celebrity names using word boundary regex.
        """
        changes: list[str] = []
        lines = text.split("\n")
        new_lines: list[str] = []

        # Sort names by length descending to match longest multi-word names first
        sorted_names = sorted(self.celebrity_blacklist, key=len, reverse=True)

        for line_idx, line in enumerate(lines, start=1):
            line_result = line
            for name in sorted_names:
                escaped = re.escape(name)
                # Word boundary check - handle special symbols like $ in A$AP
                pattern = rf"\b{escaped}\b"
                matches = list(re.finditer(pattern, line_result, flags=re.IGNORECASE))
                if matches:
                    for match in reversed(matches):
                        matched_str = match.group(0)
                        start, end = match.span()
                        line_result = line_result[:start] + replacement_token + line_result[end:]
                        changes.append(
                            f"Line {line_idx}: Replaced celebrity name '{matched_str}' with '{replacement_token}'"
                        )
            new_lines.append(line_result)

        return "\n".join(new_lines), changes

    def filter_profanities(self, text: str) -> Tuple[str, list[str]]:
        """
        Replaces severe profanities and slurs with musical, safe homophones or mild alternatives.
        """
        changes: list[str] = []
        lines = text.split("\n")
        new_lines: list[str] = []

        # Sort keys by length descending so compound words are matched before subwords
        sorted_profanities = sorted(PROFANITY_HOMOPHONES.keys(), key=len, reverse=True)

        for line_idx, line in enumerate(lines, start=1):
            line_result = line
            for bad_word in sorted_profanities:
                replacement = PROFANITY_HOMOPHONES[bad_word]
                escaped = re.escape(bad_word)
                pattern = rf"\b{escaped}\b"

                matches = list(re.finditer(pattern, line_result, flags=re.IGNORECASE))
                if matches:
                    for match in reversed(matches):
                        matched_str = match.group(0)
                        rep = self.match_case(matched_str, replacement)
                        start, end = match.span()
                        line_result = line_result[:start] + rep + line_result[end:]
                        changes.append(
                            f"Line {line_idx}: Replaced explicit term '{matched_str}' with safe alternative '{rep}'"
                        )
            new_lines.append(line_result)

        return "\n".join(new_lines), changes

    def format_structural_tags(self, text: str) -> Tuple[str, list[str]]:
        """
        Validates and standardizes Suno structural tags like [Verse], [Chorus], [Drop].
        Fixes nested, broken, or parenthesis-wrapped cue tags.
        """
        changes: list[str] = []
        lines = text.split("\n")
        new_lines: list[str] = []

        # Build regex of recognized tags
        tags_union = "|".join(re.escape(tag) for tag in RECOGNIZED_SUNO_TAGS)

        for line_idx, line in enumerate(lines, start=1):
            stripped_line = line.strip()
            line_result = line

            if not stripped_line:
                new_lines.append(line)
                continue

            # 1. Clean nested brackets: e.g. [[Verse]] or [[Chorus 1]]
            nested_match = re.fullmatch(r"^\[+\s*([a-zA-Z0-9\s:_\-]+?)\s*\]+$", stripped_line)
            if nested_match and (stripped_line.startswith("[[") or stripped_line.endswith("]]")):
                content = nested_match.group(1).strip()
                standardized = f"[{content}]"
                changes.append(f"Line {line_idx}: Fixed nested brackets '{stripped_line}' -> '{standardized}'")
                line_result = standardized
                stripped_line = standardized

            # 2. Standalone tags in parentheses or curly braces: (Chorus), (Verse 1), {Bridge}
            parenthesis_cue = re.fullmatch(
                rf"^[\(\{{]\s*(?:{tags_union})(?:\s+[0-9]+)?(?:\s*:\s*[^\)\}}]+)?\s*[\)\}}]$",
                stripped_line,
                re.IGNORECASE
            )
            if parenthesis_cue:
                inner = stripped_line[1:-1].strip()
                standardized = f"[{inner}]"
                changes.append(f"Line {line_idx}: Converted cue brackets '{stripped_line}' -> '{standardized}'")
                line_result = standardized
                stripped_line = standardized

            # 3. Standalone cues without brackets followed by colon or end-of-line:
            # e.g., "Verse 1:", "Chorus:", "Drop:", "Bridge:"
            colon_cue = re.fullmatch(
                rf"^(?:{tags_union})(?:\s+[0-9]+)?\s*:?$",
                stripped_line,
                re.IGNORECASE
            )
            if colon_cue and not stripped_line.startswith("["):
                inner = stripped_line.rstrip(":").strip()
                standardized = f"[{inner}]"
                changes.append(f"Line {line_idx}: Formatted bare cue '{stripped_line}' -> '{standardized}'")
                line_result = standardized
                stripped_line = standardized

            # 4. Canonical capitalization for recognized tags inside square brackets
            bracket_match = re.fullmatch(r"^\[\s*([a-zA-Z0-9\s:_\-]+?)\s*\]$", stripped_line)
            if bracket_match:
                tag_content = bracket_match.group(1).strip()
                # Check if matches any recognized tag
                formatted_tag = None
                for rec_tag in RECOGNIZED_SUNO_TAGS:
                    # Match exact tag or tag with number / qualifier (e.g. Verse 1, Chorus 2, Guitar Solo)
                    pattern = rf"^({re.escape(rec_tag)})(\s+[0-9]+|\s*:\s*.+)?$"
                    m = re.match(pattern, tag_content, flags=re.IGNORECASE)
                    if m:
                        suffix = m.group(2)
                        if suffix and suffix.strip():
                            formatted_tag = f"[{rec_tag} {suffix.strip()}]"
                        else:
                            formatted_tag = f"[{rec_tag}]"
                        break

                if formatted_tag and formatted_tag != stripped_line:
                    changes.append(f"Line {line_idx}: Standardized tag case '{stripped_line}' -> '{formatted_tag}'")
                    line_result = formatted_tag

            new_lines.append(line_result)

        return "\n".join(new_lines), changes

    def analyze_limits(self, text: str) -> Tuple[list[str], dict[str, Any]]:
        """
        Checks for line length exceeding 80 chars, prompt length exceeding 3000 chars,
        and computes lyrics statistics.
        """
        warnings: list[str] = []
        lines = text.split("\n")

        for idx, line in enumerate(lines, start=1):
            if len(line) > 80:
                warnings.append(
                    f"Line {idx} length is {len(line)} characters (exceeds recommended 80-char limit). "
                    "Long lines may cause unnatural vocal pacing or cut-offs in Suno."
                )

        total_chars = len(text)
        if total_chars > 3000:
            warnings.append(
                f"Total lyrics length is {total_chars} characters (exceeds Suno 3,000 character prompt limit). "
                "Lyrics beyond 3,000 characters may be truncated during song generation."
            )

        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        # Average singing tempo: ~130 words per minute
        estimated_seconds = int((word_count / 130.0) * 60) if word_count else 0
        minutes = estimated_seconds // 60
        seconds = estimated_seconds % 60

        stats = {
            "total_characters": total_chars,
            "total_lines": len(lines),
            "word_count": word_count,
            "estimated_duration": f"{minutes}m {seconds:02d}s" if estimated_seconds > 0 else "0s"
        }

        return warnings, stats

    def sanitize(
        self,
        text: str,
        strip_celebrities: bool = True,
        filter_profanity: bool = True,
        normalize_unicode: bool = True,
        format_tags: bool = True,
        check_limits: bool = True
    ) -> LyricsSanitizeResult:
        """
        Executes the full lyrics sanitization pipeline according to enabled toggles.
        """
        if not text:
            return LyricsSanitizeResult(
                original_text="",
                sanitized_text="",
                changes=["Input text is empty."],
                warnings=[],
                stats={"total_characters": 0, "total_lines": 0, "word_count": 0, "estimated_duration": "0s"}
            )

        current_text = text
        all_changes: list[str] = []

        # 1. Normalize Unicode
        if normalize_unicode:
            current_text, norm_changes = self.normalize_unicode(current_text)
            all_changes.extend(norm_changes)

        # 2. Filter Celebrities
        if strip_celebrities:
            current_text, celeb_changes = self.filter_celebrities(current_text)
            all_changes.extend(celeb_changes)

        # 3. Filter Profanities
        if filter_profanity:
            current_text, prof_changes = self.filter_profanities(current_text)
            all_changes.extend(prof_changes)

        # 4. Standardize Structural Tags
        if format_tags:
            current_text, tag_changes = self.format_structural_tags(current_text)
            all_changes.extend(tag_changes)

        # 5. Check Line Limits and Stats
        warnings: list[str] = []
        stats: dict[str, Any] = {}
        if check_limits:
            warnings, stats = self.analyze_limits(current_text)

        if not all_changes:
            all_changes.append("No prohibited words, celebrity names, or formatting errors detected.")

        return LyricsSanitizeResult(
            original_text=text,
            sanitized_text=current_text,
            changes=all_changes,
            warnings=warnings,
            stats=stats
        )
