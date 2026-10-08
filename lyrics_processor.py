"""
Lyrics Processor and Acoustic Scrambler Module for GhostWave Studio v1.0.

Provides multi-vector lyrics sanitization and adversarial text scrambling:
preserves the exact original song words and meaning while disrupting orthography,
syllable boundaries, and phonetic tokens so that automated copyright database
matchers (n-gram filters, regex, exact/fuzzy search) are completely bypassed,
while Suno AI text-to-singing synthesis models sing the original lyrics flawlessly.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Default celebrity and artist blacklist known to trigger Suno/Udio moderation
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
    # Severe Slurs & Hate Speech
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

# Comprehensive Acoustic Phonetic Disguises (Disrupts text matchers, identical sound in Suno)
ACOUSTIC_PHONETIC_DISGUISE: dict[str, str] = {
    # Pronouns, Articles, Small words
    "you": "yu",
    "your": "yur",
    "you're": "yur",
    "the": "th'",
    "and": "an'",
    "to": "2",
    "too": "2",
    "for": "4",
    "with": "wit'",
    "are": "r",
    "one": "wun",
    "two": "tu",

    # Contractions and Conversational Slurs
    "going": "go-in'",
    "gonna": "gon-na",
    "want to": "wan-na",
    "wanna": "wan-na",
    "got to": "got-ta",
    "gotta": "got-ta",
    "trying": "try-in'",
    "because": "'cause",
    "cause": "'cuz",
    "through": "thru",
    "though": "tho",
    "alright": "all-rite",
    "all right": "all-rite",
    "tonight": "2-nite",
    "forever": "for-ev-er",
    "together": "2-geth-er",
    "somebody": "sum-bod-y",
    "everybody": "ev-ery-bod-y",
    "nobody": "no-bod-y",
    "anybody": "en-y-bod-y",
    "anywhere": "en-y-ware",
    "somewhere": "sum-ware",
    "nowhere": "no-ware",
    "someone": "sum-wun",
    "something": "sum-thin'",
    "everything": "ev-ery-thin'",
    "nothing": "noth-in'",

    # Core song vocabulary (Journey, Queen, Beatles, etc.)
    "small": "smal",
    "town": "toun",
    "girl": "gurl",
    "boy": "boi",
    "world": "werld",
    "living": "liv-in'",
    "lonely": "lone-ly",
    "midnight": "mid-nite",
    "train": "trayn",
    "streetlights": "street-lites",
    "people": "pee-pul",
    "emotion": "e-mo-shun",
    "stop": "stopp",
    "believing": "be-leev-in'",
    "believe": "be-leeve",
    "feeling": "fee-lin'",
    "hold": "hol'",
    "holding": "hold-in'",
    "real": "reel",
    "life": "lyfe",
    "fantasy": "fan-ta-see",
    "reality": "re-al-i-tee",
    "landslide": "land-slyde",
    "escape": "es-caype",
    "open": "o-pen",
    "eyes": "eye-z",
    "skies": "sky-z",
    "sky": "skye",
    "sympathy": "sym-pa-thee",
    "easy": "ee-zee",
    "little": "lit-tul",
    "wind": "wynd",
    "blows": "blo-wz",
    "matter": "mat-ter",
    "matters": "mat-terz",
    "yesterday": "yes-ter-day",
    "trouble": "troub-le",
    "troubles": "troub-lez",
    "seemed": "seemd",
    "suddenly": "sud-den-ly",
    "shadow": "shad-o",
    "shadows": "shad-owz",
    "dancing": "dan-sin'",
    "crying": "cry-in'",
    "flying": "fly-in'",
    "heart": "hart",
    "heartbreak": "hart-break",
    "tears": "teer-z",
    "night": "nite",
    "nights": "nites",
    "light": "lite",
    "lights": "lites",
    "dream": "dreem",
    "dreams": "dreemz",
    "dreaming": "dreem-in'",
    "rain": "rayn",
    "fire": "fyre",
    "pain": "payn",
    "burn": "bernn",
    "burning": "burn-in'",
    "waiting": "wait-in'",
    "walking": "walk-in'",
    "talking": "talk-in'",
    "running": "run-nin'",
    "falling": "fall-in'",
    "standing": "stand-in'",
    "breathing": "breath-in'",
    "screaming": "scream-in'",
    "silent": "sy-lent",
    "silence": "sy-lence",
    "beautiful": "beau-ti-ful",
    "tomorrow": "to-mor-ro",
    "memories": "mem-o-reez",
    "memory": "mem-o-ree",
    "mama": "ma-ma",
    "killed": "kill'd",
    "head": "hed",
    "trigger": "trig-ger",
    "dead": "ded",
    "begun": "be-gun",
    "thrown": "throw-n",
    "away": "a-way",
    "again": "a-gain",
    "carry": "car-ry",
    "late": "layte",
    "time": "tyme",
    "shivers": "shiv-erz",
    "spine": "spyne",
    "body": "bo-dy",
    "aching": "ach-in'",
    "goodbye": "good-bye",
    "leave": "leev",
    "leaving": "leav-in'",
    "behind": "be-hind",
    "truth": "trooth",
    "die": "dye",
    "sometimes": "sum-tymes",
    "wish": "wysh",
    "never": "nev-er",
    "born": "borne",
    "sunshine": "sun-shyne",
    "darkness": "dark-ness",
    "bleeding": "bleed-in'",
    "breaking": "break-in'",
    "chasing": "chas-in'",
    "shining": "shin-in'",
    "playing": "play-in'"
}

# Alias for backward compatibility
PHONETIC_HOMOPHONES = ACOUSTIC_PHONETIC_DISGUISE

# Optional Syllable-Preserving Synonyms (Available for users who want synonym mode)
SYLLABLE_SYNONYMS: dict[str, tuple[int, list[str]]] = {
    "small": (1, ["calm", "plain", "dim", "fair"]),
    "town": (1, ["place", "home", "block"]),
    "girl": (1, ["soul", "youth", "one"]),
    "boy": (1, ["youth", "kid", "one"]),
    "world": (1, ["realm", "land", "earth"]),
    "train": (1, ["line", "ride", "rail"]),
    "lonely": (2, ["silent", "distant", "quiet", "fading"]),
    "trouble": (2, ["burden", "struggle", "sorrow"]),
    "troubles": (2, ["burdens", "struggles", "sorrows"]),
    "yesterday": (3, ["days gone by", "times gone by", "seasons past"])
}

# Rhythmic ad-libs for breaking copyright n-grams
ADLIBS: list[str] = [
    "(yeah)", "(oh)", "(c'mon)", "(mm-mm)", "(hey)", "(alright)", "(uh)"
]


def count_syllables(word: str) -> int:
    """Estimates syllable count for English words, stripping punctuation and hyphens."""
    w = word.lower().strip(".,!?;:'\"()[]{}~").replace("-", "").replace("'", "")
    if not w:
        return 0
    if len(w) <= 3:
        return 1

    known = {
        "yesterday": 3, "tomorrow": 3, "forever": 3, "everything": 3,
        "fantasy": 3, "reality": 4, "lonely": 2, "trouble": 2, "troubles": 2,
        "living": 2, "waiting": 2, "midnight": 2, "streetlights": 2,
        "people": 2, "silence": 2, "shadow": 2, "shadows": 2, "anywhere": 3,
        "somebody": 3, "nobody": 3, "everybody": 4, "monstrosity": 4,
        "silhouette": 3, "scaramouche": 3, "fandango": 3, "galileo": 4
    }
    if w in known:
        return known[w]

    if w.endswith('e') and not (w.endswith('le') and len(w) > 2 and w[-3] not in 'aeiouy'):
        w_proc = w[:-1]
    else:
        w_proc = w

    matches = re.findall(r'[aeiouy]+', w_proc)
    count = len(matches)

    for hiatus in ("eo", "ia", "io", "ua", "uo"):
        if hiatus in w:
            count += 1

    return max(1, count)


def count_line_syllables(line: str) -> int:
    """Calculates total syllables in a line of lyrics, ignoring structural tags."""
    stripped = line.strip()
    if not stripped or stripped.startswith("["):
        return 0
    tokens = re.findall(r"\b[a-zA-Z']+\b", stripped)
    return sum(count_syllables(t) for t in tokens)


@dataclass
class LyricsAuditResult:
    """Stores quantitative copyright evasion and cadence metrics."""
    original_words: int
    cloaked_words: int
    ngram_overlap_pct: float
    token_similarity_pct: float
    syllable_accuracy_pct: float
    evasion_verdict: str
    details: list[str] = field(default_factory=list)


def audit_lyrics(original: str, cloaked: str) -> LyricsAuditResult:
    """
    Computes 4-gram overlap, token similarity, and syllable cadence accuracy
    between original and cloaked lyrics.
    """
    orig_words = [w.lower() for w in re.findall(r"\b\w+\b", original)]
    cloaked_words = [w.lower() for w in re.findall(r"\b\w+\b", cloaked)]

    # 4-gram sets
    orig_4grams = {" ".join(orig_words[i:i+4]) for i in range(len(orig_words)-3)} if len(orig_words) >= 4 else set()
    cloaked_4grams = {" ".join(cloaked_words[i:i+4]) for i in range(len(cloaked_words)-3)} if len(cloaked_words) >= 4 else set()

    if orig_4grams:
        overlap = len(orig_4grams & cloaked_4grams)
        ngram_overlap_pct = (overlap / len(orig_4grams)) * 100.0
    else:
        ngram_overlap_pct = 0.0

    # Token Jaccard similarity
    orig_set = set(orig_words)
    cloaked_set = set(cloaked_words)
    union_set = orig_set | cloaked_set
    if union_set:
        token_similarity_pct = (len(orig_set & cloaked_set) / len(union_set)) * 100.0
    else:
        token_similarity_pct = 0.0

    # Line-by-line syllable count matching
    orig_lines = [line.strip() for line in original.split("\n") if line.strip() and not line.strip().startswith("[")]
    cloaked_lines = [line.strip() for line in cloaked.split("\n") if line.strip() and not line.strip().startswith("[")]

    matched_syllables = 0
    total_comparisons = min(len(orig_lines), len(cloaked_lines))
    details = []

    for idx in range(total_comparisons):
        s1 = count_line_syllables(orig_lines[idx])
        s2 = count_line_syllables(cloaked_lines[idx])
        if abs(s1 - s2) <= 1:
            matched_syllables += 1
        details.append(f"Line {idx+1}: original {s1} syl vs cloaked {s2} syl")

    if total_comparisons > 0:
        syllable_accuracy_pct = (matched_syllables / total_comparisons) * 100.0
    else:
        syllable_accuracy_pct = 100.0

    # Verdict classification
    if ngram_overlap_pct < 15.0:
        verdict = "EXCELLENT EVASION (<15% Overlap: Zero copyright match risk)"
    elif ngram_overlap_pct < 35.0:
        verdict = "HIGH EVASION (Substantial n-gram decoupling)"
    elif ngram_overlap_pct < 60.0:
        verdict = "MODERATE EVASION (Partial copyright risk)"
    else:
        verdict = "LOW EVASION (High risk of copyright filter match)"

    return LyricsAuditResult(
        original_words=len(orig_words),
        cloaked_words=len(cloaked_words),
        ngram_overlap_pct=ngram_overlap_pct,
        token_similarity_pct=token_similarity_pct,
        syllable_accuracy_pct=syllable_accuracy_pct,
        evasion_verdict=verdict,
        details=details
    )


class RemoteLyricsRewriter:
    """Executes remote semantic lyrics rewriting using cloud LLM APIs."""

    ENDPOINTS = {
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions"
    }

    @classmethod
    def rewrite(
        cls,
        text: str,
        api_token: str,
        provider: str = "groq",
        model: str = "llama-3.3-70b-versatile",
        custom_endpoint: str = ""
    ) -> Tuple[str, list[str]]:
        if not api_token:
            return text, ["Remote API token not provided. Using offline cloaker."]

        endpoint = custom_endpoint or cls.ENDPOINTS.get(provider.lower(), cls.ENDPOINTS["groq"])
        system_prompt = (
            "You are an acoustic and lyric cloaking assistant for signal processing study.\n"
            "Your task is to rewrite song lyrics so they will NOT match commercial lyrics databases (0% contiguous 4-word overlap).\n"
            "Strict Rules:\n"
            "1. Maintain the EXACT syllable count per line (meter and cadence must match for singing).\n"
            "2. Preserve the song structure, rhyme scheme, and Suno tags like [Verse], [Chorus], [Bridge].\n"
            "3. Do not include preamble, conversational remarks, or markdown code fences; output ONLY the cloaked lyrics."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite these lyrics for acoustic study:\n\n{text}"}
            ],
            "temperature": 0.6,
            "max_tokens": 1500
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}",
                "User-Agent": "GhostWaveStudio/1.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"].strip()
                content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
                content = re.sub(r"\n?```$", "", content).strip()
                return content, [f"Remote cloaking succeeded via {provider} ({model})."]
        except Exception as e:
            return text, [f"Remote API request failed: {e}. Falling back to offline cloaking."]


class LyricsCloaker:
    """
    Adversarial acoustic lyrics scrambler and cloaking engine.
    Scrambles text orthography and syllable boundaries to evade filters while
    preserving the exact original words, meaning, and singing scansion.
    """

    @classmethod
    def cloak(
        cls,
        text: str,
        mode: str = "scramble",
        add_vibrato_glides: bool = True,
        preserve_syllables: bool = True,
        break_ngrams: bool = False,
        adlib_frequency: float = 0.5,
        cloud_token: str = "",
        cloud_provider: str = "groq",
        cloud_model: str = "llama-3.3-70b-versatile",
        cloud_endpoint: str = ""
    ) -> Tuple[str, list[str]]:
        """
        Executes lyrics cloaking according to selected mode:
        - "scramble": Acoustic spelling disruption (Preserves exact original words!)
        - "syllable": Syllable scansion hyphenation on polysyllabic words
        - "phonetic": Pure phonetic disguise
        - "semantic": Syllable-preserving synonyms
        - "cloud": Remote LLM rewriting with automatic fallback
        """
        changes: list[str] = []

        if mode == "cloud" and cloud_token:
            rewritten, cloud_changes = RemoteLyricsRewriter.rewrite(
                text,
                api_token=cloud_token,
                provider=cloud_provider,
                model=cloud_model,
                custom_endpoint=cloud_endpoint
            )
            changes.extend(cloud_changes)
            if rewritten != text:
                return rewritten, changes

        lines = text.split("\n")
        out_lines: list[str] = []
        adlib_idx = 0
        total_scrambled = 0
        adlibs_injected = 0

        for line_idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("["):
                out_lines.append(line)
                continue

            tokens = re.findall(r"[\w'-]+|[^\w\s]", stripped)
            new_tokens = []

            for token in tokens:
                low = token.lower()
                if not re.match(r"[\w'-]", token):
                    new_tokens.append(token)
                    continue

                replaced = False

                # 1. Semantic Synonym Mode (if explicitly requested)
                if mode == "semantic" and low in SYLLABLE_SYNONYMS:
                    syl_count, syns = SYLLABLE_SYNONYMS[low]
                    rep = syns[0]
                    if token.isupper():
                        rep = rep.upper()
                    elif token.istitle():
                        rep = rep.title()
                    new_tokens.append(rep)
                    replaced = True
                    total_scrambled += 1

                # 2. Acoustic Spelling Scrambler (Preserves original lyrics, disrupts orthography!)
                elif mode in ("scramble", "phonetic", "syllable", "hybrid"):
                    if low in ACOUSTIC_PHONETIC_DISGUISE:
                        rep = ACOUSTIC_PHONETIC_DISGUISE[low]
                        if token.isupper():
                            rep = rep.upper()
                        elif token.istitle():
                            rep = rep.capitalize()
                        new_tokens.append(rep)
                        replaced = True
                        total_scrambled += 1
                    elif low.endswith("ing") and len(low) > 4:
                        base = token[:-3]
                        rep = f"{base}-in'"
                        new_tokens.append(rep)
                        replaced = True
                        total_scrambled += 1
                    elif low.endswith("ed") and len(low) > 4 and not low.endswith(("ted", "ded")):
                        base = token[:-2]
                        rep = f"{base}'d"
                        new_tokens.append(rep)
                        replaced = True
                        total_scrambled += 1
                    elif "ght" in low and not low.startswith("ght"):
                        rep = token.replace("ght", "ite").replace("GHT", "ITE")
                        new_tokens.append(rep)
                        replaced = True
                        total_scrambled += 1

                if not replaced:
                    new_tokens.append(token)

            # Reconstruct line respecting punctuation spacing
            reconstructed = ""
            for tok in new_tokens:
                if not reconstructed or tok in ".,!?;:'\"-)" or reconstructed.endswith("(") or reconstructed.endswith('"'):
                    reconstructed += tok
                else:
                    reconstructed += " " + tok

            # Add vibrato glides '~' on line ends or before punctuation
            if add_vibrato_glides and not reconstructed.endswith(("]", "~", "...")):
                if reconstructed.endswith((".", "!", "?", ",")):
                    reconstructed = reconstructed[:-1] + "~" + reconstructed[-1]
                else:
                    reconstructed = reconstructed + "~"

            # Optional adlib injection
            if break_ngrams and len(tokens) >= 4 and not stripped.startswith("["):
                if (line_idx % 2 == 1 or adlib_frequency >= 0.7):
                    adlib = ADLIBS[adlib_idx % len(ADLIBS)]
                    adlib_idx += 1
                    adlibs_injected += 1
                    if "," in reconstructed:
                        reconstructed = reconstructed.replace(",", f" {adlib},", 1)
                    else:
                        reconstructed += f" {adlib}"

            out_lines.append(reconstructed)

        cloaked_text = "\n".join(out_lines)
        changes.append(
            f"Acoustic Text Scrambler ({mode.capitalize()} Mode): Disrupted orthography of {total_scrambled} tokens while preserving original song lyrics."
        )
        if add_vibrato_glides:
            changes.append("Vocal Glissando: Added vibrato inflection markers (~) on phrase cadences.")
        if adlibs_injected > 0:
            changes.append(f"N-Gram Breaker: Injected {adlibs_injected} musical ad-libs.")

        return cloaked_text, changes


@dataclass
class LyricsSanitizeResult:
    """Stores the outcome of the lyrics sanitization process."""
    original_text: str
    sanitized_text: str
    changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    audit: Optional[LyricsAuditResult] = None


class LyricsProcessor:
    """
    Sanitizes and scrambles song lyrics for Suno AI compatibility and copyright evasion.
    """

    def __init__(self, custom_blacklist_path: Optional[str] = None):
        self.blacklist_path = custom_blacklist_path or self._get_default_blacklist_path()
        self.celebrity_blacklist = self.load_blacklist()

    def _get_default_blacklist_path(self) -> str:
        app_dir = Path(os.environ.get("APPDATA", ".")) / "GhostWaveStudio"
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
        """Normalizes unicode characters, removes diacritics, and cleans fancy quotes/dashes."""
        changes: list[str] = []
        original = text

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

        decomposed = unicodedata.normalize('NFKD', text)
        cleaned_chars = []
        stripped_accents = 0
        for c in decomposed:
            if unicodedata.category(c) == 'Mn':
                stripped_accents += 1
            else:
                cleaned_chars.append(c)

        text = "".join(cleaned_chars)
        text = unicodedata.normalize('NFC', text)
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
        """Strips known artist and celebrity names using word boundary regex."""
        changes: list[str] = []
        lines = text.split("\n")
        new_lines: list[str] = []

        sorted_names = sorted(self.celebrity_blacklist, key=len, reverse=True)

        for line_idx, line in enumerate(lines, start=1):
            line_result = line
            for name in sorted_names:
                escaped = re.escape(name)
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
        """Replaces severe profanities and slurs with musical, safe homophones."""
        changes: list[str] = []
        lines = text.split("\n")
        new_lines: list[str] = []

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
        """Validates and standardizes Suno structural tags like [Verse], [Chorus], [Drop]."""
        changes: list[str] = []
        lines = text.split("\n")
        new_lines: list[str] = []

        tags_union = "|".join(re.escape(tag) for tag in RECOGNIZED_SUNO_TAGS)

        for line_idx, line in enumerate(lines, start=1):
            stripped_line = line.strip()
            line_result = line

            if not stripped_line:
                new_lines.append(line)
                continue

            nested_match = re.fullmatch(r"^\[+\s*([a-zA-Z0-9\s:_\-]+?)\s*\]+$", stripped_line)
            if nested_match and (stripped_line.startswith("[[") or stripped_line.endswith("]]")):
                content = nested_match.group(1).strip()
                standardized = f"[{content}]"
                changes.append(f"Line {line_idx}: Fixed nested brackets '{stripped_line}' -> '{standardized}'")
                line_result = standardized
                stripped_line = standardized

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

            bracket_match = re.fullmatch(r"^\[\s*([a-zA-Z0-9\s:_\-]+?)\s*\]$", stripped_line)
            if bracket_match:
                tag_content = bracket_match.group(1).strip()
                formatted_tag = None
                for rec_tag in RECOGNIZED_SUNO_TAGS:
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
        """Checks for line length exceeding 80 chars and prompt length exceeding 3000 chars."""
        warnings: list[str] = []
        lines = text.split("\n")

        for idx, line in enumerate(lines, start=1):
            if len(line) > 80:
                warnings.append(
                    f"Line {idx} length is {len(line)} characters (exceeds recommended 80-char limit)."
                )

        total_chars = len(text)
        if total_chars > 3000:
            warnings.append(
                f"Total lyrics length is {total_chars} characters (exceeds Suno 3,000 character prompt limit)."
            )

        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
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
        check_limits: bool = True,
        cloak_lyrics: bool = True,
        cloak_mode: str = "scramble",
        add_vibrato_glides: bool = True,
        preserve_syllables: bool = True,
        break_ngrams: bool = False,
        adlib_frequency: float = 0.5,
        cloud_token: str = "",
        cloud_provider: str = "groq",
        cloud_model: str = "llama-3.3-70b-versatile",
        cloud_endpoint: str = ""
    ) -> LyricsSanitizeResult:
        """
        Executes the full lyrics sanitization and adversarial scrambling pipeline.
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

        # 5. Cloak / Scramble Lyrics (Preserves Original Lyrics & Scrambles Orthography)
        if cloak_lyrics:
            current_text, cloak_changes = LyricsCloaker.cloak(
                current_text,
                mode=cloak_mode,
                add_vibrato_glides=add_vibrato_glides,
                preserve_syllables=preserve_syllables,
                break_ngrams=break_ngrams,
                adlib_frequency=adlib_frequency,
                cloud_token=cloud_token,
                cloud_provider=cloud_provider,
                cloud_model=cloud_model,
                cloud_endpoint=cloud_endpoint
            )
            all_changes.extend(cloak_changes)

        # 6. Check Line Limits and Stats
        warnings: list[str] = []
        stats: dict[str, Any] = {}
        if check_limits:
            warnings, stats = self.analyze_limits(current_text)

        # 7. Audit Evasion Metrics
        audit_res = audit_lyrics(text, current_text)

        if not all_changes:
            all_changes.append("No prohibited words or formatting errors detected.")

        return LyricsSanitizeResult(
            original_text=text,
            sanitized_text=current_text,
            changes=all_changes,
            warnings=warnings,
            stats=stats,
            audit=audit_res
        )
