"""Bengali narrative feature extraction for trauma disclosure analysis."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Sequence


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


_NEGATION: frozenset[str] = frozenset(_nfc(w) for w in [
    "না", "নয়", "নেই", "নাই", "নহে", "নাকি",
    "কখনো না", "মোটেই না", "মোটেও না", "একদমই না",
])

_FIRST_PERSON: frozenset[str] = frozenset(_nfc(w) for w in [
    "আমি", "আমার", "আমাকে", "আমাদের", "আমরা",
    "আমাতে", "আমায়", "মোর", "মোরা", "মোরে",
])

_PAST_MARKERS: frozenset[str] = frozenset(_nfc(w) for w in [
    "ছিলাম", "ছিল", "ছিলে", "ছিলেন", "ছিলি",
    "করেছিলাম", "করেছিল", "গিয়েছিলাম", "হয়েছিল",
    "পেয়েছিলাম", "দিয়েছিল", "বলেছিলাম", "দেখেছিলাম",
    "গেছিলাম", "এসেছিলাম", "নিয়েছিলাম", "বলেছিল",
    "খেয়েছিলাম", "শুনেছিলাম",
])

_PRESENT_MARKERS: frozenset[str] = frozenset(_nfc(w) for w in [
    "করছি", "করছে", "করছেন", "আছি", "আছে", "আছেন",
    "যাচ্ছি", "যাচ্ছে", "হচ্ছে", "হচ্ছি", "বলছি", "বলছে",
    "দেখছি", "দেখছে", "শুনছি", "শুনছে", "পাচ্ছি", "পাচ্ছে",
    "চলছে", "ভাবছি",
])

_EMOTIONAL: frozenset[str] = frozenset(_nfc(w) for w in [
    "ভয়", "ভীত", "আতঙ্ক", "ভয়ঙ্কর", "কষ্ট", "ব্যথা", "যন্ত্রণা",
    "দুঃখ", "দুঃখিত", "কান্না", "রাগ", "ক্রোধ", "ঘৃণা",
    "বিরক্ত", "হতাশ", "নিরাশ", "লজ্জা", "অপমান", "লাঞ্ছনা",
    "অসহায়", "একা", "বিষণ্ণ", "উদ্বিগ্ন", "চিন্তিত",
    "ভালো", "খুশি", "আনন্দ", "সুখ", "হাসি", "আশা",
    "ভালোবাসা", "প্রেম", "শান্তি", "স্বস্তি",
])

_HEDGING: frozenset[str] = frozenset(_nfc(w) for w in [
    "হয়তো", "হয়তোবা", "সম্ভবত", "সম্ভব", "বোধহয়",
    "যেন", "যেনবা", "অনেকটা", "কিছুটা",
])

_DISCOURSE_MARKERS: frozenset[str] = frozenset(_nfc(w) for w in [
    "তারপর", "তারপরে", "এরপর", "এরপরে", "পরে", "আগে",
    "কিন্তু", "তবে", "যদিও", "তাহলে", "সেজন্য",
    "কারণ", "তাই", "সুতরাং", "ফলে", "যখন", "তখন",
    "এবং", "আর", "বরং",
])

_TRAUMA_KEYWORDS: frozenset[str] = frozenset(_nfc(w) for w in [
    "নির্যাতন", "নির্যাতিত", "আঘাত", "মারধর", "মেরেছিল",
    "জোর", "জোরপূর্বক", "ধর্ষণ", "যৌন", "অশ্লীল",
    "ছুঁয়েছিল", "স্পর্শ", "হুমকি", "বাঁচাও", "লুকিয়ে",
])


@dataclass
class NarrativeFeatures:
    """Eight float features extracted from one Bengali narrative string (all in [0.0, 1.0])."""
    type_token_ratio:       float
    negation_ratio:         float
    first_person_ratio:     float
    temporal_consistency:   float
    emotional_word_density: float
    hedging_ratio:          float
    disclosure_score:       float
    coherence_score:        float

    def __repr__(self) -> str:
        lines = ["NarrativeFeatures("]
        for field_name, val in self.__dict__.items():
            lines.append(f"  {field_name}={val:.4f},")
        lines.append(")")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)

    def to_vector(self) -> list[float]:
        return list(self.__dict__.values())


class NarrativeFeatureExtractor:
    """Rule-based extractor producing 8 linguistic features per narrative.

    No network access or ML model required — all word lists are embedded
    as frozensets of NFC-normalised Unicode strings.
    """

    def __init__(self, min_tokens: int = 1) -> None:
        self.min_tokens = min_tokens

    def _tokenize(self, text: str) -> list[str]:
        text = _nfc(text)
        raw = text.split()
        cleaned: list[str] = []
        for tok in raw:
            tok = tok.strip("।॥.,;:!?'\"""()[]{}—–")
            tok = _nfc(tok)
            if tok:
                cleaned.append(tok)
        return cleaned

    def _type_token_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return min(1.0, len(set(tokens)) / len(tokens))

    def _ratio(self, tokens: list[str], lexicon: frozenset[str]) -> float:
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in lexicon)
        return hits / len(tokens)

    def _temporal_consistency(self, tokens: list[str]) -> float:
        """1.0 if text uses only one tense family, lower if mixed."""
        n_past    = sum(1 for t in tokens if t in _PAST_MARKERS)
        n_present = sum(1 for t in tokens if t in _PRESENT_MARKERS)
        total = n_past + n_present
        if total == 0:
            return 1.0
        return max(n_past, n_present) / total

    def _disclosure_score(
        self,
        tokens: list[str],
        emotional_density: float,
        first_person_ratio: float,
    ) -> float:
        """Weighted composite: emotional density + first-person + trauma keywords."""
        trauma_hits = sum(1 for t in tokens if t in _TRAUMA_KEYWORDS)
        trauma_ratio = min(1.0, trauma_hits / max(1, len(tokens)) * 5)
        score = (
            0.40 * emotional_density
            + 0.30 * first_person_ratio
            + 0.30 * trauma_ratio
        )
        return min(1.0, score)

    def _coherence_score(self, tokens: list[str]) -> float:
        """Length adequacy + discourse markers + vocabulary richness."""
        n = len(tokens)
        if n == 0:
            return 0.0
        length_score    = min(1.0, n / 20.0)
        dm_hits         = sum(1 for t in tokens if t in _DISCOURSE_MARKERS)
        discourse_score = min(1.0, dm_hits / max(1, n) * 10)
        type_frac       = len(set(tokens)) / n
        return min(1.0, max(0.0,
            0.40 * length_score + 0.30 * discourse_score + 0.30 * type_frac
        ))

    def extract(self, text: str) -> NarrativeFeatures:
        tokens = self._tokenize(text)

        if len(tokens) < self.min_tokens:
            return NarrativeFeatures(
                type_token_ratio        = 0.0,
                negation_ratio          = 0.0,
                first_person_ratio      = 0.0,
                temporal_consistency    = 0.0,
                emotional_word_density  = 0.0,
                hedging_ratio           = 0.0,
                disclosure_score        = 0.0,
                coherence_score         = 0.0,
            )

        ttr     = self._type_token_ratio(tokens)
        neg_r   = self._ratio(tokens, _NEGATION)
        fp_r    = self._ratio(tokens, _FIRST_PERSON)
        temp_c  = self._temporal_consistency(tokens)
        emo_d   = self._ratio(tokens, _EMOTIONAL)
        hedge_r = self._ratio(tokens, _HEDGING)
        disc_s  = self._disclosure_score(tokens, emo_d, fp_r)
        coh_s   = self._coherence_score(tokens)

        return NarrativeFeatures(
            type_token_ratio        = ttr,
            negation_ratio          = neg_r,
            first_person_ratio      = fp_r,
            temporal_consistency    = temp_c,
            emotional_word_density  = emo_d,
            hedging_ratio           = hedge_r,
            disclosure_score        = disc_s,
            coherence_score         = coh_s,
        )

    def extract_batch(self, texts: Sequence[str]) -> list[NarrativeFeatures]:
        return [self.extract(t) for t in texts]
