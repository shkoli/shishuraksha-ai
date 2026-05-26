#!/usr/bin/env python3
"""Synthetic data generator for XAI-MPSCAP-BD.

Generates 500 samples each of:
  1. Questionnaire scores (SDQ + CPSS adapted for Bangladesh, ages 5-17)
       → data/synthetic/questionnaire.csv
  2. Bengali narrative texts (trauma / non-trauma)
       → data/synthetic/narratives.csv
  3. Drawing feature vectors (20 HTP markers, binary)
       → data/synthetic/drawings.csv

Class distribution: 80% non-abuse (label=0), 20% abuse (label=1).
Seed is fixed for reproducibility.

Noise layers applied to simulate real-world messiness:
  - Questionnaire: ±15% measurement error, 10% reporter misclassification,
    5% missing subscales
  - Drawings: 20% marker detection errors, ±0.1 burden-score jitter
  - Text: 15% trauma→non-trauma (partial disclosure), 10% non-trauma→trauma
  - Labels: 8% overall inter-rater disagreement noise
"""

from __future__ import annotations

import io
import random
import sys
import time
from pathlib import Path

# Force UTF-8 output so Bengali text prints correctly on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ── globals ───────────────────────────────────────────────────────────────────

SEED = 42
N_SAMPLES = 500
ABUSE_RATIO = 0.20
OUTPUT_DIR = Path("data/synthetic")

# ── SDQ subscale names ────────────────────────────────────────────────────────
# 5 subscales × 5 items × max 2 points = 0-10 per subscale
# Total difficulties = emotional + conduct + hyperactivity + peer  (0-40)
# Prosocial is reverse-valenced (high = good)
SDQ_SUBSCALES = [
    "sdq_emotional",
    "sdq_conduct",
    "sdq_hyperactivity",
    "sdq_peer",
    "sdq_prosocial",
]

# ── CPSS subscale names ───────────────────────────────────────────────────────
# Child PTSD Symptom Scale (DSM-5 adapted, Bangladesh context)
# Intrusion   : 5 items × 3 = 0-15
# Avoidance   : 7 items × 3 = 0-21
# Neg cogn    : 7 items × 3 = 0-21
# Arousal     : 6 items × 3 = 0-18
# Total max   : 0-75
CPSS_SUBSCALES = [
    "cpss_intrusion",
    "cpss_avoidance",
    "cpss_neg_cognitions",
    "cpss_arousal",
]

# ── HTP (House-Tree-Person) marker names — 20 binary features ─────────────────
HTP_MARKERS = [
    "omit_house",
    "omit_tree",
    "omit_person",
    "dark_shading",
    "heavy_line_pressure",
    "tiny_figure",
    "figure_off_center",
    "facial_features_omitted",
    "hands_omitted",
    "feet_omitted",
    "ground_line_absent",
    "sun_absent",
    "clouds_present",
    "encapsulation",
    "compartmentalization",
    "barriers",
    "aggressive_imagery",
    "regressive_content",
    "figure_eyes_closed",
    "figure_mouth_omitted",
]

# ── Baseline probability of each HTP marker for non-abuse children ────────────
# Sourced from HTP literature (Buck 1992; DiLeo 1983) + Bangladesh pilot data
HTP_BASE_PROBS: dict[str, float] = {
    "omit_house":                0.04,
    "omit_tree":                 0.04,
    "omit_person":               0.03,
    "dark_shading":              0.07,
    "heavy_line_pressure":       0.09,
    "tiny_figure":               0.06,
    "figure_off_center":         0.10,
    "facial_features_omitted":   0.06,
    "hands_omitted":             0.08,
    "feet_omitted":              0.07,
    "ground_line_absent":        0.12,
    "sun_absent":                0.20,
    "clouds_present":            0.15,
    "encapsulation":             0.03,
    "compartmentalization":      0.03,
    "barriers":                  0.04,
    "aggressive_imagery":        0.03,
    "regressive_content":        0.04,
    "figure_eyes_closed":        0.04,
    "figure_mouth_omitted":      0.05,
}

# Multipliers applied to baseline probs for abuse cases.
# Values derived from clinical drawing research (Malchiodi 1998; Klepsch 1982).
HTP_ABUSE_MULTIPLIERS: dict[str, float] = {
    "omit_house":                6.0,
    "omit_tree":                 5.0,
    "omit_person":               7.0,
    "dark_shading":              9.0,
    "heavy_line_pressure":       8.0,
    "tiny_figure":               7.0,
    "figure_off_center":         5.0,
    "facial_features_omitted":   9.0,
    "hands_omitted":             8.0,
    "feet_omitted":              7.0,
    "ground_line_absent":        4.0,
    "sun_absent":                3.0,
    "clouds_present":            2.0,
    "encapsulation":            10.0,
    "compartmentalization":      9.0,
    "barriers":                  8.0,
    "aggressive_imagery":       11.0,
    "regressive_content":       10.0,
    "figure_eyes_closed":        8.0,
    "figure_mouth_omitted":      9.0,
}

# ── Bengali narrative sentence pools ─────────────────────────────────────────
# Non-trauma (label=0): everyday life, school, family, positive interactions
NON_TRAUMA_SENTENCES: list[str] = [
    "আমি প্রতিদিন স্কুলে যাই এবং মনোযোগ দিয়ে পড়াশোনা করি।",
    "আমার পরিবার আমাকে অনেক ভালোবাসে এবং সবসময় আমার পাশে থাকে।",
    "আমি খেলাধুলা করতে খুব পছন্দ করি, বিশেষ করে ক্রিকেট।",
    "আমাদের বাড়িতে সুখ আছে, মা-বাবা সবসময় আমার যত্ন নেন।",
    "স্কুলে আমার প্রিয় বিষয় গণিত, শিক্ষক আমাকে অনেক সাহায্য করেন।",
    "আমি রাতে ভালো ঘুমাই এবং সকালে সতেজ অনুভব করি।",
    "আমার ছোট ভাইয়ের সাথে আমি প্রতিদিন খেলি এবং মজা করি।",
    "বাবা আমাকে প্রতি সপ্তাহে বাজারে নিয়ে যান এবং আমার পছন্দের জিনিস কিনে দেন।",
    "আমি আঁকতে ভালোবাসি, রঙিন ছবি এঁকে দেওয়ালে লাগাই।",
    "আমার দাদু প্রতি রাতে সুন্দর সুন্দর গল্প বলেন, আমি মনোযোগ দিয়ে শুনি।",
    "ঈদের সময় নতুন জামা পরে সবার সাথে দেখা করতে যাই, অনেক আনন্দ হয়।",
    "আমি পরীক্ষায় ভালো ফলাফল করি এবং মা-বাবা খুশি হন।",
    "প্রতিদিন সকালে ব্যায়াম করি, শরীর সুস্থ ও সবল থাকে।",
    "আমার বন্ধু রাহেলা আমার সাথে বসে পড়ে, আমরা একে অপরকে সাহায্য করি।",
    "বৃষ্টির দিনে জানালার পাশে বসে বই পড়ি, অনেক আরামদায়ক লাগে।",
    "আমাদের স্কুলে বার্ষিক ক্রীড়া প্রতিযোগিতা হয়, আমি দৌড়ে পুরস্কার পেয়েছি।",
    "মা আমাকে রান্না শেখাচ্ছেন, এখন ছোট ছোট রান্না নিজেই করতে পারি।",
    "আমার পছন্দের খাবার ভাত, ডাল আর মাছ ভাজি — মা সুন্দর করে রান্না করেন।",
    "বাবার সাথে গ্রামে যাই, নদীর ধারে বসে সময় কাটাই।",
    "আমি ভবিষ্যতে ডাক্তার হতে চাই, তাই মনোযোগ দিয়ে বিজ্ঞান পড়ি।",
    "স্কুলের শিক্ষক আমাকে অনেক উৎসাহ দেন, আমি পড়তে আগ্রহী হই।",
    "আমাদের পাড়ায় সবাই মিলে ঈদ উদযাপন করি, খুব আনন্দ হয়।",
    "আমার মা আমাকে প্রতিদিন গল্প বলেন, আমি ঘুমানোর আগে তাঁর কাছে থাকি।",
    "বাবা-মা আমাকে সবসময় সততার শিক্ষা দেন, আমি তা মেনে চলি।",
    "আমি স্কুলে নতুন বন্ধু পেয়েছি, সে আমার সাথে খেলে এবং হাসে।",
    "আমাদের বাড়িতে একটা বিড়াল আছে, সে আমার সাথে ঘুরে বেড়ায়।",
    "পরীক্ষার আগে মা আমাকে পড়তে সাহায্য করেন, আমি ভয় পাই না।",
    "আমি গান গাইতে পারি, স্কুলের অনুষ্ঠানে গান গেয়েছি।",
    "ছুটির দিনে পরিবারের সবাই মিলে পিকনিকে যাই, অনেক মজা হয়।",
    "আমার বাবা আমাকে সাইকেল শিখিয়েছেন, এখন একাই চালাতে পারি।",
]

# Trauma narratives (label=1): PTSD symptoms, abuse disclosure cues,
# distress, avoidance, fear — written in realistic child voice.
TRAUMA_SENTENCES: list[str] = [
    "আমি রাতে ঘুমাতে পারি না, ভয়ের স্বপ্ন দেখি এবং ঘুম থেকে উঠে কাঁদি।",
    "কেউ আমাকে কষ্ট দিয়েছে কিন্তু আমি কাউকে বলতে পারছি না কারণ ভয় লাগে।",
    "আমি একা থাকতে খুব ভয় পাই, মনে হয় কেউ আমার ক্ষতি করবে।",
    "আমার পেট সবসময় ব্যথা করে এবং স্কুলে যেতে একদম ইচ্ছে করে না।",
    "সেই ঘটনার কথা ভুলতে পারছি না, চোখ বন্ধ করলেই সব মনে পড়ে।",
    "আমি মনে করতে চাই না কিন্তু সেই ঘটনা বারবার মাথায় আসে।",
    "কেউ আমাকে এমন কিছু করতে বলেছে যা আমার পছন্দ ছিল না।",
    "আমি কারো সাথে কথা বলতে চাই না, সবকিছু থেকে দূরে থাকতে চাই।",
    "আমার শরীরে ব্যথা আছে কিন্তু বাড়িতে কেউ বুঝতে পারছে না।",
    "আমি স্কুলে মনোযোগ দিতে পারছি না, মাথায় সবসময় খারাপ চিন্তা ঘোরে।",
    "রাতে অন্ধকারে ঘরে থাকতে পারি না, বাতি জ্বালিয়ে রাখি।",
    "কেউ একজন আমাকে ভয় দেখায়, তার কথা মনে হলেই কাঁদতে ইচ্ছে করে।",
    "আমি আর আগের মতো খেলতে পারি না, ভেতরে অনেক কষ্ট জমে আছে।",
    "বাড়িতে প্রায়ই মারামারি হয়, আমি ভয়ে কোণায় বসে থাকি।",
    "আমার খাওয়ার ইচ্ছা নেই, কিছুই ভালো লাগছে না।",
    "কেউ আমার সাথে খুব খারাপ ব্যবহার করেছে, এখনও মনে হলে কষ্ট লাগে।",
    "আমি রাস্তায় বের হতে ভয় পাই, মনে হয় কেউ আমাকে অনুসরণ করছে।",
    "আমার বিশ্বাস করার কেউ নেই, সব কষ্ট নিজের মধ্যে লুকিয়ে রাখি।",
    "সেই ঘটনার পর থেকে আমি পরিবর্তন হয়ে গেছি, আর আনন্দ পাই না।",
    "আমাকে জোর করে অনেক কিছু করানো হয়েছে, কাউকে বলতে পারছি না।",
    "ঘরে কেউ মদ খেয়ে আমাকে মারে, আমি লুকিয়ে কাঁদি।",
    "স্কুলের একজন বড় মানুষ আমাকে ভয় দেখায়, তার কাছে যেতে ভয় লাগে।",
    "আমি ভুলতে পারছি না যা হয়েছে, বারবার মাথায় আসে।",
    "পরিবারে অনেক সমস্যা, বাবা-মা সবসময় ঝগড়া করেন এবং আমি কাঁদি।",
    "আমার মনে হয় আমি কারো কাছে নিরাপদ না, সবসময় সতর্ক থাকি।",
    "রাতে হঠাৎ ঘুম ভেঙে যায়, বুকের মধ্যে ধড়ফড় করে।",
    "আমি কাউকে স্পর্শ করতে দিতে চাই না, ভয় লাগে।",
    "বাবা আমাকে মারেন, মাও কিছু বলেন না, একা কষ্ট সহ্য করি।",
    "আমার মনে হয় আমি অনেক খারাপ মানুষ, কেউ আমাকে ভালোবাসে না।",
    "সেই লোকটার কথা মনে হলেই আমার সারা শরীর কাঁপতে থাকে।",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _clip_int(value: float, lo: int, hi: int) -> int:
    """Round and clamp a float to an integer range [lo, hi]."""
    return int(max(lo, min(hi, round(value))))


def _assign_labels(n: int, abuse_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Return an array of 0/1 labels with the requested abuse prevalence."""
    n_abuse = round(n * abuse_ratio)
    n_normal = n - n_abuse
    labels = np.array([0] * n_normal + [1] * n_abuse, dtype=np.int8)
    rng.shuffle(labels)
    return labels


def _add_label_noise(
    labels: np.ndarray, noise_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Flip noise_rate fraction of labels to simulate inter-rater disagreement."""
    noisy = labels.copy()
    flip_mask = rng.random(len(labels)) < noise_rate
    noisy[flip_mask] = (1 - noisy[flip_mask]).astype(np.int8)
    return noisy


# ── 1. Questionnaire generator ────────────────────────────────────────────────

def generate_questionnaire(
    n: int,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate SDQ + CPSS scores with realistic measurement noise.

    Noise applied:
      - 10% of cases: scores drawn from opposite-class distribution
        (reporter misclassification)
      - 5% of cases: 1-3 random subscales set to NaN (missing data)
      - All scores: ±15% uniform measurement error
    """
    divisions = [
        "Dhaka", "Chittagong", "Rajshahi", "Khulna",
        "Barisal", "Sylhet", "Rangpur", "Mymensingh",
    ]

    # (mean, std, max) tuples for each subscale per class
    # SDQ subscales (max = 10)
    sdq_params: dict[str, dict[int, tuple[float, float, int]]] = {
        "sdq_emotional":     {0: (2.0, 1.8, 10), 1: (7.2, 1.4, 10)},
        "sdq_conduct":       {0: (1.4, 1.4, 10), 1: (4.8, 1.6, 10)},
        "sdq_hyperactivity": {0: (3.2, 1.8, 10), 1: (5.5, 1.7, 10)},
        "sdq_peer":          {0: (1.6, 1.4, 10), 1: (5.0, 1.7, 10)},
        "sdq_prosocial":     {0: (8.1, 1.2, 10), 1: (4.5, 1.8, 10)},
    }
    # CPSS subscales
    cpss_params: dict[str, dict[int, tuple[float, float, int]]] = {
        "cpss_intrusion":       {0: (1.5, 1.2, 15), 1: (10.5, 2.2, 15)},
        "cpss_avoidance":       {0: (2.1, 1.8, 21), 1: (15.0, 3.0, 21)},
        "cpss_neg_cognitions":  {0: (2.0, 1.7, 21), 1: (13.5, 3.2, 21)},
        "cpss_arousal":         {0: (1.8, 1.4, 18), 1: (11.0, 2.5, 18)},
    }

    all_subscales = SDQ_SUBSCALES + CPSS_SUBSCALES

    rows = []
    for i in range(n):
        label = int(labels[i])
        age = int(rng.integers(5, 18))
        gender = rng.choice(["male", "female"])
        division = rng.choice(divisions)
        ses = rng.choice(["low", "lower_middle", "upper_middle", "high"],
                         p=[0.45, 0.30, 0.18, 0.07])

        # 10% reporter misclassification: generate scores from opposite class
        gen_label = 1 - label if rng.random() < 0.10 else label

        # 5% missing subscales: nullify 1-3 random subscales
        missing_subscales: set[str] = set()
        if rng.random() < 0.05:
            n_miss = int(rng.integers(1, 4))
            chosen = rng.choice(all_subscales, size=min(n_miss, len(all_subscales)),
                                replace=False)
            missing_subscales = set(chosen.tolist())

        row: dict = {
            "sample_id": f"S{i + 1:04d}",
            "age": age,
            "age_group": "child" if age <= 11 else "adolescent",
            "gender": gender,
            "division": division,
            "ses": ses,
        }

        # SDQ — with ±15% measurement error and possible NaN
        for subscale, params in sdq_params.items():
            if subscale in missing_subscales:
                row[subscale] = np.nan
            else:
                mu, sigma, hi = params[gen_label]
                base = _clip_int(rng.normal(mu, sigma), 0, hi)
                noise = rng.uniform(-0.15, 0.15)
                row[subscale] = _clip_int(base * (1.0 + noise), 0, hi)

        sdq_diff_components = [
            row["sdq_emotional"], row["sdq_conduct"],
            row["sdq_hyperactivity"], row["sdq_peer"],
        ]
        if any(pd.isna(v) for v in sdq_diff_components):
            sdq_diff: float | int = np.nan
            row["sdq_borderline"] = np.nan
            row["sdq_abnormal"] = np.nan
        else:
            sdq_diff = int(sum(sdq_diff_components))  # type: ignore[arg-type]
            row["sdq_borderline"] = int(14 <= sdq_diff <= 16)
            row["sdq_abnormal"] = int(sdq_diff >= 17)
        row["sdq_total_difficulties"] = sdq_diff

        # CPSS — with ±15% measurement error and possible NaN
        for subscale, params in cpss_params.items():
            if subscale in missing_subscales:
                row[subscale] = np.nan
            else:
                mu, sigma, hi = params[gen_label]
                base = _clip_int(rng.normal(mu, sigma), 0, hi)
                noise = rng.uniform(-0.15, 0.15)
                row[subscale] = _clip_int(base * (1.0 + noise), 0, hi)

        cpss_components = [
            row["cpss_intrusion"], row["cpss_avoidance"],
            row["cpss_neg_cognitions"], row["cpss_arousal"],
        ]
        if any(pd.isna(v) for v in cpss_components):
            cpss_total: float | int = np.nan
            row["cpss_moderate_plus"] = np.nan
            row["cpss_severe"] = np.nan
        else:
            cpss_total = int(sum(cpss_components))  # type: ignore[arg-type]
            row["cpss_moderate_plus"] = int(cpss_total >= 11)
            row["cpss_severe"] = int(cpss_total >= 25)
        row["cpss_total"] = cpss_total

        row["label"] = label
        rows.append(row)

    return pd.DataFrame(rows)


# ── 2. Bengali narrative generator ───────────────────────────────────────────

def generate_narratives(
    n: int,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate Bengali narratives with realistic cross-class contamination.

    Noise applied:
      - 15% of trauma (label=1) cases: use non-trauma templates only
        (partial disclosure — child conceals trauma)
      - 10% of non-trauma (label=0) cases: use trauma templates
        (false positive — distress without abuse)
      - Remaining trauma cases retain existing 40% partial-disclosure mixing
    """
    rows = []
    for i in range(n):
        label = int(labels[i])

        if label == 1:
            if rng.random() < 0.15:
                # Partial disclosure: trauma case narrates using non-trauma language
                n_sent = int(rng.integers(2, 5))
                sentences = random.sample(NON_TRAUMA_SENTENCES,
                                          min(n_sent, len(NON_TRAUMA_SENTENCES)))
            else:
                n_trauma = int(rng.integers(1, 4))
                sentences = random.sample(TRAUMA_SENTENCES,
                                          min(n_trauma, len(TRAUMA_SENTENCES)))
                # 40% chance of one disguising non-trauma sentence
                if rng.random() < 0.40:
                    sentences.insert(
                        int(rng.integers(0, len(sentences) + 1)),
                        random.choice(NON_TRAUMA_SENTENCES),
                    )
        else:
            if rng.random() < 0.10:
                # False positive: non-trauma child uses trauma-like language
                n_sent = int(rng.integers(1, 3))
                sentences = random.sample(TRAUMA_SENTENCES,
                                          min(n_sent, len(TRAUMA_SENTENCES)))
            else:
                n_sent = int(rng.integers(2, 5))
                sentences = random.sample(NON_TRAUMA_SENTENCES,
                                          min(n_sent, len(NON_TRAUMA_SENTENCES)))

        text = " ".join(sentences)
        rows.append({
            "sample_id": f"S{i + 1:04d}",
            "text": text,
            "n_sentences": len(sentences),
            "char_count": len(text),
            "word_count": len(text.split()),
            "label": label,
        })

    return pd.DataFrame(rows)


# ── 3. HTP drawing feature generator ─────────────────────────────────────────

def generate_drawings(
    n: int,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate 20-dim binary HTP marker vectors with detection noise.

    Noise applied:
      - Each marker has a 20% independent probability of being flipped
        (simulates imperfect automated or human marker detection)
      - marker_burden_pct receives ±0.1 Gaussian noise (in [0,1] space)
        before clamping to [0, 100]
    """
    rows = []
    for i in range(n):
        label = int(labels[i])
        row: dict = {"sample_id": f"S{i + 1:04d}"}
        total = 0
        for marker in HTP_MARKERS:
            p = HTP_BASE_PROBS[marker]
            if label == 1:
                p = min(p * HTP_ABUSE_MULTIPLIERS[marker], 0.92)
            val = int(rng.random() < p)
            # 20% random flip — imperfect marker detection
            if rng.random() < 0.20:
                val = 1 - val
            row[marker] = val
            total += val
        row["total_trauma_markers"] = total
        # ±0.1 jitter in [0,1] space → ±10 pct-point noise
        burden_noise = rng.normal(0.0, 0.1)
        row["marker_burden_pct"] = round(
            float(np.clip(total / len(HTP_MARKERS) * 100.0 + burden_noise * 100.0, 0.0, 100.0)),
            1,
        )
        row["label"] = label
        rows.append(row)

    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()

    print("=" * 65)
    print("  XAI-MPSCAP-BD — Synthetic Data Generator  [with noise]")
    print(f"  Samples: {N_SAMPLES}  |  Abuse ratio: {ABUSE_RATIO:.0%}  |  Seed: {SEED}")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed RNGs for reproducibility
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    # Shared label array — same 80/20 split across all modalities
    labels = _assign_labels(N_SAMPLES, ABUSE_RATIO, rng)
    n_abuse = int(labels.sum())
    n_normal = N_SAMPLES - n_abuse
    print(
        f"\n  Class distribution (true) -> Non-abuse: {n_normal} ({n_normal/N_SAMPLES:.0%})  "
        f"| Abuse: {n_abuse} ({n_abuse/N_SAMPLES:.0%})"
    )

    # 8% inter-rater disagreement: flip labels before writing to CSV
    # (generated using the same rng to stay deterministic)
    labels_noisy = _add_label_noise(labels, 0.08, rng)
    n_flipped = int((labels != labels_noisy).sum())
    print(f"  Label noise (8%): {n_flipped} labels flipped  "
          f"-> recorded abuse: {int(labels_noisy.sum())} "
          f"({labels_noisy.mean():.0%})")

    # ── 1. Questionnaire ──────────────────────────────────────────────────────
    print("\n─── [1/3] Questionnaire (SDQ + CPSS) ───")
    df_q = generate_questionnaire(N_SAMPLES, labels, rng)
    df_q["label"] = labels_noisy          # overwrite with noisy labels
    q_path = OUTPUT_DIR / "questionnaire.csv"
    df_q.to_csv(q_path, index=False, encoding="utf-8")
    print(f"  Saved {len(df_q):,} rows  ->  {q_path}")

    g0 = df_q[df_q.label == 0]
    g1 = df_q[df_q.label == 1]
    print(f"\n  SDQ Total Difficulties  (NaN rows excluded from stats)")
    print(f"    Non-abuse  : mean={g0['sdq_total_difficulties'].mean():.1f}  "
          f"std={g0['sdq_total_difficulties'].std():.1f}  "
          f"range=[{g0['sdq_total_difficulties'].min()}, {g0['sdq_total_difficulties'].max()}]")
    print(f"    Abuse      : mean={g1['sdq_total_difficulties'].mean():.1f}  "
          f"std={g1['sdq_total_difficulties'].std():.1f}  "
          f"range=[{g1['sdq_total_difficulties'].min()}, {g1['sdq_total_difficulties'].max()}]")
    print(f"    Abnormal (≥17): non-abuse {g0['sdq_abnormal'].mean():.0%}  | abuse {g1['sdq_abnormal'].mean():.0%}")
    missing_q = df_q[SDQ_SUBSCALES + CPSS_SUBSCALES].isna().any(axis=1).sum()
    print(f"    Rows with ≥1 missing subscale: {missing_q} ({missing_q/N_SAMPLES:.0%})")

    print(f"\n  CPSS Total")
    print(f"    Non-abuse  : mean={g0['cpss_total'].mean():.1f}  "
          f"std={g0['cpss_total'].std():.1f}  "
          f"range=[{g0['cpss_total'].min()}, {g0['cpss_total'].max()}]")
    print(f"    Abuse      : mean={g1['cpss_total'].mean():.1f}  "
          f"std={g1['cpss_total'].std():.1f}  "
          f"range=[{g1['cpss_total'].min()}, {g1['cpss_total'].max()}]")
    print(f"    Severe (≥25): non-abuse {g0['cpss_severe'].mean():.0%}  | abuse {g1['cpss_severe'].mean():.0%}")

    print(f"\n  Age distribution: mean={df_q['age'].mean():.1f}  "
          f"children={( df_q['age_group']=='child').sum()}  "
          f"adolescents={(df_q['age_group']=='adolescent').sum()}")

    # ── 2. Bengali narratives ─────────────────────────────────────────────────
    print("\n─── [2/3] Bengali Narratives ───")
    df_n = generate_narratives(N_SAMPLES, labels, rng)
    df_n["label"] = labels_noisy          # overwrite with noisy labels
    n_path = OUTPUT_DIR / "narratives.csv"
    # utf-8-sig adds BOM so Excel opens Bengali text correctly
    df_n.to_csv(n_path, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(df_n):,} rows  ->  {n_path}")

    g0n = df_n[df_n.label == 0]
    g1n = df_n[df_n.label == 1]
    print(f"\n  Text statistics")
    print(f"    Non-abuse  : avg {g0n['char_count'].mean():.0f} chars / "
          f"{g0n['word_count'].mean():.0f} words per sample")
    print(f"    Abuse      : avg {g1n['char_count'].mean():.0f} chars / "
          f"{g1n['word_count'].mean():.0f} words per sample")

    print("\n  Example non-abuse text:")
    sample_0 = df_n[df_n.label == 0].iloc[0]["text"]
    print(f"    \"{sample_0[:90]}{'…' if len(sample_0) > 90 else ''}\"")
    print("\n  Example abuse text:")
    sample_1 = df_n[df_n.label == 1].iloc[0]["text"]
    print(f"    \"{sample_1[:90]}{'…' if len(sample_1) > 90 else ''}\"")

    # ── 3. HTP drawings ───────────────────────────────────────────────────────
    print("\n─── [3/3] HTP Drawing Feature Vectors ───")
    df_d = generate_drawings(N_SAMPLES, labels, rng)
    df_d["label"] = labels_noisy          # overwrite with noisy labels
    d_path = OUTPUT_DIR / "drawings.csv"
    df_d.to_csv(d_path, index=False, encoding="utf-8")
    print(f"  Saved {len(df_d):,} rows  ->  {d_path}")

    g0d = df_d[df_d.label == 0]
    g1d = df_d[df_d.label == 1]
    print(f"\n  Trauma marker burden (out of {len(HTP_MARKERS)})")
    print(f"    Non-abuse  : mean={g0d['total_trauma_markers'].mean():.2f}  "
          f"std={g0d['total_trauma_markers'].std():.2f}  "
          f"max={g0d['total_trauma_markers'].max()}")
    print(f"    Abuse      : mean={g1d['total_trauma_markers'].mean():.2f}  "
          f"std={g1d['total_trauma_markers'].std():.2f}  "
          f"max={g1d['total_trauma_markers'].max()}")

    print("\n  Top-5 most discriminative markers (abuse prevalence):")
    marker_disc = {
        m: g1d[m].mean() - g0d[m].mean()
        for m in HTP_MARKERS
    }
    for mk, diff in sorted(marker_disc.items(), key=lambda x: -x[1])[:5]:
        print(
            f"    {mk:<30s}  abuse={g1d[mk].mean():.2%}  "
            f"non-abuse={g0d[mk].mean():.2%}  Δ={diff:+.2%}"
        )

    # ── summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    total_rows = len(df_q) + len(df_n) + len(df_d)
    total_bytes = sum(p.stat().st_size for p in [q_path, n_path, d_path])

    print("\n" + "=" * 65)
    print(f"  Done in {elapsed:.2f}s")
    print(f"  3 files  |  {total_rows:,} total rows  |  {total_bytes / 1024:.1f} KB on disk")
    print(f"  Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
