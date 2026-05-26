"""Validated psychiatric questionnaire instruments for Bangladesh child/adolescent screening.

Three instruments, each with bilingual items (English + Bengali):

  BangladeshSDQ  — Strengths & Difficulties Questionnaire (Goodman 1997; BD-adapted)
                   25 items · 5 subscales · ages 5-17
  CPSS           — Child PTSD Symptom Scale (Foa et al. 2001)
                   17 items · 3 subscales · ages 8-17
  CSBI_BD        — Child Sexual Behavior Inventory, Bangladesh Adaptation (Friedrich 1997)
                   38 items · 7 subscales · age-gated (5-11 parent-report / 12-17 self-report)

All instruments are administered in clinical and school settings across Bangladesh.
QuestionnaireInstrument is kept as a backward-compatible alias for BangladeshSDQ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class InstrumentType(str, Enum):
    SDQ  = "SDQ"
    CPSS = "CPSS"
    CSBI = "CSBI"


@dataclass(frozen=True)
class ResponseOption:
    value: int
    label_en: str
    label_bn: str

    def __repr__(self) -> str:
        return f"ResponseOption({self.value}, {self.label_en!r})"


@dataclass(frozen=True)
class Item:
    """A single bilingual questionnaire item with full scoring metadata.

    Attributes:
        item_id:        Unique identifier, e.g. 'SDQ_01'.
        subscale:       Subscale this item belongs to.
        text_en:        Full item text in English.
        text_bn:        Full item text in Bengali.
        options:        Ordered tuple of ResponseOption objects.
        reverse_scored: If True, scored value = max_score - raw_value.
        min_age:        Minimum applicable age (inclusive).
        max_age:        Maximum applicable age (inclusive).
        reporter:       'self' | 'parent' — who completes this item.
    """
    item_id:        str
    subscale:       str
    text_en:        str
    text_bn:        str
    options:        tuple[ResponseOption, ...]
    reverse_scored: bool = False
    min_age:        int  = 5
    max_age:        int  = 17
    reporter:       str  = "self"

    @property
    def max_score(self) -> int:
        return max(o.value for o in self.options)

    @property
    def min_score(self) -> int:
        return min(o.value for o in self.options)

    @property
    def n_options(self) -> int:
        return len(self.options)

    def score(self, raw_value: int) -> int:
        """Return the scored value, applying reverse-scoring where flagged."""
        if raw_value < self.min_score or raw_value > self.max_score:
            raise ValueError(
                f"Item {self.item_id}: raw_value {raw_value} outside "
                f"[{self.min_score}, {self.max_score}]"
            )
        return (self.max_score - raw_value) if self.reverse_scored else raw_value

    def is_applicable(self, age: int) -> bool:
        return self.min_age <= age <= self.max_age

    def __repr__(self) -> str:
        rev = " [R]" if self.reverse_scored else ""
        return (
            f"Item(id={self.item_id!r}, subscale={self.subscale!r}{rev}, "
            f"ages={self.min_age}-{self.max_age}, "
            f"text_en={self.text_en[:50]!r})"
        )


_SDQ_OPTIONS: tuple[ResponseOption, ...] = (
    ResponseOption(0, "Not True",       "সত্য নয়"),
    ResponseOption(1, "Somewhat True",  "কিছুটা সত্য"),
    ResponseOption(2, "Certainly True", "অবশ্যই সত্য"),
)

_CPSS_OPTIONS: tuple[ResponseOption, ...] = (
    ResponseOption(0, "Not at all",               "একদমই নয়"),
    ResponseOption(1, "Once a week or less",       "সপ্তাহে একবার বা কম"),
    ResponseOption(2, "2–4 times a week",          "সপ্তাহে ২–৪ বার"),
    ResponseOption(3, "5 or more times a week",    "সপ্তাহে ৫ বার বা বেশি"),
)

_CSBI_OPTIONS: tuple[ResponseOption, ...] = (
    ResponseOption(0, "Never",                    "কখনো নয়"),
    ResponseOption(1, "Less than once a month",   "মাসে একবারের কম"),
    ResponseOption(2, "1–3 times a month",        "মাসে ১–৩ বার"),
    ResponseOption(3, "At least once a week",     "সপ্তাহে অন্তত একবার"),
)


class BaseInstrument:
    instrument_type: InstrumentType
    name_en:         str = ""
    name_bn:         str = ""
    min_age:         int = 5
    max_age:         int = 17

    def __init__(self) -> None:
        self._items:        list[Item]            = []
        self._subscale_map: dict[str, list[str]]  = {}
        self._build()
        self._item_index: dict[str, Item] = {it.item_id: it for it in self._items}

    def _build(self) -> None:
        raise NotImplementedError

    def _register(self, item: Item) -> None:
        self._items.append(item)
        self._subscale_map.setdefault(item.subscale, []).append(item.item_id)

    def get_items(self, subscale: str | None = None, age: int | None = None) -> list[Item]:
        items = list(self._items)
        if subscale is not None:
            ids = set(self._subscale_map.get(subscale, []))
            items = [it for it in items if it.item_id in ids]
        if age is not None:
            items = [it for it in items if it.is_applicable(age)]
        return items

    def get_item(self, item_id: str) -> Item | None:
        return self._item_index.get(item_id)

    @property
    def subscale_names(self) -> list[str]:
        return list(self._subscale_map.keys())

    @property
    def n_items(self) -> int:
        return len(self._items)

    def validate_age(self, age: int) -> tuple[bool, str]:
        """Return (ok, message). ok is False if age is outside instrument range."""
        if age < self.min_age or age > self.max_age:
            return False, (
                f"{self.name_en} is validated for ages {self.min_age}–{self.max_age}; "
                f"received age {age}."
            )
        return True, ""

    def validate_responses(
        self,
        responses: dict[str, int],
        age: int | None = None,
    ) -> tuple[bool, list[str]]:
        """Returns (is_valid, list_of_error_messages)."""
        errors: list[str] = []
        applicable = self.get_items(age=age)
        applicable_ids = {it.item_id for it in applicable}

        for item_id, value in responses.items():
            if item_id not in self._item_index:
                errors.append(f"Unknown item ID: {item_id!r}")
                continue
            item = self._item_index[item_id]
            if age is not None and not item.is_applicable(age):
                errors.append(
                    f"Item {item_id} is not applicable at age {age} "
                    f"(valid ages: {item.min_age}–{item.max_age})"
                )
                continue
            if not (item.min_score <= value <= item.max_score):
                errors.append(
                    f"Item {item_id}: value {value} outside valid range "
                    f"[{item.min_score}, {item.max_score}]"
                )

        missing = applicable_ids - set(responses.keys())
        if missing:
            errors.append(f"Missing responses for: {sorted(missing)}")

        return len(errors) == 0, errors

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_items={self.n_items}, "
            f"subscales={self.subscale_names})"
        )

    def __iter__(self) -> Iterator[Item]:
        return iter(self._items)


class BangladeshSDQ(BaseInstrument):
    """Strengths and Difficulties Questionnaire — Bangladesh adaptation.

    25 items, 5 subscales (5 items each).
    3-point response scale: 0=Not True / 1=Somewhat True / 2=Certainly True.
    Reverse-scored items: SDQ_07, SDQ_11, SDQ_14, SDQ_21, SDQ_25.

    Subscales:
      prosocial      : higher = better (NOT included in total difficulties)
      hyperactivity  : higher = worse
      emotional      : higher = worse
      conduct        : higher = worse
      peer           : higher = worse
    Total difficulties (0–40) = hyperactivity + emotional + conduct + peer.

    Normative cutoffs (Goodman 2001; Bangladesh community pilot 2022):
      total_difficulties  normal ≤ 13 | borderline 14–16 | abnormal ≥ 17
    """

    instrument_type = InstrumentType.SDQ
    name_en         = "Strengths and Difficulties Questionnaire"
    name_bn         = "শক্তি ও কঠিনতার প্রশ্নমালা"
    min_age         = 5
    max_age         = 17

    SUBSCALE_MAX: dict[str, int] = {
        "prosocial":     10,
        "hyperactivity": 10,
        "emotional":     10,
        "conduct":       10,
        "peer":          10,
    }
    TOTAL_DIFFICULTIES_MAX = 40

    CUTOFFS = {
        "normal":     (0,  13),
        "borderline": (14, 16),
        "abnormal":   (17, 40),
    }

    def _build(self) -> None:
        _O = _SDQ_OPTIONS

        # Prosocial (items 1, 4, 9, 17, 20)
        self._register(Item(
            "SDQ_01", "prosocial",
            "I try to be nice to other people. I care about their feelings.",
            "আমি অন্যদের প্রতি ভালো ব্যবহার করার চেষ্টা করি। তাদের অনুভূতির কথা মাথায় রাখি।",
            _O,
        ))
        self._register(Item(
            "SDQ_04", "prosocial",
            "I usually share with others, for example books, games, or food.",
            "আমি সাধারণত অন্যদের সাথে জিনিসপত্র ভাগ করি, যেমন বই, খেলনা বা খাবার।",
            _O,
        ))
        self._register(Item(
            "SDQ_09", "prosocial",
            "I am helpful if someone is hurt, upset, or feeling ill.",
            "কেউ আহত, মন খারাপ বা অসুস্থ হলে আমি সাহায্য করি।",
            _O,
        ))
        self._register(Item(
            "SDQ_17", "prosocial",
            "I am kind to younger children.",
            "আমি ছোট শিশুদের প্রতি সদয়।",
            _O,
        ))
        self._register(Item(
            "SDQ_20", "prosocial",
            "I often volunteer to help others (parents, teachers, or children).",
            "আমি প্রায়ই স্বেচ্ছায় অন্যদের সাহায্য করি, যেমন বাবা-মা, শিক্ষক বা অন্য শিশুদের।",
            _O,
        ))

        # Hyperactivity (items 2, 10, 15, 21R, 25R)
        self._register(Item(
            "SDQ_02", "hyperactivity",
            "I am restless, I cannot stay still for long.",
            "আমি অস্থির থাকি, বেশিক্ষণ স্থির থাকতে পারি না।",
            _O,
        ))
        self._register(Item(
            "SDQ_10", "hyperactivity",
            "I am constantly fidgeting or squirming.",
            "আমি ক্রমাগত ছটফট করি বা নড়াচড়া করি।",
            _O,
        ))
        self._register(Item(
            "SDQ_15", "hyperactivity",
            "I am easily distracted. I find it difficult to concentrate.",
            "আমি সহজেই বিক্ষিপ্ত হই। মনোযোগ দিতে কষ্ট হয়।",
            _O,
        ))
        self._register(Item(
            "SDQ_21", "hyperactivity",
            "I think things out before acting.",
            "আমি কাজ করার আগে ভেবে নিই।",
            _O, reverse_scored=True,
        ))
        self._register(Item(
            "SDQ_25", "hyperactivity",
            "I finish the work I am doing. My attention is good.",
            "আমি যে কাজ শুরু করি তা শেষ করি। আমার মনোযোগ ভালো।",
            _O, reverse_scored=True,
        ))

        # Emotional (items 3, 8, 13, 16, 24)
        self._register(Item(
            "SDQ_03", "emotional",
            "I get a lot of headaches, stomach-aches, or sickness.",
            "আমার প্রায়ই মাথাব্যথা, পেটব্যথা বা শারীরিক অসুস্থতা হয়।",
            _O,
        ))
        self._register(Item(
            "SDQ_08", "emotional",
            "I worry a lot.",
            "আমি অনেক বেশি চিন্তা করি।",
            _O,
        ))
        self._register(Item(
            "SDQ_13", "emotional",
            "I am often unhappy, depressed, or tearful.",
            "আমি প্রায়ই অসুখী, বিষণ্ণ বা কাঁদছি এমন মনে হয়।",
            _O,
        ))
        self._register(Item(
            "SDQ_16", "emotional",
            "I am nervous in new situations. I easily lose confidence.",
            "নতুন পরিস্থিতিতে আমি নার্ভাস হই। সহজেই আত্মবিশ্বাস হারিয়ে ফেলি।",
            _O,
        ))
        self._register(Item(
            "SDQ_24", "emotional",
            "I have many fears. I am easily scared.",
            "আমার অনেক ভয় আছে। আমি সহজেই ভয় পাই।",
            _O,
        ))

        # Conduct (items 5, 7R, 12, 18, 22)
        self._register(Item(
            "SDQ_05", "conduct",
            "I get very angry and often lose my temper.",
            "আমি খুব রেগে যাই এবং প্রায়ই মেজাজ হারাই।",
            _O,
        ))
        self._register(Item(
            "SDQ_07", "conduct",
            "I usually do as I am told.",
            "আমি সাধারণত যা বলা হয় তাই করি।",
            _O, reverse_scored=True,
        ))
        self._register(Item(
            "SDQ_12", "conduct",
            "I fight a lot. I can make other people do what I want.",
            "আমি অনেক ঝগড়া করি। আমি অন্যদের দিয়ে আমার ইচ্ছামতো কাজ করাতে পারি।",
            _O,
        ))
        self._register(Item(
            "SDQ_18", "conduct",
            "I am often accused of lying or cheating.",
            "আমাকে প্রায়ই মিথ্যা বলা বা প্রতারণার জন্য দোষ দেওয়া হয়।",
            _O,
        ))
        self._register(Item(
            "SDQ_22", "conduct",
            "I take things that are not mine from home, school, or elsewhere.",
            "আমি বাড়ি, স্কুল বা অন্য জায়গা থেকে যা আমার নয় তা নিয়ে নিই।",
            _O,
        ))

        # Peer problems (items 6, 11R, 14R, 19, 23)
        self._register(Item(
            "SDQ_06", "peer",
            "I would rather be alone than with people of my age.",
            "আমি সমবয়সীদের সাথে থাকার চেয়ে একা থাকতে বেশি পছন্দ করি।",
            _O,
        ))
        self._register(Item(
            "SDQ_11", "peer",
            "I have at least one good friend.",
            "আমার অন্তত একজন ভালো বন্ধু আছে।",
            _O, reverse_scored=True,
        ))
        self._register(Item(
            "SDQ_14", "peer",
            "Other people of my age generally like me.",
            "আমার বয়সী অন্যরা সাধারণত আমাকে পছন্দ করে।",
            _O, reverse_scored=True,
        ))
        self._register(Item(
            "SDQ_19", "peer",
            "Other children or young people pick on me or bully me.",
            "অন্য শিশুরা বা তরুণরা আমাকে বিরক্ত করে বা ধমকায়।",
            _O,
        ))
        self._register(Item(
            "SDQ_23", "peer",
            "I get along better with adults than with people of my own age.",
            "আমি সমবয়সীদের চেয়ে বড়দের সাথে বেশি মিলি।",
            _O,
        ))


class CPSS(BaseInstrument):
    """Child PTSD Symptom Scale — Bangladesh clinical adaptation.

    17 items, 3 subscales, 4-point frequency scale.
    Based on DSM-IV PTSD criteria (Foa, Johnson, Feeny & Treadwell 2001).

    Subscales:
      re_experiencing  : 5 items · max 15
      avoidance        : 7 items · max 21
      arousal          : 5 items · max 15
    Total: 0–51.

    Clinical cutoffs (Foa et al. 2001; adapted):
      no_ptsd    : 0–10
      subclinical: 11–20
      clinical   : 21–34
      severe     : 35–51
    """

    instrument_type = InstrumentType.CPSS
    name_en         = "Child PTSD Symptom Scale"
    name_bn         = "শিশু পিটিএসডি উপসর্গ মাপকাঠি"
    min_age         = 8
    max_age         = 17

    SUBSCALE_MAX: dict[str, int] = {
        "re_experiencing": 15,
        "avoidance":       21,
        "arousal":         15,
    }
    TOTAL_MAX = 51

    CUTOFFS = {
        "no_ptsd":     (0,  10),
        "subclinical": (11, 20),
        "clinical":    (21, 34),
        "severe":      (35, 51),
    }

    def _build(self) -> None:
        _O = _CPSS_OPTIONS

        # Re-experiencing (items 1–5)
        self._register(Item(
            "CPSS_01", "re_experiencing",
            "Having upsetting thoughts or images about the traumatic event that came into your head when you didn't want them to.",
            "যে কষ্টের ঘটনা হয়েছে তার বিষয়ে কষ্টদায়ক চিন্তা বা ছবি মাথায় আসে যখন আমি চাই না।",
            _O,
        ))
        self._register(Item(
            "CPSS_02", "re_experiencing",
            "Having bad dreams or nightmares about the traumatic event.",
            "সেই কষ্টের ঘটনা নিয়ে খারাপ স্বপ্ন বা দুঃস্বপ্ন দেখি।",
            _O,
        ))
        self._register(Item(
            "CPSS_03", "re_experiencing",
            "Acting or feeling as if the traumatic event were happening again (flashbacks).",
            "মনে হয় সেই ঘটনা আবার ঘটছে অথবা আমি সেটা আবার অনুভব করছি (ফ্ল্যাশব্যাক)।",
            _O,
        ))
        self._register(Item(
            "CPSS_04", "re_experiencing",
            "Feeling very upset when you are reminded of the traumatic event.",
            "সেই ঘটনার কথা মনে হলে অনেক কষ্ট বা অস্বস্তি অনুভব করি।",
            _O,
        ))
        self._register(Item(
            "CPSS_05", "re_experiencing",
            "Having physical reactions when reminded of the traumatic event "
            "(heart beating fast, trouble breathing, sweating).",
            "সেই ঘটনার কথা মনে হলে শারীরিক প্রতিক্রিয়া হয় "
            "(হৃদপিণ্ড দ্রুত চলে, শ্বাস নিতে কষ্ট হয়, ঘাম হয়)।",
            _O,
        ))

        # Avoidance (items 6–12)
        self._register(Item(
            "CPSS_06", "avoidance",
            "Trying not to think about, talk about, or have feelings about the traumatic event.",
            "সেই ঘটনার কথা না ভাবা, না বলা বা সেই অনুভূতি এড়িয়ে চলার চেষ্টা করি।",
            _O,
        ))
        self._register(Item(
            "CPSS_07", "avoidance",
            "Trying to stay away from activities, places, people, or things that remind you of the traumatic event.",
            "যেসব কাজ, জায়গা, মানুষ বা জিনিস সেই ঘটনার কথা মনে করিয়ে দেয় সেগুলো এড়িয়ে চলার চেষ্টা করি।",
            _O,
        ))
        self._register(Item(
            "CPSS_08", "avoidance",
            "Not being able to remember an important part of the traumatic event.",
            "সেই ঘটনার কোনো গুরুত্বপূর্ণ অংশ মনে করতে পারি না।",
            _O,
        ))
        self._register(Item(
            "CPSS_09", "avoidance",
            "Having much less interest or participating much less in important activities.",
            "আগে যে গুরুত্বপূর্ণ কাজকর্ম করতাম সেগুলোতে আগ্রহ বা অংশগ্রহণ অনেক কমে গেছে।",
            _O,
        ))
        self._register(Item(
            "CPSS_10", "avoidance",
            "Feeling distant or cut off from people around you.",
            "আশেপাশের মানুষদের থেকে দূরে বা বিচ্ছিন্ন মনে হয়।",
            _O,
        ))
        self._register(Item(
            "CPSS_11", "avoidance",
            "Feeling emotionally numb — unable to have loving feelings or to cry when sad.",
            "আবেগহীন মনে হয় — ভালোবাসার অনুভূতি বা কষ্ট পেলেও কাঁদতে পারি না।",
            _O,
        ))
        self._register(Item(
            "CPSS_12", "avoidance",
            "Feeling as if your future plans or hopes will not come true.",
            "মনে হয় ভবিষ্যতের পরিকল্পনা বা স্বপ্ন পূরণ হবে না।",
            _O,
        ))

        # Arousal (items 13–17)
        self._register(Item(
            "CPSS_13", "arousal",
            "Having trouble falling or staying asleep.",
            "ঘুমিয়ে পড়তে বা ঘুমিয়ে থাকতে কষ্ট হয়।",
            _O,
        ))
        self._register(Item(
            "CPSS_14", "arousal",
            "Feeling irritable or having fits of anger.",
            "বিরক্ত মনে হয় বা হঠাৎ রেগে যাই।",
            _O,
        ))
        self._register(Item(
            "CPSS_15", "arousal",
            "Having trouble concentrating.",
            "মনোযোগ দিতে কষ্ট হয়।",
            _O,
        ))
        self._register(Item(
            "CPSS_16", "arousal",
            "Being overly alert — on guard or watchful even when there is no need.",
            "অতিরিক্ত সতর্ক থাকি — প্রয়োজন না থাকলেও সবসময় পাহারা দিই বা নজর রাখি।",
            _O,
        ))
        self._register(Item(
            "CPSS_17", "arousal",
            "Being jumpy or easily startled, for example when someone walks up behind you.",
            "সহজেই চমকে উঠি বা ভয় পাই, যেমন কেউ পেছন থেকে এলে।",
            _O,
        ))


class CSBI_BD(BaseInstrument):
    """Child Sexual Behavior Inventory — Bangladesh Adaptation.

    38 items · 7 subscales · 4-point frequency scale.
    Adapted from Friedrich et al. (1992, 1997) for Bangladeshi clinical context.

    Age gating:
      5–11  : parent/caregiver report (reporter='parent')
      12–17 : self-report (reporter='self')

    Subscales:
      self_stimulation    : 7 items
      boundary_problems   : 7 items
      sexual_anxiety      : 6 items
      sexual_interest     : 8 items
      sexual_intrusiveness: 5 items
      sexual_knowledge    : 3 items
      voyeuristic_behavior: 2 items

    Normative cutoffs (T-scores; Friedrich 1997; BD adaptation):
      typical    : T ≤ 60  (within 1 SD of community mean)
      elevated   : T 61–65
      clinical   : T ≥ 66  (2+ SD above mean — warrants clinical attention)
    """

    instrument_type = InstrumentType.CSBI
    name_en         = "Child Sexual Behavior Inventory — Bangladesh Adaptation"
    name_bn         = "শিশু যৌন আচরণ তালিকা — বাংলাদেশ অভিযোজন"
    min_age         = 5
    max_age         = 17

    SUBSCALE_MAX: dict[str, int] = {
        "self_stimulation":     21,
        "boundary_problems":    21,
        "sexual_anxiety":       18,
        "sexual_interest":      24,
        "sexual_intrusiveness": 15,
        "sexual_knowledge":      9,
        "voyeuristic_behavior":  6,
    }
    TOTAL_MAX = 114

    CUTOFFS = {
        "typical":  (0,  27),
        "elevated": (28, 38),
        "clinical": (39, 114),
    }

    CUTOFFS_CHILD      = {"typical": (0, 20), "elevated": (21, 28), "clinical": (29, 114)}
    CUTOFFS_ADOLESCENT = {"typical": (0, 32), "elevated": (33, 44), "clinical": (45, 114)}

    def _build(self) -> None:
        _O = _CSBI_OPTIONS

        # Self-stimulation (7 items) — parent-report for <12, self-report for 12+
        self._register(Item(
            "CSBI_01", "self_stimulation",
            "Touches or rubs own private parts at home.",
            "বাড়িতে নিজের গোপন অঙ্গ স্পর্শ করে বা ঘষে।",
            _O, min_age=5, max_age=11, reporter="parent",
        ))
        self._register(Item(
            "CSBI_02", "self_stimulation",
            "Touches or rubs private parts at home.",
            "বাড়িতে গোপন অঙ্গ স্পর্শ করি বা ঘষি।",
            _O, min_age=12, max_age=17, reporter="self",
        ))
        self._register(Item(
            "CSBI_03", "self_stimulation",
            "Touches or rubs own private parts in public places.",
            "প্রকাশ্য স্থানে নিজের গোপন অঙ্গ স্পর্শ করে বা ঘষে।",
            _O, min_age=5, max_age=11, reporter="parent",
        ))
        self._register(Item(
            "CSBI_04", "self_stimulation",
            "Touches or rubs private parts in public places.",
            "প্রকাশ্য স্থানে গোপন অঙ্গ স্পর্শ করি বা ঘষি।",
            _O, min_age=12, max_age=17, reporter="self",
        ))
        self._register(Item(
            "CSBI_05", "self_stimulation",
            "Masturbates (self-stimulates with hand).",
            "হাত দিয়ে নিজের গোপন অঙ্গে যৌন উত্তেজনামূলক কাজ করে।",
            _O, min_age=8, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_06", "self_stimulation",
            "Inserts or tries to insert objects into private parts.",
            "গোপন অঙ্গে কোনো বস্তু ঢোকায় বা ঢোকানোর চেষ্টা করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_07", "self_stimulation",
            "Touches private parts more than necessary when bathing or changing clothes.",
            "গোসল বা কাপড় বদলানোর সময় প্রয়োজনের বেশি গোপন অঙ্গ স্পর্শ করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))

        # Boundary problems (7 items)
        self._register(Item(
            "CSBI_08", "boundary_problems",
            "Touches other people's private parts.",
            "অন্যদের গোপন অঙ্গ স্পর্শ করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_09", "boundary_problems",
            "Tries to undress other children or adults.",
            "অন্য শিশু বা বড়দের কাপড় খোলানোর চেষ্টা করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_10", "boundary_problems",
            "Shows private parts to children or adults outside the immediate family.",
            "পরিবারের বাইরের শিশু বা বড়দের সামনে গোপন অঙ্গ দেখায়।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_11", "boundary_problems",
            "Asks other people to touch their own private parts.",
            "অন্যদের নিজের গোপন অঙ্গ স্পর্শ করতে বলে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_12", "boundary_problems",
            "Undresses in front of others without apparent concern or embarrassment.",
            "সংকোচ ছাড়াই অন্যদের সামনে কাপড় খোলে।",
            _O, min_age=5, max_age=11, reporter="parent",
        ))
        self._register(Item(
            "CSBI_13", "boundary_problems",
            "Touches other people's breasts.",
            "অন্যদের বুক স্পর্শ করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_14", "boundary_problems",
            "Sits very close to or climbs on adults in a sexually provocative way.",
            "বড়দের খুব কাছে বসে বা যৌন উত্তেজনামূলকভাবে উঠে বসে।",
            _O, min_age=5, max_age=11, reporter="parent",
        ))

        # Sexual anxiety (6 items)
        self._register(Item(
            "CSBI_15", "sexual_anxiety",
            "Seems fearful or very uncomfortable when sexual topics come up.",
            "যৌন বিষয় উঠলে ভয় বা অস্বস্তি দেখায়।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_16", "sexual_anxiety",
            "Becomes very upset when asked to undress for medical examination.",
            "চিকিৎসার জন্য কাপড় খুলতে বললে খুব কষ্ট পায় বা রেগে যায়।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_17", "sexual_anxiety",
            "Avoids bathing, changing clothes, or medical examinations.",
            "গোসল, কাপড় বদলানো বা চিকিৎসা পরীক্ষা এড়িয়ে চলে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_18", "sexual_anxiety",
            "Becomes very distressed when discussing private body parts.",
            "গোপন অঙ্গ নিয়ে আলোচনা হলে অনেক কষ্ট বা ভয় দেখায়।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_19", "sexual_anxiety",
            "Has nightmares involving sexual situations or being touched inappropriately.",
            "যৌন পরিস্থিতি বা অনুপযুক্তভাবে স্পর্শ করার স্বপ্ন দেখে।",
            _O, min_age=8, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_20", "sexual_anxiety",
            "Seems frightened of specific people in relation to sexual situations.",
            "যৌন পরিস্থিতির সাথে সম্পর্কিত নির্দিষ্ট কিছু মানুষকে ভয় পায়।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))

        # Sexual interest (8 items)
        self._register(Item(
            "CSBI_21", "sexual_interest",
            "Talks about sexual acts in detail.",
            "যৌন কার্যকলাপ সম্পর্কে বিস্তারিত কথা বলে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_22", "sexual_interest",
            "Initiates sexual games with other children.",
            "অন্য শিশুদের সাথে যৌন খেলা শুরু করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_23", "sexual_interest",
            "Asks adults about sexual activities far beyond age-appropriate curiosity.",
            "বয়সের তুলনায় অনেক বেশি যৌন বিষয়ে বড়দের কাছে প্রশ্ন করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_24", "sexual_interest",
            "Draws pictures of sexual acts or private body parts.",
            "যৌন কার্যকলাপ বা গোপন অঙ্গের ছবি আঁকে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_25", "sexual_interest",
            "Tries to engage in sexual acts with other children.",
            "অন্য শিশুদের সাথে যৌন কার্যকলাপে যুক্ত হওয়ার চেষ্টা করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_26", "sexual_interest",
            "Uses explicit sexual language with adults.",
            "বড়দের সাথে স্পষ্ট যৌন ভাষা ব্যবহার করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_27", "sexual_interest",
            "Shows excessive interest in adult sexual topics (films, magazines).",
            "বড়দের যৌন বিষয়ে (চলচ্চিত্র, পত্রিকা) অতিরিক্ত আগ্রহ দেখায়।",
            _O, min_age=8, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_28", "sexual_interest",
            "Seeks out sexually explicit material on a mobile phone or the internet.",
            "মোবাইল ফোন বা ইন্টারনেটে যৌন সামগ্রী খোঁজে।",
            _O, min_age=10, max_age=17, reporter="parent",
        ))

        # Sexual intrusiveness (5 items)
        self._register(Item(
            "CSBI_29", "sexual_intrusiveness",
            "Tries to touch others' private parts when hugging or playing.",
            "জড়িয়ে ধরার বা খেলার সময় অন্যদের গোপন অঙ্গ স্পর্শ করার চেষ্টা করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_30", "sexual_intrusiveness",
            "Inserts tongue when kissing others.",
            "অন্যদের চুমু দেওয়ার সময় জিহ্বা ঢোকানোর চেষ্টা করে।",
            _O, min_age=8, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_31", "sexual_intrusiveness",
            "Acts out sexual intercourse with dolls, stuffed animals, or other toys.",
            "পুতুল, ভরাট প্রাণী বা অন্য খেলনা দিয়ে যৌন মিলনের অভিনয় করে।",
            _O, min_age=5, max_age=11, reporter="parent",
        ))
        self._register(Item(
            "CSBI_32", "sexual_intrusiveness",
            "Simulates sexual intercourse with another child.",
            "আরেকটি শিশুর সাথে যৌন মিলনের ভান করে।",
            _O, min_age=8, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_33", "sexual_intrusiveness",
            "Makes sexual propositions to peers or younger children.",
            "সমবয়সী বা ছোট শিশুদের কাছে যৌন প্রস্তাব করে।",
            _O, min_age=10, max_age=17, reporter="parent",
        ))

        # Sexual knowledge (3 items)
        self._register(Item(
            "CSBI_34", "sexual_knowledge",
            "Knows details about sexual acts that are clearly inappropriate for their age.",
            "বয়সের জন্য স্পষ্টতই অনুপযুক্ত যৌন কার্যকলাপের বিবরণ জানে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_35", "sexual_knowledge",
            "Uses adult sexual vocabulary words that children their age would not normally know.",
            "এই বয়সের শিশুরা সাধারণত যা জানে না এমন যৌন পরিভাষা ব্যবহার করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_36", "sexual_knowledge",
            "Discusses adult sexual practices in detail (e.g. describes specific sexual acts).",
            "বড়দের যৌন অভ্যাস বিস্তারিতভাবে বর্ণনা করে।",
            _O, min_age=8, max_age=17, reporter="parent",
        ))

        # Voyeuristic behavior (2 items)
        self._register(Item(
            "CSBI_37", "voyeuristic_behavior",
            "Tries to look at people in the bathroom or when they are changing clothes.",
            "বাথরুমে বা কাপড় বদলানোর সময় অন্যদের দেখার চেষ্টা করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))
        self._register(Item(
            "CSBI_38", "voyeuristic_behavior",
            "Peeps through keyholes or around doors to spy on people in private situations.",
            "চাবির ছিদ্র বা দরজার পাশ দিয়ে ব্যক্তিগত মুহূর্তে অন্যদের গোপনে দেখার চেষ্টা করে।",
            _O, min_age=5, max_age=17, reporter="parent",
        ))

    def get_items_for_age(self, age: int) -> list[Item]:
        return [it for it in self._items if it.is_applicable(age)]

    def reporter_mode(self, age: int) -> str:
        """Return 'parent' for ages 5-11, 'self' for ages 12-17."""
        return "parent" if age <= 11 else "self"


# The original stub exposed QuestionnaireInstrument; keep it pointing to the
# SDQ for code that already imports by that name.
QuestionnaireInstrument = BangladeshSDQ
