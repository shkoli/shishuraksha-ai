"""HTP trauma marker definitions and extraction from binary marker vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


# ═══════════════════════════════════════════════════════════════════════════════
# HTPMarker definition
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HTPMarker:
    """One clinically-validated HTP trauma indicator (weights sum to 1.0 across all 20)."""
    name:           str
    description:    str
    weight:         float
    detection_hint: str
    category:       str


# ── 20 markers in canonical order ────────────────────────────────────────────
# Categories: figure (8), house (6), tree (4), global (2)
# Weights within each category sum proportionally; overall total = 1.0

MARKERS: list[HTPMarker] = [
    # ── FIGURE (8) — indices 0-7 ──────────────────────────────────────────
    HTPMarker(
        name="tiny_figure",
        description="Disproportionately small human figure relative to page",
        weight=0.060,
        detection_hint="Figure height < 25% of page height",
        category="figure",
    ),
    HTPMarker(
        name="missing_hands",
        description="Hands absent or concealed (pockets/back)",
        weight=0.055,
        detection_hint="No visible hand endpoints on arm stubs",
        category="figure",
    ),
    HTPMarker(
        name="missing_feet",
        description="Feet or lower limbs omitted",
        weight=0.045,
        detection_hint="Legs terminate without foot representation",
        category="figure",
    ),
    HTPMarker(
        name="faceless_figure",
        description="Facial features absent or heavily obscured",
        weight=0.070,
        detection_hint="Blank oval or cross-hatched face area",
        category="figure",
    ),
    HTPMarker(
        name="limb_distortion",
        description="Limbs markedly asymmetric, oversized, or fragmented",
        weight=0.055,
        detection_hint="Arm/leg length ratio > 2:1 between sides",
        category="figure",
    ),
    HTPMarker(
        name="encapsulation",
        description="Figure enclosed in box, circle, or heavy outline",
        weight=0.065,
        detection_hint="Closed geometric boundary surrounding entire figure",
        category="figure",
    ),
    HTPMarker(
        name="heavy_shading_figure",
        description="Excessive dark shading over body or specific body part",
        weight=0.060,
        detection_hint="Mean pixel intensity in figure region < 80/255",
        category="figure",
    ),
    HTPMarker(
        name="aggressive_imagery",
        description="Weapons, blood, or explicit violent content in drawing",
        weight=0.080,
        detection_hint="Objects identifiable as weapons or violent symbols",
        category="figure",
    ),
    # ── HOUSE (6) — indices 8-13 ─────────────────────────────────────────
    HTPMarker(
        name="no_door",
        description="House drawn without a door (inaccessibility theme)",
        weight=0.045,
        detection_hint="No rectangular opening at ground level of house",
        category="house",
    ),
    HTPMarker(
        name="no_windows",
        description="House has no windows (isolation/withdrawal theme)",
        weight=0.040,
        detection_hint="No window shapes on facade",
        category="house",
    ),
    HTPMarker(
        name="damaged_roof",
        description="Roof is broken, missing, or heavily distorted",
        weight=0.045,
        detection_hint="Irregular or absent roofline",
        category="house",
    ),
    HTPMarker(
        name="chimney_smoke_excessive",
        description="Disproportionately large or agitated smoke from chimney",
        weight=0.030,
        detection_hint="Smoke plume height > chimney height × 3",
        category="house",
    ),
    HTPMarker(
        name="house_transparency",
        description="Interior visible through walls (transparency / x-ray drawing)",
        weight=0.040,
        detection_hint="Internal rooms or furniture visible through opaque walls",
        category="house",
    ),
    HTPMarker(
        name="isolated_house",
        description="House placed at extreme page edge with no ground context",
        weight=0.035,
        detection_hint="House bounding box within 5% margin of page edge",
        category="house",
    ),
    # ── TREE (4) — indices 14-17 ─────────────────────────────────────────
    HTPMarker(
        name="dead_tree",
        description="Tree with no leaves, broken branches, or bark scarring",
        weight=0.045,
        detection_hint="No foliage mass; bare stick branches only",
        category="tree",
    ),
    HTPMarker(
        name="cut_tree",
        description="Tree trunk severed or shown with a visible cut/wound",
        weight=0.055,
        detection_hint="Horizontal cut mark or stump present on trunk",
        category="tree",
    ),
    HTPMarker(
        name="hollow_trunk",
        description="Visible hollow or cavity in the trunk",
        weight=0.040,
        detection_hint="Dark oval or rectangular recess inside trunk outline",
        category="tree",
    ),
    HTPMarker(
        name="falling_tree",
        description="Tree drawn at oblique angle (> 30°) suggesting instability",
        weight=0.035,
        detection_hint="Trunk axis angle from vertical > 30 degrees",
        category="tree",
    ),
    # ── GLOBAL (2) — indices 18-19 ────────────────────────────────────────
    HTPMarker(
        name="heavy_line_pressure",
        description="Overall heavy line pressure indicating tension/anxiety",
        weight=0.045,
        detection_hint="Mean stroke width > 3 px on 224×224 normalised image",
        category="global",
    ),
    HTPMarker(
        name="ground_line_absence",
        description="No ground line drawn; figures float (insecurity theme)",
        weight=0.055,
        detection_hint="No horizontal baseline beneath figures or house",
        category="global",
    ),
]

assert len(MARKERS) == 20, "Exactly 20 HTP markers required"
_WEIGHT_SUM = sum(m.weight for m in MARKERS)
assert abs(_WEIGHT_SUM - 1.0) < 1e-6, f"Marker weights must sum to 1.0, got {_WEIGHT_SUM}"

# Index lookup
_MARKER_BY_NAME: dict[str, HTPMarker] = {m.name: m for m in MARKERS}


# ═══════════════════════════════════════════════════════════════════════════════
# HTPMarkerExtractor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarkerExtractionResult:
    """Output of HTPMarkerExtractor.extract()."""
    marker_scores:  dict[str, float]   # name → weighted score (0.0 or weight×100)
    present_markers: list[str]          # names of activated markers
    burden_score:   float               # composite 0-100
    category_scores: dict[str, float]  # per-category burden contribution

    def __repr__(self) -> str:
        present = ", ".join(self.present_markers) if self.present_markers else "none"
        cats = ", ".join(f"{k}={v:.1f}" for k, v in self.category_scores.items())
        return (
            f"MarkerExtractionResult(\n"
            f"  burden_score={self.burden_score:.2f},\n"
            f"  present_markers=[{present}],\n"
            f"  category_scores={{{cats}}},\n"
            f"  marker_scores={{...{len(self.marker_scores)} entries}}\n"
            f")"
        )


class HTPMarkerExtractor:
    """Converts a 20-dim binary HTP marker vector into clinical scores."""

    def __init__(self, markers: list[HTPMarker] | None = None) -> None:
        self.markers = markers if markers is not None else MARKERS
        if len(self.markers) != 20:
            raise ValueError(f"Expected 20 markers, got {len(self.markers)}")

    # ------------------------------------------------------------------
    def extract(self, binary_vector: list[int] | list[float]) -> MarkerExtractionResult:
        if len(binary_vector) != len(self.markers):
            raise ValueError(
                f"Vector length {len(binary_vector)} != {len(self.markers)} markers"
            )

        marker_scores:   dict[str, float] = {}
        present_markers: list[str]        = []
        category_raw:    dict[str, float] = {}

        for marker, flag in zip(self.markers, binary_vector):
            active = bool(flag)
            score = marker.weight * 100.0 if active else 0.0
            marker_scores[marker.name] = score
            if active:
                present_markers.append(marker.name)
            category_raw[marker.category] = (
                category_raw.get(marker.category, 0.0) + score
            )

        burden_score = sum(marker_scores.values())  # already in [0, 100]

        return MarkerExtractionResult(
            marker_scores    = marker_scores,
            present_markers  = present_markers,
            burden_score     = round(burden_score, 4),
            category_scores  = {k: round(v, 4) for k, v in category_raw.items()},
        )

    def marker_names(self) -> list[str]:
        return [m.name for m in self.markers]

    def get_marker(self, name: str) -> HTPMarker:
        return _MARKER_BY_NAME[name]
