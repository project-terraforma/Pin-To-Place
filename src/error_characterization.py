"""
error_characterization.py
=========================
Reliability layer for the Pin-To-Place model.

For each place, this module:
  1. Computes the offset between the current Overture pin and ground truth (if missing).
  2. Classifies offset severity into four buckets.
  3. Assigns one or more failure modes based on category, offset, and LLM annotation.
  4. Computes a risk score (low / medium / high).
  5. Recommends an action: keep_original_pin | review_manually | allow_model_correction.

This is intentionally a *modular add-on* — it reads an existing CSV and writes a new one.
It does NOT touch the training pipeline or model weights.

Usage (CLI):
    python src/error_characterization.py \
        --input  data/processed/ground_truth_combined.csv \
        --output data/processed/error_characterization.csv

Or import individual functions in a notebook:
    from src.error_characterization import characterize_errors
    result_df = characterize_errors(df)

Authors: project-terraforma / errorChar branch
"""

import argparse
import json
import sys
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────
# 1.  OFFSET CALCULATION
# ─────────────────────────────────────────────

def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine great-circle distance in meters.
    Replicates src/metrics.py so this module has zero internal dependencies.
    """
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def ensure_offset_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    If `offset_haversine_m` is already in the DataFrame, leave it alone.
    Otherwise compute it from current_lat/lon vs gt_lat/gt_lon columns.

    Supports two common column naming conventions:
      - current_lat / current_lon  (ground_truth_combined.csv)
      - lat / lon                  (older pipeline outputs)
    """
    df = df.copy()

    # Already computed — nothing to do
    if "offset_haversine_m" in df.columns:
        return df

    # Detect which column names are present
    lat_col = "current_lat" if "current_lat" in df.columns else "lat"
    lon_col = "current_lon" if "current_lon" in df.columns else "lon"

    required = {lat_col, lon_col, "gt_lat", "gt_lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot compute offset. Missing columns: {missing}. "
            "Please supply a CSV that has current lat/lon and gt_lat/gt_lon."
        )

    df["offset_haversine_m"] = df.apply(
        lambda r: haversine_meters(r[lat_col], r[lon_col], r["gt_lat"], r["gt_lon"]),
        axis=1,
    )
    print(f"  [offset] Computed offset_haversine_m from {lat_col}/{lon_col} → gt_lat/gt_lon")
    return df


# ─────────────────────────────────────────────
# 2.  OFFSET SEVERITY CLASSIFICATION
# ─────────────────────────────────────────────

# Severity thresholds (meters)
SEVERITY_THRESHOLDS = {
    "already_accurate": (0, 5),     # pin is effectively correct
    "minor_offset":     (5, 15),    # small drift, usually acceptable
    "moderate_offset":  (15, 40),   # notable; worth reviewing
    "large_offset":     (40, float("inf")),  # definitely wrong
}


def classify_severity(offset_m: float) -> str:
    """
    Map a numeric offset (metres) to a named severity bucket.

    Buckets (from OKR taxonomy):
        already_accurate  :  0 – 5 m
        minor_offset      :  5 – 15 m
        moderate_offset   : 15 – 40 m
        large_offset      :  > 40 m
    """
    if offset_m <= 5:
        return "already_accurate"
    elif offset_m <= 15:
        return "minor_offset"
    elif offset_m <= 40:
        return "moderate_offset"
    else:
        return "large_offset"


# ─────────────────────────────────────────────
# 3.  FAILURE MODE ASSIGNMENT
# ─────────────────────────────────────────────

# Keywords that suggest each failure mode when found in the LLM reasoning text.
# We check for these strings (case-insensitive) in gt_reasoning / annotation fields.
FAILURE_MODE_KEYWORDS = {
    "entrance_mismatch": [
        "entrance", "entry", "front door", "storefront", "access point",
    ],
    "centroid_bias": [
        "center", "centroid", "middle", "geometric center",
    ],
    "parking_lot_bias": [
        "parking", "parking lot", "parking area", "lot entrance",
    ],
    "pedestrian_access_issue": [
        "sidewalk", "pedestrian", "crosswalk", "curb cut", "accessible path",
        "no sidewalk", "no pedestrian",
    ],
    "multi_tenant_risk": [
        "multi-tenant", "multi tenant", "suite", "unit", "mall", "shopping center",
        "strip mall", "floor",
    ],
    "open_space_ambiguity": [
        "park", "open space", "trail", "campground", "field", "green space",
        "no building", "no structure",
    ],
    "rural_or_sparse_area_risk": [
        "rural", "sparse", "remote", "undeveloped", "no nearby building",
        "residential area",
    ],
}

# Category strings that are strong signals for specific failure modes
# (even when LLM text doesn't explicitly say so)
CATEGORY_FAILURE_MAP = {
    "parking_lot_bias":       ["parking", "garage", "car_park"],
    "multi_tenant_risk":      ["mall", "shopping_center", "strip_mall", "plaza"],
    "open_space_ambiguity":   ["park", "campground", "resort", "open_space",
                               "national_park", "recreation_area", "beach"],
    "rural_or_sparse_area_risk": ["campground", "farm", "ranch", "rural"],
    "pedestrian_access_issue": ["transit_station", "bus_stop", "ferry_terminal"],
}


def assign_failure_modes(row: pd.Series) -> list[str]:
    """
    Return a list of failure mode labels for a single place row.

    Logic (in priority order):
      1. geocoder_disagreement  — detected if geocoder source columns exist and diverge.
      2. Annotation keyword matching  — scan gt_reasoning / llm_annotation text.
      3. Category-based rules  — apply CATEGORY_FAILURE_MAP.
      4. Structural rules  — based on tier_label, place_complexity, pin_ambiguity,
         parking_lot_crossing, sidewalk_visible.
      5. Multi-tenant tier  — always flagged for multi_tenant_risk.
    """
    modes = set()
    offset = float(row.get("offset_haversine_m", 0) or 0)

    # ── Rule 0: geocoder disagreement ──────────────────────────────────────
    # Look for geocoder source columns (e.g. geocoder_1_lat, geocoder_2_lat, …)
    geocoder_cols = [c for c in row.index if c.startswith("geocoder_") and c.endswith("_lat")]
    if len(geocoder_cols) >= 2:
        geo_lats = [float(row[c]) for c in geocoder_cols if pd.notna(row[c])]
        if geo_lats and (max(geo_lats) - min(geo_lats)) * 111_320 > 20:  # >20 m spread
            modes.add("geocoder_disagreement")

    # ── Rule 1: Annotation text keyword matching ──────────────────────────
    annotation_text = ""
    for col in ["gt_reasoning", "llm_annotation", "annotation", "reasoning"]:
        if col in row.index and pd.notna(row[col]):
            annotation_text += " " + str(row[col]).lower()

    for mode, keywords in FAILURE_MODE_KEYWORDS.items():
        if any(kw in annotation_text for kw in keywords):
            modes.add(mode)

    # ── Rule 2: Category-based rules ─────────────────────────────────────
    category = str(row.get("category_primary", "") or "").lower()
    for mode, cat_patterns in CATEGORY_FAILURE_MAP.items():
        if any(pat in category for pat in cat_patterns):
            modes.add(mode)

    # ── Rule 3: Structural column rules ──────────────────────────────────
    tier = str(row.get("tier_label", "") or "").lower()
    complexity = str(row.get("place_complexity", "") or "").lower()
    ambiguity = str(row.get("pin_ambiguity", "") or "").lower()

    if "multi_tenant" in tier or complexity == "multi_tenant":
        modes.add("multi_tenant_risk")

    if "open_space" in tier:
        modes.add("open_space_ambiguity")

    if "no_building" in tier and offset > 5:
        # No physical structure to anchor the pin — centroid often used
        modes.add("centroid_bias")

    if ambiguity == "high" and offset > 10:
        # LLM annotator found the location genuinely ambiguous
        modes.add("entrance_mismatch")

    # parking_lot_crossing column (from task_aware_evaluation)
    if row.get("parking_lot_crossing") is True:
        modes.add("parking_lot_bias")

    # sidewalk_visible column — False means no visible pedestrian path
    if row.get("sidewalk_visible") is False:
        modes.add("pedestrian_access_issue")

    # If nothing matched but offset is large, flag centroid_bias as default
    if not modes and offset > 40:
        modes.add("centroid_bias")

    # Return sorted list for deterministic output
    return sorted(modes) if modes else ["none"]


# ─────────────────────────────────────────────
# 4.  RISK SCORE
# ─────────────────────────────────────────────

def compute_risk_score(row: pd.Series, failure_modes: list[str]) -> str:
    """
    Combine severity, failure modes, and annotation confidence into a risk level.

    high_risk   → pin is likely wrong AND moving it could hurt users
    medium_risk → pin may be wrong; human review advised
    low_risk    → pin is probably fine; safe to leave as-is

    The scoring is intentionally transparent so students can audit it.
    """
    score = 0
    offset = float(row.get("offset_haversine_m", 0) or 0)
    confidence = float(row.get("gt_confidence", 1.0) or 1.0)

    # ── Offset points ──────────────────────────────────────────────────────
    if offset > 40:
        score += 3
    elif offset > 15:
        score += 2
    elif offset > 5:
        score += 1
    # 0–5 m → 0 points

    # ── Failure mode points ────────────────────────────────────────────────
    HIGH_IMPACT_MODES = {
        "entrance_mismatch", "multi_tenant_risk", "parking_lot_bias",
        "pedestrian_access_issue", "geocoder_disagreement",
    }
    medium_impact_modes = {
        "centroid_bias", "open_space_ambiguity",
        "rural_or_sparse_area_risk",
    }
    for mode in failure_modes:
        if mode in HIGH_IMPACT_MODES:
            score += 2
        elif mode in medium_impact_modes:
            score += 1

    # ── LLM confidence penalty ─────────────────────────────────────────────
    # Low annotator confidence = the ground truth itself is uncertain
    if confidence < 0.5:
        score += 1

    # ── Ambiguity field ───────────────────────────────────────────────────
    if str(row.get("pin_ambiguity", "")).lower() == "high":
        score += 1

    # ── Map score → label ─────────────────────────────────────────────────
    if score >= 5:
        return "high_risk"
    elif score >= 2:
        return "medium_risk"
    else:
        return "low_risk"


# ─────────────────────────────────────────────
# 5.  ACTION RECOMMENDATION
# ─────────────────────────────────────────────

def recommend_action(row: pd.Series, severity: str, risk: str) -> str:
    """
    Decide what to do with this pin.

    keep_original_pin     → pin is already correct; do NOT move it (baseline protection)
    review_manually       → uncertain; a human should look before any change
    allow_model_correction → model correction is safe; likely an improvement

    This directly supports OKR 2.1:
    "Maintain 0.0m median baseline offset while ensuring regression rate < 1%."
    By labelling pins as keep_original_pin we prevent the model from
    accidentally moving accurate pins.
    """
    confidence = float(row.get("gt_confidence", 1.0) or 1.0)
    should_move = row.get("should_move", None)

    # Ground-truth annotator explicitly said DO NOT move
    if should_move is False:
        return "keep_original_pin"

    # Pin is already accurate (within 5 m) and risk is low
    if severity == "already_accurate" and risk == "low_risk":
        return "keep_original_pin"

    # High risk → always escalate to human
    if risk == "high_risk":
        return "review_manually"

    # LLM annotator was uncertain → review first
    if confidence < 0.6:
        return "review_manually"

    # Medium risk with moderate/large offset → model can try to fix it
    if severity in ("moderate_offset", "large_offset") and risk == "medium_risk":
        return "allow_model_correction"

    # Minor offset, low/medium risk → keep the pin
    return "keep_original_pin"


# ─────────────────────────────────────────────
# 6.  MAIN CHARACTERIZATION FUNCTION
# ─────────────────────────────────────────────

def characterize_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full error characterization pipeline on a DataFrame.

    Adds four new columns:
        offset_severity   – already_accurate / minor_offset / moderate_offset / large_offset
        failure_modes     – pipe-separated list (e.g. "entrance_mismatch|multi_tenant_risk")
        risk_level        – low_risk / medium_risk / high_risk
        action            – keep_original_pin / review_manually / allow_model_correction

    Returns the enriched DataFrame (does not mutate the input).
    """
    df = ensure_offset_column(df)

    severities = []
    failure_modes_list = []
    risks = []
    actions = []

    for _, row in df.iterrows():
        offset = float(row.get("offset_haversine_m", 0) or 0)

        sev = classify_severity(offset)
        modes = assign_failure_modes(row)
        risk = compute_risk_score(row, modes)
        action = recommend_action(row, sev, risk)

        severities.append(sev)
        failure_modes_list.append("|".join(modes))
        risks.append(risk)
        actions.append(action)

    result = df.copy()
    result["offset_severity"] = severities
    result["failure_modes"] = failure_modes_list
    result["risk_level"] = risks
    result["action"] = actions

    return result


# ─────────────────────────────────────────────
# 7.  SUMMARY STATISTICS
# ─────────────────────────────────────────────

def generate_summary(df: pd.DataFrame) -> dict:
    """
    Produce a human-readable summary dict with:
      - failure mode counts
      - risk level counts
      - median offset by category
      - top-5 highest-risk examples
      - % pins that should NOT be moved (baseline protection metric)

    OKR connection:
      `pct_keep_original` tells you how many pins the system will protect.
      The complement (`pct_allow_correction`) should not exceed the regression
      budget from OKR 2.1 (keep regression rate < 1%).
    """
    summary = {}

    # ── Failure mode counts ────────────────────────────────────────────────
    # Each row may have multiple modes (pipe-separated), so we split them out.
    all_modes = []
    for modes_str in df["failure_modes"]:
        all_modes.extend(modes_str.split("|"))

    mode_counts = {}
    for m in all_modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1
    summary["failure_mode_counts"] = dict(
        sorted(mode_counts.items(), key=lambda x: -x[1])
    )

    # ── Risk level counts ──────────────────────────────────────────────────
    summary["risk_level_counts"] = df["risk_level"].value_counts().to_dict()

    # ── Action counts ──────────────────────────────────────────────────────
    summary["action_counts"] = df["action"].value_counts().to_dict()

    # ── Baseline protection metric (key OKR metric) ────────────────────────
    n_total = len(df)
    n_keep = (df["action"] == "keep_original_pin").sum()
    n_allow = (df["action"] == "allow_model_correction").sum()
    summary["pct_keep_original"] = round(n_keep / n_total * 100, 2)
    summary["pct_allow_correction"] = round(n_allow / n_total * 100, 2)
    summary["pct_review_manually"] = round(
        (df["action"] == "review_manually").sum() / n_total * 100, 2
    )

    # ── Median offset by category ─────────────────────────────────────────
    if "category_primary" in df.columns:
        median_by_cat = (
            df.groupby("category_primary")["offset_haversine_m"]
            .median()
            .sort_values(ascending=False)
            .head(20)
            .round(2)
            .to_dict()
        )
        summary["median_offset_by_category_top20"] = median_by_cat

    # ── Highest-risk examples ──────────────────────────────────────────────
    high_risk = df[df["risk_level"] == "high_risk"].copy()
    high_risk_sorted = high_risk.sort_values("offset_haversine_m", ascending=False)
    cols_to_show = [c for c in ["id", "name", "category_primary", "tier_label",
                                 "offset_haversine_m", "failure_modes",
                                 "risk_level", "action"] if c in df.columns]
    summary["highest_risk_examples"] = (
        high_risk_sorted[cols_to_show].head(5).to_dict(orient="records")
    )

    # ── Severity distribution ──────────────────────────────────────────────
    summary["severity_counts"] = df["offset_severity"].value_counts().to_dict()

    return summary


def print_summary(summary: dict) -> None:
    """Pretty-print the summary dict to stdout."""
    print("\n" + "═" * 60)
    print("  ERROR CHARACTERIZATION SUMMARY")
    print("═" * 60)

    print("\n📊 Offset Severity Distribution:")
    for k, v in summary.get("severity_counts", {}).items():
        print(f"    {k:<25}  {v:>5}")

    print("\n⚠️  Failure Mode Counts:")
    for k, v in summary.get("failure_mode_counts", {}).items():
        print(f"    {k:<30}  {v:>5}")

    print("\n🔴 Risk Level Counts:")
    for k, v in summary.get("risk_level_counts", {}).items():
        print(f"    {k:<20}  {v:>5}")

    print("\n✅ Action Recommendation Counts:")
    for k, v in summary.get("action_counts", {}).items():
        print(f"    {k:<30}  {v:>5}")

    print("\n🛡️  Baseline Protection Metrics (OKR 2.1):")
    print(f"    Pins labelled keep_original_pin : {summary['pct_keep_original']:.1f}%")
    print(f"    Pins labelled allow_correction  : {summary['pct_allow_correction']:.1f}%")
    print(f"    Pins labelled review_manually   : {summary['pct_review_manually']:.1f}%")
    print(
        f"\n    ► If regression rate must stay < 1%, at most "
        f"{summary['pct_allow_correction']:.1f}% of pins "
        "should be handed to model correction."
    )

    print("\n📍 Top Median Offset by Category (top 10):")
    for k, v in list(summary.get("median_offset_by_category_top20", {}).items())[:10]:
        print(f"    {k:<35}  {v:>6.1f} m")

    print("\n🚨 Highest-Risk Pin Examples:")
    for ex in summary.get("highest_risk_examples", []):
        print(
            f"    [{ex.get('risk_level','?')}] {ex.get('name','?')[:35]:<35} "
            f"  offset={ex.get('offset_haversine_m',0):.1f}m "
            f"  modes={ex.get('failure_modes','')}"
        )
    print("\n" + "═" * 60)


# ─────────────────────────────────────────────
# 8.  CLI ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Error Characterization — reliability layer for the Pin-To-Place model.\n"
            "Reads a ground-truth CSV and outputs an enriched CSV with failure modes,\n"
            "risk levels, and action recommendations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV (e.g. data/processed/ground_truth_combined.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/error_characterization.csv",
        help="Path to write the output CSV (default: data/processed/error_characterization.csv)",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional: also write summary statistics to this JSON file.",
    )
    parser.add_argument(
        "--no-print-summary",
        action="store_true",
        help="Suppress the printed summary table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load input ─────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── Run characterization ───────────────────────────────────────────────
    print("Running error characterization …")
    result_df = characterize_errors(df)

    # ── Write output CSV ───────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"  Wrote {len(result_df):,} rows → {output_path}")

    # ── Generate and print summary ─────────────────────────────────────────
    summary = generate_summary(result_df)

    if not args.no_print_summary:
        print_summary(summary)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # highest_risk_examples may contain NaN — coerce to None for JSON
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=lambda x: None if pd.isna(x) else x)
        print(f"  Wrote summary → {summary_path}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
