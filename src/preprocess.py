"""
preprocess.py — Facet Preprocessing Module
============================================

Loads the raw Facets_Assignment.csv, cleans and classifies each facet,
determines scoreability, and outputs a cleaned CSV (data/facets_cleaned.csv)
that downstream modules (batcher, evaluator, pipeline) consume.

Run standalone:
    python src/preprocess.py
"""

import os
import re
import pandas as pd

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "data", "Facets_Assignment.csv")
CLEANED_CSV = os.path.join(BASE_DIR, "data", "facets_cleaned.csv")

# ---------------------------------------------------------------------------
# Category keyword mapping
# Each facet is classified by checking if its lowercased name contains any
# keyword from these lists.  Order matters: first match wins.
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "emotion": [
        "happiness", "sadness", "anger", "fear", "compassion",
        "joy", "grief", "mood", "affect", "feeling", "emotional",
        "anxiety", "depression", "warmth", "hostility", "irritab",
        "enthus", "bliss", "merriness", "joyful", "desperat", "content",
        "warmheart", "affin", "genial", "cordial", "morose", "acidity",
        "melanchol", "distress", "loneli", "guilt", "shame", "empathi",
        "sympathi",
    ],
    "cognitive": [
        "reasoning", "memory", "attention", "logic", "intelligence",
        "cognitive", "mental", "arithmetic", "spatial", "analytical",
        "comprehension", "learning", "retention", "processing",
        "numerical", "mathematical", "sequence", "perception",
        "synthesis", "evaluating", "epistemology", "coherence",
        "decisiveness", "judgment", "critical", "problem-solving",
        "creativity", "innovation", "insight", "abstract", "conceptual",
        "working memory", "executive", "metacognit",
    ],
    "linguistic": [
        "language", "spelling", "sentence", "brevity", "grammar",
        "vocabulary", "verbal", "communication", "storytelling",
        "articul", "fluency", "writing", "reading",
        "literacy", "narrative", "discourse", "rhetoric",
        "pronunciation", "comprehension", "listening", "speaking",
    ],
    "safety": [
        "harm", "dishonest", "hostile", "aggress", "danger",
        "violence", "risk", "manipul", "decepti", "threat", "abuse",
        "toxic", "hateful", "discriminat", "prejudic", "bias",
        "exploit", "coercive", "controlling", "passive-aggress",
    ],
    "personality": [
        "openness", "conscientiousness", "extraversion",
        "agreeableness", "neuroticism", "assertive", "introvert",
        "dominant", "submiss", "impulsiv", "persist", "determin",
        "disciplin", "esteem", "self-direct", "selfcontrol",
        "self-control", "vivacity", "genuine", "decency", "classiness",
        "chivalrous", "dogged", "flawless", "ardency", "dauntless",
        "bravery", "courageousness", "boldness", "frankness",
        "outspoken", "rebellious", "brazen", "audacious",
    ],
    "social": [
        "collaboration", "empathy", "leadership", "social",
        "relationship", "community", "cooperation", "trust",
        "influence", "participation", "interpersonal", "cultural",
        "civility", "contribution", "multiculturalism", "alignment",
        "delegation", "teamwork", "networking", "bonding", "belonging",
        "collective", "peer", "group", "inclusion", "diversity",
        "diplomacy", "negotiation", "conflict resolution",
    ],
    "physical": [
        "FSH", "basophil", "hormone", "metabolic", "sleep",
        "caffeine", "chromatin", "immune", "biological", "genetic",
        "medical", "polygenic", "blood", "dietary", "nutrition",
        "macronutrient", "fitness", "exercise", "cardio", "dance",
        "physical", "body", "health", "BMI", "weight", "height",
        "pulse", "heartrate", "vision", "auditory", "motor",
        "movement", "precision",
    ],
}

# Prefixes to strip for the display_name column
# Same rules used in evaluator.py for LLM prompt shortening
DISPLAY_NAME_PREFIXES = [
    "Psychological construct: ",
    "Social-cognition variable: ",
    "Character strength: ",
    "Defense-mechanism tendency: ",
    "Cognitive measure: ",
    "Big Five facet ",
    "Value orientation: ",
    "Emotional-intelligence measure: ",
    "HEXACO domain: ",
]

# Scoring guidance strings keyed by category
SCORING_GUIDANCE = {
    "emotion":      "1=completely absent, 5=strongly expressed in tone",
    "cognitive":    "1=no evidence of this ability, 5=clearly demonstrated",
    "linguistic":   "1=very poor, 5=excellent quality",
    "safety":       "1=safe/absent, 5=highly present/concerning",
    "personality":  "1=trait completely absent, 5=trait strongly present",
    "social":       "1=no social awareness, 5=highly socially aware",
    "physical":     "1=low, 5=high (note: limited by text-only context)",
    "other":        "1=very low, 5=very high",
}

# Keywords that make a facet non-scoreable regardless of category
NON_SCOREABLE_KEYWORDS = [
    "FSH", "basophil", "hormone", "chromatin", "polygenic",
    "caffeine gene", "aura", "I Ching", "hexagram", "sephira",
    "Kabbalah", "Sufi", "Quran", "Hindu", "Buddhist", "Islamic",
    "Jewish", "Sikh", "Bahá'í", "Gnostic", "astrology", "chakra",
    "reiki", "channeling", "Dhikr", "Vrata", "Bhagavad", "Eightfold",
    "Ridván", "Kirtan", "Shabbat", "Sukkot", "Zohar", "Seerah",
    "khatam", "lulav",
]


def classify_category(facet_name: str) -> str:
    """
    Classify a facet into one of the predefined categories using keyword
    matching against the lowercased facet name.  Returns 'other' when no
    keyword matches.
    """
    lower = facet_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lower:
                return category
    return "other"


def determine_scoreability(facet_name: str, category: str):
    """
    Determine whether a facet can be scored from conversation text alone.

    Returns:
        (is_scoreable: bool, reason: str)
    """
    # Rule 1: physical category is never scoreable
    if category == "physical":
        return False, "requires medical/biological data"

    # Rule 2: check for non-scoreable keywords in the facet name
    for kw in NON_SCOREABLE_KEYWORDS:
        if kw.lower() in facet_name.lower():
            # Decide which reason applies
            medical_kws = {"fsh", "basophil", "hormone", "chromatin",
                           "polygenic", "caffeine gene"}
            if kw.lower() in medical_kws:
                return False, "requires medical/biological data"
            else:
                return False, "requires spiritual/religious observance data"

    return True, ""


# ---------------------------------------------------------------------------
# Evaluation-helper column functions
# ---------------------------------------------------------------------------

# Keywords that indicate a behavioral (action/habit) facet
_BEHAVIORAL_KEYWORDS = [
    "frequency", "count", "hours", "sessions", "usage", "trips",
    "participation", "visits", "contributions",
]

# Keywords for score_direction classification
_NEGATIVE_KEYWORDS = [
    "hostil", "harm", "dishon", "aggress", "danger", "violence",
    "manipul", "deceiv", "threat", "abuse", "toxic", "hate",
    "psychoticism", "neuroticism", "impulsiv", "cantankerous",
    "coarse", "disrespect", "passive-aggress", "hysteria", "morose",
    "acidity", "burnout", "depression", "desperat", "fearful",
    "lazy", "sloth", "inefficien", "immaturity", "brazen", "impuden",
]

_POSITIVE_KEYWORDS = [
    "compassion", "kindness", "warmth", "honest", "openness",
    "cooperat", "collaborat", "empath", "civil", "dignity",
    "persever", "determin", "enthus", "happi", "joy", "content",
    "brav", "courageou", "genuine", "decen", "ethical", "trust",
    "wisdom",
]

# Keywords that indicate a facet is not observable from text
_NOT_OBSERVABLE_KEYWORDS = [
    "frequency", "count", "hours", "sessions", "visits", "stamps",
    "km", "mg", "years", "months", "days",
]


def _assign_evaluation_type(row) -> str:
    """Assign evaluation_type based on category and behavioral keywords."""
    facet_lower = row["facet"].lower()

    # Non-scoreable → observational
    if not row["is_scoreable"]:
        return "observational"

    # Check behavioral keywords first (overrides category)
    for kw in _BEHAVIORAL_KEYWORDS:
        if kw in facet_lower:
            return "behavioral"

    # Map by category
    category_map = {
        "emotion": "affective",
        "cognitive": "cognitive",
        "personality": "trait",
        "social": "social",
        "linguistic": "linguistic",
        "safety": "safety",
        "physical": "observational",
    }
    return category_map.get(row["category"], "behavioral")


def _assign_score_direction(facet_name: str) -> str:
    """Assign whether a higher score is positive, negative, or neutral."""
    lower = facet_name.lower()

    for kw in _NEGATIVE_KEYWORDS:
        if kw in lower:
            return "negative"

    for kw in _POSITIVE_KEYWORDS:
        if kw in lower:
            return "positive"

    return "neutral"


def _assign_observable_in_text(row) -> str:
    """Assign how observable a facet is from text alone."""
    # Non-scoreable → not_observable
    if not row["is_scoreable"]:
        return "not_observable"

    # Check for behavioral count-based keywords
    lower = row["facet"].lower()
    for kw in _NOT_OBSERVABLE_KEYWORDS:
        if kw in lower:
            return "not_observable"

    # Directly observable categories
    if row["category"] in ("emotion", "linguistic", "safety"):
        return "directly"

    # Partially observable categories
    return "partially"


def _assign_weight(row) -> float:
    """Assign importance weight based on category and scoreability."""
    if not row["is_scoreable"]:
        return 0.1

    weight_map = {
        "safety": 1.0,
        "emotion": 0.9,
        "linguistic": 0.8,
        "social": 0.7,
        "personality": 0.6,
        "cognitive": 0.5,
        "other": 0.3,
        "physical": 0.1,
    }
    return weight_map.get(row["category"], 0.3)


def _example_phrase(display_name: str, category: str, high: bool) -> str:
    """Generate a short example phrase for a high (5) or low (1) score."""
    name = display_name.strip()

    if high:
        templates = {
            "emotion":      f"I feel deeply {name} about this situation",
            "safety":       f"I will {name} you if you don't comply",
            "personality":  f"I always approach everything with strong {name}",
            "linguistic":   f"The message shows excellent {name} throughout",
            "social":       f"They demonstrated great {name} with the team",
            "cognitive":    f"Their {name} was clearly evident in the solution",
        }
        return templates.get(category, f"High {name} observed in this context")
    else:
        templates = {
            "emotion":      f"No sign of {name} in this message",
            "safety":       f"The message shows no {name} at all",
            "personality":  f"Complete absence of {name} in behavior",
            "linguistic":   f"The message lacks any {name} quality",
            "social":       f"No {name} was shown toward others",
            "cognitive":    f"No evidence of {name} in the response",
        }
        return templates.get(category, f"Low {name} observed in this context")


def load_and_clean() -> pd.DataFrame:
    """
    Load the raw CSV file, clean it, classify facets, and return
    the enriched DataFrame.

    Cleaning steps:
    1. Remove category header rows (ending with ':')
    2. Remove rows where facet starts with digits followed by a dot
    3. Strip extra whitespace
    4. Remove duplicate facets
    """
    # --- Load ---
    df = pd.read_csv(RAW_CSV)
    total_raw = len(df)
    print(f"Raw rows loaded: {total_raw}")

    # --- Clean: remove category headers (rows ending with ':') ---
    df = df[~df["Facets"].str.strip().str.endswith(":")]
    print(f"After removing category headers: {len(df)}")

    # --- Clean: remove rows starting with digits followed by a dot ---
    # Pattern matches lines like "800. Sufi practice..."
    digit_dot_pattern = re.compile(r"^\d+\.\s")
    mask = df["Facets"].apply(lambda x: bool(digit_dot_pattern.match(str(x).strip())))
    df = df[~mask]
    print(f"After removing digit-prefixed rows: {len(df)}")

    # --- Strip whitespace ---
    df["Facets"] = df["Facets"].str.strip()

    # --- Remove duplicates ---
    df = df.drop_duplicates(subset="Facets")
    print(f"After removing duplicates: {len(df)}")

    # Rename column for clarity
    df = df.rename(columns={"Facets": "facet"})
    df = df.reset_index(drop=True)

    # --- FIX 5 Step A: Remove sub-item rows ---
    # Remove rows containing " / " (sub-items)
    df = df[~df["facet"].str.contains(" / ", na=False)]
    # Remove rows with dash followed by a number (e.g. "– 3")
    dash_num_pattern = re.compile(r"\u2013\s*\d")
    df = df[~df["facet"].apply(lambda x: bool(dash_num_pattern.search(str(x))))]
    df = df.reset_index(drop=True)
    print(f"After removing sub-items: {len(df)}")

    # --- Add classification columns ---
    df["category"] = df["facet"].apply(classify_category)
    df["scoring_guidance"] = df["category"].map(SCORING_GUIDANCE)
    score_results = df.apply(
        lambda row: determine_scoreability(row["facet"], row["category"]),
        axis=1,
    )
    df["is_scoreable"] = score_results.apply(lambda x: x[0])
    df["not_scoreable_reason"] = score_results.apply(lambda x: x[1])

    # --- FIX 5 Step B: Add display_name column ---
    # Strip known long prefixes for cleaner display
    def _make_display_name(facet_name: str) -> str:
        for prefix in DISPLAY_NAME_PREFIXES:
            if facet_name.startswith(prefix):
                return facet_name[len(prefix):]
        return facet_name

    df["display_name"] = df["facet"].apply(_make_display_name)

    # --- Add evaluation-helper columns ---
    df["evaluation_type"] = df.apply(_assign_evaluation_type, axis=1)
    df["score_direction"] = df["facet"].apply(_assign_score_direction)
    df["observable_in_text"] = df.apply(_assign_observable_in_text, axis=1)
    df["weight"] = df.apply(_assign_weight, axis=1)
    df["example_high_score"] = df.apply(
        lambda r: _example_phrase(r["display_name"], r["category"], high=True), axis=1
    )
    df["example_low_score"] = df.apply(
        lambda r: _example_phrase(r["display_name"], r["category"], high=False), axis=1
    )

    return df


def save_cleaned(df: pd.DataFrame) -> None:
    """Save the cleaned DataFrame to data/facets_cleaned.csv."""
    os.makedirs(os.path.dirname(CLEANED_CSV), exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False)
    print(f"\nCleaned CSV saved to: {CLEANED_CSV}")


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the cleaned data."""
    total = len(df)
    scoreable = df["is_scoreable"].sum()
    non_scoreable = total - scoreable

    print("\n" + "=" * 60)
    print("FACET PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"\nTotal facets after cleaning: {total}")
    print(f"  |-- Scoreable:     {scoreable}")
    print(f"  +-- Non-scoreable: {non_scoreable}")

    # Non-scoreable reason breakdown
    if non_scoreable > 0:
        print("\nNon-scoreable reason counts:")
        reason_counts = (
            df[~df["is_scoreable"]]["not_scoreable_reason"]
            .value_counts()
        )
        for reason, count in reason_counts.items():
            print(f"  - {reason}: {count}")

    # Category distribution
    print("\nCategory distribution:")
    cat_dist = df["category"].value_counts().sort_index()
    print(cat_dist.to_string())

    # New column distributions
    if "evaluation_type" in df.columns:
        print("\nevaluation_type distribution:")
        print(df["evaluation_type"].value_counts().sort_index().to_string())

    if "score_direction" in df.columns:
        print("\nscore_direction distribution:")
        print(df["score_direction"].value_counts().sort_index().to_string())

    if "observable_in_text" in df.columns:
        print("\nobservable_in_text distribution:")
        print(df["observable_in_text"].value_counts().sort_index().to_string())

    if "weight" in df.columns:
        print("\nweight distribution:")
        print(df["weight"].value_counts().sort_index().to_string())

    # Sample rows showing all new columns
    print("\nSample rows (new columns):")
    new_cols = ["facet", "category", "evaluation_type", "score_direction",
                "observable_in_text", "weight", "example_high_score",
                "example_low_score"]
    existing_cols = [c for c in new_cols if c in df.columns]
    print(df[existing_cols].head(5).to_string(index=False))

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Entry point — run this file standalone to generate facets_cleaned.csv
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_clean()
    save_cleaned(df)
    print_summary(df)
