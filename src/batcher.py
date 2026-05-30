"""
SCALING ARCHITECTURE NOTE:
Facets are processed in batches of 20 per LLM call.
Current: ~200 scoreable facets = ~10 batches = ~10 API calls
At 5000 facets: ~250 batches = ~250 API calls
No code changes needed - just more iterations of the same loop.
This is the architectural decision that satisfies the
'must support 5000 facets without redesign' requirement.
"""

"""
batcher.py — Facet Batching Module
====================================

Loads the cleaned facet data and provides functions to split scoreable
facets into batches for LLM evaluation, and to retrieve non-scoreable
facets separately.

Batch size of 20 is intentional:
  - Small enough for accurate per-facet scoring (no LLM confusion)
  - Large enough to minimize total API calls
  - Scales linearly: 5000 facets → 250 batches, zero code changes
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_CSV = os.path.join(BASE_DIR, "data", "facets_cleaned.csv")

# Batch size — the number of facets sent per LLM call
BATCH_SIZE = 20


def _load_cleaned_data() -> pd.DataFrame:
    """Load the cleaned facets CSV. Raises FileNotFoundError if missing."""
    if not os.path.exists(CLEANED_CSV):
        raise FileNotFoundError(
            f"Cleaned CSV not found at {CLEANED_CSV}. "
            "Run `python src/preprocess.py` first."
        )
    return pd.read_csv(CLEANED_CSV)


def get_scoreable_batches() -> list:
    """
    Filter only scoreable facets and split them into batches of BATCH_SIZE.

    Returns:
        list[list[dict]] — each inner list is a batch containing dicts
        with keys: facet_name, category, scoring_guidance
    """
    df = _load_cleaned_data()
    scoreable = df[df["is_scoreable"] == True].reset_index(drop=True)

    # Build list of facet dicts
    facet_dicts = []
    for _, row in scoreable.iterrows():
        facet_dicts.append({
            "facet_name": row["facet"],
            "category": row["category"],
            "scoring_guidance": row["scoring_guidance"],
        })

    # Split into batches of BATCH_SIZE
    batches = [
        facet_dicts[i : i + BATCH_SIZE]
        for i in range(0, len(facet_dicts), BATCH_SIZE)
    ]

    return batches


def get_non_scoreable_facets() -> list:
    """
    Filter rows where is_scoreable is False.

    Returns:
        list[dict] — each dict has keys: facet_name, category,
                      not_scoreable_reason
    """
    df = _load_cleaned_data()
    non_scoreable = df[df["is_scoreable"] == False].reset_index(drop=True)

    results = []
    for _, row in non_scoreable.iterrows():
        results.append({
            "facet_name": row["facet"],
            "category": row["category"],
            "not_scoreable_reason": row["not_scoreable_reason"],
        })

    return results


def get_all_scoreable_facets() -> list:
    """
    Return a flat list of all scoreable facet names.

    Returns:
        list[str] — facet names where is_scoreable is True
    """
    df = _load_cleaned_data()
    scoreable = df[df["is_scoreable"] == True]
    return scoreable["facet"].tolist()


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    batches = get_scoreable_batches()
    non_scoreable = get_non_scoreable_facets()
    all_scoreable = get_all_scoreable_facets()

    print(f"Total scoreable facets : {len(all_scoreable)}")
    print(f"Number of batches      : {len(batches)}")
    print(f"Batch size             : {BATCH_SIZE}")
    print(f"Non-scoreable facets   : {len(non_scoreable)}")

    if batches:
        print(f"\nSample batch (first 3 items from batch 1):")
        for item in batches[0][:3]:
            print(f"  - {item['facet_name']} [{item['category']}]")
