"""
pipeline.py — Results Pipeline Module
=======================================

Orchestrates the full evaluation flow and transforms raw results
into structured DataFrames for display and export.

Functions:
  - run_pipeline(): evaluate a conversation and return sorted DataFrame
  - results_to_csv(): export results to CSV
  - get_summary_stats(): compute aggregate statistics
"""

import os
import sys

import pandas as pd
import numpy as np

# Add parent directory to path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import evaluator


def run_pipeline(conversation_text: str) -> pd.DataFrame:
    """
    Run the full evaluation pipeline on a conversation turn.

    Steps:
    1. Call evaluator.evaluate_conversation() to get all facet scores
    2. Convert results dict to a structured DataFrame
    3. Replace None scores with "N/A" for display
    4. Sort: scoreable facets first (by score descending),
       non-scoreable at the bottom

    Args:
        conversation_text: The conversation text to evaluate.

    Returns:
        pd.DataFrame with columns: facet, category, score, confidence,
        is_scoreable, scoring_guidance, note
    """
    # Step 1: Get raw results from the evaluator
    results = evaluator.evaluate_conversation(conversation_text)

    # Step 2: Convert to DataFrame
    rows = []
    for facet_name, data in results.items():
        rows.append({
            "facet": facet_name,
            "category": data["category"],
            "score": data["score"],
            "confidence": data["confidence"],
            "is_scoreable": data["is_scoreable"],
            "scoring_guidance": data["scoring_guidance"],
            "note": data["note"],
        })
    df = pd.DataFrame(rows)

    # Add display_name column from cleaned CSV for cleaner UI display
    cleaned_csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "facets_cleaned.csv"
    )
    if os.path.exists(cleaned_csv_path):
        cleaned_df = pd.read_csv(cleaned_csv_path)
        if "display_name" in cleaned_df.columns:
            display_map = dict(zip(cleaned_df["facet"], cleaned_df["display_name"]))
            df["display_name"] = df["facet"].map(display_map).fillna(df["facet"])
        else:
            df["display_name"] = df["facet"]
    else:
        df["display_name"] = df["facet"]

    # Cap confidence at 0.95 to prevent 1.0 values in output
    df['confidence'] = df['confidence'].clip(upper=0.95)

    # Step 3: Replace None scores with "N/A" for display
    df["score"] = df["score"].apply(lambda x: "N/A" if x is None else x)

    # Step 4: Sort — scoreable first (by score desc), non-scoreable at bottom
    # Create sort keys
    df["_is_scoreable_int"] = df["is_scoreable"].astype(int)
    df["_score_numeric"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

    df = df.sort_values(
        by=["_is_scoreable_int", "_score_numeric"],
        ascending=[False, False],  # scoreable first, then high scores first
    ).reset_index(drop=True)

    # Drop helper columns
    df = df.drop(columns=["_is_scoreable_int", "_score_numeric"])

    return df


def results_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the results DataFrame to a CSV file.

    Args:
        df: Results DataFrame from run_pipeline().
        output_path: File path for the output CSV.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute aggregate statistics from the results DataFrame.

    Args:
        df: Results DataFrame from run_pipeline().

    Returns:
        dict with keys:
          - total_facets: int
          - scoreable_count: int
          - non_scoreable_count: int
          - avg_score: float (scoreable only, ignoring "N/A")
          - avg_confidence: float (scoreable only)
          - high_confidence_count: int (confidence > 0.7)
          - category_breakdown: dict {category: avg_score}
    """
    total = len(df)
    scoreable_mask = df["is_scoreable"] == True
    scoreable_df = df[scoreable_mask].copy()
    non_scoreable_count = total - len(scoreable_df)

    # Convert scores to numeric for computation (ignore "N/A")
    scoreable_df["score_num"] = pd.to_numeric(
        scoreable_df["score"], errors="coerce"
    )

    avg_score = float(scoreable_df["score_num"].mean()) if len(scoreable_df) > 0 else 0.0
    avg_confidence = float(scoreable_df["confidence"].mean()) if len(scoreable_df) > 0 else 0.0

    # High confidence count (confidence > 0.7 among scoreable)
    high_conf = int((scoreable_df["confidence"] > 0.7).sum())

    # Category breakdown — average score per category (scoreable only)
    category_breakdown = {}
    if len(scoreable_df) > 0:
        cat_groups = scoreable_df.groupby("category")["score_num"].mean()
        category_breakdown = cat_groups.to_dict()

    return {
        "total_facets": total,
        "scoreable_count": len(scoreable_df),
        "non_scoreable_count": non_scoreable_count,
        "avg_score": round(avg_score, 2),
        "avg_confidence": round(avg_confidence, 2),
        "high_confidence_count": high_conf,
        "category_breakdown": {k: round(v, 2) for k, v in category_breakdown.items()},
    }


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_text = (
        "Thank you so much for helping me! I was really struggling "
        "with this problem, and your explanation made it so clear."
    )
    print("Running pipeline test...")
    df = run_pipeline(test_text)
    print(f"\nResults DataFrame shape: {df.shape}")
    print(df.head(10).to_string(index=False))

    stats = get_summary_stats(df)
    print(f"\nSummary stats: {stats}")
