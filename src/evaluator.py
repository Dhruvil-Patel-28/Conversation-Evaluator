"""
evaluator.py — LLM-Based Conversation Evaluator
=================================================

Uses Groq API with Llama 3.1 8B Instant to score conversation text
against batches of psychological, linguistic, and behavioral facets.

Key design decisions:
  - Markdown stripping before JSON parsing (LLMs sometimes wrap output)
  - Fuzzy name matching to handle slight LLM-returned name variations
  - Retry logic with stricter prompt on first parse failure
  - Graceful fallback (score=3, confidence=0.0) if both attempts fail
  - time.sleep(1) between batch calls for Groq rate-limit compliance
"""

import json
import logging
import os
import re
import time
import sys

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# Add parent directory to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import batcher

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # Load GROQ_API_KEY from .env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL = "llama-3.1-8b-instant"

# System prompt — instructs the LLM on its role and output format
SYSTEM_PROMPT = (
    "You are an expert conversation evaluator. "
    "Analyze conversation turns and score psychological, linguistic, "
    "and behavioral facets accurately. "
    "Always respond with valid JSON only. No explanation, no markdown, "
    "no code blocks. Raw JSON only."
)


def _get_client() -> Groq:
    """Instantiate and return a Groq client using the env-loaded API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Copy .env.example to .env and add your key."
        )
    return Groq(api_key=api_key)


def _strip_markdown(text: str) -> str:
    """
    Remove markdown code-block wrappers (```json ... ``` or ``` ... ```)
    that LLMs sometimes add around their JSON responses.
    """
    # Remove ```json or ``` at start and ``` at end
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _fuzzy_match_facet(returned_name: str, expected_names: list) -> str:
    """
    Match an LLM-returned facet name to the expected list using
    lowercased, stripped comparison.  Returns the expected name if
    matched, otherwise returns the original returned_name.
    """
    returned_lower = returned_name.strip().lower()
    for expected in expected_names:
        if expected.strip().lower() == returned_lower:
            return expected
    return returned_name


def _build_user_prompt(conversation_text: str, batch: list) -> str:
    """
    Build the user prompt for a single batch of facets.

    Args:
        conversation_text: The conversation turn to evaluate.
        batch: List of dicts with keys: facet_name, scoring_guidance.

    Returns:
        Formatted user prompt string.
    """
    # Build numbered facet list
    facet_lines = []
    for i, item in enumerate(batch, 1):
        facet_lines.append(
            f"{i}. {item['facet_name']} - {item['scoring_guidance']}"
        )
    facet_list_str = "\n".join(facet_lines)

    # Build example JSON keys from batch
    example_keys = [item["facet_name"] for item in batch[:2]]
    example_json = "{\n"
    if len(example_keys) >= 1:
        example_json += f'    "{example_keys[0]}": {{"score": 3, "confidence": 0.85}}'
    if len(example_keys) >= 2:
        example_json += f',\n    "{example_keys[1]}": {{"score": 1, "confidence": 0.60}}'
    example_json += "\n}"

    prompt = (
        f"Conversation turn:\n{conversation_text}\n\n"
        f"Score each facet below from 1 to 5:\n"
        f"1 = extremely low or completely absent\n"
        f"2 = low or slightly present\n"
        f"3 = moderate or somewhat present\n"
        f"4 = high or clearly present\n"
        f"5 = extremely high or strongly dominant\n\n"
        f"Also provide confidence (0.0 to 1.0) indicating how confident "
        f"you are based on available text evidence.\n"
        f"Low confidence means the facet is hard to judge from this text.\n\n"
        f"Facets to score:\n{facet_list_str}\n\n"
        f"Respond ONLY in this exact JSON format, no other text:\n"
        f"{example_json}"
    )
    return prompt


def evaluate_batch(conversation_text: str, batch: list) -> dict:
    """
    Evaluate a single batch of facets against a conversation turn.

    Args:
        conversation_text: The conversation text to evaluate.
        batch: List of dicts with keys: facet_name, category,
               scoring_guidance.

    Returns:
        dict mapping facet_name → {score: int, confidence: float}
    """
    client = _get_client()
    expected_names = [item["facet_name"] for item in batch]

    # --- FIX 4: Shorten long-prefix facet names for the LLM prompt ---
    # Prefixes to strip (local to this call, not global)
    prefixes_to_strip = [
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

    short_name_map = {}  # short_name → original_name
    shortened_batch = []
    for item in batch:
        original = item["facet_name"]
        short = original
        for prefix in prefixes_to_strip:
            if short.startswith(prefix):
                short = short[len(prefix):]
                break
        short_name_map[short] = original
        shortened_batch.append({
            "facet_name": short,
            "category": item["category"],
            "scoring_guidance": item["scoring_guidance"],
        })

    # Build prompt using shortened names
    short_expected = [item["facet_name"] for item in shortened_batch]
    user_prompt = _build_user_prompt(conversation_text, shortened_batch)

    # --- First attempt ---
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content
        cleaned = _strip_markdown(raw)
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"First parse attempt failed: {e}. Retrying...")
        parsed = None

    # --- Retry with stricter prompt ---
    if parsed is None:
        try:
            stricter_prompt = (
                user_prompt + "\n\nIMPORTANT: Your response must be ONLY "
                "valid JSON. No markdown, no explanation, no code blocks. "
                "Start directly with { and end with }."
            )
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": stricter_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content
            cleaned = _strip_markdown(raw)
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                f"Retry also failed: {e}. "
                f"Using fallback scores (score=3, confidence=0.0) "
                f"for this batch."
            )
            # Fallback: assign default scores using original names
            return {
                name: {"score": 3, "confidence": 0.0}
                for name in expected_names
            }

    # --- Map LLM-returned keys back to original facet names ---
    result = {}
    for returned_key, values in parsed.items():
        # First try fuzzy matching against short names
        matched_short = _fuzzy_match_facet(returned_key, short_expected)
        # Then map short name back to original name
        original_name = short_name_map.get(matched_short, matched_short)
        score = values.get("score", 3)
        confidence = values.get("confidence", 0.0)
        # Clamp values to valid ranges
        score = max(1, min(5, int(score)))
        confidence = max(0.0, min(1.0, float(confidence)))
        result[original_name] = {"score": score, "confidence": confidence}

    # Fill in any facets the LLM missed (using original names)
    for name in expected_names:
        if name not in result:
            logger.warning(
                f"LLM did not return score for '{name}'. "
                f"Using fallback."
            )
            result[name] = {"score": 3, "confidence": 0.0}

    return result


def evaluate_non_scoreable(non_scoreable_facets: list) -> dict:
    """
    Generate results for non-scoreable facets without any API call.

    Args:
        non_scoreable_facets: List of dicts with keys: facet_name,
                              category, not_scoreable_reason.

    Returns:
        dict mapping facet_name → {score: None, confidence: 0.0,
                                    note: reason_string}
    """
    result = {}
    for item in non_scoreable_facets:
        result[item["facet_name"]] = {
            "score": None,
            "confidence": 0.0,
            "note": item["not_scoreable_reason"],
        }
    return result


def evaluate_conversation(conversation_text: str) -> dict:
    """
    Full evaluation pipeline: score all facets for a conversation turn.

    Steps:
    1. Get scoreable batches from batcher
    2. Get non-scoreable facets from batcher
    3. Loop through batches with progress bar, calling evaluate_batch()
    4. Handle non-scoreable facets (no API call)
    5. Merge everything into a unified result dict

    Args:
        conversation_text: The conversation text to evaluate.

    Returns:
        dict mapping facet_name → {
            score: int or None,
            confidence: float,
            category: str,
            scoring_guidance: str,
            is_scoreable: bool,
            note: str
        }
    """
    # Get batches and non-scoreable lists
    scoreable_batches = batcher.get_scoreable_batches()
    non_scoreable_list = batcher.get_non_scoreable_facets()

    # Load cleaned data for category/guidance lookup
    import pandas as pd
    cleaned_df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "facets_cleaned.csv"
        )
    )
    # Build lookup dicts
    category_lookup = dict(zip(cleaned_df["facet"], cleaned_df["category"]))
    guidance_lookup = dict(zip(cleaned_df["facet"], cleaned_df["scoring_guidance"]))

    # --- Evaluate scoreable batches ---
    all_scored = {}
    print(f"\nEvaluating {len(scoreable_batches)} batches "
          f"({sum(len(b) for b in scoreable_batches)} scoreable facets)...")

    for batch in tqdm(scoreable_batches, desc="Scoring batches"):
        batch_result = evaluate_batch(conversation_text, batch)
        all_scored.update(batch_result)
        # Rate-limit compliance: sleep between batch calls
        time.sleep(1)

    # --- Handle non-scoreable facets ---
    non_scored = evaluate_non_scoreable(non_scoreable_list)

    # --- Merge into unified result ---
    final_result = {}

    # Add scoreable results with metadata
    for facet_name, scores in all_scored.items():
        final_result[facet_name] = {
            "score": scores["score"],
            "confidence": scores["confidence"],
            "category": category_lookup.get(facet_name, "other"),
            "scoring_guidance": guidance_lookup.get(facet_name, ""),
            "is_scoreable": True,
            "note": "",
        }

    # Add non-scoreable results with metadata
    for facet_name, data in non_scored.items():
        final_result[facet_name] = {
            "score": None,
            "confidence": 0.0,
            "category": category_lookup.get(facet_name, "other"),
            "scoring_guidance": guidance_lookup.get(facet_name, ""),
            "is_scoreable": False,
            "note": data["note"],
        }

    return final_result


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_text = (
        "I really appreciate your help! You've been so kind and patient "
        "with me. I was feeling overwhelmed but now I feel much better."
    )
    print("Running test evaluation...")
    results = evaluate_conversation(test_text)
    print(f"\nTotal facets evaluated: {len(results)}")

    # Show a few sample results
    scoreable = {k: v for k, v in results.items() if v["is_scoreable"]}
    non_scoreable = {k: v for k, v in results.items() if not v["is_scoreable"]}
    print(f"Scoreable: {len(scoreable)}, Non-scoreable: {len(non_scoreable)}")

    print("\nSample scoreable results:")
    for name, data in list(scoreable.items())[:5]:
        print(f"  {name}: score={data['score']}, "
              f"confidence={data['confidence']:.2f}")
