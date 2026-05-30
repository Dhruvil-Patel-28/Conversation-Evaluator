"""
generate_samples.py — Sample Conversation Generator & Scorer
=============================================================

Defines 50 hardcoded sample conversations across 5 types:
  - 10 positive/helpful
  - 10 hostile/rude
  - 10 emotional/distressed
  - 10 technical/analytical
  - 10 casual/neutral

Runs each through the evaluation pipeline and saves:
  - samples/conversations_with_scores.json
  - samples/scores_summary.csv

Usage:
    python samples/generate_samples.py

Prerequisites:
    1. Run `python src/preprocess.py` first to generate facets_cleaned.csv
    2. Set GROQ_API_KEY in .env file
"""

import json
import os
import sys

import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import pipeline

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
SCORES_JSON = os.path.join(SAMPLES_DIR, "conversations_with_scores.json")
SUMMARY_CSV = os.path.join(SAMPLES_DIR, "scores_summary.csv")

# ---------------------------------------------------------------------------
# 50 hardcoded sample conversations
# ---------------------------------------------------------------------------
SAMPLE_CONVERSATIONS = [
    # ==================== POSITIVE / HELPFUL (1-10) ====================
    {
        "id": 1,
        "type": "positive",
        "conversation": (
            "Thank you so much for walking me through that process! "
            "You've been incredibly patient and thorough. I really "
            "appreciate your kindness and willingness to help."
        ),
    },
    {
        "id": 2,
        "type": "positive",
        "conversation": (
            "I just wanted to let you know that your advice yesterday "
            "completely changed how I approach my work. I reorganized "
            "everything and I'm already more productive. You're amazing!"
        ),
    },
    {
        "id": 3,
        "type": "positive",
        "conversation": (
            "Hey, welcome to the team! Don't worry about asking too many "
            "questions—that's how we all learned. I'll be here whenever "
            "you need guidance, so feel free to reach out anytime."
        ),
    },
    {
        "id": 4,
        "type": "positive",
        "conversation": (
            "I noticed you were struggling with the report, so I put "
            "together a quick template for you. Hopefully it saves you "
            "some time. Let me know if you want me to review your draft!"
        ),
    },
    {
        "id": 5,
        "type": "positive",
        "conversation": (
            "That's a wonderful idea! I think it could really make a "
            "difference for our customers. Let me help you put together "
            "a proposal so we can present it to the team next week."
        ),
    },
    {
        "id": 6,
        "type": "positive",
        "conversation": (
            "Your presentation was fantastic—clear, well-structured, "
            "and engaging. The way you explained the data made complex "
            "concepts accessible to everyone in the room."
        ),
    },
    {
        "id": 7,
        "type": "positive",
        "conversation": (
            "I completely understand your concern, and I want to make "
            "sure we resolve this for you. Let me escalate this to our "
            "senior team, and I'll personally follow up by end of day."
        ),
    },
    {
        "id": 8,
        "type": "positive",
        "conversation": (
            "You've been doing such a great job this quarter. Your "
            "attention to detail and consistent follow-through have "
            "really set a high standard for the whole department."
        ),
    },
    {
        "id": 9,
        "type": "positive",
        "conversation": (
            "Don't be too hard on yourself—everyone makes mistakes! "
            "What matters is how you learn from them. I think you "
            "handled the situation really well, all things considered."
        ),
    },
    {
        "id": 10,
        "type": "positive",
        "conversation": (
            "I brought some extra coffee for everyone today! Figured "
            "we could all use a pick-me-up on this Monday morning. "
            "Let's have a great week, team!"
        ),
    },

    # ==================== HOSTILE / RUDE (11-20) ====================
    {
        "id": 11,
        "type": "hostile",
        "conversation": (
            "This is the worst customer service I've ever experienced. "
            "You people are completely incompetent and I'm sick of being "
            "transferred around. Fix this now or I'm filing a complaint."
        ),
    },
    {
        "id": 12,
        "type": "hostile",
        "conversation": (
            "I don't care about your excuses. You promised delivery by "
            "Friday and it's Monday. Your company is a joke. I want a "
            "full refund immediately or I'm contacting my lawyer."
        ),
    },
    {
        "id": 13,
        "type": "hostile",
        "conversation": (
            "Are you even listening to me? I've explained this three "
            "times already. Maybe if you actually paid attention instead "
            "of reading from a script, we'd get somewhere."
        ),
    },
    {
        "id": 14,
        "type": "hostile",
        "conversation": (
            "Your opinion is garbage and nobody asked for it. Stop "
            "pretending you know what you're talking about. People like "
            "you are the reason these discussions go nowhere."
        ),
    },
    {
        "id": 15,
        "type": "hostile",
        "conversation": (
            "I can't believe they hired someone this incompetent. Every "
            "single report you submit has errors. Do us all a favor and "
            "actually learn to do your job properly."
        ),
    },
    {
        "id": 16,
        "type": "hostile",
        "conversation": (
            "Whatever. I'm done wasting my time talking to someone who "
            "clearly doesn't understand basic concepts. This whole thing "
            "is a waste and you're making it worse."
        ),
    },
    {
        "id": 17,
        "type": "hostile",
        "conversation": (
            "Stop giving me the runaround. I asked a simple question "
            "and all I get is corporate nonsense. Just be honest for "
            "once in your life—can you do that?"
        ),
    },
    {
        "id": 18,
        "type": "hostile",
        "conversation": (
            "You think you're so smart, don't you? Your idea is terrible "
            "and everyone knows it. I've seen interns come up with better "
            "solutions on their first day."
        ),
    },
    {
        "id": 19,
        "type": "hostile",
        "conversation": (
            "This meeting is pointless. We've been going in circles for "
            "an hour because some people refuse to shut up and listen. "
            "I'm leaving—let me know when you have something useful to say."
        ),
    },
    {
        "id": 20,
        "type": "hostile",
        "conversation": (
            "Honestly, your work ethic is pathetic. You show up late, "
            "leave early, and somehow still manage to mess everything up. "
            "I'm done covering for you."
        ),
    },

    # ==================== EMOTIONAL / DISTRESSED (21-30) ====================
    {
        "id": 21,
        "type": "distressed",
        "conversation": (
            "I don't know what to do anymore. Everything feels like it's "
            "falling apart and I can't stop crying. I just feel so lost "
            "and overwhelmed by everything happening around me."
        ),
    },
    {
        "id": 22,
        "type": "distressed",
        "conversation": (
            "I've been having panic attacks almost every day now. My "
            "chest gets tight and I can't breathe. I'm scared something "
            "is seriously wrong with me but I'm afraid to find out."
        ),
    },
    {
        "id": 23,
        "type": "distressed",
        "conversation": (
            "I lost my job last week and I don't know how I'm going to "
            "pay rent. I feel like such a failure. My family is depending "
            "on me and I'm letting everyone down."
        ),
    },
    {
        "id": 24,
        "type": "distressed",
        "conversation": (
            "Sometimes I wonder if anyone even notices when I'm not "
            "around. I feel invisible most days. It's like I'm going "
            "through the motions but nothing really matters."
        ),
    },
    {
        "id": 25,
        "type": "distressed",
        "conversation": (
            "My best friend just told me they're moving away, and I "
            "feel like I'm being abandoned all over again. I know it's "
            "not about me, but it hurts so much."
        ),
    },
    {
        "id": 26,
        "type": "distressed",
        "conversation": (
            "I can't sleep anymore. I lie awake for hours replaying "
            "every mistake I've ever made. The anxiety is eating me "
            "alive and I don't know how to make it stop."
        ),
    },
    {
        "id": 27,
        "type": "distressed",
        "conversation": (
            "I'm trying so hard to keep it together at work, but inside "
            "I'm breaking. I smile and say I'm fine, but the truth is "
            "I haven't felt okay in months."
        ),
    },
    {
        "id": 28,
        "type": "distressed",
        "conversation": (
            "The grief just hits me out of nowhere. One moment I'm fine "
            "and the next I'm sobbing. It's been a year since they "
            "passed but it still feels like yesterday."
        ),
    },
    {
        "id": 29,
        "type": "distressed",
        "conversation": (
            "I feel like I'm drowning in responsibilities. Work, family, "
            "bills—it never ends. I used to enjoy things but now I just "
            "feel numb. Is this what burnout feels like?"
        ),
    },
    {
        "id": 30,
        "type": "distressed",
        "conversation": (
            "I'm so frustrated with myself. I keep making the same "
            "mistakes and I can't seem to change no matter how hard I "
            "try. I feel stuck in this cycle of disappointment."
        ),
    },

    # ==================== TECHNICAL / ANALYTICAL (31-40) ====================
    {
        "id": 31,
        "type": "technical",
        "conversation": (
            "The regression analysis shows a p-value of 0.003, which is "
            "well below our significance threshold of 0.05. The R-squared "
            "of 0.87 indicates strong predictive power."
        ),
    },
    {
        "id": 32,
        "type": "technical",
        "conversation": (
            "I've identified a memory leak in the connection pooling "
            "module. The issue is in the cleanup handler—it's not "
            "releasing resources when the timeout exception is raised. "
            "I'll push a fix with proper try-finally blocks."
        ),
    },
    {
        "id": 33,
        "type": "technical",
        "conversation": (
            "Based on the benchmarks, switching from a hash map to a "
            "B-tree index reduced query latency from 45ms to 12ms for "
            "range queries, though point lookups are slightly slower."
        ),
    },
    {
        "id": 34,
        "type": "technical",
        "conversation": (
            "The architecture uses a microservices pattern with event "
            "sourcing. Each service publishes domain events to Kafka, "
            "and downstream consumers build their own read models. "
            "This ensures eventual consistency across bounded contexts."
        ),
    },
    {
        "id": 35,
        "type": "technical",
        "conversation": (
            "Looking at the data pipeline metrics, throughput peaked at "
            "2.3 million records per minute during the ETL window. "
            "The bottleneck appears to be the serialization step, not I/O."
        ),
    },
    {
        "id": 36,
        "type": "technical",
        "conversation": (
            "We need to refactor the authentication flow. The current "
            "implementation stores JWTs in local storage, which is "
            "vulnerable to XSS. We should switch to HTTP-only cookies "
            "with CSRF tokens."
        ),
    },
    {
        "id": 37,
        "type": "technical",
        "conversation": (
            "The A/B test results are conclusive. Variant B showed a "
            "14.2% increase in conversion rate with 95% confidence. "
            "I recommend rolling it out to 100% of users by next sprint."
        ),
    },
    {
        "id": 38,
        "type": "technical",
        "conversation": (
            "I've traced the latency spike to our DNS resolution layer. "
            "The TTL was set to 30 seconds, causing excessive lookups. "
            "Increasing it to 300 seconds reduced p99 latency by 38%."
        ),
    },
    {
        "id": 39,
        "type": "technical",
        "conversation": (
            "The model achieved an F1 score of 0.92 on the validation "
            "set after hyperparameter tuning. Precision is 0.94 and "
            "recall is 0.90. I think we're ready for production deployment."
        ),
    },
    {
        "id": 40,
        "type": "technical",
        "conversation": (
            "The database schema needs normalization. We have redundant "
            "user data in three tables, causing inconsistencies on update. "
            "I propose a migration plan that consolidates into a single "
            "source of truth with foreign key references."
        ),
    },

    # ==================== CASUAL / NEUTRAL (41-50) ====================
    {
        "id": 41,
        "type": "casual",
        "conversation": (
            "Hey! Did you catch the game last night? It was a pretty "
            "close match. I thought they were going to lose in the "
            "last quarter but they pulled through somehow."
        ),
    },
    {
        "id": 42,
        "type": "casual",
        "conversation": (
            "I'm thinking about trying that new Italian place downtown "
            "for lunch. Have you been there yet? I heard their pasta is "
            "really good."
        ),
    },
    {
        "id": 43,
        "type": "casual",
        "conversation": (
            "The weather's been pretty nice lately. I might go for a "
            "walk after work if it stays this way. It's good to get "
            "some fresh air, you know?"
        ),
    },
    {
        "id": 44,
        "type": "casual",
        "conversation": (
            "Just finished reorganizing my desk. It's funny how cleaning "
            "up your workspace makes you feel like you've accomplished "
            "something. Now back to actual work, I guess."
        ),
    },
    {
        "id": 45,
        "type": "casual",
        "conversation": (
            "I started watching that show everyone's been talking about. "
            "It's okay so far, nothing mind-blowing. Maybe it gets "
            "better after the first few episodes."
        ),
    },
    {
        "id": 46,
        "type": "casual",
        "conversation": (
            "Morning! How was your weekend? I didn't do much—mostly "
            "just stayed home and caught up on some reading. Sometimes "
            "a quiet weekend is exactly what you need."
        ),
    },
    {
        "id": 47,
        "type": "casual",
        "conversation": (
            "Someone brought donuts to the office today. I tried to "
            "resist but the chocolate one was calling my name. No "
            "regrets though, it was totally worth it."
        ),
    },
    {
        "id": 48,
        "type": "casual",
        "conversation": (
            "I need to get my car serviced this week. It's been making "
            "a weird noise, probably nothing serious. I'll drop it off "
            "at the shop on Thursday."
        ),
    },
    {
        "id": 49,
        "type": "casual",
        "conversation": (
            "Did you see the email about the team outing? Sounds like "
            "they're planning bowling or something. Could be fun if "
            "enough people show up."
        ),
    },
    {
        "id": 50,
        "type": "casual",
        "conversation": (
            "Just got back from my lunch break. Grabbed a sandwich from "
            "that deli around the corner. It was decent but nothing "
            "special. Anyway, what are you working on?"
        ),
    },
]


def run_all_samples():
    """
    Process all 50 sample conversations through the evaluation pipeline.
    Saves results as JSON and summary CSV.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    all_results = []
    summary_rows = []

    total = len(SAMPLE_CONVERSATIONS)
    for i, sample in enumerate(SAMPLE_CONVERSATIONS, 1):
        print(f"\n{'='*60}")
        print(f"Processing sample {i}/{total}: "
              f"[{sample['type']}] ID={sample['id']}")
        print(f"{'='*60}")

        # Run through the pipeline
        df = pipeline.run_pipeline(sample["conversation"])

        # Build scores dict for JSON output
        scores_dict = {}
        for _, row in df.iterrows():
            scores_dict[row["facet"]] = {
                "score": row["score"] if row["score"] != "N/A" else None,
                "confidence": row["confidence"],
                "category": row["category"],
                "is_scoreable": row["is_scoreable"],
                "note": row["note"],
            }

        # Add to results
        result_entry = {
            "conversation_id": sample["id"],
            "type": sample["type"],
            "conversation": sample["conversation"],
            "scores": scores_dict,
        }
        all_results.append(result_entry)

        # Build summary rows for CSV
        for _, row in df.iterrows():
            summary_rows.append({
                "conversation_id": sample["id"],
                "type": sample["type"],
                "facet": row["facet"],
                "score": row["score"],
                "confidence": row["confidence"],
                "category": row["category"],
                "is_scoreable": row["is_scoreable"],
                "note": row["note"],
            })

        print(f"  [OK] Scored {len(df)} facets")

    # --- Save JSON ---
    with open(SCORES_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved scores JSON: {SCORES_JSON}")

    # --- Save summary CSV ---
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[OK] Saved summary CSV: {SUMMARY_CSV}")

    # --- Print completion stats ---
    print(f"\n{'='*60}")
    print("SAMPLE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total conversations processed: {len(all_results)}")
    print(f"Total facet scores generated:  {len(summary_rows)}")
    type_counts = {}
    for r in all_results:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
    print(f"By type: {type_counts}")


if __name__ == "__main__":
    run_all_samples()
