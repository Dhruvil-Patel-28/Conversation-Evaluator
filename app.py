"""
app.py — Streamlit UI for the Conversation Evaluator
======================================================

A production-ready web interface that lets users evaluate conversation
turns across 300 psychological, linguistic, and behavioral facets
using an open-weights LLM (Llama 3.1 8B via Groq).

Features:
  - Sample conversation quick-load buttons
  - Free-text conversation input
  - Real-time evaluation with progress feedback
  - Interactive filtering by category, score, and confidence
  - Summary metrics and visualizations
  - Full CSV download including all 300 facets

Usage:
    streamlit run app.py
"""

import os
import sys

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment and path setup
# ---------------------------------------------------------------------------
# Load .env at startup (for GROQ_API_KEY)
load_dotenv()

# Add project root to path for module imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src import pipeline

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Conversation Evaluator",
    page_icon="💬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished, premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef0ff 100%);
        border: 1px solid #e0e4f5;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Sample button styling */
    div[data-testid="stHorizontalBlock"] > div > div > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    /* Sidebar info box */
    .info-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border: 1px solid #667eea30;
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea40, transparent);
        margin: 1.2rem 0;
    }

    /* Status badges */
    .status-scored {
        color: #059669;
        font-weight: 600;
    }
    .status-not-scoreable {
        color: #9ca3af;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sample conversations for quick testing
# ---------------------------------------------------------------------------
SAMPLES = {
    "Positive": (
        "Thank you so much for walking me through that process! "
        "You've been incredibly patient and thorough. I really "
        "appreciate your kindness and willingness to help."
    ),
    "Hostile": (
        "This is the worst customer service I've ever experienced. "
        "You people are completely incompetent and I'm sick of being "
        "transferred around. Fix this now or I'm filing a complaint."
    ),
    "Distressed": (
        "I don't know what to do anymore. Everything feels like it's "
        "falling apart and I can't stop crying. I just feel so lost "
        "and overwhelmed by everything happening around me."
    ),
    "Technical": (
        "The regression analysis shows a p-value of 0.003, which is "
        "well below our significance threshold of 0.05. The R-squared "
        "of 0.87 indicates strong predictive power."
    ),
    "Casual": (
        "Hey! Did you catch the game last night? It was a pretty "
        "close match. I thought they were going to lose in the "
        "last quarter but they pulled through somehow."
    ),
}


# ---------------------------------------------------------------------------
# Sidebar — Filters and Info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Filters & Info")

    # Model info box
    st.markdown("""
    <div class="info-box">
        <strong>🤖 Model:</strong> Llama 3.1 8B (Meta, open-weights)<br>
        <strong>📊 Facets:</strong> 300 dimensions<br>
        <strong>📈 Scale:</strong> Supports 5,000+ facets
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Category filter
    all_categories = [
        "emotion", "personality", "cognitive", "linguistic",
        "safety", "social", "physical", "other",
    ]
    selected_categories = st.multiselect(
        "📁 Filter by Category",
        options=all_categories,
        default=all_categories,
        help="Select which facet categories to display",
    )

    # Score range filter
    score_range = st.slider(
        "🎯 Score Range",
        min_value=1,
        max_value=5,
        value=(1, 5),
        help="Filter facets by score range",
    )

    # Confidence filter
    high_confidence_only = st.checkbox(
        "🔒 High confidence only (>0.7)",
        value=False,
        help="Show only facets with confidence above 0.7",
    )

    # Show non-scoreable toggle
    show_non_scoreable = st.checkbox(
        "👁️ Show non-scoreable facets",
        value=False,
        help="Include facets that cannot be scored from text alone",
    )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

# Header
st.markdown('<p class="main-title">💬 Conversation Evaluator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Score any conversation across 300 psychological, '
    'linguistic and behavioral dimensions</p>',
    unsafe_allow_html=True,
)

# --- Callback for sample buttons ---
def _load_sample(sample_key):
    """Callback: set the text area value to the selected sample."""
    st.session_state["input_area"] = SAMPLES[sample_key]

# --- Section 1: Sample conversations ---
with st.expander("🧪 Try a sample conversation", expanded=False):
    st.markdown("Click a button to load a sample conversation:")
    cols = st.columns(5)
    sample_keys = list(SAMPLES.keys())
    for idx, col in enumerate(cols):
        with col:
            st.button(
                f"{'😊🔴😢🔬💬'[idx]} {sample_keys[idx]}",
                key=f"sample_{sample_keys[idx]}",
                width='stretch',
                on_click=_load_sample,
                args=(sample_keys[idx],),
            )

# --- Section 2: Input ---
conversation_text = st.text_area(
    "✍️ Enter a conversation turn",
    height=150,
    placeholder="Paste or type a conversation turn here...",
    key="input_area",
)

evaluate_clicked = st.button(
    "🚀 Evaluate",
    type="primary",
    width='stretch',
)

# --- Section 3: Results ---
if evaluate_clicked and conversation_text.strip():
    with st.spinner("⏳ Evaluating across 300 facets..."):
        # Run the evaluation pipeline
        results_df = pipeline.run_pipeline(conversation_text.strip())
        stats = pipeline.get_summary_stats(results_df)

        # Store in session state for persistence
        st.session_state["results_df"] = results_df
        st.session_state["stats"] = stats

elif evaluate_clicked and not conversation_text.strip():
    st.warning("⚠️ Please enter a conversation turn before evaluating.")

# Display results if available
if "results_df" in st.session_state and "stats" in st.session_state:
    results_df = st.session_state["results_df"]
    stats = st.session_state["stats"]

    st.markdown("---")

    # Row 1: Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{stats['scoreable_count']}</p>
            <p class="metric-label">Total Facets Scored</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{stats['avg_score']}</p>
            <p class="metric-label">Average Score</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{stats['avg_confidence']}</p>
            <p class="metric-label">Average Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{stats['high_confidence_count']}</p>
            <p class="metric-label">High Confidence (>0.7)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### 📊 Average Score by Category")
        if stats["category_breakdown"]:
            cat_df = pd.DataFrame(
                list(stats["category_breakdown"].items()),
                columns=["Category", "Avg Score"],
            ).set_index("Category").sort_values("Avg Score", ascending=False)
            st.bar_chart(cat_df)
        else:
            st.info("No scoreable data to display.")

    with chart_col2:
        st.markdown("##### 📊 Facet Count by Category")
        cat_count = results_df["category"].value_counts()
        cat_count_df = pd.DataFrame(
            {"Category": cat_count.index, "Count": cat_count.values}
        ).set_index("Category").sort_values("Count", ascending=False)
        st.bar_chart(cat_count_df)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: Filtered results table
    st.markdown("##### 📋 Detailed Results")

    # Apply sidebar filters
    display_df = results_df.copy()

    # Category filter
    display_df = display_df[display_df["category"].isin(selected_categories)]

    # Show/hide non-scoreable
    if not show_non_scoreable:
        display_df = display_df[display_df["is_scoreable"] == True]

    # Score range filter (only for scoreable)
    scoreable_mask = display_df["is_scoreable"] == True
    if scoreable_mask.any():
        score_numeric = pd.to_numeric(
            display_df.loc[scoreable_mask, "score"], errors="coerce"
        )
        score_filter = (score_numeric >= score_range[0]) & (
            score_numeric <= score_range[1]
        )
        # Keep non-scoreable rows (if shown) + filtered scoreable rows
        non_scoreable_rows = display_df[~scoreable_mask]
        scoreable_rows = display_df[scoreable_mask][score_filter.values]
        display_df = pd.concat([scoreable_rows, non_scoreable_rows])

    # High confidence filter
    if high_confidence_only:
        display_df = display_df[
            (display_df["confidence"] > 0.7) | (display_df["is_scoreable"] == False)
        ]

    # Build display columns — FIX 6: use display_name for cleaner table
    facet_col = "display_name" if "display_name" in display_df.columns else "facet"
    display_table = display_df[[facet_col, "category", "score", "confidence"]].copy()
    display_table["Status"] = display_df["is_scoreable"].apply(
        lambda x: "✅ Scored" if x else "⬜ Not Scoreable"
    )
    display_table.columns = ["Facet", "Category", "Score", "Confidence", "Status"]

    st.dataframe(
        display_table,
        height=400,
        width='stretch',
    )

    st.markdown(f"*Showing {len(display_table)} of {len(results_df)} facets*")

    # Row 4: Download button
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full Results CSV",
        data=csv_data,
        file_name="conversation_evaluation_results.csv",
        mime="text/csv",
        width='stretch',
    )
