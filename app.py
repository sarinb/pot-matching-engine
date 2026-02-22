"""Streamlit UI for testing the Proof of Talk Matching Engine."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.matching.config import settings  # noqa: E402
from src.matching.engine import (  # noqa: E402
    load_minimal_profiles,
    load_profiles_from_json,
    load_sample_profiles,
    run,
)
from src.matching.models import AttendeeProfile, MatchBriefing  # noqa: E402

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="PoT Matching Engine", layout="wide")
st.title("Proof of Talk — Matching Engine Prototype")

TICKET_OPTIONS = ["delegate", "sponsor", "speaker", "vip"]
ROLE_OPTIONS = [
    "deploying_capital", "raising_capital", "exploring_partnerships",
    "seeking_technology", "regulatory_policy", "media_content",
]

_UPLOAD_HELP = """\
Upload a JSON array of attendee profiles.  **Minimal format** (5 fields):

```json
[
  {
    "name": "Amara Okafor",
    "title": "Director of Digital Assets",
    "company": "Abu Dhabi Sovereign Wealth Fund",
    "product": "Sovereign wealth fund deploying $200M into tokenized RWA",
    "looking_for": "Deploy $200M into tokenized RWA + blockchain infra"
  }
]
```

Optional enrichment fields: `id`, `company_description`, `ticket_type`,
`sector`, `stage`, `funding_raised`, `key_facts`, `roles_at_event`,
`stated_goal`.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_profile_preview(
    profiles: list[AttendeeProfile], expanded: bool = True,
) -> None:
    for i, p in enumerate(profiles):
        with st.expander(f"{i+1}. {p.name} — {p.title}, {p.company}", expanded=expanded):
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Name:** {p.name}")
                st.markdown(f"**Title:** {p.title}")
                st.markdown(f"**Company:** {p.company}")
                if p.product:
                    st.markdown(f"**Product:** {p.product}")
                if p.company_description and p.company_description != p.product:
                    st.markdown(f"**Description:** {p.company_description}")
            with cols[1]:
                st.markdown(f"**Goal:** {p.stated_goal or p.looking_for or '—'}")
                if p.sector:
                    st.markdown(f"**Sector:** {p.sector}")
                if p.stage:
                    st.markdown(f"**Stage:** {p.stage}")
                if p.funding_raised:
                    st.markdown(f"**Funding:** {p.funding_raised}")
                if p.roles_at_event:
                    st.markdown(f"**Roles:** {', '.join(p.roles_at_event)}")
                if p.key_facts:
                    st.markdown("**Key facts:**")
                    for f in p.key_facts:
                        st.markdown(f"- {f}")


def _run_engine(profiles: list[AttendeeProfile]) -> list[MatchBriefing] | None:
    if len(profiles) < 2:
        st.error("Need at least 2 profiles to run matching.")
        return None

    if not settings.anthropic_api_key:
        st.error("No Anthropic API key found.  Set ANTHROPIC_API_KEY in .env")
        return None

    skip = st.session_state.get("skip_explanations", False)

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def _cb(label: str, frac: float) -> None:
        progress_bar.progress(min(frac, 1.0))
        status_text.text(label)

    try:
        briefings = run(
            profiles,
            skip_explanations=skip,
            progress_callback=_cb,
        )
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        return briefings
    except Exception as e:
        st.error(f"Engine error: {e}")
        return None


def _render_briefings(
    briefings: list[MatchBriefing],
    profiles: list[AttendeeProfile],
) -> None:
    st.markdown("---")
    st.header("Match Results")

    # Heatmap
    names = [p.name for p in profiles]
    matrix_data = {name: {n: 0.0 for n in names} for name in names}
    for b in briefings:
        for m in b.matches:
            a_name = b.attendee_name
            b_profile = next((p for p in profiles if p.id == m.pair.attendee_b_id), None)
            if b_profile:
                matrix_data[a_name][b_profile.name] = m.pair.composite

    df = pd.DataFrame(matrix_data, index=names, columns=names)
    st.subheader("Match Matrix Heatmap")
    try:
        styled = df.style.background_gradient(cmap="YlOrRd", vmin=0, vmax=1).format("{:.2f}")
        st.dataframe(styled, use_container_width=True)
    except ImportError:
        st.dataframe(df.round(2), use_container_width=True)

    # Per-attendee briefings
    st.subheader("Individual Briefings")
    for briefing in briefings:
        with st.expander(f"**{briefing.attendee_name}** — Top {len(briefing.matches)} Matches"):
            for rank, match in enumerate(briefing.matches, 1):
                other = next(
                    (p for p in profiles if p.id == match.pair.attendee_b_id),
                    None,
                )
                if not other:
                    continue

                st.markdown(f"#### #{rank}: {other.name} — {other.title}, {other.company}")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Composite", f"{match.pair.composite:.2f}")
                col2.metric("Complementarity", f"{match.pair.scores.complementarity:.2f}")
                col3.metric("Transaction-Ready", f"{match.pair.scores.transaction_readiness:.2f}")
                col4.metric("Non-Obvious", f"{match.pair.scores.non_obvious:.2f}")

                if match.pair.transaction_type:
                    st.caption(f"Transaction type: **{match.pair.transaction_type.replace('_', ' ').title()}**")

                # Dimension one-liners
                dim = match.dimension_explanations
                if dim.complementarity or dim.transaction_readiness or dim.non_obvious:
                    dcol1, dcol2, dcol3 = st.columns(3)
                    if dim.complementarity:
                        dcol1.markdown(f"**Complementarity:** {_safe(dim.complementarity)}")
                    if dim.transaction_readiness:
                        dcol2.markdown(f"**Readiness:** {_safe(dim.transaction_readiness)}")
                    if dim.non_obvious:
                        dcol3.markdown(f"**Non-obvious:** {_safe(dim.non_obvious)}")

                # Main explanation
                if match.explanation and match.explanation != "(explanations skipped)":
                    st.info(_safe(match.explanation))

                st.markdown("---")


def _safe(text: str) -> str:
    """Escape dollar signs to prevent Streamlit LaTeX rendering."""
    return text.replace("$", r"\$")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    st.session_state["skip_explanations"] = st.checkbox(
        "Skip explanations (fast mode)", value=False,
    )
    st.markdown("---")
    st.caption("API key is loaded from `.env` file.")
    if settings.anthropic_api_key:
        st.success("API key loaded")
    else:
        st.warning("No API key found")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_sample, tab_upload, tab_builder = st.tabs([
    "Sample Profiles", "Upload JSON", "Interactive Builder",
])

# --- Tab 1: Sample Profiles ---
with tab_sample:
    st.subheader("Run with built-in test profiles")
    profile_set = st.radio(
        "Profile set",
        ["Detailed (enriched fields)", "Minimal (persona-format only)"],
        horizontal=True,
    )
    if profile_set.startswith("Detailed"):
        profiles = load_sample_profiles()
    else:
        profiles = load_minimal_profiles()

    _render_profile_preview(profiles, expanded=True)

    if st.button("Run Matching Engine", key="run_sample", type="primary"):
        briefings = _run_engine(profiles)
        if briefings:
            _render_briefings(briefings, profiles)


# --- Tab 2: Upload JSON ---
with tab_upload:
    st.subheader("Upload custom profiles")
    st.markdown(_UPLOAD_HELP)
    uploaded = st.file_uploader("Upload JSON", type=["json"])
    if uploaded:
        try:
            raw = json.loads(uploaded.read())
            profiles = load_profiles_from_json(raw)
            st.success(f"Loaded {len(profiles)} profiles")
            _render_profile_preview(profiles, expanded=True)

            if st.button("Run Matching Engine", key="run_upload", type="primary"):
                briefings = _run_engine(profiles)
                if briefings:
                    _render_briefings(briefings, profiles)
        except Exception as e:
            st.error(f"Error loading JSON: {e}")


# --- Tab 3: Interactive Builder ---
with tab_builder:
    st.subheader("Build profiles interactively")

    if "builder_profiles" not in st.session_state:
        st.session_state.builder_profiles = []

    with st.form("add_profile", clear_on_submit=True):
        st.markdown("**Add a new profile**")

        name = st.text_input("Name *")
        title = st.text_input("Title *")
        company = st.text_input("Company *")
        product = st.text_area(
            "Product / Offering *",
            help="What does this person's company do? One sentence.",
        )
        looking_for = st.text_area(
            "Looking for *",
            help="What outcome would make this event worth their time?",
        )

        with st.expander("Advanced (optional)"):
            col1, col2 = st.columns(2)
            with col1:
                ticket = st.selectbox("Ticket type", TICKET_OPTIONS)
                sector = st.text_input("Sector (e.g. tokenized_securities)")
                stage = st.text_input("Stage (e.g. series_b, growth_vc)")
            with col2:
                funding = st.text_input("Funding raised (e.g. $40M)")
                roles = st.multiselect(
                    "Roles at event (auto-inferred if empty)",
                    ROLE_OPTIONS,
                )
            key_facts_raw = st.text_area(
                "Key facts (one per line)",
                help="Verifiable claims — funding, customers, mandates, etc.",
            )

        submitted = st.form_submit_button("Add Profile")
        if submitted:
            if not all([name, title, company, looking_for]):
                st.error("Fill in all required fields (marked with *).")
            elif len(st.session_state.builder_profiles) >= 20:
                st.error("Maximum 20 profiles.")
            else:
                key_facts = [
                    f.strip()
                    for f in (key_facts_raw or "").strip().split("\n")
                    if f.strip()
                ]
                profile = AttendeeProfile(
                    name=name,
                    title=title,
                    company=company,
                    product=product or None,
                    looking_for=looking_for,
                    ticket_type=ticket,
                    roles_at_event=roles if roles else None,
                    sector=sector or None,
                    stage=stage or None,
                    funding_raised=funding or None,
                    key_facts=key_facts,
                )
                st.session_state.builder_profiles.append(profile)
                st.success(f"Added {name}. Total: {len(st.session_state.builder_profiles)}")

    if st.session_state.builder_profiles:
        st.markdown(f"**{len(st.session_state.builder_profiles)} profiles added:**")
        for i, p in enumerate(st.session_state.builder_profiles):
            cols = st.columns([4, 1])
            cols[0].markdown(f"{i+1}. **{p.name}** — {p.title}, {p.company}")
            if cols[1].button("Remove", key=f"rm_{i}"):
                st.session_state.builder_profiles.pop(i)
                st.rerun()

        col_run, col_clear = st.columns(2)
        if col_run.button("Run Matching Engine", key="run_builder", type="primary"):
            briefings = _run_engine(st.session_state.builder_profiles)
            if briefings:
                _render_briefings(briefings, st.session_state.builder_profiles)
        if col_clear.button("Clear All", key="clear_builder"):
            st.session_state.builder_profiles = []
            st.rerun()
