from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from google_sheets_oauth import (
    GoogleOAuthConfigError,
    build_google_auth_url,
    consume_google_oauth_callback,
    disconnect_google_token,
    extract_sheet_reference,
    get_sheet_rows,
    get_spreadsheet_metadata,
    get_valid_google_token,
    list_sheet_titles,
    resolve_sheet_title,
)
from scoring import (
    COMPLIANCE_FLAG_COLUMNS,
    DISPLAY_COLUMNS,
    PARAMETERS,
    RAW_SECTION_COLUMNS,
    build_scored_dataset,
    build_scored_dataset_from_rows,
    parameter_failure_summary,
    safe_number,
    safe_percent,
    summarize_dimension,
    top_findings,
)


st.set_page_config(
    page_title="QA Audit Intelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCORE_MODES = {
    "Without Fatal": "Overall Score (Without Fatal)",
    "With Fatal": "Overall Score (With Fatal)",
}

GROUP_OPTIONS = [
    "Agent Name",
    "Quality Auditor",
    "Call/Chat",
    "Supervisor name",
    "LOB",
    "Sub-LOB",
    "Reason for call",
]

SECTION_ORDER = list(RAW_SECTION_COLUMNS.keys())

CUSTOM_CSS = """
<style>
    :root {
        --page-bg: #f4f7fb;
        --panel: rgba(255,255,255,0.84);
        --panel-solid: #ffffff;
        --border: rgba(15, 23, 42, 0.08);
        --text: #0f172a;
        --muted: #516174;
        --brand: #3b82f6;
        --brand-soft: rgba(59,130,246,0.10);
        --brand-strong: #1d4ed8;
        --green: #16a34a;
        --amber: #d97706;
        --rose: #dc2626;
        --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        --radius-xl: 24px;
        --radius-lg: 18px;
        --radius-md: 14px;
    }

    .main > div {
        padding-top: 1rem;
    }
    .block-container {
        max-width: 1600px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }
    .app-shell {
        background:
            radial-gradient(circle at top right, rgba(59,130,246,0.12), transparent 24%),
            radial-gradient(circle at top left, rgba(147,197,253,0.18), transparent 20%),
            linear-gradient(180deg, #f8fbff 0%, #f4f7fb 42%, #eef3f8 100%);
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92));
        border: 1px solid rgba(255,255,255,0.7);
        border-radius: 28px;
        padding: 1.35rem 1.4rem 1.25rem 1.4rem;
        box-shadow: 0 28px 60px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(16px);
        margin-bottom: 1rem;
    }
    .hero-title {
        color: var(--text);
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin-bottom: 0.85rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.34rem 0.72rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.9);
        color: var(--brand-strong);
        border: 1px solid rgba(59,130,246,0.16);
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.45rem;
        margin-bottom: 0.4rem;
    }
    .kpi-card {
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(247,250,252,0.94));
        border-radius: 22px;
        padding: 1rem 1rem 0.95rem;
        min-height: 132px;
        box-shadow: var(--shadow);
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748b;
        margin-bottom: 0.25rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 1.95rem;
        line-height: 1.1;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0.18rem;
    }
    .kpi-sub {
        font-size: 0.92rem;
        color: var(--muted);
    }
    .panel-card {
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1rem 1rem 0.45rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(250,252,255,0.94));
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }
    .panel-title {
        color: var(--text);
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 0.1rem;
    }
    .panel-copy {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 0.8rem;
    }
    .section-note {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: -0.2rem;
        margin-bottom: 0.85rem;
    }
    .warning-shell {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255,247,237,0.95), rgba(255,251,235,0.95));
        border: 1px solid rgba(245, 158, 11, 0.18);
        color: #92400e;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.55rem;
        margin-bottom: 0.6rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        background: rgba(226,232,240,0.68);
        color: #334155;
        font-weight: 700;
        border: 1px solid rgba(148,163,184,0.18);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(219,234,254,0.95) !important;
        color: #1d4ed8 !important;
        border-color: rgba(59,130,246,0.22) !important;
    }
    .stDataFrame, .stTable {
        border-radius: 18px;
        overflow: hidden;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbff 0%, #eff6ff 100%);
        border-right: 1px solid rgba(59,130,246,0.08);
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.82rem;
    }
</style>
"""

st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_title(title: str, copy: str = "") -> None:
    st.markdown(
        f"""
        <div class="panel-title">{title}</div>
        <div class="panel-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_from_path(path: str, mtime: float):
    return build_scored_dataset(path)


@st.cache_data(show_spinner=False)
def load_from_bytes(file_bytes: bytes, filename: str):
    return build_scored_dataset(file_bytes)


def figure_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=48, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
        font=dict(color="#0f172a"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    return fig


def to_display_name(path: str | None) -> str:
    return Path(path).name if path else "-"


def last_modified_text(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return "-"
    ts = pd.to_datetime(os.path.getmtime(path), unit="s")
    return ts.strftime("%Y-%m-%d %H:%M")


def google_redirect_if_needed() -> None:
    redirect_url = st.session_state.pop("google_auth_redirect_url", None)
    if redirect_url:
        components.html(
            f"<script>window.parent.location.href = {json.dumps(redirect_url)};</script>",
            height=0,
        )
        st.stop()


def _google_session_dataset_matches(spreadsheet_id: str, sheet_title: str) -> bool:
    loaded = st.session_state.get("google_loaded_dataset")
    return bool(loaded and loaded.get("spreadsheet_id") == spreadsheet_id and loaded.get("sheet_title") == sheet_title)


def _load_google_dataset(spreadsheet_id: str, sheet_title: str, sheet_url: str):
    rows = get_sheet_rows(spreadsheet_id, sheet_title, data_range="A3:AO")
    df, long_df, compliance_df = build_scored_dataset_from_rows(rows, start_row=3)
    metadata = get_spreadsheet_metadata(spreadsheet_id)
    spreadsheet_title = metadata.get("properties", {}).get("title", "Google Sheet")
    payload = {
        "df": df,
        "long_df": long_df,
        "compliance_df": compliance_df,
        "spreadsheet_id": spreadsheet_id,
        "sheet_title": sheet_title,
        "sheet_url": sheet_url,
        "spreadsheet_title": spreadsheet_title,
        "loaded_at": pd.Timestamp.utcnow(),
    }
    st.session_state["google_loaded_dataset"] = payload
    return payload


def _load_google_sheet_ui():
    st.sidebar.subheader("Google Sheets source")
    st.sidebar.caption("Paste a Google Sheets link. The app will request Google OAuth and read the sheet with the signed-in user's access.")

    default_sheet_value = st.session_state.get("google_pending_sheet_value", "")
    sheet_value = st.sidebar.text_input("Google Sheets link", value=default_sheet_value, key="google_sheet_link")

    try:
        callback_sheet = consume_google_oauth_callback()
        if callback_sheet and not sheet_value:
            st.session_state["google_sheet_link"] = callback_sheet
            sheet_value = callback_sheet
            st.toast("Google authorization completed.")
    except Exception as exc:
        st.sidebar.error(str(exc))

    if not sheet_value:
        st.sidebar.info("Paste a Google Sheets link to continue.")
        return None, None, None, "Google Sheets (OAuth 2.0)", None, "-"

    try:
        sheet_ref = extract_sheet_reference(sheet_value)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return None, None, None, "Google Sheets (OAuth 2.0)", None, "-"

    with st.sidebar.expander("Parsed spreadsheet", expanded=False):
        st.code(sheet_ref.spreadsheet_id, language=None)

    token = None
    try:
        token = get_valid_google_token()
    except GoogleOAuthConfigError as exc:
        st.sidebar.error(str(exc))
        return None, None, None, "Google Sheets (OAuth 2.0)", None, "-"
    except Exception as exc:
        st.sidebar.error(f"Google token refresh failed: {exc}")

    action_col_1, action_col_2 = st.sidebar.columns(2)
    load_clicked = action_col_1.button("Authenticate / Load", type="primary")
    disconnect_clicked = action_col_2.button("Disconnect")

    if disconnect_clicked:
        disconnect_google_token()
        st.session_state.pop("google_loaded_dataset", None)
        st.rerun()

    if token:
        st.sidebar.success("Google session active")
    else:
        st.sidebar.warning("Google authorization required")

    metadata = None
    titles: list[str] = []
    selected_title = None
    source_path = sheet_value
    workbook_name = "Google Sheet"

    if token:
        try:
            metadata = get_spreadsheet_metadata(sheet_ref.spreadsheet_id)
            titles = list_sheet_titles(metadata)
            workbook_name = metadata.get("properties", {}).get("title", "Google Sheet")
            default_title = resolve_sheet_title(metadata, gid=sheet_ref.gid)
            if titles:
                current_selection = st.session_state.get("google_selected_sheet_title", default_title)
                default_index = titles.index(current_selection) if current_selection in titles else (titles.index(default_title) if default_title in titles else 0)
                selected_title = st.sidebar.selectbox("Worksheet tab", titles, index=default_index, key="google_selected_sheet_title")
                source_path = f"{sheet_value} | {selected_title}"
                workbook_name = f"{workbook_name} · {selected_title}"
        except Exception as exc:
            st.sidebar.error(f"Could not inspect the spreadsheet: {exc}")

    should_load_now = False
    if load_clicked:
        if not token:
            st.session_state["google_auth_redirect_url"] = build_google_auth_url(sheet_ref)
            st.rerun()
        elif selected_title:
            should_load_now = True

    if token and selected_title and not _google_session_dataset_matches(sheet_ref.spreadsheet_id, selected_title):
        should_load_now = True

    if token and selected_title and st.sidebar.button("Refresh Google Sheet"):
        should_load_now = True

    if should_load_now and token and selected_title:
        try:
            payload = _load_google_dataset(sheet_ref.spreadsheet_id, selected_title, sheet_value)
            source_path = f"{sheet_value} | {selected_title}"
            workbook_name = f"{payload['spreadsheet_title']} · {selected_title}"
        except Exception as exc:
            st.sidebar.error(f"Could not load Google Sheet data: {exc}")

    loaded = st.session_state.get("google_loaded_dataset")
    if loaded and loaded.get("spreadsheet_id") == sheet_ref.spreadsheet_id:
        source_path = f"{sheet_value} | {loaded['sheet_title']}"
        workbook_name = f"{loaded['spreadsheet_title']} · {loaded['sheet_title']}"
        return (
            loaded["df"],
            loaded["long_df"],
            loaded["compliance_df"],
            "Google Sheets (OAuth 2.0)",
            source_path,
            workbook_name,
        )

    return None, None, None, "Google Sheets (OAuth 2.0)", source_path, workbook_name


def load_data_ui():
    google_redirect_if_needed()

    st.sidebar.header("Source mode")
    source_mode = st.sidebar.radio(
        "How should the app read the audit data?",
        ["Google Sheets (OAuth 2.0)", "Synced workbook path", "Upload workbook"],
    )

    if source_mode == "Google Sheets (OAuth 2.0)":
        return _load_google_sheet_ui()

    st.sidebar.subheader("Workbook source")
    default_path = os.getenv("AUDIT_FILE_PATH", "data/qa_audit.xlsx")
    workbook_name = "-"
    source_path = None

    if source_mode == "Synced workbook path":
        source_path = st.sidebar.text_input("Workbook path", value=default_path)
        if source_path and os.path.exists(source_path):
            workbook_name = to_display_name(source_path)
            df, long_df, compliance_df = load_from_path(source_path, os.path.getmtime(source_path))
        else:
            st.sidebar.warning("Path not found yet. Update the path or switch source mode.")
            return None, None, None, source_mode, source_path, workbook_name
    else:
        uploaded = st.sidebar.file_uploader("Upload an .xlsx workbook", type=["xlsx"])
        if uploaded is None:
            st.sidebar.info("Upload your audit workbook to get started.")
            return None, None, None, source_mode, source_path, workbook_name
        workbook_name = uploaded.name
        file_bytes = uploaded.getvalue()
        df, long_df, compliance_df = load_from_bytes(file_bytes, uploaded.name)

    return df, long_df, compliance_df, source_mode, source_path, workbook_name


def sequential_multiselect_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    filter_order = [
        "Call/Chat",
        "Quality Auditor",
        "Agent Name",
        "Supervisor name",
        "LOB",
        "Sub-LOB",
        "Reason for call",
    ]
    filtered = df.copy()
    selections: dict[str, list[str]] = {}

    for column in filter_order:
        options = sorted(filtered[column].dropna().astype(str).unique().tolist())
        selections[column] = st.sidebar.multiselect(column, options=options, key=f"filter_{column}")
        if selections[column]:
            filtered = filtered[filtered[column].astype(str).isin(selections[column])]

    return filtered, selections


def apply_filters(df: pd.DataFrame, long_df: pd.DataFrame, compliance_df: pd.DataFrame):
    st.sidebar.header("Filters")
    date_field = st.sidebar.selectbox("Date field", ["Call Date", "Audit Date"])

    valid_dates = pd.to_datetime(df[date_field], errors="coerce").dropna()
    min_date = valid_dates.min().date() if not valid_dates.empty else date.today()
    max_date = valid_dates.max().date() if not valid_dates.empty else date.today()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    filtered = df[df[date_field].between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")].copy()

    score_mode_label = st.sidebar.radio("Score mode", list(SCORE_MODES.keys()), horizontal=True)
    score_col = SCORE_MODES[score_mode_label]

    filtered, selections = sequential_multiselect_filters(filtered)

    min_audits = st.sidebar.slider("Minimum audits per entity in ranking tables", min_value=1, max_value=max(1, int(filtered.shape[0] or 1)), value=1)

    fatal_only = st.sidebar.toggle("Only fatal audits", value=False)
    if fatal_only:
        filtered = filtered[filtered["Fatal Count"] > 0]

    defect_only = st.sidebar.toggle("Only audits with at least one defect", value=False)
    if defect_only:
        filtered = filtered[filtered["Failed Parameter Count"] > 0]

    record_ids = filtered["Record ID"].tolist()
    long_filtered = long_df[long_df["Record ID"].isin(record_ids)].copy()
    compliance_filtered = compliance_df[compliance_df["Record ID"].isin(record_ids)].copy()

    return {
        "df": filtered,
        "long_df": long_filtered,
        "compliance_df": compliance_filtered,
        "score_col": score_col,
        "score_mode_label": score_mode_label,
        "date_field": date_field,
        "start_date": start_date,
        "end_date": end_date,
        "min_audits": min_audits,
        "selections": selections,
    }


def group_summary(df: pd.DataFrame, score_col: str, group_col: str, min_audits: int = 1) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            Audits=("Record ID", "count"),
            Avg_Score=(score_col, "mean"),
            Avg_Defect=("Defect %", "mean"),
            Fatal_Rate=("Fatal Count", lambda s: float((s > 0).mean())),
            Avg_Failed_Parameters=("Failed Parameter Count", "mean"),
        )
        .reset_index()
    )
    grouped[group_col] = grouped[group_col].fillna("Unknown")
    grouped = grouped[grouped["Audits"] >= min_audits].sort_values(["Avg_Score", "Audits"], ascending=[False, False])
    return grouped


def hero_banner(source_mode: str, workbook_name: str, source_path: str | None, df: pd.DataFrame, score_mode_label: str) -> None:
    chips = [
        f"<span class='pill'>Source · {source_mode}</span>",
        f"<span class='pill'>Loaded file · {workbook_name}</span>",
        f"<span class='pill'>Audits · {len(df):,}</span>",
        f"<span class='pill'>Score mode · {score_mode_label}</span>",
    ]
    if source_path and source_mode == "Synced workbook path":
        chips.append(f"<span class='pill'>Updated · {last_modified_text(source_path)}</span>")

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">QA Audit Intelligence</div>
            <div class="hero-subtitle">A GitHub-ready, Streamlit Cloud-ready QA dashboard built for weighted scoring, drill-down analysis, and Google Sheets OAuth access.</div>
            {''.join(chips)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_chips(ctx: dict[str, Any]) -> None:
    chips: list[str] = []
    for key, values in ctx["selections"].items():
        if values:
            joined = ", ".join(values[:2]) + (" + more" if len(values) > 2 else "")
            chips.append(f"<span class='pill'>{key} · {joined}</span>")
    chips.append(f"<span class='pill'>{ctx['date_field']} · {ctx['start_date']} to {ctx['end_date']}</span>")
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)


def render_overview(df: pd.DataFrame, long_df: pd.DataFrame, score_col: str, date_field: str) -> None:
    panel_title("Executive overview", "Fast read on quality, severity, and where operational risk is clustering.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Average score", safe_percent(df[score_col].mean()), "Across all filtered audits")
    with c2:
        metric_card("Fatal rate", safe_percent((df["Fatal Count"] > 0).mean()), "Share of audits containing at least one fatal")
    with c3:
        metric_card("Defect rate", safe_percent(df["Defect %"].mean()), "Average defect share across applicable checks")
    with c4:
        metric_card("Agents covered", safe_number(df["Agent Name"].nunique(), 0), "Unique agents in the filtered view")
    with c5:
        metric_card("QA scorers", safe_number(df["Quality Auditor"].nunique(), 0), "Unique auditors in the filtered view")

    left, right = st.columns([1.35, 1])
    with left:
        daily = (
            df.groupby(date_field, dropna=False)
            .agg(Avg_Score=(score_col, "mean"), Audits=("Record ID", "count"), Fatal_Rate=("Fatal Count", lambda s: float((s > 0).mean())))
            .reset_index()
            .sort_values(date_field)
        )
        if not daily.empty:
            fig = px.line(daily, x=date_field, y="Avg_Score", markers=True)
            fig.add_bar(x=daily[date_field], y=daily["Audits"], yaxis="y2", opacity=0.18, name="Audits")
            fig.update_layout(
                yaxis=dict(title="Average score", tickformat=".0%"),
                yaxis2=dict(title="Audits", overlaying="y", side="right", showgrid=False),
            )
            st.plotly_chart(figure_layout(fig, 420), use_container_width=True)
        else:
            st.info("No trend data available for the selected date range.")

    with right:
        section_rows = []
        for section in SECTION_ORDER:
            col = f"Section Score :: {section}"
            if col in df.columns:
                section_rows.append({"Section": section, "Average Score": df[col].mean()})
        section_df = pd.DataFrame(section_rows).sort_values("Average Score", ascending=True)
        if not section_df.empty:
            fig = px.bar(section_df, x="Average Score", y="Section", orientation="h", text="Average Score")
            fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(figure_layout(fig, 420), use_container_width=True)
        else:
            st.info("No section-level data available.")

    b1, b2 = st.columns([1, 1])
    with b1:
        if not df.empty:
            fig = px.histogram(df, x=score_col, nbins=14, color="Call/Chat")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(figure_layout(fig, 360), use_container_width=True)
    with b2:
        if not df.empty:
            scatter = df.copy()
            scatter["Audit Label"] = scatter["Record ID"] + " · " + scatter["Agent Name"].fillna("Unknown")
            fig = px.scatter(
                scatter,
                x="Defect %",
                y=score_col,
                size="Applicable Parameter Count",
                color="Quality Auditor",
                hover_name="Audit Label",
                hover_data={score_col: ':.1%', 'Defect %': ':.1%'},
            )
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(figure_layout(fig, 360), use_container_width=True)

    findings = top_findings(long_df)
    if not findings.empty:
        st.markdown("### Where quality is breaking")
        st.dataframe(findings.head(12), use_container_width=True, hide_index=True)


def render_team_breakdown(df: pd.DataFrame, score_col: str, min_audits: int) -> None:
    panel_title("Performance lens", "Compare agents, scorers, or business slices with sample-size-aware rankings.")
    controls = st.columns([1, 1, 1.2])
    primary_group = controls[0].selectbox("Primary grouping", GROUP_OPTIONS, index=0)
    secondary_group = controls[1].selectbox("Secondary split", ["None"] + [c for c in GROUP_OPTIONS if c != primary_group], index=1)
    sort_metric = controls[2].selectbox("Sort ranking by", ["Avg_Score", "Fatal_Rate", "Avg_Defect", "Audits"], index=0)

    summary = group_summary(df, score_col, primary_group, min_audits=min_audits)
    if summary.empty:
        st.info("No grouped data available with the current filters and minimum-audit threshold.")
        return

    chart_df = summary.head(15).sort_values(sort_metric, ascending=(sort_metric not in {"Avg_Score", "Audits"}))
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.bar(chart_df, x=primary_group, y="Avg_Score", color="Fatal_Rate", hover_data=["Audits", "Avg_Defect"])
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(categoryorder="total descending")
        st.plotly_chart(figure_layout(fig, 430), use_container_width=True)
    with c2:
        fig = px.scatter(
            summary,
            x="Avg_Defect",
            y="Avg_Score",
            size="Audits",
            color="Fatal_Rate",
            hover_name=primary_group,
            hover_data={"Avg_Score": ':.1%', "Avg_Defect": ':.1%', "Fatal_Rate": ':.1%'},
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure_layout(fig, 430), use_container_width=True)

    if secondary_group != "None":
        split = (
            df.groupby([primary_group, secondary_group], dropna=False)
            .agg(Avg_Score=(score_col, "mean"), Audits=("Record ID", "count"))
            .reset_index()
        )
        split[primary_group] = split[primary_group].fillna("Unknown")
        split[secondary_group] = split[secondary_group].fillna("Unknown")
        split = split[split["Audits"] >= min_audits]
        if not split.empty:
            fig = px.bar(split, x=primary_group, y="Avg_Score", color=secondary_group, barmode="group")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(figure_layout(fig, 420), use_container_width=True)

    pretty = summary.copy()
    pretty["Avg_Score"] = pretty["Avg_Score"].map(safe_percent)
    pretty["Avg_Defect"] = pretty["Avg_Defect"].map(safe_percent)
    pretty["Fatal_Rate"] = pretty["Fatal_Rate"].map(safe_percent)
    pretty["Avg_Failed_Parameters"] = pretty["Avg_Failed_Parameters"].map(lambda v: safe_number(v, 2))
    st.dataframe(pretty, use_container_width=True, hide_index=True)


def render_qa_lens(df: pd.DataFrame, long_df: pd.DataFrame, score_col: str, min_audits: int) -> None:
    panel_title("QA scorer view", "See how Belina, Sami, or any other scorer distribute scores and where their audits surface issues.")
    qa_summary = group_summary(df, score_col, "Quality Auditor", min_audits=min_audits)
    if qa_summary.empty:
        st.info("No scorer-level data available.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(qa_summary, x="Quality Auditor", y="Avg_Score", color="Fatal_Rate", text="Audits")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure_layout(fig, 380), use_container_width=True)
    with c2:
        fig = px.box(df, x="Quality Auditor", y=score_col, points="all", color="Quality Auditor")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure_layout(fig, 380), use_container_width=True)

    qa_issue = (
        long_df[long_df["Status"].isin(["Fail", "Fatal", "Partial"])]
        .groupby(["Quality Auditor", "Parameter"])
        .size()
        .reset_index(name="Observations")
    )
    if not qa_issue.empty:
        heatmap = qa_issue.pivot(index="Parameter", columns="Quality Auditor", values="Observations").fillna(0)
        fig = px.imshow(heatmap, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        st.plotly_chart(figure_layout(fig, 520), use_container_width=True)
    else:
        st.info("No issue observations found for the current QA filters.")


def render_trends(df: pd.DataFrame, score_col: str, date_field: str) -> None:
    panel_title("Trend explorer", "Track score, defect rate, and fatal intensity over time at the grain that best fits the current range.")
    grain = st.selectbox("Time grain", ["Day", "Week", "Month"])
    trend = df.copy()
    if grain == "Day":
        trend["Bucket"] = pd.to_datetime(trend[date_field]).dt.date.astype(str)
    elif grain == "Week":
        trend["Bucket"] = pd.to_datetime(trend[date_field]).dt.to_period("W").astype(str)
    else:
        trend["Bucket"] = pd.to_datetime(trend[date_field]).dt.to_period("M").astype(str)

    agg = (
        trend.groupby("Bucket", dropna=False)
        .agg(
            Avg_Score=(score_col, "mean"),
            Avg_Defect=("Defect %", "mean"),
            Fatal_Rate=("Fatal Count", lambda s: float((s > 0).mean())),
            Audits=("Record ID", "count"),
        )
        .reset_index()
    )
    agg = agg.sort_values("Bucket")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg["Bucket"], y=agg["Avg_Score"], mode="lines+markers", name="Average score"))
        fig.add_trace(go.Scatter(x=agg["Bucket"], y=agg["Avg_Defect"], mode="lines+markers", name="Defect rate"))
        fig.add_trace(go.Scatter(x=agg["Bucket"], y=agg["Fatal_Rate"], mode="lines+markers", name="Fatal rate"))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure_layout(fig, 420), use_container_width=True)
    with c2:
        fig = px.bar(agg, x="Bucket", y="Audits", color="Fatal_Rate")
        st.plotly_chart(figure_layout(fig, 420), use_container_width=True)


def render_findings(df: pd.DataFrame, long_df: pd.DataFrame, compliance_df: pd.DataFrame) -> None:
    panel_title("Issue lab", "Prioritize the checks and compliance flags that are causing the most drag.")

    fail_summary = parameter_failure_summary(long_df)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        if not fail_summary.empty:
            fig = px.bar(fail_summary.head(15), x="Failure Rate", y="Parameter", orientation="h", text="Failed Audits")
            fig.update_xaxes(tickformat=".0%")
            fig.update_traces(textposition="outside")
            st.plotly_chart(figure_layout(fig, 500), use_container_width=True)
        else:
            st.info("No parameter failures in the current slice.")
    with c2:
        if not compliance_df.empty:
            comp = (
                compliance_df.groupby("Flag", dropna=False)
                .agg(Positive_Rate=("Positive", "mean"), Positive_Count=("Positive", "sum"))
                .reset_index()
                .sort_values("Positive_Rate", ascending=False)
            )
            fig = px.bar(comp, x="Positive_Rate", y="Flag", orientation="h", text="Positive_Count")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(figure_layout(fig, 500), use_container_width=True)
        else:
            st.info("No compliance data available.")

    status_counts = (
        long_df[long_df["Applicable"]]
        .groupby(["Section", "Status"]) 
        .size()
        .reset_index(name="Count")
    )
    if not status_counts.empty:
        fig = px.bar(status_counts, x="Section", y="Count", color="Status", barmode="stack")
        st.plotly_chart(figure_layout(fig, 420), use_container_width=True)

    if not fail_summary.empty:
        display_fail = fail_summary.copy()
        display_fail["Failure Rate"] = display_fail["Failure Rate"].map(safe_percent)
        st.dataframe(display_fail, use_container_width=True, hide_index=True)


def render_records(df: pd.DataFrame, long_df: pd.DataFrame, score_col: str) -> None:
    panel_title("Audit explorer", "Search specific audits, compare raw responses, and inspect each record without opening Excel.")
    search = st.text_input("Search by agent, record ID, reason, or QA scorer")
    record_view = df.copy()
    if search:
        mask = (
            record_view["Record ID"].astype(str).str.contains(search, case=False, na=False)
            | record_view["Agent Name"].astype(str).str.contains(search, case=False, na=False)
            | record_view["Reason for call"].astype(str).str.contains(search, case=False, na=False)
            | record_view["Quality Auditor"].astype(str).str.contains(search, case=False, na=False)
        )
        record_view = record_view[mask]

    st.dataframe(record_view[DISPLAY_COLUMNS], use_container_width=True, hide_index=True)
    if record_view.empty:
        return

    selected_record = st.selectbox("Open record", record_view["Record ID"].tolist())
    row = record_view.loc[record_view["Record ID"] == selected_record].iloc[0]
    st.markdown(
        f"""
        <div class="warning-shell">
            <strong>{row['Record ID']}</strong> · {row['Agent Name']} · {row['Call/Chat']} · {row['Quality Auditor']}<br>
            Score: <strong>{safe_percent(row[score_col])}</strong> · Defect %: <strong>{safe_percent(row['Defect %'])}</strong> · Fatal Count: <strong>{int(row['Fatal Count'])}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_rows = []
    for section in SECTION_ORDER:
        col = f"Section Score :: {section}"
        if col in row.index:
            section_rows.append({"Section": section, "Score": row[col]})
    section_df = pd.DataFrame(section_rows)

    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        if not section_df.empty:
            fig = px.bar(section_df, x="Score", y="Section", orientation="h", text="Score")
            fig.update_xaxes(tickformat=".0%")
            fig.update_traces(texttemplate="%{text:.0%}")
            st.plotly_chart(figure_layout(fig, 360), use_container_width=True)
    with c2:
        record_long = long_df[long_df["Record ID"] == selected_record].copy()
        if not record_long.empty:
            record_long["Score % Display"] = record_long["Score %"].map(safe_percent)
            st.dataframe(
                record_long[["Section", "Parameter", "Raw Response", "Status", "Score", "Score % Display", "Applicable", "Fatal"]],
                use_container_width=True,
                hide_index=True,
            )


def render_setup_guide() -> None:
    panel_title("Deployment and repo hygiene", "This project is packaged to be safe for GitHub and Streamlit Cloud once you add secrets locally and in the Streamlit app settings.")
    st.markdown(
        """
        - `secrets.toml` is intentionally excluded from git.
        - `data/` is kept empty except for placeholders, so raw QA files are not committed.
        - OAuth tokens are stored only in Streamlit session state, not on disk.
        - The app supports three source modes: Google Sheets via OAuth, synced local workbook path, and manual upload.
        - The scoring engine recalculates everything from raw responses, so workbook formula errors do not affect the dashboard.
        """
    )
    st.code(
        """git init
git add .
git commit -m "Initial QA dashboard"
git remote add origin <your-repo-url>
git push -u origin main""",
        language="bash",
    )


def main() -> None:
    try:
        df, long_df, compliance_df, source_mode, source_path, workbook_name = load_data_ui()
    except Exception as exc:
        st.error(f"Could not load the dataset: {exc}")
        st.stop()

    if df is None or df.empty:
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-title">QA Audit Intelligence</div>
                <div class="hero-subtitle">Connect a Google Sheet, point to a synced workbook path, or upload a workbook to start exploring the data.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_setup_guide()
        return

    ctx = apply_filters(df, long_df, compliance_df)
    filtered_df = ctx["df"]
    filtered_long = ctx["long_df"]
    filtered_compliance = ctx["compliance_df"]

    hero_banner(source_mode, workbook_name, source_path, filtered_df, ctx["score_mode_label"])
    filter_chips(ctx)

    tabs = st.tabs([
        "Overview",
        "Performance lens",
        "QA scorer view",
        "Trend explorer",
        "Issue lab",
        "Audit explorer",
        "Setup",
    ])

    with tabs[0]:
        render_overview(filtered_df, filtered_long, ctx["score_col"], ctx["date_field"])
    with tabs[1]:
        render_team_breakdown(filtered_df, ctx["score_col"], ctx["min_audits"])
    with tabs[2]:
        render_qa_lens(filtered_df, filtered_long, ctx["score_col"], ctx["min_audits"])
    with tabs[3]:
        render_trends(filtered_df, ctx["score_col"], ctx["date_field"])
    with tabs[4]:
        render_findings(filtered_df, filtered_long, filtered_compliance)
    with tabs[5]:
        render_records(filtered_df, filtered_long, ctx["score_col"])
    with tabs[6]:
        render_setup_guide()


if __name__ == "__main__":
    main()
