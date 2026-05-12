from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from google_sources import (
    disconnect_google,
    extract_sheet_ref,
    get_auth_url,
    get_credentials,
    handle_oauth_callback,
    list_worksheets,
    load_google_oauth_dataset,
    load_published_google_sheet,
)
from scoring_engine import (
    FILTER_COLUMNS,
    META_COLUMNS_CANONICAL,
    ScoredDataset,
    display_number,
    display_percent,
    group_summary,
    load_dataset_from_upload,
    parameter_failure_summary,
)

st.set_page_config(
    page_title="QA Audit Intelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCORE_MODES = {
    "With fatal rule": "Overall Score (With Fatal)",
    "Without fatal rule": "Overall Score (Without Fatal)",
}

APP_CSS = """
<style>
[data-testid="stSidebar"] {background: linear-gradient(180deg,#F8FAFC 0%,#EEF4FF 100%);}
.block-container {padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1600px;}
.hero {
    border: 1px solid rgba(15,23,42,.08);
    border-radius: 28px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1rem;
    background: radial-gradient(circle at top right, rgba(124,58,237,.14), transparent 28%),
                linear-gradient(135deg, rgba(255,255,255,.98), rgba(248,250,252,.95));
    box-shadow: 0 22px 55px rgba(15,23,42,.08);
}
.hero h1 {font-size: 2rem; line-height: 1.05; margin: 0 0 .35rem 0; color:#0F172A;}
.hero p {font-size: .98rem; color:#475569; margin: 0 0 .75rem 0;}
.pill {display:inline-flex; align-items:center; gap:.35rem; padding:.34rem .72rem; border-radius:999px; background:#FFFFFF; color:#4338CA; border:1px solid rgba(67,56,202,.15); font-size:.8rem; font-weight:700; margin:.15rem .32rem .15rem 0;}
.card {border:1px solid rgba(15,23,42,.08); border-radius:22px; padding:1rem; background:linear-gradient(180deg,#FFFFFF,#F8FAFC); box-shadow:0 14px 35px rgba(15,23,42,.07); min-height:126px;}
.card-label {font-size:.75rem; font-weight:800; text-transform:uppercase; letter-spacing:.04em; color:#64748B; margin-bottom:.28rem;}
.card-value {font-size:1.85rem; font-weight:850; color:#0F172A; line-height:1.1;}
.card-sub {font-size:.88rem; color:#64748B; margin-top:.25rem;}
.section-title {font-size:1.1rem; font-weight:850; color:#0F172A; margin:.45rem 0 .15rem;}
.section-copy {font-size:.92rem; color:#64748B; margin-bottom:.75rem;}
.warning {padding:.8rem 1rem; border-radius:18px; background:#FFF7ED; border:1px solid #FED7AA; color:#9A3412;}
.good {padding:.8rem 1rem; border-radius:18px; background:#F0FDF4; border:1px solid #BBF7D0; color:#166534;}
.small-muted {font-size:.82rem; color:#64748B;}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, copy: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div><div class='section-copy'>{copy}</div>", unsafe_allow_html=True)


def fig_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=42, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        font=dict(color="#0F172A"),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.22)", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def _load_upload_cached(data: bytes, filename: str) -> ScoredDataset:
    class UploadedLike:
        def __init__(self, name: str, payload: bytes):
            self.name = name
            self._payload = payload

        def getvalue(self) -> bytes:
            return self._payload

    return load_dataset_from_upload(UploadedLike(filename, data))


@st.cache_data(show_spinner=False)
def _load_published_cached(url: str) -> ScoredDataset:
    return load_published_google_sheet(url)


@st.cache_data(show_spinner=False)
def _load_oauth_cached(spreadsheet_id: str, audit_sheet_title: str, token_fingerprint: str) -> ScoredDataset:
    # token_fingerprint only invalidates cache when the Google session changes.
    del token_fingerprint
    return load_google_oauth_dataset(spreadsheet_id, audit_sheet_title)


def source_picker() -> ScoredDataset | None:
    handle_oauth_callback()
    st.sidebar.header("Data source")
    source_mode = st.sidebar.radio(
        "Choose how to load QA data",
        ["Upload .xlsx or .csv", "Google published link", "Google OAuth"],
        index=0,
    )

    if source_mode == "Upload .xlsx or .csv":
        uploaded = st.sidebar.file_uploader("Upload workbook or CSV", type=["xlsx", "xlsm", "csv"])
        if uploaded is None:
            return None
        return _load_upload_cached(uploaded.getvalue(), uploaded.name)

    if source_mode == "Google published link":
        st.sidebar.caption("Use the published Google Sheets web page or CSV link. If the Scoring tab is not visible publicly, fallback scoring is used.")
        published_url = st.sidebar.text_input("Published Google Sheet link")
        if not published_url:
            return None
        if st.sidebar.button("Load published sheet", type="primary") or "published_dataset_url" not in st.session_state or st.session_state.get("published_dataset_url") != published_url:
            st.session_state["published_dataset_url"] = published_url
            return _load_published_cached(published_url)
        return _load_published_cached(published_url)

    st.sidebar.caption("Use this for private Google Sheets. The Google Cloud OAuth redirect URI must exactly match your Streamlit app URL.")
    pending_url = st.session_state.pop("pending_google_sheet_url", "")
    sheet_url = st.sidebar.text_input("Private Google Sheet link", value=pending_url, key="oauth_sheet_url")
    if st.sidebar.button("Disconnect Google"):
        disconnect_google()
        st.rerun()

    creds = get_credentials()
    if not sheet_url:
        return None
    try:
        sheet_ref = extract_sheet_ref(sheet_url)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return None

    if creds is None:
        try:
            auth_url = get_auth_url(sheet_url)
            st.sidebar.link_button("Authenticate with Google", auth_url, type="primary")
        except Exception as exc:
            st.sidebar.error(str(exc))
        return None

    st.sidebar.success("Google session active")
    try:
        worksheets = list_worksheets(sheet_ref.spreadsheet_id)
    except Exception as exc:
        st.sidebar.error(f"Could not inspect spreadsheet: {exc}")
        return None
    if not worksheets:
        st.sidebar.error("No worksheets were found in this spreadsheet.")
        return None

    titles = [w["title"] for w in worksheets]
    default_index = 0
    if sheet_ref.gid:
        for idx, w in enumerate(worksheets):
            if w.get("gid") == sheet_ref.gid:
                default_index = idx
                break
    selected_sheet = st.sidebar.selectbox("Audit worksheet", titles, index=default_index)
    token_fingerprint = str(hash(st.session_state.get("google_credentials", "")))
    if st.sidebar.button("Refresh Google data"):
        _load_oauth_cached.clear()
    return _load_oauth_cached(sheet_ref.spreadsheet_id, selected_sheet, token_fingerprint)


def apply_filters(dataset: ScoredDataset) -> dict[str, Any]:
    df = dataset.df.copy()
    long_df = dataset.long_df.copy()
    compliance_df = dataset.compliance_df.copy()

    st.sidebar.header("Filters")
    score_mode_label = st.sidebar.radio("Score mode", list(SCORE_MODES.keys()), horizontal=False)
    score_col = SCORE_MODES[score_mode_label]

    available_date_fields = [c for c in ["Call Date", "Audit Date"] if c in df.columns and df[c].notna().any()]
    date_field = st.sidebar.selectbox("Date field", available_date_fields or ["Audit Date"])
    if available_date_fields:
        valid_dates = pd.to_datetime(df[date_field], errors="coerce").dropna()
        min_date = valid_dates.min().date() if not valid_dates.empty else date.today()
        max_date = valid_dates.max().date() if not valid_dates.empty else date.today()
        date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range
        df = df[df[date_field].between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")]
    else:
        start_date = end_date = None

    selections: dict[str, list[str]] = {}
    filtered = df.copy()
    for col in FILTER_COLUMNS:
        if col not in filtered.columns:
            continue
        options = sorted([x for x in filtered[col].dropna().astype(str).unique().tolist() if x.strip()])
        selected = st.sidebar.multiselect(col, options, key=f"flt_{col}")
        selections[col] = selected
        if selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]

    min_audits = st.sidebar.slider("Minimum audits in ranking", 1, max(1, int(filtered.shape[0] or 1)), 1)
    if st.sidebar.toggle("Only fatal audits", value=False):
        filtered = filtered[filtered["Has Fatal"]]
    if st.sidebar.toggle("Only audits with defects", value=False):
        filtered = filtered[filtered["Failed Parameter Count"].gt(0)]

    ids = set(filtered["Record ID"].tolist())
    return {
        "df": filtered,
        "long_df": long_df[long_df["Record ID"].isin(ids)].copy(),
        "compliance_df": compliance_df[compliance_df["Record ID"].isin(ids)].copy(),
        "score_col": score_col,
        "score_mode_label": score_mode_label,
        "date_field": date_field,
        "start_date": start_date,
        "end_date": end_date,
        "min_audits": min_audits,
        "selections": selections,
    }


def hero(dataset: ScoredDataset, ctx: dict[str, Any]) -> None:
    diag = dataset.diagnostics
    chips = [
        f"<span class='pill'>Source · {diag.get('source_name', 'Loaded data')}</span>",
        f"<span class='pill'>Audits · {len(ctx['df']):,}</span>",
        f"<span class='pill'>Score · {ctx['score_mode_label']}</span>",
        f"<span class='pill'>Header row · {diag.get('detected_header_row_excel', '-')}</span>",
    ]
    if diag.get("used_fallback_scoring"):
        chips.append("<span class='pill'>Scoring · fallback rules</span>")
    else:
        chips.append("<span class='pill'>Scoring · workbook Scoring tab</span>")

    st.markdown(
        f"""
        <div class="hero">
            <h1>QA Audit Intelligence</h1>
            <p>Weighted QA analytics that recalculates scores from raw audit responses only. Columns are matched by names, not fixed Excel positions.</p>
            {''.join(chips)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(df: pd.DataFrame, long_df: pd.DataFrame, ctx: dict[str, Any]) -> None:
    score_col = ctx["score_col"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Average score", display_percent(df[score_col].mean()), "Filtered audits")
    with c2:
        kpi_card("Audits", display_number(len(df)), "Rows after filters")
    with c3:
        kpi_card("Fatal rate", display_percent(df["Has Fatal"].mean()), "Any Fatal in raw audit cells")
    with c4:
        kpi_card("Defect rate", display_percent(df["Defect %"].mean()), "Failed or partial applicable checks")
    with c5:
        kpi_card("Agents", display_number(df["Agent Name"].nunique()), "Unique agents")

    left, right = st.columns([1.35, 1])
    with left:
        section_title("Score trend", "Audit volume is shown as bars, average score as the line.")
        date_field = ctx["date_field"]
        if date_field in df.columns and df[date_field].notna().any():
            daily = (
                df.groupby(date_field, dropna=False)
                .agg(Avg_Score=(score_col, "mean"), Audits=("Record ID", "count"), Fatal_Rate=("Has Fatal", "mean"))
                .reset_index()
                .sort_values(date_field)
            )
            fig = go.Figure()
            fig.add_bar(x=daily[date_field], y=daily["Audits"], name="Audits", opacity=.28, yaxis="y2")
            fig.add_scatter(x=daily[date_field], y=daily["Avg_Score"], name="Average score", mode="lines+markers")
            fig.update_layout(yaxis=dict(tickformat=".0%", title="Average score"), yaxis2=dict(title="Audits", overlaying="y", side="right", showgrid=False))
            st.plotly_chart(fig_layout(fig, 420), use_container_width=True)
        else:
            st.info("No valid date field found for trend analysis.")

    with right:
        section_title("Section score", "Every section is reweighted based only on applicable cells.")
        section_cols = [c for c in df.columns if c.startswith("Section Score :: ")]
        rows = [{"Section": c.replace("Section Score :: ", ""), "Average Score": df[c].mean()} for c in section_cols]
        section_df = pd.DataFrame(rows).sort_values("Average Score") if rows else pd.DataFrame()
        if not section_df.empty:
            fig = px.bar(section_df, x="Average Score", y="Section", orientation="h", text="Average Score")
            fig.update_xaxes(tickformat=".0%")
            fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
            st.plotly_chart(fig_layout(fig, 420), use_container_width=True)
        else:
            st.info("No section-level scores available.")

    c1, c2 = st.columns([1, 1])
    with c1:
        section_title("Score distribution", "Useful for spotting a cluster of low or fatal audits.")
        fig = px.histogram(df, x=score_col, color="Call/Chat" if "Call/Chat" in df.columns else None, nbins=16)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig_layout(fig, 360), use_container_width=True)
    with c2:
        section_title("Top issue drivers", "Parameters with the highest issue rate in the current slice.")
        findings = parameter_failure_summary(long_df)
        if not findings.empty:
            fig = px.bar(findings.head(10), x="Failure_Rate", y="Parameter", orientation="h", text="Issue_Audits", color="Section")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig_layout(fig, 360), use_container_width=True)
        else:
            st.info("No issue drivers in this filtered view.")


def render_breakdowns(df: pd.DataFrame, ctx: dict[str, Any]) -> None:
    score_col = ctx["score_col"]
    section_title("Breakdown explorer", "Compare scores by agent, QA scorer, call/chat, or business dimension.")
    col1, col2, col3 = st.columns([1, 1, 1])
    group_options = [c for c in FILTER_COLUMNS if c in df.columns] + ["Audit Month"]
    primary = col1.selectbox("Primary breakdown", group_options, index=group_options.index("Agent Name") if "Agent Name" in group_options else 0)
    split_options = ["None"] + [c for c in group_options if c != primary]
    split = col2.selectbox("Optional split", split_options, index=split_options.index("Quality Auditor") if "Quality Auditor" in split_options else 0)
    sort_by = col3.selectbox("Sort by", ["Avg_Score", "Fatal_Rate", "Avg_Defect", "Audits"], index=0)

    summary = group_summary(df, primary, score_col, min_audits=ctx["min_audits"])
    if summary.empty:
        st.info("No data available with current filters and minimum audit threshold.")
        return

    chart_df = summary.head(20).copy()
    chart_df = chart_df.sort_values(sort_by, ascending=(sort_by not in {"Avg_Score", "Audits"}))
    left, right = st.columns([1.25, 1])
    with left:
        fig = px.bar(chart_df, x=primary, y="Avg_Score", color="Fatal_Rate", hover_data=["Audits", "Avg_Defect"])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_layout(fig, 430), use_container_width=True)
    with right:
        fig = px.scatter(summary, x="Avg_Defect", y="Avg_Score", size="Audits", color="Fatal_Rate", hover_name=primary)
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_layout(fig, 430), use_container_width=True)

    if split != "None":
        split_df = (
            df.groupby([primary, split], dropna=False)
            .agg(Avg_Score=(score_col, "mean"), Audits=("Record ID", "count"))
            .reset_index()
        )
        split_df = split_df[split_df["Audits"] >= ctx["min_audits"]]
        if not split_df.empty:
            fig = px.bar(split_df, x=primary, y="Avg_Score", color=split, barmode="group", hover_data=["Audits"])
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_layout(fig, 450), use_container_width=True)

    pretty = summary.copy()
    for col in ["Avg_Score", "Avg_Without_Fatal", "Avg_With_Fatal", "Avg_Defect", "Fatal_Rate"]:
        pretty[col] = pretty[col].map(display_percent)
    pretty["Avg_Failed_Parameters"] = pretty["Avg_Failed_Parameters"].map(lambda x: display_number(x, 2))
    pretty["Avg_Applicable_Parameters"] = pretty["Avg_Applicable_Parameters"].map(lambda x: display_number(x, 2))
    st.dataframe(pretty, use_container_width=True, hide_index=True)


def render_qa_view(df: pd.DataFrame, long_df: pd.DataFrame, ctx: dict[str, Any]) -> None:
    score_col = ctx["score_col"]
    section_title("QA scorer view", "Compare Belina, Ayusha, or any scorer after all active filters are applied.")
    qa_summary = group_summary(df, "Quality Auditor", score_col, min_audits=ctx["min_audits"])
    if qa_summary.empty:
        st.info("No QA scorer data available.")
        return

    left, right = st.columns([1, 1])
    with left:
        fig = px.bar(qa_summary, x="Quality Auditor", y="Avg_Score", color="Fatal_Rate", text="Audits")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_layout(fig, 390), use_container_width=True)
    with right:
        fig = px.box(df, x="Quality Auditor", y=score_col, color="Call/Chat" if "Call/Chat" in df.columns else None, points="all")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_layout(fig, 390), use_container_width=True)

    issue = long_df[long_df["Applicable"] & long_df["Status"].isin(["Fail", "Partial", "Fatal"])]
    if not issue.empty:
        heat = issue.groupby(["Parameter", "Quality Auditor"]).size().reset_index(name="Issues")
        pivot = heat.pivot(index="Parameter", columns="Quality Auditor", values="Issues").fillna(0)
        fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Purples")
        st.plotly_chart(fig_layout(fig, 560), use_container_width=True)


def render_findings(long_df: pd.DataFrame, compliance_df: pd.DataFrame) -> None:
    section_title("Parameter and compliance findings", "Scored parameters affect score. CMM compliance flags are record-keeping only.")
    findings = parameter_failure_summary(long_df)
    left, right = st.columns([1.15, 1])
    with left:
        if not findings.empty:
            fig = px.bar(findings.head(18), x="Failure_Rate", y="Parameter", orientation="h", color="Section", text="Issue_Audits")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig_layout(fig, 540), use_container_width=True)
        else:
            st.info("No scored parameter issues found.")
    with right:
        if not compliance_df.empty:
            comp = (
                compliance_df.groupby("Flag", dropna=False)
                .agg(Positive_Count=("Positive", "sum"), Observations=("Record ID", "count"), Positive_Rate=("Positive", "mean"))
                .reset_index()
                .sort_values("Positive_Rate", ascending=False)
            )
            fig = px.bar(comp, x="Positive_Rate", y="Flag", orientation="h", text="Positive_Count")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig_layout(fig, 540), use_container_width=True)
        else:
            st.info("No compliance flags available.")

    if not findings.empty:
        table = findings.copy()
        table["Failure_Rate"] = table["Failure_Rate"].map(display_percent)
        table["Avg_Score_Pct"] = table["Avg_Score_Pct"].map(display_percent)
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_records(df: pd.DataFrame, long_df: pd.DataFrame, ctx: dict[str, Any]) -> None:
    score_col = ctx["score_col"]
    section_title("Audit explorer", "Inspect a specific audit and see how every applicable parameter was scored.")
    search = st.text_input("Search by agent, QA scorer, reason, record ID, or mobile/link")
    view = df.copy()
    if search:
        fields = ["Record ID", "Agent Name", "Quality Auditor", "Reason for call", "Mobile Number", "Agent's Response"]
        mask = False
        for field in fields:
            if field in view.columns:
                mask = mask | view[field].astype(str).str.contains(search, case=False, na=False)
        view = view[mask]

    display_cols = [
        "Record ID", "Call Date", "Audit Date", "Quality Auditor", "Call/Chat", "Agent Name",
        "Supervisor name", "LOB", "Sub-LOB", "Reason for call", score_col, "Defect %", "Has Fatal", "Failed Parameter Count",
    ]
    display_cols = [c for c in display_cols if c in view.columns]
    styled_view = view[display_cols].copy()
    for col in [score_col, "Defect %"]:
        if col in styled_view.columns:
            styled_view[col] = styled_view[col].map(display_percent)
    st.dataframe(styled_view, use_container_width=True, hide_index=True)

    if view.empty:
        return
    selected = st.selectbox("Open audit record", view["Record ID"].tolist())
    record = view[view["Record ID"] == selected].iloc[0]
    st.markdown(
        f"<div class='warning'><b>{selected}</b> · {record.get('Agent Name','Unknown')} · {record.get('Quality Auditor','Unknown')} · Score: <b>{display_percent(record.get(score_col))}</b> · Defect: <b>{display_percent(record.get('Defect %'))}</b> · Fatal: <b>{record.get('Has Fatal')}</b></div>",
        unsafe_allow_html=True,
    )
    record_long = long_df[long_df["Record ID"] == selected].copy()
    if not record_long.empty:
        record_long["Score %"] = record_long["Score %"].map(display_percent)
        st.dataframe(
            record_long[["Section", "Parameter", "Matched Column", "Raw Response", "Status", "Max Score", "Score", "Score %", "Applicable", "Fatal"]],
            use_container_width=True,
            hide_index=True,
        )


def render_audit(dataset: ScoredDataset) -> None:
    section_title("Data audit", "Checks that keep the app resilient when the workbook structure changes.")
    diag = dataset.diagnostics
    if diag.get("unmatched_scoring_parameters"):
        st.markdown("<div class='warning'><b>Unmatched scoring parameters found.</b> These parameters were not used until the column name is fixed or added.</div>", unsafe_allow_html=True)
        st.write(diag.get("unmatched_scoring_parameters"))
    else:
        st.markdown("<div class='good'><b>All scoring parameters matched.</b> No score column is tied to fixed Excel positions.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Detected header row", diag.get("detected_header_row_excel", "-"))
    c2.metric("Raw end column", diag.get("raw_data_end_column_excel", "-"))
    c3.metric("Records loaded", diag.get("records_loaded", "-"))

    if diag.get("inferred_headers"):
        st.markdown("#### Inferred blank headers")
        st.dataframe(pd.DataFrame(diag["inferred_headers"]), use_container_width=True, hide_index=True)

    unknown = pd.DataFrame(diag.get("unknown_scoring_values", []))
    st.markdown("#### Unknown scoring values")
    if unknown.empty:
        st.success("No unknown scoring responses found.")
    else:
        st.warning("These raw responses were ignored because they do not exist in the scoring rule and are not common Y/N/BE/ME/EE/Fatal values.")
        st.dataframe(unknown, use_container_width=True, hide_index=True)

    st.markdown("#### Scoring rules used")
    st.dataframe(dataset.scoring_df, use_container_width=True, hide_index=True)


def render_setup() -> None:
    section_title("Setup guide", "How to sync data and deploy safely.")
    st.markdown(
        """
        **Scoring formula used in Python**

        `Applicable` = response is not empty, `NA`, `N/A`, `-`, or equivalent.  
        `Parameter Score` = value from the Scoring sheet. `Fatal` scores `0` for that parameter.  
        `Overall Score Without Fatal` = `SUM(parameter scores) / SUM(max score of applicable parameters)`.  
        `Overall Score With Fatal` = `0` if any raw audit cell up to `Escalation Denied` contains `Fatal`, otherwise the same as without fatal.  
        `Defect %` = failed, partial, or fatal applicable parameters divided by total applicable parameters.  
        CMM compliance fields have max score `0`, so they are tracked only and do not affect the score.

        **Google Sheets sync options**

        1. For private sheets, use **Google OAuth** and add the OAuth credentials in Streamlit secrets.  
        2. For public sheets, publish the tab as a web page and use the **Google published link** mode.  
        3. For manual analysis, upload `.xlsx`, `.xlsm`, or `.csv`.
        """
    )
    st.code(
        """# .streamlit/secrets.toml
[google_oauth]
client_id = "YOUR_GOOGLE_CLIENT_ID"
client_secret = "YOUR_GOOGLE_CLIENT_SECRET"
redirect_uri = "https://YOUR-APP.streamlit.app/"

[google_sheets]
scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
""",
        language="toml",
    )
    st.markdown(
        """
        **Important:** do not commit `secrets.toml`. Rotate any OAuth client secret that has been pasted into chat, tickets, screenshots, or source code.
        """
    )


def empty_state() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>QA Audit Intelligence</h1>
            <p>Upload the workbook, paste a published Google Sheet link, or connect a private Google Sheet with OAuth.</p>
            <span class="pill">No fixed column positions</span>
            <span class="pill">Formula-based scoring</span>
            <span class="pill">Fatal-aware score modes</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_setup()


def main() -> None:
    try:
        dataset = source_picker()
    except Exception as exc:
        st.error(f"Could not load the selected source: {exc}")
        st.stop()

    if dataset is None or dataset.df.empty:
        empty_state()
        return

    ctx = apply_filters(dataset)
    df = ctx["df"]
    long_df = ctx["long_df"]
    compliance_df = ctx["compliance_df"]

    hero(dataset, ctx)
    if df.empty:
        st.warning("No records match the selected filters.")
        render_audit(dataset)
        return

    tabs = st.tabs(["Overview", "Breakdowns", "QA scorer view", "Findings", "Audit explorer", "Data audit", "Setup"])
    with tabs[0]:
        render_overview(df, long_df, ctx)
    with tabs[1]:
        render_breakdowns(df, ctx)
    with tabs[2]:
        render_qa_view(df, long_df, ctx)
    with tabs[3]:
        render_findings(long_df, compliance_df)
    with tabs[4]:
        render_records(df, long_df, ctx)
    with tabs[5]:
        render_audit(dataset)
    with tabs[6]:
        render_setup()


if __name__ == "__main__":
    main()
