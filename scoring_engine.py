from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_SCORING_ROWS: list[dict[str, Any]] = [
    {"Parameter": "Call Compliance", "Sub Parameter": "Greeting", "Max Score": 2, "Scoring": "Y=2,N=0"},
    {"Parameter": "Call Compliance", "Sub Parameter": "Permission", "Max Score": 2, "Scoring": "Y=2,N=0"},
    {"Parameter": "Call Compliance", "Sub Parameter": "Closing", "Max Score": 2, "Scoring": "Y=2,N=0"},
    {"Parameter": "Call Etiquette", "Sub Parameter": "Active Listening", "Max Score": 6, "Scoring": "BE=0,ME=3,EE=6"},
    {"Parameter": "Call Etiquette", "Sub Parameter": "Empathy/Apology", "Max Score": 6, "Scoring": "BE=0,ME=3,EE=6"},
    {"Parameter": "Call Etiquette", "Sub Parameter": "Politeness", "Max Score": 6, "Scoring": "BE=0,ME=3,EE=6"},
    {"Parameter": "Call Etiquette", "Sub Parameter": "Preferred Language", "Max Score": 3, "Scoring": "BE=0,ME=2,EE=3"},
    {"Parameter": "Call Etiquette", "Sub Parameter": "Voice Clarity/Tone", "Max Score": 3, "Scoring": "BE=0,ME=2,EE=3"},
    {"Parameter": "Call Etiquette", "Sub Parameter": "Context Setting", "Max Score": 3, "Scoring": "BE=0,ME=2,EE=3"},
    {"Parameter": "Call Etiquette", "Sub Parameter": "Grammar & Sentence", "Max Score": 6, "Scoring": "BE=0,ME=3,EE=6"},
    {"Parameter": "Query Resolution", "Sub Parameter": "Probing", "Max Score": 6, "Scoring": "BE=0,ME=3,EE=6"},
    {"Parameter": "Query Resolution", "Sub Parameter": "Correct Resolution", "Max Score": 10, "Scoring": "Y=10,N=0"},
    {"Parameter": "Query Resolution", "Sub Parameter": "Complete Resolution", "Max Score": 10, "Scoring": "Y=10,N=0"},
    {"Parameter": "Query Resolution", "Sub Parameter": "TAT informed", "Max Score": 3, "Scoring": "Y=3,N=0"},
    {"Parameter": "Call Compliance", "Sub Parameter": "Hold (<=1min / 3x)", "Max Score": 3, "Scoring": "Y=3,N=0"},
    {"Parameter": "Call Compliance", "Sub Parameter": "Transfer/Escalation", "Max Score": 2, "Scoring": "Y=2,N=0"},
    {"Parameter": "Call Compliance", "Sub Parameter": "Complain/Ticket Raised", "Max Score": 3, "Scoring": "Y=3,N=0"},
    {"Parameter": "Query Resolution", "Sub Parameter": "Response Time (<10s)", "Max Score": 4, "Scoring": "Y=4,N=0"},
    {"Parameter": "Call Compliance", "Sub Parameter": "All Queries Tagged", "Max Score": 4, "Scoring": "Y=4,N=0"},
    {"Parameter": "Call Compliance", "Sub Parameter": "Complete Tagging", "Max Score": 4, "Scoring": "Y=4,N=0"},
    {"Parameter": "Call Compliance", "Sub Parameter": "Correct Tagging", "Max Score": 6, "Scoring": "Y=6,N=0"},
    {"Parameter": "Sales $ Compliance", "Sub Parameter": "Upsell/Promotions", "Max Score": 4, "Scoring": "Y=4,N=0"},
    {"Parameter": "Sales $ Compliance", "Sub Parameter": "Waiver/Discount/Coupon", "Max Score": 2, "Scoring": "Y=2,N=0"},
    {"Parameter": "CMM Compliance", "Sub Parameter": "Condescending/Rude/Abuse", "Max Score": 0, "Scoring": "Y,N"},
    {"Parameter": "CMM Compliance", "Sub Parameter": "Disconnect Line", "Max Score": 0, "Scoring": "Y,N"},
    {"Parameter": "CMM Compliance", "Sub Parameter": "Personal Info Violation", "Max Score": 0, "Scoring": "Y,N"},
    {"Parameter": "CMM Compliance", "Sub Parameter": "Complaints tagged wrongly", "Max Score": 0, "Scoring": "Y,N"},
    {"Parameter": "CMM Compliance", "Sub Parameter": "Blind Transfer", "Max Score": 0, "Scoring": "Y,N"},
    {"Parameter": "CMM Compliance", "Sub Parameter": "Escalation Denied", "Max Score": 0, "Scoring": "Y,N"},
]

META_COLUMNS_CANONICAL = [
    "S.No.",
    "Call Date",
    "Audit Date",
    "Audit Month",
    "Quality Auditor",
    "Call/Chat",
    "Agent Name",
    "Supervisor name",
    "LOB",
    "Sub-LOB",
    "Mobile Number",
    "Reason for call",
    "Agent's Response",
]

FILTER_COLUMNS = [
    "Audit Month",
    "Quality Auditor",
    "Call/Chat",
    "Agent Name",
    "Supervisor name",
    "LOB",
    "Sub-LOB",
    "Reason for call",
]

BREAKDOWN_COLUMNS = [
    "Audit Date",
    "Audit Month",
    "Quality Auditor",
    "Call/Chat",
    "Agent Name",
    "Supervisor name",
    "LOB",
    "Sub-LOB",
    "Reason for call",
]

NA_VALUES = {"", "NA", "N/A", "NONE", "NULL", "-", "NIL", "NOT APPLICABLE", "NAN"}
FATAL_VALUES = {"FATAL"}
YES_VALUES = {"Y", "YES", "TRUE", "PASS", "PASSED", "1"}
NO_VALUES = {"N", "NO", "FALSE", "FAIL", "FAILED", "0"}


@dataclass(frozen=True)
class ScoringRule:
    section: str
    parameter: str
    max_score: float
    scoring_text: str
    mapping: dict[str, float]
    key: str


@dataclass
class ScoredDataset:
    df: pd.DataFrame
    long_df: pd.DataFrame
    compliance_df: pd.DataFrame
    scoring_df: pd.DataFrame
    diagnostics: dict[str, Any]
    data_view: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    existing_calc_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def normalize_header(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = text.replace("≤", "<=")
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "", text)
    replacements = {
        "qualityauditor": "qualityauditor",
        "auditor": "qualityauditor",
        "callchat": "callchat",
        "callorchat": "callchat",
        "agent": "agentname",
        "agentname": "agentname",
        "supervisor": "supervisorname",
        "supervisorname": "supervisorname",
        "sublineofbusiness": "sublob",
        "sublob": "sublob",
        "reasonforcallchat": "reasonforcall",
        "reasonforcall": "reasonforcall",
        "agentsresponse": "agentsresponse",
        "mobile": "mobilenumber",
        "phonenumber": "mobilenumber",
        "contactnumber": "mobilenumber",
        "holdupto1min3x": "hold1min3x",
        "hold1min3x": "hold1min3x",
        "hold1minx3": "hold1min3x",
        "hold1min3": "hold1min3x",
        "waiverdiscount": "waiverdiscountcoupon",
        "waiverdiscounts": "waiverdiscountcoupon",
        "waiverdiscountcoupon": "waiverdiscountcoupon",
        "complaintraised": "complainticketraised",
        "complainticketraised": "complainticketraised",
        "complaintticketraised": "complainticketraised",
        "complaintstaggedwrongly": "complaintstaggedwrongly",
        "complainttaggedwrongly": "complaintstaggedwrongly",
        "overallscorewithfatal": "ovaallscorewithfatal",
        "ovaallscorewithfatal": "ovaallscorewithfatal",
        "overallscorewithoutfatal": "ovaallscorewithoutfatal",
        "ovaallscorewithoutfatal": "ovaallscorewithoutfatal",
        "defect": "defectpct",
        "defectpercent": "defectpct",
        "defectpct": "defectpct",
    }
    return replacements.get(text, text)


def normalize_response(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text.upper()


def make_unique_headers(headers: Iterable[Any]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for idx, value in enumerate(headers):
        text = "" if pd.isna(value) else str(value).strip()
        if not text:
            text = f"Unnamed {idx + 1}"
        base = text
        count = seen.get(base, 0)
        if count:
            text = f"{base}__{count + 1}"
        seen[base] = count + 1
        out.append(text)
    return out


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    return SequenceMatcher(None, a, b).ratio()


def parse_scoring_mapping(scoring_text: Any) -> dict[str, float]:
    if scoring_text is None or pd.isna(scoring_text):
        return {}
    mapping: dict[str, float] = {}
    for part in str(scoring_text).split(","):
        token = part.strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            mapping[normalize_response(key)] = float(str(value).strip().replace("%", ""))
        except ValueError:
            continue
    return mapping


def detect_header_row(raw: pd.DataFrame) -> int:
    required = {"qualityauditor", "agentname"}
    strong = {"escalationdenied", "greeting"}
    best_idx = 0
    best_score = -1
    for idx in range(min(len(raw), 20)):
        normalized = {normalize_header(v) for v in raw.iloc[idx].tolist()}
        score = sum(1 for key in required if key in normalized) * 3 + sum(1 for key in strong if key in normalized) * 2
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_score < 6:
        raise ValueError("Could not detect the audit header row. Expected Quality Auditor, Agent Name, Greeting, and Escalation Denied.")
    return best_idx


def find_first_matching_header(headers: list[Any], target: str, threshold: float = 0.87) -> int | None:
    target_norm = normalize_header(target)
    candidates = [(idx, normalize_header(h)) for idx, h in enumerate(headers)]
    for idx, norm in candidates:
        if norm == target_norm:
            return idx
    best = max(((idx, similarity(norm, target_norm)) for idx, norm in candidates), key=lambda x: x[1], default=(None, 0.0))
    if best[0] is not None and best[1] >= threshold:
        return int(best[0])
    return None


def find_scoring_header_row(raw: pd.DataFrame) -> int | None:
    for idx in range(min(len(raw), 20)):
        normalized = {normalize_header(v) for v in raw.iloc[idx].tolist()}
        if {"parameter", "subparameter", "maxscore"}.issubset(normalized):
            return idx
    return None


def parse_scoring_sheet(scoring_raw: pd.DataFrame | None) -> tuple[pd.DataFrame, list[ScoringRule], dict[str, Any]]:
    diagnostics: dict[str, Any] = {"used_fallback_scoring": False}
    if scoring_raw is None or scoring_raw.empty:
        scoring_df = pd.DataFrame(DEFAULT_SCORING_ROWS)
        diagnostics["used_fallback_scoring"] = True
        col_map = {"Parameter": "Parameter", "Sub Parameter": "Sub Parameter", "Max Score": "Max Score", "Scoring": "Scoring"}
    else:
        header_idx = find_scoring_header_row(scoring_raw)
        if header_idx is None:
            scoring_df = pd.DataFrame(DEFAULT_SCORING_ROWS)
            diagnostics["used_fallback_scoring"] = True
            col_map = {"Parameter": "Parameter", "Sub Parameter": "Sub Parameter", "Max Score": "Max Score", "Scoring": "Scoring"}
        else:
            headers = make_unique_headers(scoring_raw.iloc[header_idx].tolist())
            scoring_df = scoring_raw.iloc[header_idx + 1 :].copy()
            scoring_df.columns = headers
            col_map = {}
            for wanted in ["Parameter", "Sub Parameter", "Max Score", "Scoring"]:
                found = find_first_matching_header(list(scoring_df.columns), wanted, threshold=0.82)
                if found is not None:
                    col_map[wanted] = scoring_df.columns[found]
            if not {"Parameter", "Sub Parameter", "Max Score"}.issubset(col_map):
                scoring_df = pd.DataFrame(DEFAULT_SCORING_ROWS)
                diagnostics["used_fallback_scoring"] = True
                col_map = {"Parameter": "Parameter", "Sub Parameter": "Sub Parameter", "Max Score": "Max Score", "Scoring": "Scoring"}

    clean = pd.DataFrame(
        {
            "Parameter": scoring_df[col_map["Parameter"]],
            "Sub Parameter": scoring_df[col_map["Sub Parameter"]],
            "Max Score": pd.to_numeric(scoring_df[col_map["Max Score"]], errors="coerce"),
            "Scoring": scoring_df[col_map.get("Scoring", col_map["Sub Parameter"])].fillna(""),
        }
    )
    clean = clean.dropna(subset=["Sub Parameter", "Max Score"], how="any")
    clean = clean[clean["Sub Parameter"].astype(str).str.strip().ne("")]
    clean["Parameter"] = clean["Parameter"].astype(str).str.strip()
    clean["Sub Parameter"] = clean["Sub Parameter"].astype(str).str.strip()
    clean["Rule Key"] = clean["Sub Parameter"].map(normalize_header)

    rules: list[ScoringRule] = []
    for _, row in clean.iterrows():
        rules.append(
            ScoringRule(
                section=str(row["Parameter"]).strip(),
                parameter=str(row["Sub Parameter"]).strip(),
                max_score=float(row["Max Score"]),
                scoring_text=str(row.get("Scoring", "") or ""),
                mapping=parse_scoring_mapping(row.get("Scoring", "")),
                key=normalize_header(row["Sub Parameter"]),
            )
        )
    return clean.reset_index(drop=True), rules, diagnostics


def repair_headers_by_scoring_sequence(headers: list[Any], rules: list[ScoringRule], end_idx: int) -> tuple[list[str], list[dict[str, Any]]]:
    repaired = ["" if pd.isna(h) else str(h).strip() for h in headers[: end_idx + 1]]
    scoring_params = [r.parameter for r in rules]
    scoring_keys = [r.key for r in rules]
    inferred: list[dict[str, Any]] = []

    start_candidates = [find_first_matching_header(repaired, r.parameter, threshold=0.90) for r in rules[:5]]
    start_candidates = [x for x in start_candidates if x is not None]
    if not start_candidates:
        return make_unique_headers(repaired), inferred
    start = min(start_candidates)

    expected_pos = 0
    for col_idx in range(start, end_idx + 1):
        current = repaired[col_idx].strip()
        current_key = normalize_header(current)
        if current_key:
            matches = [i for i in range(expected_pos, len(scoring_keys)) if similarity(current_key, scoring_keys[i]) >= 0.88]
            if matches:
                expected_pos = matches[0] + 1
            continue
        if expected_pos < len(scoring_params):
            inferred_name = scoring_params[expected_pos]
            repaired[col_idx] = inferred_name
            inferred.append({"Column Number": col_idx + 1, "Inferred Header": inferred_name})
            expected_pos += 1
    return make_unique_headers(repaired), inferred


def _label_trailing_column(top_value: Any, header_value: Any, rules: list[ScoringRule], mode: str) -> str:
    header = "" if pd.isna(header_value) else str(header_value).strip()
    top = "" if pd.isna(top_value) else str(top_value).strip()
    if not header:
        header = top or "Workbook Column"

    h_norm = normalize_header(header)
    top_norm = normalize_header(top)
    rule_keys = {r.key for r in rules}
    section_keys = {normalize_header(r.section) for r in rules}

    if h_norm in {"ovaallscorewithfatal", "ovaallscorewithoutfatal", "defectpct"}:
        return f"Dataset Metric :: {header}"
    if mode == "parameter_compliance":
        return f"Dataset Parameter Compliance :: {header}"
    if mode == "defect":
        return f"Dataset Defect Flag :: {header}"
    if h_norm in section_keys or h_norm in {"disposition", "empowerment", "salescompliance", "salescompliance"}:
        return f"Dataset Section Score :: {header}"
    if h_norm in rule_keys or any(similarity(h_norm, r.key) >= 0.88 for r in rules):
        return f"Dataset Earned Score :: {header}"
    return f"Dataset Extra :: {header}"


def build_trailing_headers(header_values: list[Any], top_values: list[Any], end_idx: int, rules: list[ScoringRule]) -> list[str]:
    labels: list[str] = []
    mode = "earned"
    for abs_idx in range(end_idx + 1, len(header_values)):
        top = top_values[abs_idx] if abs_idx < len(top_values) else ""
        top_norm = normalize_header(top)
        if top_norm == "complianceagainstparameters":
            mode = "parameter_compliance"
        elif top_norm == "defect":
            mode = "defect"
        elif normalize_header(header_values[abs_idx]) in {"ovaallscorewithfatal", "ovaallscorewithoutfatal", "defectpct"}:
            mode = "metrics"
        labels.append(_label_trailing_column(top, header_values[abs_idx], rules, mode))
    return make_unique_headers(labels)


def prepare_audit_table(raw: pd.DataFrame, rules: list[ScoringRule]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if raw.empty:
        raise ValueError("Audit sheet is empty.")
    raw = raw.dropna(how="all")
    header_idx = detect_header_row(raw)
    header_values = raw.iloc[header_idx].tolist()
    top_values = raw.iloc[header_idx - 1].tolist() if header_idx > 0 else [""] * len(header_values)
    end_idx = find_first_matching_header(header_values, "Escalation Denied", threshold=0.86)
    if end_idx is None:
        raise ValueError("Could not find the raw-data end column: Escalation Denied.")

    headers, inferred = repair_headers_by_scoring_sequence(header_values, rules, end_idx)
    data = raw.iloc[header_idx + 1 :, : end_idx + 1].copy()
    data.columns = headers
    data = data.dropna(how="all")

    trailing = raw.iloc[header_idx + 1 :, end_idx + 1 :].copy()
    if trailing.shape[1] > 0:
        trailing.columns = build_trailing_headers(header_values, top_values, end_idx, rules)
        trailing = trailing.iloc[: len(data)].reset_index(drop=True)
    else:
        trailing = pd.DataFrame(index=range(len(data)))

    if "Agent Name" in data.columns:
        keep_mask = data["Agent Name"].notna()
        data = data[keep_mask].reset_index(drop=True)
        trailing = trailing[keep_mask.to_numpy()].reset_index(drop=True) if not trailing.empty else trailing

    diagnostics = {
        "detected_header_row_excel": header_idx + 1,
        "raw_data_end_column_excel": end_idx + 1,
        "raw_data_end_header": headers[end_idx],
        "inferred_headers": inferred,
        "raw_columns_used": headers,
        "existing_dataset_columns_detected": list(trailing.columns),
    }
    return data.reset_index(drop=True), trailing.reset_index(drop=True), diagnostics


def find_column(df: pd.DataFrame, wanted: str, threshold: float = 0.86) -> str | None:
    target = normalize_header(wanted)
    for col in df.columns:
        if normalize_header(col) == target:
            return col
    scored = [(col, similarity(normalize_header(col), target)) for col in df.columns]
    best_col, best_score = max(scored, key=lambda x: x[1], default=(None, 0.0))
    if best_col is not None and best_score >= threshold:
        return best_col
    return None


def coerce_date_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0 and numeric.dropna().between(25000, 70000).mean() > 0.5:
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def score_response(response: Any, rule: ScoringRule) -> tuple[float | None, bool, str, str]:
    value = normalize_response(response)
    if value in NA_VALUES:
        return None, False, "NA", value
    if value in FATAL_VALUES:
        return 0.0, True, "Fatal", value

    score: float | None = None
    if value in rule.mapping:
        score = rule.mapping[value]
    elif value in YES_VALUES:
        score = rule.max_score
    elif value in NO_VALUES:
        score = 0.0
    elif value == "EE":
        score = rule.mapping.get("EE", rule.max_score)
    elif value == "ME":
        score = rule.mapping.get("ME", round(rule.max_score / 2, 4))
    elif value == "BE":
        score = rule.mapping.get("BE", 0.0)
    else:
        try:
            numeric = float(value.replace("%", "")) if isinstance(value, str) else float(value)
            if 0 <= numeric <= 1:
                score = numeric * rule.max_score
            elif 0 <= numeric <= rule.max_score:
                score = numeric
        except Exception:
            score = None

    if score is None:
        return None, False, "Unknown", value

    score = max(0.0, min(float(score), rule.max_score))
    if np.isclose(score, rule.max_score):
        status = "Pass"
    elif np.isclose(score, 0.0):
        status = "Fail"
    else:
        status = "Partial"
    return score, True, status, value


def _find_prefixed_column(df: pd.DataFrame, prefix: str, target: str, threshold: float = 0.84) -> str | None:
    prefix_norm = normalize_header(prefix)
    target_norm = normalize_header(target)
    candidates: list[tuple[str, float]] = []
    for col in df.columns:
        if not str(col).startswith(prefix):
            continue
        suffix = str(col).split("::", 1)[-1].strip()
        candidates.append((col, similarity(normalize_header(suffix), target_norm)))
    if not candidates:
        return None
    best_col, best_score = max(candidates, key=lambda x: x[1])
    if best_score >= threshold:
        return best_col
    return None


def _numeric_or_nan(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip().replace("%", "")
        if text == "":
            return np.nan
        try:
            num = float(text)
            return num / 100 if "%" in value and num > 1 else num
        except Exception:
            return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _values_match(dataset_value: Any, computed_value: Any, tolerance: float) -> bool:
    dv = _numeric_or_nan(dataset_value)
    cv = _numeric_or_nan(computed_value)
    if pd.isna(dv) and pd.isna(cv):
        return True
    if pd.isna(dv) != pd.isna(cv):
        return False
    return abs(float(dv) - float(cv)) <= tolerance


def build_comparison_df(df: pd.DataFrame, long_df: pd.DataFrame, existing_calc_df: pd.DataFrame, rules: list[ScoringRule], tolerance: float = 0.005) -> pd.DataFrame:
    if existing_calc_df.empty:
        return pd.DataFrame()

    base_cols = [c for c in ["Record ID", "Audit Date", "Audit Month", "Quality Auditor", "Call/Chat", "Agent Name", "Supervisor name", "LOB", "Sub-LOB", "Reason for call"] if c in df.columns]
    meta = df[base_cols].set_index("Record ID") if "Record ID" in base_cols else pd.DataFrame()
    existing = existing_calc_df.set_index("Record ID") if "Record ID" in existing_calc_df.columns else existing_calc_df.copy()
    records: list[dict[str, Any]] = []

    def add_record(record_id: str, metric_type: str, metric: str, dataset_value: Any, computed_value: Any) -> None:
        row: dict[str, Any] = {"Record ID": record_id, "Metric Type": metric_type, "Metric": metric, "Dataset Value": dataset_value, "Computed Value": computed_value}
        for col in meta.columns:
            row[col] = meta.loc[record_id, col] if record_id in meta.index else np.nan
        dv = _numeric_or_nan(dataset_value)
        cv = _numeric_or_nan(computed_value)
        row["Difference"] = (dv - cv) if not (pd.isna(dv) or pd.isna(cv)) else np.nan
        row["Match"] = _values_match(dataset_value, computed_value, tolerance)
        records.append(row)

    metric_map = {
        "Ova_All_Score (With Fatal)": "Overall Score (With Fatal)",
        "Ova_All_Score (Without Fatal)": "Overall Score (Without Fatal)",
        "Defect%": "Defect %",
    }
    for dataset_label, computed_col in metric_map.items():
        existing_col = _find_prefixed_column(existing_calc_df, "Dataset Metric ::", dataset_label, threshold=0.80)
        if existing_col and computed_col in df.columns:
            for record_id, computed_value in df.set_index("Record ID")[computed_col].items():
                dataset_value = existing.loc[record_id, existing_col] if record_id in existing.index else np.nan
                add_record(record_id, "Overall metric", dataset_label, dataset_value, computed_value)

    for section_col in [c for c in df.columns if c.startswith("Section Score ::")]:
        section = section_col.split("::", 1)[1].strip()
        existing_col = _find_prefixed_column(existing_calc_df, "Dataset Section Score ::", section, threshold=0.80)
        if existing_col:
            for record_id, computed_value in df.set_index("Record ID")[section_col].items():
                dataset_value = existing.loc[record_id, existing_col] if record_id in existing.index else np.nan
                add_record(record_id, "Section score", section, dataset_value, computed_value)

    score_lookup = long_df.set_index(["Record ID", "Parameter"])
    for rule in rules:
        if rule.max_score <= 0:
            continue
        earned_col = _find_prefixed_column(existing_calc_df, "Dataset Earned Score ::", rule.parameter, threshold=0.78)
        compliance_col = _find_prefixed_column(existing_calc_df, "Dataset Parameter Compliance ::", rule.parameter, threshold=0.78)
        defect_col = _find_prefixed_column(existing_calc_df, "Dataset Defect Flag ::", rule.parameter, threshold=0.78)
        for record_id in df["Record ID"]:
            if (record_id, rule.parameter) not in score_lookup.index:
                continue
            row = score_lookup.loc[(record_id, rule.parameter)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if earned_col:
                add_record(record_id, "Parameter earned score", rule.parameter, existing.loc[record_id, earned_col] if record_id in existing.index else np.nan, row.get("Score"))
            if compliance_col:
                score_pct = row.get("Score %")
                add_record(record_id, "Parameter score ratio", rule.parameter, existing.loc[record_id, compliance_col] if record_id in existing.index else np.nan, score_pct)
            if defect_col:
                computed_defect = 1.0 if bool(row.get("Applicable")) and str(row.get("Status")) in {"Fail", "Partial", "Fatal"} else np.nan
                add_record(record_id, "Parameter defect flag", rule.parameter, existing.loc[record_id, defect_col] if record_id in existing.index else np.nan, computed_defect)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values(["Match", "Metric Type", "Metric", "Record ID"], ascending=[True, True, True, True]).reset_index(drop=True)


def build_data_view(df: pd.DataFrame, audit_df: pd.DataFrame, long_df: pd.DataFrame, rules: list[ScoringRule], rule_to_column: dict[str, str | None]) -> pd.DataFrame:
    meta_cols = [c for c in META_COLUMNS_CANONICAL + ["Record ID"] if c in df.columns]
    computed_cols = [
        "Applicable Max Score",
        "Earned Score",
        "Applicable Parameter Count",
        "Failed Parameter Count",
        "Fatal Count",
        "Has Fatal",
        "Overall Score (With Fatal)",
        "Overall Score (Without Fatal)",
        "Defect %",
    ]
    computed_cols += [c for c in df.columns if c.startswith("Section Score ::") or c.startswith("Section Earned ::") or c.startswith("Section Max ::")]
    view = df[meta_cols + [c for c in computed_cols if c in df.columns]].copy()

    entered = audit_df[["Record ID"]].copy()
    used_cols: set[str] = set()
    for rule in rules:
        col = rule_to_column.get(rule.parameter)
        if col and col in audit_df.columns and col not in used_cols:
            entered[f"Entered :: {rule.parameter}"] = audit_df[col]
            used_cols.add(col)
    view = view.merge(entered, on="Record ID", how="left")

    if not long_df.empty:
        score_wide = long_df.pivot(index="Record ID", columns="Parameter", values="Score")
        score_wide.columns = [f"Computed Score :: {c}" for c in score_wide.columns]
        status_wide = long_df.pivot(index="Record ID", columns="Parameter", values="Status")
        status_wide.columns = [f"Computed Status :: {c}" for c in status_wide.columns]
        applicable_wide = long_df.pivot(index="Record ID", columns="Parameter", values="Applicable")
        applicable_wide.columns = [f"Computed Applicable :: {c}" for c in applicable_wide.columns]
        view = view.merge(score_wide.reset_index(), on="Record ID", how="left")
        view = view.merge(status_wide.reset_index(), on="Record ID", how="left")
        view = view.merge(applicable_wide.reset_index(), on="Record ID", how="left")
    return view


def build_scored_dataset_from_frames(audit_raw: pd.DataFrame, scoring_raw: pd.DataFrame | None = None, source_name: str = "Workbook") -> ScoredDataset:
    scoring_df, rules, scoring_diag = parse_scoring_sheet(scoring_raw)
    audit_df, existing_calc_df, audit_diag = prepare_audit_table(audit_raw, rules)
    diagnostics: dict[str, Any] = {"source_name": source_name, **scoring_diag, **audit_diag}

    rename_map: dict[str, str] = {}
    for canonical in META_COLUMNS_CANONICAL:
        found = find_column(audit_df, canonical, threshold=0.82)
        if found:
            rename_map[found] = canonical
    audit_df = audit_df.rename(columns=rename_map)
    for col in META_COLUMNS_CANONICAL:
        if col not in audit_df.columns:
            audit_df[col] = np.nan

    rule_to_column: dict[str, str | None] = {}
    unmatched: list[str] = []
    for rule in rules:
        found = find_column(audit_df, rule.parameter, threshold=0.84)
        if found is None:
            unmatched.append(rule.parameter)
        rule_to_column[rule.parameter] = found
    diagnostics["unmatched_scoring_parameters"] = unmatched

    for date_col in ["Call Date", "Audit Date"]:
        audit_df[date_col] = coerce_date_series(audit_df[date_col])
    if audit_df["Audit Month"].isna().all():
        audit_df["Audit Month"] = audit_df["Audit Date"].dt.strftime("%b")
    audit_df["Record ID"] = [f"QA-{i + 1:05d}" for i in range(len(audit_df))]
    if not existing_calc_df.empty:
        existing_calc_df = existing_calc_df.iloc[: len(audit_df)].reset_index(drop=True)
        existing_calc_df.insert(0, "Record ID", audit_df["Record ID"].values)

    master_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    compliance_rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []

    scored_rules = [r for r in rules if r.max_score > 0]
    compliance_rules = [r for r in rules if r.max_score <= 0]
    sections = list(dict.fromkeys([r.section for r in scored_rules]))
    response_columns = [c for c in audit_df.columns if c not in META_COLUMNS_CANONICAL and c != "Record ID"]

    for _, raw_row in audit_df.iterrows():
        record_id = raw_row["Record ID"]
        base = {col: raw_row.get(col, np.nan) for col in META_COLUMNS_CANONICAL}
        base["Record ID"] = record_id

        scores: list[float] = []
        maxes: list[float] = []
        applicable_count = 0
        failed_count = 0
        fatal_count = 0
        section_scores: dict[str, list[float]] = {s: [] for s in sections}
        section_maxes: dict[str, list[float]] = {s: [] for s in sections}
        fatal_anywhere = any(normalize_response(raw_row.get(c)) in FATAL_VALUES for c in response_columns)

        for rule in scored_rules:
            col = rule_to_column.get(rule.parameter)
            raw_value = raw_row.get(col) if col else np.nan
            score, applicable, status, norm_value = score_response(raw_value, rule)
            is_fatal = norm_value in FATAL_VALUES
            if applicable:
                applicable_count += 1
                scores.append(float(score or 0.0))
                maxes.append(rule.max_score)
                section_scores[rule.section].append(float(score or 0.0))
                section_maxes[rule.section].append(rule.max_score)
                if status in {"Fail", "Partial", "Fatal"} or float(score or 0.0) < rule.max_score:
                    failed_count += 1
                if is_fatal:
                    fatal_count += 1
            elif status == "Unknown" and str(raw_value).strip() != "":
                unknown_rows.append({"Record ID": record_id, "Agent Name": raw_row.get("Agent Name"), "Quality Auditor": raw_row.get("Quality Auditor"), "Parameter": rule.parameter, "Raw Response": raw_value})

            long_rows.append(
                {
                    "Record ID": record_id,
                    "Call Date": raw_row.get("Call Date"),
                    "Audit Date": raw_row.get("Audit Date"),
                    "Audit Month": raw_row.get("Audit Month"),
                    "Agent Name": raw_row.get("Agent Name"),
                    "Quality Auditor": raw_row.get("Quality Auditor"),
                    "Call/Chat": raw_row.get("Call/Chat"),
                    "Supervisor name": raw_row.get("Supervisor name"),
                    "LOB": raw_row.get("LOB"),
                    "Sub-LOB": raw_row.get("Sub-LOB"),
                    "Reason for call": raw_row.get("Reason for call"),
                    "Section": rule.section,
                    "Parameter": rule.parameter,
                    "Matched Column": col,
                    "Raw Response": raw_value,
                    "Normalized Response": norm_value,
                    "Max Score": rule.max_score,
                    "Score": score,
                    "Applicable": applicable,
                    "Status": status,
                    "Fatal": is_fatal,
                    "Score %": (score / rule.max_score) if applicable and rule.max_score else np.nan,
                }
            )

        total_score = float(np.nansum(scores)) if scores else np.nan
        total_max = float(np.nansum(maxes)) if maxes else np.nan
        without_fatal = total_score / total_max if total_max and total_max > 0 else np.nan
        with_fatal = 0.0 if fatal_anywhere else without_fatal
        defect_pct = failed_count / applicable_count if applicable_count else np.nan

        base.update(
            {
                "Applicable Max Score": total_max,
                "Earned Score": total_score,
                "Applicable Parameter Count": applicable_count,
                "Failed Parameter Count": failed_count,
                "Fatal Count": fatal_count,
                "Has Fatal": bool(fatal_anywhere),
                "Overall Score (Without Fatal)": without_fatal,
                "Overall Score (With Fatal)": with_fatal,
                "Defect %": defect_pct,
            }
        )
        for section in sections:
            sec_max = float(np.nansum(section_maxes[section])) if section_maxes[section] else np.nan
            sec_score = float(np.nansum(section_scores[section])) if section_scores[section] else np.nan
            base[f"Section Score :: {section}"] = sec_score / sec_max if sec_max and sec_max > 0 else np.nan
            base[f"Section Max :: {section}"] = sec_max
            base[f"Section Earned :: {section}"] = sec_score

        for rule in compliance_rules:
            col = rule_to_column.get(rule.parameter)
            raw_value = raw_row.get(col) if col else np.nan
            norm = normalize_response(raw_value)
            if norm not in NA_VALUES:
                compliance_rows.append(
                    {
                        "Record ID": record_id,
                        "Call Date": raw_row.get("Call Date"),
                        "Audit Date": raw_row.get("Audit Date"),
                        "Audit Month": raw_row.get("Audit Month"),
                        "Agent Name": raw_row.get("Agent Name"),
                        "Quality Auditor": raw_row.get("Quality Auditor"),
                        "Call/Chat": raw_row.get("Call/Chat"),
                        "Supervisor name": raw_row.get("Supervisor name"),
                        "LOB": raw_row.get("LOB"),
                        "Sub-LOB": raw_row.get("Sub-LOB"),
                        "Reason for call": raw_row.get("Reason for call"),
                        "Flag": rule.parameter,
                        "Raw Response": raw_value,
                        "Positive": norm in YES_VALUES or norm in FATAL_VALUES,
                    }
                )
        master_rows.append(base)

    df = pd.DataFrame(master_rows)
    long_df = pd.DataFrame(long_rows)
    compliance_df = pd.DataFrame(compliance_rows)
    data_view = build_data_view(df, audit_df, long_df, rules, rule_to_column)
    comparison_df = build_comparison_df(df, long_df, existing_calc_df, rules)

    diagnostics["unknown_scoring_values"] = unknown_rows
    diagnostics["records_loaded"] = len(df)
    diagnostics["scored_parameters"] = len(scored_rules)
    diagnostics["compliance_flags"] = len(compliance_rules)
    diagnostics["comparison_rows"] = len(comparison_df)
    diagnostics["comparison_mismatches"] = int((~comparison_df["Match"]).sum()) if not comparison_df.empty and "Match" in comparison_df else 0
    return ScoredDataset(df=df, long_df=long_df, compliance_df=compliance_df, scoring_df=scoring_df, diagnostics=diagnostics, data_view=data_view, comparison_df=comparison_df, existing_calc_df=existing_calc_df)


def pick_sheet_name(excel: pd.ExcelFile, keyword: str, default_index: int = 0) -> str:
    keyword_norm = normalize_header(keyword)
    for name in excel.sheet_names:
        if keyword_norm in normalize_header(name):
            return name
    return excel.sheet_names[default_index]


def load_workbook_dataset(file_obj: Any, filename: str | None = None) -> ScoredDataset:
    excel = pd.ExcelFile(file_obj)
    audit_sheet = pick_sheet_name(excel, "Audit", default_index=0)
    scoring_sheet = None
    for name in excel.sheet_names:
        if "scoring" in normalize_header(name):
            scoring_sheet = name
            break
    audit_raw = pd.read_excel(excel, sheet_name=audit_sheet, header=None, dtype=object)
    scoring_raw = pd.read_excel(excel, sheet_name=scoring_sheet, header=None, dtype=object) if scoring_sheet else None
    dataset = build_scored_dataset_from_frames(audit_raw, scoring_raw, source_name=filename or str(file_obj))
    dataset.diagnostics["audit_sheet_name"] = audit_sheet
    dataset.diagnostics["scoring_sheet_name"] = scoring_sheet or "Fallback scoring"
    return dataset


def ragged_csv_to_frame(text: str) -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    normalized = [r + [None] * (width - len(r)) for r in rows]
    return pd.DataFrame(normalized, dtype=object)


def load_csv_dataset(file_obj: Any, filename: str | None = None) -> ScoredDataset:
    text = file_obj.read().decode("utf-8-sig", errors="replace") if hasattr(file_obj, "read") else str(file_obj)
    raw = ragged_csv_to_frame(text)
    return build_scored_dataset_from_frames(raw, None, source_name=filename or "CSV")


def load_dataset_from_upload(uploaded_file: Any) -> ScoredDataset:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".csv"):
        return load_csv_dataset(io.BytesIO(data), uploaded_file.name)
    if name.endswith(".xlsx") or name.endswith(".xlsm"):
        return load_workbook_dataset(io.BytesIO(data), uploaded_file.name)
    raise ValueError("Unsupported file type. Upload .xlsx, .xlsm, or .csv.")


def display_percent(value: Any, decimals: int = 1) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value) * 100:.{decimals}f}%"
    except Exception:
        return "-"


def display_number(value: Any, decimals: int = 0) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "-"


def group_summary(df: pd.DataFrame, group_col: str, score_col: str, min_audits: int = 1) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            Audits=("Record ID", "count"),
            Avg_Score=(score_col, "mean"),
            Avg_Without_Fatal=("Overall Score (Without Fatal)", "mean"),
            Avg_With_Fatal=("Overall Score (With Fatal)", "mean"),
            Avg_Defect=("Defect %", "mean"),
            Fatal_Rate=("Has Fatal", "mean"),
            Avg_Failed_Parameters=("Failed Parameter Count", "mean"),
            Avg_Applicable_Parameters=("Applicable Parameter Count", "mean"),
        )
        .reset_index()
    )
    grouped[group_col] = grouped[group_col].fillna("Unknown")
    grouped = grouped[grouped["Audits"] >= min_audits]
    return grouped.sort_values(["Avg_Score", "Audits"], ascending=[False, False])


def parameter_failure_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    applicable = long_df[long_df["Applicable"]].copy()
    if applicable.empty:
        return pd.DataFrame()
    applicable["Is Issue"] = applicable["Status"].isin(["Fail", "Partial", "Fatal"])
    out = (
        applicable.groupby(["Section", "Parameter"], dropna=False)
        .agg(
            Applicable_Audits=("Record ID", "count"),
            Issue_Audits=("Is Issue", "sum"),
            Failure_Rate=("Is Issue", "mean"),
            Avg_Score_Pct=("Score %", "mean"),
            Fatal_Count=("Fatal", "sum"),
        )
        .reset_index()
        .sort_values(["Failure_Rate", "Issue_Audits"], ascending=[False, False])
    )
    return out
