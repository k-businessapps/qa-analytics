from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any
import json
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import pandas as pd
import requests
import streamlit as st
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from scoring_engine import build_scored_dataset_from_frames, normalize_header


DEFAULT_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


@dataclass(frozen=True)
class SheetRef:
    spreadsheet_id: str
    gid: str | None = None


def extract_sheet_ref(url_or_id: str) -> SheetRef:
    value = url_or_id.strip()
    if not value:
        raise ValueError("Paste a Google Sheets URL or spreadsheet ID.")
    if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", value):
        return SheetRef(spreadsheet_id=value, gid=None)
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", value)
    if not match:
        raise ValueError("Could not read a Google spreadsheet ID from that link.")
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    gid = None
    if "gid" in query and query["gid"]:
        gid = query["gid"][0]
    else:
        fragment_qs = parse_qs(parsed.fragment)
        if "gid" in fragment_qs and fragment_qs["gid"]:
            gid = fragment_qs["gid"][0]
        else:
            gid_match = re.search(r"gid=([0-9]+)", value)
            gid = gid_match.group(1) if gid_match else None
    return SheetRef(spreadsheet_id=match.group(1), gid=gid)


def build_published_csv_url(url: str) -> str:
    parsed = urlparse(url.strip())
    query = parse_qs(parsed.query)
    gid = None
    if "gid" in query and query["gid"]:
        gid = query["gid"][0]
    else:
        gid_match = re.search(r"gid=([0-9]+)", url)
        gid = gid_match.group(1) if gid_match else None

    if "/pub" in parsed.path:
        new_query = query.copy()
        new_query["output"] = ["csv"]
        if gid:
            new_query["gid"] = [gid]
        query_string = urlencode({k: v[0] for k, v in new_query.items()})
        return urlunparse(parsed._replace(query=query_string, fragment=""))

    ref = extract_sheet_ref(url)
    params = {"format": "csv", "id": ref.spreadsheet_id}
    if gid or ref.gid:
        params["gid"] = gid or ref.gid or "0"
    return f"https://docs.google.com/spreadsheets/d/{ref.spreadsheet_id}/export?{urlencode(params)}"


def load_published_google_sheet(url: str):
    """Load a public/published Google Sheet without OAuth.

    For pubhtml links, this first tries to parse all available HTML tables. If a
    Scoring table is visible, it uses that. Otherwise it falls back to the built-in
    sample scoring rules.
    """
    scoring_raw = None
    audit_raw = None
    source_note = "Published Google Sheet"

    if "pubhtml" in url:
        try:
            tables = pd.read_html(url, header=None)
            for table in tables:
                sample_values = {normalize_header(x) for x in table.head(5).astype(str).values.flatten().tolist()}
                if {"parameter", "subparameter", "maxscore"}.issubset(sample_values):
                    scoring_raw = table
                elif {"qualityauditor", "agentname"}.intersection(sample_values):
                    audit_raw = table if audit_raw is None else audit_raw
        except Exception:
            audit_raw = None

    if audit_raw is None:
        csv_url = build_published_csv_url(url)
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()
        audit_raw = pd.read_csv(io.StringIO(response.text), header=None, dtype=object)
        source_note = "Published Google Sheet CSV"

    return build_scored_dataset_from_frames(audit_raw, scoring_raw, source_name=source_note)


def _get_secret_value(section: str, key: str, default: Any = None) -> Any:
    try:
        block = st.secrets.get(section, {})
        if hasattr(block, "get"):
            return block.get(key, default)
    except Exception:
        pass
    return default


def get_oauth_config() -> dict[str, Any]:
    client_id = _get_secret_value("google_oauth", "client_id")
    client_secret = _get_secret_value("google_oauth", "client_secret")
    redirect_uri = _get_secret_value("google_oauth", "redirect_uri")
    scopes = _get_secret_value("google_sheets", "scopes", DEFAULT_SCOPES)
    if not client_id or not client_secret or not redirect_uri:
        raise RuntimeError(
            "Google OAuth is not configured. Add google_oauth.client_id, google_oauth.client_secret, and google_oauth.redirect_uri to Streamlit secrets."
        )
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "scopes": list(scopes),
    }


def get_auth_url(sheet_url: str) -> str:
    """Build a Google OAuth URL without PKCE.

    Streamlit Cloud reruns the script after the Google redirect. If the OAuth
    helper generates a PKCE code_challenge during the first run, the matching
    code_verifier can be lost by the time Google redirects back. That produces:
    `(invalid_grant) Missing code verifier`.

    This app uses a confidential web-client OAuth flow instead. The token
    exchange is authenticated with the client_secret from Streamlit secrets, so
    no PKCE verifier has to survive the redirect.
    """
    cfg = get_oauth_config()
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "response_type": "code",
        "scope": " ".join(cfg["scopes"]),
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "state": sheet_url,
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


def _create_credentials_from_token_payload(payload: dict[str, Any]) -> Credentials:
    cfg = get_oauth_config()
    return Credentials(
        token=payload.get("access_token"),
        refresh_token=payload.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cfg["client_id"],
        client_secret=cfg["client_secret"],
        scopes=cfg["scopes"],
    )


def handle_oauth_callback() -> None:
    params = st.query_params

    if "error" in params:
        error = params.get("error")
        if isinstance(error, list):
            error = error[0]
        st.query_params.clear()
        raise RuntimeError(f"Google OAuth failed: {error}")

    if "code" not in params:
        return

    code = params.get("code")
    state = params.get("state", "")
    if isinstance(code, list):
        code = code[0]
    if isinstance(state, list):
        state = state[0]

    cfg = get_oauth_config()
    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "redirect_uri": cfg["redirect_uri"],
            "grant_type": "authorization_code",
        },
        timeout=30,
    )

    if not response.ok:
        try:
            details = response.json()
        except Exception:
            details = response.text
        st.query_params.clear()
        raise RuntimeError(f"Google token exchange failed: {details}")

    creds = _create_credentials_from_token_payload(response.json())
    st.session_state["google_credentials"] = creds.to_json()
    st.session_state["data_source_mode"] = "Google OAuth"
    st.session_state["oauth_callback_completed"] = True
    if state:
        st.session_state["pending_google_sheet_url"] = state
        st.session_state["oauth_sheet_url"] = state
    st.query_params.clear()
    st.rerun()


def get_credentials() -> Credentials | None:
    raw = st.session_state.get("google_credentials")
    if not raw:
        return None
    cfg = get_oauth_config()
    creds = Credentials.from_authorized_user_info(
        raw if isinstance(raw, dict) else json.loads(raw),
        cfg["scopes"],
    )
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
        st.session_state["google_credentials"] = creds.to_json()
    if creds and creds.valid:
        return creds
    return None

def disconnect_google() -> None:
    st.session_state.pop("google_credentials", None)
    st.session_state.pop("oauth_loaded_dataset", None)


def sheets_service():
    creds = get_credentials()
    if creds is None:
        return None
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def list_worksheets(spreadsheet_id: str) -> list[dict[str, Any]]:
    service = sheets_service()
    if service is None:
        return []
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    out: list[dict[str, Any]] = []
    for sheet in meta.get("sheets", []):
        props = sheet.get("properties", {})
        out.append({"title": props.get("title"), "gid": str(props.get("sheetId"))})
    return out


def get_values(spreadsheet_id: str, sheet_title: str) -> list[list[Any]]:
    service = sheets_service()
    if service is None:
        return []
    escaped = sheet_title.replace("'", "''")
    result = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=f"'{escaped}'").execute()
    return result.get("values", [])


def values_to_frame(values: list[list[Any]]) -> pd.DataFrame:
    if not values:
        return pd.DataFrame()
    width = max(len(r) for r in values)
    normalized = [r + [None] * (width - len(r)) for r in values]
    return pd.DataFrame(normalized, dtype=object)


def load_google_oauth_dataset(spreadsheet_id: str, audit_sheet_title: str):
    worksheets = list_worksheets(spreadsheet_id)
    scoring_title = None
    for sheet in worksheets:
        title = sheet.get("title") or ""
        if "scoring" in normalize_header(title):
            scoring_title = title
            break
    audit_raw = values_to_frame(get_values(spreadsheet_id, audit_sheet_title))
    scoring_raw = values_to_frame(get_values(spreadsheet_id, scoring_title)) if scoring_title else None
    dataset = build_scored_dataset_from_frames(audit_raw, scoring_raw, source_name=f"Google OAuth · {audit_sheet_title}")
    dataset.diagnostics["audit_sheet_name"] = audit_sheet_title
    dataset.diagnostics["scoring_sheet_name"] = scoring_title or "Fallback scoring"
    return dataset
