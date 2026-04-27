from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlparse

import requests
import streamlit as st

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_SHEETS_API_BASE = "https://sheets.googleapis.com/v4"
GOOGLE_SCOPE_SHEETS_READONLY = "https://www.googleapis.com/auth/spreadsheets.readonly"


class GoogleOAuthConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class GoogleSheetReference:
    spreadsheet_id: str
    gid: str | None = None
    original_value: str | None = None



def get_google_oauth_config() -> dict[str, str]:
    if "google_oauth" not in st.secrets:
        raise GoogleOAuthConfigError(
            "Missing [google_oauth] in Streamlit secrets. Add client_id, client_secret, and redirect_uri."
        )

    cfg = st.secrets["google_oauth"]
    required = ["client_id", "client_secret", "redirect_uri"]
    missing = [key for key in required if key not in cfg or not str(cfg[key]).strip()]
    if missing:
        raise GoogleOAuthConfigError(
            f"Missing Google OAuth secret values: {', '.join(missing)}"
        )

    return {
        "client_id": str(cfg["client_id"]),
        "client_secret": str(cfg["client_secret"]),
        "redirect_uri": str(cfg["redirect_uri"]),
        "scope": str(cfg.get("scope", GOOGLE_SCOPE_SHEETS_READONLY)),
    }



def extract_sheet_reference(value: str) -> GoogleSheetReference:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Enter a Google Sheets link or a spreadsheet ID.")

    if "/spreadsheets/d/" in raw:
        parsed = urlparse(raw)
        path_parts = [part for part in parsed.path.split("/") if part]
        spreadsheet_id = None
        for idx, part in enumerate(path_parts):
            if part == "d" and idx + 1 < len(path_parts):
                spreadsheet_id = path_parts[idx + 1]
                break
        if not spreadsheet_id:
            raise ValueError("Could not extract a spreadsheet ID from the Google Sheets URL.")
        fragment_params = parse_qs(parsed.fragment)
        gid = fragment_params.get("gid", [None])[0]
        return GoogleSheetReference(spreadsheet_id=spreadsheet_id, gid=gid, original_value=raw)

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if all(ch in allowed for ch in raw):
        return GoogleSheetReference(spreadsheet_id=raw, gid=None, original_value=raw)

    raise ValueError("That does not look like a valid Google Sheets link or spreadsheet ID.")



def _store_pending_sheet(ref: GoogleSheetReference) -> None:
    st.session_state["google_pending_sheet_id"] = ref.spreadsheet_id
    st.session_state["google_pending_sheet_gid"] = ref.gid
    st.session_state["google_pending_sheet_value"] = ref.original_value



def build_google_auth_url(sheet_ref: GoogleSheetReference) -> str:
    cfg = get_google_oauth_config()
    state = secrets.token_urlsafe(24)
    st.session_state["google_oauth_state"] = state
    _store_pending_sheet(sheet_ref)

    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "response_type": "code",
        "scope": cfg["scope"],
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"



def _store_token(token_payload: dict[str, Any]) -> dict[str, Any]:
    now = int(time.time())
    expires_in = int(token_payload.get("expires_in", 3600))
    stored = {
        "access_token": token_payload["access_token"],
        "refresh_token": token_payload.get("refresh_token", st.session_state.get("google_token", {}).get("refresh_token")),
        "token_type": token_payload.get("token_type", "Bearer"),
        "scope": token_payload.get("scope", GOOGLE_SCOPE_SHEETS_READONLY),
        "expires_at": now + max(expires_in - 60, 60),
    }
    st.session_state["google_token"] = stored
    return stored



def exchange_code_for_token(code: str) -> dict[str, Any]:
    cfg = get_google_oauth_config()
    response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "redirect_uri": cfg["redirect_uri"],
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    response.raise_for_status()
    return _store_token(response.json())



def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    cfg = get_google_oauth_config()
    response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    response.raise_for_status()
    return _store_token(response.json())



def get_valid_google_token() -> dict[str, Any] | None:
    token = st.session_state.get("google_token")
    if not token:
        return None

    now = int(time.time())
    if token.get("expires_at", 0) > now and token.get("access_token"):
        return token

    refresh_token = token.get("refresh_token")
    if not refresh_token:
        return None

    return refresh_access_token(refresh_token)



def disconnect_google_token() -> None:
    for key in [
        "google_token",
        "google_oauth_state",
        "google_pending_sheet_id",
        "google_pending_sheet_gid",
        "google_pending_sheet_value",
        "google_selected_sheet_title",
    ]:
        st.session_state.pop(key, None)



def consume_google_oauth_callback() -> str | None:
    query_params = st.query_params
    if not query_params:
        return None

    if "error" in query_params:
        error_value = query_params.get("error")
        error_text = error_value[0] if isinstance(error_value, list) else error_value
        st.query_params.clear()
        raise RuntimeError(f"Google authorization failed: {error_text}")

    code = query_params.get("code")
    state = query_params.get("state")
    if isinstance(code, list):
        code = code[0]
    if isinstance(state, list):
        state = state[0]

    if not code:
        return None

    expected_state = st.session_state.get("google_oauth_state")
    if expected_state and state != expected_state:
        st.query_params.clear()
        raise RuntimeError("Google authorization state mismatch. Please try again.")

    exchange_code_for_token(code)
    st.query_params.clear()
    return st.session_state.get("google_pending_sheet_value")



def _authorized_get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    token = get_valid_google_token()
    if not token:
        raise RuntimeError("Google authorization is required before reading Sheets.")

    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token['access_token']}"},
        params=params,
        timeout=30,
    )

    if response.status_code == 401 and token.get("refresh_token"):
        refreshed = refresh_access_token(token["refresh_token"])
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {refreshed['access_token']}"},
            params=params,
            timeout=30,
        )

    response.raise_for_status()
    return response.json()



def get_spreadsheet_metadata(spreadsheet_id: str) -> dict[str, Any]:
    return _authorized_get(
        f"{GOOGLE_SHEETS_API_BASE}/spreadsheets/{spreadsheet_id}",
        params={
            "fields": "properties.title,sheets.properties(sheetId,title,index)",
        },
    )



def resolve_sheet_title(metadata: dict[str, Any], gid: str | None = None, preferred_title: str = "Audit sheet") -> str:
    sheets = metadata.get("sheets", [])
    if not sheets:
        raise RuntimeError("This spreadsheet does not contain any visible sheets.")

    if gid is not None:
        for sheet in sheets:
            props = sheet.get("properties", {})
            if str(props.get("sheetId")) == str(gid):
                return str(props.get("title"))

    for sheet in sheets:
        title = str(sheet.get("properties", {}).get("title", ""))
        if title == preferred_title:
            return title

    return str(sheets[0].get("properties", {}).get("title", "Sheet1"))



def list_sheet_titles(metadata: dict[str, Any]) -> list[str]:
    return [str(sheet.get("properties", {}).get("title", "")) for sheet in metadata.get("sheets", []) if sheet.get("properties", {}).get("title")]



def get_sheet_rows(spreadsheet_id: str, sheet_title: str, data_range: str = "A3:AO") -> list[list[Any]]:
    range_a1 = f"'{sheet_title}'!{data_range}"
    encoded_range = quote(range_a1, safe="")
    payload = _authorized_get(
        f"{GOOGLE_SHEETS_API_BASE}/spreadsheets/{spreadsheet_id}/values/{encoded_range}",
        params={
            "majorDimension": "ROWS",
            "valueRenderOption": "FORMATTED_VALUE",
        },
    )
    return payload.get("values", [])
