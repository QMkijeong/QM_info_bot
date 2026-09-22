# -*- coding: utf-8 -*-
"""
판정 케이스 로깅 — Google 스프레드시트에 자동 적재.

목적: MD가 "이 판정이 왜 나왔는지" 나중에 리뷰하고, 오탐 사례를 모아서
      category_rules.py / gemini_service.py 프롬프트를 계속 보강하기 위함.

설정 방법 (최초 1회):
  1. Google Cloud Console에서 서비스 계정 생성 + Google Sheets API 활성화
  2. 서비스 계정 JSON 키 발급
  3. 로그를 쌓을 Google 스프레드시트를 만들고, 그 서비스 계정 이메일을
     "편집자"로 공유
  4. Streamlit Cloud Secrets에 아래처럼 등록:

     [gcp_service_account]
     type = "service_account"
     project_id = "..."
     private_key_id = "..."
     private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
     client_email = "...@....iam.gserviceaccount.com"
     ... (JSON 키 파일의 나머지 필드 그대로)

     SHEET_URL = "https://docs.google.com/spreadsheets/d/xxxx/edit"

설정 전에는 로깅이 조용히 스킵된다 (앱이 죽지 않도록).
"""

import datetime
import json
import uuid

_HEADER = [
    "timestamp", "row_type", "case_id", "product_name",
    "category_id", "category_name", "group3_confidence",
    "trail", "feedback_text", "raw_signals_json",
]


def _get_worksheet():
    try:
        import streamlit as st
        import gspread
        from google.oauth2.service_account import Credentials

        if "gcp_service_account" not in st.secrets or "SHEET_URL" not in st.secrets:
            return None

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=scopes
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(st.secrets["SHEET_URL"])
        ws = sh.sheet1

        # 헤더가 비어있으면 한 번 세팅
        first_row = ws.row_values(1)
        if first_row != _HEADER:
            ws.update("A1", [_HEADER])

        return ws
    except Exception:
        # 로깅 설정이 안 됐거나 일시적으로 실패해도, 앱의 본 기능(카테고리 판정)은
        # 절대 막히면 안 되므로 조용히 넘어간다.
        return None


def log_case(*, case_id: str, product_name: str, category_id: str,
             category_name: str, group3_confidence: str, trail: list,
             signals: dict) -> None:
    ws = _get_worksheet()
    if ws is None:
        return
    try:
        ws.append_row([
            datetime.datetime.now().isoformat(timespec="seconds"),
            "case",
            case_id,
            product_name or "",
            category_id,
            category_name,
            group3_confidence or "",
            " | ".join(trail),
            "",
            json.dumps(signals, ensure_ascii=False),
        ])
    except Exception:
        pass


def log_feedback(*, case_id: str, feedback_text: str) -> None:
    ws = _get_worksheet()
    if ws is None:
        return
    try:
        ws.append_row([
            datetime.datetime.now().isoformat(timespec="seconds"),
            "feedback",
            case_id,
            "", "", "", "",
            "",
            feedback_text,
            "",
        ])
    except Exception:
        pass


def new_case_id() -> str:
    return uuid.uuid4().hex[:10]


def logging_configured() -> bool:
    return _get_worksheet() is not None
