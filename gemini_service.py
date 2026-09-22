# -*- coding: utf-8 -*-
"""
Gemini 호출 레이어.

설계 원칙 (오탐률을 낮추기 위한 핵심):
  1) Gemini는 "추출/판독"만 한다. 카테고리 확정 판단은 절대 여기서 하지 않는다.
     (판단은 classifier.py의 규칙 엔진이 전담)
  2) 라벨에 실제로 적혀있지 않은 값은 절대 추론해서 채우지 않는다.
     -> 모든 필드에 "visible" 플래그를 강제하고, 없으면 null.
  3) 이미지 품질이 낮으면 추출 단계로 아예 넘어가지 않는다 (품질 게이트 우선).
  4) temperature=0 으로 고정 (창의성 불필요, 일관된 판독이 목적).
"""

import json
import os
from typing import List

import google.generativeai as genai
from PIL import Image

from category_rules import GROUP3

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# 3군(일반 재화) 후보 이름 목록 — Gemini가 이 목록 중에서만 고르도록 enum으로 강제한다.
# "미상" 은 Gemini 스스로 확신이 없을 때 쓰는 탈출구 (지어내지 말라는 뜻).
_GROUP3_NAMES = [c.name for c in GROUP3] + ["미상"]

_GENERATION_CONFIG = {
    "temperature": 0,
    "response_mime_type": "application/json",
}


def _get_api_key() -> str:
    """
    키 조회 우선순위:
      1) Streamlit Cloud의 Secrets (st.secrets["GOOGLE_API_KEY"]) — 배포 환경 기본값
      2) 로컬 환경변수 (GOOGLE_API_KEY) — 로컬 개발/테스트용
    회사 공식 키 하나를 Streamlit Cloud Secrets에 등록해두면, MD는 아무 설정 없이
    링크만 열어도 이 키로 동작한다.
    """
    try:
        import streamlit as st
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass  # Streamlit 밖(예: 단독 스크립트/테스트)에서 호출된 경우 무시하고 아래로

    return os.environ.get("GOOGLE_API_KEY", "")


def _get_model(response_schema: dict = None):
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY 를 찾을 수 없습니다. "
            "배포 환경(Streamlit Cloud)이면 App settings → Secrets 에 등록하고, "
            "로컬이면 환경변수로 설정하세요. "
            "(AI Studio: https://aistudio.google.com/apikey)"
        )
    genai.configure(api_key=api_key)
    cfg = dict(_GENERATION_CONFIG)
    if response_schema is not None:
        cfg["response_schema"] = response_schema
    return genai.GenerativeModel(MODEL_NAME, generation_config=cfg)


# ---------------------------------------------------------------------------
# 1단계: 이미지 품질 게이트
# ---------------------------------------------------------------------------
QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "image_quality": {
            "type": "string",
            "enum": ["good", "blurry", "too_dark", "cropped", "not_a_label", "glare"],
        },
        "readable_regions": {"type": "array", "items": {"type": "string"}},
        "unreadable_regions": {"type": "array", "items": {"type": "string"}},
        "overall_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "recommendation": {"type": "string", "enum": ["proceed", "request_reupload"]},
        "reupload_reason_ko": {
            "type": "string",
            "description": "재촬영이 필요하면 MD가 이해할 수 있는 친절한 한국어 사유",
        },
    },
    "required": ["image_quality", "overall_confidence", "recommendation"],
}

QUALITY_PROMPT = """\
당신은 상품 한글표시사항 이미지의 "판독 가능 여부"만 평가합니다.
절대로 상품 정보를 추출하거나 해석하지 마세요. 지금은 사진이 읽을 수 있는 상태인지만 봅니다.

체크리스트:
- 초점이 맞아 글자가 선명한가?
- 조명이 충분한가? (너무 어둡거나 반사광으로 안 보이는 부분은 없는가?)
- 표시사항 전체가 잘리지 않고 프레임 안에 들어와 있는가?
- 애초에 이게 "표시사항(글자가 인쇄된 라벨)" 사진이 맞는가? (제품 전체샷/포장 박스 겉면 사진만 있고
  표시사항 글자 영역이 안 보이면 not_a_label 로 판정)

중요 — recommendation 판단 기준 (재촬영 요청은 최후의 수단입니다):
- request_reupload 는 "이 사진 전체가 실질적으로 못 쓸 수준"일 때만 선택하세요.
  예: 전체적으로 심하게 흐려서 글자를 거의 못 읽음 / 너무 어두워서 대부분 안 보임 /
  표시사항이 아예 안 찍힘(not_a_label) / 핵심 영역 대부분이 잘려나감.
- 반대로, 사진은 전반적으로 읽을 수 있는데 "특정 항목 한두 개"만 안 보이거나 흐린 경우
  (예: 인증번호 숫자 일부만 흐림, 하단 한 줄만 잘림)는 request_reupload 하지 마세요.
  proceed로 판정하고, unreadable_regions 에 그 항목만 적어주면 됩니다.
  그 항목은 다음 단계에서 null로 처리되고, 필요하면 담당자에게 별도로 확인 질문이 뜨니
  사진 전체를 다시 받을 필요가 없습니다.
- 즉 "완벽하지 않다"와 "못 쓴다"를 구분하세요. 기준은 후자입니다.
"""


def assess_image_quality(images: List[Image.Image]) -> dict:
    model = _get_model(response_schema=QUALITY_SCHEMA)
    resp = model.generate_content([QUALITY_PROMPT, *images])
    return json.loads(resp.text)


# ---------------------------------------------------------------------------
# 2단계: 구조화 신호 추출 (카테고리 판정용 signal dict)
# ---------------------------------------------------------------------------
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string", "nullable": True},
        "product_type_text": {
            "type": "string",
            "nullable": True,
            "description": "라벨에 적힌 '식품의 유형', '품목', '제품종류' 등 원문 그대로",
        },
        "product_category_hint": {
            "type": "string",
            "nullable": True,
            "description": "라벨에서 유추 가능한 품목 관련 텍스트(원문 인용). 절대 추론해서 지어내지 말 것.",
        },
        "group3_best_guess": {
            "type": "string",
            "enum": _GROUP3_NAMES,
            "description": (
                "이 상품이 일반 재화 카테고리 중 무엇에 가장 가까운지 의미상으로 판단한 결과. "
                "표현이 라벨 문구와 정확히 일치하지 않아도(예: '자켓'='재킷') 의미가 같으면 그 카테고리로 고른다. "
                "확신이 서지 않으면 억지로 고르지 말고 '미상'을 선택한다."
            ),
        },
        "group3_confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "group3_best_guess 판단에 대한 확신도. 제품유형이 라벨/제품명에 명확히 드러나면 high, 정황상 추정이면 medium/low.",
        },
        "group3_evidence_text": {
            "type": "string",
            "nullable": True,
            "description": "group3_best_guess 판단의 근거가 된 라벨상의 원문 문구 (감사 로그/검수용)",
        },
        "certifications": {
            "type": "array",
            "description": "라벨에 실제로 인쇄된 인증/허가/신고/등록 관련 문구를 모두 나열",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "예: 의료기기허가, 화장품책임판매업자, KC 안전인증 등 원문에 가깝게"},
                    "number": {"type": "string", "nullable": True},
                    "raw_text": {"type": "string", "description": "라벨에 적힌 원문 그대로"},
                },
                "required": ["type", "raw_text"],
            },
        },
        "functional_food_notice": {"type": "boolean", "description": "건강기능식품 기능정보 표시 문구 존재 여부"},
        "age_label": {"type": "string", "nullable": True, "description": "사용연령/권장사용연령 표시 원문"},
        "food_type_label": {"type": "boolean", "description": "'식품의 유형' 항목이 명시적으로 존재하는지"},
        "food_type_text": {
            "type": "string",
            "nullable": True,
            "description": "'식품의 유형' 항목에 적힌 원문 그대로 (예: '수산물', '축산물', '농산물', '기타 수산물가공품', '조미건어물' 등)",
        },
        "is_food_item": {"type": "boolean", "description": "이 상품 자체가 식품(먹는 것)인지"},
        "food_item_report_number_present": {
            "type": "boolean",
            "nullable": True,
            "description": (
                "'품목보고번호' 또는 '품목제조보고번호' 항목이 라벨에 존재하는지. "
                "농수축산물(원물)에는 이 번호가 없고, 가공식품에는 있는 것이 원칙이라 "
                "농수축산물/가공식품을 가르는 가장 확실한 신호다. 안 보이면 null."
            ),
        },
        "nutrition_label_present": {
            "type": "boolean",
            "nullable": True,
            "description": (
                "'영양정보' 표시(나트륨/탄수화물/지방/단백질 등 표)가 라벨에 있는지. "
                "가공식품에 주로 있고 농수축산물(원물)에는 없다. 단, 일부 가공식품은 "
                "영양정보표시 의무화 이전 상품이라 없을 수도 있으니 보조 신호로만 쓸 것. 안 보이면 null."
            ),
        },
        "all_ingredients_listed": {
            "type": "boolean",
            "description": "'전성분' 표시(화장품법에 따라 기재·표시하여야 하는 모든 성분) 존재 여부. 이 문구는 화장품에만 등장하는 고유 표현이므로 정확히 판독할 것.",
        },
        "shelf_life_or_expiry": {"type": "boolean", "description": "사용기한/개봉후사용기간 표시 존재 여부"},
        "usage_method_listed": {"type": "boolean", "description": "'사용방법' 항목 표시 존재 여부"},
        "has_book_metadata": {"type": "boolean", "description": "도서명/저자/출판사 등 서적 메타데이터 존재 여부"},
        "unclear_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "판정에 필요할 것 같은데 라벨에서 찾지 못한 항목들",
        },
    },
    "required": ["certifications", "group3_best_guess", "group3_confidence"],
}

EXTRACTION_PROMPT = """\
당신은 한국 전자상거래 상품정보고시(공정거래위원회 고시) 카테고리 판정을 위해
한글표시사항 이미지에서 정보를 추출하고, 일반 재화 종류를 의미 기준으로 분류하는 역할을 합니다.

절대 규칙 (사실 추출 필드 — certifications, age_label 등):
1. 라벨에 실제로 인쇄/기재된 문구만 추출한다. 없는 내용을 추론하거나 짐작해서 채우지 않는다.
2. 확실하지 않으면 해당 필드는 null 또는 빈 값으로 둔다. (모르면 모른다고 답하는 것이 정답이다)
3. certifications의 raw_text는 라벨에서 그대로 옮겨 적은 원문이어야 한다 (사람이 나중에 원본 대조 가능하도록).
4. 여러 장의 이미지가 주어지면 한 상품의 표시사항이 여러 면에 나뉜 것으로 보고 통합해서 추출한다.

group3_best_guess (품목 종류 판단)에 대해서만은 다른 규칙이 적용됩니다:
- 이건 사실 인용이 아니라 "의미 판단"입니다. 라벨/제품명에 목록의 단어가 정확히 똑같이
  적혀있지 않아도, 같은 뜻이면 해당 카테고리로 분류하세요.
  (예: "자켓", "숏패딩", "가디건" → 모두 의류로 판단)
- 단, 이 판단이 애매하거나 라벨 정보만으론 두 카테고리 중 어느 쪽인지 확신이 안 서면
  절대 임의로 찍지 말고 group3_confidence를 low로 낮추거나 '미상'을 선택하세요.
  틀린 확정보다 낮은 확신도로 사람에게 재확인받는 게 always 안전합니다.

농수축산물(원물) vs 가공식품 판단 시 주의 (food_item_report_number_present, nutrition_label_present):
- 껍질제거/절단/탈골/단순 냉동·건조/데침 같은 "단순 처리"만 거친 원물은 여전히 농수축산물입니다.
  이런 단순처리 문구가 보인다고 해서 가공식품으로 오판하지 마세요.
- 반대로, 조미/염장/훈제 같은 "첨가물 혼합"이나 분쇄·성형처럼 원형을 알아볼 수 없게 만드는
  공정, 또는 자숙·레토르트 같은 본질적 가열이 있으면 가공식품입니다.
- 가장 확실한 판별 신호는 라벨에 '품목보고번호'(또는 품목제조보고번호)가 있는지입니다.
  있으면 가공식품, 원물에는 이 번호 자체가 없습니다. 영양정보표시 유무는 보조 신호로만
  참고하세요 (일부 가공식품은 영양정보표시 의무화 이전 상품이라 없을 수 있습니다).
"""


def extract_signals(images: List[Image.Image]) -> dict:
    model = _get_model(response_schema=EXTRACTION_SCHEMA)
    resp = model.generate_content([EXTRACTION_PROMPT, *images])
    return json.loads(resp.text)
