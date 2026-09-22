# -*- coding: utf-8 -*-
"""
전자상거래 상품정보고시 카테고리 판정 규칙표

우선순위 게이트 구조:
  GROUP 1 (확정 트리거) -> GROUP 2 (식품 원물/가공) -> GROUP 3 (일반 재화, 품명+KC 기반)
  -> GROUP 4 (기타재화, catch-all)

각 카테고리는 법정고시 번호(legal_id)를 갖고, xlsx 사내 간소화 기준의 항목명을
그대로 트리거 근거로 사용한다. 여기서 "trigger_signals"는 gemini_service.py가
추출하는 signal dict의 key와 1:1로 대응한다.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Category:
    legal_id: str          # 고시상 번호 (예: "18")
    name: str               # 상품종류명 (사내 xlsx 기준 명칭)
    group: int              # 1~4군
    priority: int           # 그룹 내 판정 순서 (낮을수록 먼저 체크)
    match: Callable[[dict], bool]      # signals -> 확정 매칭 여부
    ambiguous_with: Optional[list] = field(default_factory=list)  # 3군 내 헷갈리는 상대 legal_id


def _has(signals: dict, key: str) -> bool:
    v = signals.get(key)
    if isinstance(v, list):
        return len(v) > 0
    return bool(v)


def _cert_contains(signals: dict, *keywords: str) -> bool:
    """certifications: [{type, number, raw_text}, ...] 안에서 type/raw_text 키워드 매칭."""
    certs = signals.get("certifications") or []
    for c in certs:
        blob = f"{c.get('type', '')} {c.get('raw_text', '')}"
        if any(k in blob for k in keywords):
            return True
    return False


# ---------------------------------------------------------------------------
# GROUP 1 — 확정 트리거 (법정 번호/고유문구, 오판 여지 거의 없음)
# ---------------------------------------------------------------------------
GROUP1 = [
    Category(
        legal_id="16", name="의료기기", group=1, priority=1,
        match=lambda s: _cert_contains(s, "의료기기"),
    ),
    Category(
        legal_id="18", name="화장품", group=1, priority=2,
        # "전성분" 표시는 화장품법에서만 쓰이는 고유 문구라 그 자체로 확정 신호.
        # 인허가 문구(화장품제조업자 등)만 있어도 확정. 전성분이 안 보이면
        # 사용기한 + 사용방법처럼 화장품 특유의 문구 조합으로 보조 확정.
        match=lambda s: (
            _cert_contains(s, "화장품제조업자", "화장품책임판매업자", "맞춤형화장품판매업자")
            or _has(s, "all_ingredients_listed")
            or (_has(s, "shelf_life_or_expiry") and _has(s, "usage_method_listed"))
        ),
    ),
    Category(
        legal_id="22", name="건강기능식품", group=1, priority=3,
        match=lambda s: _cert_contains(s, "건강기능식품") or _has(s, "functional_food_notice"),
    ),
    Category(
        legal_id="23", name="어린이제품", group=1, priority=4,
        match=lambda s: (
            _cert_contains(s, "안전인증", "안전확인", "공급자적합성확인", "어린이제품")
            and _has(s, "age_label")
        ),
    ),
    Category(
        legal_id="37", name="생활화학제품", group=1, priority=5,
        match=lambda s: _cert_contains(s, "안전확인대상생활화학제품", "생활화학제품"),
    ),
    Category(
        legal_id="38", name="살생물제품", group=1, priority=6,
        match=lambda s: _cert_contains(s, "살생물제품", "살생물처리제품"),
    ),
]

# ---------------------------------------------------------------------------
# GROUP 2 — 식품 원물/가공 판정
# ---------------------------------------------------------------------------
GROUP2 = [
    Category(
        legal_id="21", name="가공식품", group=2, priority=1,
        match=lambda s: _has(s, "food_type_label"),
    ),
    Category(
        legal_id="20", name="농수축산물", group=2, priority=2,
        match=lambda s: s.get("is_food_item") is True and not _has(s, "food_type_label"),
    ),
]

# ---------------------------------------------------------------------------
# GROUP 3 — 일반 재화
#   여기는 코드에서 키워드로 매칭하지 않는다. "이 상품이 무슨 종류냐"는 의미 판단이라
#   Gemini가 group3_best_guess(enum 강제) + group3_confidence 로 직접 분류하고,
#   classifier.py 가 확신도에 따라 바로 확정하거나 MD에게 Y/N으로 재확인한다.
#   (match 필드는 이 그룹에서는 사용하지 않음 — 항상 False)
# ---------------------------------------------------------------------------
GROUP3 = [
    Category("1", "의류", 3, 1, lambda s: False),
    Category("2", "구두/신발", 3, 2, lambda s: False),
    Category("3", "가방", 3, 3, lambda s: False),
    Category("4", "패션잡화", 3, 4, lambda s: False),
    Category("5", "침구류/커튼", 3, 5, lambda s: False),
    Category("6", "가구", 3, 6, lambda s: False),
    Category("7", "영상가전", 3, 7, lambda s: False),
    Category("8", "가정용전기제품", 3, 8, lambda s: False),
    Category("9", "계절가전", 3, 9, lambda s: False),
    Category("10", "사무용기기", 3, 10, lambda s: False),
    Category("11", "광학기기", 3, 11, lambda s: False),
    Category("12", "소형전자", 3, 12, lambda s: False),
    Category("13", "휴대형통신기기", 3, 13, lambda s: False),
    Category("14", "내비게이션", 3, 14, lambda s: False),
    Category("15", "자동차용품", 3, 15, lambda s: False),
    Category("17", "주방용품", 3, 16, lambda s: False),
    Category("19", "귀금속/보석/시계류", 3, 17, lambda s: False),
    Category("24", "악기", 3, 18, lambda s: False),
    Category("25", "스포츠용품", 3, 19, lambda s: False),
    Category("26", "서적", 3, 20, lambda s: False),
    Category("34", "상품권/쿠폰", 3, 21, lambda s: False),
    Category("35", "모바일쿠폰", 3, 22, lambda s: False),
]


# ---------------------------------------------------------------------------
# GROUP 4 — 기타재화 (catch-all)
# ---------------------------------------------------------------------------
GROUP4 = [
    Category("40", "기타재화", 4, 1, lambda s: True),
]

ALL_GROUPS = {1: GROUP1, 2: GROUP2, 3: GROUP3, 4: GROUP4}

# 카테고리 자체가 법정 40개에 없어 관행적으로 기타재화로 귀결되는 품목
# (규칙 매칭이 하나도 안 걸렸을 때 MD에게 보여줄 대표 품목 안내용)
NO_DEDICATED_CATEGORY_EXAMPLES = ["반려동물(사료)", "반려동물(용품)", "위생용품", "의약외품"]


# ---------------------------------------------------------------------------
# 3군 내부에서 라벨만으로 확정이 애매한 지점 — 반드시 Y/N으로 확인
# ---------------------------------------------------------------------------
DISAMBIGUATION_QUESTIONS = {
    frozenset({"1", "5"}): {
        "question": "이 상품은 '입는 옷(의류)'인가요, 아니면 '침구/커튼류(이불, 베개, 커튼 등)'인가요?",
        "options": {"의류": "1", "침구/커튼류": "5"},
    },
    frozenset({"4", "19"}): {
        "question": "이 상품은 모자·벨트 같은 일반 패션잡화인가요, 아니면 귀금속·보석·시계류인가요?",
        "options": {"패션잡화": "4", "귀금속/보석/시계류": "19"},
    },
    "17_food_contact": {
        "question": "이 주방용품은 음식(식품)과 직접 닿는 용도인가요? (예: 그릇, 냄비 / 아니오: 도마 받침대, 행주걸이 등 비접촉 용품)",
        "options": {"네, 식품과 직접 접촉합니다": "food_contact_yes", "아니오, 접촉하지 않습니다": "food_contact_no"},
    },
    "23_child_use": {
        "question": "이 상품은 만 13세 이하 어린이가 사용하도록 만들어졌거나 그렇게 판매될 예정인가요?",
        "options": {"네": "child_yes", "아니오": "child_no"},
    },
    "food_processed": {
        "question": "이 식품은 제조·가공 공정을 거쳤나요? (예: 조미, 가열, 혼합 등 / 아니오: 세척·선별만 거친 원물)",
        "options": {"가공했습니다": "processed_yes", "원물 그대로입니다": "processed_no"},
    },
    "no_dedicated_category": {
        "question": "이 상품이 아래 예시 중 하나에 해당하나요? 맞으면 '기타재화' 서식으로 안내해드립니다.",
        "options": {ex: "40" for ex in NO_DEDICATED_CATEGORY_EXAMPLES} | {"해당 없음 / 다른 상품입니다": "unresolved"},
    },
}
