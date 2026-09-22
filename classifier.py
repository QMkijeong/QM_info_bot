# -*- coding: utf-8 -*-
"""
카테고리 판정 규칙 엔진.

Gemini는 여기서 절대 판단하지 않는다 — 입력은 이미 추출된 signals(dict) 뿐이고,
그 signals를 category_rules.py 의 결정 순서(1군->2군->3군->4군)대로 코드가 매칭한다.
확정할 수 없으면 반드시 pending_question 을 반환하고, 카테고리를 임의로 추측하지 않는다.

호출 패턴 (Streamlit app.py 참고):
    result = classify(signals, answers={})
    while result.status == "needs_question":
        # UI에 result.question 을 보여주고 MD가 고른 값을 answers에 누적
        answers[result.question["key"]] = chosen_value
        result = classify(signals, answers)
    # result.status == "confirmed" -> result.category, result.trail
"""

from dataclasses import dataclass, field
from typing import Optional

from category_rules import (
    GROUP1, GROUP2, GROUP3, GROUP4,
    DISAMBIGUATION_QUESTIONS, Category, _cert_contains, _has,
)

_BY_ID = {c.legal_id: c for c in GROUP1 + GROUP2 + GROUP3 + GROUP4}


@dataclass
class ClassificationResult:
    status: str  # "confirmed" | "needs_question"
    category: Optional[Category] = None
    trail: list = field(default_factory=list)   # 판정 근거 감사 로그 (MD/개발자 확인용)
    question: Optional[dict] = None             # {"key":..., "text":..., "options": {label: value}}


def _confirmed(cat: Category, trail: list) -> ClassificationResult:
    return ClassificationResult(status="confirmed", category=cat, trail=trail)


def _ask(key: str, text: str, options: dict, trail: list) -> ClassificationResult:
    return ClassificationResult(
        status="needs_question",
        trail=trail,
        question={"key": key, "text": text, "options": options},
    )


def classify(signals: dict, answers: dict = None) -> ClassificationResult:
    answers = answers or {}
    trail = list(signals.get("_trail", []))

    # ------------------------------------------------------------------
    # GROUP 1 — 확정 트리거
    # ------------------------------------------------------------------
    g1_matches = [c for c in GROUP1 if c.match(signals)]

    if len(g1_matches) == 1:
        cat = g1_matches[0]
        trail.append(f"[1군] '{cat.name}' 확정 트리거 감지 → 확정")
        return _confirmed(cat, trail)

    if len(g1_matches) > 1:
        key = "group1_conflict"
        ids = tuple(c.legal_id for c in g1_matches)
        if answers.get(f"{key}:{ids}"):
            chosen = answers[f"{key}:{ids}"]
            cat = _BY_ID[chosen]
            trail.append(f"[1군] 복수 트리거 충돌({[c.name for c in g1_matches]}) → MD 확인 결과 '{cat.name}'")
            return _confirmed(cat, trail)
        trail.append(f"[1군] 복수 트리거 동시 감지: {[c.name for c in g1_matches]} → MD 확인 필요")
        return _ask(
            f"{key}:{ids}",
            "라벨에서 서로 다른 카테고리의 확정 문구가 동시에 감지됐습니다. 이 상품은 어디에 가장 가깝나요?",
            {c.name: c.legal_id for c in g1_matches},
            trail,
        )

    # 어린이제품 부분 매칭 (KC 인증정보는 있는데 사용연령 표시가 안 읽힌 경우)
    if _cert_contains(signals, "안전인증", "안전확인", "공급자적합성확인") and not _has(signals, "age_label"):
        key = "23_child_use"
        if key in answers:
            if answers[key] == "child_yes":
                trail.append("[1군-보완] KC 어린이제품 인증 감지 + MD 확인(어린이용 맞음) → 어린이제품 확정")
                return _confirmed(_BY_ID["23"], trail)
            trail.append("[1군-보완] KC 인증 감지했으나 MD가 어린이용 아님으로 확인 → 3군으로 진행")
        else:
            trail.append("[1군-보완] KC 안전인증류 감지, 사용연령 표시는 미확인 → MD 확인 필요")
            q = DISAMBIGUATION_QUESTIONS["23_child_use"]
            return _ask(key, q["question"], q["options"], trail)

    # ------------------------------------------------------------------
    # GROUP 2 — 식품 원물/가공
    #   1순위 신호: 품목보고번호 유무 (있으면 가공식품, 원물엔 없음)
    #   2순위 신호: 영양정보표시 유무 (있으면 가공식품 쪽 근거. 없다고 원물 확정은 안 함
    #              — 영양정보표시 의무화가 2028년부터라 일부 가공식품엔 없을 수 있음)
    #   둘 다 불명확하면 MD에게 직접 질문.
    # ------------------------------------------------------------------
    if signals.get("is_food_item") is True:
        report_no = signals.get("food_item_report_number_present")
        nutrition = signals.get("nutrition_label_present")

        if report_no is True:
            trail.append("[2군] 품목보고번호 존재 → 가공식품 확정")
            return _confirmed(_BY_ID["21"], trail)
        if nutrition is True:
            trail.append("[2군] 영양정보표시 존재(품목보고번호는 미확인) → 가공식품 확정")
            return _confirmed(_BY_ID["21"], trail)
        if report_no is False and nutrition is not True:
            trail.append("[2군] 품목보고번호 없음 + 영양정보표시도 없음(단순처리 원물 추정) → 농수축산물 확정")
            return _confirmed(_BY_ID["20"], trail)

        key = "food_processed"
        if key in answers:
            cat_id = "21" if answers[key] == "processed_yes" else "20"
            trail.append(f"[2군] MD 확인({answers[key]}) → {_BY_ID[cat_id].name} 확정")
            return _confirmed(_BY_ID[cat_id], trail)
        trail.append("[2군] 품목보고번호/영양정보표시 모두 불명확 → MD 확인 필요")
        q = DISAMBIGUATION_QUESTIONS["food_processed"]
        return _ask(key, q["question"], q["options"], trail)

    # ------------------------------------------------------------------
    # GROUP 3 — 일반 재화 (Gemini의 의미 판단 결과 + 확신도 기반)
    #   product_category_hint 문구를 코드가 키워드로 매칭하지 않는다. "자켓/재킷"
    #   같은 표현 차이는 Gemini가 의미로 이해해서 group3_best_guess 로 이미 분류해뒀고,
    #   여기서는 그 결과를 확신도에 따라 확정하거나 사람에게 재확인만 시킨다.
    # ------------------------------------------------------------------
    guess_name = signals.get("group3_best_guess")
    confidence = signals.get("group3_confidence")
    guess_cat = next((c for c in GROUP3 if c.name == guess_name), None)

    if guess_cat is not None:
        if confidence == "high":
            evidence = signals.get("group3_evidence_text")
            trail.append(
                f"[3군] Gemini 의미판단 '{guess_cat.name}' (확신도 high"
                + (f", 근거: '{evidence}'" if evidence else "") + ") → 확정"
            )
            return _confirmed(guess_cat, trail)

        # medium/low 확신도 -> 추측은 하되, MD에게 Y/N으로 재확인
        confirm_key = f"group3_confirm:{guess_cat.legal_id}"
        if confirm_key in answers:
            if answers[confirm_key] == "yes":
                trail.append(f"[3군] Gemini 추정 '{guess_cat.name}'(확신도 {confidence}) → MD 확인(맞음) → 확정")
                return _confirmed(guess_cat, trail)
            trail.append(f"[3군] Gemini 추정 '{guess_cat.name}' → MD가 아니라고 확인 → 전체 목록 재선택 필요")
            # 아래 manual_pick 질문으로 계속 진행
        else:
            trail.append(f"[3군] Gemini 추정 '{guess_cat.name}' (확신도 {confidence}) → MD 확인 필요")
            return _ask(
                confirm_key,
                f"이 상품, 혹시 '{guess_cat.name}' 카테고리 맞을까요?",
                {"네, 맞습니다": "yes", "아니요, 다릅니다": "no"},
                trail,
            )

    # Gemini가 '미상'을 골랐거나, MD가 추정을 거부한 경우 -> 전체 목록에서 직접 선택
    manual_key = "group3_manual_pick"
    if manual_key in answers:
        picked = answers[manual_key]
        if picked in _BY_ID:
            cat = _BY_ID[picked]
            trail.append(f"[3군] MD가 직접 '{cat.name}' 선택 → 확정")
            return _confirmed(cat, trail)
        # 목록에 없음 -> 기타재화로 이동 (아래 GROUP4 로직에서 처리)
        trail.append("[3군] MD가 목록에 해당 없음으로 응답 → 기타재화 후보로 이동")
    else:
        trail.append("[3군] Gemini 의미판단 불가('미상') → MD 직접 선택 필요")
        options = {c.name: c.legal_id for c in GROUP3}
        options["목록에 없음"] = "none"
        return _ask(manual_key, "이 상품과 가장 가까운 카테고리를 직접 골라주세요.", options, trail)

    # ------------------------------------------------------------------
    # GROUP 4 — 기타재화 (catch-all, 최종 확인만 거치고 확정)
    # ------------------------------------------------------------------
    key = "no_dedicated_category"
    if key in answers:
        trail.append("[4군] MD 최종 확인 → 기타재화 확정")
        return _confirmed(_BY_ID["40"], trail)
    trail.append("[4군] 1~3군 어디에도 매칭되지 않음 → 기타재화 후보, MD 최종 확인 필요")
    q = DISAMBIGUATION_QUESTIONS["no_dedicated_category"]
    return _ask(key, q["question"], q["options"], trail)
