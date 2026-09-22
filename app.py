# -*- coding: utf-8 -*-
"""
상품정보고시 카테고리 추천 챗봇 — Streamlit 프로토타입

실행:
    pip install streamlit google-generativeai pillow
    export GOOGLE_API_KEY=<AI Studio에서 발급받은 키>
    streamlit run app.py

흐름:
    이미지 업로드 -> [1단계] 이미지 품질 게이트 -> [2단계] 신호 추출(Gemini)
    -> [3단계] 규칙 엔진 판정 (classifier.py) -> 애매하면 Y/N 질문 반복 -> 카테고리 확정
"""

import streamlit as st
from PIL import Image

import gemini_service
import logging_service
from classifier import classify

st.set_page_config(page_title="상품정보고시 카테고리 추천", page_icon="🏷️", layout="centered")

st.markdown(
    """
    <style>
    :root { --brand: #5f0080; }

    .brand-title{
        font-size:26px; font-weight:700;
        display:flex; align-items:baseline; gap:11px; flex-wrap:wrap;
        margin-bottom: 0.25rem;
    }
    .brand-watermark{
        display:inline-flex; align-items:center; position:relative; top:-3px;
        font-family:'JetBrains Mono', ui-monospace, monospace;
        font-size:10.5px; font-weight:600; letter-spacing:.24em; text-transform:uppercase;
        color: var(--brand); opacity:.4;
        padding:3.5px 10px 3.5px 9px; border:1px solid currentColor; border-radius:999px;
        transform:rotate(-4deg); user-select:none; white-space:nowrap;
        transition:opacity .18s ease, transform .18s ease;
    }
    .brand-watermark::before{ content:'✦'; margin-right:5px; font-size:7.5px; }
    .brand-title:hover .brand-watermark{ opacity:.65; transform:rotate(-4deg) scale(1.03); }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in [
    ("stage", "upload"),        # upload -> quality_checked -> extracted -> done
    ("images", []),
    ("quality", None),
    ("signals", None),
    ("answers", {}),
    ("result", None),
    ("case_id", None),
    ("logged", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def reset_all():
    for key, default in [
        ("stage", "upload"), ("images", []), ("quality", None),
        ("signals", None), ("answers", {}), ("result", None),
        ("case_id", None), ("logged", False),
    ]:
        st.session_state[key] = default


st.markdown(
    """
    <div class="brand-title">
        <span class="brand-title-main">🏷️ 상품정보고시 카테고리 추천</span>
        <span class="brand-watermark">By QM</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("한글표시사항 이미지를 올리면 전자상거래 상품정보제공고시 40개 카테고리 중 알맞은 것을 추천합니다.")

with st.sidebar:
    st.markdown("### 진행 단계")
    st.write(f"현재 단계: **{st.session_state.stage}**")
    if st.button("처음부터 다시"):
        reset_all()
        st.rerun()

# ---------------------------------------------------------------------------
# 업로드
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "한글표시사항 이미지 (앞면/뒷면 등 여러 장 가능)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("분석 시작", type="primary"):
    st.session_state.images = [Image.open(f) for f in uploaded_files]
    st.session_state.stage = "checking_quality"
    st.session_state.quality = None
    st.session_state.signals = None
    st.session_state.answers = {}
    st.session_state.result = None
    st.rerun()

if st.session_state.images:
    st.image(st.session_state.images, width=160)

# ---------------------------------------------------------------------------
# 1단계: 품질 게이트
# ---------------------------------------------------------------------------
if st.session_state.stage == "checking_quality":
    with st.spinner("이미지 판독 가능 여부를 확인하는 중..."):
        try:
            quality = gemini_service.assess_image_quality(st.session_state.images)
        except Exception as e:
            st.error(f"Gemini 호출 중 오류가 발생했습니다: {e}")
            st.stop()
    st.session_state.quality = quality

    if quality.get("recommendation") == "request_reupload":
        st.session_state.stage = "upload"
        st.warning(
            "⚠️ 이미지를 다시 받아야 할 것 같아요.\n\n"
            f"**사유**: {quality.get('reupload_reason_ko', '표시사항이 선명하게 보이지 않습니다.')}"
        )
        if quality.get("unreadable_regions"):
            st.caption("잘 안 읽힌 영역: " + ", ".join(quality["unreadable_regions"]))
        st.info("다시 촬영한 이미지를 위에 업로드하고 '분석 시작'을 눌러주세요.")
    else:
        st.session_state.stage = "extracting"
        st.rerun()

# ---------------------------------------------------------------------------
# 2단계: 신호 추출
# ---------------------------------------------------------------------------
if st.session_state.stage == "extracting":
    with st.spinner("표시사항에서 정보를 추출하는 중..."):
        try:
            signals = gemini_service.extract_signals(st.session_state.images)
        except Exception as e:
            st.error(f"Gemini 호출 중 오류가 발생했습니다: {e}")
            st.stop()
    st.session_state.signals = signals
    st.session_state.case_id = logging_service.new_case_id()
    st.session_state.stage = "classifying"
    st.rerun()

# ---------------------------------------------------------------------------
# 3단계: 규칙 엔진 판정 (+ 필요시 Y/N 질문)
# ---------------------------------------------------------------------------
if st.session_state.stage == "classifying":
    result = classify(st.session_state.signals, st.session_state.answers)
    st.session_state.result = result

    if result.status == "needs_question":
        st.subheader("🙋 확인이 필요해요")
        st.write(result.question["text"])
        choice = st.radio(
            "선택", list(result.question["options"].keys()),
            key=f"q_{result.question['key']}", label_visibility="collapsed",
        )
        if st.button("답변 제출"):
            value = result.question["options"][choice]
            if value != "unresolved":
                st.session_state.answers[result.question["key"]] = value
                st.rerun()
            else:
                st.error("담당 MD 또는 상품기획팀에 직접 문의가 필요한 케이스입니다.")
                st.stop()
    else:
        st.session_state.stage = "done"
        st.rerun()

# ---------------------------------------------------------------------------
# 완료
# ---------------------------------------------------------------------------
if st.session_state.stage == "done" and st.session_state.result:
    cat = st.session_state.result.category
    st.success(f"✅ 추천 카테고리: **({cat.legal_id}) {cat.name}**")

    with st.expander("판정 근거 (감사 로그)"):
        for line in st.session_state.result.trail:
            st.write("- " + line)

    with st.expander("Gemini 추출 원본 신호값 (검수용)"):
        st.json(st.session_state.signals)

    # 판정이 확정될 때마다 케이스를 한 번만 자동으로 기록 (재실행돼도 중복 기록 방지)
    if not st.session_state.logged:
        logging_service.log_case(
            case_id=st.session_state.case_id,
            product_name=(st.session_state.signals or {}).get("product_name", ""),
            category_id=cat.legal_id,
            category_name=cat.name,
            group3_confidence=(st.session_state.signals or {}).get("group3_confidence", ""),
            trail=st.session_state.result.trail,
            signals=st.session_state.signals or {},
        )
        st.session_state.logged = True

    if not logging_service.logging_configured():
        st.caption(
            "⚠️ 케이스 자동 기록이 아직 설정되지 않았습니다 "
            "(Secrets에 Google 스프레드시트 연동 정보가 없음). 지금은 이 결과가 어디에도 저장되지 않아요."
        )

    st.caption("이 추천이 실제와 다르면 하단 피드백으로 알려주세요 — 규칙표 보강에 사용됩니다.")
    feedback = st.text_area("피드백 (선택)", placeholder="예: 이건 사실 위생용품인데 기타재화로 갔어야...")
    if st.button("피드백 제출"):
        if feedback.strip():
            logging_service.log_feedback(case_id=st.session_state.case_id, feedback_text=feedback.strip())
        st.success("피드백이 기록되었습니다. 감사합니다!")
