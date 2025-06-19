## app_cart_ocr.py  (2025-06-19 패치)

import base64, json, re, time
from io import BytesIO
from typing import List

import streamlit as st
import pandas as pd
from openai import OpenAI, APIConnectionError, RateLimitError

# ───────────── 1. Vision → DataFrame 함수 ─────────────
def extract_cart_df(
    image_bytes: bytes,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> pd.DataFrame:
    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode()

    prompt = (
        "이 이미지는 쇼핑몰 장바구니입니다. "
        "각 행을 {\"품명\":\"...\",\"수량\":정수,\"단가\":정수,\"총액\":정수} "
        "형식의 JSON 배열로만 반환하세요."
    )
    schema = {
        "name": "extract_items",
        "description": "Extract cart items from image",
        "parameters": {
            "type":"object",
            "properties":{
                "items":{
                    "type":"array",
                    "items":{
                        "type":"object",
                        "properties":{
                            "품명":{"type":"string"},
                            "수량":{"type":"integer"},
                            "단가":{"type":"integer"},
                            "총액":{"type":"integer"}
                        },
                        "required":["품명","수량","단가","총액"]
                    }
                }
            },
            "required":["items"]
        }
    }

    for i in range(max_retries + 1):
        try:
            res = client.chat.completions.create(
                model=model,
                tools=[{"type":"function","function":schema}],
                tool_choice="auto",
                messages=[{
                    "role":"user",
                    "content":[
                        {"type":"text","text":prompt},
                        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}
                    ]
                }]
            )
            break
        except (RateLimitError, APIConnectionError):
            if i == max_retries:
                return pd.DataFrame(columns=["품명","수량","단가","총액"])
            time.sleep(2 ** i)

    choice = res.choices[0]
    items: List[dict] = []

    # ── 수정된 파싱 블록 ──
    if choice.message.tool_calls:
        arguments = choice.message.tool_calls[0].function.arguments
        if isinstance(arguments, str):          # 문자열이면 JSON 파싱
            arguments = json.loads(arguments)
        items = arguments.get("items", [])
    else:
        raw = choice.message.content.strip()
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", raw, re.S)
            if m:
                items = json.loads(m.group())

    return pd.DataFrame(items, columns=["품명","수량","단가","총액"])

# ───────────── 2. Streamlit UI ─────────────
st.set_page_config(page_title="🛒 장바구니 → Excel", layout="centered")
st.title("🛒 장바구니 캡처 → 품목 Excel 에듀파인 변환기 ")

upload = st.file_uploader("장바구니 캡처(JPG/PNG)", ["jpg","jpeg","png"])
if upload:
    st.image(upload, caption="업로드된 이미지", use_container_width=True)   # ← 경고 해결

    with st.spinner("처리 중…"):
        df = extract_cart_df(upload.read(), st.secrets["OPENAI_API_KEY"])

    if df.empty:
        st.warning("❗️품목을 추출하지 못했습니다.")
    else:
        st.subheader("📋 추출 결과")
        st.dataframe(df, use_container_width=True)

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            df.to_excel(w, index=False, sheet_name="장바구니")

        st.download_button(
            "📥 Excel 다운로드",
            data=buf.getvalue(),
            file_name="cart_items.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
