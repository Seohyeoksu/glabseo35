import base64
import json
import re
import time
from io import BytesIO
from typing import List, Tuple
from datetime import datetime

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
        "품목별 총액과 수량에 맞추어서 단가를 조정해주세요. "
        "각 행을 {\"내용\":\"...\",\"규격\":\"...\",\"수량\":정수,\"예상단가\":정수} "
        "형식의 JSON 배열로만 반환하세요. 규격이 없으면 빈 문자열로 처리하세요."
    )
    
    schema = {
        "name": "extract_items",
        "description": "Extract cart items from image",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "내용": {"type": "string"},
                            "규격": {"type": "string"},
                            "수량": {"type": "integer"},
                            "예상단가": {"type": "integer"}
                        },
                        "required": ["내용", "규격", "수량", "예상단가"]
                    }
                }
            },
            "required": ["items"]
        }
    }
    
    for i in range(max_retries + 1):
        try:
            res = client.chat.completions.create(
                model=model,
                tools=[{"type": "function", "function": schema}],
                tool_choice="auto",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }]
            )
            break
        except (RateLimitError, APIConnectionError):
            if i == max_retries:
                return pd.DataFrame(columns=["내용", "규격", "수량", "예상단가"])
            time.sleep(2 ** i)
    
    choice = res.choices[0]
    items: List[dict] = []
    
    if choice.message.tool_calls:
        arguments = choice.message.tool_calls[0].function.arguments
        if isinstance(arguments, str):
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
    
    return pd.DataFrame(items, columns=["내용", "규격", "수량", "예상단가"])


# ───────────── 2. 지출품의서 생성 함수 ─────────────
def create_expense_report(df: pd.DataFrame, title: str, purpose: str) -> Tuple[str, str]:
    """지출품의서 품의개요와 지출품의서 생성"""
    
    # 총액 계산
    df['금액'] = df['수량'] * df['예상단가']
    total_amount = df['금액'].sum()
    
    # 한글 금액 변환
    def number_to_korean(num):
        units = ['', '만', '억', '조']
        nums = ['영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
        result = []
        
        num_str = str(int(num))
        num_len = len(num_str)
        
        for i, digit in enumerate(num_str):
            if digit != '0':
                result.append(nums[int(digit)])
                unit_pos = (num_len - i - 1) // 4
                digit_pos = (num_len - i - 1) % 4
                
                if digit_pos == 3:
                    result.append('천')
                elif digit_pos == 2:
                    result.append('백')
                elif digit_pos == 1:
                    result.append('십')
                
                if digit_pos == 0 and unit_pos > 0:
                    result.append(units[unit_pos])
        
        return ''.join(result) + '원정'
    
    korean_amount = number_to_korean(total_amount)
    
    # 산출근거 생성
    calculation_details = []
    for _, row in df.iterrows():
        if row['규격']:
            item_desc = f"{row['내용']}({row['규격']})"
        else:
            item_desc = row['내용']
        
        detail = f"{item_desc} : {row['예상단가']:,}원 × {row['수량']}개 = {row['금액']:,}원"
        calculation_details.append(detail)
    
    # 품의개요 생성
    overview = f"""{title}
────────────────────────────────────────────────────────────────────
품 의
개 요   {title}을 다음과 같이 구입하고자 합니다.
        1. 목적  : {purpose}
        2. 품목  : 
"""
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if row['규격']:
            overview += f"           {i}) {row['내용']} ({row['규격']})\n"
        else:
            overview += f"           {i}) {row['내용']}\n"
    
    overview += f"""        3. 소요예산 : 금 {total_amount:,}원(금{korean_amount})
        4. 산출근거 : 
"""
    
    for detail in calculation_details:
        overview += f"           - {detail}\n"
    
    overview += """
붙임  지출품의서 1부.  끝."""
    
    # 지출품의서 생성
    current_year = datetime.now().year
    current_date = datetime.now().strftime("%Y. %m. %d.")
    
    expense_report = f"""────────────────────────────────────────────────────────────────────
                          지 출 품 의 서
────────────────────────────────────────────────────────────────────
회계연도 : {current_year}년                             품의번호 : 
────────────────────────────────────────────────────────────────────
제   목  {title}
────────────────────────────────────────────────────────────────────
품 의
개 요   1. 일시 : {current_date}
        2. {title}을 위한 물품 구입을 다음과 같이 실시하고자 합니다.
           가. 목적 : {purpose}
           나. 소요예산 : 금 {total_amount:,}원(금{korean_amount})
           다. 산출내역 :
"""
    
    for i, detail in enumerate(calculation_details, 1):
        expense_report += f"               {i}) {detail}\n"
    
    expense_report += """
붙임  지출품의서 1부.  끝."""
    
    return overview, expense_report


# ───────────── 3. Streamlit UI ─────────────
st.set_page_config(page_title="🛒 장바구니 → Excel & 지출품의서", layout="centered")
st.title("🛒 장바구니 캡처 → Excel & 지출품의서 변환기")

# 여러 이미지 업로드
uploads = st.file_uploader(
    "장바구니 캡처(JPG/PNG) - 여러 개 선택 가능",
    ["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploads:
    # 업로드된 이미지들 표시
    cols = st.columns(min(len(uploads), 3))
    for i, upload in enumerate(uploads):
        with cols[i % 3]:
            st.image(upload, caption=f"이미지 {i+1}", use_container_width=True)
    
    # 지출품의서 정보 입력
    st.subheader("📝 지출품의서 정보")
    col1, col2 = st.columns(2)
    with col1:
        doc_title = st.text_input("제목", placeholder="예: 교무실 프린터 토너 구입")
    with col2:
        doc_purpose = st.text_input("목적", placeholder="예: 인쇄 사항 발생으로 교체 필요")
    
    # 처리 버튼
    if st.button("🚀 처리 시작", type="primary"):
        all_dfs = []
        
        with st.spinner("이미지 처리 중..."):
            for i, upload in enumerate(uploads):
                st.write(f"이미지 {i+1} 처리 중...")
                df = extract_cart_df(upload.read(), st.secrets["OPENAI_API_KEY"])
                if not df.empty:
                    all_dfs.append(df)
        
        if all_dfs:
            # 모든 데이터 합치기
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # 결과 표시
            st.subheader("📋 추출 결과")
            st.dataframe(combined_df, use_container_width=True)
            
            # 총액 표시
            combined_df['금액'] = combined_df['수량'] * combined_df['예상단가']
            total = combined_df['금액'].sum()
            st.info(f"💰 총 합계: {total:,}원")
            
            # Excel 다운로드
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                combined_df.to_excel(writer, index=False, sheet_name="품목내역")
                
                # 워크북과 워크시트 가져오기
                workbook = writer.book
                worksheet = writer.sheets["품목내역"]
                
                # 헤더 포맷
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D7E4BD',
                    'border': 1
                })
                
                # 숫자 포맷
                number_format = workbook.add_format({'num_format': '#,##0'})
                
                # 헤더 스타일 적용
                for col_num, value in enumerate(combined_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # 숫자 컬럼에 포맷 적용
                worksheet.set_column('C:D', 15, number_format)  # 수량, 예상단가
                worksheet.set_column('E:E', 15, number_format)  # 금액
                
                # 열 너비 조정
                worksheet.set_column('A:A', 40)  # 내용
                worksheet.set_column('B:B', 20)  # 규격
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Excel 다운로드",
                    data=buf.getvalue(),
                    file_name="품목내역.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            
            # 지출품의서 생성
            if doc_title and doc_purpose:
                st.subheader("📄 지출품의서")
                
                overview, expense_report = create_expense_report(
                    combined_df[['내용', '규격', '수량', '예상단가']],
                    doc_title,
                    doc_purpose
                )
                
                # 품의개요 표시
                st.text_area("품의개요", overview, height=300)
                
                # 지출품의서 표시
                st.text_area("지출품의서", expense_report, height=300)
                
                # 텍스트 파일로 다운로드
                doc_content = f"{'='*70}\n품의개요\n{'='*70}\n\n{overview}\n\n\n{'='*70}\n지출품의서\n{'='*70}\n\n{expense_report}"
                
                with col2:
                    st.download_button(
                        "📥 지출품의서 다운로드",
                        data=doc_content.encode('utf-8'),
                        file_name="지출품의서.txt",
                        mime="text/plain",
                    )
            else:
                if not doc_title or not doc_purpose:
                    st.info("💡 지출품의서를 생성하려면 제목과 목적을 입력해주세요.")
        else:
            st.warning("❗️ 품목을 추출하지 못했습니다.")
else:
    st.info("👆 장바구니 캡처 이미지를 업로드해주세요. 여러 개 선택 가능합니다.")

# 사용 안내
with st.expander("📖 사용 안내"):
    st.markdown("""
    ### 사용 방법
    1. **이미지 업로드**: 장바구니 캡처 이미지를 선택합니다 (여러 개 가능)
    2. **지출품의서 정보**: 제목과 목적을 입력합니다 (선택사항)
    3. **처리 시작**: 버튼을 클릭하여 처리를 시작합니다
    4. **다운로드**: Excel 파일과 지출품의서를 다운로드합니다
    
    ### 출력 형식
    - **Excel**: 내용, 규격, 수량, 예상단가 컬럼
    - **지출품의서**: 품의개요와 지출품의서 양식
    """)
