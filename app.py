from collections import Counter
import streamlit as st
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from konlpy.tag import Okt
import fitz  # PyMuPDF 라이브러리 임포트

# --- Okt 형태소 분석기 초기화 ---
okt = Okt()

# --- 불용어 및 기본 설정 (이전과 동일하게 유지 또는 사용자 정의 목록 사용) ---
STOPWORDS = [ # 사용자의 최신 불용어 목록으로 교체해주세요
    # 예시: '수', '것', '때', '등', ...
    # (이전에 제공된 길고 구체적인 불용어 목록을 여기에 넣어주세요)
    '이', '그', '저', '것', '수', '때', '등', '및', '년', '월', '일', '좀', '중', '위해',
    '그것', '이것', '저것', '여기', '저기', '거기', '자신', '자체', '대한', '통해', '관련',
    '여러', '가지', '다른', '부분', '경우', '정도', '사이', '문제', '내용', '결과', '과정',
    '사용', '생각', '지금', '현재', '당시', '때문에', '면서', '동안', '위한', '따라',
    '대해', '통한', '관련된', '있음', '없음', '같음', '사항', '활동', '모습', '분야',
    '능력', '역량', '자세', '태도', '노력', '바탕', '역할', '학습', '이해',
    '항상', '매우', '다소', '특히', '가장', '더욱', '적극적', '구체적', '다양한', '꾸준히',
    '뛰어남', '우수함', '보임', '발휘함', '참여함', '탐구함', '발전함', '향상됨', '함양함',
    '만듦', '발표함', '제시함', '제출함', '바', '점', '측면', '과제', '조사', '주제',
    '자료', '발표', '토론', '보고서', '탐구', '연구', '프로젝트', '실험', '수업', '시간',
    '이용', '참여',
    '고', '한', '터', '이후', '이전', '내', '외', '속',
    '열심히',
    '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉', '열',
    '첫째', '둘째', '셋째', '다음', '먼저', '비롯', '비롯한', '등등', '기타',
    '활용', '실시', '진행', '수행', '제작', '경험',
    '관찰', '기록', '정리',
    '기반', '향상', '발전', '성장',
    '수준', '관심', '흥미', '호기심', '질문', '제안',
    '해결', '도움', '협력', '소통', '관계', '중심', '대상', '방법', '원리', '개념',
    '의미', '중요성', '필요성', '가치', '다양성', '창의성', '적극성', '성실성', '책임감',
    '자기주도', '모범', '리더십', '팔로우십', '공동체', '배려', '나눔', '봉사',
    '교과', '과목', '단원', '영역',
    '학기', '학년', '학교', '교내', '교외',
    '대회', '행사', '캠프', '동아리', '부서', '조직', '단체', '기관', '시설',
    '학생', '교사', '친구', '우리', '모둠', '팀',
    '시작', '마무리', '완성',
    '드러냄', '갖춤', '지님', '인정됨', '확인됨', '관찰됨',
    '우수', '뛰어남', '탁월', '미흡', '부족',
    '관련하여', '대하여', '바탕으로', '중심으로', '통하여', '비추어', '앞서',
    '기록함', '기재함', '작성함',
    '됨', '함', '높음', '낮음', '많음', '적음',
    '기대됨', '요망됨'
]
MIN_NOUN_LEN = 2
MIN_WORD_COUNT_FOR_W2V = 1

# --- PDF 텍스트 추출 함수 ---
def extract_text_from_pdf(uploaded_file):
    """PyMuPDF를 사용하여 업로드된 PDF 파일에서 텍스트를 추출합니다."""
    text = ""
    try:
        # 업로드된 파일은 BytesIO 객체이므로, stream으로 바로 열 수 있습니다.
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
    except Exception as e:
        st.error(f"PDF 처리 중 오류 발생: {e}")
        st.error("올바른 PDF 파일인지, 손상되지 않았는지 확인해주세요. 암호화된 PDF는 처리할 수 없습니다.")
        return None
    return text

# --- 기존 함수들 (extract_meaningful_nouns, get_keywords_from_nouns) 은 동일하게 사용 ---
def extract_meaningful_nouns(text):
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.]+", "", str(text)).strip()
    text = re.sub(r"\s+", " ", text)
    if not text: return []
    nouns = okt.nouns(text)
    meaningful_nouns = []
    for noun in nouns:
        if (noun not in STOPWORDS and len(noun) >= MIN_NOUN_LEN and not noun.isnumeric()):
            meaningful_nouns.append(noun)
    return meaningful_nouns

def get_keywords_from_nouns(noun_list):
    if not noun_list: return [], []
    word_counts = Counter(noun_list)
    sorted_keywords_with_counts = word_counts.most_common()
    wordset = [item[0] for item in sorted_keywords_with_counts]
    wordsetcount = [item[1] for item in sorted_keywords_with_counts]
    return wordset, wordsetcount

# --- Streamlit UI ---
st.set_page_config(page_title="생기부 분석기", layout="wide")
st.title("📝 생기부 키워드 분석 및 연관 문장 추천")
st.markdown("""
KoNLPy 형태소 분석기를 사용하여 명사 위주로 키워드를 추출하고,
Word2Vec 모델을 통해 유사 단어 및 관련 높은 문장을 찾아줍니다.
**PDF 파일을 업로드하거나 텍스트를 직접 입력하여 분석할 수 있습니다.**
""")

# --- 입력 방식 선택 ---
# st.subheader("1. 분석할 생기부 데이터 입력") # 소제목 추가
# input_method = st.radio(
#     "입력 방식을 선택하세요:",
#     ('텍스트 직접 입력', 'PDF 파일 업로드')
# )

raw_sentence_input = None # 초기화
# if input_method == '텍스트 직접 입력':
#     raw_sentence_input_area = st.text_area("분석할 생기부 내용을 입력하세요:", height=250, placeholder="여기에 텍스트를 입력해주세요...")
#     if raw_sentence_input_area.strip():
#         raw_sentence_input = raw_sentence_input_area
# else: # PDF 파일 업로드
#     uploaded_pdf_file = st.file_uploader("생기부 PDF 파일을 업로드하세요.", type="pdf")
#     if uploaded_pdf_file is not None:
#         with st.spinner("PDF 파일을 읽고 있습니다..."):
#             raw_sentence_input = extract_text_from_pdf(uploaded_pdf_file)
#             if raw_sentence_input:
#                 st.success("PDF 파일에서 텍스트를 성공적으로 추출했습니다!")
#                 # 추출된 텍스트 일부를 보여주어 확인 (선택 사항)
#                 # st.text_area("추출된 텍스트 (일부):", raw_sentence_input[:1000] + "...", height=100, disabled=True)
#             else:
#                 st.error("PDF에서 텍스트를 추출하지 못했습니다. 파일 내용을 확인해주세요.")

# --- 더 간단한 입력 방식: 파일 업로더와 텍스트 영역을 모두 표시하고, 파일이 있으면 파일 우선 ---
st.subheader("1. 분석할 생기부 데이터 입력")
st.markdown("이 링크는 [새 탭에서 Naver](https://www.naver.com)가 열립니다.", unsafe_allow_html=True) # 일반 마크다운은 target 지원 안함
uploaded_pdf_file = st.file_uploader("생기부 PDF 파일 업로드 (PDF 업로드 시 아래 텍스트 입력 내용은 무시됩니다):", type="pdf")
raw_sentence_input_area = st.text_area("또는, 생기부 내용을 여기에 직접 붙여넣으세요:", height=200, placeholder="PDF를 업로드하지 않을 경우 여기에 텍스트를 입력해주세요...")

if uploaded_pdf_file is not None:
    with st.spinner("PDF 파일을 읽고 분석 준비 중입니다..."):
        extracted_text_from_pdf = extract_text_from_pdf(uploaded_pdf_file)
        if extracted_text_from_pdf:
            raw_sentence_input = extracted_text_from_pdf
            st.success("PDF 파일에서 텍스트를 성공적으로 추출했습니다!")
            # st.info("PDF 내용으로 분석을 진행합니다.") # 사용자에게 알림
        else:
            st.error("PDF에서 텍스트를 추출하지 못했습니다. 파일이 올바른지 확인하거나, 아래 텍스트 영역에 직접 입력해주세요.")
            # PDF 추출 실패 시 텍스트 영역 입력을 사용하도록 raw_sentence_input을 None으로 유지
elif raw_sentence_input_area.strip():
    raw_sentence_input = raw_sentence_input_area
    # st.info("입력된 텍스트로 분석을 진행합니다.")
else:
    # 파일도 없고 텍스트 입력도 없는 경우
    pass


# --- 분석 시작 버튼 및 로직 (raw_sentence_input이 채워졌을 때만 활성화되도록) ---
if raw_sentence_input and raw_sentence_input.strip(): # raw_sentence_input이 None이 아니고, 공백만 있는 문자열이 아닐 때
    if st.button("분석 시작 ✨"):
        with st.spinner('텍스트를 분석 중입니다... 잠시만 기다려주세요 (KoNLPy 첫 실행 시 시간이 더 걸릴 수 있습니다) ⏳'):
            # (이하 분석 로직은 이전과 거의 동일. raw_sentence_input을 사용)
            # 1. 전체 문서에서 의미 있는 명사 추출 (키워드 분석용)
            all_document_nouns = extract_meaningful_nouns(raw_sentence_input)

            if not all_document_nouns:
                st.error("분석할 의미 있는 명사가 없습니다. 입력 내용을 확인하거나 불용어 설정을 점검해주세요.")
            else:
                st.subheader("🔑 주요 키워드 (명사, 빈도순)")
                keywords, keyword_counts = get_keywords_from_nouns(all_document_nouns)
                
                if not keywords:
                    st.warning("키워드를 추출하지 못했습니다.")
                else:
                    keyword_df = pd.DataFrame({'키워드': keywords, '빈도수': keyword_counts})
                    st.dataframe(keyword_df.head(15))

                    # 2. Word2Vec 모델 학습을 위한 문장 단위 명사 리스트 준비
                    raw_sentences = re.split(r'(?<=[.?!])\s+', raw_sentence_input.strip())
                    sentences_for_w2v = []
                    original_sentences_for_display = []

                    for sentence_text in raw_sentences:
                        sentence_text_cleaned = sentence_text.strip()
                        if sentence_text_cleaned:
                            sentence_nouns = extract_meaningful_nouns(sentence_text_cleaned)
                            if sentence_nouns:
                                sentences_for_w2v.append(sentence_nouns)
                                original_sentences_for_display.append(sentence_text_cleaned)
                    
                    if not sentences_for_w2v or len(sentences_for_w2v) < 1 :
                        st.error("Word2Vec 모델 학습을 위한 문장(명사 기반) 데이터가 부족합니다.")
                    else:
                        try:
                            model = Word2Vec(sentences_for_w2v, vector_size=100, window=5, min_count=MIN_WORD_COUNT_FOR_W2V, workers=4, sg=1)
                            st.success("Word2Vec 모델 학습 완료! (문장 내 명사 기반)")

                            # 3. 주요 키워드와 유사한 단어 찾기
                            st.subheader("🔗 주요 키워드와 유사한 단어 (Word2Vec)")
                            # ... (이전 유사 단어 찾기 로직과 동일) ...
                            num_similar_words_to_show = 5
                            displayed_similar_count = 0
                            for keyword_to_check in keywords[:10]: 
                                if displayed_similar_count >= 5: 
                                    break
                                if keyword_to_check in model.wv:
                                    similar_words = model.wv.most_similar(keyword_to_check, topn=num_similar_words_to_show)
                                    st.write(f"**'{keyword_to_check}'**와 유사한 단어:")
                                    st.write([f"{word} (유사도: {similarity:.2f})" for word, similarity in similar_words])
                                    displayed_similar_count +=1
                            if displayed_similar_count == 0:
                                st.info("주요 키워드에 대한 유사 단어를 모델에서 찾을 수 없었습니다.")

                            # 4. 키워드와 연관성 높은 문장 찾기
                            st.subheader("📜 키워드와 연관성 높은 문장")
                            # ... (이전 연관 문장 찾기 로직과 동일) ...
                            num_top_sentences = 3 
                            displayed_sentence_count = 0
                            for i in range(min(len(keywords), 10)): 
                                if displayed_sentence_count >= 5: 
                                    break
                                main_keyword = keywords[i]
                                if main_keyword not in model.wv:
                                    continue

                                sentence_similarities = []
                                for idx, sentence_nouns in enumerate(sentences_for_w2v): 
                                    if not sentence_nouns: 
                                        continue
                                    vectors = [model.wv[token] for token in sentence_nouns if token in model.wv]
                                    if not vectors:
                                        continue
                                    sentence_vector = np.mean(vectors, axis=0)
                                    keyword_vector = model.wv[main_keyword]
                                    similarity_score = cosine_similarity([sentence_vector], [keyword_vector])[0][0]
                                    if idx < len(original_sentences_for_display):
                                      sentence_similarities.append({
                                          'sentence': original_sentences_for_display[idx],
                                          'similarity': similarity_score
                                      })
                                if sentence_similarities:
                                    st.markdown(f"--- \n#### '{main_keyword}' 관련 문장:")
                                    sorted_sentences = sorted(sentence_similarities, key=lambda x: x['similarity'], reverse=True)
                                    for item in sorted_sentences[:num_top_sentences]:
                                        st.markdown(f"> {item['sentence']} *(유사도: {item['similarity']:.3f})*")
                                    displayed_sentence_count +=1
                            if displayed_sentence_count == 0:
                                st.info("주요 키워드에 대한 연관 문장을 찾을 수 없었습니다.")
                        except Exception as e:
                            st.error(f"Word2Vec 모델 학습 또는 유사도 분석 중 오류 발생: {e}")
                            # ... (오류 메시지)
else:
    if not uploaded_pdf_file and not raw_sentence_input_area.strip(): # 아무것도 입력되지 않았을 때 안내
        st.info("생기부 PDF 파일을 업로드하거나, 텍스트를 직접 입력해주세요.")


# --- 사이드바 (이전과 동일) ---
st.sidebar.header("ℹ️ 사용 방법")
st.sidebar.markdown("""
1.  **생기부 데이터 입력:**
    * **PDF 파일 업로드:** 'PDF 파일 업로드' 섹션에서 파일을 선택합니다. (권장)
    * **텍스트 직접 입력:** PDF가 없을 경우, 아래 텍스트 영역에 내용을 붙여넣습니다.
2.  **분석 시작:** '분석 시작 ✨' 버튼을 클릭합니다. (데이터가 입력되면 버튼이 나타납니다.)
3.  **결과 확인:**
    * **주요 키워드**, **유사 단어**, **연관성 높은 문장**을 확인합니다.

**팁:**
* PDF 파일은 텍스트 기반이어야 정확한 분석이 가능합니다. (이미지 스캔 PDF는 지원 X)
* 불용어 목록은 앱 코드 내에서 직접 수정하여 분석의 질을 높일 수 있습니다.
""")
# ... (나머지 사이드바 내용)
st.sidebar.header("⚙️ 설정값 정보")
st.sidebar.markdown(f"""
-   추출 명사 최소 길이: `{MIN_NOUN_LEN}`
-   Word2Vec 최소 단어 빈도: `{MIN_WORD_COUNT_FOR_W2V}`
""")
st.sidebar.markdown("---")
st.sidebar.caption("Made with Streamlit, KoNLPy & PyMuPDF")
