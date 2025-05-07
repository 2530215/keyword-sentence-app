from collections import Counter
import streamlit as st
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from konlpy.tag import Okt
import fitz
#from github import Github
import requests
import json

# --- Okt 형태소 분석기 초기화 ---
okt = Okt()

# --- 불용어 및 기본 설정 (사용자님의 최신 불용어 목록 사용) ---
STOPWORDS = [
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
    '상원','고등학교','상원고등학교','번호','표현','설명','이름','성함'
]
MIN_NOUN_LEN = 2
MIN_WORD_COUNT_FOR_W2V = 1

# --- PDF 텍스트 추출 함수 (이전과 동일) ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
    except Exception as e:
        st.error(f"PDF 처리 중 오류 발생: {e}")
        return None
    return text

# --- 명사 추출 함수 (이전과 동일) ---
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

# --- 빈도수 기반 키워드 추출 함수 (이전과 동일) ---
def get_keywords_from_nouns_by_freq(noun_list): # 함수 이름 변경
    if not noun_list: return [], []
    word_counts = Counter(noun_list)
    sorted_keywords_with_counts = word_counts.most_common()
    wordset = [item[0] for item in sorted_keywords_with_counts]
    wordsetcount = [item[1] for item in sorted_keywords_with_counts]
    return wordset, wordsetcount

# --- Streamlit UI (일부만 표시, 핵심 로직 위주) ---
st.set_page_config(page_title="생기부 분석기", layout="wide")
st.title("📝 생기부 키워드 분석 및 연관 문장 추천")
# ... (UI 상단 마크다운, 파일 업로드, 텍스트 입력 부분은 이전과 거의 동일하게 유지) ...
st.markdown("""
KoNLPy 형태소 분석기를 사용하여 명사 위주로 키워드를 추출하고,
Word2Vec 모델을 통해 유사 단어 및 관련 높은 문장을 찾아줍니다.
**PDF 파일을 업로드하거나 텍스트를 직접 입력하여 분석할 수 있습니다.**
""")
st.subheader("1. 분석할 생기부 데이터 입력")
st.markdown("[카카오톡에서 생기부 pdf 다운받는법(권장!!)](https://blog.naver.com/needtime0514/223256443411)", unsafe_allow_html=True)
st.markdown("[정부24에서 생기부 pdf 다운받는법](https://blog.naver.com/leeyju4/223208661500)", unsafe_allow_html=True)

uploaded_pdf_file = st.file_uploader("생기부 PDF 파일 업로드 (PDF 업로드 시 아래 텍스트 입력 내용은 무시됩니다):", type="pdf")
raw_sentence_input_area = st.text_area("또는, 생기부 내용을 여기에 직접 붙여넣으세요:", height=200, placeholder="PDF를 업로드하지 않을 경우 여기에 텍스트를 입력해주세요...")

raw_sentence_input = None
if uploaded_pdf_file is not None:
    with st.spinner("PDF 파일을 읽고 분석 준비 중입니다..."):
        extracted_text_from_pdf = extract_text_from_pdf(uploaded_pdf_file)
        if extracted_text_from_pdf:
            raw_sentence_input = extracted_text_from_pdf
            st.success("PDF 파일에서 텍스트를 성공적으로 추출했습니다!")
        else:
            st.error("PDF에서 텍스트를 추출하지 못했습니다. 파일이 올바른지 확인하거나, 아래 텍스트 영역에 직접 입력해주세요.")
elif raw_sentence_input_area.strip():
    raw_sentence_input = raw_sentence_input_area
else:
    pass

# ... (이전 코드들은 동일하게 유지) ...

if raw_sentence_input and raw_sentence_input.strip():
    if st.button("분석 시작 ✨"):
        with st.spinner('텍스트를 분석 중입니다... (KoNLPy/Word2Vec 첫 실행 시 시간이 더 걸릴 수 있습니다) ⏳'):
            all_document_nouns = extract_meaningful_nouns(raw_sentence_input)

            if not all_document_nouns:
                st.error("분석할 의미 있는 명사가 없습니다. 입력 내용을 확인하거나 불용어 설정을 점검해주세요.")
            else:
                # --- 1. 빈도수 기반 키워드 표시 (기존 방식) ---
                st.subheader("🔑 주요 키워드 (단순 빈도수 기반)")
                keywords_freq_raw, keyword_counts_freq_raw = get_keywords_from_nouns_by_freq(all_document_nouns)
                
                # "상원" 및 기타 명시적으로 제거하고 싶은 단어들 리스트
                explicit_remove_list = ["상원"] # 필요에 따라 추가

                # 빈도수 기반 키워드에서 "상원" 등 제거
                keywords_freq_filtered = []
                keyword_counts_freq_filtered = []
                for kw, count in zip(keywords_freq_raw, keyword_counts_freq_raw):
                    if kw not in explicit_remove_list:
                        keywords_freq_filtered.append(kw)
                        keyword_counts_freq_filtered.append(count)
                
                if not keywords_freq_filtered:
                    st.warning("빈도수 기반 키워드를 추출하지 못했습니다 (필터링 후).")
                else:
                    keyword_df_freq = pd.DataFrame({'키워드': keywords_freq_filtered, '빈도수': keyword_counts_freq_filtered})
                    st.dataframe(keyword_df_freq.head(10))

                # --- Word2Vec 모델 학습 (이전과 동일) ---
                # ... (sentences_for_w2v, model 학습 로직은 그대로) ...
                raw_sentences = re.split(r'(?<=[.?!])\s+', raw_sentence_input.strip())
                sentences_for_w2v = []
                original_sentences_for_display = []
                for sentence_text in raw_sentences:
                    sentence_text_cleaned = sentence_text.strip()
                    if sentence_text_cleaned:
                        # Word2Vec 학습 데이터에는 "상원"이 불용어 처리되어 빠지는 것이 이상적이지만,
                        # 만약 extract_meaningful_nouns에서 여전히 문제가 있다면 여기에도 영향.
                        # 하지만 extract_meaningful_nouns의 STOPWORDS는 계속 유지/개선해야 함.
                        sentence_nouns = extract_meaningful_nouns(sentence_text_cleaned)
                        if sentence_nouns:
                            sentences_for_w2v.append(sentence_nouns)
                            original_sentences_for_display.append(sentence_text_cleaned)
                
                model = None # 초기화
                if not sentences_for_w2v or len(sentences_for_w2v) < 1:
                    st.error("Word2Vec 모델 학습을 위한 문장(명사 기반) 데이터가 부족합니다.")
                else:
                    try:
                        model = Word2Vec(sentences_for_w2v, vector_size=100, window=5, min_count=MIN_WORD_COUNT_FOR_W2V, workers=4, sg=1)
                        st.success("Word2Vec 모델 학습 완료!")
                    except Exception as e:
                        st.error(f"Word2Vec 모델 학습 중 오류 발생: {e}")


                # --- 2. 의미 기반 키워드 추출 (문서 벡터와 유사도) ---
                if model:
                    st.subheader("🌟 주요 키워드 (문서 전체 의미 기반)")
                    doc_vector_sum = np.zeros(model.vector_size)
                    word_count_for_doc_vector = 0
                    valid_nouns_for_doc_vector = [noun for noun in all_document_nouns if noun in model.wv and noun not in explicit_remove_list] # 여기서도 제거
                    
                    if not valid_nouns_for_doc_vector:
                        st.warning("문서 대표 벡터를 계산하거나 의미 기반 키워드를 추출할 단어가 모델에 없습니다 (필터링 후).")
                    else:
                        for word in valid_nouns_for_doc_vector:
                            doc_vector_sum += model.wv[word]
                            word_count_for_doc_vector += 1
                        
                        if word_count_for_doc_vector > 0:
                            document_vector = doc_vector_sum / word_count_for_doc_vector
                            candidate_keywords = sorted(list(set(valid_nouns_for_doc_vector))) # 이미 explicit_remove_list 제외됨
                            
                            keyword_similarities_to_doc = []
                            for keyword_candidate in candidate_keywords: # 이미 explicit_remove_list 제외됨
                                try:
                                    similarity = cosine_similarity([model.wv[keyword_candidate]], [document_vector])[0][0]
                                    keyword_similarities_to_doc.append((keyword_candidate, similarity))
                                except KeyError:
                                    continue 
                            
                            if keyword_similarities_to_doc:
                                sorted_keywords_by_meaning_raw = sorted(keyword_similarities_to_doc, key=lambda item: item[1], reverse=True)
                                
                                # 의미 기반 키워드에서 "상원" 등 제거 (이미 candidate_keywords에서 고려했지만, 한번 더 확인 가능)
                                keywords_meaning_filtered = []
                                keyword_scores_meaning_filtered = []
                                for kw, score in sorted_keywords_by_meaning_raw:
                                    if kw not in explicit_remove_list: # 이중 체크 또는 여기서만 처리
                                        keywords_meaning_filtered.append(kw)
                                        keyword_scores_meaning_filtered.append(score)

                                if not keywords_meaning_filtered:
                                    st.warning("의미 기반 키워드를 추출하지 못했습니다 (필터링 후).")
                                else:
                                    keyword_df_meaning = pd.DataFrame({'키워드': keywords_meaning_filtered, '문서 대표 벡터와의 유사도': keyword_scores_meaning_filtered})
                                    st.dataframe(keyword_df_meaning.head(15))
                            else:
                                st.warning("의미 기반 키워드를 추출하지 못했습니다.")
                        else:
                             st.warning("문서 대표 벡터 계산에 사용될 유효한 단어가 없습니다.")
                else:
                    st.warning("Word2Vec 모델이 없어 의미 기반 키워드를 추출할 수 없습니다.")

                # --- 3. 키워드와 연관성 높은 문장 찾기 ---
                if model:
                    st.subheader("📜 연관성 높은 문장")
                    # 연관 문장 찾기 대상도 필터링된 키워드 사용
                    target_keywords_for_sentence = keywords_freq_filtered # 필터링된 빈도수 기반 키워드 사용
                    # ... (이하 로직 동일, target_keywords_for_sentence 사용) ...
                    displayed_sentence_count = 0
                    for i in range(min(len(target_keywords_for_sentence), 10)):
                        if displayed_sentence_count >= 10: break
                        main_keyword = target_keywords_for_sentence[i]
                        if main_keyword not in model.wv: continue
                        sentence_similarities = []
                        for idx, sentence_nouns in enumerate(sentences_for_w2v):
                            if not sentence_nouns: continue
                            vectors = [model.wv[token] for token in sentence_nouns if token in model.wv]
                            if not vectors: continue
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
                            num_top_sentences = 3
                            for item in sorted_sentences[:num_top_sentences]:
                                st.markdown(f"> {item['sentence']} *(유사도: {item['similarity']:.3f})*")
                            displayed_sentence_count +=1
                    if displayed_sentence_count == 0:
                        st.info("주요 키워드에 대한 연관 문장을 찾을 수 없었습니다.")


# --- 사이드바 (이전과 동일) ---
# ... (사이드바 코드는 그대로 유지) ...
st.sidebar.header("ℹ️ 사용 방법")
st.sidebar.markdown("""
1.  **생기부 데이터 입력:**
    * **PDF 파일 업로드:** 'PDF 파일 업로드' 섹션에서 파일을 선택합니다. (권장)
    * **텍스트 직접 입력:** PDF가 없을 경우, 아래 텍스트 영역에 내용을 붙여넣습니다.
2.  **분석 시작:** '분석 시작 ✨' 버튼을 클릭합니다. (데이터가 입력되면 버튼이 나타납니다.)
3.  **결과 확인:**
    * **주요 키워드 (빈도수 기반)**: 단순히 자주 등장하는 명사입니다.
    * **주요 키워드 (의미 기반)**: 문서 전체의 주제와 관련성이 높은 명사입니다.
    * **유사 단어**, **연관성 높은 문장**을 확인합니다.

**팁:**
* PDF 파일은 텍스트 기반이어야 정확한 분석이 가능합니다. (이미지 스캔 PDF는 지원 X)
* 불용어 목록은 앱 코드 내에서 직접 수정하여 분석의 질을 높일 수 있습니다.
""")
st.sidebar.header("⚙️ 설정값 정보")
st.sidebar.markdown(f"""
-   추출 명사 최소 길이: `{MIN_NOUN_LEN}`
-   Word2Vec 최소 단어 빈도: `{MIN_WORD_COUNT_FOR_W2V}`
""")


st.markdown("---") # 구분선 추가
st.header("📝 프로그램 피드백")
st.markdown("상품을 받으려면 구글 폼 링크를 통해 피드백을 작성해 주세요!")
st.markdown("[구글 폼 링크](https://docs.google.com/forms/d/e/1FAIpQLSdqbJDR3ASS1IXqSh2dyo15xrl08sefT9N3-p7bJ1XzyWhvew/viewform?usp=header)", unsafe_allow_html=True)


st.sidebar.markdown("---")
st.sidebar.caption("Made with Streamlit, KoNLPy, PyMuPDF & Word2Vec")
