from collections import Counter
import streamlit as st
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from konlpy.tag import Okt # KoNLPy Okt 추가

# --- Okt 형태소 분석기 초기화 ---
okt = Okt()

# --- 불용어 및 기본 설정 ---
# 불용어 목록 (생기부 내용에 맞춰 계속 추가/수정하시는 것이 좋습니다)
STOPWORDS = [
    # --- 기존에 사용하시던 목록 ---
    '수', '것', '때', '등', '이', '그', '저', '년', '월', '일', '좀', '중', '위해', '및',
    '그것', '이것', '저것', '여기', '저기', '거기', '자신', '자체', '대한', '통해', '관련',
    '여러', '가지', '다른', '부분', '경우', '정도', '사이', '문제', '내용', '결과', '과정',
    '사용', '생각', '지금', '현재', '당시', '때문에', '면서', '동안', '위한', '따라',
    '대해', '통한', '관련된', '있음', '없음', '같음', '사항', '활동', '모습', '분야', # '모습' 중복 제거됨
    '능력', # '모습'은 위에서 이미 처리, '역량', '자세', '태도', '노력', '바탕', '역할', '학습', '이해'는 아래 목록과 중복될 수 있어 확인
    '항상', '매우', '다소', '특히', '가장', '더욱', '적극적', '구체적', '다양한', '꾸준히',
    '뛰어남', '우수함', '보임', '발휘함', '참여함', '탐구함', '발전함', '향상됨', '함양함', # '참여함', '탐구함', '발전함', '향상됨' 등은 아래 목록과 유사/중복 가능
    '만듦', '발표함', '제시함', '제출함', '바', '점', '측면', '과제', '조사', '주제', # '과제', '조사', '주제', '발표함' 등은 아래 목록과 중복 가능
    '자료', '발표', '토론', '보고서', '탐구', '연구', '프로젝트', '실험', '수업', '시간', # '발표', '토론', '보고서', '탐구', '연구', '프로젝트', '실험', '수업', '시간' 등은 아래 목록과 중복 가능
    '이용', '참여', # '참여'는 아래 목록과 중복

    # --- 제가 제안드린 확장 목록 (위 목록과 중복되는 부분은 통합/정리) ---
    # 1. 일반적인 불용어 (확장)
    '고', '한', '터', '이후', '이전', '내', '외', '속',
    '열심히', # '매우', '다소', '특히', '가장', '더욱', '항상', '꾸준히'는 기존 목록에 있음
    '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉', '열', # 기존 목록에 숫자 없음
    '첫째', '둘째', '셋째', '다음', '먼저', '비롯', '비롯한', '등등', '기타',

    # 2. 생기부 특화 동사성 명사 및 일반 명사
    '활용', '실시', '진행', '수행', '제작', '경험', # '참여', '이용', '노력', '학습', '이해', '탐구', '연구', '조사', '발표', '토론', '보고서', '과제', '주제', '자료', '수업', '시간', '프로젝트', '실험' 등은 기존 목록과 중복 또는 유사하여 포함됨
    '관찰', '기록', '정리',
    '역량', '자세', '태도', '바탕', '역할', '기반', '향상', '발전', '성장', # '능력'은 기존 목록에 있음
    '수준', '관심', '흥미', '호기심', '질문', '제안', # '점', '측면'은 기존 목록에 있음. '제시'는 '제시함'과 유사
    '해결', '도움', '협력', '소통', '관계', '중심', '대상', '방법', '원리', '개념',
    '의미', '중요성', '필요성', '가치', '다양성', '창의성', '적극성', '성실성', '책임감', # '적극성'은 '적극적'과 유사
    '자기주도', '모범', '리더십', '팔로우십', '공동체', '배려', '나눔', '봉사',
    '교과', '과목', '단원', '영역', # '분야'는 기존 목록에 있음
    '학기', '학년', '학교', '교내', '교외',
    '대회', '행사', '캠프', '동아리', '부서', '조직', '단체', '기관', '시설',
    '학생', '교사', '친구', '우리', '모둠', '팀', # '자신'은 기존 목록에 있음
    '시작', '마무리', '완성', # '제출'은 기존 목록에 있음
    '향상됨', '발전함', '성장함', '노력함', '참여함', '탐구함', '연구함', '발표함', # 기존 목록과 중복되는 동사 파생 명사 정리
    '드러냄', '갖춤', '지님', '인정됨', '확인됨', '관찰됨', # '보임'은 기존 목록에 있음
    '우수', '뛰어남', '탁월', '미흡', '부족', # '우수함', '뛰어남'은 기존 목록과 유사
    '관련하여', '대하여', '바탕으로', '중심으로', '통하여', '비추어', '앞서',
    '기록함', '기재함', '작성함',

    # 3. 생기부 서술어에서 파생된 명사 (형식적인 표현)
    '됨', '함', '높음', '낮음', '많음', '적음', # '있음', '없음', '보임', '인정됨', '확인됨', '관찰됨' 등은 기존 또는 위 목록과 중복
    '기대됨', '요망됨'
]
MIN_NOUN_LEN = 2 # 추출할 명사의 최소 길이 (한 글자 명사 제외)
MIN_WORD_COUNT_FOR_W2V = 1 # Word2Vec 학습 시 단어의 최소 등장 빈도

# --- 새로운 전처리 함수 (KoNLPy Okt 사용) ---
def extract_meaningful_nouns(text):
    """
    KoNLPy Okt를 사용하여 텍스트에서 의미 있는 명사를 추출합니다.
    특수문자 제거, 불용어 처리, 짧은 단어/숫자형 단어 필터링을 수행합니다.
    """
    # 1. 기본적인 특수문자 및 공백 정리
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.]+", "", str(text)).strip() # 마침표는 문장 분리 위해 유지 시도
    text = re.sub(r"\s+", " ", text) # 중복 공백 제거

    if not text:
        return []

    # 2. Okt를 사용한 명사 추출
    nouns = okt.nouns(text)

    # 3. 필터링
    meaningful_nouns = []
    for noun in nouns:
        if (
            noun not in STOPWORDS
            and len(noun) >= MIN_NOUN_LEN
            and not noun.isnumeric() # 숫자만으로 이루어진 단어 제외
        ):
            meaningful_nouns.append(noun)
    return meaningful_nouns

# --- 기존 키워드 추출 함수 (Counter 사용) ---
def get_keywords_from_nouns(noun_list):
    """명사 리스트에서 빈도수 기반으로 키워드와 빈도수를 정렬하여 반환합니다."""
    if not noun_list:
        return [], []
    word_counts = Counter(noun_list)
    # most_common()은 (단어, 빈도수) 튜플의 리스트를 반환
    sorted_keywords_with_counts = word_counts.most_common()
    
    # 키워드 리스트와 빈도수 리스트로 분리
    wordset = [item[0] for item in sorted_keywords_with_counts]
    wordsetcount = [item[1] for item in sorted_keywords_with_counts]
    
    return wordset, wordsetcount


# --- Streamlit UI ---
st.set_page_config(page_title="생기부 분석기", layout="wide") # 페이지 넓게 사용
st.title("📝 생기부 키워드 분석 및 연관 문장 추천")
st.markdown("""
KoNLPy 형태소 분석기를 사용하여 명사 위주로 키워드를 추출하고,
Word2Vec 모델을 통해 유사 단어 및 관련 높은 문장을 찾아줍니다.
""")

raw_sentence_input = st.text_area("분석할 생기부 내용을 입력하세요:", height=250, placeholder="여기에 텍스트를 입력해주세요...")

if st.button("분석 시작 ✨"):
    if raw_sentence_input.strip(): # 입력 내용이 실제로 있는지 확인
        with st.spinner('텍스트를 분석 중입니다... 잠시만 기다려주세요 (KoNLPy 첫 실행 시 시간이 더 걸릴 수 있습니다) ⏳'):
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
                    st.dataframe(keyword_df.head(15)) # 상위 15개 키워드 표시

                    # 2. Word2Vec 모델 학습을 위한 문장 단위 명사 리스트 준비
                    # 마침표, 물음표, 느낌표를 기준으로 문장 분리 (더 정교한 문장 분리 로직이 필요할 수 있음)
                    raw_sentences = re.split(r'(?<=[.?!])\s+', raw_sentence_input.strip()) # 문장 구분자 유지하며 분리
                    
                    sentences_for_w2v = []
                    original_sentences_for_display = [] # 유사 문장 표시 시 원본 문장 사용 위함

                    for sentence_text in raw_sentences:
                        sentence_text_cleaned = sentence_text.strip()
                        if sentence_text_cleaned:
                            sentence_nouns = extract_meaningful_nouns(sentence_text_cleaned)
                            if sentence_nouns: # 명사가 하나라도 있는 문장만 학습에 사용
                                sentences_for_w2v.append(sentence_nouns)
                                original_sentences_for_display.append(sentence_text_cleaned)
                    
                    if not sentences_for_w2v or len(sentences_for_w2v) < 1 : # 학습할 문장이 너무 적은 경우
                        st.error("Word2Vec 모델 학습을 위한 문장(명사 기반) 데이터가 부족합니다.")
                    else:
                        try:
                            model = Word2Vec(sentences_for_w2v, vector_size=100, window=5, min_count=MIN_WORD_COUNT_FOR_W2V, workers=4, sg=1)
                            st.success("Word2Vec 모델 학습 완료! (문장 내 명사 기반)")


                            # 3. 키워드와 연관성 높은 문장 찾기
                            st.subheader("📜 키워드와 연관성 높은 문장")
                            num_top_sentences = 3 # 각 키워드별로 보여줄 상위 문장 수
                            displayed_sentence_count = 0

                            for i in range(min(len(keywords), 10)): # 상위 10개 키워드에 대해 시도
                                if displayed_sentence_count >= 5: # 최대 5개 키워드에 대한 연관문장 표시
                                    break
                                main_keyword = keywords[i]
                                if main_keyword not in model.wv:
                                    # st.write(f"키워드 '{main_keyword}'에 대한 벡터가 없어 문장 유사도를 계산할 수 없습니다.")
                                    continue

                                sentence_similarities = []
                                for idx, sentence_nouns in enumerate(sentences_for_w2v): # 학습에 사용된 명사 리스트 기준
                                    if not sentence_nouns: # 명사가 없는 문장은 건너뜀
                                        continue

                                    vectors = [model.wv[token] for token in sentence_nouns if token in model.wv]
                                    if not vectors:
                                        # 해당 문장의 명사들이 모델 어휘에 없는 경우
                                        continue
                                    
                                    sentence_vector = np.mean(vectors, axis=0)
                                    keyword_vector = model.wv[main_keyword]
                                    
                                    # 코사인 유사도 계산, 1차원 배열로 변환 후 접근
                                    similarity_score = cosine_similarity([sentence_vector], [keyword_vector])[0][0]
                                    
                                    # original_sentences_for_display 에서 원본 문장 가져오기
                                    # sentences_for_w2v 와 original_sentences_for_display는 길이가 다를 수 있으므로 주의.
                                    # 가장 안전한 방법은 sentences_for_w2v 만들 때 원본 문장도 같이 저장하는 것
                                    # 현재 코드는 original_sentences_for_display 와 sentences_for_w2v 를 동기화 시킴
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
                            st.error("입력된 텍스트의 양이 충분한지, 다양한 단어가 포함되어 있는지 확인해주세요.")
                            st.error("특히, Word2Vec 모델은 학습 데이터의 질과 양, 그리고 min_count 설정에 민감합니다.")
        # 분석 완료 후 스피너 자동 종료
    else:
        st.warning("분석할 내용을 입력해주세요.")

# --- 사이드바 ---
st.sidebar.header("ℹ️ 사용 방법")
st.sidebar.markdown("""
1.  **생기부 내용 입력:** 중앙의 텍스트 입력창에 분석하고 싶은 생기부 내용을 복사하여 붙여넣거나 직접 입력합니다.
2.  **분석 시작:** '분석 시작 ✨' 버튼을 클릭합니다.
3.  **결과 확인:**
    * **주요 키워드:** 텍스트에서 자주 등장하는 명사들이 빈도순으로 표시됩니다. (불용어 등 제외)
    * **연관성 높은 문장:** 주요 키워드와 가장 관련성이 높은 문장들이 추천됩니다.

**팁:**
* 더 정확한 분석을 위해 충분한 양의 텍스트를 입력하는 것이 좋습니다.
* **불용어 목록 (`STOPWORDS`)**은 앱 코드 내에서 직접 수정하여 분석의 질을 높일 수 있습니다. (현재는 기본적인 단어들만 포함)
""")
st.sidebar.header("⚙️ 설정값 정보")
st.sidebar.markdown(f"""
-   추출 명사 최소 길이: `{MIN_NOUN_LEN}`
-   Word2Vec 최소 단어 빈도: `{MIN_WORD_COUNT_FOR_W2V}`
""")
st.sidebar.markdown("---")
st.sidebar.caption("Made with Streamlit & KoNLPy")
