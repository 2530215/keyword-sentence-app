from collections import Counter
import streamlit as st
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from konlpy.tag import Okt
import fitz
from github import Github

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
    '상원','고등학교','상원고등학교','번호','표현','설명','표현'
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
st.markdown("[정부24에서 생기부 pdf 다운받는법](https://blog.naver.com/leeyju4/223208661500)", unsafe_allow_html=True)
st.markdown("[카카오톡에서 생기부 pdf 다운받는법](https://blog.naver.com/needtime0514/223256443411)", unsafe_allow_html=True)

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

if raw_sentence_input and raw_sentence_input.strip():
    if st.button("분석 시작 ✨"):
        with st.spinner('텍스트를 분석 중입니다... (KoNLPy/Word2Vec 첫 실행 시 시간이 더 걸릴 수 있습니다) ⏳'):
            all_document_nouns = extract_meaningful_nouns(raw_sentence_input)

            if not all_document_nouns:
                st.error("분석할 의미 있는 명사가 없습니다. 입력 내용을 확인하거나 불용어 설정을 점검해주세요.")
            else:
                # --- 1. 빈도수 기반 키워드 표시 (기존 방식) ---
                st.subheader("🔑 주요 키워드 (단순 빈도수 기반)")
                keywords_freq, keyword_counts_freq = get_keywords_from_nouns_by_freq(all_document_nouns)
                if not keywords_freq:
                    st.warning("빈도수 기반 키워드를 추출하지 못했습니다.")
                else:
                    keyword_df_freq = pd.DataFrame({'키워드': keywords_freq, '빈도수': keyword_counts_freq})
                    st.dataframe(keyword_df_freq.head(10)) # 상위 10개 표시

                # --- Word2Vec 모델 학습 (이전과 동일) ---
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
                
                if not sentences_for_w2v or len(sentences_for_w2v) < 1:
                    st.error("Word2Vec 모델 학습을 위한 문장(명사 기반) 데이터가 부족합니다.")
                    # 의미 기반 키워드 추출은 Word2Vec 모델이 필요하므로 여기서 중단될 수 있음
                    model = None # 모델이 없음을 명시
                else:
                    try:
                        model = Word2Vec(sentences_for_w2v, vector_size=100, window=5, min_count=MIN_WORD_COUNT_FOR_W2V, workers=4, sg=1)
                        st.success("Word2Vec 모델 학습 완료!")
                    except Exception as e:
                        st.error(f"Word2Vec 모델 학습 중 오류 발생: {e}")
                        model = None # 모델 학습 실패

                # --- 2. 의미 기반 키워드 추출 (문서 벡터와 유사도) ---
                if model: # Word2Vec 모델이 성공적으로 학습된 경우에만 실행
                    st.subheader("🌟 주요 키워드 (문서 전체 의미 기반)")
                    
                    # 2a. 문서 대표 벡터 계산
                    doc_vector_sum = np.zeros(model.vector_size)
                    word_count_for_doc_vector = 0
                    # all_document_nouns 중에서 모델 어휘에 있는 단어들만 사용
                    valid_nouns_for_doc_vector = [noun for noun in all_document_nouns if noun in model.wv]
                    
                    if not valid_nouns_for_doc_vector:
                        st.warning("문서 대표 벡터를 계산할 단어가 모델에 없습니다.")
                    else:
                        for word in valid_nouns_for_doc_vector:
                            doc_vector_sum += model.wv[word]
                            word_count_for_doc_vector += 1
                        
                        if word_count_for_doc_vector > 0:
                            document_vector = doc_vector_sum / word_count_for_doc_vector
                            
                            # 2b. 각 고유 명사와 문서 대표 벡터 간 유사도 계산
                            # 키워드 후보는 all_document_nouns의 고유한 명사들 중 모델에 있는 것들
                            candidate_keywords = sorted(list(set(valid_nouns_for_doc_vector))) # 고유 명사 정렬
                            
                            keyword_similarities_to_doc = []
                            for keyword_candidate in candidate_keywords:
                                try:
                                    similarity = cosine_similarity([model.wv[keyword_candidate]], [document_vector])[0][0]
                                    keyword_similarities_to_doc.append((keyword_candidate, similarity))
                                except KeyError:
                                    # 이론상 valid_nouns_for_doc_vector에 있으므로 이 에러는 안나야 함
                                    continue 
                            
                            # 2c. 유사도 높은 순으로 정렬 및 표시
                            if keyword_similarities_to_doc:
                                sorted_keywords_by_meaning = sorted(keyword_similarities_to_doc, key=lambda item: item[1], reverse=True)
                                
                                keywords_meaning = [item[0] for item in sorted_keywords_by_meaning]
                                keyword_scores_meaning = [item[1] for item in sorted_keywords_by_meaning]
                                
                                keyword_df_meaning = pd.DataFrame({'키워드': keywords_meaning, '문서 대표 벡터와의 유사도': keyword_scores_meaning})
                                st.dataframe(keyword_df_meaning.head(15)) # 상위 15개 표시
                            else:
                                st.warning("의미 기반 키워드를 추출하지 못했습니다.")
                        else:
                             st.warning("문서 대표 벡터 계산에 사용될 유효한 단어가 없습니다.")
                else:
                    st.warning("Word2Vec 모델이 없어 의미 기반 키워드를 추출할 수 없습니다.")

                # --- 3. 주요 키워드와 유사한 단어 찾기 (Word2Vec) ---
                if model:
                    st.subheader("🔗 유사 단어 (Word2Vec)")
                    # 유사 단어 찾기의 대상 키워드는 빈도수 기반(keywords_freq) 또는 의미 기반(keywords_meaning) 중 선택 가능
                    # 여기서는 빈도수 기반 상위 키워드를 사용
                    target_keywords_for_similar = keywords_freq 
                    displayed_similar_count = 0
                    for keyword_to_check in target_keywords_for_similar[:10]:
                        if displayed_similar_count >= 5: break
                        if keyword_to_check in model.wv:
                            similar_words = model.wv.most_similar(keyword_to_check, topn=5)
                            st.write(f"**'{keyword_to_check}'**와 유사한 단어:")
                            st.write([f"{word} (유사도: {similarity:.2f})" for word, similarity in similar_words])
                            displayed_similar_count += 1
                    if displayed_similar_count == 0:
                        st.info("주요 키워드에 대한 유사 단어를 모델에서 찾을 수 없었습니다.")

                # --- 4. 키워드와 연관성 높은 문장 찾기 ---
                if model:
                    st.subheader("📜 연관성 높은 문장")
                    # 연관 문장 찾기의 대상 키워드도 빈도수 기반(keywords_freq) 또는 의미 기반(keywords_meaning) 중 선택
                    target_keywords_for_sentence = keywords_freq
                    # ... (이하 연관 문장 찾기 로직은 이전과 동일, target_keywords_for_sentence 사용) ...
                    num_top_sentences = 3
                    displayed_sentence_count = 0
                    for i in range(min(len(target_keywords_for_sentence), 10)):
                        if displayed_sentence_count >= 5: break
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
                            for item in sorted_sentences[:num_top_sentences]:
                                st.markdown(f"> {item['sentence']} *(유사도: {item['similarity']:.3f})*")
                            displayed_sentence_count +=1
                    if displayed_sentence_count == 0:
                        st.info("주요 키워드에 대한 연관 문장을 찾을 수 없었습니다.")
else:
    if not uploaded_pdf_file and not raw_sentence_input_area.strip():
        st.info("생기부 PDF 파일을 업로드하거나, 텍스트를 직접 입력해주세요.")

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
st.markdown("프로그램 사용 경험에 대한 소중한 의견을 남겨주세요! 버그 리포트, 개선 아이디어, 칭찬 모두 환영합니다. 😊")

# 사용자 이름 또는 닉네임 입력 (선택 사항) - 상품 증정용으로 유지
user_info = st.text_input(
    "학번+이름 (선택 사항):",
    placeholder="추후 상품 증정에 사용됩니다. 그 외 사용X",
    key="user_info_input_v3" # 이전 key와 다르게 하거나, 동일하게 사용해도 무방 (단, session_state 초기화 시 주의)
)

# 피드백 내용 입력
feedback_text = st.text_area(
    "피드백 내용:",
    placeholder="여기에 자세한 내용을 작성해주세요.",
    height=150,
    key="feedback_text_area_v3" # 이전 key와 다르게 하거나, 동일하게 사용해도 무방
)

# 제출 버튼
submit_button = st.button("피드백 제출하기", key="feedback_submit_button_v3")

# --- 피드백 처리 로직 (GitHub Issue 생성) ---
if submit_button:
    if feedback_text.strip(): # 피드백 내용이 비어있지 않다면
        try:
            # Streamlit Secrets에서 토큰 및 저장소 정보 가져오기
            gh_token = st.secrets.get("GITHUB_TOKEN")
            repo_name = st.secrets.get("GITHUB_REPO")

            if not gh_token or not repo_name:
                st.error("GitHub 토큰 또는 저장소 정보가 설정되지 않았습니다. 앱 관리자에게 문의하세요.")
            else:
                g = Github(gh_token)
                repo = g.get_repo(repo_name)

                # 이슈 제목 및 본문 구성 (피드백 유형 제거)
                submitter_id_for_title = user_info.strip() if user_info.strip() else "익명 사용자"
                # 피드백 내용의 일부를 제목에 포함시키거나, 단순히 "피드백 제출" 등으로 할 수 있습니다.
                # 여기서는 제출자 정보만으로 제목을 구성합니다.
                issue_title = f"피드백 제출: {submitter_id_for_title}"

                issue_body = f"""
**제출자 정보 (상품 증정용, 선택 사항):** {user_info.strip() if user_info.strip() else "미입력"}
---
**내용:**
{feedback_text}
"""
                # 이슈 생성
                created_issue = repo.create_issue(title=issue_title, body=issue_body)
                st.success("소중한 피드백이 성공적으로 제출되었습니다! 감사합니다.")
                st.markdown(f"제출된 내용은 [여기]({created_issue.html_url})에서 (개발자가) 확인할 수 있습니다.")
                st.info("피드백 내용은 GitHub 저장소의 'Issues' 탭에 기록됩니다.")

        except Exception as e:
            st.error(f"피드백 제출 중 오류가 발생했습니다: {e}")
            st.error("잠시 후 다시 시도해주세요. 문제가 지속되면 앱 관리자에게 알려주세요.")
    else: # 피드백 내용이 비어있다면
        st.error("피드백 내용을 작성해주세요! 😅")

st.sidebar.markdown("---")
st.sidebar.caption("Made with Streamlit, KoNLPy, PyMuPDF & Word2Vec")

st.sidebar.markdown("---")
st.sidebar.caption("Made with Streamlit, KoNLPy & PyMuPDF")
