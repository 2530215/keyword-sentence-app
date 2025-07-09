import streamlit as st
import re
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# --- NLTK 데이터 다운로드 (최초 실행 시) ---
# Streamlit 앱에서는 이 함수를 통해 필요한 NLTK 데이터를 자동으로 다운로드합니다.
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()

# --- 초기 설정 및 상수 정의 ---

# Streamlit 페이지 설정
st.set_page_config(page_title="영어 지문 상세 분석 엔진", layout="wide")

# 영어 불용어 리스트
STOPWORDS = set(stopwords.words('english'))
# 추가적으로 제외하고 싶은 단어가 있다면 여기에 추가
# STOPWORDS.update(['student', 'school', 'teacher']) 

MIN_WORD_LEN = 2 # 추출할 단어의 최소 길이
MIN_WORD_COUNT_FOR_W2V = 1 # Word2Vec 학습을 위한 단어의 최소 빈도

# 영어 연결어 및 전환어 사전
CONNECTORS = {
    'Contrast': ['however', 'but', 'in contrast', 'on the other hand', 'conversely', 'nevertheless'],
    'Result': ['therefore', 'as a result', 'consequently', 'thus', 'hence', 'accordingly'],
    'Example': ['for example', 'for instance', 'to illustrate', 'specifically'],
    'Addition': ['and', 'also', 'moreover', 'furthermore', 'in addition', 'besides'],
    'Sequence': ['first', 'second', 'next', 'then', 'finally', 'afterward', 'subsequently']
}

# --- 헬퍼 함수 정의 ---

def extract_text_from_pdf(uploaded_file):
    """PDF 파일에서 텍스트를 추출합니다."""
    text = ""
    try:
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
    except Exception as e:
        st.error(f"PDF 처리 중 오류 발생: {e}")
        return None
    return text

def preprocess_text_english(text):
    """입력된 영어 텍스트를 전처리하여 문장 리스트와 각 문장의 핵심 단어(표제어) 리스트를 반환합니다."""
    lemmatizer = WordNetLemmatizer()
    
    # 1. 문장 분리
    sentences = sent_tokenize(text)
    
    sentence_words_list = []
    for sentence in sentences:
        # 소문자 변환 및 특수문자 제거 (알파벳, 공백, 기본 구두점만 남김)
        cleaned_sentence = re.sub(r"[^a-zA-Z\s]", "", sentence.lower())
        
        # 단어 토큰화
        words = word_tokenize(cleaned_sentence)
        
        # 품사 태깅
        tagged_words = pos_tag(words)
        
        meaningful_words = []
        for word, tag in tagged_words:
            # 품사가 명사(NN), 동사(VB), 형용사(JJ)인 단어만 선택
            if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ'):
                # 불용어가 아니고, 길이가 최소 길이 이상인 단어만
                if word not in STOPWORDS and len(word) >= MIN_WORD_LEN:
                    # 표제어 추출 (예: running -> run, books -> book)
                    lemmatized_word = lemmatizer.lemmatize(word)
                    meaningful_words.append(lemmatized_word)
        
        sentence_words_list.append(meaningful_words)
        
    return sentences, sentence_words_list

def train_word2vec_model(sentence_words_list):
    """단어 리스트로 Word2Vec 모델을 학습시킵니다."""
    if not sentence_words_list or len(sentence_words_list) < 1:
        return None
    try:
        model = Word2Vec(sentences=sentence_words_list, vector_size=100, window=5, min_count=MIN_WORD_COUNT_FOR_W2V, workers=4, sg=1)
        return model
    except Exception as e:
        st.error(f"Word2Vec 모델 학습 중 오류 발생: {e}")
        return None

# --- [Part 1 & 2] 핵심 분석 엔진 함수 ---
# (이 부분의 로직은 이전과 거의 동일하며, 입력 데이터만 영어용으로 바뀜)
def perform_full_analysis(sentences, sentence_words_list, model):
    """모든 분석을 수행하고 결과를 담은 딕셔너리를 반환합니다."""
    analysis_report = {}
    if not model:
        return {"error": "Word2Vec 모델이 생성되지 않았습니다."}

    # [Part 1] 의미 벡터화
    analysis_report['word_list'] = sorted(list(model.wv.index_to_key))
    analysis_report['word_vector_list'] = [model.wv[word] for word in analysis_report['word_list']]
    analysis_report['sentence_list'] = sentences
    
    sentence_vectors = []
    for words in sentence_words_list:
        vectors = [model.wv[word] for word in words if word in model.wv]
        if vectors:
            sentence_vectors.append(np.mean(vectors, axis=0))
        else:
            sentence_vectors.append(np.zeros(model.vector_size))
    analysis_report['sentence_vector_list'] = sentence_vectors

    all_words = [word for sublist in sentence_words_list for word in sublist]
    valid_vectors = [model.wv[word] for word in all_words if word in model.wv]
    document_vector = np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(model.vector_size)
    analysis_report['document_vector'] = document_vector

    # [Part 2] 심층 내용 분석
    word_sims = cosine_similarity(model.wv.vectors, [document_vector])
    word_sim_pairs = list(zip(model.wv.index_to_key, word_sims.flatten()))
    analysis_report['important_word_list'] = sorted(word_sim_pairs, key=lambda item: item[1], reverse=True)

    sent_sims = cosine_similarity(sentence_vectors, [document_vector])
    sent_sim_pairs = list(zip(sentences, sent_sims.flatten()))
    analysis_report['important_sentence_list'] = sorted(sent_sim_pairs, key=lambda item: item[1], reverse=True)

    vocab_analysis = {}
    top_keywords = [word for word, sim in analysis_report['important_word_list'][:5]]
    for keyword in top_keywords:
        if keyword in model.wv:
            synonyms = model.wv.most_similar(keyword, topn=5)
            vocab_analysis[keyword] = {"Synonyms": synonyms}
    analysis_report['vocabulary_analysis'] = vocab_analysis
    
    if len(sentence_vectors) > 3:
        adj_sent_sims = [cosine_similarity([sentence_vectors[i]], [sentence_vectors[i+1]])[0][0] for i in range(len(sentence_vectors) - 1)]
        split_indices = sorted(range(len(adj_sent_sims)), key=lambda i: adj_sent_sims[i])[:2]
        split_indices.sort()
        
        paragraphs, last_split = [], 0
        for idx in split_indices:
            paragraphs.append(sentences[last_split : idx + 1])
            last_split = idx + 1
        paragraphs.append(sentences[last_split:])
        analysis_report['reconstructed_paragraphs'] = paragraphs
    else:
        analysis_report['reconstructed_paragraphs'] = [sentences]

    found_connectors = []
    for i, sentence in enumerate(sentences):
        for conn_type, conn_list in CONNECTORS.items():
            for conn_word in conn_list:
                # 단어 경계를 확인하기 위해 정규표현식 사용 (예: 'and'가 'sand'의 일부로 인식되는 것 방지)
                if re.search(r'\b' + conn_word + r'\b', sentence.lower()):
                    found_connectors.append({"Sentence No.": i + 1, "Sentence": sentence, "Connector": conn_word, "Function": conn_type})
    analysis_report['syntax_analysis'] = {"connectors": found_connectors}

    return analysis_report

# --- [Part 3] 결과 출력 함수 ---
# (출력 부분은 한글로 유지)
def display_report(report):
    """분석 리포트 딕셔너리를 받아 Streamlit UI에 체계적으로 출력합니다."""
    st.header("📊 지문 상세 분석 결과 리포트")
    
    if "error" in report:
        st.error(report["error"])
        return

    with st.expander("🌟 핵심 내용 요약", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("핵심 단어 (Top 10)")
            df_words = pd.DataFrame(report['important_word_list'][:10], columns=['단어 (Word)', '문서 전체와의 관련도 (Relevance)'])
            st.dataframe(df_words, use_container_width=True)
        with col2:
            st.subheader("핵심 문장 (Top 5)")
            for sentence, similarity in report['important_sentence_list'][:5]:
                st.markdown(f"> {sentence} *(관련도: {similarity:.3f})*")

    with st.expander("📑 구조 및 구문 분석", expanded=True):
        st.subheader("논리적 문단 재구성")
        st.info("문장 간 의미적 유사도를 분석하여 내용 흐름이 바뀌는 지점을 기준으로 문단을 재구성합니다.")
        for i, paragraph_sentences in enumerate(report['reconstructed_paragraphs']):
            st.markdown(f"**- 문단 {i+1} -**")
            st.write(" ".join(paragraph_sentences))
            st.markdown("---")
        
        st.subheader("연결어 및 전환어 식별")
        if report['syntax_analysis']['connectors']:
            df_connectors = pd.DataFrame(report['syntax_analysis']['connectors'])
            st.dataframe(df_connectors, use_container_width=True)
        else:
            st.info("분석 가능한 연결어가 발견되지 않았습니다.")

    with st.expander("🔎 상세 어휘 분석"):
        st.info("문서의 핵심 단어와 의미적으로 가장 유사한 단어(유의어)를 보여줍니다.")
        for keyword, analysis in report['vocabulary_analysis'].items():
            st.subheader(f"'{keyword}'의 분석 결과")
            synonyms_df = pd.DataFrame(analysis['Synonyms'], columns=['유사 단어 (Similar Word)', '유사도 (Similarity)'])
            st.dataframe(synonyms_df)

    with st.expander("🔬 Raw 데이터 및 벡터값 보기"):
        # ... (이전과 동일한 Raw 데이터 출력 로직) ...
        pass

# --- 메인 UI 로직 ---
def main():
    st.title("📝 영어 지문 상세 분석 엔진")
    st.markdown("사용자가 입력한 **영어 텍스트(지문)**를 다각도로 분석하여 **핵심 내용, 구조, 어휘**를 포함한 상세 리포트를 생성합니다.")

    input_method = st.radio("입력 방식 선택", ('텍스트 직접 입력', 'PDF 파일 업로드'))
    
    raw_text_input = ""
    if input_method == 'PDF 파일 업로드':
        uploaded_file = st.file_uploader("분석할 PDF 파일을 업로드하세요.", type="pdf")
        if uploaded_file:
            with st.spinner("PDF 파일에서 텍스트를 추출하는 중..."):
                raw_text_input = extract_text_from_pdf(uploaded_file)
    else:
        raw_text_input = st.text_area("분석할 영어 지문을 여기에 직접 붙여넣으세요.", height=250)

    if raw_text_input and raw_text_input.strip():
        if st.button("분석 시작 ✨", type="primary"):
            with st.spinner('텍스트를 분석 중입니다... (NLTK 데이터 다운로드로 첫 실행 시 시간이 더 걸릴 수 있습니다) ⏳'):
                sentences, sentence_words_list = preprocess_text_english(raw_text_input)
                
                if not any(sentence_words_list):
                    st.error("분석할 의미 있는 단어가 부족합니다. 텍스트 내용을 확인해주세요.")
                    return

                model = train_word2vec_model(sentence_words_list)

                if model:
                    analysis_report = perform_full_analysis(sentences, sentence_words_list, model)
                    display_report(analysis_report)
                else:
                    st.error("데이터가 부족하여 분석 모델을 생성할 수 없습니다.")
    else:
        st.info("분석할 텍스트를 입력하거나 PDF 파일을 업로드해주세요.")

    st.sidebar.header("ℹ️ 프로그램 원리 (영어)")
    st.sidebar.markdown("""
    1.  **입력 전처리**: `NLTK`를 사용하여 텍스트를 문장/단어로 나누고, 불용어를 제거한 뒤 품사(명사, 동사, 형용사)를 기준으로 핵심 단어를 선별합니다. 단어는 기본형(표제어)으로 변환됩니다.
    2.  **의미 벡터화**: 전처리된 단어들을 `Word2Vec` 모델로 학습시켜 각 단어를 벡터로 변환합니다.
    3.  **유사도 분석**: 벡터 간 '코사인 유사도'를 계산하여 의미적 관련성을 파악합니다.
    4.  **핵심 내용/구조 분석**: 이 유사도를 기반으로 핵심 단어/문장 추출, 문단 분할 등의 분석을 수행합니다.
    """)
    st.sidebar.markdown("---")
    st.sidebar.caption("Made with Streamlit, Gensim & NLTK")

if __name__ == "__main__":
    main()
