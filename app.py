import streamlit as st
import re
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk # NLTK 라이브러리

# --- ================================================================== ---
# ---               오류 해결을 위한 핵심 코드 (NLTK 설정)               ---
# --- =================================----------------================= ---
# Streamlit의 캐시 기능을 사용하여 앱 세션당 딱 한 번만 실행되도록 합니다.
# 이렇게 하면 앱이 시작될 때 필요한 모든 데이터가 준비되었는지 확인하고,
# 없는 경우에만 다운로드하여 LookupError를 원천적으로 방지합니다.
@st.cache_resource
def setup_nltk():
    """
    NLTK의 필수 데이터 패키지를 다운로드하는 함수.
    앱 실행 시 가장 먼저 호출되어야 합니다.
    """
    nltk.download('punkt') # 문장 토큰화(sent_tokenize)에 필요
    nltk.download('stopwords') # 불용어(stopwords)에 필요
    nltk.download('averaged_perceptron_tagger') # 품사 태깅(pos_tag)에 필요
    nltk.download('wordnet') # 표제어 추출(lemmatize)에 필요

# --- 앱 실행 시 가장 먼저 NLTK 설정을 수행 ---
setup_nltk()
# --- ================================================================== ---


# --- 초기 설정 및 상수 정의 ---
st.set_page_config(page_title="영어 지문 상세 분석 엔진", layout="wide")

STOPWORDS = set(stopwords.words('english'))
MIN_WORD_LEN = 2
MIN_WORD_COUNT_FOR_W2V = 1

CONNECTORS = {
    'Contrast': ['however', 'but', 'in contrast', 'on the other hand', 'conversely', 'nevertheless'],
    'Result': ['therefore', 'as a result', 'consequently', 'thus', 'hence', 'accordingly'],
    'Example': ['for example', 'for instance', 'to illustrate', 'specifically'],
    'Addition': ['and', 'also', 'moreover', 'furthermore', 'in addition', 'besides'],
    'Sequence': ['first', 'second', 'next', 'then', 'finally', 'afterward', 'subsequently']
}

# --- 헬퍼 함수 정의 ---

def extract_text_from_pdf(uploaded_file):
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
    lemmatizer = WordNetLemmatizer()
    sentences = nltk.sent_tokenize(text)
    
    sentence_words_list = []
    for sentence in sentences:
        cleaned_sentence = re.sub(r"[^a-zA-Z\s]", "", sentence.lower())
        words = nltk.word_tokenize(cleaned_sentence)
        tagged_words = nltk.pos_tag(words)
        
        meaningful_words = []
        for word, tag in tagged_words:
            if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ'):
                if word not in STOPWORDS and len(word) >= MIN_WORD_LEN:
                    lemmatized_word = lemmatizer.lemmatize(word)
                    meaningful_words.append(lemmatized_word)
        
        sentence_words_list.append(meaningful_words)
        
    return sentences, sentence_words_list

def train_word2vec_model(sentence_words_list):
    if not sentence_words_list or len(sentence_words_list) < 1:
        return None
    try:
        model = Word2Vec(sentences=sentence_words_list, vector_size=100, window=5, min_count=MIN_WORD_COUNT_FOR_W2V, workers=4, sg=1)
        return model
    except Exception as e:
        st.error(f"Word2Vec 모델 학습 중 오류 발생: {e}")
        return None

def perform_full_analysis(sentences, sentence_words_list, model):
    analysis_report = {}
    if not model:
        return {"error": "Word2Vec 모델이 생성되지 않았습니다."}

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
                if re.search(r'\b' + conn_word + r'\b', sentence.lower()):
                    found_connectors.append({"Sentence No.": i + 1, "Sentence": sentence, "Connector": conn_word, "Function": conn_type})
    analysis_report['syntax_analysis'] = {"connectors": found_connectors}

    return analysis_report

def display_report(report):
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
        pass

def main():
    st.title("📝 영어 지문 상세 분석 엔진")
    st.markdown("사용자가 입력한 **영어 텍스트(지문)**를 다각도로 분석하여 **핵심 내용, 구조, 어휘**를 포함한 상세 리포트를 생성합니다.")

    # NLTK 설정이 완료되었음을 사용자에게 알릴 수 있습니다 (선택 사항)
    st.sidebar.success("언어 분석 리소스 준비 완료!")

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
            with st.spinner('텍스트를 분석 중입니다...'):
                # 이제 NLTK 함수들은 안전하게 호출됩니다.
                from nltk.stem import WordNetLemmatizer
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
    1.  **리소스 준비**: 앱 시작 시 `NLTK`의 `punkt`, `stopwords` 등 필수 데이터 패키지를 먼저 다운로드하여 `LookupError`를 방지합니다.
    2.  **입력 전처리**: `NLTK`를 사용하여 텍스트를 문장/단어로 나누고, 불용어를 제거한 뒤 품사(명사, 동사, 형용사)를 기준으로 핵심 단어를 선별합니다. 단어는 기본형(표제어)으로 변환됩니다.
    3.  **의미 벡터화**: 전처리된 단어들을 `Word2Vec` 모델로 학습시켜 각 단어를 벡터로 변환합니다.
    4.  **유사도 분석**: 벡터 간 '코사인 유사도'를 계산하여 의미적 관련성을 파악합니다.
    5.  **핵심 내용/구조 분석**: 이 유사도를 기반으로 핵심 단어/문장 추출, 문단 분할 등의 분석을 수행합니다.
    """)
    st.sidebar.markdown("---")
    st.sidebar.caption("Made with Streamlit, Gensim & NLTK")

if __name__ == "__main__":
    main()
