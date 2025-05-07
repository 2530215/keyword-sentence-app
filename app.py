from collections import Counter
import streamlit as st
# import urllib.request # 웹에서 직접 데이터를 가져오는 부분이 아니라면 필요 없을 수 있습니다.
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re # 문장 분리 시 정규표현식 사용

# --- 기존 코드의 함수들 ---
def processing(input_factor):
    processed_words = []
    for word in input_factor:
        a = word
        alist = list(a)
        if not alist: # 빈 문자열 처리
            continue
        # 조사 및 어미 제거 (더 정교한 처리를 위해 KoNLPy 같은 형태소 분석기 사용을 고려해볼 수 있습니다)
        if (alist[-1]=="을")or(alist[-1]=="를")or(alist[-1]=="은")or(alist[-1]=="는")or(alist[-1]=="이")or(alist[-1]=="가"):
            processed_words.append("".join(alist[:-1]))
        elif(alist[-1]=="와")or(alist[-1]=="과"):
            processed_words.append("".join(alist[:-1]))
        elif len(alist) > 1 and ((alist[-2:]==list("에서"))or(alist[-2:]==list("께서"))): # 슬라이싱 오류 방지
            processed_words.append("".join(alist[:-2]))
        elif len(alist) > 1 and ((alist[-2:]==list("하고"))or(alist[-2:]==list("하다"))or(alist[-2:]==list("하며"))or(alist[-2:]==list("하는"))): # 슬라이싱 오류 방지
            processed_words.append("".join(alist[:-2]))
        else:
            processed_words.append(a)
    return [word for word in processed_words if word] # 빈 문자열 최종 제거

def get_keywords(rawword_list):
    wordset = list(set(rawword_list))
    wordsetcount = []
    for element in wordset:
        wordsetcount.append(rawword_list.count(element))

    # 빈도수 기준으로 단어 정렬 (Counter 객체 사용하면 더 간결)
    word_counts = Counter(rawword_list)
    sorted_wordset = [word for word, count in word_counts.most_common()]
    sorted_wordsetcount = [count for word, count in word_counts.most_common()]

    return sorted_wordset, sorted_wordsetcount

# --- Streamlit UI ---
st.title("📝 생기부 키워드 분석 및 연관 문장 추천")

raw_sentence_input = st.text_area("분석할 생기부 내용을 입력하세요:", height=200, placeholder="여기에 텍스트를 입력해주세요...")

if st.button("분석 시작 ✨"):
    if raw_sentence_input:
        with st.spinner('텍스트를 분석 중입니다... 잠시만 기다려주세요 ⏳'):
            raw_word_list_original = raw_sentence_input.split()
            processed_word_list = processing(list(raw_word_list_original)) # 원본 유지를 위해 복사본 사용

            if not processed_word_list:
                st.error("분석할 단어가 없습니다. 텍스트를 확인해주세요.")
            else:
                st.subheader("🔑 추출된 키워드 (빈도순)")
                wordset, wordsetcount = get_keywords(processed_word_list)
                keyword_df = pd.DataFrame({'키워드': wordset, '빈도수': wordsetcount})
                st.dataframe(keyword_df.head(10)) # 상위 10개 키워드 표시

                # Word2Vec 모델 학습
                # Word2Vec 학습을 위해서는 문장 리스트가 필요합니다.
                # 현재 rawword1은 단어 리스트의 리스트로 되어 있는데, gensim은 보통 토큰화된 문장들의 리스트를 기대합니다.
                # 예: [['첫', '문장', '단어들'], ['두번째', '문장', '단어들']]
                # 여기서는 전체 텍스트를 하나의 문서로 보고, 그 안의 단어들로 학습합니다.
                # 더 나은 성능을 위해서는 실제 문장 단위로 나누어 학습하는 것이 좋습니다.
                sentences_for_w2v = [processed_word_list] # 단일 문서로 취급
                if not any(sentences_for_w2v): # 모든 문장이 비어있지 않은지 확인
                    st.error("Word2Vec 모델 학습을 위한 데이터가 부족합니다.")
                else:
                    try:
                        model = Word2Vec(sentences_for_w2v, vector_size=100, window=3, min_count=1, workers=4, sg=1) # sg=1 for skip-gram
                        st.success("Word2Vec 모델 학습 완료!")

                        st.subheader("🔗 주요 키워드와 유사한 단어")
                        num_similar_words_to_show = 5
                        # wordset에서 상위 N개 키워드에 대해 유사 단어 표시 (모델에 없는 단어 예외 처리)
                        for i in range(min(len(wordset), 5)): # 상위 5개 또는 그 이하
                            keyword_to_check = wordset[i]
                            if keyword_to_check in model.wv:
                                similar_words = model.wv.most_similar(keyword_to_check, topn=num_similar_words_to_show)
                                st.write(f"**'{keyword_to_check}'**와 유사한 단어:")
                                st.write([f"{word} ({similarity:.2f})" for word, similarity in similar_words])
                            else:
                                st.write(f"**'{keyword_to_check}'**는(은) 모델의 어휘 사전에 없습니다.")

                        # 문장 유사도 분석
                        st.subheader("📜 키워드와 연관성 높은 문장")
                        # 문장 분리 (마침표, 물음표, 느낌표 기준)
                        splited_sentences_raw = re.split(r'[.?!]\s*', raw_sentence_input)
                        splited_sentences = [sent.strip() for sent in splited_sentences_raw if sent.strip()]

                        if not splited_sentences:
                            st.warning("분석할 문장이 없습니다.")
                        else:
                            num_top_sentences = 3 # 각 키워드별로 보여줄 상위 문장 수

                            # 상위 키워드 몇 개에 대해서만 문장 유사도 계산 (예: 상위 3개)
                            for i in range(min(len(wordset), 3)):
                                main_keyword = wordset[i]
                                if main_keyword not in model.wv:
                                    st.write(f"키워드 '{main_keyword}'에 대한 벡터가 없어 문장 유사도를 계산할 수 없습니다.")
                                    continue

                                st.markdown(f"--- \n#### '{main_keyword}' 관련 문장:")
                                sentence_similarities = []

                                for sentence_text in splited_sentences:
                                    sentence_words_original = sentence_text.split()
                                    processed_sentence_words = processing(list(sentence_words_original))

                                    if not processed_sentence_words:
                                        continue

                                    # 문장 벡터 계산 (모델에 있는 단어들만 사용)
                                    vectors = [model.wv[token] for token in processed_sentence_words if token in model.wv]
                                    if not vectors:
                                        sentence_similarities.append({'sentence': sentence_text, 'similarity': 0.0})
                                        continue

                                    sentence_vector = np.mean(vectors, axis=0)
                                    keyword_vector = model.wv[main_keyword]
                                    similarity = cosine_similarity([sentence_vector], [keyword_vector])[0][0]
                                    sentence_similarities.append({'sentence': sentence_text, 'similarity': similarity})

                                if sentence_similarities:
                                    # 유사도 기준으로 문장 정렬
                                    sorted_sentences = sorted(sentence_similarities, key=lambda x: x['similarity'], reverse=True)
                                    for item in sorted_sentences[:num_top_sentences]:
                                        st.markdown(f"> {item['sentence']} *(유사도: {item['similarity']:.3f})*")
                                else:
                                    st.write(f"'{main_keyword}'와(과) 관련된 문장을 찾을 수 없거나, 문장 내 단어들이 모델에 없습니다.")
                    except Exception as e:
                        st.error(f"Word2Vec 모델 학습 또는 유사도 분석 중 오류 발생: {e}")
                        st.error("입력된 텍스트의 양이 충분한지, 다양한 단어가 포함되어 있는지 확인해주세요.")
                        st.error("특히, Word2Vec 모델은 학습 데이터의 질과 양에 민감합니다.")

    else:
        st.warning("분석할 내용을 입력해주세요.")

st.sidebar.header("ℹ️ 사용 방법")
st.sidebar.markdown("""
1.  **생기부 내용 입력:** 중앙의 텍스트 입력창에 분석하고 싶은 생기부 내용을 복사하여 붙여넣거나 직접 입력합니다.
2.  **분석 시작:** '분석 시작 ✨' 버튼을 클릭합니다.
3.  **결과 확인:**
    * **추출된 키워드:** 텍스트에서 자주 등장하는 단어들이 빈도순으로 표시됩니다.
    * **주요 키워드와 유사한 단어:** 주요 키워드와 의미적으로 유사한 단어들이 나타납니다.
    * **키워드와 연관성 높은 문장:** 주요 키워드와 가장 관련성이 높은 문장들이 추천됩니다.

**팁:** 더 정확한 분석을 위해 충분한 양의 텍스트를 입력하는 것이 좋습니다.
""")
