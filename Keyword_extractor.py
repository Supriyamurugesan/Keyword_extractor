import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
kw_model = KeyBERT(model=SentenceTransformer('all-MiniLM-L6-v2'))

# Streamlit UI
st.set_page_config(page_title="Smart Keyword Extractor", layout="centered")
st.title("🔑Keyword Extractor")

input_text = st.text_area("📄 Paste your text below:", height=200)

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

def get_unique_keywords(text, top_n=10, final_count=5):
    raw_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n)
    seen = set()
    unique_keywords = []

    for word, score in raw_keywords:
        lemma = lemmatizer.lemmatize(word.lower())
        if lemma not in seen:
            seen.add(lemma)
            unique_keywords.append(word)
        if len(unique_keywords) == final_count:
            break

    return unique_keywords

if st.button("🔍 Extract Keywords"):
    if not input_text or len(input_text.split()) < 10:
        st.warning("⚠️ Please enter at least 10 words.")
    else:
        try:
            cleaned = clean_text(input_text)
            keywords = get_unique_keywords(cleaned)

            st.subheader("✨ Unique Keywords")
            st.success(", ".join(keywords))

            st.info(f"📝 Words in Text: **{len(input_text.split())}** | 🎯 Unique Keywords: **{len(keywords)}**")

        except Exception as error:
            st.error(f"🚫 Error occurred: {error}")
