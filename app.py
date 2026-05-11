"""
Pokémon TCG Chatbot — A closed-domain NLP chatbot using TF-IDF and cosine similarity.
Built with Streamlit, scikit-learn, and NLTK.
Author: rmpalgo
"""

import json
import random
import string
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)


@st.cache_data
def load_knowledge_base():
    with open("knowledge_base.json", "r") as f:
        return json.load(f)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


@st.cache_resource
def build_model():
    kb = load_knowledge_base()
    corpus = []
    tags = []
    for entry in kb:
        for pattern in entry["patterns"]:
            corpus.append(preprocess(pattern))
            tags.append(entry["tag"])

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix, tags, kb


CONFIDENCE_THRESHOLD = 0.15


def get_response(user_input, vectorizer, tfidf_matrix, tags, kb):
    processed = preprocess(user_input)

    if not processed.strip():
        default = [e for e in kb if e["tag"] == "default"][0]
        return random.choice(default["responses"]), "default"

    user_vec = vectorizer.transform([processed])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < CONFIDENCE_THRESHOLD:
        default = [e for e in kb if e["tag"] == "default"][0]
        return random.choice(default["responses"]), "default"

    matched_tag = tags[best_idx]
    matched_entry = [e for e in kb if e["tag"] == matched_tag][0]
    return random.choice(matched_entry["responses"]), matched_tag


def get_suggestions(tag, kb):
    entry = next((e for e in kb if e["tag"] == tag), None)
    if entry and entry.get("suggestions"):
        return entry["suggestions"]
    default = next((e for e in kb if e["tag"] == "default"), None)
    return default.get("suggestions", []) if default else []


def main():
    st.set_page_config(
        page_title="Pokémon TCG Chatbot",
        page_icon="🎴",
        layout="centered",
    )

    st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .chat-header {
        text-align: center;
        padding: 1rem 0;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button {
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        border: 1px solid #e0e0e0;
        background: #f8f9fa;
        color: #333;
        white-space: nowrap;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: #e8f4fd;
        border-color: #4a9eda;
        color: #1a73e8;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='chat-header'>", unsafe_allow_html=True)
    st.title("🎴 Pokémon TCG Chatbot")
    st.caption(
        "Ask me anything about the Pokémon Trading Card Game — rules, cards, "
        "strategies, sets, and more!"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("About This Chatbot")
        st.markdown(
            """
            **Type:** Closed-domain NLP chatbot

            **NLP Methods:**
            - TF-IDF Vectorization
            - Cosine Similarity
            - NLTK Lemmatization
            - Stop-word Removal

            **Domain:** Pokémon Trading Card Game

            **Topics I can help with:**
            - Game rules & how to play
            - Card types & energy
            - Deck building tips
            - Competitive formats
            - Card values & rarity
            - Pokémon ex cards
            - TCG Live & TCG Pocket
            - Tournament info
            - And more!

            ---
            *An NLP-powered Pokémon TCG assistant*
            """
        )

    vectorizer, tfidf_matrix, tags, kb = build_model()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello, Trainer! 🎴 I'm your Pokémon TCG assistant. "
                "Ask me about rules, cards, deck building, strategies, "
                "or anything else about the Pokémon Trading Card Game!",
                "tag": "greeting",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "assistant" and last_msg.get("tag"):
        suggestions = get_suggestions(last_msg["tag"], kb)
        if suggestions:
            cols = st.columns(len(suggestions))
            for i, (label, query) in enumerate(suggestions):
                with cols[i]:
                    if st.button(label, key=f"sug_{len(st.session_state.messages)}_{i}"):
                        st.session_state._suggestion_query = query
                        st.rerun()

    suggestion_query = st.session_state.pop("_suggestion_query", None)
    prompt = suggestion_query or st.chat_input("Ask me about Pokémon TCG...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response, matched_tag = get_response(prompt, vectorizer, tfidf_matrix, tags, kb)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "tag": matched_tag,
        })
        with st.chat_message("assistant"):
            st.markdown(response)

        st.rerun()


if __name__ == "__main__":
    main()
