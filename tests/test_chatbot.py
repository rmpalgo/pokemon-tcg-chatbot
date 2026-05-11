"""
Tests for the Pokémon TCG chatbot NLP pipeline.
Verifies preprocessing, model building, and response generation.
"""

import json
import string
import pytest
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Ensure NLTK data is available ────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ── Preprocessing (mirrors app.py logic) ─────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def knowledge_base():
    with open("knowledge_base.json", "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model(knowledge_base):
    corpus = []
    tags = []
    for entry in knowledge_base:
        for pattern in entry["patterns"]:
            corpus.append(preprocess(pattern))
            tags.append(entry["tag"])
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix, tags


# ── Knowledge Base Structure Tests ───────────────────────────────────────────
class TestKnowledgeBase:
    def test_is_list(self, knowledge_base):
        assert isinstance(knowledge_base, list)

    def test_has_intents(self, knowledge_base):
        assert len(knowledge_base) >= 10, "KB should have at least 10 intents"

    def test_all_entries_have_required_keys(self, knowledge_base):
        for entry in knowledge_base:
            assert "tag" in entry, f"Missing 'tag' key"
            assert "patterns" in entry, f"Missing 'patterns' in {entry.get('tag')}"
            assert "responses" in entry, f"Missing 'responses' in {entry.get('tag')}"

    def test_all_responses_nonempty(self, knowledge_base):
        for entry in knowledge_base:
            assert len(entry["responses"]) > 0, f"Empty responses for {entry['tag']}"

    def test_has_default_intent(self, knowledge_base):
        tags = [e["tag"] for e in knowledge_base]
        assert "default" in tags, "KB must include a 'default' fallback intent"

    def test_unique_tags(self, knowledge_base):
        tags = [e["tag"] for e in knowledge_base]
        assert len(tags) == len(set(tags)), "Duplicate tags found in KB"


# ── Preprocessing Tests ──────────────────────────────────────────────────────
class TestPreprocessing:
    def test_lowercase(self):
        assert "hello" in preprocess("HELLO World")

    def test_punctuation_removed(self):
        result = preprocess("What's up?!")
        assert "?" not in result
        assert "!" not in result

    def test_stopwords_removed(self):
        result = preprocess("what is the pokemon tcg")
        assert "the" not in result.split()
        assert "is" not in result.split()

    def test_lemmatization(self):
        result = preprocess("cards types wolves")
        assert "card" in result.split(), "Expected 'cards' -> 'card'"
        assert "type" in result.split(), "Expected 'types' -> 'type'"
        assert "wolf" in result.split(), "Expected 'wolves' -> 'wolf'"

    def test_empty_input(self):
        result = preprocess("")
        assert result == ""


# ── Model / Similarity Tests ────────────────────────────────────────────────
class TestModel:
    def test_vectorizer_fitted(self, model):
        vectorizer, _, _ = model
        assert hasattr(vectorizer, "vocabulary_")

    def test_matrix_shape(self, model, knowledge_base):
        _, tfidf_matrix, tags = model
        total_patterns = sum(len(e["patterns"]) for e in knowledge_base if e["patterns"])
        assert tfidf_matrix.shape[0] == total_patterns
        assert len(tags) == total_patterns

    def test_greeting_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("hello")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "greeting"

    def test_rules_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("teach me the basic rules")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "how_to_play"

    def test_deck_building_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("tips for building a deck")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "deck_building"

    def test_gibberish_low_confidence(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("xyzzy foobar blargh")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        assert np.max(sims) < 0.15, "Gibberish should score below threshold"

    def test_card_types_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("what types of cards are there")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "card_types"

    def test_energy_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("what are the energy types")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "energy_types"


# ── Pricing Intent Tests ───────────────────────────────────────────────────
class TestPricingIntents:
    def test_series_overview_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("which series are the most valuable")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "series_overview"

    def test_trending_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("what sets are going up in price")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "trending_sets"

    def test_most_traded_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("which sets have the most sales")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "most_traded_sets"

    def test_most_valuable_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("most expensive pokemon sets")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "most_valuable_sets"

    def test_price_spread_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("which sets have chase cards")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "price_spread"

    def test_series_timeline_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("when did each series come out")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "series_timeline"

    def test_pricing_intents_exist(self, knowledge_base):
        tags = [e["tag"] for e in knowledge_base]
        pricing_tags = [
            "series_overview", "set_pricing_summary", "trending_sets",
            "most_traded_sets", "most_valuable_sets", "rarity_info",
            "series_timeline", "price_spread"
        ]
        for tag in pricing_tags:
            assert tag in tags, f"Missing pricing intent: {tag}"

    def test_top_cards_intents_exist(self, knowledge_base):
        tags = [e["tag"] for e in knowledge_base]
        top_cards_tags = [
            "top_cards_overview", "top_cards_vintage",
            "top_cards_modern", "graded_cards"
        ]
        for tag in top_cards_tags:
            assert tag in tags, f"Missing top cards intent: {tag}"

    def test_total_intent_count(self, knowledge_base):
        assert len(knowledge_base) == 47, f"Expected 47 intents, got {len(knowledge_base)}"

    def test_top_cards_overview_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("top 5 cards in each series")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "top_cards_overview"

    def test_graded_cards_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("how much are graded cards worth")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "graded_cards"

    def test_enriched_intents_exist(self, knowledge_base):
        """Verify the 5 new intents from card data enrichment exist."""
        tags = [e["tag"] for e in knowledge_base]
        enriched_tags = [
            "pokemon_types", "card_variants", "banned_cards",
            "regulation_marks", "card_mechanics_history"
        ]
        for tag in enriched_tags:
            assert tag in tags, f"Missing enriched intent: {tag}"

    def test_pokemon_types_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("how many pokemon types are in the tcg")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "pokemon_types"

    def test_banned_cards_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("what cards are banned in expanded")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "banned_cards"

    def test_regulation_marks_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("what are regulation marks on cards")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "regulation_marks"

    def test_card_mechanics_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("difference between gx vmax and vstar")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "card_mechanics_history"

    def test_card_variants_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("what is a reverse holofoil card")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "card_variants"

    def test_pricing_tools_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("best website to check card prices")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "pricing_tools"

    def test_price_tracking_match(self, model):
        vectorizer, tfidf_matrix, tags = model
        user_vec = vectorizer.transform([preprocess("what affects card prices and when to buy")])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_tag = tags[np.argmax(sims)]
        assert best_tag == "price_tracking"

    def test_pricing_platform_intents_exist(self, knowledge_base):
        tags = [e["tag"] for e in knowledge_base]
        assert "pricing_tools" in tags, "Missing pricing_tools intent"
        assert "price_tracking" in tags, "Missing price_tracking intent"

    def test_no_internal_field_names(self, knowledge_base):
        """Ensure no database column names leak into responses."""
        internal_terms = [
            "loose_price", "cib_price", "new_price", "sales_volume",
            "tcg_id", "tcgplayer_url", "image_url", "set_abbr",
            "loose_min", "loose_max", "loose_median", "loose_mean",
            "dc_set_id", "series_id", "set_id", "dc_set_name",
            "enriched_id", "enriched_set_id", "enriched_match"
        ]
        for intent in knowledge_base:
            for resp in intent["responses"]:
                for term in internal_terms:
                    assert term not in resp.lower(), \
                        f"Internal field '{term}' leaked in {intent['tag']}"


# ── Suggestion Prompt Tests ─────────────────────────────────────────────────
class TestSuggestions:
    def test_all_intents_have_suggestions(self, knowledge_base):
        """Every intent should have a non-empty suggestions list."""
        for entry in knowledge_base:
            assert "suggestions" in entry, \
                f"Missing 'suggestions' key in {entry['tag']}"
            assert len(entry["suggestions"]) >= 1, \
                f"Empty suggestions for {entry['tag']}"

    def test_suggestions_are_pairs(self, knowledge_base):
        """Each suggestion should be a [label, query] pair."""
        for entry in knowledge_base:
            for sug in entry.get("suggestions", []):
                assert isinstance(sug, list) and len(sug) == 2, \
                    f"Bad suggestion format in {entry['tag']}: {sug}"
                assert isinstance(sug[0], str) and len(sug[0]) > 0, \
                    f"Empty label in {entry['tag']}"
                assert isinstance(sug[1], str) and len(sug[1]) > 0, \
                    f"Empty query in {entry['tag']}"

    def test_suggestions_max_count(self, knowledge_base):
        """No intent should have more than 4 suggestions (UI space)."""
        for entry in knowledge_base:
            assert len(entry.get("suggestions", [])) <= 4, \
                f"Too many suggestions in {entry['tag']}: {len(entry['suggestions'])}"

    def test_no_self_referencing_suggestions(self, knowledge_base):
        """Suggestions should not point back to the same intent's patterns."""
        tag_patterns = {}
        for entry in knowledge_base:
            tag_patterns[entry["tag"]] = [p.lower() for p in entry["patterns"]]
        for entry in knowledge_base:
            for label, query in entry.get("suggestions", []):
                # The query shouldn't exactly match one of this intent's own patterns
                assert query.lower() not in tag_patterns.get(entry["tag"], []), \
                    f"Self-referencing suggestion in {entry['tag']}: '{query}'"
