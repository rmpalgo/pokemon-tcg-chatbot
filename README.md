# Pokémon TCG Chatbot

A closed-domain NLP chatbot for the Pokémon Trading Card Game, built with TF-IDF vectorization and cosine similarity.

## Features

- **47 intents** covering rules, gameplay, card types, deck building, pricing, tournaments, and more
- **357 training patterns** for robust intent matching
- **Suggestion chips** — contextual follow-up prompts appear after each response to guide the conversation
- **NLP preprocessing pipeline** — tokenization, lemmatization, stop-word removal, punctuation stripping
- **Confidence threshold** — unrecognized queries get a helpful fallback response instead of a bad match

## Quick Start

```bash
# Install dependencies
make install

# Run locally
make run

# Run tests
make test

# Lint
make lint

# Run everything (install, lint, test, validate)
make all
```

## Deployment

This app deploys to **Streamlit Community Cloud** connected to this repo.

### CI/CD:
- GitHub Actions runs lint + tests across Python 3.10, 3.11, and 3.12 on every push and PR
- Streamlit Cloud auto-redeploys when `main` is updated
- See `.github/workflows/ci.yml` for the pipeline

## Project Structure

```
pokemon-tcg-chatbot/
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── .streamlit/config.toml      # Streamlit theme config
├── tests/
│   └── test_chatbot.py         # Pytest suite (44 tests across 5 classes)
├── app.py                      # Main chatbot application
├── knowledge_base.json         # 47 intents, 357 patterns, suggestion prompts
├── nlp_pipeline_diagram.jsx    # React visualization of the NLP pipeline
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Dev dependencies (flake8, pytest)
├── Makefile                    # Build/test/run commands
└── README.md
```

## NLP Methods

- **TF-IDF Vectorization** (scikit-learn) — converts text to numerical feature vectors weighted by term importance
- **Cosine Similarity** — matches user input to the closest known pattern in the knowledge base
- **NLTK Preprocessing** — tokenization, WordNet lemmatization, stop-word removal
- **Confidence Threshold** (0.15) — queries below this score trigger a fallback response

## Knowledge Base Categories

| Category | Intents | Topics |
|----------|---------|--------|
| General | 4 | Greetings, goodbyes, thanks, what is the TCG |
| Rules | 14 | How to play, card types, energy, evolution, conditions, retreating |
| Game | 15 | Deck building, formats, tournaments, popular decks, TCG Live/Pocket |
| Pricing | 8 | Set values, trends, most traded/valuable sets, price spreads |
| Cards | 4 | Top vintage/modern cards, graded cards, card overview |
| Enriched | 7 | Pokémon types, variants, banned cards, regulation marks, mechanics, pricing tools |

## Test Coverage

- **TestKnowledgeBase** — structure validation (required keys, unique tags, default intent)
- **TestPreprocessing** — lowercase, punctuation, stop-words, lemmatization, empty input
- **TestModel** — vectorizer fitting, matrix shape, intent matching accuracy
- **TestPricingIntents** — pricing/cards intent matching and existence checks, field name leak detection
- **TestSuggestions** — all intents have suggestions, valid format, max count, no self-references

## Built With

- [Streamlit](https://streamlit.io/) — web UI framework
- [scikit-learn](https://scikit-learn.org/) — TF-IDF and cosine similarity
- [NLTK](https://www.nltk.org/) — natural language preprocessing
- [NumPy](https://numpy.org/) — numerical operations

---

*An NLP-powered Pokémon TCG assistant*
