.PHONY: install lint test validate run clean all

# Default target
all: install lint test validate

# Install all dependencies (app + dev)
install:
	pip install -r requirements.txt -r requirements-dev.txt

# Lint with flake8
lint:
	flake8 app.py tests/ --max-line-length 120 --ignore E501,W503

# Run tests with pytest
test:
	python -m pytest tests/ -v --tb=short

# Validate the knowledge base JSON structure
validate:
	python -c "\
	import json, sys; \
	kb = json.load(open('knowledge_base.json')); \
	assert isinstance(kb, list), 'KB must be a list'; \
	for e in kb: \
	    assert 'tag' in e, f'Missing tag'; \
	    assert 'patterns' in e, f'Missing patterns in {e[\"tag\"]}'; \
	    assert 'responses' in e, f'Missing responses in {e[\"tag\"]}'; \
	    assert isinstance(e['responses'], list) and len(e['responses']) > 0, f'Empty responses in {e[\"tag\"]}'; \
	print(f'OK: {len(kb)} intents, {sum(len(e[\"patterns\"]) for e in kb)} patterns'); \
	"

# Run the Streamlit app locally
run:
	streamlit run app.py

# Clean caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache
