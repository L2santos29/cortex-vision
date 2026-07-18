.PHONY: run test clean setup env

env:
	@echo "📄 Checking .env file..."
	@if [ ! -f .env ]; then \
		echo "   No .env found — copying from .env.example"; \
		sed 's/change-me-to-a-secure-random-key/dev-key-change-in-production/' .env.example > .env; \
		echo "   ✅ Created .env with development key"; \
	fi

run: setup env
	@echo "🚀 Starting server at http://localhost:8000"
	@echo "   API docs: http://localhost:8000/docs"
	@export $$(grep -v '^#' .env | xargs) && .venv/bin/python -m src.main

setup: .venv
	@.venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@.venv/bin/pip install -q -r requirements.txt

.venv:
	@echo "📦 Creating virtual environment..."
	@python3 -m venv .venv

test: setup env
	@.venv/bin/pip install -q pytest pytest-cov
	@export API_KEY=test-key-for-testing && .venv/bin/python -m pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	@rm -rf .venv uploads output
	@echo "🧹 Cleaned up .venv, uploads, and output"

shell: setup env
	@export $$(grep -v '^#' .env | xargs) && .venv/bin/python -c "from src.detector import Detector; d = Detector(); print('✅ Detector ready')"
	@echo "🐍 Virtual env active. Run: source .venv/bin/activate"
