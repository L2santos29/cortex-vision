.PHONY: run test clean setup

run: setup
	@echo "🚀 Starting server at http://localhost:8000"
	@echo "   API docs: http://localhost:8000/docs"
	@.venv/bin/python -m src.main

setup: .venv
	@.venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@.venv/bin/pip install -q -r requirements.txt

.venv:
	@echo "📦 Creating virtual environment..."
	@python3 -m venv .venv

test: setup
	@.venv/bin/pip install -q pytest
	@.venv/bin/python -m pytest tests/ -v

clean:
	@rm -rf .venv uploads output
	@echo "🧹 Cleaned up .venv, uploads, and output"

shell: setup
	@.venv/bin/python -c "from src.detector import Detector; d = Detector(); print('✅ Detector ready')"
	@echo "🐍 Virtual env active. Run: source .venv/bin/activate"
