# ODH Data Processing - Test Commands

.PHONY: help test-notebooks test-notebooks-list test-all install clean

# Default target
help:
	@echo "ODH Data Processing - Test Commands:"
	@echo ""
	@echo "Notebook Tests:"
	@echo "  test-notebooks                     Run all notebook validation tests"
	@echo "  test-notebooks-list NOTEBOOKS=... Run notebook tests on specific files"
	@echo ""
	@echo "Examples:"
	@echo "  make test-notebooks"
	@echo "  make test-notebooks-list NOTEBOOKS='notebooks/example1.ipynb notebooks/example2.ipynb'"
	@echo ""
	@echo "Setup & Utilities:"
	@echo "  install                           Install test dependencies"

# Install test dependencies
install:
	pip install -r tests/requirements.txt

# Notebook validation tests
test-notebooks:
	@echo "🧪 Running notebook validation tests..."
	pytest tests/test_notebooks_validate.py -v

test-notebooks-list:
	@if [ -z "$(NOTEBOOKS)" ]; then \
		echo "❌ Please specify notebooks: make test-notebooks-list NOTEBOOKS='notebook1.ipynb notebook2.ipynb'"; \
		exit 1; \
	fi
	@echo "🧪 Running notebook validation on specified files..."
	@NOTEBOOK_LIST="$(NOTEBOOKS)"; \
	JSON_LIST=$$(echo "$$NOTEBOOK_LIST" | tr ' ' '\n' | sed 's/.*/"&"/' | tr '\n' ',' | sed 's/,$$//' | sed 's/^/[/' | sed 's/$$/]/'); \
	CHANGED_NOTEBOOKS="$$JSON_LIST" pytest tests/test_notebooks_validate.py -v
