PYTHON ?= python

.PHONY: install test lint format figures demo

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

# Hai target dưới đây được hiện thực ở các việc sau.
figures:
	@echo "make figures: chua co, thuoc viec P3.2b (TV2)"; exit 1

demo:
	$(PYTHON) -m src.ui.gui_core
