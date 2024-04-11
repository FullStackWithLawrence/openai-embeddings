SHELL := /bin/bash
ifeq ($(OS),Windows_NT)
    PYTHON = python.exe
    ACTIVATE_VENV = venv\Scripts\activate
else
    PYTHON = python3.11
    ACTIVATE_VENV = source venv/bin/activate
endif
PIP = $(PYTHON) -m pip

ifneq ("$(wildcard .env)","")
    include .env
else
$(shell echo -e "OPENAI_API_ORGANIZATION=PLEASE-ADD-ME\n\
OPENAI_API_KEY=PLEASE-ADD-ME\n\
PINECONE_API_KEY=PLEASE-ADD-ME\n\
PINECONE_ENVIRONMENT=gcp-starter\n\
PINECONE_INDEX_NAME=rag\n\
PINECONE_VECTORSTORE_TEXT_KEY=lc_id\n\
PINECONE_METRIC=dotproduct\n\
PINECONE_DIMENSIONS=1536\n\
OPENAI_CHAT_MODEL_NAME=gpt-3.5-turbo\n\
OPENAI_PROMPT_MODEL_NAME=gpt-3.5-turbo-instruct\n\
OPENAI_CHAT_TEMPERATURE=0.0\n\
OPENAI_CHAT_MAX_RETRIES=3\n\
DEBUG_MODE=True\n" >> .env)
endif

.PHONY: analyze init activate test lint clean

# Default target executed when no arguments are given to make.
all: help

analyze:
	cloc . --exclude-ext=svg,json,zip --vcs=git

init:
	make clean && \
	$(PYTHON) -m venv venv && \
	$(ACTIVATE_VENV) && \
	$(PIP) install --upgrade pip && \
	$(PIP) install -r requirements.txt && \
	npm install && \
	pre-commit install

activate:
	. venv/bin/activate

test:
	cd models && pytest -v -s tests/
	python -m setup_test

lint:
	pre-commit run --all-files && \
	pylint models && \
	flake8 . && \
	isort . && \
	black .

clean:
	rm -rf venv && rm -rf node_modules && \
	find ./models/ -name __pycache__ -type d -exec rm -rf {} +

release:
	git commit -m "fix: force a new release" --allow-empty && git push

######################
# HELP
######################

help:
	@echo '===================================================================='
	@echo 'analyze             - generate code analysis report'
	@echo 'init            - create a Python virtual environment and install dependencies'
	@echo 'activate        - activate the Python virtual environment'
	@echo 'test            - run Python unit tests'
	@echo 'lint            - run Python linting'
	@echo 'clean           - destroy the Python virtual environment'
