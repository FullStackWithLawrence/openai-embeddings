SHELL := /bin/bash

ifneq ("$(wildcard .env)","")
    include .env
else
    $(shell echo -e "OPENAI_API_ORGANIZATION=PLEASE-ADD-ME\nOPENAI_API_KEY=PLEASE-ADD-ME\nPINECONE_API_KEY=PLEASE-ADD-ME\nPINECONE_ENVIRONMENT=gcp-starter\nPINECONE_INDEX_NAME=hsr\nPINECONE_VECTORSTORE_TEXT_KEY=lc_id\nOPENAI_CHAT_MODEL_NAME=gpt-3.5-turbo\nOPENAI_PROMPT_MODEL_NAME=text-davinci-003\nOPENAI_CHAT_TEMPERATURE=0.0\nOPENAI_CHAT_MAX_RETRIES=3\nDEBUG_MODE=True\n" >> .env)
endif

.PHONY: analyze init activate test lint clean

# Default target executed when no arguments are given to make.
all: help

analyze:
	cloc . --exclude-ext=svg,json,zip --vcs=git
init:
	npm install && \
	python3.11 -m venv venv && \
	source venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	pre-commit install

activate:
	. venv/bin/activate

test:
	cd models && pytest -v -s tests/
	python -m setup_test

lint:
	pre-commit run --all-files && \
	black .

clean:
	rm -rf venv && rm -rf node_modules


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
