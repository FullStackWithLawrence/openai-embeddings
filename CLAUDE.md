# CLAUDE.md

Guidance for Claude Code (and other coding agents) working in this repository.

openai-embeddings is a teaching demo from Full Stack With Lawrence. It shows
system prompting, prompt templates, LangChain, a PDF loader that writes OpenAI
embeddings to a Pinecone index, and Retrieval Augmented Generation over a
hybrid (dense + BM25 sparse) search retriever. Settings are managed with
Pydantic. The code is small (about 1,800 lines under `models/`), and readers
learn from it, so clarity matters more than cleverness.

Detailed know-how lives in the Agent Skills under `skills/` (symlinked to
`.claude/skills`). **Load `openai-embeddings-development` first for any
change.** Its table says which narrower skill covers each kind of work. This
file holds only what applies to nearly every task.

## Hard rules

1. **Never run git write commands.** "Write a git commit" means print the
   command as text. Never run `git commit`, `git add`, `git reset`, `git push`
   or anything else that changes the index or history. See
   `openai-embeddings-git-commits`.
2. **The tests call live OpenAI and Pinecone.** They read the keys in `.env`,
   send billable prompts and embeddings, and `models/tests/test_pinecone.py`
   deletes and recreates the configured index (`PINECONE_INDEX_NAME`,
   default `openai-embeddings`). That's accepted for this demo. Don't touch
   other indexes in the account. See `openai-embeddings-testing`.
3. **Run the tests yourself** after any change, and report failures honestly,
   including pre-existing ones.
4. **Never read, print or edit `.env`.** It holds live API keys. To learn a
   non-secret value, `grep` for that one variable.
5. **The pre-commit hooks are the lint standard** (black, isort, flake8,
   pylint, bandit, codespell, prettier, and the pre-commit-hooks checks). See
   `openai-embeddings-code-quality`.
6. **This repository is public.** No API keys, organization IDs or private
   data in code, tests, docs or skills.

## Repository map

| Path                                | Contents                                                             |
| ----------------------------------- | -------------------------------------------------------------------- |
| `models/conf.py`                    | Pydantic `Settings` singleton (`settings`), read from env and `.env` |
| `models/pinecone.py`                | `PineconeIndex`: create, delete, initialize an index, load PDFs      |
| `models/hybrid_search_retriever.py` | `HybridSearchRetriever`: chat, templates, RAG                        |
| `models/prompt_templates.py`        | `PromptTemplate`s used by the examples                               |
| `models/examples/`                  | Command-line examples (`python -m models.examples.<name>`)           |
| `models/tests/`                     | pytest tests, mock data in `mock_data/`                              |
| `models/yt.py`                      | Script from the YouTube video; not imported by anything              |
| `requirements/`                     | `base.txt` (runtime, unpinned), `local.txt` (dev and test)           |
| `data/`                             | Sample PDFs for the RAG example                                      |
| `skills/`                           | Agent Skills for developing this repository                          |
| `.github/workflows/`                | Tests on Python 3.12, semantic-release on `main`                     |

## Common commands

```bash
make init                      # fresh venv + requirements + npm + pre-commit
source venv/bin/activate
make test                      # pytest models/tests, then setup_test
make coverage                  # tests with coverage; fails under 95%
python -m pytest -q models/tests/test_configuration.py   # one module
git diff --name-only | xargs pre-commit run --files     # lint changed files
```

The Makefile's `PYTHON` picks the interpreter for `make init` (currently
`python3.14`). CI uses 3.12. Both must keep working.

## Conventions

- **Docstrings** are reStructuredText (`:param:`, `:returns:`, `:raises:`)
  with a one-line summary. See `openai-embeddings-docstrings`.
- **Logging:** every message starts with its origin, `logger_prefix`
  (`__name__`) then `.function_name()`, with %-style arguments. See
  `openai-embeddings-logging`.
- **Typing:** `Any` only when the value really can be anything. See
  `openai-embeddings-python-typing`.
- **Coverage:** at least 95% of `models/`, measured with `make coverage`.

## Definition of done

- The change matches the style of the surrounding code.
- Tests cover it, and you ran them.
- Pre-commit hooks pass on the changed files.
- Your summary says what changed, what you verified, and what you skipped.
