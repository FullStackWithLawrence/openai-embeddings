---
name: openai-embeddings-development
description: Start here for any change to the openai-embeddings repository (the models/ package, its command-line examples, prompt templates, Pinecone index helper, hybrid search retriever, settings, tests, requirements, commits). Gives the rules, the code layout and conventions, and an index of the narrower openai-embeddings-* skills to load for each kind of work.
---

# openai-embeddings Development

A teaching demo of OpenAI embeddings, Pinecone and LangChain. People read this
code to learn from it, often alongside a YouTube video. Keep it short,
readable and conventional, and prefer the plainest working LangChain API over
a clever one.

## How the code fits together

- `models/conf.py` builds one frozen Pydantic `Settings` object, `settings`,
  at import time. Defaults come from `SettingsDefaults`, which reads
  `os.environ` after `load_dotenv()`. A real environment variable wins over
  `.env`, because `load_dotenv()` doesn't override. API keys are `SecretStr`:
  call `.get_secret_value()` only where a client needs the key, never in a log.
- `models/pinecone.py`, `PineconeIndex`: lazy properties for the Pinecone
  client, the index, the `PineconeVectorStore`, the `OpenAIEmbeddings` and the
  text splitter. `create()`, `delete()`, `initialize()` (delete then create)
  and `pdf_loader()` change the live index.
- `models/hybrid_search_retriever.py`, `HybridSearchRetriever`: a `ChatOpenAI`
  with LangChain's in-memory cache, a `BM25Encoder` and a
  `PineconeHybridSearchRetriever`. `cached_chat_request()`,
  `prompt_with_template()` and `rag()` are what the examples call.
- `models/examples/*.py`: thin `argparse` scripts, run as
  `python -m models.examples.<name>`. The README's Usage section lists them;
  keep it in sync when an example changes.
- `models/yt.py` is the video's script. Nothing imports it, but it must still
  import cleanly.

## Conventions

- Every Python file starts with `# -*- coding: utf-8 -*-` (flake8 C101 and the
  `fix-encoding-pragma` hook check it), then a module docstring.
- Docstrings on every public class, method and property, in reST field-list
  style. See `openai-embeddings-docstrings`.
- Logging: every message starts with `logger_prefix` and the function name,
  with %-style arguments. Never log a key. See `openai-embeddings-logging`.
- Annotations: `Any` only when the value really can be anything. See
  `openai-embeddings-python-typing`.
- Lazy read-only properties (`_x = None`, then build on first access) for
  clients and other expensive objects, as in `PineconeIndex` and
  `HybridSearchRetriever`.
- Narrow pylint disables with a reason, e.g. `# pylint: disable=import-error`
  on optional or compiled imports. Don't add blanket disables.
- New settings go in three places: `SettingsDefaults`, a `Settings` field, and
  the `dump` property, plus a test in `models/tests/test_configuration.py`.
- LangChain 1.x moved most imports: use `langchain_core.*`,
  `langchain_openai`, `langchain_community`, `langchain_text_splitters`,
  `langchain_pinecone`, and `langchain_classic` for legacy chains. Never
  `langchain.prompts`, `langchain.schema` or `langchain.chat_models`. See
  `openai-embeddings-dependencies`.

## Rules

1. Never run git write commands. See `openai-embeddings-git-commits`.
2. Never read, print or edit `.env`.
3. Run the tests yourself, knowing they use live services. See
   `openai-embeddings-testing`.
4. The pre-commit hooks are the lint standard. See
   `openai-embeddings-code-quality`.
5. Keep CI's Python (3.12, in `.github/workflows/testsPython.yml`) and the
   Makefile's `PYTHON` both working.

## Which skill to load

| Task                                                        | Skill                             |
| ----------------------------------------------------------- | --------------------------------- |
| Writing, fixing or running tests                            | `openai-embeddings-testing`       |
| Requirements, Python versions, `make init`, LangChain moves | `openai-embeddings-dependencies`  |
| Lint, format, pre-commit                                    | `openai-embeddings-code-quality`  |
| Python docstrings                                           | `openai-embeddings-docstrings`    |
| Logger calls: prefixes, levels, secrets                     | `openai-embeddings-logging`       |
| Type annotations, `Any`                                     | `openai-embeddings-python-typing` |
| Writing a commit message                                    | `openai-embeddings-git-commits`   |

## Definition of done

- The change matches the surrounding code.
- Tests cover it and you ran them. Report failures, including pre-existing
  ones.
- The pre-commit hooks pass on the changed files.
- The README's Usage section still matches the examples.
- Your summary says what changed, what you verified and what you skipped.
