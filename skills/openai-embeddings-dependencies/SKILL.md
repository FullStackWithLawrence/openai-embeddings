---
name: openai-embeddings-dependencies
description: Use when changing openai-embeddings requirements (requirements/base.txt, requirements/local.txt), the Python version (Makefile PYTHON, CI python-version, pre-commit default_language_version), fixing make init or pip resolution errors, reviewing a Dependabot bump, or fixing code broken by a LangChain, Pinecone or OpenAI release. Covers the unpinned requirements, the langchain-pinecone Python 3.14 cap and --ignore-requires-python, and where LangChain 1.x moved its imports.
---

# openai-embeddings Dependencies

## Layout

- `requirements/base.txt`: runtime packages, **unpinned**, so `make init`
  always gets the latest releases. `setup.py` reads it for
  `install_requires`, and semantic-release commits it with each release.
- `requirements/local.txt`: `-r base.txt` plus test and lint tools. This is
  what `make init` and CI install.
- Dependabot opens bump PRs, and Mergify merges them.

Because nothing is pinned, an upstream release can break `make init` or the
tests overnight. When that happens, fix the code for the new API rather than
pinning, unless there's no other way; then pin with a comment that says why
and when to remove it.

## Python versions

Three places name a Python version. Keep them consistent with what you test:

| Place                                                | Current      |
| ---------------------------------------------------- | ------------ |
| `Makefile` `PYTHON` (used by `make init`)            | `python3.14` |
| `.github/workflows/*.yml` `python-version`           | `3.12`       |
| `.pre-commit-config.yaml` `default_language_version` | `python3.12` |

## langchain-pinecone and Python 3.14

Every langchain-pinecone release, including upstream main, declares
`Requires-Python <3.14`. On 3.14, pip skips them and backtracks to 0.0.x,
which needs `simsimd<4`, which has no 3.14 wheels. The error is
`No matching distribution found for simsimd<4.0.0,>=3.6.3`.

The current release works on 3.14, so `make init` installs with
`--ignore-requires-python`. Remove that flag once a langchain-pinecone release
supports 3.14. Check with:

```bash
curl -s https://pypi.org/pypi/langchain-pinecone/json | python3 -c "import sys,json;i=json.load(sys.stdin)['info'];print(i['version'],i['requires_python'])"
```

To diagnose any similar failure, read pip's "Ignored the following versions
that require a different python version" line: it names the capped releases.

## LangChain 1.x import locations

| Old import                              | Use                                      |
| --------------------------------------- | ---------------------------------------- |
| `langchain.prompts`                     | `langchain_core.prompts`                 |
| `langchain.schema` messages             | `langchain_core.messages`                |
| `langchain.globals`                     | `langchain_core.globals`                 |
| `langchain.chat_models.ChatOpenAI`      | `langchain_openai.ChatOpenAI`            |
| `langchain.embeddings.OpenAIEmbeddings` | `langchain_openai.OpenAIEmbeddings`      |
| `langchain.text_splitter`               | `langchain_text_splitters`               |
| `langchain.vectorstores.Pinecone`       | `langchain_pinecone.PineconeVectorStore` |
| `langchain.chains.LLMChain` and friends | `langchain_classic.chains`               |

`langchain-community` is being sunset. Don't add new imports from it when a
standalone integration package exists.

After any bump, verify with a fresh venv and the tests (see
`openai-embeddings-testing`):

```bash
grep -rn "from langchain\.\|import langchain$" --include='*.py' models setup.py
```
