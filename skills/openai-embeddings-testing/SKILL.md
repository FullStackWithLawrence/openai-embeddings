---
name: openai-embeddings-testing
description: Use when writing, fixing or running the openai-embeddings pytest tests (models/tests, setup_test.py), or when a test fails. Covers that the tests call live OpenAI and Pinecone and delete the configured Pinecone index, how to run one module or the whole suite, using a throwaway index name, the os.environ.clear() trap in test_configuration.py, and what CI runs.
---

# openai-embeddings Testing

## The tests use live services

The original tests have no mocks for OpenAI or Pinecone. They read the keys
in `.env` (CI writes its own `.env` from repository secrets) and:

- send real, billable chat completions and embeddings
  (`test_examples.py`, `test_prompts.py`, `test_openai.py`, `test_hsr.py`);
- **delete and recreate the configured Pinecone index**
  (`test_pinecone.py`: `test_05_delete`, `test_06_create`, and
  `test_07_load_pdf`, which ends with `delete()`). After a full run, the index
  named by `PINECONE_INDEX_NAME` (default `openai-embeddings`) is gone.

That's accepted for this demo. Running the tests is fine without asking.
To keep a loaded index, point the run at another name. A real environment
variable wins over `.env`:

```bash
PINECONE_INDEX_NAME=openai-embeddings-test python -m pytest -q models/tests
```

Never delete or write to any other index in the account. To reload the demo
index afterwards, the user runs `python -m models.examples.load data/`.

`test_examples.py::test_prompt` asks the model to answer exactly `SUCCESS`.
An occasional mismatch is model drift, not a code bug: rerun once before
digging in.

## Running

From the repository root, with the venv active:

```bash
make coverage                                 # what CI runs, plus setup_test
make test                                     # the suite without coverage
python -m pytest -q models/tests/test_configuration.py   # offline, no keys used
python -m pytest -q setup_test.py
```

- The offline tests need no network: `test_configuration.py`,
  `test_exceptions.py`, `test_prompt_templates.py`, and the `*Offline`
  classes in `test_pinecone.py` and `test_hsr.py`
  (`python -m pytest -q models/tests -k "Offline or configuration or exceptions or templates"`).
- `make test` runs `cd models && pytest -v -s tests/`, then
  `python -m setup_test`.
- `setup_test.py` runs `setup.py check` with `sys.executable`, so it needs
  `setuptools` in the venv (it's in `requirements/local.txt`; Python 3.12+
  venvs no longer ship it).

## Writing tests

- One test module per source module, pytest classes named `Test<Thing>`,
  numbered methods (`test_01_...`) where order matters, as in
  `test_pinecone.py`.
- Put test files in `models/tests/mock_data/`.
- **`os.environ.clear()` in `test_configuration.py` wipes the environment for
  the rest of the process.** `settings` was already built at import, so other
  modules still work, but a new test that reads `os.environ` afterwards sees
  nothing. Prefer `@patch.dict(os.environ, {...})`, which restores it.
- A test that creates something in Pinecone deletes it before it ends.
- **Prefer offline tests for new code.** Mock the client, not the code under
  test: `offline_index()` in `test_pinecone.py` builds a `PineconeIndex` around
  a `MagicMock(spec=Pinecone)` without calling its constructor, and
  `TestHybridSearchRetrieverOffline.offline_hsr()` sets `_pinecone` and
  `_chat` to mocks. Patch classes where the module imports them, e.g.
  `patch("models.pinecone.PyPDFLoader")`.
- Pydantic validators in `models/conf.py` can be called directly
  (`Settings.check_pinecone_metric("")`). That's the only way to reach the
  branches for values Pydantic has already converted.

## Coverage

`make coverage` runs the suite under pytest-cov and writes a terminal report,
`htmlcov/` and `coverage.xml`. The settings are in `pyproject.toml`
(`[tool.coverage.*]`): `models/` is measured, `models/tests/` and
`models/yt.py` (the video's script) are omitted, and the run **fails under
95%**. Keep it there: cover new code with tests in the same change.

## CI

`.github/workflows/testsPython.yml` runs on pull requests and pushes to `main`
that touch `*.py`, on Python 3.12, through `.github/actions/tests/python`,
which installs `requirements/local.txt`, writes `.env`, runs `make coverage`
and `python -m setup_test`, then uploads `coverage.xml` to Codecov with the
`CODECOV_TOKEN` repository secret. The README shows the test and Codecov
badges.
