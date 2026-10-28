---
name: openai-embeddings-code-quality
description: Use before finishing any openai-embeddings code change, to lint and format with the repository's pre-commit hooks (codespell, prettier, black, flake8, isort, pylint, bandit, and the pre-commit-hooks checks). Explains which tools are authoritative, their configuration, and how to run them on changed files under zsh.
---

# openai-embeddings Code Quality

`.pre-commit-config.yaml` is the standard. If a hook passes, the code meets
it. Never bypass a hook (`--no-verify`, `SKIP=`, blanket `# noqa`).

## What the hooks enforce

| Hook             | Configuration                                                                                     | Notes                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| black            | `pyproject.toml`, line length 120                                                                 | Formatter. Let it reformat.                                |
| isort            | `pyproject.toml`, `profile = "black"`, 2 lines after imports                                      |                                                            |
| flake8           | `.flake8`                                                                                         | Selects **only C101**, the `# -*- coding: utf-8 -*-` line. |
| pylint           | `.pylintrc`, via `scripts/run_pylint.sh`                                                          | Runs in the active venv, so it needs `make init` first.    |
| bandit           | `-ll`                                                                                             | Medium or high severity findings fail.                     |
| codespell        | `codespell.txt` ignore list                                                                       | Add real project words to `codespell.txt`.                 |
| prettier         |                                                                                                   | Markdown, YAML, JSON.                                      |
| pre-commit-hooks | encoding pragma, whitespace, EOF, YAML/JSON/TOML, private keys, AWS credentials, debug statements |                                                            |
| commitlint       | `commitlint.config.js`                                                                            | On commit messages. See `openai-embeddings-git-commits`.   |

`fix-encoding-pragma` adds the coding line to new Python files, so the first
run on a new file fails and the second passes.

## How to run

From the repository root, with the venv active:

```bash
git diff --name-only | xargs pre-commit run --files
git ls-files --others --exclude-standard | xargs pre-commit run --files   # new files
```

- Use `xargs`. Under zsh, `pre-commit run --files $FILES` passes one argument,
  and pre-commit checks nothing.
- Hooks that modify files fail the first run, then pass. Run twice and read
  the diff.
- `make lint` runs every hook on every file, then pylint, flake8, isort and
  black on the whole tree. It touches files you didn't change.
- `default_language_version` is `python3.12`, so pre-commit needs a
  `python3.12` on the PATH even when the venv is 3.14.

## Principles

- Match the surrounding code: naming, docstrings, logging.
- Fix the cause of a finding. Suppress only with a narrow, commented
  directive (`# pylint: disable=broad-except` where a test wraps a call to
  report its exception).
- Don't reformat code you didn't otherwise change.
