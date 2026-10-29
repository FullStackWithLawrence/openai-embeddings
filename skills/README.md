# openai-embeddings Agent Skills

[![Python Unit Tests](https://github.com/FullStackWithLawrence/openai-embeddings/actions/workflows/testsPython.yml/badge.svg?branch=main)](https://github.com/FullStackWithLawrence/openai-embeddings/actions/workflows/testsPython.yml)
[![codecov](https://codecov.io/gh/FullStackWithLawrence/openai-embeddings/graph/badge.svg)](https://codecov.io/gh/FullStackWithLawrence/openai-embeddings)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/FullStackWithLawrence/openai-embeddings/blob/main/.pre-commit-config.yaml)
[![Release Notes](https://img.shields.io/github/release/FullStackWithLawrence/openai-embeddings)](https://github.com/FullStackWithLawrence/openai-embeddings/releases)
[![License: AGPL v3](https://img.shields.io/github/license/FullStackWithLawrence/openai-embeddings)](https://github.com/FullStackWithLawrence/openai-embeddings/blob/main/LICENSE)
[![Agent Skills](https://img.shields.io/badge/Agent%20Skills-8-blueviolet)](https://agentskills.io/specification)

This folder holds the [Agent Skills](https://agentskills.io/specification) for
working on this repository with an AI coding agent such as Claude Code. This
README is written for people.

A skill is a folder of written know-how that an agent loads only when a task
needs it. Each one has a `SKILL.md` file, and sometimes `references/` and
`scripts/` folders. They are developer tooling, like the `Makefile` and
`.pre-commit-config.yaml`. Nothing in `models/` loads them.

The layout follows the skills in
[smarter-sh/smarter](https://github.com/smarter-sh/smarter/tree/main/skills),
which has a longer explanation of why they exist and how to write them.

## What's here

Start with `openai-embeddings-development`. It sets out the rules for every
change, and its table says which narrower skill to load.

| Skill                             | Use it when                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| `openai-embeddings-development`   | You're making any change. Start here.                             |
| `openai-embeddings-testing`       | You're writing or running the pytest tests.                       |
| `openai-embeddings-dependencies`  | You're changing requirements, Python versions, or LangChain code. |
| `openai-embeddings-code-quality`  | You're about to finish a change and need to run the hooks.        |
| `openai-embeddings-docstrings`    | You're writing Python docstrings.                                 |
| `openai-embeddings-logging`       | You're writing or changing a logger call.                         |
| `openai-embeddings-python-typing` | You're writing type annotations, especially with `Any`.           |
| `openai-embeddings-git-commits`   | You've been asked to write a commit message.                      |

The git-commits, docstrings, logging and python-typing skills follow the same
rules as their `smarter-*` counterparts in smarter-sh/smarter. Only the parts
that name Smarter-specific code differ. When a rule changes in one repository,
change it in the other.

## How Claude Code finds them

Claude Code looks for project skills in `.claude/skills/`, not `skills/`. A
committed symlink connects the two:

```bash
# from the repository root
mkdir -p .claude
ln -s ../skills .claude/skills
```

If the symlink breaks, for example on a Windows checkout without symlink
support, Claude Code silently stops seeing the skills.

## When to add or change a skill

- You've corrected an agent on the same thing twice.
- A mistake is costly, such as one that sends billable requests or deletes
  an index.
- A pattern isn't obvious from the code.

Don't add a skill for things the linters or CI already enforce, for one-off
facts (use a comment), or for personal preferences (use `~/.claude/`). Prefer
editing an existing skill to adding one. If you add one, add it to the table
above and to the index in `openai-embeddings-development/SKILL.md`.

Writing guidelines:

- Name the folder `openai-embeddings-<topic>` and set the frontmatter `name`
  to the same value.
- Write the `description` as "Use when ...", naming the concrete situations
  that should trigger it.
- Explain why a rule exists. Point to real files rather than copying code.
- Update a skill in the same pull request that changes what it describes.
  A stale skill misleads.

## Security

A skill is a set of instructions an agent follows, often without asking.
Review changes here as carefully as changes to CI. Watch for instructions
unrelated to the skill's purpose, new scripts or network calls, and text that
tells the agent to skip hooks or confirmations. This repository is public:
never put keys or other secrets in a skill.
