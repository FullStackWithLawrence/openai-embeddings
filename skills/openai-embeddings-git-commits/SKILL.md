---
name: openai-embeddings-git-commits
description: Use whenever the user asks you to "write a git commit" (or a commit message, or the git add) in the openai-embeddings repo. Covers the never-execute rule, authorship, the multiple -m command format, conventional-commit types and scopes, and how thorough the body must be.
---

# openai-embeddings Git Commits

## The rule: write, never execute

"Write a git commit", in any wording or tone, means **output the command as
text**. It never means run it. Frustrated or emphatic phrasing ("THAT IS NOT A
GIT COMMIT", all caps, profanity) still asks for the command text. Agents have
broken this rule in exactly that situation.

- Never execute `git commit`, `git add`, `git reset`, `git restore --staged`,
  `git push`, `git tag`, `git stash`, or any other git command that writes.
  The user stages, commits, tags and pushes.
- Read-only git commands (`git status`, `git diff`, `git log`) are fine for
  working out what changed.
- Don't volunteer commit messages. Write one only when asked.

## Authorship

You are the author. Lawrence McDaniel is the co-author.

- `--author="Claude <noreply@anthropic.com>"`
- The last `-m` is `Co-authored-by: Lawrence McDaniel <lpm0073@gmail.com>`.

This is specific to this repository. The repositories published to PyPI
(smarter-python, smarter-types, cache-results) are the other way round: Lawrence McDaniel is
the author and you are the co-author, as their own commit skills say. A change
to their authorship rule doesn't apply here.

## Command format

Exactly one `git commit` command in one code block:

```console
git commit --author="Claude <noreply@anthropic.com>" \
  -m "<type>(<scope>): <subject>" \
  -m "<body paragraph>" \
  -m "<body paragraph or bullet list>" \
  -m "Co-authored-by: Lawrence McDaniel <lpm0073@gmail.com>"
```

- Use one `-m` per paragraph. git joins them with blank lines.
- No heredoc, no `-F`, and no `git add` in the same block.
- Avoid backticks, `$` and `!` inside the double-quoted messages. The shell
  would otherwise run or expand them. The user's shell is zsh, where a `!` in
  double quotes is history expansion: `"... !important ..."` aborts the whole
  command with `zsh: event not found: important`, and the commit isn't made.
  Reword instead (e.g. "important declarations"); escaping `!` with a backslash
  leaves the backslash in the message.
- Never write a CI skip instruction anywhere in the message, not even to
  describe one: `[skip ci]`, `[ci skip]`, `[no ci]`, `[skip actions]`,
  `[actions skip]`, or `skip-checks: true`. GitHub reads the whole message of
  a push's head commit, body included, and then starts no workflows: no
  Build, and no Tests on the pull request, whose required checks then wait
  forever. A commit whose body explained the `[skip ci]` problem did exactly
  that. Write "a CI skip instruction" or "semantic-release's skip tag"
  instead.
- When the user asks for the `git add` too, write it as a separate code block.
  List the paths explicitly. Leave out files that don't belong to the change
  (for example, untracked work for another task).

## Subject line

`<type>(<scope>): <subject>`, checked by commitlint
(`@commitlint/config-conventional` + `@commitlint/config-angular`) and read by
semantic-release (`release.config.js`) to version the release.

- **type:** `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `style`,
  `build`, `ci`, or `chore`.
  - `feat` is reserved for a **truly new feature**: a new Django app, a new
    React app, a new SAM resource kind, or another major addition to an existing
    app. `feat` triggers a minor release, so it must be rare. When in doubt, it
    is `fix`.
  - `fix` is a bug fix, and that includes UI and UX defects and other
    shortcomings of something that already exists, even when the fix adds
    controls, options or keyboard shortcuts. `fix` triggers a patch release.
  - Refactoring is `refactor`, never `feat`, even when it moves a feature to a
    new place.
  - Use `chore` for data-only or housekeeping changes, such as example manifests
    or model defaults. Don't use `feat` for them.
  - Anti-example: a log viewer tab that lacked a horizontal scroll bar, a wrap
    toggle and keyboard navigation got all three, plus a filter, copy and full
    screen. Its commit was written as
    `feat(chat): add keyboard navigation and a toolbar to the server logs`,
    and the user corrected it to `fix`: it repaired UI defects in an existing
    tab, however much code it took.
  - Anti-example: the read-only Server Logs window (the
    smarter-terminal-emulator app) became a real bash terminal, with a new
    WebSocket consumer, shell, ticket api and setting. Its commit was written
    as `feat(console): add a superuser bash terminal to the web console`, and
    the user corrected it to a `fix` described as a refactor of the existing
    window: reworking an existing app, even into one that does much more, is
    not a new feature.
- **scope:** the module or area that the change centers on, in lower case:
  `pinecone`, `hsr`, `conf`, `examples`, `tests`, `deps`, `skills`, ... You
  can omit it for changes that span the whole repository.
- **subject:** imperative mood, lower case, no trailing period.
- **length:** the whole header, type and scope included, must be **72
  characters or fewer** (config-angular's `header-max-length`). This is a hard
  limit: the commitlint pre-commit hook rejects a 73-character header, and the
  user has to come back for a new command. Don't count by eye, because that has
  produced 73-character headers more than once. Measure it, and aim for 70 or
  fewer to leave a margin:

  ```console
  printf %s "refactor(account): run add_plugin_examples, create_stackademy at init" | wc -c
  ```

  If it's too long, shorten the subject, for example by naming the area
  instead of listing every command or file. Don't drop the scope just to save
  characters.

- `chore(release): ...` belongs to semantic-release. Never write it yourself.

## Body

Write the body for a reader with no context, such as a reviewer six months from
now. A terse four-bullet body has been rejected as "too brief".

- Start with the problem or motivation: what was wrong or missing, and how it
  showed up.
- Then go area by area, with a short heading paragraph per area (for example
  `Passthrough client (smarter/apps/provider/clients.py):`), followed by one
  bullet per change. Each bullet says what changed and why, especially for
  non-obvious choices and alternatives that were rejected.
- Name files, classes, settings, and commands precisely.
- Cover the tests: what's new, and what they prove.
- End with the current status, such as known skips, pre-existing failures, or
  follow-ups.
- Use good grammar and plain sentences. Add links where they help.

## Before writing

1. Run `git status` and `git diff --stat` (and `git diff` on anything unclear),
   so the message describes the actual working tree, not your memory of it.
2. Ask, or check, which files belong to the commit, if that's ambiguous.
3. Measure the header with `wc -c`, as described in "Subject line", before you
   hand the command over.
4. Start the response with a one-line `Written for:` audience note, because a
   commit message is written for other readers.

See [references/canonical-git-commit.md](references/canonical-git-commit.md) for
a complete, real example.
