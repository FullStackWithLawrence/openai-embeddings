---
name: openai-embeddings-logging
description: Use when writing or changing any logger call in openai-embeddings Python code. Follows the Smarter project's logging style - every message starts with a dot-notated prefix that traces where it came from (logger_prefix, then the function name), uses lazy %-style arguments, picks its level deliberately, and never logs secrets such as the OpenAI or Pinecone API keys.
---

# openai-embeddings Logging Style

The same rules as Smarter's `smarter-logging` skill. This repository has no
`SmarterFormattedText` or waffle switches, so the prefix is plain text and
every module uses the standard library logger.

Existing code predates these rules (bare `logging.debug(...)` calls and
`print()` in `models/pinecone.py`). Apply them to log calls you write or
change.

## The rule: every message starts with its origin

Each message's first argument is the dot-notated path to the code that logged
it, followed by the function or method name. Together they read like a stack
trace: `models.pinecone.PineconeIndex.create() ...`. A message without the
prefix can't be traced, and is a defect in review.

```python
import logging

from models.conf import settings

logging.basicConfig(level=logging.DEBUG if settings.debug_mode else logging.INFO)
logger = logging.getLogger(__name__)
logger_prefix = __name__


def load(filepath: str) -> None:
    logger.info("%s.load() loading %s", logger_prefix, filepath)


class PineconeIndex:
    def create(self) -> None:
        logger.info("%s.PineconeIndex.create() creating index %s", logger_prefix, self.index_name)
```

- Use `logger_prefix = __name__`, never a hard-coded module path, which goes
  stale when the module moves.
- Write the function name with `()`, and keep it right when you rename the
  function.
- Log through the module's `logger`, not the root `logging.info()`, and not
  `print()`. The command-line examples in `models/examples/` print their
  result for the user; that's output, not logging.

## Arguments, not f-strings

Pass values as %-style arguments, so that a message that isn't emitted is never
formatted, and so that log aggregation can group messages by their template:

```python
logger.info("%s.rag() retrieved %s documents", logger_prefix, len(documents))  # yes
logger.info(f"{logger_prefix}.rag() retrieved {len(documents)} documents")     # no
```

## Levels

| Level     | Use it for                                                                                   |
| --------- | -------------------------------------------------------------------------------------------- |
| `debug`   | Detail for a developer tracing a flow: arguments, intermediate values, index stats, prompts. |
| `info`    | A normal event worth a line: an index created or deleted, a PDF loaded, a request finished.  |
| `warning` | Something unusual that was handled: a fallback, a retry, a missing optional setting.         |
| `error`   | Something failed and needs attention: a request or load that did not do its job.             |

- An expected answer is not an error. A check that answers no must not log at
  ERROR.
- Log an exception once, where it is handled, not at every level it passes
  through. Include it in the message (`"...: %s", e`), or use
  `logger.exception()` inside an `except` when the traceback matters.
- `DEBUG_MODE` in `.env` switches the level between DEBUG and INFO.

## Never log secrets

The repository is public. Never log an API key, a `SecretStr`'s
`.get_secret_value()`, an `Authorization` header, or a whole settings object
or `settings.dump` with secrets in it. Log the key's source
(`settings.openai_api_key_source`) instead. When a credential must be
identified, mask it, e.g. `key[:4] + "****"`.

## Tests

Assert on a log call by patching the module's `logger`, not by capturing output:

```python
with patch.object(pinecone_module, "logger") as logger:
    index.delete()
logger.info.assert_called_once()
```

## Checklist

- Every new or changed log call starts with `logger_prefix`, then
  `.function_name()` (or `.ClassName.method()`).
- %-style arguments, no f-strings.
- The level matches the table above.
- Nothing secret is in the message or its arguments.
