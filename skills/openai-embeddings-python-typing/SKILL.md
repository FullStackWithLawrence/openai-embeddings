---
name: openai-embeddings-python-typing
description: Use when writing or reviewing Python type annotations in openai-embeddings, especially any use of typing.Any, Optional[Any], list[Any], or dict[str, Any], or a type construction that is repeated across the codebase. Follows the Smarter project's rules - when Any is honest and when it hides a finite set of types, TYPE_CHECKING imports, pydantic.JsonValue, @overload, and when to add a shared alias.
---

# openai-embeddings Python Type Annotations

The same rules as Smarter's `smarter-python-typing` skill. An annotation is a
claim about what a value can be. Readers and type checkers (mypy is in
`requirements/local.txt`) take it at its word. `Any` makes the weakest
possible claim and also turns the type checker off for that value, so it's
only correct when the claim is literally true.

## When `Any` is acceptable

Use `Any` only where the value really can be of any type, and no narrower type
would be true.

```python
@property
def dump(self) -> dict[str, Any]:
```

This is fine. The values of the settings dump are strings, numbers, booleans
and nested dicts, and any of them may be added later.

Other honest uses:

- `*args: Any, **kwargs: Any` that are passed through to someone else's API,
  such as `Settings.__init__(self, **data: Any)`.
- Comparison dunders: `def __eq__(self, other: Any) -> bool`.
- Values stored in a generic container, such as a cache.
- Validators and normalizers whose job is to accept untrusted input of any type
  and reject or coerce it, such as the `field_validator`s in `models/conf.py`.
- A value that is only stored, compared or passed on can also be `object`,
  which accepts anything but keeps the type checker on.

## When `Any` is wrong

If you can list the types a value can take, list them. `Any` there hides
information that the reader and the type checker need.

```python
# wrong
def rag(self, human_message: Any) -> Any:

# right
def rag(self, human_message: Union[str, HumanMessage]) -> str:
```

The typical mistakes:

- **A return type with a finite set of possibilities.** Write the `Union`, or
  `Optional[...]` when the alternative is `None`. If callers must then narrow
  it, that's the truth about the function. Consider splitting it into two
  functions, each with one return type.
- **`Optional[Any]`**. `Any` already includes `None`, so it says nothing more
  than `Any`. Either the value has a real type (`Optional[Index]`), or it's
  `Any`.
- **`Any` to dodge a circular import.** Import the class under
  `if TYPE_CHECKING:` and annotate with it (as a string where needed):

  ```python
  from typing import TYPE_CHECKING, Optional

  if TYPE_CHECKING:
      from models.pinecone import PineconeIndex

  def attach(self, index: Optional["PineconeIndex"]) -> None:
  ```

- **A third-party object whose type you didn't look up**, e.g. a Pinecone
  `Index`, an `IndexDescription`, a LangChain `BaseMessage` or `Document`, an
  OpenAI response. Look it up, and import it under `TYPE_CHECKING` if
  importing it at runtime is expensive.
- **Parsed JSON.** A value that is any JSON is `pydantic.JsonValue`, not
  `Any`. A JSON object or list whose shape you know gets a Pydantic model,
  as settings do in `models/conf.py`.
- **A return type that depends on an argument's type.** Use `@overload`. A
  `TypeVar` does the same for a function that returns the very object it was
  given.
- **`:type ...: Any` or `:rtype: Any` in a docstring.** Keep the docstring in
  step with the annotation.

## Shared aliases

Smarter declares a construction that appears more than about ten times in
`smarter.lib.typing` (e.g. `SmarterDictType` for `dict[str, Any]`). This
repository is small and has no alias module yet, so write `dict[str, Any]`
where it's honest.

- A bare `dict` annotation means `dict[str, Any]` here too: spell it out.
- **An alias doesn't make a loose type right.** A dict whose values are all
  strings is `dict[str, str]`, and a structured JSON document is a Pydantic
  model.
- **Add an alias** only when a construction that carries meaning recurs about
  ten times or more across modules. Put it in `models/typing.py` (standard
  library imports only, so any module can import it), give it a docstring,
  and replace the existing uses in the same change.
- **Don't alias** single-level builtin generics of concrete types
  (`list[str]`, `dict[str, str]`) or `Optional[X]`.

## Checking your work

```bash
# Any in annotations (not prose such as "Any exception ..."), in the files you changed
git diff --name-only -- '*.py' | xargs grep -nE '(: *|-> *|\[)Any\b'
```

For each hit, ask: can I list the types this value can have? If so, list them.
