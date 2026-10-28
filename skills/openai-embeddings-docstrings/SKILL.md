---
name: openai-embeddings-docstrings
description: Use when writing or editing Python docstrings in openai-embeddings. Docstrings follow the same reStructuredText field-list style as the Smarter project (:param:, :returns:, :raises:), with a one-line summary, cross-reference syntax, and the layout that pydocstringformatter enforces.
---

# openai-embeddings Python Docstrings

The same rules as Smarter's `smarter-docstrings` skill. Docstrings explain the
code to developers, and in this repository to students learning from it.
Write for a reader who is new to LangChain, Pinecone and embeddings.

Much of the existing code has short one-line docstrings that predate these
rules. Apply the rules to code you write or change; don't rewrite docstrings
you didn't otherwise touch.

## Style: reStructuredText field lists

Use reST fields, not Google style (`Args:`, `Returns:`):

```python
def pdf_loader(self, filepath: str) -> None:
    """
    Embed every PDF in a folder and upsert the vectors into the Pinecone index.

    :param filepath: A folder of PDF files, or a single PDF file.
    :raises FileNotFoundError: when ``filepath`` doesn't exist.
    """
```

- `:param name:` for each parameter. Types come from the annotations, so don't
  repeat them in `:type:`.
- `:returns:` says what the value means, including `None` cases. Use `:rtype:`
  only when there is no annotation.
- `:raises SomeError: when ...` for each exception that callers should handle.
- Indent continuation lines by four spaces.

## Layout (pydocstringformatter enforces it)

- Opening `"""` on its own line, then a **one-line summary that ends with a
  period**, then a blank line, then the body.
- **Keep the summary to one line.** pydocstringformatter puts a period at the
  end of the first line and splits the body after it, so a two-line summary is
  mangled into "Queue the task, which." followed by the rest.
- A body paragraph starts with a capital letter. A body that starts lower case
  usually means a summary was split.
- Use one blank line between paragraphs, and between the body and the fields.
- **pydocstringformatter capitalizes the first word.** A summary that starts
  with a name, e.g. `rag() returns ...`, becomes `Rag() returns ...`. Put the
  name in double backticks, which the formatter leaves alone, or rephrase.
  Start a module docstring with a sentence, not the module's dotted name.

## Content

- **Module docstrings** say what the module is for and how it fits in.
- **Class docstrings** say what the class represents and its role. Larger
  classes can use bold pseudo-headings such as `**Features:**` and
  `**Methods:**`.
- **Functions and methods** cover purpose, parameters, return value,
  exceptions, and side effects: Pinecone indexes created, deleted or written,
  OpenAI requests sent (they're billable), cache entries stored.
- Say why, not only what, when the reason isn't obvious.
- Be accurate. A docstring that describes old behavior is worse than none.
  When you change behavior, update the docstring in the same edit.

## reST inside docstrings

- `double backticks` for literals: field names, values, commands, file
  names.
- Cross-references: ``:class:`~models.pinecone.PineconeIndex` ``,
  ``:meth:`~models.hybrid_search_retriever.HybridSearchRetriever.rag` ``, and
  `:func:`, `:attr:`, `:mod:`. The `~` shows only the last component.
- Example code: `.. code-block:: python`, then a blank line and indented
  content, or `::` at the end of a paragraph.
- Lists need a blank line before them.

## Test docstrings

A test module has a one-line docstring naming what it tests
(`"""Test PineconeIndex."""`). A test method has a one-line docstring stating
the behavior it proves, when the name alone isn't clear:
`"""Create() builds a serverless index with the configured dimensions."""`.

## Verify

Run the pre-commit hooks on the file (`openai-embeddings-code-quality`) and
review their diff.
