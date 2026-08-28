# Contributing

Thank you for helping ChromaRecover become more reliable.

1. Create a focused branch and keep changes within one pipeline concern when possible.
2. Install development dependencies with `python -m pip install ".[dev]"`. Editable installs
   are optional; on Windows Python 3.10, avoid `-e` when the repository path contains
   non-ASCII characters because the generated `.pth` may be decoded with the wrong locale.
3. Run `python -m pytest` and `python -m ruff check .` before opening a pull request.
4. New algorithms need a deterministic regression case, a hard negative and a note about
   which confidence evidence they add. Correlated duplicate evidence must not raise
   confidence.
5. Public API or schema changes must update `PROJECT_SPEC.md` and `CHANGELOG.md`.

Start with [`docs/contributor-ideas.md`](docs/contributor-ideas.md) for tasks that have a
bounded acceptance test. Use the software-bug form for packaging/API defects and the recovery
failure form for image-specific algorithm behavior. Security-sensitive reports must follow
`SECURITY.md`, not a public issue or pull request.

Do not add private images to the repository. A failure case must have an explicit license
and a `source_group_id`; all variants of one source remain in the same evaluation split.
