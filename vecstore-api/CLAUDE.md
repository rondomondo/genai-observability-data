# Project

This CLAUDE.md applies to all projects and directories (including subdirectories) that contain this file,
unless a path is excluded by `.claudeignore`.

## Context exclusions

A `.claudeignore` file in the project root lists glob patterns for paths Claude should not read or index
(dependencies, build output, secrets, large data files, etc.). Treat `.claudeignore` as authoritative:
if a path matches, do not read or summarise its contents.

## Memory

Project-level memories are stored under `~/.claude/projects/` and loaded automatically. Global preferences
and cross-project rules live in `~/.claude/`. When a task references prior decisions or past context,
check those memory files before asking the user to re-explain.

## Writing style

- Never use em dashes (--); use commas, parentheses, hyphens, or separate sentences instead
- British spelling: `realise` not `realize`, `colour` not `color`, etc.
- Stay within the ASCII character set
- No comments explaining what code does -- only why (hidden constraints, non-obvious invariants)
- No unnecessary `**kwargs` silently ignored without a docstring note

## Code style

- `ruff` for linting/formatting (`line-length = 100`)
- `mypy` with `strict = false` and `ignore_missing_imports = true`
- Line length: 100 characters max
- Python 3.12+ with full type annotations
- Import order: standard lib, third-party, local
- Naming: `snake_case` for functions/variables, `PascalCase` for classes
- Use Pydantic v2 for data validation and schemas
- Use FastAPI for HTTP endpoints
- Use `pathlib.Path` throughout; never raw string paths
- Use `copy.deepcopy` before mutating shared data structures
- All file I/O must use `encoding='utf-8'`; JSON output uses `ensure_ascii=False`
- Magic numbers must be named constants (`ALL_CAPS`)
- Private helpers are prefixed with `_`; public API is clearly separated

## Type hints

- All public function signatures must have full type hints
- Use modern union syntax: `str | None`, `list[str]`, `dict[str, Any]` -- not `Optional[str]` or `List[str]`
- `from __future__ import annotations` is not currently used; explicit imports from `typing` are required
  only for constructs unavailable as builtins (e.g. `Any`, `Union` in complex cases)
- Buffer parameters should be typed `BytesIO | bytes | str | Path` as appropriate -- not `Optional[bytes]`

## Async

- Prefer `async`/`await` for all I/O-bound work (network calls, file reads in hot paths)
- Use `httpx` (async-capable) instead of `requests` for HTTP calls
- Entry points use `asyncio.run(main())` -- do not call `asyncio.get_event_loop()` directly
- Synchronous CLI scripts that do minimal I/O are acceptable without async

## Logging

- Library and API code must use the `logging` module -- never `print` for diagnostics
- CLI scripts may use `print(..., file=sys.stderr)` for errors and warnings, `print()` for output data
- Do not configure logging inside library modules; leave that to the caller or entry point
- Log levels: `DEBUG` for trace/detail, `INFO` for progress, `WARNING` for recoverable issues, `ERROR` for failures

## Error handling

- Catch specific exceptions -- never bare `except:` or `except Exception:` without re-raising or logging
- Validate inputs early; fail fast with a clear message before doing any work
- CLI scripts exit with code 0 on success, 1 on failure -- use `sys.exit(1)` not `raise SystemExit`
- Optional third-party dependencies are imported inside a `try/except ImportError` block with a fallback or
  a clear error message

## CLI scripts

- Use `argparse` with grouped arguments and sensible defaults
- Validate all arguments in `main()` before any processing begins
- Scripts designed for pipeline composition read from stdin, write data to stdout, and send errors to stderr
- Entry point: `if __name__ == "__main__": main()`

## Section comments

Use plain section headings without decorator lines:

```python
# Section title
```

Not:

```python
# ---------------------------------------------------------------------------
# Section title
# ---------------------------------------------------------------------------
```

The dashes cause rendering problems in some editors and CI log viewers.

## Docstrings: Google style

All public functions must have Google-style docstrings.

```python
def fn(url: str, overwrite: bool = False) -> dict:
    """One-line summary

    Optional longer description when intent is non-obvious.

    Args:
        url: The redirect destination URL.
        overwrite: When True, silently replace an existing key.

    Returns:
        A dict suitable for passing to put_object as extra kwargs.

    Raises:
        ValueError: If url uses an unsupported scheme.
    """
```

Rules:

- First line is a short imperative summary with no trailing period
- `Args:`, `Returns:`, `Raises:` sections only when present
- Parameter descriptions start lowercase
- Omit obvious sections (no `Returns:` on a `None`-returning function)

## Running tests

```bash
make test
# or directly:
uv run --extra dev pytest
```

Test location and `pythonpath` settings are configured in `pyproject.toml`.

## Project structure

```
app/
  config.py          # Pydantic settings
  dependencies.py    # FastAPI dependency injection
  logging_config.py  # structlog setup
  main.py            # FastAPI app factory
  middleware/        # auth, correlation-id
  models/            # Pydantic request/response schemas
  routers/           # FastAPI route handlers
  services/          # Business logic (embedder, vectorstore, cache, etc.)
tests/
  conftest.py        # shared fixtures
  test_routers/      # HTTP-layer tests (TestClient)
  test_services/     # service unit tests
```
