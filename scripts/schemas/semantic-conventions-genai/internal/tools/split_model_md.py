#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Rewrite cross-links from the merged ``model.md`` to per-namespace pages.

Background
----------
With Weaver's ``definition/2`` layout, attribute defs live at the top-level
``attributes:`` list and Weaver renders the entire ``./model`` registry as a
single ``docs/registry/attributes/model.md`` page (named after the registry
root) rather than the per-namespace ``gen-ai.md`` / ``mcp.md`` / ``openai.md``
pages the legacy layout produced. That collapses every cross-link in the docs
to point at ``model.md``, which produces a noisy diff that drowns out real
semantic changes.

This script is a review-aid only. It rewrites generated links in ``docs/`` so
they target the same per-namespace paths v1 produced, and restores the
per-namespace bullet list in ``docs/registry/attributes/README.md``. The
merged ``model.md`` itself is left in place. Nothing here changes the actual
generation pipeline -- it just reduces diff noise so real changes are easier
to review.

Run after ``make generate-registry``::

    uv run internal/tools/split_model_md.py

Idempotent: rerunning is a no-op once no ``model.md`` references remain.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
MODEL_MD = DOCS_DIR / "registry" / "attributes" / "model.md"
ATTR_README = DOCS_DIR / "registry" / "attributes" / "README.md"

# Map attribute-key namespace prefix -> per-namespace doc filename.
NAMESPACE_PAGE = {
    "gen_ai": "gen-ai.md",
    "mcp": "mcp.md",
    "openai": "openai.md",
}

# Match a markdown link whose text is `<key>` in backticks and whose target
# is .../docs/registry/attributes/model.md (with or without anchor). The link
# target may be absolute (/docs/...) or any path ending in
# registry/attributes/model.md.
LINK_RE = re.compile(
    r"\[`(?P<key>[a-zA-Z_][\w.]*)`\]\((?P<prefix>[^)]*?registry/attributes/)model\.md(?P<anchor>#[^)]*)?\)"
)

# Match the single "Model" bullet the v2 template emits in attributes/README.md.
# Anchored with `[ \t\r]*` (not `\s*`) so we don't swallow the trailing newline
# and the blank line that follows on the next line.
README_MODEL_LINE_RE = re.compile(r"^- \[Model\]\(model\.md\)[ \t\r]*$", re.MULTILINE)
README_RESTORED = (
    "- [Gen AI](gen-ai.md)\n"
    "- [MCP](mcp.md)\n"
    "- [OpenAI](openai.md)"
)


def _page_for_key(key: str) -> str | None:
    prefix = key.split(".", 1)[0]
    return NAMESPACE_PAGE.get(prefix)


def _rewrite_links(text: str, path: Path, unmapped: list[tuple[Path, str]]) -> tuple[str, int]:
    count = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal count
        key = m.group("key")
        page = _page_for_key(key)
        if page is None:
            unmapped.append((path, key))
            return m.group(0)
        count += 1
        # Drop any anchor: the v1 per-namespace pages link without anchors,
        # which is what the existing checked-in references look like.
        return f"[`{key}`]({m.group('prefix')}{page})"

    return LINK_RE.sub(sub, text), count


def main() -> int:
    if not DOCS_DIR.is_dir():
        print(f"error: {DOCS_DIR} not found", file=sys.stderr)
        return 1

    unmapped: list[tuple[Path, str]] = []
    total_links = 0
    files_changed = 0

    for md in sorted(DOCS_DIR.rglob("*.md")):
        if md == MODEL_MD:
            continue
        original = md.read_text(encoding="utf-8")
        rewritten, n = _rewrite_links(original, md, unmapped)
        if n:
            md.write_text(rewritten, encoding="utf-8")
            files_changed += 1
            total_links += n

    # Restore the namespace bullet list in attributes/README.md.
    if ATTR_README.exists():
        text = ATTR_README.read_text(encoding="utf-8")
        new_text, n = README_MODEL_LINE_RE.subn(README_RESTORED, text, count=1)
        if n:
            ATTR_README.write_text(new_text, encoding="utf-8")
            files_changed += 1

    print(
        f"split_model_md: rewrote {total_links} link(s) across {files_changed} file(s)"
    )
    if unmapped:
        print(
            f"split_model_md: warning: {len(unmapped)} link(s) had unmapped key prefixes:",
            file=sys.stderr,
        )
        for path, key in unmapped:
            print(f"  {path.relative_to(REPO_ROOT)}: {key}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
