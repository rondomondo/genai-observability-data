#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema>=4.23", "pyyaml>=6.0"]
# ///
"""
validate_schema.py - Validate a JSON or YAML file against a JSON Schema.

Reads the target file from --file or stdin; loads the schema from --schema.
Format is auto-detected from the file extension (.json / .yaml / .yml) or
can be forced with --format. When reading from stdin, --format defaults to
yaml (which also accepts plain JSON, since JSON is valid YAML).

Errors go to stderr. A one-line ok summary goes to stdout.
Exits 0 on success, 1 on any validation or parse error.

Usage:
    python3 validate_schema.py --file infra/sdk-config.yaml --schema scripts/schemas/opentelemetry_configuration.json
    cat infra/sdk-config.yaml | python3 validate_schema.py --schema scripts/schemas/opentelemetry_configuration.json
    python3 validate_schema.py --file config.json --schema schema.json --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:
    print("Error: jsonschema is required -- pip install jsonschema", file=sys.stderr)
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: pyyaml is required -- pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def _load(text: str, fmt: str, source: str) -> Any:
    """Load JSON or YAML from a string.

    Args:
        text: raw file content.
        fmt: either "json" or "yaml".
        source: label used in error messages.

    Returns:
        Parsed Python object.

    Raises:
        SystemExit: on parse error.
    """
    try:
        if fmt == "json":
            return json.loads(text)
        return yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"Error: failed to parse {source} as {fmt.upper()} -- {exc}", file=sys.stderr)
        sys.exit(1)


def _detect_format(path: Path) -> str:
    """Return 'json' or 'yaml' based on file suffix."""
    if path.suffix.lower() == ".json":
        return "json"
    return "yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a JSON or YAML file against a JSON Schema")
    parser.add_argument("--file", metavar="FILE", help="Target file to validate (default: read from stdin)")
    parser.add_argument(
        "--schema",
        metavar="FILE",
        required=True,
        help="JSON Schema file to validate against (must be JSON)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        help="Force input format (default: auto-detect from extension, or yaml for stdin)",
    )
    args = parser.parse_args()

    schema_path = Path(args.schema)
    if not schema_path.exists():
        print(f"Error: schema file not found: {args.schema}", file=sys.stderr)
        sys.exit(1)

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: failed to load schema -- {exc}", file=sys.stderr)
        sys.exit(1)

    if args.file:
        target_path = Path(args.file)
        if not target_path.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        fmt = args.format or _detect_format(target_path)
        try:
            raw = target_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Error: could not read {args.file} -- {exc}", file=sys.stderr)
            sys.exit(1)
        source_label = args.file
    else:
        fmt = args.format or "yaml"
        raw = sys.stdin.read()
        source_label = "stdin"

    data = _load(raw, fmt, source_label)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))

    if not errors:
        print(f"  ok: {source_label} is valid against {args.schema}")
        return

    print(f"INVALID: {len(errors)} error(s) in {source_label}\n", file=sys.stderr)
    for err in errors:
        path = ".".join(str(p) for p in err.absolute_path) or "(root)"
        print(f"  {path}: {err.message}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
