#!/bin/sh
#
# Conventional Commits enforcer
#
# Format: <type>(<scope>): <description>
#
# Common types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
# A '!' suffix (e.g. feat!:) signals a breaking change.
#
# Examples:
#   feat(auth): add OAuth2 login
#   fix: correct null pointer in parser
#   docs(readme): update install steps
#   feat!: drop support for Python 3.10  (breaking change)
#
# Learn more: https://www.conventionalcommits.org
#
MSG=$(cat "$1")
PATTERN="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?(!)?: .{1,}"

if ! echo "$MSG" | grep -qE "$PATTERN"; then
  echo ""
  echo "  ✗ Invalid commit message format!"
  echo ""
  echo "  Expected: <type>(<scope>): <description>"
  echo "  Example:  feat!: drop support for Python 3.10"
  echo ""
  echo "  Types: feat fix docs style refactor perf test build ci chore revert"
  echo ""
  exit 1
fi
