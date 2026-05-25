#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace "$HOME/.claude"
chown -R "$(id -u):$(id -g)" /workspace "$HOME/.claude" 2>/dev/null || true
sed -i 's|/root/.oh-my-zsh|/home/vscode/.oh-my-zsh|g' ~/.zshrc ~/.bashrc
echo "eval \$(fnm env)" >> "/home/vscode/.zshrc"
echo "==> Container ready. Activate your venv and run 'make install'"
