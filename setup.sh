#!/bin/bash
# setup.sh  —  one-step environment setup for Linux / macOS / WSL
#
# Creates a Python virtual environment in .venv and installs all
# dependencies listed in requirements.txt.
#
# Usage:
#   bash setup.sh
#
# Then activate it with:
#   source .venv/bin/activate
#
# (On Debian/Ubuntu you may first need:  sudo apt install python3-venv)

set -e
cd "$(dirname "$0")"

# Pick an available python (python3 preferred).
PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python

echo ">> Creating virtual environment in .venv ..."
"$PY" -m venv .venv

echo ">> Upgrading pip ..."
.venv/bin/pip install --upgrade pip

echo ">> Installing requirements ..."
.venv/bin/pip install -r requirements.txt

# Optional: only needed to compare against the authors' package
# (python run.py --stage official)
# .venv/bin/pip install gosdt

echo ""
echo ">> Done. Activate the environment with:"
echo "     source .venv/bin/activate"
echo "   or run directly, e.g.:"
echo "     .venv/bin/python run.py --stage check"
