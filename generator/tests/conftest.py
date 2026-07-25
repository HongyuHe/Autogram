"""Ensure the emulator package is importable when tests run from anywhere."""

from __future__ import annotations

import sys
from pathlib import Path

# Put ``generator/`` (the parent of this tests dir) on sys.path so ``gtib_emulator``
# imports without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
