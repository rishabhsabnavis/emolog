"""Ensure the project root is importable so tests can reach emologcontext and benchmark."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
