"""Repository-wide pytest configuration."""

import importlib.util

# The figures module needs the optional matplotlib; skip its doctest collection without it.
collect_ignore = [] if importlib.util.find_spec("matplotlib") else ["src/relu_tm/viz.py"]
