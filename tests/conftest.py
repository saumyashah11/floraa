"""
Shared pytest setup. The FLORA scripts import each other via sys.path
insertion rather than being an installed package (each script does
`sys.path.insert(0, ...)` for its own directory and its dependencies) -- so
tests need the same paths available before any `from flora_xxx import ...`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in (ROOT, ROOT / "module 2", ROOT / "module 3", ROOT / "module 4"):
    sys.path.insert(0, str(p))
