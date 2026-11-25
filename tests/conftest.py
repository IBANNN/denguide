"""Point WORK_DIR at a temporary folder before any src module is imported."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ["DENGUIDE_DIR"] = tempfile.mkdtemp(prefix="denguide_test_")
