"""Locations shared by graph runtime helpers and offline tools."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONTENT_DIR = PROJECT_DIR / "graph_data"
BINDINGS_DIR = PROJECT_DIR / "graph_bindings"
REPORTS_DIR = PROJECT_DIR / "graph_reports"
SOURCE_FILE = CONTENT_DIR / "sources" / "神经网络与深度学习.md"
