import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

project = "TIDE"
copyright = f"{datetime.now():%Y}, TIDE Contributors"
author = "TIDE Contributors"
release = "0.1.0"

extensions = ["myst_parser"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "TIDE Docs"
language = "en"
