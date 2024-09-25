"""Dataset classes for citation networks."""

from pathlib import Path

DIR = Path("Data") / "dblp_v14.json"

KEYS_TO_KEEP = [
    "id",
    "title",
    "keywords",
    "year",
    "n_citation",
    "abstract",
    "authors",
    "doc_type",
    "references",
]
