"""Assemble cells/*.py into a self-contained Colab notebook.

Each file in cells/ becomes one code cell, ordered by filename. A leading
module docstring in a cell file becomes a markdown cell above it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS_DIR = HERE / "cells"
OUT = HERE / "jev_vs_modernbert_colab.ipynb"


def split_doc(source: str) -> tuple[str | None, str]:
    """Pull a leading docstring out to use as the markdown header."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, source
    doc = ast.get_docstring(tree)
    if not doc:
        return None, source
    body = source.split('"""', 2)[-1].lstrip("\n")
    return doc, body


def to_lines(text: str) -> list[str]:
    lines = text.rstrip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []


def main() -> None:
    cells = [{
        "cell_type": "markdown",
        "metadata": {},
        "source": to_lines(
            "# Jev vs fine-tuned ModernBERT - parent_queue routing\n\n"
            "Head-to-head on the fixed OpsPilot test split (3,563 tickets, "
            "`random_state=42`), 7-class `clean_v1` taxonomy.\n\n"
            "- **ModernBERT**: `shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1`, "
            "fine-tuned on 16,622 examples\n"
            "- **Jev**: zero-shot, no training data\n\n"
            "Runtime > Change runtime type > **T4 GPU** before running."
        ),
    }]

    for path in sorted(CELLS_DIR.glob("*.py")):
        doc, body = split_doc(path.read_text(encoding="utf-8"))
        if doc:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": to_lines(doc),
            })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_lines(body),
        })

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
