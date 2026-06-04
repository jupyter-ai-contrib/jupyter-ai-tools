"""Tests for _get_cell_index_from_id_ydoc against a real jupyter_ydoc.YNotebook.

These tests exercise the helper directly (no mocks) to verify it works against
the upstream YNotebook from jupyter_ydoc, which does not define a `find_cell`
method. The previous implementation called `ydoc.find_cell(...)` and silently
returned None on AttributeError, making every edit_cell/delete_cell/move_cell
call fail with a misleading "cell not found" error on installs without
jupyter-server-documents.

Closes #29.
"""

from jupyter_ydoc import YNotebook

from jupyter_ai_tools.toolkits.notebook import _get_cell_index_from_id_ydoc


def _populate(ydoc, cells):
    ydoc.set(
        {
            "cells": cells,
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def test_returns_index_when_cell_id_matches():
    ydoc = YNotebook()
    _populate(
        ydoc,
        [
            {
                "cell_type": "code",
                "source": "x = 1",
                "id": "first",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            },
            {"cell_type": "markdown", "source": "# title", "id": "second", "metadata": {}},
            {
                "cell_type": "code",
                "source": "x = 2",
                "id": "third",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            },
        ],
    )

    assert _get_cell_index_from_id_ydoc(ydoc, "first") == 0
    assert _get_cell_index_from_id_ydoc(ydoc, "second") == 1
    assert _get_cell_index_from_id_ydoc(ydoc, "third") == 2


def test_returns_none_when_cell_id_missing():
    ydoc = YNotebook()
    _populate(
        ydoc,
        [
            {
                "cell_type": "code",
                "source": "x = 1",
                "id": "only",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            },
        ],
    )

    assert _get_cell_index_from_id_ydoc(ydoc, "nonexistent") is None


def test_returns_none_for_empty_notebook():
    ydoc = YNotebook()
    _populate(ydoc, [])

    assert _get_cell_index_from_id_ydoc(ydoc, "anything") is None
