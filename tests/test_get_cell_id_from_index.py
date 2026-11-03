"""Tests for mapping JupyterLab cell numbers to notebook cell IDs."""

from unittest.mock import AsyncMock, patch

import pytest

from jupyter_ai_tools.toolkits.notebook import get_cell_id_from_index


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cell_number", "expected_id"),
    [(1, "first"), (3, "third")],
)
async def test_uses_one_based_cell_number(cell_number, expected_id):
    notebook = {"cells": [{"id": "first"}, {"id": "second"}, {"id": "third"}]}

    with patch(
        "jupyter_ai_tools.toolkits.notebook.read_notebook_json",
        new=AsyncMock(return_value=notebook),
    ):
        assert await get_cell_id_from_index("notebook.ipynb", cell_number) == expected_id


@pytest.mark.asyncio
@pytest.mark.parametrize("cell_number", [0, -1, 4])
async def test_rejects_invalid_one_based_cell_number(cell_number):
    notebook = {"cells": [{"id": "first"}, {"id": "second"}, {"id": "third"}]}

    with patch(
        "jupyter_ai_tools.toolkits.notebook.read_notebook_json",
        new=AsyncMock(return_value=notebook),
    ):
        with pytest.raises(ValueError, match="No cell_id found"):
            await get_cell_id_from_index("notebook.ipynb", cell_number)
