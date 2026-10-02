"""
Tests for the save of the notebook after an RTC-free edit in the JupyterLab web client.
"""

from unittest.mock import AsyncMock, patch

import pytest

from jupyter_ai_tools.toolkits.notebook import add_cell, delete_cell, edit_cell, insert_cell

SAVE = "jupyterlab-ai-commands:save-notebook"
OK = {"success": True, "result": {"success": True}}
FAILED = {"success": False, "error": "Cell not found"}
INFO = {"success": True, "result": {"cells": [{"cellId": "cell-1"}]}}

pytestmark = pytest.mark.asyncio

EDITS = {
    "add_cell": lambda: add_cell("nb.ipynb", "x = 1"),
    "insert_cell": lambda: insert_cell("nb.ipynb", "x = 1", insert_index=0),
    "delete_cell": lambda: delete_cell("nb.ipynb", "cell-1"),
    "edit_cell": lambda: edit_cell("nb.ipynb", "cell-1", content="x = 2"),
    "edit_cell_type": lambda: edit_cell("nb.ipynb", "cell-1", cell_type="markdown"),
}


@pytest.fixture(autouse=True)
def _rtc_free():
    with (
        patch("jupyter_ai_tools.toolkits.notebook.rtc_available", return_value=False),
        patch(
            "jupyter_ai_tools.toolkits.notebook.select_cell",
            new_callable=AsyncMock,
            return_value={"success": True},
        ),
    ):
        yield


def frontend(edit_result=OK, save_result=OK):
    """
    Mock the frontend commands, with the given results for the edits and the save.
    """

    async def run(command_id, args=None):
        if command_id == SAVE:
            return save_result
        if command_id.endswith("get-notebook-info"):
            return INFO
        return edit_result

    return patch(
        "jupyter_ai_tools.toolkits.notebook.run_lab_command",
        new_callable=AsyncMock,
        side_effect=run,
    )


def saves(mock):
    return [call.args for call in mock.call_args_list if call.args[0] == SAVE]


@pytest.mark.parametrize("edit", EDITS)
async def test_saves_after_edit(edit):
    with frontend() as mock:
        result = await EDITS[edit]()
    assert result == OK
    assert saves(mock) == [(SAVE, {"notebookPath": "nb.ipynb"})]


@pytest.mark.parametrize("edit", EDITS)
async def test_no_save_after_failed_edit(edit):
    with frontend(edit_result=FAILED) as mock:
        result = await EDITS[edit]()
    assert result == FAILED
    assert saves(mock) == []


@pytest.mark.parametrize("edit", EDITS)
async def test_reports_failed_save(edit):
    with frontend(save_result={"success": False, "error": "Command timed out"}):
        result = await EDITS[edit]()
    assert result["success"]
    assert result["save_error"] == "Command timed out"


async def test_saves_after_content_edit_when_type_change_fails():
    results = iter([OK, FAILED])

    async def run(command_id, args=None):
        return OK if command_id == SAVE else next(results)

    with patch(
        "jupyter_ai_tools.toolkits.notebook.run_lab_command",
        new_callable=AsyncMock,
        side_effect=run,
    ) as mock:
        result = await edit_cell("nb.ipynb", "cell-1", content="x = 2", cell_type="markdown")
    assert result == FAILED
    assert len(saves(mock)) == 1


async def test_no_save_without_edit():
    with frontend() as mock:
        result = await edit_cell("nb.ipynb", "cell-1")
    assert result == {"success": True}
    mock.assert_not_called()
