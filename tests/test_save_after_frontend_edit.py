from unittest.mock import AsyncMock

import pytest

from jupyter_ai_tools import utils
from jupyter_ai_tools.toolkits import notebook

pytestmark = pytest.mark.asyncio


async def test_add_cell_waits_for_save(monkeypatch):
    edit = {"success": True, "result": {"cellId": "new-cell"}}
    command = AsyncMock(side_effect=[edit, {"success": True}])
    monkeypatch.setattr(notebook, "rtc_available", lambda: False)
    monkeypatch.setattr(notebook, "run_lab_command", command)
    monkeypatch.setattr(utils, "run_lab_command", command)

    assert await notebook.add_cell("demo.ipynb", "print(42)") == edit
    command.assert_awaited_with(
        "jupyterlab-ai-commands:save-notebook", {"notebookPath": "demo.ipynb"}
    )


async def test_failed_edit_raises_without_saving(monkeypatch):
    command = AsyncMock(return_value={"success": False, "error": "Notebook missing"})
    monkeypatch.setattr(utils, "run_lab_command", command)
    with pytest.raises(RuntimeError, match="Notebook missing"):
        await utils.save_frontend_edit("demo.ipynb", await command("edit"))
    assert command.await_count == 1


async def test_failed_save_preserves_edit_result(monkeypatch):
    command = AsyncMock(return_value={"success": False, "error": "Save conflict"})
    monkeypatch.setattr(utils, "run_lab_command", command)
    result = await utils.save_frontend_edit(
        "demo.ipynb", {"success": True, "result": {"cellId": "new-cell"}}
    )
    assert result["result"]["cellId"] == "new-cell"
    assert result["save_error"] == "Save conflict"
