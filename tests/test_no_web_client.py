"""Tests for the RTC-free tools when no JupyterLab web client is connected.

jupyterlab-commands-toolkit then reports ``error_code: "no_web_client"``, and
the tools fall back to the notebook file on disk, which is the only copy of
the notebook in that case.
"""

import os
import tempfile
from unittest.mock import AsyncMock, patch

import nbformat
import pytest

from jupyter_ai_tools.toolkits.notebook import (
    add_cell,
    create_notebook,
    delete_cell,
    edit_cell,
    get_active_cell_id,
    get_active_notebook,
    insert_cell,
    read_notebook_json,
    select_cell,
)

NO_WEB_CLIENT = {
    "success": False,
    "error": "No JupyterLab web client is connected.",
    "error_code": "no_web_client",
}

COMMAND_ERROR = {"success": False, "error": "Cell not found"}

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _rtc_free():
    with patch("jupyter_ai_tools.toolkits.notebook.rtc_available", return_value=False):
        yield


@pytest.fixture
def no_web_client():
    """Make every frontend command report that no web client received it."""
    with patch(
        "jupyter_ai_tools.toolkits.notebook.run_lab_command",
        new_callable=AsyncMock,
        return_value=NO_WEB_CLIENT,
    ) as mock:
        yield mock


@pytest.fixture
def command_error():
    """Make every frontend command fail for another reason."""
    with patch(
        "jupyter_ai_tools.toolkits.notebook.run_lab_command",
        new_callable=AsyncMock,
        return_value=COMMAND_ERROR,
    ) as mock:
        yield mock


@pytest.fixture
def notebook_path():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("print(1)", id="cell-1"),
        nbformat.v4.new_markdown_cell("# Title", id="cell-2"),
    ]
    fd, path = tempfile.mkstemp(suffix=".ipynb")
    with os.fdopen(fd, "w") as f:
        nbformat.write(nb, f)
    yield path
    os.remove(path)


def _cells(path):
    with open(path) as f:
        return nbformat.read(f, as_version=nbformat.NO_CONVERT).cells


async def test_read_notebook_json_reads_the_file(no_web_client, notebook_path):
    notebook = await read_notebook_json(notebook_path)
    assert [cell["id"] for cell in notebook["cells"]] == ["cell-1", "cell-2"]


async def test_read_notebook_json_raises_command_error(command_error, notebook_path):
    with pytest.raises(RuntimeError, match="Cell not found"):
        await read_notebook_json(notebook_path)


async def test_add_cell_edits_the_file(no_web_client, notebook_path):
    result = await add_cell(notebook_path, "x = 1", cell_id="cell-1", cell_type="code")
    assert result["success"]
    assert "on disk" in result["result"]["message"]
    cells = _cells(notebook_path)
    assert [cell.source for cell in cells] == ["print(1)", "x = 1", "# Title"]
    assert result["result"]["cellId"] == cells[1].id


async def test_add_cell_above(no_web_client, notebook_path):
    await add_cell(notebook_path, "x = 1", cell_id="cell-1", add_above=True)
    assert [cell.source for cell in _cells(notebook_path)] == ["x = 1", "print(1)", "# Title"]


async def test_add_cell_returns_command_error(command_error, notebook_path):
    result = await add_cell(notebook_path, "x = 1")
    assert result == COMMAND_ERROR
    assert len(_cells(notebook_path)) == 2


async def test_insert_cell_edits_the_file(no_web_client, notebook_path):
    result = await insert_cell(notebook_path, "raw", insert_index=1, cell_type="raw")
    assert result["success"]
    cells = _cells(notebook_path)
    assert [cell.source for cell in cells] == ["print(1)", "raw", "# Title"]
    assert cells[1].cell_type == "raw"


async def test_insert_cell_appends_without_index(no_web_client, notebook_path):
    await insert_cell(notebook_path, "last")
    assert [cell.source for cell in _cells(notebook_path)][-1] == "last"


async def test_insert_cell_returns_command_error(command_error, notebook_path):
    result = await insert_cell(notebook_path, "x = 1", insert_index=0)
    assert result == COMMAND_ERROR
    assert len(_cells(notebook_path)) == 2


async def test_delete_cell_edits_the_file(no_web_client, notebook_path):
    result = await delete_cell(notebook_path, "cell-1")
    assert result["success"]
    assert [cell.id for cell in _cells(notebook_path)] == ["cell-2"]


async def test_delete_unknown_cell(no_web_client, notebook_path):
    with pytest.raises(ValueError, match="cell-3"):
        await delete_cell(notebook_path, "cell-3")


async def test_edit_cell_content_and_type(no_web_client, notebook_path):
    result = await edit_cell(notebook_path, "cell-1", content="# Heading", cell_type="markdown")
    assert result["success"]
    cell = _cells(notebook_path)[0]
    assert cell.id == "cell-1"
    assert cell.cell_type == "markdown"
    assert cell.source == "# Heading"
    assert "outputs" not in cell


async def test_edit_cell_type_only(no_web_client, notebook_path):
    result = await edit_cell(notebook_path, "cell-2", cell_type="code")
    assert result["success"]
    cell = _cells(notebook_path)[1]
    assert cell.cell_type == "code"
    assert cell.source == "# Title"
    assert cell.outputs == []


async def test_edit_cell_returns_command_error(command_error, notebook_path):
    result = await edit_cell(notebook_path, "cell-1", content="x", cell_type="markdown")
    assert result == COMMAND_ERROR
    assert _cells(notebook_path)[0].source == "print(1)"


async def test_get_active_notebook_raises(no_web_client):
    with pytest.raises(RuntimeError, match="No JupyterLab web client"):
        await get_active_notebook()


async def test_get_active_cell_id_raises(no_web_client, notebook_path):
    with pytest.raises(RuntimeError, match="No JupyterLab web client"):
        await get_active_cell_id(notebook_path)


async def test_select_cell_returns_the_error(no_web_client, notebook_path):
    result = await select_cell("cell-1", file_path=notebook_path)
    assert result == NO_WEB_CLIENT


async def test_create_notebook_reports_not_opened(notebook_path):
    os.remove(notebook_path)
    with patch(
        "jupyter_ai_tools.toolkits.jupyterlab.execute_command",
        new_callable=AsyncMock,
        return_value=NO_WEB_CLIENT,
    ):
        message = await create_notebook(notebook_path)
    assert message.startswith("Successfully created notebook")
    assert "not open in JupyterLab" in message
    assert len(_cells(notebook_path)) == 0
