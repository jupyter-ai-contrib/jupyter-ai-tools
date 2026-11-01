import os
import tempfile
from unittest.mock import AsyncMock, patch

import nbformat
import pytest

from jupyter_ai_tools.toolkits.notebook import read_cell

NO_WEB_CLIENT = {
    "success": False,
    "error": "No JupyterLab web client is connected.",
    "error_code": "no_web_client",
}

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _rtc_free():
    with patch("jupyter_ai_tools.toolkits.notebook.rtc_available", return_value=False):
        yield


@pytest.fixture(autouse=True)
def _no_web_client():
    with patch(
        "jupyter_ai_tools.toolkits.notebook.run_lab_command",
        new_callable=AsyncMock,
        return_value=NO_WEB_CLIENT,
    ):
        yield


@pytest.fixture
def notebook_with_output():
    nb = nbformat.v4.new_notebook()
    code_cell = nbformat.v4.new_code_cell(
        source='print("run code")',
        id="code-cell-1",
        execution_count=1,
        outputs=[
            nbformat.v4.new_output(
                output_type="stream",
                name="stdout",
                text="unique_stream_output_value\n",
            )
        ],
    )
    md_cell = nbformat.v4.new_markdown_cell(
        source="# Sample Title",
        id="md-cell-1",
    )
    nb.cells = [code_cell, md_cell]

    fd, path = tempfile.mkstemp(suffix=".ipynb")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    yield path
    if os.path.exists(path):
        os.remove(path)


async def test_read_cell_default_includes_outputs(notebook_with_output):
    result = await read_cell(notebook_with_output, "code-cell-1")
    assert "#### Source" in result
    assert 'print("run code")' in result
    assert "#### Output" in result
    assert "unique_stream_output_value" in result


async def test_read_cell_explicit_include_outputs_true(notebook_with_output):
    result = await read_cell(notebook_with_output, "code-cell-1", include_outputs=True)
    assert "#### Source" in result
    assert 'print("run code")' in result
    assert "#### Output" in result
    assert "unique_stream_output_value" in result


async def test_read_cell_include_outputs_false_excludes_outputs(notebook_with_output):
    result = await read_cell(notebook_with_output, "code-cell-1", include_outputs=False)
    assert "#### Source" in result
    assert 'print("run code")' in result
    assert "#### Output" not in result
    assert "unique_stream_output_value" not in result
