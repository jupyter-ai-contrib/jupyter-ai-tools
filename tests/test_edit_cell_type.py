"""Tests for edit_cell() cell_type parameter."""

import os
import tempfile
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import nbformat
import pytest

from jupyter_ai_tools.toolkits.notebook import edit_cell


# ── Helpers ──


def _make_notebook(*cells):
    """Create a notebook file with the given cells and return its path."""
    nb = nbformat.v4.new_notebook()
    nb.cells = list(cells)
    fd, path = tempfile.mkstemp(suffix=".ipynb")
    with os.fdopen(fd, "w") as f:
        nbformat.write(nb, f)
    return path


def _read_notebook(path):
    with open(path, "r") as f:
        return nbformat.read(f, as_version=nbformat.NO_CONVERT)


@contextmanager
def _nbformat_path():
    """Patch get_file_id and get_jupyter_ydoc so edit_cell takes the nbformat path."""
    with patch("jupyter_ai_tools.toolkits.notebook.get_file_id", new_callable=AsyncMock), \
         patch("jupyter_ai_tools.toolkits.notebook.get_jupyter_ydoc", new_callable=AsyncMock, return_value=None):
        yield


@contextmanager
def _ydoc_path(ydoc, resolved_cell_id="cell-1", cell_index=0):
    """Patch dependencies so edit_cell takes the YDoc path with the given mock."""
    with patch("jupyter_ai_tools.toolkits.notebook.get_file_id", new_callable=AsyncMock), \
         patch("jupyter_ai_tools.toolkits.notebook.get_jupyter_ydoc", new_callable=AsyncMock, return_value=ydoc), \
         patch("jupyter_ai_tools.toolkits.notebook._get_cell_index_from_id_ydoc", return_value=cell_index), \
         patch("jupyter_ai_tools.toolkits.notebook.normalize_filepath", side_effect=lambda x: x), \
         patch("jupyter_ai_tools.toolkits.notebook._resolve_cell_id", new_callable=AsyncMock, return_value=resolved_cell_id):
        yield


def _make_mock_ydoc(cell_type="code", source="print('hello')", cell_id="cell-1", metadata=None):
    """Build a mock YDoc with a single cell."""
    ydoc = MagicMock()
    ycell = MagicMock()
    cell_py = {
        "cell_type": cell_type,
        "source": source,
        "id": cell_id,
        "metadata": metadata or {},
    }
    if cell_type == "code":
        cell_py["outputs"] = []
        cell_py["execution_count"] = None
    ycell.to_py.return_value = cell_py
    ydoc._ycells = [ycell]
    ydoc.set_cell = MagicMock()
    return ydoc, ycell


# ── Fixtures ──


@pytest.fixture
def code_cell():
    cell = nbformat.v4.new_code_cell(source="print('hello')")
    cell.id = "cell-1"
    return cell


@pytest.fixture
def markdown_cell():
    cell = nbformat.v4.new_markdown_cell(source="# Title")
    cell.id = "cell-2"
    return cell


@pytest.fixture
def raw_cell():
    cell = nbformat.v4.new_raw_cell(source="raw text")
    cell.id = "cell-3"
    return cell


# ── nbformat path tests ──


class TestEditCellTypeNbformat:
    """Tests for cell type changes via the nbformat (filesystem) path."""

    @pytest.mark.asyncio
    async def test_code_to_markdown(self, code_cell):
        path = _make_notebook(code_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-1", cell_type="markdown")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "markdown"
            assert cell.source == "print('hello')"
            assert cell.id == "cell-1"
            assert "outputs" not in cell
            assert "execution_count" not in cell
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_code_to_raw(self, code_cell):
        path = _make_notebook(code_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-1", cell_type="raw")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "raw"
            assert cell.source == "print('hello')"
            assert cell.id == "cell-1"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_markdown_to_code(self, markdown_cell):
        path = _make_notebook(markdown_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-2", cell_type="code")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "code"
            assert cell.source == "# Title"
            assert cell.id == "cell-2"
            assert cell.outputs == []
            assert cell.execution_count is None
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_markdown_to_raw(self, markdown_cell):
        path = _make_notebook(markdown_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-2", cell_type="raw")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "raw"
            assert cell.source == "# Title"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_raw_to_code(self, raw_cell):
        path = _make_notebook(raw_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-3", cell_type="code")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "code"
            assert cell.source == "raw text"
            assert cell.outputs == []
            assert cell.execution_count is None
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_raw_to_markdown(self, raw_cell):
        path = _make_notebook(raw_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-3", cell_type="markdown")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "markdown"
            assert cell.source == "raw text"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_type_change_with_content(self, code_cell):
        path = _make_notebook(code_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-1", content="# New heading", cell_type="markdown")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "markdown"
            assert cell.source == "# New heading"
            assert cell.id == "cell-1"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_same_type_edits_content_only(self, code_cell):
        path = _make_notebook(code_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-1", content="new code", cell_type="code")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "code"
            assert cell.source == "new code"
            assert cell.outputs == []
            assert cell.execution_count is None
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_content_only_backward_compat(self, code_cell):
        """cell_type=None only changes content."""
        path = _make_notebook(code_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-1", content="updated")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "code"
            assert cell.source == "updated"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_type_only_preserves_source(self, code_cell):
        path = _make_notebook(code_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-1", cell_type="markdown")

            cell = _read_notebook(path).cells[0]
            assert cell.source == "print('hello')"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_preserves_metadata(self, code_cell):
        code_cell.metadata["custom_key"] = "custom_value"
        path = _make_notebook(code_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-1", cell_type="markdown")

            cell = _read_notebook(path).cells[0]
            assert cell.metadata.get("custom_key") == "custom_value"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_code_with_outputs_to_markdown_drops_outputs(self):
        """Outputs and execution_count should be dropped when converting code→markdown."""
        cell = nbformat.v4.new_code_cell(source="1+1")
        cell.id = "cell-out"
        cell.execution_count = 5
        cell.outputs = [nbformat.v4.new_output("execute_result", data={"text/plain": "2"})]
        path = _make_notebook(cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-out", cell_type="markdown")

            result = _read_notebook(path).cells[0]
            assert result.cell_type == "markdown"
            assert result.source == "1+1"
            assert "outputs" not in result
            assert "execution_count" not in result
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_cell_index_as_string(self, code_cell):
        """cell_id can be a numeric index string like '0'."""
        path = _make_notebook(code_cell)
        try:
            with _nbformat_path():
                await edit_cell(path, "0", cell_type="markdown")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "markdown"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_cell_not_found(self, code_cell):
        path = _make_notebook(code_cell)
        try:
            with _nbformat_path():
                with pytest.raises(ValueError, match="not found"):
                    await edit_cell(path, "nonexistent-id", content="x")
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_no_content_no_type_skips_write(self, code_cell):
        """Both content=None and cell_type=None should not write the file."""
        path = _make_notebook(code_cell)
        mtime_before = os.path.getmtime(path)
        try:
            with _nbformat_path():
                await edit_cell(path, "cell-1")

            cell = _read_notebook(path).cells[0]
            assert cell.cell_type == "code"
            assert cell.source == "print('hello')"
            # File should not have been rewritten
            assert os.path.getmtime(path) == mtime_before
        finally:
            os.unlink(path)


# ── YDoc path tests ──


class TestEditCellTypeYdoc:
    """Tests for cell type changes via the YDoc (collaborative) path."""

    @pytest.mark.asyncio
    async def test_code_to_markdown_calls_set_cell(self):
        ydoc, _ = _make_mock_ydoc(cell_type="code")

        with _ydoc_path(ydoc):
            await edit_cell("test.ipynb", "cell-1", cell_type="markdown")

        ydoc.set_cell.assert_called_once()
        new_cell = ydoc.set_cell.call_args[0][1]
        assert new_cell["cell_type"] == "markdown"
        assert new_cell["source"] == "print('hello')"
        assert new_cell["id"] == "cell-1"
        assert "outputs" not in new_cell
        assert "execution_count" not in new_cell

    @pytest.mark.asyncio
    async def test_markdown_to_code_adds_fields(self):
        ydoc, _ = _make_mock_ydoc(cell_type="markdown", source="# Title", cell_id="cell-2")

        with _ydoc_path(ydoc, resolved_cell_id="cell-2"):
            await edit_cell("test.ipynb", "cell-2", cell_type="code")

        new_cell = ydoc.set_cell.call_args[0][1]
        assert new_cell["cell_type"] == "code"
        assert new_cell["source"] == "# Title"
        assert new_cell["outputs"] == []
        assert new_cell["execution_count"] is None

    @pytest.mark.asyncio
    async def test_raw_to_markdown(self):
        ydoc, _ = _make_mock_ydoc(cell_type="raw", source="raw text", cell_id="cell-3")

        with _ydoc_path(ydoc, resolved_cell_id="cell-3"):
            await edit_cell("test.ipynb", "cell-3", cell_type="markdown")

        new_cell = ydoc.set_cell.call_args[0][1]
        assert new_cell["cell_type"] == "markdown"
        assert new_cell["source"] == "raw text"

    @pytest.mark.asyncio
    async def test_raw_to_code(self):
        ydoc, _ = _make_mock_ydoc(cell_type="raw", source="raw text", cell_id="cell-3")

        with _ydoc_path(ydoc, resolved_cell_id="cell-3"):
            await edit_cell("test.ipynb", "cell-3", cell_type="code")

        new_cell = ydoc.set_cell.call_args[0][1]
        assert new_cell["cell_type"] == "code"
        assert new_cell["outputs"] == []
        assert new_cell["execution_count"] is None

    @pytest.mark.asyncio
    async def test_markdown_to_raw(self):
        ydoc, _ = _make_mock_ydoc(cell_type="markdown", source="# Title", cell_id="cell-2")

        with _ydoc_path(ydoc, resolved_cell_id="cell-2"):
            await edit_cell("test.ipynb", "cell-2", cell_type="raw")

        new_cell = ydoc.set_cell.call_args[0][1]
        assert new_cell["cell_type"] == "raw"
        assert "outputs" not in new_cell

    @pytest.mark.asyncio
    async def test_type_change_with_content(self):
        ydoc, _ = _make_mock_ydoc(cell_type="code")

        with _ydoc_path(ydoc):
            await edit_cell("test.ipynb", "cell-1", content="# New heading", cell_type="markdown")

        new_cell = ydoc.set_cell.call_args[0][1]
        assert new_cell["source"] == "# New heading"
        assert new_cell["cell_type"] == "markdown"

    @pytest.mark.asyncio
    async def test_content_only_uses_atomic_replace(self):
        ydoc, ycell = _make_mock_ydoc()

        with _ydoc_path(ydoc), \
             patch("jupyter_ai_tools.toolkits.notebook._atomic_replace_cell_source") as mock_replace:
            await edit_cell("test.ipynb", "cell-1", content="new code")

        mock_replace.assert_called_once_with(ycell, "new code")
        ydoc.set_cell.assert_not_called()

    @pytest.mark.asyncio
    async def test_same_type_skips_set_cell(self):
        ydoc, ycell = _make_mock_ydoc(cell_type="code")

        with _ydoc_path(ydoc), \
             patch("jupyter_ai_tools.toolkits.notebook._atomic_replace_cell_source") as mock_replace:
            await edit_cell("test.ipynb", "cell-1", content="updated", cell_type="code")

        mock_replace.assert_called_once_with(ycell, "updated")
        ydoc.set_cell.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_content_no_type_is_noop(self):
        """Neither set_cell nor _atomic_replace should be called."""
        ydoc, _ = _make_mock_ydoc()

        with _ydoc_path(ydoc), \
             patch("jupyter_ai_tools.toolkits.notebook._atomic_replace_cell_source") as mock_replace:
            await edit_cell("test.ipynb", "cell-1")

        ydoc.set_cell.assert_not_called()
        mock_replace.assert_not_called()

    @pytest.mark.asyncio
    async def test_preserves_metadata(self):
        ydoc, _ = _make_mock_ydoc(metadata={"custom": "value"})

        with _ydoc_path(ydoc):
            await edit_cell("test.ipynb", "cell-1", cell_type="markdown")

        new_cell = ydoc.set_cell.call_args[0][1]
        assert new_cell["metadata"] == {"custom": "value"}

    @pytest.mark.asyncio
    async def test_type_change_with_animate(self):
        """animate=True + type change should call write_to_cell_collaboratively on the new ycell."""
        ydoc, _ = _make_mock_ydoc(cell_type="code")
        new_ycell = MagicMock()
        ydoc._ycells = [new_ycell]  # After set_cell, the ycell at index 0 is the new one

        with _ydoc_path(ydoc), \
             patch("jupyter_ai_tools.toolkits.notebook._atomic_replace_cell_source") as mock_replace, \
             patch("jupyter_ai_tools.toolkits.notebook.write_to_cell_collaboratively", new_callable=AsyncMock) as mock_write:
            await edit_cell("test.ipynb", "cell-1", content="# Animated", cell_type="markdown", animate=True)

        ydoc.set_cell.assert_called_once()
        # Should clear the new cell then write collaboratively
        mock_replace.assert_called_once_with(new_ycell, "")
        mock_write.assert_called_once_with(ydoc, new_ycell, "# Animated")

    @pytest.mark.asyncio
    async def test_cell_not_found_raises(self):
        ydoc, _ = _make_mock_ydoc()

        with _ydoc_path(ydoc, resolved_cell_id="bad-id"), \
             patch("jupyter_ai_tools.toolkits.notebook._get_cell_index_from_id_ydoc", return_value=None):
            with pytest.raises(ValueError, match="not found"):
                await edit_cell("test.ipynb", "bad-id", content="x")
