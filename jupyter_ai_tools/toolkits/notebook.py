import json
from typing import Any, Dict, Literal, Tuple

import nbformat
from jupyter_ai.tools.models import Tool, Toolkit

from ..utils import cell_to_md, get_file_id, get_jupyter_ydoc, notebook_json_to_md


def read_notebook(file_path: str, include_outputs=False) -> str:
    """Returns the complete notebook content as markdown string"""
    notebook_dict = read_notebook_json(file_path)
    notebook_md = notebook_json_to_md(notebook_dict, include_outputs=include_outputs)
    return notebook_md


def read_notebook_json(file_path: str) -> Dict[str, Any]:
    """Returns the complete notebook content and returns as json dict"""
    with open(file_path, "r:UTF-8") as f:
        notebook_dict = json.load(f)
        return notebook_dict


def read_cell(file_path: str, cell_id: str, include_outputs: bool = True) -> str:
    """Returns the notebook cell as markdown string"""
    cell, cell_index = read_cell_json(file_path, cell_id)
    cell_md = cell_to_md(cell, cell_index)
    return cell_md


def read_cell_json(file_path: str, cell_id: str) -> Tuple[Dict[str, Any], int]:
    """Returns the notebook cell as json dict and cell index"""
    notebook_json = read_notebook_json(file_path)
    cell_index = _get_cell_index_from_id_json(notebook_json, cell_id)
    if cell_index and 0 <= cell_index < len(notebook_json["cells"]):
        return notebook_json["cells"][cell_index]
    raise LookupError(f"No cell found with {cell_id=}")


def add_cell(
    file_path: str,
    content: str | None = None,
    cell_id: str | None = None,
    add_above: bool = False,
    cell_type: Literal["code", "markdown", "raw"] = "code",
):
    """Adds a new cell to the Jupyter notebook above or below a specified cell.

    This function adds a new cell to a Jupyter notebook. It first attempts to use
    the in-memory YDoc representation if the notebook is currently active. If the
    notebook is not active, it falls back to using the filesystem to read, modify,
    and write the notebook file directly.

    Args:
        file_path: The absolute path to the notebook file on the filesystem.
        content: The content of the new cell. If None, an empty cell is created.
        cell_id: The UUID of the cell to add relative to. If None,
                the cell is added at the end of the notebook.
        add_above: If True, the cell is added above the specified cell. If False,
                  it's added below the specified cell.
        cell_type: The type of cell to add ("code", "markdown", "raw").

    Returns:
        None
    """

    file_id = get_file_id(file_path)
    ydoc = get_jupyter_ydoc(file_id)

    if ydoc:
        cells_count = ydoc.cell_number()
        cell_index = _get_cell_index_from_id_ydoc(ydoc, cell_id) if cell_id else None
        insert_index = _determine_insert_index(cells_count, cell_index, add_above)
        ycell = ydoc.create_ycell(
            {
                "cell_type": cell_type,
                "source": content or "",
            }
        )
        ydoc.cells.insert(insert_index, ycell)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        cells_count = len(notebook.cells)
        cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id) if cell_id else None
        insert_index = _determine_insert_index(cells_count, cell_index, add_above)

        if cell_type == "code":
            notebook.cells.insert(insert_index, nbformat.v4.new_code_cell(source=content or ""))
        elif cell_type == "markdown":
            notebook.cells.insert(insert_index, nbformat.v4.new_markdown_cell(source=content or ""))
        else:
            notebook.cells.insert(insert_index, nbformat.v4.new_raw_cell(source=content or ""))

        with open(file_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)


def delete_cell(file_path: str, cell_id: str):
    """Removes a notebook cell with the specified cell ID.

    This function deletes a cell from a Jupyter notebook. It first attempts to use
    the in-memory YDoc representation if the notebook is currently active. If the
    notebook is not active, it falls back to using the filesystem to read, modify,
    and write the notebook file directly using nbformat.

    Args:
        file_path: The absolute path to the notebook file on the filesystem.
        cell_id: The UUID of the cell to delete.

    Returns:
        None
    """

    file_id = get_file_id(file_path)
    ydoc = get_jupyter_ydoc(file_id)
    if ydoc:
        cell_index = _get_cell_index_from_id_ydoc(ydoc, cell_id)
        if cell_index is not None and 0 <= cell_index < len(ydoc.cells):
            del ydoc.cells[cell_index]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id)
        if cell_index is not None and 0 <= cell_index < len(notebook.cells):
            notebook.cells.pop(cell_index)

            with open(file_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)


def edit_cell(file_path: str, cell_id: str, content: str | None = None) -> None:
    """Edits the content of a notebook cell with the specified ID

    This function modifies the content of a cell in a Jupyter notebook. It first attempts to use
    the in-memory YDoc representation if the notebook is currently active. If the
    notebook is not active, it falls back to using the filesystem to read, modify,
    and write the notebook file directly using nbformat.

    Args:
        file_path: The absolute path to the notebook file on the filesystem.
        cell_id: The UUID of the cell to edit.
        content: The new content for the cell. If None, the cell content remains unchanged.

    Returns:
        None

    Raises:
        ValueError: If the cell_id is not found in the notebook.
    """

    file_id = get_file_id(file_path)
    ydoc = get_jupyter_ydoc(file_id)

    if ydoc:
        cell_index = _get_cell_index_from_id_ydoc(ydoc, cell_id)
        if cell_index is not None:
            if content is not None:
                ydoc.cells[cell_index]["source"] = content
        else:
            raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id)
        if cell_index is not None:
            if content is not None:
                notebook.cells[cell_index].source = content

            with open(file_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)
        else:
            raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")


# Note: This is currently failing with server outputs, use `read_cell` instead
def read_cell_nbformat(file_path: str, cell_id: str) -> Dict[str, Any]:
    """Returns the content and metadata of a cell with the specified ID"""

    with open(file_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id)
    if cell_index is not None:
        cell = notebook.cells[cell_index]
        return cell
    else:
        raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")


def summarize_notebook(file_id: str, max_length: int = 500) -> str:
    """Generates a summary of the notebook content"""
    pass


def _get_cell_index_from_id_json(notebook_json, cell_id: str) -> int | None:
    """Get cell index from cell_id by notebook json dict."""
    for i, cell in enumerate(notebook_json["cells"]):
        if "id" in cell and cell["id"] == cell_id:
            return i
    return None


def _get_cell_index_from_id_ydoc(ydoc, cell_id: str) -> int | None:
    """Get cell index from cell_id using YDoc interface."""
    try:
        cell_index, _ = ydoc.find_cell(cell_id)
        return cell_index
    except (AttributeError, KeyError):
        return None


def _get_cell_index_from_id_nbformat(notebook, cell_id: str) -> int | None:
    """Get cell index from cell_id using nbformat interface."""
    for i, cell in enumerate(notebook.cells):
        if hasattr(cell, "id") and cell.id == cell_id:
            return i
        elif hasattr(cell, "metadata") and cell.metadata.get("id") == cell_id:
            return i
    return None


def _determine_insert_index(cells_count: int, cell_index: str, add_above: bool) -> int:
    if cell_index is None:
        insert_index = cells_count
    else:
        if not (0 <= cell_index < cells_count):
            cell_index = max(0, min(cell_index, cells_count))
        insert_index = cell_index if add_above else cell_index + 1
    return insert_index


toolkit = Toolkit(
    name="notebook_toolkit",
    description="Tools for reading and manipulating Jupyter notebooks.",
)
toolkit.add_tool(Tool(callable=read_notebook, read=True))
toolkit.add_tool(Tool(callable=read_cell, read=True))
toolkit.add_tool(Tool(callable=add_cell, write=True))
toolkit.add_tool(Tool(callable=delete_cell, delete=True))
toolkit.add_tool(Tool(callable=edit_cell, read=True, write=True))
