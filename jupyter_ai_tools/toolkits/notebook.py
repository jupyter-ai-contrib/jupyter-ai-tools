import json
from typing import Any, Dict, Literal, Optional, Tuple

import nbformat
from jupyter_ai.tools.models import Tool, Toolkit

from ..utils import cell_to_md, get_file_id, get_jupyter_ydoc, notebook_json_to_md


async def read_notebook(file_path: str, include_outputs=False) -> str:
    """Returns the complete notebook content as markdown string.

    This function reads a Jupyter notebook file and converts its content to a markdown string.
    It uses the read_notebook_json function to read the notebook file and then converts
    the resulting JSON to markdown.

    Args:
        file_path:
            The absolute path to the notebook file on the filesystem.
        include_outputs:
            If True, cell outputs will be included in the markdown. Default is False.

    Returns:
        The notebook content as a markdown string.
    """
    notebook_dict = await read_notebook_json(file_path)
    notebook_md = notebook_json_to_md(notebook_dict, include_outputs=include_outputs)
    return notebook_md


async def read_notebook_json(file_path: str) -> Dict[str, Any]:
    """Returns the complete notebook content as a JSON dictionary.

    This function reads a Jupyter notebook file and returns its content as a
    dictionary representation of the JSON structure.

    Args:
        file_path:
            The absolute path to the notebook file on the filesystem.

    Returns:
        A dictionary containing the complete notebook structure.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        notebook_dict = json.load(f)
        return notebook_dict


async def read_cell(file_path: str, cell_id: str, include_outputs: bool = True) -> str:
    """Returns the notebook cell as a markdown string.

    This function reads a specific cell from a Jupyter notebook file and converts
    it to a markdown string. It uses the read_cell_json function to read the cell
    and then converts it to markdown.

    Args:
        file_path:
            The absolute path to the notebook file on the filesystem.
        cell_id:
            The UUID of the cell to read.
        include_outputs:
            If True, cell outputs will be included in the markdown. Default is True.

    Returns:
        The cell content as a markdown string.

    Raises:
        LookupError: If no cell with the given ID is found.
    """
    cell, cell_index = await read_cell_json(file_path, cell_id)
    cell_md = cell_to_md(cell, cell_index)
    return cell_md


async def read_cell_json(file_path: str, cell_id: str) -> Tuple[Dict[str, Any], int]:
    """Returns the notebook cell as a JSON dictionary and its index.

    This function reads a specific cell from a Jupyter notebook file and returns
    both the cell content as a dictionary and the cell's index within the notebook.

    Args:
        file_path:
            The absolute path to the notebook file on the filesystem.
        cell_id:
            The UUID of the cell to read.

    Returns:
        A tuple containing:
        - The cell as a dictionary
        - The index of the cell in the notebook

    Raises:
        LookupError: If no cell with the given ID is found.
    """
    notebook_json = await read_notebook_json(file_path)
    cell_index = _get_cell_index_from_id_json(notebook_json, cell_id)
    if cell_index and 0 <= cell_index < len(notebook_json["cells"]):
        return notebook_json["cells"][cell_index]
    raise LookupError(f"No cell found with {cell_id=}")


async def get_cell_id_from_index(file_path: str, cell_index: int) -> Optional[int]:
    """Finds the cell_id of the cell at a specific cell index.

    This function reads a Jupyter notebook file and returns the UUID of the cell
    at the specified index position.

    Args:
        file_path:
            The absolute path to the notebook file on the filesystem.
        cell_index:
            The index of the cell to find the ID for.

    Returns:
        The UUID of the cell at the specified index, or None if the index is out of range
        or if the cell does not have an ID.
    """

    cell_id = None
    notebook_json = await read_notebook_json(file_path)
    cells = notebook_json["cells"]
    if 0 <= cell_index < len(cells):
        cell_id = cells[cell_index].get("cell_id")

    if cell_id is None:
        raise ValueError("No cell_id found, use `insert_cell` based on cell index")

    return cell_id


async def add_cell(
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
        file_path:
            The absolute path to the notebook file on the filesystem.
        content:
            The content of the new cell. If None, an empty cell is created.
        cell_id:
            The UUID of the cell to add relative to. If None,
            the cell is added at the end of the notebook.
        add_above:
            If True, the cell is added above the specified cell. If False,
            it's added below the specified cell.
        cell_type:
            The type of cell to add ("code", "markdown", "raw").

    Returns:
        None
    """

    file_id = await get_file_id(file_path)
    ydoc = await get_jupyter_ydoc(file_id)

    if ydoc:
        cells_count = ydoc.cell_number
        cell_index = _get_cell_index_from_id_ydoc(ydoc, cell_id) if cell_id else None
        insert_index = _determine_insert_index(cells_count, cell_index, add_above)
        cell = {
            "cell_type": cell_type,
            "source": content or "",
        }
        if insert_index >= cells_count:
            ydoc.append_cell(cell)
        else:
            ydoc.ycells.insert(insert_index, ydoc.create_cell(cell))
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


async def insert_cell(
    file_path: str,
    content: str | None = None,
    insert_index: int | None = None,
    cell_type: Literal["code", "markdown", "raw"] = "code",
):
    """Inserts a new cell to the Jupyter notebook at the specified cell index.

    This function adds a new cell to a Jupyter notebook. It first attempts to use
    the in-memory YDoc representation if the notebook is currently active. If the
    notebook is not active, it falls back to using the filesystem to read, modify,
    and write the notebook file directly.

    Args:
        file_path:
            The absolute path to the notebook file on the filesystem.
        content:
            The content of the new cell. If None, an empty cell is created.
        insert_index:
            The index to insert the cell at.
        cell_type:
            The type of cell to add ("code", "markdown", "raw").

    Returns:
        None
    """

    file_id = await get_file_id(file_path)
    ydoc = await get_jupyter_ydoc(file_id)

    if ydoc:
        cells_count = ydoc.cell_number
        cell = {
            "cell_type": cell_type,
            "source": content or "",
        }
        if insert_index >= cells_count:
            ydoc.append_cell(cell)
        else:
            ydoc.ycells.insert(insert_index, ydoc.create_cell(cell))
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        cells_count = len(notebook.cells)

        if cell_type == "code":
            notebook.cells.insert(insert_index, nbformat.v4.new_code_cell(source=content or ""))
        elif cell_type == "markdown":
            notebook.cells.insert(insert_index, nbformat.v4.new_markdown_cell(source=content or ""))
        else:
            notebook.cells.insert(insert_index, nbformat.v4.new_raw_cell(source=content or ""))

        with open(file_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)


async def delete_cell(file_path: str, cell_id: str):
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

    file_id = await get_file_id(file_path)
    ydoc = await get_jupyter_ydoc(file_id)
    if ydoc:
        cell_index = _get_cell_index_from_id_ydoc(ydoc, cell_id)
        if cell_index is not None and 0 <= cell_index < len(ydoc.ycells):
            del ydoc.ycells[cell_index]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id)
        if cell_index is not None and 0 <= cell_index < len(notebook.cells):
            notebook.cells.pop(cell_index)

            with open(file_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)

    if not cell_index:
        raise ValueError(f"Could not find cell index for {cell_id=}")


async def edit_cell(file_path: str, cell_id: str, content: str) -> None:
    """Edits the content of a notebook cell with the specified ID

    This function modifies the content of a cell in a Jupyter notebook. It first attempts to use
    the in-memory YDoc representation if the notebook is currently active. If the
    notebook is not active, it falls back to using the filesystem to read, modify,
    and write the notebook file directly using nbformat.

    Args:
        file_path:
            The absolute path to the notebook file on the filesystem.
        cell_id:
            The UUID of the cell to edit.
        content:
            The new content for the cell. If None, the cell content remains unchanged.

    Returns:
        None

    Raises:
        ValueError: If the cell_id is not found in the notebook.
    """

    file_id = await get_file_id(file_path)
    ydoc = await get_jupyter_ydoc(file_id)

    if ydoc:
        cell_index = _get_cell_index_from_id_ydoc(ydoc, cell_id)
        if cell_index is not None:
            ydoc.ycells[cell_index]["source"] = content
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id)
        if cell_index is not None:
            notebook.cells[cell_index].source = content

            with open(file_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)

    raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")


# Note: This is currently failing with server outputs, use `read_cell` instead
def read_cell_nbformat(file_path: str, cell_id: str) -> Dict[str, Any]:
    """Returns the content and metadata of a cell with the specified ID.

    This function reads a specific cell from a Jupyter notebook file using the nbformat
    library and returns the cell's content and metadata.

    Note: This function is currently not functioning properly with server outputs.
    Use `read_cell` instead.

    Args:
        file_path:
            The absolute path to the notebook file on the filesystem.
        cell_id:
            The UUID of the cell to read.

    Returns:
        The cell as a dictionary containing its content and metadata.

    Raises:
        ValueError: If no cell with the given ID is found.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id)
    if cell_index is not None:
        cell = notebook.cells[cell_index]
        return cell
    else:
        raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")


def _get_cell_index_from_id_json(notebook_json, cell_id: str) -> int | None:
    """Get cell index from cell_id by notebook json dict.

    Args:
        notebook_json:
            The notebook as a JSON dictionary.
        cell_id:
            The UUID of the cell to find.

    Returns:
        The index of the cell in the notebook, or None if not found.
    """
    for i, cell in enumerate(notebook_json["cells"]):
        if "id" in cell and cell["id"] == cell_id:
            return i
    return None


def _get_cell_index_from_id_ydoc(ydoc, cell_id: str) -> int | None:
    """Get cell index from cell_id using YDoc interface.

    Args:
        ydoc:
            The YDoc object representing the notebook.
        cell_id:
            The UUID of the cell to find.

    Returns:
        The index of the cell in the notebook, or None if not found.
    """
    try:
        cell_index, _ = ydoc.find_cell(cell_id)
        return cell_index
    except (AttributeError, KeyError):
        return None


def _get_cell_index_from_id_nbformat(notebook, cell_id: str) -> int | None:
    """Get cell index from cell_id using nbformat interface.

    Args:
        notebook:
            The nbformat notebook object.
        cell_id:
            The UUID of the cell to find.

    Returns:
        The index of the cell in the notebook, or None if not found.
    """
    for i, cell in enumerate(notebook.cells):
        if hasattr(cell, "id") and cell.id == cell_id:
            return i
        elif hasattr(cell, "metadata") and cell.metadata.get("id") == cell_id:
            return i
    return None


def _determine_insert_index(cells_count: int, cell_index: Optional[int], add_above: bool) -> int:
    """Determine the index where a new cell should be inserted.

    Args:
        cells_count:
            The total number of cells in the notebook.
        cell_index:
            The index of the reference cell, or None to append at the end.
        add_above:
            If True, insert above the reference cell; if False, insert below.

    Returns:
        The index where the new cell should be inserted.
    """
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
toolkit.add_tool(Tool(callable=add_cell, read=True, write=True))
toolkit.add_tool(Tool(callable=insert_cell, read=True, write=True))
toolkit.add_tool(Tool(callable=delete_cell, delete=True))
toolkit.add_tool(Tool(callable=edit_cell, read=True, write=True))
toolkit.add_tool(Tool(callable=get_cell_id_from_index, read=True))
