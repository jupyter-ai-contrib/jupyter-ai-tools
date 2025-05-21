import asyncio
import difflib
import json
from typing import Any, Dict

from jupyter_ydoc import YNotebook


# Delete a cell
async def delete_cell(ynotebook: YNotebook, index: int) -> str:
    """
    Delete the cell at the specified index and return its source.

    Parameters:
        ynotebook (YNotebook): The notebook to modify.
        index (int): The index of the cell to delete.

    Returns:
        str: The source of the deleted cell, or an error message.
    """
    try:
        cell = ynotebook.get_cell(index)
        ynotebook._ycells.pop(index)
        return f"✅ Cut cell {index} :\n{cell['source']}"
    except Exception as e:
        return f"❌ Error cutting cell {index}: {str(e)}"


# Overwrite cell contents
async def write_to_cell(ynotebook: YNotebook, index: int, content: str, stream: bool = True) -> str:
    """
    Overwrite the source of a notebook cell at the given index.

    Parameters:
        ynotebook (YNotebook): The notebook to modify.
        index (int): The index of the cell to overwrite.
        content (str): The new content to write.
        stream (bool): Whether to simulate gradual updates (default: True).

    Returns:
        str: Success or error message.
    """
    try:
        ycell = ynotebook.get_cell(index)
        old = ycell["source"]
        new = content

        if not stream:
            ycell["source"] = new
            ynotebook.set_cell(index, ycell)
            return f"✅ Overwrote cell {index}."

        sm = difflib.SequenceMatcher(None, old, new)
        result = list(old)
        cursor = 0

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                cursor += i2 - i1
            elif tag == "delete":
                for offset in reversed(range(i2 - i1)):
                    del result[cursor + offset]
                    ycell["source"] = ''.join(result)
                    ynotebook.set_cell(index, ycell)
                    await asyncio.sleep(0.03)
            elif tag == "insert":
                for c in new[j1:j2]:
                    result.insert(cursor, c)
                    cursor += 1
                    ycell["source"] = ''.join(result)
                    ynotebook.set_cell(index, ycell)
                    await asyncio.sleep(0.03)
            elif tag == "replace":
                for _ in range(i2 - i1):
                    result.pop(cursor)
                    ycell["source"] = ''.join(result)
                    ynotebook.set_cell(index, ycell)
                    await asyncio.sleep(0.03)
                for c in new[j1:j2]:
                    result.insert(cursor, c)
                    cursor += 1
                    ycell["source"] = ''.join(result)
                    ynotebook.set_cell(index, ycell)
                    await asyncio.sleep(0.03)

        return f"✅ Updated cell {index}."
    except Exception as e:
        return f"❌ Error editing cell {index}: {str(e)}"


# Add a new cell
async def add_cell(ynotebook: YNotebook, index: int, cell_type: str = "code") -> str:
    """
    Insert a new blank cell at the specified index.

    Parameters:
        ynotebook (YNotebook): The notebook to modify.
        index (int): The index at which to insert the cell.
        cell_type (str): The type of cell to insert (default: "code").

    Returns:
        str: Success or error message.
    """
    try:
        new_cell: Dict[str, Any] = {
            "cell_type": cell_type,
            "source": "",
            "metadata": {},
        }
        if cell_type == "code":
            new_cell["outputs"] = []
            new_cell["execution_count"] = None

        ycell = ynotebook.create_ycell(new_cell)
        ynotebook._ycells.insert(index, ycell)

        return f"✅ Added {cell_type} cell at index {index}."
    except Exception as e:
        return f"❌ Error adding cell at index {index}: {str(e)}"


# Get the index of the last cell
async def get_max_cell_index(ynotebook: YNotebook) -> int:
    """
    Return the index of the last cell in the notebook.

    Parameters:
        ynotebook (YNotebook): The notebook to query.

    Returns:
        int: The highest valid cell index.
    """
    try:
        return len(ynotebook._ycells) - 1
    except Exception as e:
        raise RuntimeError(f"❌ Error getting max cell index: {str(e)}")


# Read a specific cell
async def read_cell(ynotebook: YNotebook, index: int) -> str:
    """
    Return the full content of a specific notebook cell.

    Parameters:
        ynotebook (YNotebook): The notebook to read from.
        index (int): The index of the cell to read.

    Returns:
        str: JSON-formatted cell data or error message.
    """
    try:
        if 0 <= index < len(ynotebook._ycells):
            cell_data = ynotebook.get_cell(index)
            return json.dumps(cell_data, indent=2)
        else:
            return f"❌ Invalid cell index: {index}"
    except Exception as e:
        return f"❌ Error reading cell {index}: {str(e)}"


# Read the entire notebook
async def read_notebook(ynotebook: YNotebook) -> str:
    """
    Return the full notebook content as a JSON-formatted list of cells.

    Parameters:
        ynotebook (YNotebook): The notebook to read.

    Returns:
        str: JSON-formatted list of cells or an error message.
    """
    try:
        cells = [ynotebook.get_cell(i) for i in range(len(ynotebook._ycells))]
        return json.dumps(cells, indent=2)
    except Exception as e:
        return f"❌ Error reading notebook: {str(e)}"
