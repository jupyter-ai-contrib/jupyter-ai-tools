import asyncio
import difflib
from functools import lru_cache
import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import nbformat
from jupyter_ydoc import YNotebook
from pycrdt import Assoc, Text

from ..utils import (
    cell_to_md,
    get_file_id,
    get_global_awareness,
    get_jupyter_ydoc,
    normalize_filepath,
    notebook_json_to_md,
)


def _is_uuid_like(value: str) -> bool:
    """Check if a string looks like a UUID v4"""
    if not isinstance(value, str):
        return False
    # UUID v4 pattern: 8-4-4-4-12 hexadecimal characters
    uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    return bool(re.match(uuid_pattern, value, re.IGNORECASE))


def _is_index_like(value: str) -> bool:
    """Check if a string looks like a numeric index"""
    if not isinstance(value, str):
        return False
    try:
        int(value)
        return True
    except ValueError:
        return False


async def _resolve_cell_id(file_path: str, cell_id_or_index: str) -> str:
    """
    Resolve a cell_id parameter that might be either a UUID or an index.
    If it's an index, convert it to the actual cell_id.
    """
    if _is_uuid_like(cell_id_or_index):
        return cell_id_or_index
    elif _is_index_like(cell_id_or_index):
        index = int(cell_id_or_index)
        try:
            actual_cell_id = await get_cell_id_from_index(file_path, index)
            return actual_cell_id
        except Exception as e:
            raise ValueError(f"Invalid cell index {index}: {str(e)}")
    else:
        # Assume it's a cell_id and let the downstream function handle validation
        return cell_id_or_index


def clean_text(text: Union[str, list, None]) -> Optional[str]:
    """Clean and format text output.

    Args:
        text: Text data that might be string, list, or None

    Returns:
        Cleaned text string or None
    """
    if text is None:
        return None
    if isinstance(text, list):
        return "".join(str(item) for item in text)
    return str(text)


def process_notebook_output(output_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a Jupyter notebook cell output into a standardized format.

    Args:
        output_data: Raw output data from notebook cell

    Returns:
        Processed output dictionary with standardized format
    """
    output_type = output_data.get("output_type")

    if output_type == "stream":
        return {"output_type": output_type, "text": clean_text(output_data.get("text", ""))}

    elif output_type in ["execute_result", "display_data"]:
        data = output_data.get("data", {})
        return {
            "output_type": output_type,
            "text": clean_text(data.get("text/plain")),
            "image": extract_image_data(data) if data else None,
        }

    elif output_type == "error":
        error_text = f"{output_data.get('ename', '')}: {output_data.get('evalue', '')}\n{chr(10).join(output_data.get('traceback', []))}"
        return {"output_type": output_type, "text": clean_text(error_text)}

    return output_data


def extract_image_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract image data from notebook output data.

    Args:
        data: Output data dictionary that may contain various MIME types

    Returns:
        Extracted image data or None
    """
    for mime_type in ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/svg+xml"]:
        if mime_type in data:
            return {"mime_type": mime_type, "data": data[mime_type]}
    return None


def format_notebook_cell(cell_data: Dict[str, Any], cell_index: int, language: str, include_full_outputs: bool = False) -> Dict[str, Any]:
    """Format a Jupyter notebook cell into a standardized format.

    Args:
        cell_data: Raw cell data from notebook JSON
        cell_index: Index of the cell in the notebook
        language: Programming language of the notebook
        include_full_outputs: Whether to include full outputs or truncate large ones
    """
    cell_id = cell_data.get("id", f"cell-{cell_index}")

    formatted_cell = {
        "cellType": cell_data["cell_type"],
        "source": "".join(cell_data["source"]) if isinstance(cell_data["source"], list) else cell_data["source"],
        "execution_count": cell_data.get("execution_count") if cell_data["cell_type"] == "code" else None,
        "cell_id": cell_id,
    }

    if cell_data["cell_type"] == "code":
        formatted_cell["language"] = language

    if cell_data["cell_type"] == "code" and cell_data.get("outputs"):
        processed_outputs = [process_notebook_output(output) for output in cell_data["outputs"]]

        if not include_full_outputs and len(json.dumps(processed_outputs)) > 10000:
            formatted_cell["outputs"] = [{
                "output_type": "stream",
                "text": f"Outputs are too large to include. Use command with: cat <notebook_path> | jq '.cells[{cell_index}].outputs'",
            }]
        else:
            formatted_cell["outputs"] = processed_outputs

    return formatted_cell


async def read_notebook_cells(notebook_path: str, specific_cell_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read and process cells from a Jupyter notebook file.

    Args:
        notebook_path: Path to the notebook file
        specific_cell_id: Optional cell ID to return only that cell

    Returns:
        List of formatted cell dictionaries

    Raises:
        FileNotFoundError: If notebook file doesn't exist
        ValueError: If specific cell ID is not found
    """
    resolved_path = normalize_filepath(notebook_path)

    with open(resolved_path, "r", encoding="utf-8") as file:
        notebook_data = json.load(file)

    language = notebook_data.get("metadata", {}).get("language_info", {}).get("name", "python")

    if specific_cell_id:
        target_cell = None
        cell_index = -1
        for i, cell in enumerate(notebook_data["cells"]):
            if cell.get("id") == specific_cell_id:
                target_cell = cell
                cell_index = i
                break
        if target_cell is None:
            raise ValueError(f'Cell with ID "{specific_cell_id}" not found in notebook')
        return [format_notebook_cell(target_cell, cell_index, language, include_full_outputs=True)]

    return [
        format_notebook_cell(cell, index, language, include_full_outputs=False)
        for index, cell in enumerate(notebook_data["cells"])
    ]


async def read_notebook(file_path: str, include_outputs=False) -> str:
    """Returns the complete notebook content as markdown string.

    This function reads a Jupyter notebook file and converts its content to a markdown string.
    It uses the read_notebook_json function to read the notebook file and then converts
    the resulting JSON to markdown.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        include_outputs:
            If True, cell outputs will be included in the markdown. Default is False.

    Returns:
        The notebook content as a markdown string.
    """
    try:
        file_path = normalize_filepath(file_path)
        notebook_dict = await read_notebook_json(file_path)
        notebook_md = notebook_json_to_md(notebook_dict, include_outputs=include_outputs)
        return notebook_md
    except Exception:
        raise


async def read_notebook_json(file_path: str) -> Dict[str, Any]:
    """Returns the complete notebook content as a JSON dictionary.

    This function reads a Jupyter notebook file and returns its content as a
    dictionary representation of the JSON structure.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.

    Returns:
        A dictionary containing the complete notebook structure.
    """
    try:
        file_path = normalize_filepath(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            notebook_dict = json.load(f)
            return notebook_dict
    except Exception:
        raise


async def read_cell(file_path: str, cell_id: str, include_outputs: bool = True) -> str:
    """Returns the notebook cell as a markdown string.

    This function reads a specific cell from a Jupyter notebook file and converts
    it to a markdown string. It uses the read_cell_json function to read the cell
    and then converts it to markdown.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        cell_id:
            The UUID of the cell to read, or a numeric index as string.
        include_outputs:
            If True, cell outputs will be included in the markdown. Default is True.

    Returns:
        The cell content as a markdown string.

    Raises:
        LookupError: If no cell with the given ID is found.
    """
    try:
        file_path = normalize_filepath(file_path)
        # Resolve cell_id in case it's an index
        resolved_cell_id = await _resolve_cell_id(file_path, cell_id)
        cell, cell_index = await read_cell_json(file_path, resolved_cell_id)
        cell_md = cell_to_md(cell, cell_index)
        return cell_md
    except Exception:
        raise


async def read_cell_json(file_path: str, cell_id: str) -> Tuple[Dict[str, Any], int]:
    """Returns the notebook cell as a JSON dictionary and its index.

    This function reads a specific cell from a Jupyter notebook file and returns
    both the cell content as a dictionary and the cell's index within the notebook.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        cell_id:
            The UUID of the cell to read, or a numeric index as string.

    Returns:
        A tuple containing:
        - The cell as a dictionary
        - The index of the cell in the notebook

    Raises:
        LookupError: If no cell with the given ID is found.
    """
    try:
        file_path = normalize_filepath(file_path)
        # Resolve cell_id in case it's an index
        resolved_cell_id = await _resolve_cell_id(file_path, cell_id)
        notebook_json = await read_notebook_json(file_path)
        cell_index = _get_cell_index_from_id_json(notebook_json, resolved_cell_id)

        if cell_index is not None and 0 <= cell_index < len(notebook_json["cells"]):
            cell = notebook_json["cells"][cell_index]
            return cell, cell_index

        raise LookupError(f"No cell found with {cell_id=}")

    except Exception:
        raise


async def get_cell_id_from_index(file_path: str, cell_index: int) -> str:
    """Finds the cell_id of the cell at a specific cell index.

    This function reads a Jupyter notebook file and returns the UUID of the cell
    at the specified index position.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        cell_index:
            The index of the cell to find the ID for.

    Returns:
        The UUID of the cell at the specified index, or None if the index is out of range
        or if the cell does not have an ID.
    """
    try:
        file_path = normalize_filepath(file_path)
        cell_id = None
        notebook_json = await read_notebook_json(file_path)
        cells = notebook_json["cells"]

        if 0 <= cell_index < len(cells):
            cell_id = cells[cell_index].get("id")
        else:
            cell_id = None

        if cell_id is None:
            raise ValueError("No cell_id found, use `insert_cell` based on cell index")

        return cell_id

    except Exception:
        raise


async def add_cell(
    file_path: str,
    content: Optional[str] = None,
    cell_id: Optional[str] = None,
    add_above: bool = False,
    cell_type: Literal["code", "markdown", "raw"] = "code",
    animate: bool = False,
):
    """Adds a new cell to the Jupyter notebook above or below a specified cell.

    This function adds a new cell to a Jupyter notebook. It first attempts to use
    the in-memory YDoc representation if the notebook is currently active. If the
    notebook is not active, it falls back to using the filesystem to read, modify,
    and write the notebook file directly.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        content:
            The content of the new cell. If None, an empty cell is created.
        cell_id:
            The UUID of the cell to add relative to, or a numeric index as string. If None,
            the cell is added at the end of the notebook.
        add_above:
            If True, the cell is added above the specified cell. If False,
            it's added below the specified cell.
        cell_type:
            The type of cell to add ("code", "markdown", "raw").

    Returns:
        None
    """
    try:
        file_path = normalize_filepath(file_path)
        # Resolve cell_id in case it's an index
        resolved_cell_id = await _resolve_cell_id(file_path, cell_id) if cell_id else None

        file_id = await get_file_id(file_path)
        ydoc: YNotebook = await get_jupyter_ydoc(file_id)

        if ydoc:
            cells_count = ydoc.cell_number
            cell_index = (
                _get_cell_index_from_id_ydoc(ydoc, resolved_cell_id) if resolved_cell_id else None
            )
            insert_index = _determine_insert_index(cells_count, cell_index, add_above)

            cell = {
                "cell_type": cell_type,
                "source": "",
            }
            if cell_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            ycell = ydoc.create_ycell(cell)
            if insert_index >= cells_count:
                ydoc.ycells.append(ycell)
            else:
                ydoc.ycells.insert(insert_index, ycell)
            if animate:
                await write_to_cell_collaboratively(ydoc, ycell, content or "")
            else:
                _atomic_replace_cell_source(ycell, content or "")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

            cells_count = len(notebook.cells)
            cell_index = (
                _get_cell_index_from_id_nbformat(notebook, resolved_cell_id)
                if resolved_cell_id
                else None
            )
            insert_index = _determine_insert_index(cells_count, cell_index, add_above)

            if cell_type == "code":
                notebook.cells.insert(insert_index, nbformat.v4.new_code_cell(source=content or ""))
            elif cell_type == "markdown":
                notebook.cells.insert(
                    insert_index, nbformat.v4.new_markdown_cell(source=content or "")
                )
            else:
                notebook.cells.insert(insert_index, nbformat.v4.new_raw_cell(source=content or ""))

            with open(file_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)

        return None
    except Exception:
        raise


async def insert_cell(
    file_path: str,
    content: Optional[str] = None,
    insert_index: Optional[int] = None,
    cell_type: Literal["code", "markdown", "raw"] = "code",
    animate: bool = False,
):
    """Inserts a new cell to the Jupyter notebook at the specified cell index.

    This function adds a new cell to a Jupyter notebook. It first attempts to use
    the in-memory YDoc representation if the notebook is currently active. If the
    notebook is not active, it falls back to using the filesystem to read, modify,
    and write the notebook file directly.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        content:
            The content of the new cell. If None, an empty cell is created.
        insert_index:
            The index to insert the cell at.
        cell_type:
            The type of cell to add ("code", "markdown", "raw").

    Returns:
        None
    """
    try:
        file_path = normalize_filepath(file_path)
        file_id = await get_file_id(file_path)
        ydoc = await get_jupyter_ydoc(file_id)

        if ydoc:
            cells_count = ydoc.cell_number

            cell = {
                "cell_type": cell_type,
                "source": "",
            }
            if cell_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            ycell = ydoc.create_ycell(cell)
            if insert_index >= cells_count:
                ydoc.ycells.append(ycell)
            else:
                ydoc.ycells.insert(insert_index, ycell)
            if animate:
                await write_to_cell_collaboratively(ydoc, ycell, content or "")
            else:
                _atomic_replace_cell_source(ycell, content or "")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

            cells_count = len(notebook.cells)

            if cell_type == "code":
                notebook.cells.insert(insert_index, nbformat.v4.new_code_cell(source=content or ""))
            elif cell_type == "markdown":
                notebook.cells.insert(
                    insert_index, nbformat.v4.new_markdown_cell(source=content or "")
                )
            else:
                notebook.cells.insert(insert_index, nbformat.v4.new_raw_cell(source=content or ""))

            with open(file_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)

    except Exception:
        raise


async def delete_cell(file_path: str, cell_id: str):
    """Removes a notebook cell with the specified cell ID.

    This function deletes a cell from a Jupyter notebook. It first attempts to use
    the in-memory YDoc representation if the notebook is currently active. If the
    notebook is not active, it falls back to using the filesystem to read, modify,
    and write the notebook file directly using nbformat.

    Args:
        file_path: The relative path to the notebook file on the filesystem.
        cell_id: The UUID of the cell to delete, or a numeric index as string.

    Returns:
        None
    """
    try:
        file_path = normalize_filepath(file_path)
        # Resolve cell_id in case it's an index
        resolved_cell_id = await _resolve_cell_id(file_path, cell_id)

        file_id = await get_file_id(file_path)
        ydoc = await get_jupyter_ydoc(file_id)

        if ydoc:
            cell_index = _get_cell_index_from_id_ydoc(ydoc, resolved_cell_id)
            if cell_index is not None and 0 <= cell_index < len(ydoc.ycells):
                del ydoc.ycells[cell_index]
            else:
                pass  # Cell not found in ydoc
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

            cell_index = _get_cell_index_from_id_nbformat(notebook, resolved_cell_id)
            if cell_index is not None and 0 <= cell_index < len(notebook.cells):
                notebook.cells.pop(cell_index)
                with open(file_path, "w", encoding="utf-8") as f:
                    nbformat.write(notebook, f)
            else:
                pass  # Cell not found in notebook

        if cell_index is None:
            raise ValueError(f"Could not find cell index for {cell_id=}")

        return {"success": True}
    except Exception:
        raise


def get_cursor_details(
    cell_source: Text, start_index: int, stop_index: Optional[int] = None
) -> Dict[str, Any]:
    """
    Creates cursor details for collaborative notebook cursor positioning.

    This function constructs the cursor details object required by the YNotebook
    awareness system to show cursor positions in collaborative editing environments.
    It handles both single cursor positions and text selections.

    Args:
        cell_source: The YText source object representing the cell content
        start_index: The starting position of the cursor (0-based index)
        stop_index: The ending position for selections (optional)

    Returns:
        dict: Cursor details object with head, anchor, and selection state

    Example:
        >>> details = get_cursor_details(cell_source, 10)  # Single cursor at position 10
        >>> details = get_cursor_details(cell_source, 5, 15)  # Selection from 5 to 15
    """
    # Create sticky index for the head position (where cursor starts)
    head_sticky_index = cell_source.sticky_index(start_index, Assoc.BEFORE)
    head_sticky_index_data = head_sticky_index.to_json()

    # Initialize cursor details with default values
    cursor_details: Dict[str, Any] = {"primary": True, "empty": True}

    # Set the head position (where cursor starts)
    cursor_details["head"] = {
        "type": head_sticky_index_data["item"],
        "tname": None,
        "item": head_sticky_index_data["item"],
        "assoc": 0,
    }

    # By default, anchor is same as head (no selection)
    cursor_details["anchor"] = cursor_details["head"]

    # If stop_index is provided, create a selection
    if stop_index is not None:
        anchor_sticky_index = cell_source.sticky_index(stop_index, Assoc.BEFORE)
        anchor_sticky_index_data = anchor_sticky_index.to_json()
        cursor_details["anchor"] = {
            "type": anchor_sticky_index_data["item"],
            "tname": None,
            "item": anchor_sticky_index_data["item"],
            "assoc": 0,
        }
        cursor_details["empty"] = False  # Not empty when there's a selection

    return cursor_details


def set_cursor_in_ynotebook(
    ynotebook: YNotebook, cell_source: Text, start_index: int, stop_index: Optional[int] = None
) -> None:
    """
    Sets the cursor position in a collaborative notebook environment.

    This function updates the cursor position in the YNotebook awareness system,
    which allows other collaborators to see where the cursor is positioned.
    It handles both single cursor positions and text selections.

    Args:
        ynotebook: The YNotebook instance representing the collaborative notebook
        cell_source: The YText source object representing the cell content
        start_index: The starting position of the cursor (0-based index)
        stop_index: The ending position for selections (optional)

    Returns:
        None: This function does not return a value

    Note:
        This function silently ignores any errors that occur during cursor setting
        to avoid breaking the main collaborative editing operations.

    Example:
        >>> set_cursor_in_ynotebook(ynotebook, cell_source, 10)  # Set cursor at position 10
        >>> set_cursor_in_ynotebook(ynotebook, cell_source, 5, 15)  # Select text from 5 to 15
    """
    try:
        # Get cursor details for the specified position/selection
        details = get_cursor_details(cell_source, start_index, stop_index=stop_index)

        # Update the awareness system with the cursor position
        if ynotebook.awareness:
            ynotebook.awareness.set_local_state_field("cursors", [details])
    except Exception:
        # Silently ignore cursor setting errors to avoid breaking main operations
        # This is intentional - cursor positioning is a visual enhancement, not critical
        pass


def _atomic_replace_cell_source(ycell, content: str) -> None:
    """Atomically replaces cell source: del old content then insert new, no await between ops."""
    old_content = ycell.to_py().get("source", "")
    if old_content == content:
        return
    cell_source = ycell["source"]
    del cell_source[0:len(old_content)]
    cell_source.insert(0, content)


async def write_to_cell_collaboratively(
    ynotebook, ycell, content: str, typing_speed: float = 0.1
) -> bool:
    """
    Writes content to a Jupyter notebook cell with collaborative typing simulation.

    This function provides a collaborative writing experience by applying text changes
    incrementally with visual feedback. It uses a diff-based approach to compute the
    minimal set of changes needed and applies them with cursor positioning and timing
    delays to simulate natural typing behavior.

    The function handles three types of operations:
    - Delete: Removes text with visual highlighting
    - Insert: Adds text word-by-word with typing delays
    - Replace: Combines delete and insert operations

    Args:
        ynotebook: The YNotebook instance representing the collaborative notebook
        ycell: The YCell instance representing the specific cell to modify
        content: The new content to write to the cell
        typing_speed: Delay in seconds between typing operations (default: 0.1)

    Returns:
        bool: True if the operation completed successfully

    Raises:
        ValueError: If ynotebook/ycell is None or typing_speed is negative
        TypeError: If content is not a string
        RuntimeError: If cell content extraction or writing fails

    Example:
        >>> # Write with default typing speed
        >>> success = await write_to_cell_collaboratively(ynotebook, ycell, "print('Hello')")
        >>>
        >>> # Write with custom typing speed (faster)
        >>> success = await write_to_cell_collaboratively(
        ...     ynotebook, ycell, "print('World')", typing_speed=0.05
        ... )
    """
    # Input validation
    if ynotebook is None:
        raise ValueError("ynotebook cannot be None")
    if ycell is None:
        raise ValueError("ycell cannot be None")
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    if typing_speed < 0:
        raise ValueError("typing_speed must be non-negative")

    try:
        # Extract current cell content
        cell = ycell.to_py()
        old_content = cell.get("source", "")
        cell_source = ycell["source"]  # YText object for collaborative editing
        new_content = content

        # Early return if content is unchanged
        if old_content == new_content:
            return True

    except Exception as e:
        raise RuntimeError(f"Failed to extract cell content: {e}")

    try:
        # Compute the minimal set of changes needed using difflib
        sequence_matcher = difflib.SequenceMatcher(None, old_content, new_content)
        cursor_position = 0

        # Set initial cursor position
        _safe_set_cursor(ynotebook, cell_source, cursor_position)

        # Apply each change operation sequentially
        for operation, old_start, old_end, new_start, new_end in sequence_matcher.get_opcodes():
            if operation == "equal":
                # No changes needed for this segment, just advance cursor
                cursor_position += old_end - old_start

            elif operation == "delete":
                # Remove text with visual feedback
                delete_length = old_end - old_start
                await _handle_delete_operation(
                    ynotebook, cell_source, cursor_position, delete_length, typing_speed
                )
                # Cursor stays at same position after deletion

            elif operation == "insert":
                # Add text with typing simulation
                cursor_position = await _handle_insert_operation(
                    ynotebook,
                    cell_source,
                    cursor_position,
                    new_content,
                    new_start,
                    new_end,
                    typing_speed,
                )

            elif operation == "replace":
                # Combine delete and insert operations
                delete_length = old_end - old_start
                cursor_position = await _handle_replace_operation(
                    ynotebook,
                    cell_source,
                    cursor_position,
                    new_content,
                    delete_length,
                    new_start,
                    new_end,
                    typing_speed,
                )

        # Set final cursor position at the end of the content
        _safe_set_cursor(ynotebook, cell_source, cursor_position)

        return True

    except Exception as e:
        raise RuntimeError(f"Failed to write cell content collaboratively: {e}")


async def _handle_delete_operation(
    ynotebook, cell_source, cursor_position: int, delete_length: int, typing_speed: float
) -> None:
    """
    Handle deletion of text chunks with visual feedback.

    This function provides visual feedback during deletion by first highlighting
    the text to be deleted, then removing it after a delay to simulate natural
    deletion behavior in collaborative environments.

    Args:
        ynotebook: The YNotebook instance for cursor positioning
        cell_source: The YText source object representing the cell content
        cursor_position: Current cursor position in the text
        delete_length: Number of characters to delete from cursor position
        typing_speed: Base delay between operations in seconds

    Returns:
        None
    """
    # Highlight the text chunk that will be deleted (visual feedback)
    _safe_set_cursor(ynotebook, cell_source, cursor_position, cursor_position + delete_length)
    await asyncio.sleep(min(0.3, typing_speed * 3))  # Cap highlight duration at 0.3s

    # Perform the actual deletion
    del cell_source[cursor_position : cursor_position + delete_length]
    await asyncio.sleep(typing_speed)


async def _handle_insert_operation(
    ynotebook,
    cell_source,
    cursor_position: int,
    new_content: str,
    new_start: int,
    new_end: int,
    typing_speed: float,
) -> int:
    """
    Handle insertion of text with word-by-word typing simulation.

    This function simulates natural typing behavior by inserting text word-by-word
    with appropriate delays and cursor positioning. It handles both regular text
    and whitespace-only content appropriately.

    Args:
        ynotebook: The YNotebook instance for cursor positioning
        cell_source: The YText source object representing the cell content
        cursor_position: Current cursor position in the text
        new_content: The complete new content string
        new_start: Start index of text to insert in the new content
        new_end: End index of text to insert in the new content
        typing_speed: Base delay between typing operations in seconds

    Returns:
        int: The new cursor position after insertion
    """
    text_to_insert = new_content[new_start:new_end]
    words = text_to_insert.split()

    # Handle whitespace-only or empty insertions
    if not words or text_to_insert.strip() == "":
        cell_source.insert(cursor_position, text_to_insert)
        cursor_position += len(text_to_insert)
        _safe_set_cursor(ynotebook, cell_source, cursor_position)
        await asyncio.sleep(typing_speed)
        return cursor_position

    # Insert text word-by-word with proper spacing and punctuation
    current_pos = 0
    for word in words:
        # Find the position of this word in the text
        word_start = text_to_insert.find(word, current_pos)

        # Insert any whitespace or punctuation before the word
        if word_start > current_pos:
            prefix = text_to_insert[current_pos:word_start]
            cell_source.insert(cursor_position, prefix)
            cursor_position += len(prefix)

        # Insert the word itself
        cell_source.insert(cursor_position, word)
        cursor_position += len(word)
        current_pos = word_start + len(word)

        # Update cursor position and pause for typing effect
        _safe_set_cursor(ynotebook, cell_source, cursor_position)
        await asyncio.sleep(typing_speed)

    # Insert any remaining text after the last word (punctuation, etc.)
    if current_pos < len(text_to_insert):
        suffix = text_to_insert[current_pos:]
        cell_source.insert(cursor_position, suffix)
        cursor_position += len(suffix)
        _safe_set_cursor(ynotebook, cell_source, cursor_position)

    return cursor_position


async def _handle_replace_operation(
    ynotebook,
    cell_source,
    cursor_position: int,
    new_content: str,
    delete_length: int,
    new_start: int,
    new_end: int,
    typing_speed: float,
) -> int:
    """
    Handle replacement operations by deleting then inserting.

    This function simulates natural text replacement behavior by first deleting
    the old text (with visual feedback) and then inserting the new text with
    typing simulation. A pause is added between operations to make the replacement
    feel more natural.

    Args:
        ynotebook: The YNotebook instance for cursor positioning
        cell_source: The YText source object representing the cell content
        cursor_position: Current cursor position in the text
        new_content: The complete new content string
        delete_length: Number of characters to delete from cursor position
        new_start: Start index of replacement text in the new content
        new_end: End index of replacement text in the new content
        typing_speed: Base delay between typing operations in seconds

    Returns:
        int: The new cursor position after replacement
    """
    # First, delete the old text with visual feedback
    await _handle_delete_operation(
        ynotebook, cell_source, cursor_position, delete_length, typing_speed
    )

    # Brief pause between deletion and insertion for natural feel
    await asyncio.sleep(typing_speed * 2)

    # Then, insert the new text with typing simulation
    cursor_position = await _handle_insert_operation(
        ynotebook, cell_source, cursor_position, new_content, new_start, new_end, typing_speed
    )

    return cursor_position


def _safe_set_cursor(
    ynotebook: YNotebook, cell_source: Text, cursor_position: int, stop_cursor: Optional[int] = None
) -> None:
    """
    Safely set cursor position with error handling.

    This function wraps the cursor positioning logic to prevent errors from
    breaking the main collaborative writing operations. Since cursor positioning
    is a visual enhancement rather than a core functionality, errors are silently
    ignored to maintain robustness.

    Args:
        ynotebook: The YNotebook instance for cursor positioning
        cell_source: The YText source object representing the cell content
        cursor_position: The cursor position to set
        stop_cursor: Optional end position for text selections

    Returns:
        None

    Note:
        This function silently ignores all exceptions to prevent cursor
        positioning errors from interfering with the main editing operations.
    """
    try:
        set_cursor_in_ynotebook(ynotebook, cell_source, cursor_position, stop_cursor)
    except Exception:
        # Silently ignore cursor setting errors to avoid breaking the main operation
        # Cursor positioning is a visual enhancement, not critical functionality
        pass


async def get_active_notebook(username: Optional[str] = None) -> Optional[str]:
    """Returns path for the currently active notebook.

    Args:
        username: Optional username to return a specific user's active notebook

    Returns:
        File path for the first active notebook. If username is provided, then
        returns the active notebook for that specific user.
    """
    awareness = await get_global_awareness()
    if not awareness:
        return
    for _, state in awareness.states.items():
        _username = state.get("user", {}).get("username", None)
        if username and username != _username:
            continue

        if (active_notebook := state.get("current")) and "notebook" in active_notebook:
            return active_notebook.replace("notebook:", "")

        if documents := state.get("documents"):
            notebooks = [doc for doc in documents if doc.endswith('.ipynb')]
            if len(notebooks) == 1:
                return notebooks[0]


def _get_active_cell_id_from_ydoc(ydoc: YNotebook, username: Optional[str] = None) -> Optional[str]:
    """Internal helper: Returns the active cell id from a ydoc instance."""
    if not ydoc or not ydoc.awareness:
        return None

    for _, state in ydoc.awareness.states.items():
        _username = state.get("user", {}).get("username", None)
        if username and username != _username:
            continue

        if active_cell_id := state.get("activeCellId"):
            return active_cell_id

    return None


async def get_active_cell_id(notebook_path: str, username: Optional[str] = None) -> Optional[str]:
    """Returns the active (selected) cell ID without reading the full notebook.

    Prefer this over reading the entire notebook when you only need the
    currently selected cell's ID.

    Args:
        notebook_path: Path to the notebook file
        username: Optional username to return a specific user's active cell

    Returns:
        The active cell ID for the notebook, or None if no active cell found
    """
    file_path = normalize_filepath(notebook_path)
    file_id = await get_file_id(file_path)
    ydoc = await get_jupyter_ydoc(file_id)

    return _get_active_cell_id_from_ydoc(ydoc, username)


async def get_open_documents(username: Optional[str] = None) -> Optional[List[str]]:
    """Returns all open documents for the user, excluding chat files.

    Args:
        username: Optional username to return a specific user's open documents

    Returns:
        List of file paths for all open documents (excluding .chat files).
        Returns None if no documents are found or awareness is unavailable.
    """
    awareness = await get_global_awareness()
    if not awareness:
        return None

    for _, state in awareness.states.items():
        _username = state.get("user", {}).get("username", None)
        if username and username != _username:
            continue

        if documents := state.get("documents"):
            filtered_documents = [doc for doc in documents if not doc.endswith('.chat')]
            return filtered_documents if filtered_documents else None

    return None


async def select_cell(cell_id: str, username: Optional[str] = None) -> dict:
    """Selects a cell in the active notebook by navigating to it using cursor movements.

    Args:
        cell_id: The UUID of the cell to select, or a numeric index as string
        username: Optional username to get the active cell for that specific user

    Returns:
        dict: A dictionary containing the response from the last cursor movement

    Raises:
        ValueError: If the cell_id is not found in the notebook
        RuntimeError: If there is no active notebook or notebook is not currently open
    """
    from jupyterlab_commands_toolkit.tools import execute_command

    try:
        file_path = await get_active_notebook(username)
        if not file_path:
            raise RuntimeError("No active notebook found. Please open a notebook first.")

        resolved_cell_id = await _resolve_cell_id(file_path, cell_id)

        file_id = await get_file_id(file_path)
        ydoc = await get_jupyter_ydoc(file_id)

        if not ydoc:
            raise RuntimeError(f"Notebook at {file_path} is not currently open")

        target_cell_index = _get_cell_index_from_id_ydoc(ydoc, resolved_cell_id)
        if target_cell_index is None:
            raise ValueError(f"Cell with ID {cell_id} not found in notebook")

        active_cell_id = _get_active_cell_id_from_ydoc(ydoc, username)
        if not active_cell_id:
            raise RuntimeError("No active cell found. Make sure the notebook is focused.")

        active_cell_index = _get_cell_index_from_id_ydoc(ydoc, active_cell_id)
        if active_cell_index is None:
            raise RuntimeError(f"Active cell {active_cell_id} not found in notebook")

        distance = target_cell_index - active_cell_index

        if distance == 0:
            return {"success": True, "result": "Already at target cell"}

        result = None
        if distance > 0:
            for _ in range(distance):
                result = await execute_command("notebook:move-cursor-down")
        else:
            for _ in range(abs(distance)):
                result = await execute_command("notebook:move-cursor-up")

        return result

    except Exception:
        raise


async def edit_cell(file_path: str, cell_id: str, content: str, animate: bool = False) -> None:
    """Edits the content of a notebook cell with the specified ID

    This function modifies the content of a cell in a Jupyter notebook. It first attempts to use
    the in-memory YDoc representation if the notebook is currently active. If the
    notebook is not active, it falls back to using the filesystem to read, modify,
    and write the notebook file directly using nbformat.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        cell_id:
            The UUID of the cell to edit, or a numeric index as string.
        content:
            The new content for the cell. If None, the cell content remains unchanged.

    Returns:
        None

    Raises:
        ValueError: If the cell_id is not found in the notebook.
    """
    try:
        file_path = normalize_filepath(file_path)
        # Resolve cell_id in case it's an index
        resolved_cell_id = await _resolve_cell_id(file_path, cell_id)

        file_id = await get_file_id(file_path)
        ydoc = await get_jupyter_ydoc(file_id)

        if ydoc:
            cell_index = _get_cell_index_from_id_ydoc(ydoc, resolved_cell_id)
            if cell_index is not None:
                ycell = ydoc._ycells[cell_index]
                if animate:
                    await write_to_cell_collaboratively(ydoc, ycell, content)
                else:
                    _atomic_replace_cell_source(ycell, content)
            else:
                raise ValueError(f"Cell with {cell_id=} not found in notebook")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

            cell_index = _get_cell_index_from_id_nbformat(notebook, resolved_cell_id)
            if cell_index is not None:
                notebook.cells[cell_index].source = content
                with open(file_path, "w", encoding="utf-8") as f:
                    nbformat.write(notebook, f)
            else:
                raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")

        return {"success": True}
    except Exception:
        raise


# Note: This is currently failing with server outputs, use `read_cell` instead
def read_cell_nbformat(file_path: str, cell_id: str) -> Dict[str, Any]:
    """Returns the content and metadata of a cell with the specified ID.

    This function reads a specific cell from a Jupyter notebook file using the nbformat
    library and returns the cell's content and metadata.

    Note: This function is currently not functioning properly with server outputs.
    Use `read_cell` instead.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        cell_id:
            The UUID of the cell to read.

    Returns:
        The cell as a dictionary containing its content and metadata.

    Raises:
        ValueError: If no cell with the given ID is found.
    """
    file_path = normalize_filepath(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id)
    if cell_index is not None:
        cell = notebook.cells[cell_index]
        return cell
    else:
        raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")


def _get_cell_index_from_id_json(notebook_json, cell_id: str) -> Optional[int]:
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


def _get_cell_index_from_id_ydoc(ydoc, cell_id: str) -> Optional[int]:
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


def _get_cell_index_from_id_nbformat(notebook, cell_id: str) -> Optional[int]:
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


@lru_cache(maxsize=1)
def list_available_kernelspecs():
    """Lists all available Jupyter kernels and their details."""
    from jupyter_client.kernelspec import KernelSpecManager

    ksm = KernelSpecManager()
    kernels = ksm.find_kernel_specs()
    specs = []
    for kernel_name, _ in kernels.items():
        try:
            spec = ksm.get_kernel_spec(kernel_name)
            specs.append({"name": kernel_name, "display_name": spec.display_name, "language": spec.language})
        except Exception:
            specs = [{"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}]
    return specs


async def create_notebook(file_path: str, kernel_name: str = None) -> str:
    """Creates a new Jupyter notebook at the specified file path.

    The new notebook starts with one empty default cell. Account for this
    when adding cells.

    Args:
        file_path:
            The path where the new notebook should be created.
        kernel_name:
            Optional kernel name (e.g. "python3"). Defaults to the first
            available kernel. Use "python3" for general-purpose notebooks.

    Returns:
        A success message or error message.
    """
    try:
        from .jupyterlab import open_file

        file_path = normalize_filepath(file_path)

        if os.path.exists(file_path):
            raise FileExistsError(f"Notebook at path {file_path} already exists.")

        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        notebook = nbformat.v4.new_notebook()

        kernelspecs = list_available_kernelspecs()
        if kernel_name:
            spec = next((s for s in kernelspecs if s["name"] == kernel_name), None)
            if not spec:
                available = [s["name"] for s in kernelspecs]
                raise ValueError(f"Kernel '{kernel_name}' not found. Available: {available}")
        else:
            spec = kernelspecs[0]

        notebook["metadata"] = {"kernelspec": spec}

        with open(file_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

        await open_file(file_path)
        return f"Successfully created and opened notebook: {file_path}"

    except Exception as e:
        return f"Error: Failed to create notebook: {str(e)}"


toolkit = [
    read_notebook,
    read_notebook_cells,
    read_cell,
    add_cell,
    insert_cell,
    delete_cell,
    edit_cell,
    select_cell,
    get_cell_id_from_index,
    get_active_notebook,
    get_active_cell_id,
    get_open_documents,
    create_notebook,
]
