import json
import asyncio
import difflib

from typing import Any, Dict, Literal, Optional, Tuple

import nbformat
from jupyter_ai.tools.models import Tool, Toolkit
from pycrdt import Awareness, Doc, Text, Assoc
from jupyter_ydoc import YNotebook

from ..utils import cell_to_md, get_file_id, get_jupyter_ydoc, notebook_json_to_md, get_global_awareness, collaborative_tool
import re


def _is_uuid_like(value: str) -> bool:
    """Check if a string looks like a UUID v4"""
    if not isinstance(value, str):
        return False
    # UUID v4 pattern: 8-4-4-4-12 hexadecimal characters
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
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
        notebook_dict = await read_notebook_json(file_path)
        notebook_md = notebook_json_to_md(notebook_dict, include_outputs=include_outputs)
        return notebook_md
    except Exception as e:
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
        with open(file_path, "r", encoding="utf-8") as f:
            notebook_dict = json.load(f)
            return notebook_dict
    except Exception as e:
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
        # Resolve cell_id in case it's an index
        resolved_cell_id = await _resolve_cell_id(file_path, cell_id)
        cell, cell_index = await read_cell_json(file_path, resolved_cell_id)
        cell_md = cell_to_md(cell, cell_index)
        return cell_md
    except Exception as e:
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
        # Resolve cell_id in case it's an index
        resolved_cell_id = await _resolve_cell_id(file_path, cell_id)
        notebook_json = await read_notebook_json(file_path)
        cell_index = _get_cell_index_from_id_json(notebook_json, resolved_cell_id)
        
        if cell_index is not None and 0 <= cell_index < len(notebook_json["cells"]):
            cell = notebook_json["cells"][cell_index]
            return cell, cell_index
        
        raise LookupError(f"No cell found with {cell_id=}")
        
    except Exception as e:
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
        
    except Exception as e:
        raise


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
        
        # Resolve cell_id in case it's an index
        resolved_cell_id = await _resolve_cell_id(file_path, cell_id) if cell_id else None
        
        file_id = await get_file_id(file_path)
        ydoc: YNotebook = await get_jupyter_ydoc(file_id)

        if ydoc:
            cells_count = ydoc.cell_number
            cell_index = _get_cell_index_from_id_ydoc(ydoc, resolved_cell_id) if resolved_cell_id else None
            insert_index = _determine_insert_index(cells_count, cell_index, add_above)
            
            cell = {
                "cell_type": cell_type,
                "source": "",
            }
            ycell = ydoc.create_ycell(cell)
            if insert_index >= cells_count:
                ydoc.ycells.append(ycell)
            else:
                ydoc.ycells.insert(insert_index, ycell)
            await write_to_cell_collaboratively(ydoc, ycell, content or "")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

            cells_count = len(notebook.cells)
            cell_index = _get_cell_index_from_id_nbformat(notebook, resolved_cell_id) if resolved_cell_id else None
            insert_index = _determine_insert_index(cells_count, cell_index, add_above)

            if cell_type == "code":
                notebook.cells.insert(insert_index, nbformat.v4.new_code_cell(source=content or ""))
            elif cell_type == "markdown":
                notebook.cells.insert(insert_index, nbformat.v4.new_markdown_cell(source=content or ""))
            else:
                notebook.cells.insert(insert_index, nbformat.v4.new_raw_cell(source=content or ""))

            with open(file_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)
            
    except Exception as e:
        raise


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
        
        file_id = await get_file_id(file_path)
        ydoc = await get_jupyter_ydoc(file_id)

        if ydoc:
            cells_count = ydoc.cell_number
            
            cell = {
                "cell_type": cell_type,
                "source": "",
            }
            ycell = ydoc.create_ycell(cell)
            if insert_index >= cells_count:
                ydoc.ycells.append(ycell)
            else:
                ydoc.ycells.insert(insert_index, ycell)
            await write_to_cell_collaboratively(ydoc, ycell, content or "")
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
            
    except Exception as e:
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
            
    except Exception as e:
        raise


def get_cursor_details(cell_source: Text, start_index: int, stop_index: Optional[int] = None) -> Dict[str, Any]:
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
        "assoc": 0
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
            "assoc": 0
        }
        cursor_details["empty"] = False  # Not empty when there's a selection
    
    return cursor_details


def set_cursor_in_ynotebook(ynotebook: YNotebook, cell_source: Text, start_index: int, stop_index: Optional[int] = None) -> None:
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


async def write_to_cell_collaboratively(ynotebook, ycell, content: str, typing_speed: float = 0.1) -> bool:
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
                    ynotebook, cell_source, cursor_position, new_content, new_start, new_end, typing_speed
                )
                
            elif operation == "replace":
                # Combine delete and insert operations
                delete_length = old_end - old_start
                cursor_position = await _handle_replace_operation(
                    ynotebook, cell_source, cursor_position, new_content, 
                    delete_length, new_start, new_end, typing_speed
                )
        
        # Set final cursor position at the end of the content
        _safe_set_cursor(ynotebook, cell_source, cursor_position)
        
        return True
        
    except Exception as e:
        raise RuntimeError(f"Failed to write cell content collaboratively: {e}")


async def _handle_delete_operation(ynotebook, cell_source, cursor_position: int, delete_length: int, typing_speed: float) -> None:
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
    del cell_source[cursor_position:cursor_position + delete_length]
    await asyncio.sleep(typing_speed)


async def _handle_insert_operation(ynotebook, cell_source, cursor_position: int, new_content: str, new_start: int, new_end: int, typing_speed: float) -> int:
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


async def _handle_replace_operation(ynotebook, cell_source, cursor_position: int, new_content: str, delete_length: int, new_start: int, new_end: int, typing_speed: float) -> int:
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
    await _handle_delete_operation(ynotebook, cell_source, cursor_position, delete_length, typing_speed)
    
    # Brief pause between deletion and insertion for natural feel
    await asyncio.sleep(typing_speed * 2)
    
    # Then, insert the new text with typing simulation
    cursor_position = await _handle_insert_operation(ynotebook, cell_source, cursor_position, new_content, new_start, new_end, typing_speed)
    
    return cursor_position


def _safe_set_cursor(ynotebook: YNotebook, cell_source: Text, cursor_position: int, stop_cursor: Optional[int] = None) -> None:
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


async def edit_cell(file_path: str, cell_id: str, content: str) -> None:
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
        
        # Resolve cell_id in case it's an index
        resolved_cell_id = await _resolve_cell_id(file_path, cell_id)
        
        file_id = await get_file_id(file_path)
        ydoc = await get_jupyter_ydoc(file_id)

        if ydoc:
            cell_index = _get_cell_index_from_id_ydoc(ydoc, resolved_cell_id)
            if cell_index is not None:
                ycell = ydoc._ycells[cell_index]
                await write_to_cell_collaboratively(ydoc, ycell, content)
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
                
    except Exception as e:
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




# async def set_persona_awareness(
#     file_path: str,
#     username: str,
#     name: str,
#     display_name: str,
#     initials: str,
#     avatar_url: str = "",
#     color: str = "var(--jp-collaborator-color1)",
#     mention_name: str = "",
#     current: str = "",
#     documents: list = None
# ) -> str:
#     """Sets user awareness information in the notebook's YDoc.
    
#     This function sets both the local and global "user" field in the notebook's
#     awareness state with the provided user information based on the Jupyter Server user model.
    
#     Args:
#         file_path: The relative path to the notebook file on the filesystem.
#         username: The username of the user
#         name: The full name of the user
#         display_name: The display name for the user
#         initials: User's initials for avatar display
#         avatar_url: URL to the user's avatar image (optional)
#         color: CSS color variable for user identification (optional)
#         mention_name: The mention name for @-mentions (optional, defaults to @username)
#         current: Current context/status string (optional)
#         documents: List of documents the user is working with (optional)
    
#     Returns:
#         Success message or error message
#     """
#     try:
#         print(f"DEBUG: set_user_awareness called with file_path='{file_path}', username='{username}'")
        
#         file_id = await get_file_id(file_path)
#         ydoc = await get_jupyter_ydoc(file_id)
#         global_awareness = await get_global_awareness()
        
#         if not ydoc:
#             return f"Error: Could not access notebook document for {file_path}. Notebook may not be open."
        
#         # Set default mention_name if not provided
#         if not mention_name:
#             mention_name = f"@{username}"
            
#         # Set default documents list if not provided
#         if documents is None:
#             documents = [file_path]
        
#         # Create user model based on Jupyter Server user model
#         user_model = {
#             "username": username,
#             "name": name,
#             "display_name": display_name,
#             "initials": initials,
#             "avatar_url": avatar_url,
#             "color": color,
#             "mention_name": mention_name
#         }
        
#         print(f"DEBUG: set_user_awareness setting user awareness with model: {user_model}")
        
#         # Set the local user field in the notebook's awareness
#         ydoc.awareness.set_local_state_field("user", user_model)
        
#         # Create global awareness state with user, current, and documents fields
#         global_state = {
#             "user": user_model,
#             "current": current,
#             "documents": documents
#         }
        
#         print(f"DEBUG: set_user_awareness setting global awareness state: {global_state}")
        
#         # Set the global awareness state
#         global_awareness.set_local_state(global_state)
        
#         print("DEBUG: set_user_awareness completed successfully")
#         return f"Successfully set user awareness for {display_name} ({username}) in notebook {file_path} with current='{current}' and {len(documents)} documents"
        
#     except Exception as e:
#         print(f"ERROR: set_user_awareness failed for {file_path}, username={username}: {str(e)}")
#         return f"Error setting user awareness: {str(e)}"


toolkit = Toolkit(
    name="notebook_toolkit",
    description="Tools for reading and manipulating Jupyter notebooks.",
)
toolkit.add_tool(Tool(callable=read_notebook, read=True))
# toolkit.add_tool(Tool(callable=write_to_cell, read=True, write=True))
toolkit.add_tool(Tool(callable=read_cell, read=True))
toolkit.add_tool(Tool(callable=add_cell, read=True, write=True))
toolkit.add_tool(Tool(callable=insert_cell, read=True, write=True))
toolkit.add_tool(Tool(callable=delete_cell, delete=True))
toolkit.add_tool(Tool(callable=edit_cell, read=True, write=True))
toolkit.add_tool(Tool(callable=get_cell_id_from_index, read=True))
