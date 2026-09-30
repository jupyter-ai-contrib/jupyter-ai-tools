import asyncio
import difflib
import json
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

import nbformat
from jupyter_ydoc import YNotebook
from pycrdt import Assoc, Text

from ..utils import (
    cell_to_md,
    command_result,
    get_file_id,
    get_global_awareness,
    get_jupyter_ydoc,
    no_web_client,
    normalize_filepath,
    notebook_json_to_md,
    rtc_available,
    run_lab_command,
)

if TYPE_CHECKING:
    from mcp.types import ImageContent

logger = logging.getLogger(__name__)


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
        ename = output_data.get("ename", "")
        evalue = output_data.get("evalue", "")
        traceback = "\n".join(output_data.get("traceback", []))
        error_text = f"{ename}: {evalue}\n{traceback}"
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


def format_notebook_cell(
    cell_data: Dict[str, Any], cell_index: int, language: str, include_full_outputs: bool = False
) -> Dict[str, Any]:
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
        "source": (
            "".join(cell_data["source"])
            if isinstance(cell_data["source"], list)
            else cell_data["source"]
        ),
        "execution_count": (
            cell_data.get("execution_count") if cell_data["cell_type"] == "code" else None
        ),
        "cell_id": cell_id,
    }

    if cell_data["cell_type"] == "code":
        formatted_cell["language"] = language

    if cell_data["cell_type"] == "code" and cell_data.get("outputs"):
        processed_outputs = [process_notebook_output(output) for output in cell_data["outputs"]]

        if not include_full_outputs and len(json.dumps(processed_outputs)) > 10000:
            formatted_cell["outputs"] = [
                {
                    "output_type": "stream",
                    "text": (
                        "Outputs are too large to include. Use command with: "
                        f"cat <notebook_path> | jq '.cells[{cell_index}].outputs'"
                    ),
                }
            ]
        else:
            formatted_cell["outputs"] = processed_outputs

    return formatted_cell


async def read_notebook_cells(
    notebook_path: str, specific_cell_id: Optional[str] = None
) -> List[Dict[str, Any]]:
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
    notebook_data = await read_notebook_json(notebook_path)

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
        notebook_dict = await read_notebook_json(file_path)
        notebook_md = notebook_json_to_md(notebook_dict, include_outputs=include_outputs)
        return notebook_md
    except Exception:
        raise


async def read_notebook_json(file_path: str) -> Dict[str, Any]:
    """Returns the complete notebook content as a JSON dictionary.

    This is the single choke point every read tool goes through to obtain the
    notebook as an nbformat dict.

    When no RTC provider is active (the RTC-free case), the write tools mutate
    the live in-browser notebook model via jupyterlab-ai-commands *without*
    saving to disk -- saving is left to the human. Reading from disk would
    therefore return stale content (see issue #39). To stay consistent with the
    write tools, we read the live model through the
    ``jupyterlab-ai-commands:get-notebook-content`` frontend command, which
    returns the current (possibly unsaved) nbformat JSON.

    When an RTC provider is active, the server owns a live YDoc that is synced
    to the filesystem, so the on-disk content is already current and we read it
    directly. The same applies when no web client is connected: nothing holds a
    live model, so the file on disk is read.

    Args:
        file_path:
            The relative path to the notebook file. Passed to the frontend
            command as-is (mirroring the write tools); normalized to an absolute
            filesystem path only for the on-disk read.

    Returns:
        A dictionary containing the complete notebook structure.
    """
    if not rtc_available():
        # RTC-free: read the live model (mirrors how the write tools operate).
        res = await run_lab_command(
            "jupyterlab-ai-commands:get-notebook-content",
            {"notebookPath": file_path},
        )
        if not no_web_client(res):
            return command_result(res)["content"]

    normalized_path = normalize_filepath(file_path)
    with open(normalized_path, "r", encoding="utf-8") as f:
        notebook_dict = json.load(f)
        return notebook_dict


async def read_cell(file_path: str, cell_id: str, include_outputs: bool = True) -> str:
    """Returns the notebook cell as a markdown string.

    This function reads a specific cell from a Jupyter notebook file and converts
    it to a markdown string. It uses the read_cell_json function to read the cell
    and then converts it to markdown.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        cell_id:
            The nbformat id of the cell.
        include_outputs:
            If True, cell outputs will be included in the markdown. Default is True.

    Returns:
        The cell content as a markdown string.

    Raises:
        LookupError: If no cell with the given ID is found.
    """
    try:
        # Resolve cell_id in case it's an index
        resolved_cell_id = cell_id
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
            The nbformat id of the cell.

    Returns:
        A tuple containing:
        - The cell as a dictionary
        - The index of the cell in the notebook

    Raises:
        LookupError: If no cell with the given ID is found.
    """
    try:
        # Resolve cell_id in case it's an index
        resolved_cell_id = cell_id
        notebook_json = await read_notebook_json(file_path)
        cell_index = _get_cell_index_from_id_json(notebook_json, resolved_cell_id)

        if cell_index is not None and 0 <= cell_index < len(notebook_json["cells"]):
            cell = notebook_json["cells"][cell_index]
            return cell, cell_index

        raise LookupError(f"No cell found with {cell_id=}")

    except Exception:
        raise


async def read_cell_image(
    file_path: str,
    cell_id: str,
    output_index: Optional[int] = None,
) -> Optional["ImageContent"]:
    """Returns a single image from a cell's outputs as an MCP ImageContent.

    Returns the first supported image found. Use ``output_index`` to access a
    specific output. Use ``read_cell`` to discover the cell's outputs (note:
    image outputs are not yet enumerated there — see issue #27).

    Reads a specific cell from a Jupyter notebook and returns the first
    supported image found in its outputs (typically a matplotlib plot or other
    rich display_data) as an ``mcp.types.ImageContent``. The base64 payload
    already stored in the .ipynb JSON is passed through unchanged; no
    decode/re-encode round-trip is performed.

    Why the ImageContent return type (and not a base64 string)
    ----------------------------------------------------------
    Multimodal LLMs route inputs through two separate channels:

    * the **text channel**, which tokenizes text for the language model, and
    * the **image channel**, which feeds bytes into a vision encoder.

    A base64 string returned as plain text from an MCP tool lands in the text
    channel — the model sees tens of thousands of opaque tokens, not a plot.
    ``mcp.types.ImageContent`` is the protocol-level signal that tells the
    consumer (e.g. jupyter-ai) "wire these bytes into the vision channel of
    the downstream LLM call." Without it, returning the image is effectively
    indistinguishable from returning nothing useful.

    As a design aside: OpenAI's chat completions API accepts
    ``data:image/png;base64,...`` URLs directly in ``image_url`` inputs, but
    Claude and Gemini do not — their SDKs require explicit image content
    blocks with separate data/media-type fields. Either way, the tool-result
    boundary here is MCP, so ``ImageContent`` is the only correct return.

    MCP is an optional extra of ``jupyter-ai-tools``. The import is lazy; if
    the ``mcp`` package is not installed, calling this function raises a
    clear ``RuntimeError`` pointing at ``pip install jupyter-ai-tools[mcp]``.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        cell_id:
            The nbformat id of the cell.
        output_index:
            If provided, inspect only that single output. If None (default),
            scan all outputs and return the first supported image found.

    Returns:
        An ``ImageContent`` for the first image/png, image/jpeg, or image/gif
        found, or None if no supported image is present.

        image/svg+xml is intentionally skipped — most multimodal providers do
        not accept SVG as image input. TODO: rasterize SVG to PNG, or return
        the raw XML as text.

    Raises:
        LookupError: If no cell with the given ID is found.
        IndexError: If output_index is provided but is out of range for the
            cell's outputs.
        RuntimeError: If the ``mcp`` package is not installed.
    """
    try:
        from mcp.types import ImageContent
    except ImportError as e:
        raise RuntimeError(
            "read_cell_image requires the 'mcp' package. "
            "Install it with: pip install jupyter-ai-tools[mcp]  "
            "(or: pip install mcp)"
        ) from e

    cell, _ = await read_cell_json(file_path, cell_id)
    outputs = cell.get("outputs") or []

    if output_index is not None:
        if not 0 <= output_index < len(outputs):
            raise IndexError(
                f"output_index {output_index} out of range " f"(cell has {len(outputs)} outputs)"
            )
        candidates = [outputs[output_index]]
    else:
        candidates = outputs

    supported_mime_types = ("image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp")

    for output in candidates:
        if output.get("output_type") not in ("display_data", "execute_result"):
            continue
        data = output.get("data", {})
        for mime_type in supported_mime_types:
            if mime_type not in data:
                continue
            payload = data[mime_type]
            # nbformat may store base64 as a list of strings or with trailing
            # whitespace; normalize to a clean single string.
            if isinstance(payload, list):
                payload = "".join(payload)
            payload = "".join(payload.split())
            if mime_type == "image/gif":
                logger.warning(
                    "Returning image/gif from %s (cell_id=%s); multimodal "
                    "provider support for GIFs is limited (e.g. Claude uses "
                    "the first frame only, OpenAI may reject).",
                    file_path,
                    cell_id,
                )
            # Normalize the non-standard image/jpg alias to image/jpeg.
            reported_mime = "image/jpeg" if mime_type == "image/jpg" else mime_type
            # NOTE: Returns on first match. Multi-image support tracked in issue #27.
            # Build via model_validate with the JSON key so this works across
            # mcp versions: older mcp names the field ``mimeType``; newer mcp
            # renamed it to ``mime_type`` with ``mimeType`` as the alias.
            return ImageContent.model_validate(
                {"type": "image", "data": payload, "mimeType": reported_mime}
            )

    return None


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

    except Exception:
        raise


_NEW_CELL = {
    "code": nbformat.v4.new_code_cell,
    "markdown": nbformat.v4.new_markdown_cell,
    "raw": nbformat.v4.new_raw_cell,
}


def _read_notebook_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=nbformat.NO_CONVERT)


def _write_notebook_file(file_path: str, notebook) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)


def _file_result(message: str, **fields) -> dict:
    """Build the result of a write tool that edited the notebook file on disk.

    The message tells the agent that the edit went to the file, not to a
    notebook open in JupyterLab.
    """
    return {"success": True, "result": {"message": message, **fields}}


def _add_cell_to_file(
    file_path: str,
    content: Optional[str],
    cell_id: Optional[str],
    add_above: bool,
    cell_type: str,
) -> str:
    """Add a cell to the notebook file on disk and return the id of the new cell."""
    notebook = _read_notebook_file(file_path)
    cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id) if cell_id else None
    insert_index = _determine_insert_index(len(notebook.cells), cell_index, add_above)
    return _insert_cell_in_file(file_path, content, insert_index, cell_type, notebook)


def _insert_cell_in_file(
    file_path: str,
    content: Optional[str],
    insert_index: Optional[int],
    cell_type: str,
    notebook=None,
) -> str:
    """Insert a cell in the notebook file on disk and return the id of the new cell."""
    if notebook is None:
        notebook = _read_notebook_file(file_path)
    if insert_index is None:
        insert_index = len(notebook.cells)
    cell = _NEW_CELL[cell_type](source=content or "")
    notebook.cells.insert(insert_index, cell)
    _write_notebook_file(file_path, notebook)
    return cell.id


def _delete_cell_from_file(file_path: str, cell_id: str) -> Optional[int]:
    """Delete a cell from the notebook file on disk and return its former index."""
    notebook = _read_notebook_file(file_path)
    cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id)
    if cell_index is not None:
        notebook.cells.pop(cell_index)
        _write_notebook_file(file_path, notebook)
    return cell_index


def _edit_cell_in_file(
    file_path: str,
    cell_id: str,
    content: Optional[str],
    cell_type: Optional[str],
) -> None:
    """Edit the content and/or type of a cell in the notebook file on disk."""
    notebook = _read_notebook_file(file_path)
    cell_index = _get_cell_index_from_id_nbformat(notebook, cell_id)
    if cell_index is None:
        raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")

    old_cell = notebook.cells[cell_index]
    source = content if content is not None else old_cell.source

    if cell_type is not None and cell_type != old_cell.cell_type:
        new_cell = _NEW_CELL[cell_type](source=source)
        if getattr(old_cell, "id", None):
            new_cell.id = old_cell.id
        new_cell.metadata.update(old_cell.get("metadata", {}))
        notebook.cells[cell_index] = new_cell
    elif content is not None:
        old_cell.source = content
    else:
        return
    _write_notebook_file(file_path, notebook)


def _is_single_empty_notebook(ydoc: YNotebook) -> bool:
    """True iff the notebook has exactly one cell and it is empty.

    Mirrors jupyterlab-ai-commands' add-cell behavior, which replaces a lone
    empty first cell instead of appending a new one, so the RTC and RTC-free
    paths agree.
    """
    try:
        cells = ydoc.get().get("cells", [])
        if len(cells) != 1:
            return False
        source = cells[0].get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        return not source.strip()
    except Exception:
        return False


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
            The nbformat id of the cell to add relative to. If None,
            the cell is added at the end of the notebook.
        add_above:
            If True, the cell is added above the specified cell. If False,
            it's added below the specified cell.
        cell_type:
            The type of cell to add ("code", "markdown", "raw").

    Returns:
        None
    """
    if not rtc_available():
        # RTC-free: drive the JupyterLab frontend via jupyterlab-ai-commands.
        result = await run_lab_command(
            "jupyterlab-ai-commands:add-cell",
            {
                "notebookPath": file_path,
                "referenceCellId": cell_id,
                "content": content or "",
                "cellType": cell_type,
                "position": "above" if add_above else "below",
            },
        )
        if not no_web_client(result):
            return result
        new_cell_id = _add_cell_to_file(
            normalize_filepath(file_path), content, cell_id, add_above, cell_type
        )
        return _file_result(
            f"{cell_type} cell added to the notebook file on disk, "
            "as the notebook is not open in JupyterLab",
            cellId=new_cell_id,
        )
    try:
        file_path = normalize_filepath(file_path)
        # Resolve cell_id in case it's an index
        resolved_cell_id = cell_id

        file_id = await get_file_id(file_path)
        ydoc: YNotebook = await get_jupyter_ydoc(file_id)

        if ydoc:
            cells_count = ydoc.cell_number
            cell_index = (
                _get_cell_index_from_id_ydoc(ydoc, resolved_cell_id) if resolved_cell_id else None
            )
            insert_index = _determine_insert_index(cells_count, cell_index, add_above)

            cell: Dict[str, Any] = {
                "cell_type": cell_type,
                "source": "",
            }
            if cell_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            ycell = ydoc.create_ycell(cell)
            if _is_single_empty_notebook(ydoc):
                # Match jupyterlab-ai-commands: replace a single empty first
                # cell instead of adding a new one.
                del ydoc.ycells[0]
                ydoc.ycells.append(ycell)
            elif insert_index >= cells_count:
                ydoc.ycells.append(ycell)
            else:
                ydoc.ycells.insert(insert_index, ycell)
            if animate:
                await write_to_cell_collaboratively(ydoc, ycell, content or "")
            else:
                _atomic_replace_cell_source(ycell, content or "")
        else:
            _add_cell_to_file(file_path, content, resolved_cell_id, add_above, cell_type)

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
    if not rtc_available():
        # RTC-free: jupyterlab-ai-commands add-cell is reference-cell based, so
        # translate the target index into a (reference cell, position).
        info = await run_lab_command(
            "jupyterlab-ai-commands:get-notebook-info", {"notebookPath": file_path}
        )
        if no_web_client(info):
            new_cell_id = _insert_cell_in_file(
                normalize_filepath(file_path), content, insert_index, cell_type
            )
            return _file_result(
                f"{cell_type} cell inserted in the notebook file on disk, "
                "as the notebook is not open in JupyterLab",
                cellId=new_cell_id,
            )
        if not info.get("success"):
            return info
        cells = (info.get("result") or {}).get("cells") or []
        idx = insert_index if insert_index is not None else len(cells)
        if not cells:
            ref, position = None, "below"
        elif idx <= 0:
            ref, position = cells[0]["cellId"], "above"
        elif idx >= len(cells):
            ref, position = cells[-1]["cellId"], "below"
        else:
            ref, position = cells[idx]["cellId"], "above"
        return await run_lab_command(
            "jupyterlab-ai-commands:add-cell",
            {
                "notebookPath": file_path,
                "referenceCellId": ref,
                "content": content or "",
                "cellType": cell_type,
                "position": position,
            },
        )
    try:
        file_path = normalize_filepath(file_path)
        file_id = await get_file_id(file_path)
        ydoc = await get_jupyter_ydoc(file_id)

        if ydoc:
            cells_count = ydoc.cell_number

            cell: Dict[str, Any] = {
                "cell_type": cell_type,
                "source": "",
            }
            if cell_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            ycell = ydoc.create_ycell(cell)
            if _is_single_empty_notebook(ydoc):
                del ydoc.ycells[0]
                ydoc.ycells.append(ycell)
            elif insert_index is None or insert_index >= cells_count:
                ydoc.ycells.append(ycell)
            else:
                ydoc.ycells.insert(insert_index, ycell)
            if animate:
                await write_to_cell_collaboratively(ydoc, ycell, content or "")
            else:
                _atomic_replace_cell_source(ycell, content or "")
        else:
            _insert_cell_in_file(file_path, content, insert_index, cell_type)

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
        cell_id: The nbformat id of the cell to delete.

    Returns:
        None
    """
    if not rtc_available():
        result = await run_lab_command(
            "jupyterlab-ai-commands:delete-cell",
            {"notebookPath": file_path, "cellId": cell_id},
        )
        if not no_web_client(result):
            return result
        if _delete_cell_from_file(normalize_filepath(file_path), cell_id) is None:
            raise ValueError(f"Could not find cell index for {cell_id=}")
        return _file_result(
            "Cell deleted from the notebook file on disk, "
            "as the notebook is not open in JupyterLab",
            cellId=cell_id,
        )
    try:
        file_path = normalize_filepath(file_path)
        # Resolve cell_id in case it's an index
        resolved_cell_id = cell_id

        file_id = await get_file_id(file_path)
        ydoc = await get_jupyter_ydoc(file_id)

        if ydoc:
            cell_index = _get_cell_index_from_id_ydoc(ydoc, resolved_cell_id)
            if cell_index is not None and 0 <= cell_index < len(ydoc.ycells):
                del ydoc.ycells[cell_index]
            else:
                pass  # Cell not found in ydoc
        else:
            cell_index = _delete_cell_from_file(file_path, resolved_cell_id)

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
    """Atomically replaces cell source: clear then insert new, no await between ops."""
    old_content = ycell.to_py().get("source", "")
    if old_content == content:
        return
    cell_source = ycell["source"]
    cell_source.clear()
    cell_source += content


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
    if not rtc_available():
        resp = await run_lab_command("jupyterlab-ai-commands:get-notebook-info", {})
        if no_web_client(resp):
            raise RuntimeError(resp["error"])
        return (resp.get("result") or {}).get("notebookPath")
    awareness = await get_global_awareness()
    if not awareness:
        return None
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

    return None


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
    if not rtc_available():
        resp = await run_lab_command(
            "jupyterlab-ai-commands:get-notebook-info", {"notebookPath": notebook_path}
        )
        if no_web_client(resp):
            raise RuntimeError(resp["error"])
        return (resp.get("result") or {}).get("activeCellId")
    file_path = normalize_filepath(notebook_path)
    file_id = await get_file_id(file_path)
    ydoc = await get_jupyter_ydoc(file_id)

    return _get_active_cell_id_from_ydoc(ydoc, username)


async def select_cell(
    cell_id: str, username: Optional[str] = None, file_path: Optional[str] = None
) -> dict:
    """Selects a cell in the active notebook by navigating to it using cursor movements.

    Args:
        cell_id: The nbformat id of the cell to select
        username: Optional username to get the active cell for that specific user
        file_path: Optional path to the notebook file. If provided, uses this path
                   instead of deriving the notebook from awareness state.

    Returns:
        dict: A dictionary containing the response from the last cursor movement

    Raises:
        ValueError: If the cell_id is not found in the notebook
        RuntimeError: If there is no active notebook or notebook is not currently open
    """
    from jupyterlab_commands_toolkit.tools import execute_command

    if not rtc_available():
        # RTC-free: read the current + target cell from the frontend
        # (get-notebook-info), then navigate with the same core move-cursor
        # commands the RTC path uses.
        target_path = file_path or await get_active_notebook(username)
        if not target_path:
            raise RuntimeError("No active notebook found. Please open a notebook first.")
        info = await run_lab_command(
            "jupyterlab-ai-commands:get-notebook-info", {"notebookPath": target_path}
        )
        if not info.get("success"):
            return info
        info_result = info.get("result") or {}
        cells = info_result.get("cells") or []
        ids = [c.get("cellId") for c in cells]
        if cell_id not in ids:
            raise ValueError(f"Cell with ID {cell_id} not found in notebook")
        target_index = ids.index(cell_id)
        active_id = info_result.get("activeCellId")
        active_index = ids.index(active_id) if active_id in ids else 0
        distance = target_index - active_index
        if distance == 0:
            return {"success": True, "result": "Already at target cell"}
        cmd = "notebook:move-cursor-down" if distance > 0 else "notebook:move-cursor-up"
        move_result: dict = {}
        for _ in range(abs(distance)):
            move_result = await execute_command(cmd)
        return move_result

    try:
        if not file_path:
            file_path = await get_active_notebook(username)
        if not file_path:
            raise RuntimeError("No active notebook found. Please open a notebook first.")

        resolved_cell_id = cell_id

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

        result: dict = {}
        if distance > 0:
            for _ in range(distance):
                result = await execute_command("notebook:move-cursor-down")
        else:
            for _ in range(abs(distance)):
                result = await execute_command("notebook:move-cursor-up")

        return result

    except Exception:
        raise


async def edit_cell(
    file_path: str,
    cell_id: str,
    content: Optional[str] = None,
    cell_type: Optional[Literal["code", "markdown", "raw"]] = None,
    animate: bool = False,
) -> dict:
    """Edits the content and/or type of a notebook cell with the specified ID.

    This function modifies a cell in a Jupyter notebook. It first attempts to use
    the in-memory YDoc representation if the notebook is currently active. If the
    notebook is not active, it falls back to using the filesystem to read, modify,
    and write the notebook file directly using nbformat.

    When changing cell type, the cell is replaced via ``set_cell()`` to ensure
    type-specific fields (outputs, execution_count) are correctly added or removed.

    Args:
        file_path:
            The relative path to the notebook file on the filesystem.
        cell_id:
            The nbformat id of the cell to edit.
        content:
            The new content for the cell. If None, the existing content is preserved.
        cell_type:
            The new cell type ("code", "markdown", "raw"). If None, the type is unchanged.
        animate:
            If True, simulate collaborative typing when changing content.

    Returns:
        A dictionary ``{"success": True}`` once the edit completes.

    Raises:
        ValueError: If the cell_id is not found in the notebook.
    """
    if not rtc_available():
        rtc_free_result: dict = {"success": True}
        if content is not None:
            rtc_free_result = await run_lab_command(
                "jupyterlab-ai-commands:set-cell-content",
                {
                    "notebookPath": file_path,
                    "cellId": cell_id,
                    "content": content,
                    "showDiff": False,
                },
            )
        if cell_type is not None and rtc_free_result.get("success"):
            # change-cell-to-* act on the selected cell, so select it first.
            rtc_free_result = await select_cell(cell_id, file_path=file_path)
            if rtc_free_result.get("success"):
                type_command = {
                    "code": "notebook:change-cell-to-code",
                    "markdown": "notebook:change-cell-to-markdown",
                    "raw": "notebook:change-cell-to-raw",
                }[cell_type]
                rtc_free_result = await run_lab_command(type_command)
        if no_web_client(rtc_free_result):
            _edit_cell_in_file(normalize_filepath(file_path), cell_id, content, cell_type)
            return _file_result(
                "Cell edited in the notebook file on disk, "
                "as the notebook is not open in JupyterLab",
                cellId=cell_id,
            )
        return rtc_free_result
    try:
        file_path = normalize_filepath(file_path)
        # Resolve cell_id in case it's an index
        resolved_cell_id = cell_id

        file_id = await get_file_id(file_path)
        ydoc = await get_jupyter_ydoc(file_id)

        if ydoc:
            cell_index = _get_cell_index_from_id_ydoc(ydoc, resolved_cell_id)
            if cell_index is None:
                raise ValueError(f"Cell with {cell_id=} not found in notebook")

            ycell = ydoc._ycells[cell_index]
            old_cell = ycell.to_py()
            old_type = old_cell.get("cell_type")
            needs_type_change = cell_type is not None and cell_type != old_type
            source = content if content is not None else old_cell.get("source", "")

            if needs_type_change:
                new_cell = {
                    "cell_type": cell_type,
                    "source": source,
                    "id": old_cell.get("id", str(uuid4())),
                    "metadata": old_cell.get("metadata", {}),
                }
                if cell_type == "code":
                    new_cell["outputs"] = []
                    new_cell["execution_count"] = None
                ydoc.set_cell(cell_index, new_cell)
                if animate and content is not None:
                    # After replacement, write content collaboratively on the new ycell
                    new_ycell = ydoc._ycells[cell_index]
                    _atomic_replace_cell_source(new_ycell, "")
                    await write_to_cell_collaboratively(ydoc, new_ycell, source)
            else:
                if content is not None:
                    if animate:
                        await write_to_cell_collaboratively(ydoc, ycell, content)
                    else:
                        _atomic_replace_cell_source(ycell, content)
        else:
            _edit_cell_in_file(file_path, resolved_cell_id, content, cell_type)

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
    for i, ycell in enumerate(ydoc.ycells):
        if ycell.get("id") == cell_id:
            return i
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
            specs.append(
                {"name": kernel_name, "display_name": spec.display_name, "language": spec.language}
            )
        except Exception:
            specs = [
                {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
            ]
    return specs


async def create_notebook(file_path: str, kernel_name: Optional[str] = None) -> str:
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

        _write_notebook_file(file_path, notebook)

        opened = await open_file(file_path)
        if not opened.get("success"):
            return (
                f"Successfully created notebook: {file_path}. "
                f"It is not open in JupyterLab: {opened.get('error')}"
            )
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
    create_notebook,
]
