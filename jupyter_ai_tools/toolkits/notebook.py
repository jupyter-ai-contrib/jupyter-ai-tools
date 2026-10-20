import json
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import nbformat

from ..adapters._notebook_helpers import (
    get_cell_index_from_id_json,
    get_cell_index_from_id_nbformat,
    process_notebook_output,
    write_notebook_file,
)
from ..adapters.notebook import get_adapter
from ..utils import (
    cell_to_md,
    normalize_filepath,
    notebook_json_to_md,
)

if TYPE_CHECKING:
    from mcp.types import ImageContent

logger = logging.getLogger(__name__)


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

    With RTC, reads from disk (the YDoc is kept in sync with the filesystem).
    Without RTC, reads the live model from the JupyterLab frontend (avoiding
    stale disk content when the notebook has unsaved edits), falling back to
    disk when no browser is connected.

    Args:
        file_path: The relative path to the notebook file.

    Returns:
        A dictionary containing the complete notebook structure.
    """
    accessor = await get_adapter(file_path)
    return await accessor.read(file_path)


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
        cell_index = get_cell_index_from_id_json(notebook_json, resolved_cell_id)

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


async def add_cell(
    file_path: str,
    content: Optional[str] = None,
    cell_id: Optional[str] = None,
    add_above: bool = False,
    cell_type: Literal["code", "markdown", "raw"] = "code",
    animate: bool = False,
):
    """Adds a new cell to the Jupyter notebook above or below a specified cell.

    Args:
        file_path: The relative path to the notebook file on the filesystem.
        content: The content of the new cell. If None, an empty cell is created.
        cell_id: The nbformat id of the reference cell. If None, adds at the end.
        add_above: If True, the cell is added above the reference cell.
        cell_type: The type of cell to add ("code", "markdown", "raw").

    Returns:
        None
    """
    accessor = await get_adapter(file_path)
    return await accessor.add_cell(file_path, content, cell_id, add_above, cell_type, animate)


async def insert_cell(
    file_path: str,
    content: Optional[str] = None,
    insert_index: Optional[int] = None,
    cell_type: Literal["code", "markdown", "raw"] = "code",
    animate: bool = False,
):
    """Inserts a new cell to the Jupyter notebook at the specified cell index.

    Args:
        file_path: The relative path to the notebook file on the filesystem.
        content: The content of the new cell. If None, an empty cell is created.
        insert_index: The index to insert the cell at.
        cell_type: The type of cell to add ("code", "markdown", "raw").

    Returns:
        None
    """
    accessor = await get_adapter(file_path)
    return await accessor.insert_cell(file_path, content, insert_index, cell_type, animate)


async def delete_cell(file_path: str, cell_id: str):
    """Removes a notebook cell with the specified cell ID.

    Args:
        file_path: The relative path to the notebook file on the filesystem.
        cell_id: The nbformat id of the cell to delete.

    Returns:
        None
    """
    accessor = await get_adapter(file_path)
    return await accessor.delete_cell(file_path, cell_id)


async def get_active_notebook(username: Optional[str] = None) -> Optional[str]:
    """Returns path for the currently active notebook.

    Args:
        username: Optional username to return a specific user's active notebook

    Returns:
        File path for the first active notebook. If username is provided, then
        returns the active notebook for that specific user.
    """
    accessor = await get_adapter()
    return await accessor.get_active_notebook(username)


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
    accessor = await get_adapter(notebook_path)
    return await accessor.get_active_cell_id(notebook_path, username)


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
    target_path = file_path or await get_active_notebook(username)
    if not target_path:
        raise RuntimeError("No active notebook found. Please open a notebook first.")
    accessor = await get_adapter(target_path)
    return await accessor.select_cell(cell_id, username, target_path)


async def edit_cell(
    file_path: str,
    cell_id: str,
    content: Optional[str] = None,
    cell_type: Optional[Literal["code", "markdown", "raw"]] = None,
    animate: bool = False,
) -> dict:
    """Edits the content and/or type of a notebook cell with the specified ID.

    When changing cell type, the cell is replaced via ``set_cell()`` to ensure
    type-specific fields (outputs, execution_count) are correctly added or removed.

    Args:
        file_path: The relative path to the notebook file on the filesystem.
        cell_id: The nbformat id of the cell to edit.
        content: The new content for the cell. If None, existing content is preserved.
        cell_type: The new cell type. If None, the type is unchanged.
        animate: If True, simulate collaborative typing when changing content.

    Returns:
        A dictionary ``{"success": True}`` once the edit completes.

    Raises:
        ValueError: If the cell_id is not found in the notebook.
    """
    accessor = await get_adapter(file_path)
    return await accessor.edit_cell(file_path, cell_id, content, cell_type, animate)


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

    cell_index = get_cell_index_from_id_nbformat(notebook, cell_id)
    if cell_index is not None:
        cell = notebook.cells[cell_index]
        return cell
    else:
        raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")


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

        write_notebook_file(file_path, notebook)

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
