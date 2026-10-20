from typing import Optional

import nbformat


NEW_CELL = {
    "code": nbformat.v4.new_code_cell,
    "markdown": nbformat.v4.new_markdown_cell,
    "raw": nbformat.v4.new_raw_cell,
}


def read_notebook_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=nbformat.NO_CONVERT)


def write_notebook_file(file_path: str, notebook) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)


def get_cell_index_from_id_json(notebook_json, cell_id: str) -> Optional[int]:
    """Get cell index from cell_id by notebook json dict."""
    for i, cell in enumerate(notebook_json["cells"]):
        if "id" in cell and cell["id"] == cell_id:
            return i
    return None


def get_cell_index_from_id_nbformat(notebook, cell_id: str) -> Optional[int]:
    """Get cell index from cell_id using nbformat interface."""
    for i, cell in enumerate(notebook.cells):
        if hasattr(cell, "id") and cell.id == cell_id:
            return i
        elif hasattr(cell, "metadata") and cell.metadata.get("id") == cell_id:
            return i
    return None


def determine_insert_index(cells_count: int, cell_index: Optional[int], add_above: bool) -> int:
    """Determine the index where a new cell should be inserted."""
    if cell_index is None:
        insert_index = cells_count
    else:
        if not (0 <= cell_index < cells_count):
            cell_index = max(0, min(cell_index, cells_count))
        insert_index = cell_index if add_above else cell_index + 1
    return insert_index
