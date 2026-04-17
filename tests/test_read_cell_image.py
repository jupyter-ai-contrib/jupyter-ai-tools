import base64
import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mcp.types import ImageContent

from jupyter_ai_tools.toolkits.notebook import read_cell_image

PNG_CELL_ID = "11111111-aaaa-4aaa-aaaa-aaaaaaaaaaaa"
JPEG_CELL_ID = "22222222-bbbb-4bbb-bbbb-bbbbbbbbbbbb"
JPG_CELL_ID = "33333333-cccc-4ccc-cccc-cccccccccccc"
GIF_CELL_ID = "44444444-dddd-4ddd-dddd-dddddddddddd"
SVG_CELL_ID = "55555555-eeee-4eee-eeee-eeeeeeeeeeee"
TEXT_ONLY_CELL_ID = "66666666-ffff-4fff-ffff-ffffffffffff"
MULTI_OUTPUT_CELL_ID = "77777777-1111-4111-1111-111111111111"
STREAM_ONLY_CELL_ID = "88888888-2222-4222-2222-222222222222"
WHITESPACE_CELL_ID = "99999999-3333-4333-3333-333333333333"
LIST_PAYLOAD_CELL_ID = "aaaaaaaa-4444-4444-4444-444444444444"


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode()


def _image_output(mime_type: str, payload):
    return {
        "output_type": "display_data",
        "data": {mime_type: payload, "text/plain": ["<Figure>"]},
        "metadata": {},
    }


def _text_output():
    return {
        "output_type": "execute_result",
        "execution_count": 1,
        "data": {"text/plain": ["42"]},
        "metadata": {},
    }


def _stream_output():
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": "hello\n",
    }


def _cell(cell_id: str, outputs):
    return {
        "cell_type": "code",
        "execution_count": 1,
        "id": cell_id,
        "metadata": {},
        "source": "",
        "outputs": outputs,
    }


def _build_notebook() -> dict:
    return {
        "cells": [
            _cell(PNG_CELL_ID, [_image_output("image/png", _b64(b"fake-png-bytes"))]),
            _cell(JPEG_CELL_ID, [_image_output("image/jpeg", _b64(b"fake-jpeg-bytes"))]),
            _cell(JPG_CELL_ID, [_image_output("image/jpg", _b64(b"fake-jpg-bytes"))]),
            _cell(GIF_CELL_ID, [_image_output("image/gif", _b64(b"fake-gif-bytes"))]),
            _cell(
                SVG_CELL_ID,
                [
                    {
                        "output_type": "display_data",
                        "data": {"image/svg+xml": "<svg/>", "text/plain": ["<Figure>"]},
                        "metadata": {},
                    }
                ],
            ),
            _cell(TEXT_ONLY_CELL_ID, [_text_output()]),
            _cell(
                MULTI_OUTPUT_CELL_ID,
                [
                    _text_output(),
                    _image_output("image/png", _b64(b"multi-first-png")),
                    _image_output("image/png", _b64(b"multi-second-png")),
                ],
            ),
            _cell(STREAM_ONLY_CELL_ID, [_stream_output()]),
            _cell(
                WHITESPACE_CELL_ID,
                [_image_output("image/png", _b64(b"whitespace-png") + "\n  \n")],
            ),
            _cell(
                LIST_PAYLOAD_CELL_ID,
                [
                    _image_output(
                        "image/png",
                        [_b64(b"list-payload")[:5], _b64(b"list-payload")[5:]],
                    )
                ],
            ),
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


@pytest.fixture
def notebook_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        nb_path = Path(temp_dir) / "nb.ipynb"
        nb_path.write_text(json.dumps(_build_notebook()))

        with patch("jupyter_ai_tools.utils.get_serverapp") as mock_serverapp:
            mock_app = MagicMock()
            mock_app.root_dir = temp_dir
            mock_serverapp.return_value = mock_app
            yield str(nb_path)


@pytest.mark.asyncio
async def test_returns_png_image_content(notebook_path):
    result = await read_cell_image(notebook_path, PNG_CELL_ID)

    assert isinstance(result, ImageContent)
    assert result.type == "image"
    assert result.mimeType == "image/png"
    assert base64.b64decode(result.data) == b"fake-png-bytes"


@pytest.mark.asyncio
async def test_returns_jpeg_image_content(notebook_path):
    result = await read_cell_image(notebook_path, JPEG_CELL_ID)

    assert isinstance(result, ImageContent)
    assert result.mimeType == "image/jpeg"
    assert base64.b64decode(result.data) == b"fake-jpeg-bytes"


@pytest.mark.asyncio
async def test_normalizes_jpg_alias_to_jpeg(notebook_path):
    result = await read_cell_image(notebook_path, JPG_CELL_ID)

    assert isinstance(result, ImageContent)
    assert result.mimeType == "image/jpeg"
    assert base64.b64decode(result.data) == b"fake-jpg-bytes"


@pytest.mark.asyncio
async def test_returns_gif_and_logs_warning(notebook_path, caplog):
    with caplog.at_level(logging.WARNING, logger="jupyter_ai_tools.toolkits.notebook"):
        result = await read_cell_image(notebook_path, GIF_CELL_ID)

    assert isinstance(result, ImageContent)
    assert result.mimeType == "image/gif"
    assert base64.b64decode(result.data) == b"fake-gif-bytes"
    assert any("image/gif" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_svg_returns_none(notebook_path):
    # TODO: when SVG handling is added, update this test accordingly.
    result = await read_cell_image(notebook_path, SVG_CELL_ID)

    assert result is None


@pytest.mark.asyncio
async def test_text_only_output_returns_none(notebook_path):
    result = await read_cell_image(notebook_path, TEXT_ONLY_CELL_ID)

    assert result is None


@pytest.mark.asyncio
async def test_stream_only_output_returns_none(notebook_path):
    result = await read_cell_image(notebook_path, STREAM_ONLY_CELL_ID)

    assert result is None


@pytest.mark.asyncio
async def test_scans_all_outputs_and_returns_first_image(notebook_path):
    # First output is text, second is the "earlier" PNG — should pick that one.
    result = await read_cell_image(notebook_path, MULTI_OUTPUT_CELL_ID)

    assert isinstance(result, ImageContent)
    assert base64.b64decode(result.data) == b"multi-first-png"


@pytest.mark.asyncio
async def test_output_index_selects_specific_output(notebook_path):
    result = await read_cell_image(notebook_path, MULTI_OUTPUT_CELL_ID, output_index=2)

    assert isinstance(result, ImageContent)
    assert base64.b64decode(result.data) == b"multi-second-png"


@pytest.mark.asyncio
async def test_output_index_skips_non_image_output_returns_none(notebook_path):
    result = await read_cell_image(notebook_path, MULTI_OUTPUT_CELL_ID, output_index=0)

    assert result is None


@pytest.mark.asyncio
async def test_output_index_out_of_range_raises(notebook_path):
    with pytest.raises(IndexError):
        await read_cell_image(notebook_path, PNG_CELL_ID, output_index=99)


@pytest.mark.asyncio
async def test_missing_cell_raises_lookup_error(notebook_path):
    with pytest.raises(LookupError):
        await read_cell_image(notebook_path, "deadbeef-0000-4000-8000-000000000000")


@pytest.mark.asyncio
async def test_base64_whitespace_is_stripped(notebook_path):
    result = await read_cell_image(notebook_path, WHITESPACE_CELL_ID)

    assert isinstance(result, ImageContent)
    # No whitespace in the stored payload.
    assert result.data == result.data.strip()
    assert "\n" not in result.data and " " not in result.data
    assert base64.b64decode(result.data) == b"whitespace-png"


@pytest.mark.asyncio
async def test_base64_list_payload_is_joined(notebook_path):
    result = await read_cell_image(notebook_path, LIST_PAYLOAD_CELL_ID)

    assert isinstance(result, ImageContent)
    assert base64.b64decode(result.data) == b"list-payload"


@pytest.mark.asyncio
async def test_raises_runtime_error_when_mcp_is_missing(notebook_path):
    """When the optional mcp dep is not installed, surface a clear error."""
    removed = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "mcp" or name.startswith("mcp.")
    }
    try:
        with patch.dict(sys.modules, {"mcp": None, "mcp.types": None}):
            with pytest.raises(RuntimeError, match="jupyter-ai-tools\\[mcp\\]"):
                await read_cell_image(notebook_path, PNG_CELL_ID)
    finally:
        sys.modules.update(removed)
