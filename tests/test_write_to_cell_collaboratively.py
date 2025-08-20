"""
Unit tests for the write_to_cell_collaboratively function and its helper functions.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from jupyter_ai_tools.toolkits.notebook import (
    _handle_delete_operation,
    _handle_insert_operation,
    _handle_replace_operation,
    _safe_set_cursor,
    write_to_cell_collaboratively,
)


class TestWriteToCellCollaboratively:
    """Test cases for write_to_cell_collaboratively function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_ynotebook = Mock()
        self.mock_ycell = MagicMock()
        self.mock_source = MagicMock()

        # Mock ycell behavior
        self.mock_ycell.to_py.return_value = {"source": "old content"}
        self.mock_ycell.__getitem__.return_value = self.mock_source

        # Mock source behavior
        self.mock_source.insert = Mock()
        self.mock_source.__delitem__ = Mock()
        self.mock_source.__getitem__ = Mock(return_value="old content")

    @pytest.mark.asyncio
    async def test_input_validation_none_ynotebook(self):
        """Test that None ynotebook raises ValueError."""
        with pytest.raises(ValueError, match="ynotebook cannot be None"):
            await write_to_cell_collaboratively(None, self.mock_ycell, "content")

    @pytest.mark.asyncio
    async def test_input_validation_none_ycell(self):
        """Test that None ycell raises ValueError."""
        with pytest.raises(ValueError, match="ycell cannot be None"):
            await write_to_cell_collaboratively(self.mock_ynotebook, None, "content")

    @pytest.mark.asyncio
    async def test_input_validation_non_string_content(self):
        """Test that non-string content raises TypeError."""
        with pytest.raises(TypeError, match="content must be a string"):
            await write_to_cell_collaboratively(self.mock_ynotebook, self.mock_ycell, 123)  # type: ignore

    @pytest.mark.asyncio
    async def test_input_validation_negative_typing_speed(self):
        """Test that negative typing_speed raises ValueError."""
        with pytest.raises(ValueError, match="typing_speed must be non-negative"):
            await write_to_cell_collaboratively(
                self.mock_ynotebook, self.mock_ycell, "content", typing_speed=-1
            )

    @pytest.mark.asyncio
    async def test_same_content_returns_true(self):
        """Test that same content returns True immediately."""
        self.mock_ycell.to_py.return_value = {"source": "same content"}

        result = await write_to_cell_collaboratively(
            self.mock_ynotebook, self.mock_ycell, "same content"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_cell_extraction_error(self):
        """Test handling of cell content extraction errors."""
        self.mock_ycell.to_py.side_effect = Exception("Cell extraction failed")

        with pytest.raises(RuntimeError, match="Failed to extract cell content"):
            await write_to_cell_collaboratively(self.mock_ynotebook, self.mock_ycell, "new content")

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._safe_set_cursor")
    @patch("jupyter_ai_tools.toolkits.notebook.difflib.SequenceMatcher")
    async def test_successful_write_operation(self, mock_sequence_matcher, mock_safe_set_cursor):
        """Test successful write operation."""
        # Mock SequenceMatcher to return simple equal operation
        mock_sm = Mock()
        mock_sm.get_opcodes.return_value = [("equal", 0, 5, 0, 5)]
        mock_sequence_matcher.return_value = mock_sm

        result = await write_to_cell_collaboratively(
            self.mock_ynotebook, self.mock_ycell, "new content"
        )

        assert result is True
        mock_safe_set_cursor.assert_called()

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._safe_set_cursor")
    @patch("jupyter_ai_tools.toolkits.notebook.difflib.SequenceMatcher")
    async def test_difflib_error_handling(self, mock_sequence_matcher, mock_safe_set_cursor):
        """Test handling of difflib errors."""
        mock_sequence_matcher.side_effect = Exception("Difflib error")

        with pytest.raises(RuntimeError, match="Failed to write cell content collaboratively"):
            await write_to_cell_collaboratively(self.mock_ynotebook, self.mock_ycell, "new content")

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._safe_set_cursor")
    @patch("jupyter_ai_tools.toolkits.notebook._handle_delete_operation")
    @patch("jupyter_ai_tools.toolkits.notebook.difflib.SequenceMatcher")
    async def test_delete_operation_called(
        self, mock_sequence_matcher, mock_delete_op, mock_safe_set_cursor
    ):
        """Test that delete operation is called for delete opcodes."""
        mock_sm = Mock()
        mock_sm.get_opcodes.return_value = [("delete", 0, 5, 0, 0)]
        mock_sequence_matcher.return_value = mock_sm
        mock_delete_op.return_value = None

        result = await write_to_cell_collaboratively(
            self.mock_ynotebook, self.mock_ycell, "new content"
        )

        assert result is True
        mock_delete_op.assert_called_once()

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._safe_set_cursor")
    @patch("jupyter_ai_tools.toolkits.notebook._handle_insert_operation")
    @patch("jupyter_ai_tools.toolkits.notebook.difflib.SequenceMatcher")
    async def test_insert_operation_called(
        self, mock_sequence_matcher, mock_insert_op, mock_safe_set_cursor
    ):
        """Test that insert operation is called for insert opcodes."""
        mock_sm = Mock()
        mock_sm.get_opcodes.return_value = [("insert", 0, 0, 0, 5)]
        mock_sequence_matcher.return_value = mock_sm
        mock_insert_op.return_value = 5

        result = await write_to_cell_collaboratively(
            self.mock_ynotebook, self.mock_ycell, "new content"
        )

        assert result is True
        mock_insert_op.assert_called_once()

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._safe_set_cursor")
    @patch("jupyter_ai_tools.toolkits.notebook._handle_replace_operation")
    @patch("jupyter_ai_tools.toolkits.notebook.difflib.SequenceMatcher")
    async def test_replace_operation_called(
        self, mock_sequence_matcher, mock_replace_op, mock_safe_set_cursor
    ):
        """Test that replace operation is called for replace opcodes."""
        mock_sm = Mock()
        mock_sm.get_opcodes.return_value = [("replace", 0, 5, 0, 7)]
        mock_sequence_matcher.return_value = mock_sm
        mock_replace_op.return_value = 7

        result = await write_to_cell_collaboratively(
            self.mock_ynotebook, self.mock_ycell, "new content"
        )

        assert result is True
        mock_replace_op.assert_called_once()


class TestHandleDeleteOperation:
    """Test cases for _handle_delete_operation function."""

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._safe_set_cursor")
    @patch("jupyter_ai_tools.toolkits.notebook.asyncio.sleep")
    async def test_delete_operation(self, mock_sleep, mock_safe_set_cursor):
        """Test delete operation with proper timing."""
        mock_ynotebook = Mock()
        mock_old_ = MagicMock()
        mock_old_.__delitem__ = Mock()

        await _handle_delete_operation(mock_ynotebook, mock_old_, 0, 5, 0.1)

        # Check that cursor was set and sleep was called
        mock_safe_set_cursor.assert_called_with(mock_ynotebook, mock_old_, 0, 5)
        assert mock_sleep.call_count == 2  # Two sleep calls
        mock_old_.__delitem__.assert_called_with(slice(0, 5))

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._safe_set_cursor")
    @patch("jupyter_ai_tools.toolkits.notebook.asyncio.sleep")
    async def test_delete_operation_with_fast_typing(self, mock_sleep, mock_safe_set_cursor):
        """Test delete operation respects maximum sleep time."""
        mock_ynotebook = Mock()
        mock_old_ = MagicMock()
        mock_old_.__delitem__ = Mock()

        await _handle_delete_operation(mock_ynotebook, mock_old_, 0, 5, 1.0)

        # Should use min(0.3, 1.0 * 3) = 0.3 for first sleep
        mock_sleep.assert_any_call(0.3)


class TestHandleInsertOperation:
    """Test cases for _handle_insert_operation function."""

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._safe_set_cursor")
    @patch("jupyter_ai_tools.toolkits.notebook.asyncio.sleep")
    async def test_insert_whitespace_only(self, mock_sleep, mock_safe_set_cursor):
        """Test insertion of whitespace-only content."""
        mock_ynotebook = Mock()
        mock_old_ = Mock()
        mock_old_.insert = Mock()

        result = await _handle_insert_operation(mock_ynotebook, mock_old_, 0, "  \n  ", 0, 4, 0.1)

        assert result == 4
        # Check that the entire whitespace string was inserted
        mock_old_.insert.assert_called_once()
        call_args = mock_old_.insert.call_args
        assert call_args[0][0] == 0  # cursor position
        # The actual string inserted should be the entire whitespace string
        inserted_text = call_args[0][1]
        assert len(inserted_text) == 4  # Should be 4 characters
        mock_safe_set_cursor.assert_called_with(mock_ynotebook, mock_old_, 4)

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._safe_set_cursor")
    @patch("jupyter_ai_tools.toolkits.notebook.asyncio.sleep")
    async def test_insert_words(self, mock_sleep, mock_safe_set_cursor):
        """Test insertion of words with proper spacing."""
        mock_ynotebook = Mock()
        mock_old_ = Mock()
        mock_old_.insert = Mock()

        result = await _handle_insert_operation(
            mock_ynotebook, mock_old_, 0, "hello world", 0, 11, 0.1
        )

        assert result == 11
        # Should insert "hello" and "world" separately
        assert mock_old_.insert.call_count >= 2
        mock_safe_set_cursor.assert_called()

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._safe_set_cursor")
    @patch("jupyter_ai_tools.toolkits.notebook.asyncio.sleep")
    async def test_insert_with_suffix(self, mock_sleep, mock_safe_set_cursor):
        """Test insertion handles remaining text after last word."""
        mock_ynotebook = Mock()
        mock_old_ = Mock()
        mock_old_.insert = Mock()

        result = await _handle_insert_operation(mock_ynotebook, mock_old_, 0, "hello!", 0, 6, 0.1)

        assert result == 6
        mock_old_.insert.assert_called()
        mock_safe_set_cursor.assert_called()


class TestHandleReplaceOperation:
    """Test cases for _handle_replace_operation function."""

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook._handle_delete_operation")
    @patch("jupyter_ai_tools.toolkits.notebook._handle_insert_operation")
    @patch("jupyter_ai_tools.toolkits.notebook.asyncio.sleep")
    async def test_replace_operation(self, mock_sleep, mock_insert_op, mock_delete_op):
        """Test replace operation calls delete then insert."""
        mock_ynotebook = Mock()
        mock_old_ = Mock()
        mock_delete_op.return_value = None
        mock_insert_op.return_value = 10

        result = await _handle_replace_operation(
            mock_ynotebook, mock_old_, 0, "new content", 5, 0, 10, 0.1
        )

        assert result == 10
        mock_delete_op.assert_called_once_with(mock_ynotebook, mock_old_, 0, 5, 0.1)
        mock_insert_op.assert_called_once_with(
            mock_ynotebook, mock_old_, 0, "new content", 0, 10, 0.1
        )
        mock_sleep.assert_called_with(0.2)  # typing_speed * 2


class TestSafeSetCursor:
    """Test cases for _safe_set_cursor function."""

    @patch("jupyter_ai_tools.toolkits.notebook.set_cursor_in_ynotebook")
    def test_safe_set_cursor_success(self, mock_set_cursor):
        """Test successful cursor setting."""
        mock_ynotebook = Mock()
        mock_old_ = Mock()

        _safe_set_cursor(mock_ynotebook, mock_old_, 5)

        mock_set_cursor.assert_called_once_with(mock_ynotebook, mock_old_, 5, None)

    @patch("jupyter_ai_tools.toolkits.notebook.set_cursor_in_ynotebook")
    def test_safe_set_cursor_with_stop(self, mock_set_cursor):
        """Test cursor setting with stop position."""
        mock_ynotebook = Mock()
        mock_old_ = Mock()

        _safe_set_cursor(mock_ynotebook, mock_old_, 5, 10)

        mock_set_cursor.assert_called_once_with(mock_ynotebook, mock_old_, 5, 10)

    @patch("jupyter_ai_tools.toolkits.notebook.set_cursor_in_ynotebook")
    def test_safe_set_cursor_handles_exception(self, mock_set_cursor):
        """Test that cursor setting exceptions are handled gracefully."""
        mock_set_cursor.side_effect = Exception("Cursor error")
        mock_ynotebook = Mock()
        mock_old_ = Mock()

        # Should not raise exception
        _safe_set_cursor(mock_ynotebook, mock_old_, 5)

        mock_set_cursor.assert_called_once()


class TestIntegration:
    """Integration tests for the complete functionality."""

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook.set_cursor_in_ynotebook")
    @patch("jupyter_ai_tools.toolkits.notebook.asyncio.sleep")
    async def test_full_workflow_simple_change(self, mock_sleep, mock_set_cursor):
        """Test a complete workflow with simple text change."""
        mock_ynotebook = Mock()
        mock_ycell = MagicMock()
        mock_source = MagicMock()

        # Set up mocks
        mock_ycell.to_py.return_value = {"source": "hello"}
        mock_ycell.__getitem__.return_value = mock_source
        mock_source.insert = Mock()
        mock_source.__delitem__ = Mock()

        # This should result in a replace operation: "hello" -> "world"
        result = await write_to_cell_collaboratively(
            mock_ynotebook, mock_ycell, "world", typing_speed=0.0
        )

        assert result is True
        # Should have called insert operations for typing simulation
        mock_source.insert.assert_called()

    @pytest.mark.asyncio
    @patch("jupyter_ai_tools.toolkits.notebook.set_cursor_in_ynotebook")
    @patch("jupyter_ai_tools.toolkits.notebook.asyncio.sleep")
    async def test_custom_typing_speed(self, mock_sleep, mock_set_cursor):
        """Test that custom typing speed is respected."""
        mock_ynotebook = Mock()
        mock_ycell = MagicMock()
        mock_source = MagicMock()

        mock_ycell.to_py.return_value = {"source": ""}
        mock_ycell.__getitem__.return_value = mock_source
        mock_source.insert = Mock()

        await write_to_cell_collaboratively(mock_ynotebook, mock_ycell, "test", typing_speed=0.5)

        # Should have called sleep with custom timing
        mock_sleep.assert_called()
        # At least one call should use our custom typing speed
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert 0.5 in sleep_calls


if __name__ == "__main__":
    pytest.main([__file__])
