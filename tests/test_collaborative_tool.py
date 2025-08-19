import inspect
from unittest.mock import MagicMock, patch

import pytest

from jupyter_ai_tools.utils import collaborative_tool


class TestCollaborativeTool:
    """Test suite for the collaborative_tool decorator."""

    @pytest.fixture
    def mock_user_dict(self):
        """Sample user dictionary for testing."""
        return {
            "name": "TestUser",
            "color": "var(--jp-collaborator-color1)",
            "display_name": "Test User",
            "avatar": "/test/avatar.svg"
        }

    @pytest.fixture
    def mock_global_awareness(self):
        """Mock global awareness object."""
        awareness = MagicMock()
        awareness.set_local_state = MagicMock()
        return awareness

    @pytest.fixture
    def mock_ydoc(self):
        """Mock YDoc object."""
        ydoc = MagicMock()
        ydoc.awareness = MagicMock()
        ydoc.awareness.set_local_state_field = MagicMock()
        return ydoc

    @pytest.mark.asyncio
    async def test_user_with_notebook_file_sets_awareness(self, mock_user_dict, mock_global_awareness, mock_ydoc):
        """Test that decorator sets both global and notebook awareness for .ipynb files."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(file_path: str, content: str):
            return f"processed {file_path}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=mock_global_awareness), \
             patch('jupyter_ai_tools.utils.get_file_id', return_value="test-file-id"), \
             patch('jupyter_ai_tools.utils.get_jupyter_ydoc', return_value=mock_ydoc):
            
            result = await test_func("test_notebook.ipynb", "test content")
            
            # Verify function executed
            assert result == "processed test_notebook.ipynb"
            
            # Verify global awareness was set
            mock_global_awareness.set_local_state.assert_called_once_with({
                "user": mock_user_dict,
                "current": "test_notebook.ipynb",
                "documents": ["test_notebook.ipynb"]
            })
            
            # Verify notebook awareness was set
            mock_ydoc.awareness.set_local_state_field.assert_called_once_with("user", mock_user_dict)

    @pytest.mark.asyncio
    async def test_user_with_non_notebook_file_only_sets_global_awareness(self, mock_user_dict, mock_global_awareness):
        """Test that decorator only sets global awareness for non-.ipynb files."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(file_path: str, content: str):
            return f"processed {file_path}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=mock_global_awareness), \
             patch('jupyter_ai_tools.utils.get_file_id') as mock_file_id, \
             patch('jupyter_ai_tools.utils.get_jupyter_ydoc') as mock_ydoc:
            
            result = await test_func("test_file.py", "test content")
            
            # Verify function executed
            assert result == "processed test_file.py"
            
            # Verify global awareness was set
            mock_global_awareness.set_local_state.assert_called_once_with({
                "user": mock_user_dict,
                "current": "test_file.py",
                "documents": ["test_file.py"]
            })
            
            # Verify notebook-specific functions were not called
            mock_file_id.assert_not_called()
            mock_ydoc.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_path_detection_from_kwargs(self, mock_user_dict, mock_global_awareness):
        """Test that file_path is correctly detected from kwargs."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(content: str, file_path: str):
            return f"processed {file_path}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=mock_global_awareness):
            
            result = await test_func(content="test", file_path="test.py")
            
            # Verify function executed
            assert result == "processed test.py"
            
            # Verify global awareness was set with correct file_path
            mock_global_awareness.set_local_state.assert_called_once_with({
                "user": mock_user_dict,
                "current": "test.py",
                "documents": ["test.py"]
            })

    @pytest.mark.asyncio
    async def test_file_path_detection_from_positional_args(self, mock_user_dict, mock_global_awareness):
        """Test that file_path is correctly detected from positional arguments."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(file_path: str, content: str):
            return f"processed {file_path}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=mock_global_awareness):
            
            result = await test_func("test.py", "test content")
            
            # Verify function executed
            assert result == "processed test.py"
            
            # Verify global awareness was set with correct file_path
            mock_global_awareness.set_local_state.assert_called_once_with({
                "user": mock_user_dict,
                "current": "test.py",
                "documents": ["test.py"]
            })

    @pytest.mark.asyncio
    async def test_function_without_file_path_parameter(self, mock_user_dict, mock_global_awareness):
        """Test that decorator works with functions that don't have file_path parameter."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(message: str):
            return f"processed {message}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=mock_global_awareness):
            
            result = await test_func("hello world")
            
            # Verify function executed
            assert result == "processed hello world"
            
            # Verify global awareness was set with empty file_path
            mock_global_awareness.set_local_state.assert_called_once_with({
                "user": mock_user_dict,
                "current": "",
                "documents": []
            })

    @pytest.mark.asyncio
    async def test_notebook_awareness_error_handling(self, mock_user_dict, mock_global_awareness):
        """Test that notebook awareness errors don't break function execution."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(file_path: str):
            return f"processed {file_path}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=mock_global_awareness), \
             patch('jupyter_ai_tools.utils.get_file_id', side_effect=Exception("File ID error")), \
             patch('jupyter_ai_tools.utils.get_jupyter_ydoc') as mock_ydoc:
            
            # Function should still execute despite notebook awareness error
            result = await test_func("test.ipynb")
            
            # Verify function executed normally
            assert result == "processed test.ipynb"
            
            # Verify global awareness was still set
            mock_global_awareness.set_local_state.assert_called_once_with({
                "user": mock_user_dict,
                "current": "test.ipynb",
                "documents": ["test.ipynb"]
            })
            
            # Verify ydoc was not called due to error
            mock_ydoc.assert_not_called()

    @pytest.mark.asyncio
    async def test_global_awareness_error_handling(self, mock_user_dict):
        """Test that global awareness errors don't break function execution."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(file_path: str):
            return f"processed {file_path}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', side_effect=Exception("Global awareness error")):
            
            # Function should still execute despite global awareness error
            result = await test_func("test.py")
            
            # Verify function executed normally
            assert result == "processed test.py"

    @pytest.mark.asyncio
    async def test_file_path_detection_error_handling(self, mock_user_dict, mock_global_awareness):
        """Test that file_path detection errors don't break function execution."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(weird_param: str):
            return f"processed {weird_param}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=mock_global_awareness):
            
            # Function should execute even with file_path detection issues
            result = await test_func("test data")
            
            # Verify function executed normally
            assert result == "processed test data"
            
            # Verify global awareness was set with empty file_path
            mock_global_awareness.set_local_state.assert_called_once_with({
                "user": mock_user_dict,
                "current": "",
                "documents": []
            })

    @pytest.mark.asyncio
    async def test_ydoc_none_handling(self, mock_user_dict, mock_global_awareness):
        """Test that None ydoc is handled gracefully."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(file_path: str):
            return f"processed {file_path}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=mock_global_awareness), \
             patch('jupyter_ai_tools.utils.get_file_id', return_value="test-file-id"), \
             patch('jupyter_ai_tools.utils.get_jupyter_ydoc', return_value=None):
            
            result = await test_func("test.ipynb")
            
            # Verify function executed normally
            assert result == "processed test.ipynb"
            
            # Verify global awareness was still set
            mock_global_awareness.set_local_state.assert_called_once_with({
                "user": mock_user_dict,
                "current": "test.ipynb",
                "documents": ["test.ipynb"]
            })

    @pytest.mark.asyncio
    async def test_global_awareness_none_handling(self, mock_user_dict):
        """Test that None global awareness is handled gracefully."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(file_path: str):
            return f"processed {file_path}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=None):
            
            # Function should execute even with None global awareness
            result = await test_func("test.py")
            
            # Verify function executed normally
            assert result == "processed test.py"

    @pytest.mark.asyncio
    async def test_function_signature_preservation(self, mock_user_dict):
        """Test that decorator preserves function signature and metadata."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(file_path: str, content: str, optional_param: str = "default"):
            """Test function docstring."""
            return f"processed {file_path} with {content} and {optional_param}"

        # Check that function metadata is preserved
        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring."
        
        # Check that function signature is preserved
        sig = inspect.signature(test_func)
        params = list(sig.parameters.keys())
        assert params == ["file_path", "content", "optional_param"]
        
        # Check that default values are preserved
        assert sig.parameters["optional_param"].default == "default"

    @pytest.mark.asyncio
    async def test_function_exception_propagation(self, mock_user_dict):
        """Test that exceptions from the wrapped function are properly propagated."""
        
        @collaborative_tool(user=mock_user_dict)
        async def test_func(file_path: str):
            raise ValueError("Test error")

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=MagicMock()):
            
            # Verify that the original function's exception is propagated
            with pytest.raises(ValueError, match="Test error"):
                await test_func("test.py")

    @pytest.mark.asyncio
    async def test_complex_function_signature(self, mock_user_dict, mock_global_awareness):
        """Test decorator with complex function signatures."""
        
        @collaborative_tool(user=mock_user_dict)
        async def complex_func(arg1: str, file_path: str, *args, **kwargs):
            return f"processed {arg1}, {file_path}, {args}, {kwargs}"

        with patch('jupyter_ai_tools.utils.get_global_awareness', return_value=mock_global_awareness):
            
            result = await complex_func("first", "test.py", "extra1", "extra2", key1="value1", key2="value2")
            
            # Verify function executed with all parameters
            assert "processed first, test.py, ('extra1', 'extra2'), {'key1': 'value1', 'key2': 'value2'}" in result
            
            # Verify global awareness was set with correct file_path
            mock_global_awareness.set_local_state.assert_called_once_with({
                "user": mock_user_dict,
                "current": "test.py",
                "documents": ["test.py"]
            })

    def test_decorator_factory_pattern(self, mock_user_dict):
        """Test that collaborative_tool works as a decorator factory."""
        
        # Test that it returns a decorator function
        decorator = collaborative_tool(user=mock_user_dict)
        assert callable(decorator)
        
        # Test that the decorator returns a wrapper function
        async def test_func():
            return "test"
        
        wrapped_func = decorator(test_func)
        assert callable(wrapped_func)
        assert wrapped_func.__name__ == "test_func"