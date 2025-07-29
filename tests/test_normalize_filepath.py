import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jupyter_ai_tools.utils import normalize_filepath


class TestNormalizeFilepath:
    """Test suite for the normalize_filepath function."""
    
    @pytest.fixture
    def mock_serverapp(self):
        """Fixture that provides a mocked serverapp with test root directory."""
        with patch('jupyter_ai_tools.utils.get_serverapp') as mock_serverapp:
            mock_app = MagicMock()
            mock_app.root_dir = "/test/root"
            mock_serverapp.return_value = mock_app
            yield mock_serverapp

    @pytest.mark.parametrize("test_path,expected_decoded", [
        ("notebooks/my%20notebook.ipynb", "notebooks/my notebook.ipynb"),
        ("relative/file.ipynb", "relative/file.ipynb"),
        ("folder%20with%20spaces/file%20name.ipynb", "folder with spaces/file name.ipynb"),
        ("./current/file.ipynb", "current/file.ipynb"),
        ("../parent/file.ipynb", "../parent/file.ipynb"),
        ("path%2Fwith%2Fslashes/file%2Bwith%2Bplus.ipynb", "path/with/slashes/file+with+plus.ipynb"),
        ("path/with/special%21chars%40symbols.ipynb", "path/with/special!chars@symbols.ipynb"),
        ("path//with//double//slashes.ipynb", "path/with/double/slashes.ipynb"),
        ("path/with/unicode%C3%A9chars.ipynb", "path/with/unicodeéchars.ipynb"),
        ("very/deeply/nested/path/structure/with/many/levels/file.ipynb", "very/deeply/nested/path/structure/with/many/levels/file.ipynb"),
    ])
    def test_relative_path_resolution(self, mock_serverapp, test_path, expected_decoded):
        """Test that relative paths are properly decoded and resolved against server root."""
        result = normalize_filepath(test_path)
        expected = str(Path(f"/test/root/{expected_decoded}").resolve())
        assert result == expected

    @pytest.mark.parametrize("test_path,expected_decoded", [
        ("/absolute/path/file.ipynb", "/absolute/path/file.ipynb"),
        ("/absolute/path%20with%20spaces/file.ipynb", "/absolute/path with spaces/file.ipynb"),
    ])
    def test_absolute_path_resolution(self, test_path, expected_decoded):
        """Test that absolute paths are normalized but not changed."""
        result = normalize_filepath(test_path)
        expected = str(Path(expected_decoded).resolve())
        assert result == expected

    def test_fallback_to_cwd_when_serverapp_fails(self):
        """Test that function falls back to current working directory when serverapp fails."""
        test_path = "relative/file.ipynb"
        
        with patch('jupyter_ai_tools.utils.get_serverapp', side_effect=Exception("ServerApp error")):
            result = normalize_filepath(test_path)
            expected = str(Path(os.getcwd(), "relative/file.ipynb").resolve())
            assert result == expected

    @pytest.mark.parametrize("invalid_path", [
        "",
        None,
        "   ",
        "\t\n",
    ])
    def test_invalid_path_raises_error(self, invalid_path):
        """Test that invalid paths raise ValueError."""
        with pytest.raises(ValueError, match="file_path cannot be empty"):
            normalize_filepath(invalid_path)

    def test_path_resolution_with_real_filesystem(self):
        """Test path resolution with real filesystem using temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file structure
            test_subdir = Path(temp_dir) / "test_subdir"
            test_subdir.mkdir()
            test_file = test_subdir / "test_file.ipynb"
            test_file.write_text('{"cells": []}')
            
            # Test relative path resolution
            relative_path = "test_subdir/test_file.ipynb"
            
            with patch('jupyter_ai_tools.utils.get_serverapp') as mock_serverapp:
                mock_app = MagicMock()
                mock_app.root_dir = temp_dir
                mock_serverapp.return_value = mock_app
                
                result = normalize_filepath(relative_path)
                
                # Should resolve to the actual file path
                expected = str(test_file.resolve())
                assert result == expected
                
                # Verify the resolved path actually exists
                assert Path(result).exists()

    @pytest.mark.parametrize("test_path", [
        "notebook.ipynb",
        "script.py",
        "data.csv",
        "image.png",
        "document.txt",
        "config.json",
        "style.css",
        "page.html"
    ])
    def test_various_file_extensions(self, mock_serverapp, test_path):
        """Test that function works with various file extensions."""
        result = normalize_filepath(test_path)
        expected = str(Path(f"/test/root/{test_path}").resolve())
        assert result == expected

    @pytest.mark.parametrize("root_dir", [
        "/home/user/notebooks",
        "/var/jupyter/work",
        "/tmp/jupyter_root",
        "/Users/username/Documents"
    ])
    def test_serverapp_with_different_root_dirs(self, root_dir):
        """Test that different server root directories are handled correctly."""
        test_path = "file.ipynb"
        
        with patch('jupyter_ai_tools.utils.get_serverapp') as mock_serverapp:
            mock_app = MagicMock()
            mock_app.root_dir = root_dir
            mock_serverapp.return_value = mock_app
            
            result = normalize_filepath(test_path)
            expected = str(Path(root_dir, test_path).resolve())
            assert result == expected