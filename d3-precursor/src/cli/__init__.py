"""
Command-line interface module for D3 Precursor.

This module provides the CLI functionality for the D3 Precursor tool.
"""

from .main import main
from .commands import (
    extract_command,
    generate_command,
    query_command,
    analyze_command
)

# Create a package-level export for the progress bar utilities
from .progress import (
    ProgressBar,
    spinner,
    create_progress_bar,
    track_progress
)

# Create a package-level export for the color utilities
from .colors import (
    ColorFormatter,
    success,
    error,
    warning,
    info,
    highlight,
    api_name,
    file_path,
    code,
    heading
)

__all__ = [
    'main',
    'extract_command',
    'generate_command',
    'query_command',
    'analyze_command',
    'ProgressBar',
    'spinner',
    'create_progress_bar',
    'track_progress',
    'ColorFormatter',
    'success',
    'error',
    'warning',
    'info',
    'highlight',
    'api_name',
    'file_path',
    'code',
    'heading'
] 