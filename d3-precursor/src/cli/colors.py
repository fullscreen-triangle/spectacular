"""
Colorized output utilities for CLI.

This module provides utilities for displaying colorized text in CLI applications.
"""

from typing import Dict, Any, Optional


class ColorCode:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class ColorFormatter:
    """Formatter for colorized output."""
    
    @staticmethod
    def colorize(text: str, color: str, bold: bool = False, underline: bool = False) -> str:
        """
        Colorize text with ANSI color codes.
        
        Args:
            text: The text to colorize
            color: The color to use
            bold: Whether to make the text bold
            underline: Whether to underline the text
            
        Returns:
            Colorized text
        """
        formatted = ""
        
        if bold:
            formatted += ColorCode.BOLD
        if underline:
            formatted += ColorCode.UNDERLINE
            
        formatted += color + str(text) + ColorCode.RESET
        return formatted
    
    @staticmethod
    def success(text: str) -> str:
        """Format text as a success message."""
        return ColorFormatter.colorize(text, ColorCode.GREEN, bold=True)
    
    @staticmethod
    def error(text: str) -> str:
        """Format text as an error message."""
        return ColorFormatter.colorize(text, ColorCode.RED, bold=True)
    
    @staticmethod
    def warning(text: str) -> str:
        """Format text as a warning message."""
        return ColorFormatter.colorize(text, ColorCode.YELLOW, bold=True)
    
    @staticmethod
    def info(text: str) -> str:
        """Format text as an info message."""
        return ColorFormatter.colorize(text, ColorCode.BLUE)
    
    @staticmethod
    def highlight(text: str) -> str:
        """Format text as highlighted."""
        return ColorFormatter.colorize(text, ColorCode.CYAN, bold=True)
    
    @staticmethod
    def api_name(text: str) -> str:
        """Format text as an API name."""
        return ColorFormatter.colorize(text, ColorCode.MAGENTA, bold=True)
    
    @staticmethod
    def file_path(text: str) -> str:
        """Format text as a file path."""
        return ColorFormatter.colorize(text, ColorCode.BRIGHT_BLACK, underline=True)
    
    @staticmethod
    def code(text: str) -> str:
        """Format text as code."""
        return ColorFormatter.colorize(text, ColorCode.BRIGHT_WHITE, bold=True)
    
    @staticmethod
    def heading(text: str) -> str:
        """Format text as a heading."""
        return ColorFormatter.colorize(text, ColorCode.BRIGHT_CYAN, bold=True)


# Create shorthand functions for easy use
def success(text: str) -> str:
    """Format text as a success message."""
    return ColorFormatter.success(text)


def error(text: str) -> str:
    """Format text as an error message."""
    return ColorFormatter.error(text)


def warning(text: str) -> str:
    """Format text as a warning message."""
    return ColorFormatter.warning(text)


def info(text: str) -> str:
    """Format text as an info message."""
    return ColorFormatter.info(text)


def highlight(text: str) -> str:
    """Format text as highlighted."""
    return ColorFormatter.highlight(text)


def api_name(text: str) -> str:
    """Format text as an API name."""
    return ColorFormatter.api_name(text)


def file_path(text: str) -> str:
    """Format text as a file path."""
    return ColorFormatter.file_path(text)


def code(text: str) -> str:
    """Format text as code."""
    return ColorFormatter.code(text)


def heading(text: str) -> str:
    """Format text as a heading."""
    return ColorFormatter.heading(text) 