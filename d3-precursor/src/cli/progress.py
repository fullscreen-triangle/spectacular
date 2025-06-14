"""
Progress bar utilities for CLI.

This module provides utilities for displaying progress bars in CLI applications.
"""

import sys
import time
from typing import Optional, Any, Iterator, Callable


class ProgressBar:
    """A simple progress bar for CLI applications."""
    
    def __init__(
        self, 
        total: int, 
        prefix: str = '', 
        suffix: str = '', 
        decimals: int = 1, 
        length: int = 50, 
        fill: str = '█', 
        empty: str = '░'
    ):
        """
        Initialize a progress bar.
        
        Args:
            total: Total iterations
            prefix: Prefix string
            suffix: Suffix string
            decimals: Positive number of decimals in percent complete
            length: Character length of bar
            fill: Bar fill character
            empty: Bar empty character
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.empty = empty
        self.iteration = 0
        self.start_time = time.time()
    
    def update(self, iteration: Optional[int] = None) -> None:
        """
        Update the progress bar.
        
        Args:
            iteration: The current iteration (if None, increments by 1)
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
            
        # Calculate percent
        percent = 100 * (self.iteration / float(self.total))
        
        # Calculate elapsed time and ETA
        elapsed = time.time() - self.start_time
        if self.iteration > 0:
            eta = elapsed * (self.total / self.iteration - 1)
            eta_str = f"ETA: {int(eta)}s"
        else:
            eta_str = "ETA: --"
            
        # Calculate filled length
        filled_length = int(self.length * self.iteration // self.total)
        
        # Create the bar string
        bar = self.fill * filled_length + self.empty * (self.length - filled_length)
        
        # Print the progress bar
        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent:.{self.decimals}f}% {self.suffix} {eta_str}')
        sys.stdout.flush()
        
        # Print a new line when complete
        if self.iteration >= self.total:
            total_time = time.time() - self.start_time
            sys.stdout.write(f' (Completed in {total_time:.2f}s)\n')
    
    def finish(self) -> None:
        """Finish the progress bar."""
        self.update(self.total)


def progress_iterator(iterable: Iterator[Any], description: str = 'Processing', **kwargs: Any) -> Iterator[Any]:
    """
    Wrap an iterable with a progress bar.
    
    Args:
        iterable: The iterable to wrap
        description: Description to show in the progress bar
        **kwargs: Additional arguments to pass to ProgressBar
        
    Returns:
        Iterator over the iterable with progress updates
    """
    items = list(iterable)
    total = len(items)
    progress = ProgressBar(total=total, prefix=description, **kwargs)
    
    for item in items:
        yield item
        progress.update()
    
    progress.finish()


class Spinner:
    """A simple spinner for CLI applications."""
    
    def __init__(self, message: str = 'Processing', symbols: str = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'):
        """
        Initialize a spinner.
        
        Args:
            message: The message to display
            symbols: The symbols to use for the spinner
        """
        self.message = message
        self.symbols = symbols
        self.current = 0
        self.running = False
        self.start_time = None
    
    def start(self) -> None:
        """Start the spinner."""
        self.running = True
        self.start_time = time.time()
        self._spin()
    
    def stop(self, message: Optional[str] = None) -> None:
        """
        Stop the spinner.
        
        Args:
            message: Optional message to replace the spinner
        """
        self.running = False
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if message:
            sys.stdout.write(f'\r{message} ({elapsed:.2f}s)\n')
        else:
            sys.stdout.write(f'\r{self.message} done! ({elapsed:.2f}s)\n')
        sys.stdout.flush()
    
    def _spin(self) -> None:
        """Internal method to update the spinner."""
        if not self.running:
            return
            
        symbol = self.symbols[self.current]
        self.current = (self.current + 1) % len(self.symbols)
        
        sys.stdout.write(f'\r{symbol} {self.message}')
        sys.stdout.flush()
        
        time.sleep(0.1)
        self._spin() if self.running else None


def with_spinner(func: Callable) -> Callable:
    """
    Decorator to add a spinner to a function.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with a spinner
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        message = kwargs.pop('spinner_message', 'Processing')
        spinner = Spinner(message=message)
        
        spinner.start()
        try:
            result = func(*args, **kwargs)
            spinner.stop()
            return result
        except Exception as e:
            spinner.stop(f"Error: {str(e)}")
            raise
    
    return wrapper 