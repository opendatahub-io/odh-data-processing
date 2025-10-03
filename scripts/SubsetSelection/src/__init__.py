"""SubsetSelection: Intelligent subset selection for large datasets."""

__version__ = "1.0.0"

# Import main classes when package is imported
try:
    from .api.simple import SubsetSelector
    from .core.config import ProcessingConfig
    __all__ = ["SubsetSelector", "ProcessingConfig"]
except ImportError:
    # Handle case where dependencies aren't installed yet
    __all__ = []