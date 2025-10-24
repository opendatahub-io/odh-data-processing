"""
Simple notebook validation for ODH Data Processing.
Focuses on essential checks only.
"""

from pathlib import Path
from typing import List
import nbformat
from nbformat.validator import ValidationError


class NotebookValidator:
    """Simple notebook validation for essential checks only."""
    
    def __init__(self, notebook_path: str):
        """Initialize validator with notebook path."""
        self.notebook_path = Path(notebook_path)
        self.notebook = None
    
    def load_notebook(self) -> nbformat.NotebookNode:
        """Load and validate basic notebook format."""
        try:
            self.notebook = nbformat.read(self.notebook_path, as_version=nbformat.NO_CONVERT)
            nbformat.validate(self.notebook)
            return self.notebook
        except ValidationError as e:
            raise ValueError(f"Invalid notebook format: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read notebook {self.notebook_path}: {e}")
    
    def validate_no_outputs(self) -> List[str]:
        """Validate that code cells have no outputs."""
        errors = []
        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'code' and cell.outputs:
                errors.append(f"Cell {i}: Code cell contains outputs (should be cleared)")
        return errors
    
    def validate_no_execution_counts(self) -> List[str]:
        """Validate that code cells have no execution counts."""
        errors = []
        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'code' and cell.execution_count:
                errors.append(f"Cell {i}: Code cell has execution_count {cell.execution_count} (should be null)")
        return errors
    
    def validate_parameters_cell_exists(self) -> List[str]:
        """Validate that at least one cell is tagged with 'parameters'."""
        errors = []
        has_parameters_cell = False
        
        for cell in self.notebook.cells:
            if cell.cell_type == 'code':
                tags = cell.metadata.get('tags', [])
                if 'parameters' in tags:
                    has_parameters_cell = True
                    break
        
        if not has_parameters_cell:
            errors.append("No cell tagged with 'parameters' found (required for parameterized execution)")
        
        return errors