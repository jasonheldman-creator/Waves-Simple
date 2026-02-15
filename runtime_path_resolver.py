"""
runtime_path_resolver.py

Reliable runtime path resolver for Streamlit Cloud and local development.

This module ensures that the project root directory is always in sys.path,
regardless of how the application is deployed or started. This is critical
for Streamlit Cloud deployments where sys.path behavior differs from local
development environments.

Usage:
    import runtime_path_resolver  # Import at the top of any module that needs reliable imports

The import alone is sufficient - the module automatically configures sys.path on import.
"""

import sys
import os
from pathlib import Path


def resolve_runtime_paths():
    """
    Ensure project root is in sys.path for reliable imports across environments.
    
    This handles:
    - Streamlit Cloud deployment (where cwd may differ)
    - Local development (various working directories)
    - OS-specific path handling
    - Prevents duplicate path entries
    """
    # Get the directory containing this file (should be project root)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    
    # Convert to string for sys.path
    project_root_str = str(project_root)
    
    # Add to sys.path if not already present (check at beginning for priority)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # Also ensure current working directory is set correctly
    # This helps with relative file paths for data files
    try:
        if os.getcwd() != project_root_str:
            os.chdir(project_root_str)
    except Exception:
        # If we can't change directory, at least the sys.path is correct
        pass


# Auto-execute on import
resolve_runtime_paths()
