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
    
    Note: This function does NOT change the working directory to avoid
    unexpected side effects in multi-threaded applications. Users should
    ensure their relative file paths are project-root-relative, or use
    get_project_root() to construct absolute paths.
    """
    # Get the directory containing this file (should be project root)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    
    # Convert to string for sys.path
    project_root_str = str(project_root)
    
    # Add to sys.path if not already present (check at beginning for priority)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def get_project_root():
    """
    Get the project root directory as a Path object.
    
    Use this to construct absolute paths for data files:
        from runtime_path_resolver import get_project_root
        data_file = get_project_root() / "data" / "file.csv"
    
    Returns:
        Path: The project root directory
    """
    return Path(__file__).resolve().parent


# Auto-execute on import
resolve_runtime_paths()
