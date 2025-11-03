"""
MDITRE Utilities Module

Cross-platform utilities for MDITRE package.

Available submodules:
    - path_utils: Platform-independent path handling and project structure utilities

Example:
    from mditre.utils.path_utils import get_project_root, get_data_dir

    root = get_project_root()
    data = get_data_dir()
"""

from . import path_utils

__all__ = ["path_utils"]
