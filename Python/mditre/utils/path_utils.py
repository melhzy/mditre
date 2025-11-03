"""
Cross-platform path utilities for MDITRE.

Provides platform-independent path handling that works seamlessly
across Windows, macOS, and Linux/Ubuntu. Works both in development
mode and when installed via pip.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union


def get_package_root() -> Path:
    """
    Get the MDITRE package installation directory.

    This returns the location where mditre package is installed,
    which could be in site-packages (pip install) or a development
    directory (pip install -e .).

    Returns:
        Path: Absolute path to the mditre package directory

    Raises:
        RuntimeError: If package location cannot be determined

    Examples:
        >>> root = get_package_root()
        >>> # Pip install: Path('/usr/local/lib/python3.12/site-packages/mditre')
        >>> # Dev install: Path('/home/username/mditre/Python/mditre')
    """
    try:
        # Get the directory containing the mditre package
        return Path(__file__).resolve().parent.parent
    except (NameError, AttributeError) as e:
        raise RuntimeError(
            "Cannot determine MDITRE package location. "
            "MDITRE may not be properly installed. "
            "Please reinstall using: pip install mditre\n"
            f"Error details: {e}"
        ) from e


def get_project_root() -> Optional[Path]:
    """
    Get the MDITRE project root directory (development mode only).

    This only works in development installations (pip install -e .)
    where the full repository structure exists. Returns None when
    installed via pip in site-packages.

    Returns:
        Path or None: Project root if in development mode, None otherwise

    Examples:
        >>> root = get_project_root()
        >>> if root:
        >>>     print(f"Development mode: {root}")
        >>> else:
        >>>     print("Installed via pip")
    """
    # Start from package location
    package_root = get_package_root()

    # Check if we're in development mode (package_root ends with Python/mditre)
    # Development: /path/to/mditre/Python/mditre
    # Pip install: /path/to/site-packages/mditre

    if package_root.parent.name == "Python":
        # We're in development mode
        return package_root.parent.parent
    else:
        # We're in site-packages (pip installed)
        return None


def get_python_dir() -> Path:
    """
    Get the Python implementation directory (development mode only).

    Returns the Python/ directory in development mode, or the package
    directory when installed via pip.

    Returns:
        Path: Python directory path
    """
    project_root = get_project_root()
    if project_root:
        return project_root / "Python"
    else:
        # In pip install mode, return the package root
        return get_package_root().parent


def get_r_dir() -> Optional[Path]:
    """
    Get the R implementation directory (development mode only).

    Only available in development installations. Returns None when
    installed via pip.

    Returns:
        Path or None: R directory if in development mode, None otherwise
    """
    project_root = get_project_root()
    if project_root:
        r_dir = project_root / "R"
        if r_dir.exists():
            return r_dir
    return None
    return get_project_root() / "R"


def get_data_dir(
    subdirectory: Optional[str] = None, base_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Get the data directory path.

    In development mode, returns mditre_root/data/
    When installed via pip, returns the specified base_path or current directory.

    Args:
        subdirectory: Optional subdirectory within data/ (e.g., 'raw', 'processed')
        base_path: Base path for data directory (defaults to project root or cwd)

    Returns:
        Path: Absolute path to data directory or subdirectory

    Examples:
        >>> # Development mode
        >>> get_data_dir()  # Returns mditre/data/
        >>> get_data_dir('raw')  # Returns mditre/data/raw/
        >>>
        >>> # Pip installed mode - specify your data location
        >>> get_data_dir(base_path='/my/data/location')
    """
    if base_path:
        data_dir = Path(base_path)
    else:
        project_root = get_project_root()
        if project_root:
            data_dir = project_root / "data"
        else:
            # Pip installed - use current working directory
            data_dir = Path.cwd() / "data"

    if subdirectory:
        data_dir = data_dir / subdirectory
    return data_dir


def get_output_dir(create: bool = True, base_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Get the output directory for results.

    In development mode, returns mditre_root/outputs/
    When installed via pip, returns the specified base_path or current directory.

    Args:
        create: If True, create directory if it doesn't exist
        base_path: Base path for output directory (defaults to project root or cwd)

    Returns:
        Path: Absolute path to output directory

    Examples:
        >>> # Development mode
        >>> get_output_dir()  # Returns mditre/outputs/
        >>>
        >>> # Pip installed mode - specify your output location
        >>> get_output_dir(base_path='/my/output/location')
    """
    if base_path:
        output_dir = Path(base_path)
    else:
        project_root = get_project_root()
        if project_root:
            output_dir = project_root / "outputs"
        else:
            # Pip installed - use current working directory
            output_dir = Path.cwd() / "outputs"

    if create and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_path(path: Union[str, Path], relative_to: Optional[Union[str, Path]] = None) -> Path:
    """
    Normalize a path to be platform-independent.

    Args:
        path: Path to normalize (can be string or Path object)
        relative_to: Optional base path to make relative to

    Returns:
        Path: Normalized Path object

    Examples:
        >>> # Convert Windows-style to platform-independent
        >>> normalize_path("C:\\Users\\data\\file.txt")
        Path('C:/Users/data/file.txt')  # On Windows

        >>> # Works on all platforms
        >>> normalize_path("~/data/file.txt")
        Path('/home/username/data/file.txt')  # On Linux
    """
    path = Path(path).expanduser().resolve()

    if relative_to:
        relative_to = Path(relative_to).expanduser().resolve()
        try:
            path = path.relative_to(relative_to)
        except ValueError:
            # Paths are not relative to each other
            pass

    return path


def ensure_dir_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path to ensure exists

    Returns:
        Path: Absolute path to the directory

    Raises:
        RuntimeError: If directory cannot be created due to permissions or other errors

    Examples:
        >>> ensure_dir_exists("outputs/models")
        Path('/.../mditre/outputs/models')
    """
    try:
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    except PermissionError as e:
        raise RuntimeError(
            f"Permission denied: Cannot create directory '{directory}'. "
            "Please check directory permissions or choose a different location.\n"
            f"Error details: {e}"
        ) from e
    except OSError as e:
        raise RuntimeError(
            f"OS error: Cannot create directory '{directory}'. " f"Error details: {e}"
        ) from e


def join_paths(*parts: Union[str, Path]) -> Path:
    """
    Join path parts in a platform-independent way.

    Args:
        *parts: Path components to join

    Returns:
        Path: Combined path

    Examples:
        >>> join_paths("data", "raw", "file.csv")
        Path('data/raw/file.csv')  # Platform-independent
    """
    if not parts:
        return Path()

    base = Path(parts[0])
    for part in parts[1:]:
        base = base / part
    return base


def get_platform_info() -> dict:
    """
    Get current platform information.

    Returns:
        dict: Platform details including OS, separator, home directory

    Examples:
        >>> info = get_platform_info()
        >>> # On Windows: {'os': 'Windows', 'sep': '\\', ...}
        >>> # On macOS:   {'os': 'Darwin', 'sep': '/', ...}
        >>> # On Linux:   {'os': 'Linux', 'sep': '/', ...}
    """
    return {
        "os": sys.platform,
        "os_name": os.name,
        "sep": os.sep,
        "pathsep": os.pathsep,
        "home": str(Path.home()),
        "cwd": str(Path.cwd()),
    }


def to_unix_path(path: Union[str, Path]) -> str:
    """
    Convert path to Unix-style (forward slashes).

    Useful for cross-platform config files or URLs.

    Args:
        path: Path to convert

    Returns:
        str: Path with forward slashes

    Examples:
        >>> to_unix_path("C:\\Users\\data\\file.txt")
        'C:/Users/data/file.txt'
    """
    return str(Path(path)).replace("\\", "/")


def to_platform_path(path: Union[str, Path]) -> str:
    """
    Convert path to platform-specific format.

    Args:
        path: Path to convert

    Returns:
        str: Path with platform-appropriate separators

    Examples:
        >>> # On Windows
        >>> to_platform_path("data/raw/file.txt")
        'data\\raw\\file.txt'

        >>> # On Unix
        >>> to_platform_path("data/raw/file.txt")
        'data/raw/file.txt'
    """
    return str(Path(path))


# Example usage and testing
if __name__ == "__main__":
    print("MDITRE Cross-Platform Path Utilities")
    print("=" * 70)

    # Show platform info
    info = get_platform_info()
    print(f"\nPlatform: {info['os']}")
    print(f"OS Name: {info['os_name']}")
    print(f"Path Separator: {info['sep']}")
    print(f"Home Directory: {info['home']}")
    print(f"Current Directory: {info['cwd']}")

    # Detect installation mode
    print("\n" + "=" * 70)
    project_root = get_project_root()
    if project_root:
        print("Installation Mode: Development (editable install)")
        print(f"Project Root: {project_root}")
        print(f"Python Directory: {get_python_dir()}")
        r_dir = get_r_dir()
        if r_dir:
            print(f"R Directory: {r_dir}")
        print(f"Data Directory: {get_data_dir()}")
        print(f"Output Directory: {get_output_dir(create=False)}")
    else:
        print("Installation Mode: Pip installed (site-packages)")
        print(f"Package Location: {get_package_root()}")
        print(f"Data Directory: {get_data_dir()} (defaults to cwd/data)")
        print(f"Output Directory: {get_output_dir(create=False)} (defaults to cwd/outputs)")
        print("\nNote: For pip-installed mode, specify data/output paths in your code:")
        print("  get_data_dir(base_path='/path/to/your/data')")
        print("  get_output_dir(base_path='/path/to/your/output')")

    # Show path conversions
    print("\n" + "=" * 70)
    print("Path Conversions:")
    test_path = "data/raw/file.txt"
    print(f"  Original: {test_path}")
    print(f"  Platform: {to_platform_path(test_path)}")
    print(f"  Unix: {to_unix_path(test_path)}")
    if project_root:
        print(f"  Normalized (abs): {normalize_path(test_path)}")
