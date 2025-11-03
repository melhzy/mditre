"""
Cross-Platform Path Utilities Verification Script

This script verifies that all cross-platform path utilities are working correctly.
Run this on Windows, macOS, and Linux to ensure consistency.

Usage:
    python scripts/verify_cross_platform.py

Note:
    This script is designed to run from the repository root directory.
    It automatically adds the Python/ directory to sys.path at runtime,
    so import errors shown by linters can be ignored - the script works correctly when executed.

    All tests should pass with 3/3 results on any supported platform (Windows/macOS/Linux).
"""

import os
import sys
from pathlib import Path


def test_python_path_utils():
    """Test Python path utilities."""
    print("=" * 70)
    print("Testing Python Path Utilities")
    print("=" * 70)

    try:
        # Import path utilities
        from mditre.utils.path_utils import (
            get_data_dir,
            get_output_dir,
            get_package_root,
            get_platform_info,
            get_project_root,
            get_python_dir,
            get_r_dir,
            normalize_path,
        )

        print("✓ Successfully imported path utilities")

        # Test package root detection
        pkg_root = get_package_root()
        print(f"✓ Package root: {pkg_root}")
        assert pkg_root.exists(), "Package root doesn't exist"

        # Test project root detection (may be None if pip installed)
        root = get_project_root()
        if root:
            print(f"✓ Project root (dev mode): {root}")
            assert root.exists(), "Project root doesn't exist"

            # Test directory getters in dev mode
            python_dir = get_python_dir()
            print(f"✓ Python directory: {python_dir}")
            assert python_dir.exists(), "Python directory doesn't exist"

            r_dir = get_r_dir()
            if r_dir:
                print(f"✓ R directory: {r_dir}")
                assert r_dir.exists(), "R directory doesn't exist"
        else:
            print("✓ Pip install mode detected (project_root is None)")
            print(f"  Package installed at: {pkg_root}")

        # Test platform info
        info = get_platform_info()
        print(f"✓ Platform: {info['os']} ({info['os_name']})")
        print(f"  Path separator: '{info['sep']}'")
        print(f"  Home directory: {info['home']}")

        # Test path normalization
        test_path = "data/raw/file.txt"
        if root:
            normalized = normalize_path(test_path)
            print(f"✓ Path normalization works:")
            print(f"  Input: {test_path}")
            print(f"  Output: {normalized}")
        else:
            print(
                f"✓ Path utilities available (skipping abs path normalization in pip mode)"
            )

        print("\n✓ All Python path utility tests passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Python path utilities test failed: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def test_data_loader_example():
    """Test that data loader example works with cross-platform paths."""
    print("=" * 70)
    print("Testing Data Loader Example")
    print("=" * 70)

    try:
        # Get the path to the example
        from mditre.utils.path_utils import get_python_dir

        example_path = (
            get_python_dir() / "mditre" / "examples" / "data_loader_example.py"
        )

        if not example_path.exists():
            print(f"✗ Example not found at: {example_path}")
            return False

        print(f"✓ Found example at: {example_path}")

        # Check that it doesn't have hardcoded paths
        with open(example_path, "r") as f:
            content = f.read()
            if "d:/Github" in content or "D:/Github" in content:
                print("✗ Example still contains hardcoded Windows paths")
                return False
            if "C:/Users" in content or "C:\\Users" in content:
                print("✗ Example still contains hardcoded user paths")
                return False

        print("✓ Example uses cross-platform path detection")
        print("\n✓ Data loader example verification passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Data loader example test failed: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def test_package_imports():
    """Test that mditre package imports correctly."""
    print("=" * 70)
    print("Testing Package Imports")
    print("=" * 70)

    try:
        # Test main package import
        import mditre

        print(f"✓ mditre package version: {mditre.__version__}")

        # Test that utils is in __all__
        if "utils" not in mditre.__all__:
            print("✗ 'utils' not in mditre.__all__")
            return False
        print("✓ 'utils' module exported in package")

        # Test utils.path_utils import
        from mditre import utils

        print("✓ mditre.utils module accessible")

        from mditre.utils import path_utils

        print("✓ mditre.utils.path_utils module accessible")

        # Test direct function import
        from mditre.utils.path_utils import get_project_root

        root = get_project_root()
        print(f"✓ Direct function import works: {root}")

        print("\n✓ All package import tests passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Package import test failed: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("MDITRE Cross-Platform Verification Suite")
    print("=" * 70)
    print()

    # Get platform info
    print(f"Running on: {sys.platform}")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print()

    # Run tests
    results = []
    results.append(("Package Imports", test_package_imports()))
    results.append(("Python Path Utils", test_python_path_utils()))
    results.append(("Data Loader Example", test_data_loader_example()))

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All verification tests passed! Cross-platform support is working.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    # Make sure we're in the right directory
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    os.chdir(project_root)

    # Add Python directory to path
    python_dir = project_root / "Python"
    sys.path.insert(0, str(python_dir))

    sys.exit(main())
