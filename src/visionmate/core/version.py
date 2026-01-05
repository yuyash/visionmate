"""Version information for VisionMate."""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Get version from installed package metadata.

    This works in both development and production environments:
    - Development: Reads from editable install metadata
    - Production: Reads from installed package metadata

    Returns:
        Version string (e.g., "0.3.0") or "unknown" if not found
    """
    try:
        # Try to get version from installed package metadata
        return version("visionmate")
    except PackageNotFoundError:
        # Fallback: try to read from pyproject.toml (development only)
        try:
            import tomllib
            from pathlib import Path

            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            pyproject_path = project_root / "pyproject.toml"

            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                return data.get("project", {}).get("version", "unknown")
            else:
                return "unknown"

        except Exception:
            return "unknown"


__version__ = get_version()
