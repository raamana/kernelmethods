"""Runtime version shim for source trees and built distributions."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess


def _version_from_git():
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty", "--always"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "0+unknown"
    return result.stdout.strip()


try:
    __version__ = version("kernelmethods")
except PackageNotFoundError:
    __version__ = _version_from_git()
