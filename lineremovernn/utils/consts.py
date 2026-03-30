import importlib.metadata
import platform
import sys
from pathlib import Path

import distro

ROOT = Path(__file__).parent.parent
VERSION = importlib.metadata.version("lineremovernn")
PYTHON_VERSION = ".".join(map(str, sys.version_info[:3]))
if platform.system() == "Windows":
    OS = f"Windows {platform.release()}"
elif platform.system() == "Darwin":
    OS = f"macOS {platform.mac_ver()[0]}"
elif platform.system() == "Linux":
    OS = distro.name(pretty=True)
ARCH = " ".join(platform.architecture())
