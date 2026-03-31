import importlib.metadata
import platform
import sys
from pathlib import Path

import distro
from torch.cuda import is_available as torch_cuda_available

ROOT = Path(__file__).parent.parent
SAVED = ROOT / "saved"

DEFAULT_IAM = SAVED / "datasets/iam"
DEFAULT_PAGES = SAVED / "datasets/pages"
DEFAULT_MODELS = SAVED / "models"
DEFAULT_LOGS = SAVED / "logs"

VERSION = importlib.metadata.version("lineremovernn")
PYTHON_VERSION = ".".join(map(str, sys.version_info[:3]))
if platform.system() == "Windows":
    OS = f"Windows {platform.release()}"
elif platform.system() == "Darwin":
    OS = f"macOS {platform.mac_ver()[0]}"
elif platform.system() == "Linux":
    OS = distro.name(pretty=True)
ARCH = " ".join(platform.architecture())
DEVICE = "cuda" if torch_cuda_available() else "cpu"
