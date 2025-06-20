# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""Lawrence McDaniel https://lawrencemcdaniel.com."""
import importlib.util
import os
import re
from typing import Dict


MODULE_NAME = "models"
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, MODULE_NAME))

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


def load_version() -> Dict[str, str]:
    """Stringify the __version__ module."""
    version_file_path = os.path.join(PROJECT_ROOT, "__version__.py")
    spec = importlib.util.spec_from_file_location("__version__", version_file_path)
    if spec is None:
        raise ImportError(f"Could not find version file at {version_file_path}")
    version_module = importlib.util.module_from_spec(spec)
    if version_module is None:
        raise ImportError(f"Could not load version module from {version_file_path}")
    if spec.loader is None:
        raise ImportError(f"Could not load version module from {version_file_path} - no loader found")
    spec.loader.exec_module(version_module)
    return version_module.__dict__


VERSION = load_version()


def get_semantic_version() -> str:
    """
    Return the semantic version number.

    Example valid values of __version__.py are:
    0.1.17
    0.1.17-next.1
    0.1.17-next.2
    0.1.17-next.123456
    0.1.17-next-major.1
    0.1.17-next-major.2
    0.1.17-next-major.123456

    Note:
    - pypi does not allow semantic version numbers to contain a dash.
    - pypi does not allow semantic version numbers to contain a 'v' prefix.
    - pypi does not allow semantic version numbers to contain a 'next' suffix.
    """
    version = VERSION["__version__"]
    version = re.sub(r"-next\.\d+", "", version)
    return re.sub(r"-next-major\.\d+", "", version)
