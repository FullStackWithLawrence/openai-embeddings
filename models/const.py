# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""Sales Support Model (hsr) for the LangChain project."""

import os
from pathlib import Path


MODULE_NAME = "models"
HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = str(Path(HERE).parent)
IS_USING_TFVARS = False
