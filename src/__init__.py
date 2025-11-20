"""Rad Text Engine (RaTE) - AI-powered radiology report processing and analysis."""

__version__ = "1.0.0"
__author__ = "YalaLab"
__license__ = "MIT"

from . import core
from .cli import main as cli_main
from .qc_cli import main as qc_main
from .eval_cli import main as eval_main

__all__ = ["core", "cli_main", "qc_main", "eval_main", "__version__"]
