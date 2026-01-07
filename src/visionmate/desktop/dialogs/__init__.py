"""
Dialogs for Visionmate desktop application.

This module provides various dialog windows including About dialog,
Settings dialog, and other modal dialogs.
"""

from visionmate.desktop.dialogs.about import AboutDialog
from visionmate.desktop.dialogs.selector import WindowSelectorDialog

__all__ = ["AboutDialog", "WindowSelectorDialog"]
