"""
Dialogs for Visionmate desktop application.

This module provides various dialog windows including About dialog,
Settings dialog, Log Console dialog, and other modal dialogs.
"""

from visionmate.desktop.dialogs.about import AboutDialog
from visionmate.desktop.dialogs.console import LogConsoleDialog
from visionmate.desktop.dialogs.selector import WindowSelectorDialog
from visionmate.desktop.dialogs.settings import SettingsDialog

__all__ = ["AboutDialog", "LogConsoleDialog", "WindowSelectorDialog", "SettingsDialog"]
