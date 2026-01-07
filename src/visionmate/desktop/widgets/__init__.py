"""
Desktop UI widgets for Visionmate application.

This package provides custom widgets for the control panel including
input mode selection, video input configuration, and audio input configuration.
"""

from visionmate.desktop.widgets.control import ControlContainer
from visionmate.desktop.widgets.input import VideoInputWidget
from visionmate.desktop.widgets.mode import InputModeWidget
from visionmate.desktop.widgets.preview import PreviewContainer, PreviewLayout
from visionmate.desktop.widgets.video import VideoPreviewWidget

__all__ = [
    "ControlContainer",
    "InputModeWidget",
    "VideoInputWidget",
    "VideoPreviewWidget",
    "PreviewContainer",
    "PreviewLayout",
]
