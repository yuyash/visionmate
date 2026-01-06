"""
Desktop UI widgets for Visionmate application.

This package provides custom widgets for the control panel including
input mode selection, video input configuration, and audio input configuration.
"""

from visionmate.desktop.widgets.input_mode import InputModeWidget
from visionmate.desktop.widgets.video_input import VideoInputWidget

__all__ = [
    "InputModeWidget",
    "VideoInputWidget",
]
