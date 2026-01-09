"""
Desktop UI widgets for Visionmate application.

This package provides custom widgets for the control panel including
input mode selection, video input configuration, and audio input configuration.
"""

from visionmate.desktop.widgets.action import ActionContainer
from visionmate.desktop.widgets.audio import AudioPreviewWidget
from visionmate.desktop.widgets.control import ControlContainer
from visionmate.desktop.widgets.input import AudioInputWidget, VideoInputWidget
from visionmate.desktop.widgets.mode import InputModeWidget
from visionmate.desktop.widgets.preview import PreviewContainer, PreviewLayout
from visionmate.desktop.widgets.request import RequestWidget
from visionmate.desktop.widgets.response import ResponseWidget
from visionmate.desktop.widgets.session import SessionControlWidget
from visionmate.desktop.widgets.video import VideoPreviewWidget

__all__ = [
    "ActionContainer",
    "AudioInputWidget",
    "AudioPreviewWidget",
    "ControlContainer",
    "InputModeWidget",
    "RequestWidget",
    "ResponseWidget",
    "SessionControlWidget",
    "VideoInputWidget",
    "VideoPreviewWidget",
    "PreviewContainer",
    "PreviewLayout",
]
