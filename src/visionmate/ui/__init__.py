"""UI components for VisionMate Desktop application."""

from visionmate.ui.device import DeviceControlsWidget
from visionmate.ui.input import InputModeWidget
from visionmate.ui.main import MainWindow
from visionmate.ui.preview import VideoPreviewWidget
from visionmate.ui.waveform import AudioWaveformWidget

__all__ = [
    "MainWindow",
    "VideoPreviewWidget",
    "AudioWaveformWidget",
    "DeviceControlsWidget",
    "InputModeWidget",
]
