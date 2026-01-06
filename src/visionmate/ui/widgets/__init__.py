"""Widget components for VisionMate Desktop application."""

from visionmate.ui.widgets.device import DeviceControlsWidget
from visionmate.ui.widgets.fps import FPSControlWidget
from visionmate.ui.widgets.input import InputModeWidget
from visionmate.ui.widgets.method import CaptureMethodWidget
from visionmate.ui.widgets.preview import VideoPreviewWidget
from visionmate.ui.widgets.session import SessionControlWidget
from visionmate.ui.widgets.waveform import AudioWaveformWidget

__all__ = [
    "VideoPreviewWidget",
    "AudioWaveformWidget",
    "DeviceControlsWidget",
    "InputModeWidget",
    "FPSControlWidget",
    "CaptureMethodWidget",
    "SessionControlWidget",
]
