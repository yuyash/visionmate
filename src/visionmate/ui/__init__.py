"""UI components for VisionMate Desktop application."""

from visionmate.ui.audio_waveform import AudioWaveformWidget
from visionmate.ui.device_controls import DeviceControlsWidget
from visionmate.ui.main_window import MainWindow
from visionmate.ui.video_preview import VideoPreviewWidget

__all__ = [
    "MainWindow",
    "VideoPreviewWidget",
    "AudioWaveformWidget",
    "DeviceControlsWidget",
]
