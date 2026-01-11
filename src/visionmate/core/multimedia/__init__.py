"""Multimedia processing components for real-time VLM integration.

This module contains components for intelligent frame selection, multimedia
segment building, and temporal correlation of audio and video data.
"""

from visionmate.core.multimedia.buffer import SegmentBufferManager
from visionmate.core.multimedia.detector import AudioActivityDetector
from visionmate.core.multimedia.handlers import (
    ClientSideHandler,
    EventType,
    MultimediaEvent,
    ServerSideHandler,
)
from visionmate.core.multimedia.manager import MultimediaManager
from visionmate.core.multimedia.segment import MultimediaSegmentBuilder

__all__ = [
    "AudioActivityDetector",
    "MultimediaSegmentBuilder",
    "SegmentBufferManager",
    "ServerSideHandler",
    "ClientSideHandler",
    "MultimediaManager",
    "EventType",
    "MultimediaEvent",
]
