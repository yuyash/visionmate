"""Capture manager for managing devices and video sources.

This module provides the CaptureManager class that unifies device
management and video source management into a single cohesive interface.
"""

from __future__ import annotations

import logging
from typing import Optional

from visionmate.core.capture.device import DeviceManager
from visionmate.core.capture.source import VideoSourceManager
from visionmate.core.capture.stream import StreamManager
from visionmate.core.capture.video import VideoCaptureInterface
from visionmate.core.models import DeviceMetadata, VideoFrame

logger = logging.getLogger(__name__)


class CaptureManager:
    """Manages device enumeration and video source lifecycle.

    This class provides a unified interface for:
    - Device enumeration and metadata retrieval
    - Video source lifecycle management
    - Frame collection from multiple sources

    It encapsulates both DeviceManager and VideoSourceManager to provide
    a single point of access for all capture-related operations.
    """

    def __init__(self):
        """Initialize the CaptureManager."""
        self._device_manager = DeviceManager()
        self._video_source_manager = VideoSourceManager()
        self._audio_sources: dict[str, object] = {}  # Audio capture interfaces
        self._stream_manager = StreamManager(self)
        logger.info("CaptureManager initialized")

    # ========================================================================
    # Internal Access (for components that need direct DeviceManager access)
    # ========================================================================

    def get_device_manager(self) -> DeviceManager:
        """Get the internal DeviceManager instance.

        This is provided for components that need direct access to DeviceManager,
        such as video capture implementations.

        Returns:
            DeviceManager instance
        """
        return self._device_manager

    # ========================================================================
    # Device Management (delegated to DeviceManager)
    # ========================================================================

    def get_screens(self) -> list[DeviceMetadata]:
        """Get available screens.

        Returns:
            List of DeviceMetadata objects for each available screen
        """
        return self._device_manager.get_screens()

    def get_uvc_devices(self) -> list[DeviceMetadata]:
        """Get UVC video devices.

        Returns:
            List of DeviceMetadata objects for each available UVC device
        """
        return self._device_manager.get_uvc_devices()

    def get_audio_devices(self) -> list[DeviceMetadata]:
        """Get audio input devices.

        Returns:
            List of DeviceMetadata objects for each available audio device
        """
        return self._device_manager.get_audio_devices()

    def get_device_metadata(self, device_id: str) -> DeviceMetadata:
        """Get detailed metadata for a device.

        Args:
            device_id: Device identifier

        Returns:
            DeviceMetadata object with device information
        """
        return self._device_manager.get_device_metadata(device_id)

    # ========================================================================
    # Video Source Management (delegated to VideoSourceManager)
    # ========================================================================

    def add_video_source(self, source_id: str, capture: VideoCaptureInterface) -> None:
        """Add a video source.

        Args:
            source_id: Unique identifier for the source
            capture: VideoCaptureInterface instance

        Raises:
            ValueError: If source_id already exists
        """
        self._video_source_manager.add_source(source_id, capture)

    def remove_video_source(self, source_id: str) -> None:
        """Remove a video source.

        Args:
            source_id: Source identifier

        Raises:
            KeyError: If source_id does not exist
        """
        self._video_source_manager.remove_source(source_id)

    def get_video_source(self, source_id: str) -> Optional[VideoCaptureInterface]:
        """Get a video source by ID.

        Args:
            source_id: Source identifier

        Returns:
            VideoCaptureInterface instance, or None if not found
        """
        return self._video_source_manager.get_source(source_id)

    def get_all_video_sources(self) -> dict[str, VideoCaptureInterface]:
        """Get all video sources.

        Returns:
            Dictionary mapping source IDs to VideoCaptureInterface instances
        """
        return self._video_source_manager.get_all_sources()

    def get_video_source_ids(self) -> list[str]:
        """Get list of all video source IDs.

        Returns:
            List of source IDs
        """
        return self._video_source_manager.get_source_ids()

    def get_video_source_count(self) -> int:
        """Get the number of active video sources.

        Returns:
            Number of active sources
        """
        return self._video_source_manager.get_source_count()

    def collect_frames(self) -> dict[str, Optional[VideoFrame]]:
        """Collect the latest frame from all active video sources.

        Returns:
            Dictionary mapping source IDs to VideoFrame objects
        """
        return self._video_source_manager.collect_frames()

    def stop_all_video_sources(self) -> None:
        """Stop capture for all video sources."""
        self._video_source_manager.stop_all()

    def clear_all_video_sources(self) -> None:
        """Remove all video sources and stop their captures."""
        self._video_source_manager.clear_all()

    def is_video_source_capturing(self, source_id: str) -> bool:
        """Check if a specific video source is currently capturing.

        Args:
            source_id: Source identifier

        Returns:
            True if source is capturing, False otherwise
        """
        return self._video_source_manager.is_capturing(source_id)

    def has_video_source(self, source_id: str) -> bool:
        """Check if a video source exists.

        Args:
            source_id: Source identifier

        Returns:
            True if source exists, False otherwise
        """
        return source_id in self._video_source_manager

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    def __len__(self) -> int:
        """Get the number of active video sources.

        Returns:
            Number of active sources
        """
        return len(self._video_source_manager)

    def __contains__(self, source_id: str) -> bool:
        """Check if a video source exists.

        Args:
            source_id: Source identifier

        Returns:
            True if source exists, False otherwise
        """
        return source_id in self._video_source_manager

    def __repr__(self) -> str:
        """Get string representation.

        Returns:
            String representation
        """
        return f"CaptureManager(video_sources={self.get_video_source_count()})"

    # ========================================================================
    # Audio Source Management
    # ========================================================================

    def add_audio_source(self, source_id: str, capture: object) -> None:
        """Add an audio source.

        Args:
            source_id: Unique identifier for the source
            capture: AudioCaptureInterface instance

        Raises:
            ValueError: If source_id already exists
        """
        if source_id in self._audio_sources:
            raise ValueError(f"Audio source already exists: {source_id}")

        self._audio_sources[source_id] = capture
        logger.info(f"Added audio source: {source_id}")

    def remove_audio_source(self, source_id: str) -> None:
        """Remove an audio source.

        Args:
            source_id: Source identifier

        Raises:
            KeyError: If source_id does not exist
        """
        if source_id not in self._audio_sources:
            raise KeyError(f"Audio source not found: {source_id}")

        # Clean up stream
        self._stream_manager.cleanup_stream(source_id)

        del self._audio_sources[source_id]
        logger.info(f"Removed audio source: {source_id}")

    def get_audio_source(self, source_id: str) -> Optional[object]:
        """Get an audio source by ID.

        Args:
            source_id: Source identifier

        Returns:
            AudioCaptureInterface instance, or None if not found
        """
        return self._audio_sources.get(source_id)

    def get_audio_source_ids(self) -> list[str]:
        """Get list of all audio source IDs.

        Returns:
            List of source IDs
        """
        return list(self._audio_sources.keys())

    def get_audio_source_count(self) -> int:
        """Get the number of active audio sources.

        Returns:
            Number of active sources
        """
        return len(self._audio_sources)

    def stop_all_audio_sources(self) -> None:
        """Stop capture for all audio sources."""
        for source_id, capture in self._audio_sources.items():
            try:
                if hasattr(capture, "stop_capture") and callable(capture.stop_capture):
                    capture.stop_capture()  # type: ignore
                logger.debug(f"Stopped audio source: {source_id}")
            except Exception as e:
                logger.error(f"Error stopping audio source {source_id}: {e}", exc_info=True)

    def clear_all_audio_sources(self) -> None:
        """Remove all audio sources and stop their captures."""
        self.stop_all_audio_sources()
        self._audio_sources.clear()
        logger.info("Cleared all audio sources")

    # ========================================================================
    # Stream Management (for AI modules)
    # ========================================================================

    def get_stream_manager(self) -> StreamManager:
        """Get the stream manager for AI module access.

        Returns:
            StreamManager instance
        """
        return self._stream_manager
