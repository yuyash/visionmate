"""Capture manager for managing devices and video sources.

This module provides the CaptureManager class that unifies device
management and video source management into a single cohesive interface.
"""

from __future__ import annotations

import logging
from typing import Optional

from visionmate.core.capture.device import DeviceManager
from visionmate.core.capture.source import VideoSourceManager
from visionmate.core.capture.video import VideoCaptureInterface
from visionmate.core.models import DeviceMetadata, OptimalSettings, VideoFrame

logger = logging.getLogger(__name__)


class CaptureManager:
    """Manages device enumeration and video source lifecycle.

    This class provides a unified interface for:
    - Device enumeration and metadata retrieval
    - Video source lifecycle management
    - Frame collection from multiple sources

    It encapsulates both DeviceManager and VideoSourceManager to provide
    a single point of access for all capture-related operations.

    Requirements: 1.6, 1.7, 1.8, 27.1-27.6
    """

    def __init__(self):
        """Initialize the CaptureManager."""
        self._device_manager = DeviceManager()
        self._video_source_manager = VideoSourceManager()
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

    def enumerate_screens(self) -> list[DeviceMetadata]:
        """Enumerate available screens.

        Returns:
            List of DeviceMetadata objects for each available screen

        Requirements: 1.7
        """
        return self._device_manager.enumerate_screens()

    def enumerate_uvc_devices(self) -> list[DeviceMetadata]:
        """Enumerate UVC video devices.

        Returns:
            List of DeviceMetadata objects for each available UVC device

        Requirements: 1.8
        """
        return self._device_manager.enumerate_uvc_devices()

    def enumerate_audio_devices(self) -> list[DeviceMetadata]:
        """Enumerate audio input devices.

        Returns:
            List of DeviceMetadata objects for each available audio device

        Requirements: 2.8
        """
        return self._device_manager.enumerate_audio_devices()

    def get_device_metadata(self, device_id: str) -> DeviceMetadata:
        """Get detailed metadata for a device.

        Args:
            device_id: Device identifier

        Returns:
            DeviceMetadata object with device information

        Requirements: 27.1-27.6
        """
        return self._device_manager.get_device_metadata(device_id)

    def validate_settings(
        self,
        device_id: str,
        resolution: tuple[int, int],
        fps: int,
    ) -> bool:
        """Validate if settings are supported by device.

        Args:
            device_id: Device identifier
            resolution: Resolution as (width, height)
            fps: Frame rate

        Returns:
            True if settings are supported, False otherwise

        Requirements: 27.8
        """
        return self._device_manager.validate_settings(device_id, resolution, fps)

    def suggest_optimal_settings(self, device_id: str) -> OptimalSettings:
        """Suggest optimal capture settings for device.

        Args:
            device_id: Device identifier

        Returns:
            OptimalSettings object with suggested settings

        Requirements: 27.9
        """
        return self._device_manager.suggest_optimal_settings(device_id)

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

        Requirements: 1.6
        """
        self._video_source_manager.add_source(source_id, capture)

    def remove_video_source(self, source_id: str) -> None:
        """Remove a video source.

        Args:
            source_id: Source identifier

        Raises:
            KeyError: If source_id does not exist

        Requirements: 1.6
        """
        self._video_source_manager.remove_source(source_id)

    def get_video_source(self, source_id: str) -> Optional[VideoCaptureInterface]:
        """Get a video source by ID.

        Args:
            source_id: Source identifier

        Returns:
            VideoCaptureInterface instance, or None if not found

        Requirements: 1.6
        """
        return self._video_source_manager.get_source(source_id)

    def get_all_video_sources(self) -> dict[str, VideoCaptureInterface]:
        """Get all video sources.

        Returns:
            Dictionary mapping source IDs to VideoCaptureInterface instances

        Requirements: 1.6
        """
        return self._video_source_manager.get_all_sources()

    def get_video_source_ids(self) -> list[str]:
        """Get list of all video source IDs.

        Returns:
            List of source IDs

        Requirements: 1.6
        """
        return self._video_source_manager.get_source_ids()

    def get_video_source_count(self) -> int:
        """Get the number of active video sources.

        Returns:
            Number of active sources

        Requirements: 1.6
        """
        return self._video_source_manager.get_source_count()

    def collect_frames(self) -> dict[str, Optional[VideoFrame]]:
        """Collect the latest frame from all active video sources.

        Returns:
            Dictionary mapping source IDs to VideoFrame objects

        Requirements: 1.6
        """
        return self._video_source_manager.collect_frames()

    def stop_all_video_sources(self) -> None:
        """Stop capture for all video sources.

        Requirements: 1.6
        """
        self._video_source_manager.stop_all()

    def clear_all_video_sources(self) -> None:
        """Remove all video sources and stop their captures.

        Requirements: 1.6
        """
        self._video_source_manager.clear_all()

    def is_video_source_capturing(self, source_id: str) -> bool:
        """Check if a specific video source is currently capturing.

        Args:
            source_id: Source identifier

        Returns:
            True if source is capturing, False otherwise

        Requirements: 1.6
        """
        return self._video_source_manager.is_capturing(source_id)

    def has_video_source(self, source_id: str) -> bool:
        """Check if a video source exists.

        Args:
            source_id: Source identifier

        Returns:
            True if source exists, False otherwise

        Requirements: 1.6
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
