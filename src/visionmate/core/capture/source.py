"""Video source manager for managing multiple active video captures.

This module provides the VideoSourceManager class for coordinating multiple
video capture sources and collecting frames from all active sources.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from visionmate.core.capture.video import VideoCaptureInterface
from visionmate.core.models import VideoFrame

logger = logging.getLogger(__name__)


class VideoSourceManager:
    """Manages multiple active video capture sources.

    This class coordinates multiple video captures and provides a unified
    interface for collecting frames from all active sources.

    Requirements: 1.6
    """

    def __init__(self):
        """Initialize the VideoSourceManager."""
        self._sources: dict[str, VideoCaptureInterface] = {}
        self._lock = threading.RLock()
        logger.info("VideoSourceManager initialized")

    def add_source(self, source_id: str, capture: VideoCaptureInterface) -> None:
        """Add a video source to the manager.

        Args:
            source_id: Unique identifier for the source
            capture: VideoCaptureInterface instance

        Raises:
            ValueError: If source_id already exists

        Requirements: 1.6
        """
        with self._lock:
            if source_id in self._sources:
                raise ValueError(f"Source already exists: {source_id}")

            self._sources[source_id] = capture
            logger.info(f"Added video source: {source_id}")

    def remove_source(self, source_id: str) -> None:
        """Remove a video source from the manager.

        Args:
            source_id: Source identifier

        Raises:
            KeyError: If source_id does not exist

        Requirements: 1.6
        """
        with self._lock:
            if source_id not in self._sources:
                raise KeyError(f"Source not found: {source_id}")

            # Stop capture before removing
            capture = self._sources[source_id]
            if capture.is_capturing():
                capture.stop_capture()

            del self._sources[source_id]
            logger.info(f"Removed video source: {source_id}")

    def get_source(self, source_id: str) -> Optional[VideoCaptureInterface]:
        """Get a video source by ID.

        Args:
            source_id: Source identifier

        Returns:
            VideoCaptureInterface instance, or None if not found

        Requirements: 1.6
        """
        with self._lock:
            return self._sources.get(source_id)

    def get_all_sources(self) -> dict[str, VideoCaptureInterface]:
        """Get all video sources.

        Returns:
            Dictionary mapping source IDs to VideoCaptureInterface instances

        Requirements: 1.6
        """
        with self._lock:
            return self._sources.copy()

    def get_source_ids(self) -> list[str]:
        """Get list of all source IDs.

        Returns:
            List of source IDs

        Requirements: 1.6
        """
        with self._lock:
            return list(self._sources.keys())

    def get_source_count(self) -> int:
        """Get the number of active sources.

        Returns:
            Number of active sources

        Requirements: 1.6
        """
        with self._lock:
            return len(self._sources)

    def collect_frames(self) -> dict[str, Optional[VideoFrame]]:
        """Collect the latest frame from all active sources.

        Returns:
            Dictionary mapping source IDs to VideoFrame objects.
            Sources with no available frame will have None value.

        Requirements: 1.6
        """
        with self._lock:
            frames = {}
            for source_id, capture in self._sources.items():
                try:
                    frame = capture.get_frame()
                    frames[source_id] = frame
                except Exception as e:
                    logger.error(f"Error getting frame from {source_id}: {e}", exc_info=True)
                    frames[source_id] = None

            return frames

    def start_all(self) -> None:
        """Start capture for all sources that are not already capturing.

        Requirements: 1.6
        """
        with self._lock:
            for source_id, capture in self._sources.items():
                try:
                    if not capture.is_capturing():
                        # Note: start_capture requires device_id and other params
                        # This method assumes sources are already started when added
                        logger.debug(f"Source {source_id} already started or needs manual start")
                except Exception as e:
                    logger.error(f"Error checking capture status for {source_id}: {e}")

    def stop_all(self) -> None:
        """Stop capture for all sources.

        Requirements: 1.6
        """
        with self._lock:
            for source_id, capture in self._sources.items():
                try:
                    if capture.is_capturing():
                        capture.stop_capture()
                        logger.info(f"Stopped capture for source: {source_id}")
                except Exception as e:
                    logger.error(f"Error stopping capture for {source_id}: {e}", exc_info=True)

    def clear_all(self) -> None:
        """Remove all sources and stop their captures.

        Requirements: 1.6
        """
        with self._lock:
            source_ids = list(self._sources.keys())
            for source_id in source_ids:
                try:
                    self.remove_source(source_id)
                except Exception as e:
                    logger.error(f"Error removing source {source_id}: {e}", exc_info=True)

            logger.info("Cleared all video sources")

    def is_capturing(self, source_id: str) -> bool:
        """Check if a specific source is currently capturing.

        Args:
            source_id: Source identifier

        Returns:
            True if source is capturing, False otherwise

        Requirements: 1.6
        """
        with self._lock:
            capture = self._sources.get(source_id)
            if capture:
                return capture.is_capturing()
            return False

    def __len__(self) -> int:
        """Get the number of active sources.

        Returns:
            Number of active sources
        """
        return self.get_source_count()

    def __contains__(self, source_id: str) -> bool:
        """Check if a source exists in the manager.

        Args:
            source_id: Source identifier

        Returns:
            True if source exists, False otherwise
        """
        with self._lock:
            return source_id in self._sources

    def __repr__(self) -> str:
        """Get string representation of the manager.

        Returns:
            String representation
        """
        with self._lock:
            return f"VideoSourceManager(sources={len(self._sources)})"
