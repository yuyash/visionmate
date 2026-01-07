"""Stream interface for AI modules to access captured data.

This module provides a clean interface for AI modules to access video frames
and audio chunks without needing to know about device management or capture logic.
It implements separation of concerns by abstracting away the capture implementation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

from visionmate.core.models import AudioChunk, VideoFrame

logger = logging.getLogger(__name__)


class StreamInterface(ABC):
    """Abstract base class for data streams.

    Provides a common interface for accessing captured data (video/audio)
    without exposing implementation details to AI modules.
    """

    @abstractmethod
    def get_latest_data(self) -> Optional[VideoFrame | AudioChunk]:
        """Get the most recent data item.

        Returns:
            Latest VideoFrame or AudioChunk, or None if no data available
        """
        pass

    @abstractmethod
    def register_callback(self, callback: Callable[[VideoFrame | AudioChunk], None]) -> None:
        """Register a callback to be called when new data arrives.

        Args:
            callback: Function to call with new data
        """
        pass

    @abstractmethod
    def unregister_callback(self, callback: Callable[[VideoFrame | AudioChunk], None]) -> None:
        """Unregister a previously registered callback.

        Args:
            callback: Function to unregister
        """
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Check if the stream is currently active.

        Returns:
            True if stream is active, False otherwise
        """
        pass

    @abstractmethod
    def get_source_id(self) -> str:
        """Get the source identifier for this stream.

        Returns:
            Source identifier string
        """
        pass


class VideoStream(StreamInterface):
    """Video stream interface for AI modules.

    Provides access to video frames from a capture source without
    exposing the underlying capture implementation.

    A stream is considered "active" when its underlying capture is running,
    which typically means it has been added to a preview and is capturing frames.
    """

    def __init__(self, source_id: str, capture_interface):
        """Initialize the video stream.

        Args:
            source_id: Unique identifier for the video source
            capture_interface: VideoCaptureInterface instance
        """
        self._source_id = source_id
        self._capture = capture_interface
        self._callbacks: list[Callable[[VideoFrame], None]] = []

        logger.debug(f"VideoStream initialized for source: {source_id}")

    def get_latest_data(self) -> Optional[VideoFrame]:
        """Get the most recent video frame.

        Returns:
            Latest VideoFrame, or None if no frame available
        """
        try:
            return self._capture.get_latest_frame()
        except Exception as e:
            logger.error(f"Error getting latest frame: {e}", exc_info=True)
            return None

    def get_latest_frame(self) -> Optional[VideoFrame]:
        """Get the most recent video frame (alias for get_latest_data).

        Returns:
            Latest VideoFrame, or None if no frame available
        """
        return self.get_latest_data()

    def register_callback(self, callback: Callable[[VideoFrame], None]) -> None:
        """Register a callback to be called when new frames arrive.

        Args:
            callback: Function to call with new VideoFrame
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"Registered callback for video stream: {self._source_id}")

    def unregister_callback(self, callback: Callable[[VideoFrame], None]) -> None:
        """Unregister a previously registered callback.

        Args:
            callback: Function to unregister
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Unregistered callback for video stream: {self._source_id}")

    def _notify_callbacks(self, frame: VideoFrame) -> None:
        """Notify all registered callbacks of a new frame.

        Args:
            frame: New VideoFrame to pass to callbacks
        """
        for callback in self._callbacks:
            try:
                callback(frame)
            except Exception as e:
                logger.error(f"Error in video stream callback: {e}", exc_info=True)

    def is_active(self) -> bool:
        """Check if the video stream is currently active.

        A stream is active when its underlying capture is running,
        which typically means it has been added to a preview.

        Returns:
            True if stream is capturing, False otherwise
        """
        try:
            return self._capture.is_capturing()
        except Exception:
            return False

    def get_source_id(self) -> str:
        """Get the source identifier for this stream.

        Returns:
            Source identifier string
        """
        return self._source_id

    def get_fps(self) -> int:
        """Get the frame rate of the video stream.

        Returns:
            Frame rate in frames per second
        """
        try:
            metadata = self._capture.get_device_metadata()
            return metadata.current_fps or 1
        except Exception:
            return 1

    def get_resolution(self):
        """Get the resolution of the video stream.

        Returns:
            Resolution object with width and height
        """
        try:
            metadata = self._capture.get_device_metadata()
            return metadata.current_resolution
        except Exception:
            return None


class AudioStream(StreamInterface):
    """Audio stream interface for AI modules.

    Provides access to audio chunks from a capture source without
    exposing the underlying capture implementation.

    A stream is considered "active" when its underlying capture is running,
    which typically means it has been added to a preview and is capturing audio.
    """

    def __init__(self, source_id: str, capture_interface):
        """Initialize the audio stream.

        Args:
            source_id: Unique identifier for the audio source
            capture_interface: AudioCaptureInterface instance
        """
        self._source_id = source_id
        self._capture = capture_interface
        self._callbacks: list[Callable[[AudioChunk], None]] = []

        logger.debug(f"AudioStream initialized for source: {source_id}")

    def get_latest_data(self) -> Optional[AudioChunk]:
        """Get the most recent audio chunk.

        Returns:
            Latest AudioChunk, or None if no chunk available
        """
        try:
            return self._capture.get_latest_chunk()
        except Exception as e:
            logger.error(f"Error getting latest audio chunk: {e}", exc_info=True)
            return None

    def get_latest_chunk(self) -> Optional[AudioChunk]:
        """Get the most recent audio chunk (alias for get_latest_data).

        Returns:
            Latest AudioChunk, or None if no chunk available
        """
        return self.get_latest_data()

    def register_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        """Register a callback to be called when new audio chunks arrive.

        Args:
            callback: Function to call with new AudioChunk
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"Registered callback for audio stream: {self._source_id}")

    def unregister_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        """Unregister a previously registered callback.

        Args:
            callback: Function to unregister
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Unregistered callback for audio stream: {self._source_id}")

    def _notify_callbacks(self, chunk: AudioChunk) -> None:
        """Notify all registered callbacks of a new audio chunk.

        Args:
            chunk: New AudioChunk to pass to callbacks
        """
        for callback in self._callbacks:
            try:
                callback(chunk)
            except Exception as e:
                logger.error(f"Error in audio stream callback: {e}", exc_info=True)

    def is_active(self) -> bool:
        """Check if the audio stream is currently active.

        A stream is active when its underlying capture is running,
        which typically means it has been added to a preview.

        Returns:
            True if stream is capturing, False otherwise
        """
        try:
            return self._capture.is_capturing()
        except Exception:
            return False

    def get_source_id(self) -> str:
        """Get the source identifier for this stream.

        Returns:
            Source identifier string
        """
        return self._source_id

    def get_sample_rate(self) -> int:
        """Get the sample rate of the audio stream.

        Returns:
            Sample rate in Hz
        """
        try:
            metadata = self._capture.get_device_metadata()
            return metadata.current_sample_rate or 16000
        except Exception:
            return 16000

    def get_channels(self) -> int:
        """Get the number of audio channels.

        Returns:
            Number of channels
        """
        try:
            metadata = self._capture.get_device_metadata()
            return metadata.current_channels or 1
        except Exception:
            return 1


class StreamManager:
    """Manager for video and audio streams.

    Provides a centralized interface for AI modules to access all active streams
    without needing to interact with the capture manager directly.

    Note: A stream is considered "active" if its capture is running, which typically
    means it has been added to a preview. The stream manager provides access to all
    streams that have been registered with the capture manager.
    """

    def __init__(self, capture_manager):
        """Initialize the stream manager.

        Args:
            capture_manager: CaptureManager instance
        """
        self._capture_manager = capture_manager
        self._video_streams: dict[str, VideoStream] = {}
        self._audio_streams: dict[str, AudioStream] = {}

        logger.debug("StreamManager initialized")

    def get_active_video_source_ids(self) -> list[str]:
        """Get list of all active (capturing) video source IDs.

        These are the sources that have been added to the capture manager
        and are currently capturing frames (typically shown in previews).

        Returns:
            List of video source IDs
        """
        return self._capture_manager.get_video_source_ids()

    def get_active_audio_source_ids(self) -> list[str]:
        """Get list of all active (capturing) audio source IDs.

        These are the sources that have been added to the capture manager
        and are currently capturing audio (typically shown in previews).

        Returns:
            List of audio source IDs
        """
        return self._capture_manager.get_audio_source_ids()

    def get_video_stream(self, source_id: str) -> Optional[VideoStream]:
        """Get a video stream by source ID.

        Args:
            source_id: Source identifier

        Returns:
            VideoStream instance, or None if not found
        """
        # Check cache first
        if source_id in self._video_streams:
            return self._video_streams[source_id]

        # Try to create from capture manager
        try:
            capture = self._capture_manager.get_video_source(source_id)
            if capture:
                stream = VideoStream(source_id, capture)
                self._video_streams[source_id] = stream
                return stream
        except Exception as e:
            logger.error(f"Error getting video stream: {e}", exc_info=True)

        return None

    def get_audio_stream(self, source_id: str) -> Optional[AudioStream]:
        """Get an audio stream by source ID.

        Args:
            source_id: Source identifier

        Returns:
            AudioStream instance, or None if not found
        """
        # Check cache first
        if source_id in self._audio_streams:
            return self._audio_streams[source_id]

        # Try to create from capture manager
        try:
            capture = self._capture_manager.get_audio_source(source_id)
            if capture:
                stream = AudioStream(source_id, capture)
                self._audio_streams[source_id] = stream
                return stream
        except Exception as e:
            logger.error(f"Error getting audio stream: {e}", exc_info=True)

        return None

    def get_all_video_streams(self) -> list[VideoStream]:
        """Get all active video streams.

        Returns:
            List of VideoStream instances
        """
        streams = []
        for source_id in self._capture_manager.get_video_source_ids():
            stream = self.get_video_stream(source_id)
            if stream:
                streams.append(stream)
        return streams

    def get_all_audio_streams(self) -> list[AudioStream]:
        """Get all active audio streams.

        Returns:
            List of AudioStream instances
        """
        streams = []
        for source_id in self._capture_manager.get_audio_source_ids():
            stream = self.get_audio_stream(source_id)
            if stream:
                streams.append(stream)
        return streams

    def cleanup_stream(self, source_id: str) -> None:
        """Clean up a stream when its source is removed.

        Args:
            source_id: Source identifier
        """
        if source_id in self._video_streams:
            del self._video_streams[source_id]
            logger.debug(f"Cleaned up video stream: {source_id}")

        if source_id in self._audio_streams:
            del self._audio_streams[source_id]
            logger.debug(f"Cleaned up audio stream: {source_id}")

    def cleanup_all_streams(self) -> None:
        """Clean up all streams."""
        self._video_streams.clear()
        self._audio_streams.clear()
        logger.debug("Cleaned up all streams")
