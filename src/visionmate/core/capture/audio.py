"""Audio capture interfaces and implementations.

This module provides audio capture functionality for various sources including
audio devices, UVC devices, and RTSP streams.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from threading import Lock, Thread
from typing import Optional

import cv2
import numpy as np
import sounddevice as sd

from visionmate.core.models import AudioChunk, AudioSourceType, DeviceMetadata, DeviceType

logger = logging.getLogger(__name__)


# ============================================================================
# Audio Capture Interface
# ============================================================================


class AudioCaptureInterface(ABC):
    """Abstract interface for audio capture.

    This interface defines the contract for all audio capture implementations,
    supporting various audio sources including audio devices, UVC devices,
    and RTSP streams.

    Requirements: 2.1, 2.2, 2.3
    """

    @abstractmethod
    def start_capture(
        self,
        device_id: str,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        """Start capturing audio.

        Args:
            device_id: Device identifier
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)

        Raises:
            RuntimeError: If capture is already running
            ValueError: If device_id is invalid or device not available
        """

    @abstractmethod
    def stop_capture(self) -> None:
        """Stop capturing audio.

        This method should be idempotent - calling it multiple times
        should not cause errors.
        """

    @abstractmethod
    def get_chunk(self) -> Optional[AudioChunk]:
        """Get the latest audio chunk.

        Returns:
            AudioChunk object with audio data and metadata, or None if no data available
        """

    @abstractmethod
    def is_capturing(self) -> bool:
        """Check if currently capturing.

        Returns:
            True if capture is active, False otherwise
        """

    @abstractmethod
    def get_source_info(self) -> DeviceMetadata:
        """Get source device metadata.

        Returns:
            DeviceMetadata object with device information

        Raises:
            RuntimeError: If no device is configured
        """


# ============================================================================
# Device Audio Capture
# ============================================================================


class DeviceAudioCapture(AudioCaptureInterface):
    """Audio device capture using sounddevice library.

    This implementation captures audio from system audio input devices
    using the sounddevice library. It uses a ring buffer to store audio
    chunks for continuous streaming.

    Requirements: 2.1, 2.6
    """

    def __init__(self, chunk_duration: float = 0.5):
        """Initialize DeviceAudioCapture.

        Args:
            chunk_duration: Duration of each audio chunk in seconds (default: 0.5)
        """
        self._device_id: Optional[str] = None
        self._sample_rate: int = 16000
        self._channels: int = 1
        self._chunk_duration = chunk_duration
        self._chunk_size: int = 0
        self._is_capturing = False
        self._chunk_number = 0
        self._stream: Optional[sd.InputStream] = None
        self._buffer: deque[AudioChunk] = deque(maxlen=10)  # Ring buffer
        self._buffer_lock = Lock()
        logger.debug("DeviceAudioCapture initialized with chunk_duration=%.2f", chunk_duration)

    def start_capture(
        self,
        device_id: str,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        """Start capturing audio from device.

        Args:
            device_id: Device identifier (device index as string or "audio_N" format)
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels

        Raises:
            RuntimeError: If capture is already running
            ValueError: If device_id is invalid
        """
        if self._is_capturing:
            raise RuntimeError("Audio capture is already running")

        try:
            # Parse device_id - support both "audio_N" and "N" formats
            if device_id.startswith("audio_"):
                device_index = int(device_id.split("_")[1])
            else:
                device_index = int(device_id)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid device_id: {device_id}") from e

        self._device_id = device_id
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = int(sample_rate * self._chunk_duration)
        self._chunk_number = 0

        # Create audio stream
        try:
            self._stream = sd.InputStream(
                device=device_index,
                channels=channels,
                samplerate=sample_rate,
                blocksize=self._chunk_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_capturing = True
            logger.info(
                "Started audio capture: device=%s, sample_rate=%d, channels=%d",
                device_id,
                sample_rate,
                channels,
            )
        except Exception as e:
            logger.error("Failed to start audio capture: %s", e)
            raise ValueError(f"Failed to start audio capture: {e}") from e

    def stop_capture(self) -> None:
        """Stop capturing audio."""
        if not self._is_capturing:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._is_capturing = False
        logger.info("Stopped audio capture: device=%s", self._device_id)

    def get_chunk(self) -> Optional[AudioChunk]:
        """Get the latest audio chunk from buffer.

        Returns:
            AudioChunk object, or None if buffer is empty
        """
        with self._buffer_lock:
            if self._buffer:
                return self._buffer[-1]  # Return most recent chunk
        return None

    def is_capturing(self) -> bool:
        """Check if currently capturing.

        Returns:
            True if capture is active
        """
        return self._is_capturing

    def get_source_info(self) -> DeviceMetadata:
        """Get source device metadata.

        Returns:
            DeviceMetadata object

        Raises:
            RuntimeError: If no device is configured
        """
        if self._device_id is None:
            raise RuntimeError("No device configured")

        try:
            # Parse device_id - support both "audio_N" and "N" formats
            if self._device_id.startswith("audio_"):
                device_index = int(self._device_id.split("_")[1])
            else:
                device_index = int(self._device_id)

            device_info = sd.query_devices(device_index)

            return DeviceMetadata(
                device_id=self._device_id,
                name=device_info["name"],
                device_type=DeviceType.AUDIO,
                sample_rates=[int(device_info["default_samplerate"])],
                channels=[device_info["max_input_channels"]],
                current_sample_rate=self._sample_rate if self._is_capturing else None,
                current_channels=self._channels if self._is_capturing else None,
                is_available=True,
            )
        except Exception as e:
            logger.error("Failed to get device metadata: %s", e)
            return DeviceMetadata(
                device_id=self._device_id,
                name="Unknown Device",
                device_type=DeviceType.AUDIO,
                is_available=False,
                error_message=str(e),
            )

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback for audio stream.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Stream status
        """
        if status:
            logger.warning("Audio stream status: %s", status)

        # Create audio chunk
        from datetime import datetime, timezone

        chunk = AudioChunk(
            data=indata.copy(),
            sample_rate=self._sample_rate,
            channels=self._channels,
            timestamp=datetime.now(timezone.utc),
            source_id=self._device_id or "unknown",
            source_type=AudioSourceType.DEVICE,
            chunk_number=self._chunk_number,
        )
        self._chunk_number += 1

        # Add to buffer
        with self._buffer_lock:
            self._buffer.append(chunk)


# ============================================================================
# UVC Audio Capture
# ============================================================================


class UVCAudioCapture(AudioCaptureInterface):
    """UVC device audio capture using OpenCV.

    This implementation captures audio from UVC devices that support
    audio input using OpenCV's VideoCapture API.

    Requirements: 2.2
    """

    def __init__(self, chunk_duration: float = 0.5):
        """Initialize UVCAudioCapture.

        Args:
            chunk_duration: Duration of each audio chunk in seconds (default: 0.5)
        """
        self._device_id: Optional[str] = None
        self._sample_rate: int = 16000
        self._channels: int = 1
        self._chunk_duration = chunk_duration
        self._chunk_size: int = 0
        self._is_capturing = False
        self._chunk_number = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[Thread] = None
        self._buffer: deque[AudioChunk] = deque(maxlen=10)
        self._buffer_lock = Lock()
        logger.debug("UVCAudioCapture initialized with chunk_duration=%.2f", chunk_duration)

    def start_capture(
        self,
        device_id: str,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        """Start capturing audio from UVC device.

        Args:
            device_id: Device identifier (device index as string or "audio_N" format)
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels

        Raises:
            RuntimeError: If capture is already running
            ValueError: If device cannot be opened
        """
        if self._is_capturing:
            raise RuntimeError("Audio capture is already running")

        try:
            # Parse device_id - support both "audio_N" and "N" formats
            if device_id.startswith("audio_"):
                device_index = int(device_id.split("_")[1])
            else:
                device_index = int(device_id)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid device_id: {device_id}") from e

        self._device_id = device_id
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = int(sample_rate * self._chunk_duration)
        self._chunk_number = 0

        # Open video capture (OpenCV doesn't have direct audio API)
        # Note: OpenCV's audio support is limited, this is a placeholder
        # In practice, you might need platform-specific audio extraction
        self._capture = cv2.VideoCapture(device_index)
        if not self._capture.isOpened():
            raise ValueError(f"Failed to open UVC device: {device_id}")

        self._is_capturing = True
        self._capture_thread = Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        logger.info(
            "Started UVC audio capture: device=%s, sample_rate=%d, channels=%d",
            device_id,
            sample_rate,
            channels,
        )

    def stop_capture(self) -> None:
        """Stop capturing audio."""
        if not self._is_capturing:
            return

        self._is_capturing = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        logger.info("Stopped UVC audio capture: device=%s", self._device_id)

    def get_chunk(self) -> Optional[AudioChunk]:
        """Get the latest audio chunk from buffer.

        Returns:
            AudioChunk object, or None if buffer is empty
        """
        with self._buffer_lock:
            if self._buffer:
                return self._buffer[-1]
        return None

    def is_capturing(self) -> bool:
        """Check if currently capturing.

        Returns:
            True if capture is active
        """
        return self._is_capturing

    def get_source_info(self) -> DeviceMetadata:
        """Get source device metadata.

        Returns:
            DeviceMetadata object

        Raises:
            RuntimeError: If no device is configured
        """
        if self._device_id is None:
            raise RuntimeError("No device configured")

        return DeviceMetadata(
            device_id=self._device_id,
            name=f"UVC Audio Device {self._device_id}",
            device_type=DeviceType.AUDIO,
            sample_rates=[self._sample_rate],
            channels=[self._channels],
            current_sample_rate=self._sample_rate if self._is_capturing else None,
            current_channels=self._channels if self._is_capturing else None,
            is_available=True,
        )

    def _capture_loop(self) -> None:
        """Capture loop running in separate thread.

        Note: This is a placeholder implementation. OpenCV doesn't provide
        direct audio capture API. In a real implementation, you would need
        to use platform-specific APIs or libraries to extract audio from
        UVC devices.
        """
        logger.warning("UVC audio capture is not fully implemented - placeholder only")

        # Placeholder: Generate silent audio chunks
        import time
        from datetime import datetime, timezone

        while self._is_capturing:
            # Generate silent audio chunk
            data = np.zeros((self._chunk_size, self._channels), dtype=np.float32)

            chunk = AudioChunk(
                data=data,
                sample_rate=self._sample_rate,
                channels=self._channels,
                timestamp=datetime.now(timezone.utc),
                source_id=self._device_id or "unknown",
                source_type=AudioSourceType.UVC,
                chunk_number=self._chunk_number,
            )
            self._chunk_number += 1

            with self._buffer_lock:
                self._buffer.append(chunk)

            # Sleep for chunk duration
            time.sleep(self._chunk_duration)


# ============================================================================
# RTSP Audio Capture
# ============================================================================


class RTSPAudioCapture(AudioCaptureInterface):
    """RTSP stream audio capture using OpenCV.

    This implementation captures audio from RTSP streams using OpenCV's
    VideoCapture API.

    Requirements: 2.3
    """

    def __init__(self, chunk_duration: float = 0.5):
        """Initialize RTSPAudioCapture.

        Args:
            chunk_duration: Duration of each audio chunk in seconds (default: 0.5)
        """
        self._device_id: Optional[str] = None  # RTSP URL
        self._sample_rate: int = 16000
        self._channels: int = 1
        self._chunk_duration = chunk_duration
        self._chunk_size: int = 0
        self._is_capturing = False
        self._chunk_number = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[Thread] = None
        self._buffer: deque[AudioChunk] = deque(maxlen=10)
        self._buffer_lock = Lock()
        logger.debug("RTSPAudioCapture initialized with chunk_duration=%.2f", chunk_duration)

    def start_capture(
        self,
        device_id: str,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        """Start capturing audio from RTSP stream.

        Args:
            device_id: RTSP URL
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels

        Raises:
            RuntimeError: If capture is already running
            ValueError: If RTSP stream cannot be opened
        """
        if self._is_capturing:
            raise RuntimeError("Audio capture is already running")

        self._device_id = device_id
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = int(sample_rate * self._chunk_duration)
        self._chunk_number = 0

        # Open RTSP stream
        self._capture = cv2.VideoCapture(device_id)
        if not self._capture.isOpened():
            raise ValueError(f"Failed to open RTSP stream: {device_id}")

        self._is_capturing = True
        self._capture_thread = Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        logger.info(
            "Started RTSP audio capture: url=%s, sample_rate=%d, channels=%d",
            device_id,
            sample_rate,
            channels,
        )

    def stop_capture(self) -> None:
        """Stop capturing audio."""
        if not self._is_capturing:
            return

        self._is_capturing = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        logger.info("Stopped RTSP audio capture: url=%s", self._device_id)

    def get_chunk(self) -> Optional[AudioChunk]:
        """Get the latest audio chunk from buffer.

        Returns:
            AudioChunk object, or None if buffer is empty
        """
        with self._buffer_lock:
            if self._buffer:
                return self._buffer[-1]
        return None

    def is_capturing(self) -> bool:
        """Check if currently capturing.

        Returns:
            True if capture is active
        """
        return self._is_capturing

    def get_source_info(self) -> DeviceMetadata:
        """Get source device metadata.

        Returns:
            DeviceMetadata object

        Raises:
            RuntimeError: If no device is configured
        """
        if self._device_id is None:
            raise RuntimeError("No device configured")

        return DeviceMetadata(
            device_id=self._device_id,
            name="RTSP Audio Stream",
            device_type=DeviceType.AUDIO,
            sample_rates=[self._sample_rate],
            channels=[self._channels],
            current_sample_rate=self._sample_rate if self._is_capturing else None,
            current_channels=self._channels if self._is_capturing else None,
            is_available=True,
        )

    def _capture_loop(self) -> None:
        """Capture loop running in separate thread.

        Note: This is a placeholder implementation. OpenCV doesn't provide
        direct audio capture API. In a real implementation, you would need
        to use ffmpeg or other tools to extract audio from RTSP streams.
        """
        logger.warning("RTSP audio capture is not fully implemented - placeholder only")

        # Placeholder: Generate silent audio chunks
        import time
        from datetime import datetime, timezone

        while self._is_capturing:
            # Generate silent audio chunk
            data = np.zeros((self._chunk_size, self._channels), dtype=np.float32)

            chunk = AudioChunk(
                data=data,
                sample_rate=self._sample_rate,
                channels=self._channels,
                timestamp=datetime.now(timezone.utc),
                source_id=self._device_id or "unknown",
                source_type=AudioSourceType.RTSP,
                chunk_number=self._chunk_number,
            )
            self._chunk_number += 1

            with self._buffer_lock:
                self._buffer.append(chunk)

            # Sleep for chunk duration
            time.sleep(self._chunk_duration)


# ============================================================================
# Audio Mixer
# ============================================================================


class AudioMixer:
    """Mixes multiple audio sources into a single stream.

    This class manages multiple audio capture sources and mixes their
    audio output into a single stream. It supports per-source volume
    control and automatic sample rate conversion.

    Requirements: 2.5, 2.9
    """

    def __init__(self, target_sample_rate: int = 16000, target_channels: int = 1):
        """Initialize AudioMixer.

        Args:
            target_sample_rate: Target sample rate for mixed output (default: 16000)
            target_channels: Target number of channels for mixed output (default: 1)
        """
        self._sources: dict[str, AudioCaptureInterface] = {}
        self._volumes: dict[str, float] = {}
        self._target_sample_rate = target_sample_rate
        self._target_channels = target_channels
        self._source_lock = Lock()
        self._chunk_number = 0
        logger.debug(
            "AudioMixer initialized: sample_rate=%d, channels=%d",
            target_sample_rate,
            target_channels,
        )

    def add_source(self, source: AudioCaptureInterface) -> str:
        """Add audio source to mixer.

        Args:
            source: AudioCaptureInterface instance

        Returns:
            Source ID (generated from source metadata)

        Raises:
            ValueError: If source is already added
        """
        source_info = source.get_source_info()
        source_id = source_info.device_id

        with self._source_lock:
            if source_id in self._sources:
                raise ValueError(f"Source {source_id} already added")

            self._sources[source_id] = source
            self._volumes[source_id] = 1.0  # Default volume

        logger.info("Added audio source to mixer: %s", source_id)
        return source_id

    def remove_source(self, source_id: str) -> None:
        """Remove audio source from mixer.

        Args:
            source_id: Source identifier

        Raises:
            KeyError: If source_id does not exist
        """
        with self._source_lock:
            if source_id not in self._sources:
                raise KeyError(f"Source {source_id} not found")

            del self._sources[source_id]
            del self._volumes[source_id]

        logger.info("Removed audio source from mixer: %s", source_id)

    def get_mixed_chunk(self) -> Optional[AudioChunk]:
        """Get mixed audio chunk from all sources.

        This method collects audio chunks from all active sources,
        converts them to the target sample rate and channels if needed,
        applies volume control, and mixes them into a single chunk.

        Returns:
            Mixed AudioChunk object, or None if no sources are active

        Requirements: 2.5, 2.9
        """
        with self._source_lock:
            if not self._sources:
                return None

            # Collect chunks from all sources
            chunks: list[tuple[str, AudioChunk]] = []
            for source_id, source in self._sources.items():
                if source.is_capturing():
                    chunk = source.get_chunk()
                    if chunk is not None:
                        chunks.append((source_id, chunk))

            if not chunks:
                return None

            # Convert and mix audio
            mixed_data = None
            latest_timestamp = None

            for source_id, chunk in chunks:
                # Convert to target format
                converted_data = self._convert_audio(
                    chunk.data,
                    chunk.sample_rate,
                    chunk.channels,
                    self._target_sample_rate,
                    self._target_channels,
                )

                # Apply volume
                volume = self._volumes.get(source_id, 1.0)
                converted_data = converted_data * volume

                # Mix with existing data
                if mixed_data is None:
                    mixed_data = converted_data
                else:
                    # Ensure same length (pad shorter one with zeros)
                    if len(converted_data) < len(mixed_data):
                        pad_length = len(mixed_data) - len(converted_data)
                        converted_data = np.pad(
                            converted_data,
                            ((0, pad_length), (0, 0)),
                            mode="constant",
                        )
                    elif len(converted_data) > len(mixed_data):
                        pad_length = len(converted_data) - len(mixed_data)
                        mixed_data = np.pad(
                            mixed_data,
                            ((0, pad_length), (0, 0)),
                            mode="constant",
                        )

                    # Mix (average to prevent clipping)
                    mixed_data = mixed_data + converted_data

                # Track latest timestamp
                if latest_timestamp is None or chunk.timestamp > latest_timestamp:
                    latest_timestamp = chunk.timestamp

            if mixed_data is None or latest_timestamp is None:
                return None

            # Normalize to prevent clipping (divide by number of sources)
            mixed_data = mixed_data / len(chunks)

            # Create mixed chunk
            mixed_chunk = AudioChunk(
                data=mixed_data,
                sample_rate=self._target_sample_rate,
                channels=self._target_channels,
                timestamp=latest_timestamp,
                source_id="mixed",
                source_type=AudioSourceType.DEVICE,  # Use DEVICE as default
                chunk_number=self._chunk_number,
            )
            self._chunk_number += 1

            return mixed_chunk

    def set_source_volume(self, source_id: str, volume: float) -> None:
        """Set volume for specific source.

        Args:
            source_id: Source identifier
            volume: Volume level (0.0-1.0, where 1.0 is 100%)

        Raises:
            KeyError: If source_id does not exist
            ValueError: If volume is out of range
        """
        if volume < 0.0 or volume > 1.0:
            raise ValueError(f"Volume must be between 0.0 and 1.0, got {volume}")

        with self._source_lock:
            if source_id not in self._sources:
                raise KeyError(f"Source {source_id} not found")

            self._volumes[source_id] = volume

        logger.debug("Set volume for source %s: %.2f", source_id, volume)

    def get_source_volume(self, source_id: str) -> float:
        """Get volume for specific source.

        Args:
            source_id: Source identifier

        Returns:
            Volume level (0.0-1.0)

        Raises:
            KeyError: If source_id does not exist
        """
        with self._source_lock:
            if source_id not in self._sources:
                raise KeyError(f"Source {source_id} not found")

            return self._volumes[source_id]

    def get_source_ids(self) -> list[str]:
        """Get list of all source IDs.

        Returns:
            List of source IDs
        """
        with self._source_lock:
            return list(self._sources.keys())

    def get_source_count(self) -> int:
        """Get number of sources.

        Returns:
            Number of sources
        """
        with self._source_lock:
            return len(self._sources)

    def clear_all(self) -> None:
        """Remove all sources from mixer."""
        with self._source_lock:
            self._sources.clear()
            self._volumes.clear()

        logger.info("Cleared all audio sources from mixer")

    def _convert_audio(
        self,
        data: np.ndarray,
        src_sample_rate: int,
        src_channels: int,
        dst_sample_rate: int,
        dst_channels: int,
    ) -> np.ndarray:
        """Convert audio to target format.

        Args:
            data: Input audio data (frames, channels)
            src_sample_rate: Source sample rate
            src_channels: Source number of channels
            dst_sample_rate: Target sample rate
            dst_channels: Target number of channels

        Returns:
            Converted audio data

        Requirements: 2.9
        """
        # Convert sample rate if needed
        if src_sample_rate != dst_sample_rate:
            # Simple linear interpolation for sample rate conversion
            # For production, consider using scipy.signal.resample or librosa
            src_length = len(data)
            dst_length = int(src_length * dst_sample_rate / src_sample_rate)

            # Create interpolation indices
            src_indices = np.linspace(0, src_length - 1, dst_length)

            # Interpolate each channel
            converted_data = np.zeros((dst_length, src_channels), dtype=data.dtype)
            for ch in range(src_channels):
                converted_data[:, ch] = np.interp(
                    src_indices,
                    np.arange(src_length),
                    data[:, ch],
                )
            data = converted_data

        # Convert channels if needed
        if src_channels != dst_channels:
            if dst_channels == 1 and src_channels > 1:
                # Convert to mono by averaging channels
                data = np.mean(data, axis=1, keepdims=True)
            elif dst_channels > 1 and src_channels == 1:
                # Convert mono to multi-channel by duplicating
                data = np.repeat(data, dst_channels, axis=1)
            else:
                # For other conversions, use simple channel mapping
                # This is a simplified approach
                min_channels = min(src_channels, dst_channels)
                new_data = np.zeros((len(data), dst_channels), dtype=data.dtype)
                new_data[:, :min_channels] = data[:, :min_channels]
                data = new_data

        return data

    def __len__(self) -> int:
        """Get number of sources.

        Returns:
            Number of sources
        """
        return self.get_source_count()

    def __contains__(self, source_id: str) -> bool:
        """Check if source exists.

        Args:
            source_id: Source identifier

        Returns:
            True if source exists
        """
        with self._source_lock:
            return source_id in self._sources

    def __repr__(self) -> str:
        """Get string representation.

        Returns:
            String representation
        """
        return f"AudioMixer(sources={self.get_source_count()}, sample_rate={self._target_sample_rate}, channels={self._target_channels})"
