"""Audio capture interface and implementations."""

import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import sounddevice as sd


@dataclass
class AudioChunk:
    """Audio data chunk with metadata."""

    data: np.ndarray
    sample_rate: int
    timestamp: float


class AudioCaptureInterface(ABC):
    """Abstract interface for audio capture implementations."""

    @abstractmethod
    def start_capture(self) -> None:
        """Start capturing audio."""
        pass

    @abstractmethod
    def stop_capture(self) -> None:
        """Stop capturing audio."""
        pass

    @abstractmethod
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get the latest audio chunk.

        Returns:
            Audio data as numpy array or None if no audio available
        """
        pass

    @abstractmethod
    def list_devices(self) -> List[Dict[str, Any]]:
        """List available audio input devices.

        Returns:
            List of device information dictionaries
        """
        pass


class SoundDeviceAudioCapture(AudioCaptureInterface):
    """Audio capture using sounddevice library."""

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        buffer_size: int = 100,
    ):
        """Initialize sounddevice audio capture.

        Args:
            device_id: Audio device ID (None for default device)
            sample_rate: Sample rate in Hz (default 16kHz for speech)
            chunk_size: Number of samples per chunk (default 1024)
            channels: Number of audio channels (default 1 for mono)
            buffer_size: Maximum number of chunks to buffer (default 100)
        """
        self._device_id = device_id
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._channels = channels
        self._buffer_size = buffer_size

        self._stream: Optional[sd.InputStream] = None
        self._is_capturing = False
        self._buffer_lock = threading.Lock()
        self._audio_buffer: deque = deque(maxlen=buffer_size)

    def start_capture(self) -> None:
        """Start capturing audio."""
        if self._is_capturing:
            self.stop_capture()

        self._is_capturing = True

        # Create and start audio stream
        self._stream = sd.InputStream(
            device=self._device_id,
            channels=self._channels,
            samplerate=self._sample_rate,
            blocksize=self._chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop_capture(self) -> None:
        """Stop capturing audio."""
        if not self._is_capturing:
            return

        self._is_capturing = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get the latest audio chunk.

        Returns:
            Audio data as numpy array or None if no audio available
        """
        with self._buffer_lock:
            if not self._audio_buffer:
                return None
            return self._audio_buffer[-1].copy()

    def list_devices(self) -> List[Dict[str, Any]]:
        """List available audio input devices.

        Returns:
            List of device information dictionaries with keys:
                - id: Device ID
                - name: Device name
                - channels: Number of input channels
                - sample_rate: Default sample rate
        """
        devices = []

        try:
            device_list = sd.query_devices()

            for i, device in enumerate(device_list):
                # Only include devices with input channels
                if device["max_input_channels"] > 0:
                    devices.append(
                        {
                            "id": i,
                            "name": device["name"],
                            "channels": device["max_input_channels"],
                            "sample_rate": int(device["default_samplerate"]),
                        }
                    )

        except Exception as e:
            print(f"Error listing audio devices: {e}")

        return devices

    def set_device(self, device_id: Optional[int]) -> None:
        """Set audio input device.

        Args:
            device_id: Device ID or None for default device
        """
        was_capturing = self._is_capturing

        if was_capturing:
            self.stop_capture()

        self._device_id = device_id

        if was_capturing:
            self.start_capture()

    def get_device(self) -> Optional[int]:
        """Get current audio input device ID.

        Returns:
            Device ID or None if using default device
        """
        return self._device_id

    def set_sample_rate(self, sample_rate: int) -> None:
        """Set audio sample rate.

        Args:
            sample_rate: Sample rate in Hz
        """
        was_capturing = self._is_capturing

        if was_capturing:
            self.stop_capture()

        self._sample_rate = sample_rate

        if was_capturing:
            self.start_capture()

    def get_sample_rate(self) -> int:
        """Get current sample rate.

        Returns:
            Sample rate in Hz
        """
        return self._sample_rate

    def get_buffer_size(self) -> int:
        """Get current buffer size (number of chunks).

        Returns:
            Number of chunks in buffer
        """
        with self._buffer_lock:
            return len(self._audio_buffer)

    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        with self._buffer_lock:
            self._audio_buffer.clear()

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        """Callback function for audio stream.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            print(f"Audio callback status: {status}")

        if not self._is_capturing:
            return

        # Copy audio data to buffer
        audio_chunk = indata.copy()

        with self._buffer_lock:
            self._audio_buffer.append(audio_chunk)
