"""Unit tests for audio capture module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deskmate.capture.audio_capture import (
    AudioCaptureInterface,
    AudioChunk,
    SoundDeviceAudioCapture,
)


class TestAudioCaptureInterface:
    """Test AudioCaptureInterface abstract base class."""

    def test_interface_is_abstract(self):
        """Test that AudioCaptureInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            AudioCaptureInterface()


class TestAudioChunk:
    """Test AudioChunk dataclass."""

    def test_audio_chunk_creation(self):
        """Test creating AudioChunk instance."""
        data = np.zeros((1024, 1), dtype=np.float32)
        chunk = AudioChunk(data=data, sample_rate=16000, timestamp=123.456)

        assert chunk.data.shape == (1024, 1)
        assert chunk.sample_rate == 16000
        assert chunk.timestamp == 123.456


class TestSoundDeviceAudioCapture:
    """Test SoundDeviceAudioCapture implementation."""

    def test_initialization(self):
        """Test SoundDeviceAudioCapture initializes correctly."""
        capture = SoundDeviceAudioCapture()

        assert capture.get_sample_rate() == 16000
        assert capture.get_device() is None
        assert not capture._is_capturing
        assert capture.get_buffer_size() == 0

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        capture = SoundDeviceAudioCapture(
            device_id=1, sample_rate=44100, chunk_size=2048, channels=2, buffer_size=50
        )

        assert capture.get_sample_rate() == 44100
        assert capture.get_device() == 1
        assert capture._chunk_size == 2048
        assert capture._channels == 2
        assert capture._buffer_size == 50

    @patch("sounddevice.InputStream")
    def test_start_stop_capture(self, mock_input_stream):
        """Test starting and stopping audio capture."""
        # Mock InputStream
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        capture = SoundDeviceAudioCapture()

        # Start capture
        capture.start_capture()
        assert capture._is_capturing
        mock_input_stream.assert_called_once()
        mock_stream.start.assert_called_once()

        # Stop capture
        capture.stop_capture()
        assert not capture._is_capturing
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch("sounddevice.InputStream")
    def test_restart_capture(self, mock_input_stream):
        """Test restarting capture stops previous stream."""
        # Mock InputStream
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        capture = SoundDeviceAudioCapture()

        # Start capture twice
        capture.start_capture()
        capture.start_capture()

        # Should have stopped and started again
        assert mock_stream.stop.call_count == 1
        assert mock_stream.close.call_count == 1
        assert mock_stream.start.call_count == 2

    @patch("sounddevice.query_devices")
    def test_list_devices(self, mock_query_devices):
        """Test audio device enumeration."""
        # Mock device list
        mock_query_devices.return_value = [
            {
                "name": "Microphone 1",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 44100.0,
            },
            {
                "name": "Microphone 2",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "Speaker",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
        ]

        capture = SoundDeviceAudioCapture()
        devices = capture.list_devices()

        assert isinstance(devices, list)
        # Should only include devices with input channels (2 devices)
        assert len(devices) == 2

        # Check device structure
        assert devices[0]["id"] == 0
        assert devices[0]["name"] == "Microphone 1"
        assert devices[0]["channels"] == 2
        assert devices[0]["sample_rate"] == 44100

        assert devices[1]["id"] == 1
        assert devices[1]["name"] == "Microphone 2"
        assert devices[1]["channels"] == 1
        assert devices[1]["sample_rate"] == 48000

    @patch("sounddevice.query_devices")
    def test_list_devices_error_handling(self, mock_query_devices):
        """Test device enumeration handles errors gracefully."""
        # Mock error
        mock_query_devices.side_effect = Exception("Device query failed")

        capture = SoundDeviceAudioCapture()
        devices = capture.list_devices()

        # Should return empty list on error
        assert devices == []

    @patch("sounddevice.InputStream")
    def test_set_device(self, mock_input_stream):
        """Test setting audio device."""
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        capture = SoundDeviceAudioCapture()

        # Start capture with default device
        capture.start_capture()
        assert capture.get_device() is None

        # Change device while capturing
        capture.set_device(2)
        assert capture.get_device() == 2

        # Should have restarted capture
        assert mock_stream.stop.call_count == 1
        assert mock_stream.start.call_count == 2

    @patch("sounddevice.InputStream")
    def test_set_device_while_not_capturing(self, mock_input_stream):
        """Test setting device when not capturing."""
        capture = SoundDeviceAudioCapture()

        # Set device without capturing
        capture.set_device(1)
        assert capture.get_device() == 1

        # Should not have started stream
        mock_input_stream.assert_not_called()

    @patch("sounddevice.InputStream")
    def test_set_sample_rate(self, mock_input_stream):
        """Test setting sample rate."""
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        capture = SoundDeviceAudioCapture()

        # Start capture with default sample rate
        capture.start_capture()
        assert capture.get_sample_rate() == 16000

        # Change sample rate while capturing
        capture.set_sample_rate(44100)
        assert capture.get_sample_rate() == 44100

        # Should have restarted capture
        assert mock_stream.stop.call_count == 1
        assert mock_stream.start.call_count == 2

    @patch("sounddevice.InputStream")
    def test_set_sample_rate_while_not_capturing(self, mock_input_stream):
        """Test setting sample rate when not capturing."""
        capture = SoundDeviceAudioCapture()

        # Set sample rate without capturing
        capture.set_sample_rate(48000)
        assert capture.get_sample_rate() == 48000

        # Should not have started stream
        mock_input_stream.assert_not_called()

    def test_audio_callback(self):
        """Test audio callback adds data to buffer."""
        capture = SoundDeviceAudioCapture()
        capture._is_capturing = True

        # Simulate audio callback
        audio_data = np.random.randn(1024, 1).astype(np.float32)
        capture._audio_callback(audio_data, 1024, None, None)

        # Check buffer has data
        assert capture.get_buffer_size() == 1

        # Get audio chunk
        chunk = capture.get_audio_chunk()
        assert chunk is not None
        assert chunk.shape == (1024, 1)

    def test_audio_callback_when_not_capturing(self):
        """Test audio callback ignores data when not capturing."""
        capture = SoundDeviceAudioCapture()
        capture._is_capturing = False

        # Simulate audio callback
        audio_data = np.random.randn(1024, 1).astype(np.float32)
        capture._audio_callback(audio_data, 1024, None, None)

        # Buffer should be empty
        assert capture.get_buffer_size() == 0

    def test_buffer_management(self):
        """Test audio buffer management."""
        capture = SoundDeviceAudioCapture(buffer_size=5)
        capture._is_capturing = True

        # Add multiple chunks
        for i in range(10):
            audio_data = np.ones((1024, 1), dtype=np.float32) * i
            capture._audio_callback(audio_data, 1024, None, None)

        # Buffer should be limited to 5 chunks
        assert capture.get_buffer_size() == 5

        # Latest chunk should be the last one added (value 9)
        chunk = capture.get_audio_chunk()
        assert chunk is not None
        assert np.allclose(chunk[0, 0], 9.0)

    def test_get_audio_chunk_when_empty(self):
        """Test getting audio chunk when buffer is empty."""
        capture = SoundDeviceAudioCapture()

        chunk = capture.get_audio_chunk()
        assert chunk is None

    def test_clear_buffer(self):
        """Test clearing audio buffer."""
        capture = SoundDeviceAudioCapture()
        capture._is_capturing = True

        # Add some chunks
        for _ in range(5):
            audio_data = np.random.randn(1024, 1).astype(np.float32)
            capture._audio_callback(audio_data, 1024, None, None)

        assert capture.get_buffer_size() == 5

        # Clear buffer
        capture.clear_buffer()
        assert capture.get_buffer_size() == 0

    def test_audio_callback_with_status(self, capsys):
        """Test audio callback prints status when present."""
        capture = SoundDeviceAudioCapture()
        capture._is_capturing = True

        # Mock status flags
        mock_status = MagicMock()
        mock_status.__bool__ = lambda self: True
        mock_status.__str__ = lambda self: "Input overflow"

        # Simulate callback with status
        audio_data = np.random.randn(1024, 1).astype(np.float32)
        capture._audio_callback(audio_data, 1024, None, mock_status)

        # Check that status was printed
        captured = capsys.readouterr()
        assert "Audio callback status" in captured.out

    def test_get_audio_chunk_returns_copy(self):
        """Test that get_audio_chunk returns a copy, not reference."""
        capture = SoundDeviceAudioCapture()
        capture._is_capturing = True

        # Add chunk
        audio_data = np.ones((1024, 1), dtype=np.float32)
        capture._audio_callback(audio_data, 1024, None, None)

        # Get chunk twice
        chunk1 = capture.get_audio_chunk()
        chunk2 = capture.get_audio_chunk()

        # Should be equal but not the same object
        assert np.array_equal(chunk1, chunk2)
        assert chunk1 is not chunk2

        # Modifying one should not affect the other
        chunk1[0, 0] = 999.0
        assert chunk2[0, 0] == 1.0
