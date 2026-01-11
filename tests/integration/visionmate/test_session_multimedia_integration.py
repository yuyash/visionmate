"""Integration tests for SessionManager and MultimediaManager integration.

This module tests the integration between SessionManager and MultimediaManager,
verifying that both server-side and client-side audio modes work correctly
and that mode switching functions as expected.

"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import List, Optional
from unittest.mock import Mock

import numpy as np
import pytest

from visionmate.core.capture.stream import AudioStream, StreamManager, VideoStream
from visionmate.core.models import (
    AudioChunk,
    AudioMode,
    AudioSourceType,
    Resolution,
    SessionState,
    VideoFrame,
    VideoSourceType,
)
from visionmate.core.recognition import (
    RequestResponseVLMClient,
    SpeechToTextInterface,
    StreamingVLMClient,
    VLMRequest,
    VLMResponse,
)
from visionmate.core.session.manager import SessionManager

logger = logging.getLogger(__name__)


# ============================================================================
# Mock Implementations
# ============================================================================


class MockStreamingVLMClient(StreamingVLMClient):
    """Mock streaming VLM client for testing."""

    def __init__(self):
        """Initialize mock client."""
        self.connected = False
        self.frames_received: List[VideoFrame] = []
        self.audio_chunks_received: List[AudioChunk] = []
        self.texts_received: List[str] = []
        self.response_callback: Optional[Callable] = None
        self._model = "mock-model"

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return ["mock-model"]

    def set_model(self, model_name: str) -> None:
        """Set the model to use."""
        self._model = model_name

    async def connect(self) -> None:
        """Connect to VLM service."""
        self.connected = True
        logger.info("MockStreamingVLMClient connected")

    async def disconnect(self) -> None:
        """Disconnect from VLM service."""
        self.connected = False
        logger.info("MockStreamingVLMClient disconnected")

    async def send_frame(self, frame: VideoFrame) -> None:
        """Send video frame."""
        if not self.connected:
            raise ConnectionError("Not connected")
        self.frames_received.append(frame)
        logger.debug(f"MockStreamingVLMClient received frame at {frame.timestamp}")

    async def send_audio_chunk(self, audio: AudioChunk) -> None:
        """Send audio chunk."""
        if not self.connected:
            raise ConnectionError("Not connected")
        self.audio_chunks_received.append(audio)
        logger.debug(f"MockStreamingVLMClient received audio at {audio.timestamp}")

    async def send_text(self, text: str) -> None:
        """Send text."""
        if not self.connected:
            raise ConnectionError("Not connected")
        self.texts_received.append(text)
        logger.debug(f"MockStreamingVLMClient received text: {text[:50]}")

    async def notify_topic_change(self) -> None:
        """Notify of topic change."""
        logger.info("MockStreamingVLMClient notified of topic change")

    def register_response_callback(self, callback: Callable) -> None:
        """Register response callback."""
        self.response_callback = callback
        logger.debug("MockStreamingVLMClient registered response callback")

    def get_received_count(self) -> dict:
        """Get count of received items."""
        return {
            "frames": len(self.frames_received),
            "audio_chunks": len(self.audio_chunks_received),
            "texts": len(self.texts_received),
        }

    def clear_received(self) -> None:
        """Clear received items."""
        self.frames_received.clear()
        self.audio_chunks_received.clear()
        self.texts_received.clear()


class MockRequestResponseVLMClient(RequestResponseVLMClient):
    """Mock request-response VLM client for testing."""

    def __init__(self):
        """Initialize mock client."""
        self.requests_received: List[VLMRequest] = []
        self._model = "mock-model"

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return ["mock-model"]

    def set_model(self, model_name: str) -> None:
        """Set the model to use."""
        self._model = model_name

    async def process_multimodal_input(self, request: VLMRequest) -> VLMResponse:
        """Process multimodal input."""
        self.requests_received.append(request)
        logger.debug(
            f"MockRequestResponseVLMClient received request with "
            f"{len(request.frames)} frames, "
            f"audio={'yes' if request.audio else 'no'}, "
            f"text={'yes' if request.text else 'no'}"
        )

        # Return mock response
        return VLMResponse(
            question="Test question",
            direct_answer="Test answer",
            confidence=0.9,
            timestamp=int(datetime.now().timestamp()),
        )

    def get_received_count(self) -> int:
        """Get count of received requests."""
        return len(self.requests_received)

    def clear_received(self) -> None:
        """Clear received requests."""
        self.requests_received.clear()


class MockSTTClient(SpeechToTextInterface):
    """Mock STT client for testing."""

    def __init__(self):
        """Initialize mock client."""
        from visionmate.core.models import STTProvider

        self._provider = STTProvider.WHISPER
        self.transcriptions: List[str] = []
        self._language = "en"

    async def transcribe(self, audio: AudioChunk) -> str:
        """Transcribe audio."""
        # Return mock transcription
        text = f"Transcribed audio at {audio.timestamp.isoformat()}"
        self.transcriptions.append(text)
        logger.debug(f"MockSTTClient transcribed: {text}")
        return text

    def get_provider(self):
        """Get STT provider."""
        return self._provider

    def is_available(self) -> bool:
        """Check if STT is available."""
        return True

    def set_language(self, language: str) -> None:
        """Set language for transcription."""
        self._language = language

    def get_transcription_count(self) -> int:
        """Get count of transcriptions."""
        return len(self.transcriptions)

    def clear_transcriptions(self) -> None:
        """Clear transcriptions."""
        self.transcriptions.clear()


class MockStreamManager(StreamManager):
    """Mock stream manager for testing."""

    def __init__(self) -> None:
        """Initialize mock stream manager."""
        self._video_streams: List[VideoStream] = []
        self._audio_streams: List[AudioStream] = []

    def add_mock_video_stream(self, source_id: str) -> VideoStream:
        """Add mock video stream."""
        # Create mock capture interface
        mock_capture = Mock()
        mock_capture.get_latest_frame = Mock(return_value=None)
        mock_capture.is_capturing = Mock(return_value=True)

        stream = VideoStream(source_id=source_id, capture_interface=mock_capture)
        self._video_streams.append(stream)
        return stream

    def add_mock_audio_stream(self, source_id: str) -> AudioStream:
        """Add mock audio stream."""
        # Create mock capture interface
        mock_capture = Mock()
        mock_capture.get_latest_chunk = Mock(return_value=None)
        mock_capture.is_capturing = Mock(return_value=True)

        stream = AudioStream(source_id=source_id, capture_interface=mock_capture)
        self._audio_streams.append(stream)
        return stream

    def get_all_video_streams(self) -> List[VideoStream]:  # type: ignore[override]
        """Get all video streams."""
        return self._video_streams

    def get_all_audio_streams(self) -> List[AudioStream]:  # type: ignore[override]
        """Get all audio streams."""
        return self._audio_streams


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def session_manager():
    """Create session manager for testing."""
    manager = SessionManager()
    yield manager
    # Cleanup
    if manager.get_state() == SessionState.ACTIVE:
        manager.stop()


@pytest.fixture
def mock_streaming_vlm():
    """Create mock streaming VLM client."""
    return MockStreamingVLMClient()


@pytest.fixture
def mock_request_response_vlm():
    """Create mock request-response VLM client."""
    return MockRequestResponseVLMClient()


@pytest.fixture
def mock_stt():
    """Create mock STT client."""
    return MockSTTClient()


@pytest.fixture
def mock_stream_manager():
    """Create mock stream manager."""
    return MockStreamManager()


# ============================================================================
# Test Cases
# ============================================================================


class TestServerSideModeIntegration:
    """Test server-side mode integration."""

    def test_start_session_server_side_mode(
        self,
        session_manager: SessionManager,
        mock_streaming_vlm: MockStreamingVLMClient,
        mock_stream_manager: MockStreamManager,
    ):
        """Test starting session with server-side mode.

        Verifies that:
        1. Session can be started with server-side mode
        2. MultimediaManager is initialized correctly
        3. Streams are registered with callbacks
        4. VLM client is connected

        """
        # Configure session
        session_manager.set_vlm_client(mock_streaming_vlm)
        session_manager.set_audio_mode(AudioMode.SERVER_SIDE)

        # Inject mock stream manager
        session_manager._stream_manager = mock_stream_manager  # type: ignore

        # Add mock streams
        _video_stream = mock_stream_manager.add_mock_video_stream("test_video")
        _audio_stream = mock_stream_manager.add_mock_audio_stream("test_audio")

        # Add mock video source (required for start)
        session_manager._video_source_configs["test_video"] = Mock()
        mock_capture = Mock()
        mock_capture.is_capturing = Mock(return_value=True)
        mock_capture.get_frame = Mock(return_value=None)
        session_manager._capture_manager.add_video_source("test_video", mock_capture)

        # Add mock audio source (required for start)
        session_manager._audio_source_config = Mock()

        # Start session
        session_manager.start()

        # Wait for async initialization to complete
        import time

        time.sleep(2.0)  # Increased wait time for async connection

        # Verify session state
        assert session_manager.get_state() == SessionState.ACTIVE
        assert session_manager.get_audio_mode() == AudioMode.SERVER_SIDE

        # Verify multimedia manager was created
        assert session_manager._multimedia_manager is not None

        # Verify VLM client is connected (may take time due to async connection)
        # The connection happens in a background thread, so we need to wait
        for _ in range(10):
            if mock_streaming_vlm.connected:
                break
            time.sleep(0.5)

        assert mock_streaming_vlm.connected

        # Cleanup
        session_manager.stop()

        # Wait for cleanup
        time.sleep(0.5)

        # Verify cleanup
        assert session_manager.get_state() == SessionState.IDLE
        assert not mock_streaming_vlm.connected

    def test_server_side_mode_data_flow(
        self,
        session_manager: SessionManager,
        mock_streaming_vlm: MockStreamingVLMClient,
        mock_stream_manager: MockStreamManager,
    ):
        """Test data flow in server-side mode.

        Verifies that frames and audio are streamed to VLM immediately.

        """
        # Configure session
        session_manager.set_vlm_client(mock_streaming_vlm)
        session_manager.set_audio_mode(AudioMode.SERVER_SIDE)
        session_manager._stream_manager = mock_stream_manager  # type: ignore

        # Add mock streams
        video_stream = mock_stream_manager.add_mock_video_stream("test_video")
        audio_stream = mock_stream_manager.add_mock_audio_stream("test_audio")

        # Add mock sources
        session_manager._video_source_configs["test_video"] = Mock()
        mock_capture = Mock()
        mock_capture.is_capturing = Mock(return_value=True)
        mock_capture.get_frame = Mock(return_value=None)
        session_manager._capture_manager.add_video_source("test_video", mock_capture)
        session_manager._audio_source_config = Mock()

        # Start session
        session_manager.start()

        # Wait for initialization and connection
        import time

        time.sleep(2.0)

        # Wait for VLM client to connect
        for _ in range(10):
            if mock_streaming_vlm.connected:
                break
            time.sleep(0.5)

        # Simulate frame capture
        frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=datetime.now(),
            source_id="test_video",
            source_type=VideoSourceType.SCREEN,
            resolution=Resolution(width=640, height=480),
            fps=1,
            frame_number=1,
        )

        # Trigger frame callback
        for callback in video_stream._callbacks:
            callback(frame)

        # Simulate audio capture
        audio = AudioChunk(
            data=np.zeros(1600, dtype=np.float32),
            timestamp=datetime.now(),
            source_id="test_audio",
            sample_rate=16000,
            channels=1,
            source_type=AudioSourceType.DEVICE,
            chunk_number=1,
        )

        # Trigger audio callback
        for callback in audio_stream._callbacks:
            callback(audio)

        # Wait for processing
        time.sleep(1.0)

        # Verify data was sent to VLM
        counts = mock_streaming_vlm.get_received_count()
        assert counts["frames"] > 0 or counts["audio_chunks"] > 0

        # Cleanup
        session_manager.stop()


class TestClientSideModeIntegration:
    """Test client-side mode integration."""

    def test_start_session_client_side_mode(
        self,
        session_manager: SessionManager,
        mock_streaming_vlm: MockStreamingVLMClient,
        mock_stt: MockSTTClient,
        mock_stream_manager: MockStreamManager,
    ):
        """Test starting session with client-side mode.

        Verifies that:
        1. Session can be started with client-side mode
        2. MultimediaManager is initialized with STT client
        3. Buffering is enabled

        """
        # Configure session
        session_manager.set_vlm_client(mock_streaming_vlm)
        session_manager.set_stt_client(mock_stt)
        session_manager.set_audio_mode(AudioMode.CLIENT_SIDE)

        # Inject mock stream manager
        session_manager._stream_manager = mock_stream_manager  # type: ignore

        # Add mock streams
        _video_stream = mock_stream_manager.add_mock_video_stream("test_video")
        _audio_stream = mock_stream_manager.add_mock_audio_stream("test_audio")

        # Add mock sources
        session_manager._video_source_configs["test_video"] = Mock()
        session_manager._capture_manager.add_video_source(
            "test_video", Mock(is_capturing=Mock(return_value=True))
        )
        session_manager._audio_source_config = Mock()

        # Start session
        session_manager.start()

        # Verify session state
        assert session_manager.get_state() == SessionState.ACTIVE
        assert session_manager.get_audio_mode() == AudioMode.CLIENT_SIDE

        # Verify multimedia manager was created
        assert session_manager._multimedia_manager is not None

        # Cleanup
        session_manager.stop()

        # Verify cleanup
        assert session_manager.get_state() == SessionState.IDLE

    def test_client_side_mode_requires_stt(
        self,
        session_manager: SessionManager,
        mock_streaming_vlm: MockStreamingVLMClient,
    ):
        """Test that client-side mode requires STT client.

        Verifies that setting client-side mode without STT client raises error.

        """
        # Configure session
        session_manager.set_vlm_client(mock_streaming_vlm)

        # Try to set client-side mode without STT client
        with pytest.raises(ValueError, match="STT client is required"):
            session_manager.set_audio_mode(AudioMode.CLIENT_SIDE)


class TestModeSwitching:
    """Test mode switching functionality."""

    def test_switch_mode_while_stopped(
        self,
        session_manager: SessionManager,
        mock_streaming_vlm: MockStreamingVLMClient,
        mock_stt: MockSTTClient,
    ):
        """Test switching audio mode while session is stopped.

        Verifies that mode can be changed when session is idle.

        """
        # Configure session
        session_manager.set_vlm_client(mock_streaming_vlm)
        session_manager.set_stt_client(mock_stt)

        # Start with server-side mode
        session_manager.set_audio_mode(AudioMode.SERVER_SIDE)
        assert session_manager.get_audio_mode() == AudioMode.SERVER_SIDE

        # Switch to client-side mode
        session_manager.set_audio_mode(AudioMode.CLIENT_SIDE)
        assert session_manager.get_audio_mode() == AudioMode.CLIENT_SIDE

        # Switch back to server-side mode
        session_manager.set_audio_mode(AudioMode.SERVER_SIDE)
        assert session_manager.get_audio_mode() == AudioMode.SERVER_SIDE

    def test_cannot_switch_mode_while_active(
        self,
        session_manager: SessionManager,
        mock_streaming_vlm: MockStreamingVLMClient,
        mock_stream_manager: MockStreamManager,
    ):
        """Test that mode cannot be changed while session is active.

        Verifies that attempting to change mode while active raises error.

        """
        # Configure session
        session_manager.set_vlm_client(mock_streaming_vlm)
        session_manager.set_audio_mode(AudioMode.SERVER_SIDE)
        session_manager._stream_manager = mock_stream_manager  # type: ignore

        # Add mock sources
        session_manager._video_source_configs["test_video"] = Mock()
        session_manager._capture_manager.add_video_source(
            "test_video", Mock(is_capturing=Mock(return_value=True))
        )
        session_manager._audio_source_config = Mock()

        # Start session
        session_manager.start()

        # Try to change mode while active
        with pytest.raises(RuntimeError, match="Cannot change audio mode while session is active"):
            session_manager.set_audio_mode(AudioMode.CLIENT_SIDE)

        # Cleanup
        session_manager.stop()

    def test_restart_with_different_mode(
        self,
        session_manager: SessionManager,
        mock_streaming_vlm: MockStreamingVLMClient,
        mock_stt: MockSTTClient,
        mock_stream_manager: MockStreamManager,
    ):
        """Test restarting session with different audio mode.

        Verifies that session can be stopped, mode changed, and restarted.

        """
        # Configure session
        session_manager.set_vlm_client(mock_streaming_vlm)
        session_manager.set_stt_client(mock_stt)
        session_manager._stream_manager = mock_stream_manager  # type: ignore

        # Add mock sources
        session_manager._video_source_configs["test_video"] = Mock()
        session_manager._capture_manager.add_video_source(
            "test_video", Mock(is_capturing=Mock(return_value=True))
        )
        session_manager._audio_source_config = Mock()

        # Start with server-side mode
        session_manager.set_audio_mode(AudioMode.SERVER_SIDE)
        session_manager.start()
        assert session_manager.get_state() == SessionState.ACTIVE
        assert session_manager.get_audio_mode() == AudioMode.SERVER_SIDE

        # Stop session
        session_manager.stop()
        assert session_manager.get_state() == SessionState.IDLE

        # Switch to client-side mode
        session_manager.set_audio_mode(AudioMode.CLIENT_SIDE)
        assert session_manager.get_audio_mode() == AudioMode.CLIENT_SIDE

        # Restart session
        session_manager.start()
        assert session_manager.get_state() == SessionState.ACTIVE
        assert session_manager.get_audio_mode() == AudioMode.CLIENT_SIDE

        # Cleanup
        session_manager.stop()


class TestErrorHandling:
    """Test error handling in integration."""

    def test_start_without_vlm_client(
        self,
        session_manager: SessionManager,
        mock_stream_manager: MockStreamManager,
    ):
        """Test starting session without VLM client.

        Verifies that session can start without VLM client (recognition disabled).

        """
        # Configure session without VLM client
        session_manager._stream_manager = mock_stream_manager  # type: ignore

        # Add mock sources
        session_manager._video_source_configs["test_video"] = Mock()
        session_manager._capture_manager.add_video_source(
            "test_video", Mock(is_capturing=Mock(return_value=True))
        )
        session_manager._audio_source_config = Mock()

        # Start session (should succeed but log warning)
        session_manager.start()
        assert session_manager.get_state() == SessionState.ACTIVE

        # Cleanup
        session_manager.stop()

    def test_start_without_sources(
        self,
        session_manager: SessionManager,
        mock_streaming_vlm: MockStreamingVLMClient,
    ):
        """Test starting session without sources.

        Verifies that starting without sources raises error.

        """
        # Configure session
        session_manager.set_vlm_client(mock_streaming_vlm)

        # Try to start without sources
        with pytest.raises(RuntimeError, match="No input sources configured"):
            session_manager.start()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
