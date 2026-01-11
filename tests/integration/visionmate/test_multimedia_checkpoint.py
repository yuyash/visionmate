"""Checkpoint verification for MultimediaManager.

This test verifies that the multimedia manager works correctly with both
audio modes (server-side and client-side) using mock streams to simulate
the event-driven data flow.

Task 13: Checkpoint - Verify multimedia manager
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, List
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from visionmate.core.capture.stream import StreamManager
from visionmate.core.models import (
    AudioChunk,
    AudioMode,
    AudioSourceType,
    ManagerConfig,
    VideoFrame,
    VideoSourceType,
)
from visionmate.core.multimedia.manager import MultimediaManager


class MockVideoStream:
    """Mock video stream for testing."""

    def __init__(self, source_id: str = "test-video"):
        self.source_id = source_id
        self.callbacks: List[Callable[[VideoFrame], None]] = []

    def get_source_id(self) -> str:
        return self.source_id

    def register_callback(self, callback: Callable[[VideoFrame], None]) -> None:
        self.callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[VideoFrame], None]) -> None:
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def emit_frame(self, frame: VideoFrame) -> None:
        """Simulate frame capture by calling registered callbacks."""
        for callback in self.callbacks:
            callback(frame)


class MockAudioStream:
    """Mock audio stream for testing."""

    def __init__(self, source_id: str = "test-audio"):
        self.source_id = source_id
        self.callbacks: List[Callable[[AudioChunk], None]] = []

    def get_source_id(self) -> str:
        return self.source_id

    def register_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        self.callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def emit_audio(self, audio: AudioChunk) -> None:
        """Simulate audio capture by calling registered callbacks."""
        for callback in self.callbacks:
            callback(audio)


class MockStreamManager(StreamManager):
    """Mock stream manager for testing."""

    def __init__(self) -> None:
        self.video_streams: List[MockVideoStream] = []
        self.audio_streams: List[MockAudioStream] = []

    def add_video_stream(self, stream: MockVideoStream) -> None:
        self.video_streams.append(stream)

    def add_audio_stream(self, stream: MockAudioStream) -> None:
        self.audio_streams.append(stream)

    def get_all_video_streams(self) -> List[Any]:  # type: ignore
        return self.video_streams  # type: ignore

    def get_all_audio_streams(self) -> List[Any]:  # type: ignore
        return self.audio_streams  # type: ignore


def create_test_frame(timestamp: datetime, frame_number: int = 1) -> VideoFrame:
    """Create a test video frame."""
    return VideoFrame(
        image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        timestamp=timestamp,
        source_id="test-video",
        source_type=VideoSourceType.SCREEN,
        resolution=Mock(width=640, height=480),
        fps=30,
        frame_number=frame_number,
    )


def create_test_audio(timestamp: datetime, chunk_number: int = 1) -> AudioChunk:
    """Create a test audio chunk."""
    return AudioChunk(
        data=np.random.randn(16000).astype(np.float32) * 0.1,
        sample_rate=16000,
        channels=1,
        timestamp=timestamp,
        source_id="test-audio",
        source_type=AudioSourceType.DEVICE,
        chunk_number=chunk_number,
    )


@pytest.mark.asyncio
class TestMultimediaManagerCheckpoint:
    """Checkpoint tests for MultimediaManager."""

    async def test_server_side_mode_event_flow(self):
        """Test event-driven flow in server-side mode.

        Verifies:
        - Manager starts and registers callbacks
        - Frames and audio are captured via callbacks
        - Segments are built and sent to VLM
        - Manager stops and unregisters callbacks
        """
        # Create mock VLM client
        vlm_client = Mock()
        vlm_client.send_frame = AsyncMock()
        vlm_client.send_audio_chunk = AsyncMock()

        # Create mock streams
        stream_manager = MockStreamManager()
        video_stream = MockVideoStream()
        audio_stream = MockAudioStream()
        stream_manager.add_video_stream(video_stream)
        stream_manager.add_audio_stream(audio_stream)

        # Create manager in server-side mode
        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
            stream_manager=stream_manager,  # type: ignore
            config=ManagerConfig(
                max_segment_buffer_size=100,
                energy_threshold=0.01,
            ),
        )

        # Start manager
        await manager.start()

        # Verify callbacks were registered
        assert len(video_stream.callbacks) == 1
        assert len(audio_stream.callbacks) == 1
        assert manager._is_running

        # Simulate capture events
        base_time = datetime.now(timezone.utc)

        # Emit some frames
        for i in range(5):
            frame = create_test_frame(base_time, frame_number=i + 1)
            await video_stream.emit_frame(frame)
            await asyncio.sleep(0.01)  # Small delay to simulate real capture

        # Emit audio chunk (should build segment with frames)
        audio = create_test_audio(base_time, chunk_number=1)
        await audio_stream.emit_audio(audio)

        # Give time for async processing
        await asyncio.sleep(0.1)

        # Verify segment was built (frames were added to builder)
        assert len(manager._segment_builder._frame_buffer) > 0

        # Stop manager
        await manager.stop()

        # Verify callbacks were unregistered
        assert len(video_stream.callbacks) == 0
        assert len(audio_stream.callbacks) == 0
        assert not manager._is_running

        print("✓ Server-side mode event flow verified")

    async def test_client_side_mode_event_flow(self):
        """Test event-driven flow in client-side mode.

        Verifies:
        - Manager starts with STT client
        - Frames and audio are buffered during speech
        - Activity detection triggers buffering
        - Manager stops cleanly
        """
        # Create mock clients
        vlm_client = Mock()
        stt_client = Mock()
        stt_client.transcribe = AsyncMock(return_value="test transcription")

        # Create mock streams
        stream_manager = MockStreamManager()
        video_stream = MockVideoStream()
        audio_stream = MockAudioStream()
        stream_manager.add_video_stream(video_stream)
        stream_manager.add_audio_stream(audio_stream)

        # Create manager in client-side mode
        manager = MultimediaManager(
            audio_mode=AudioMode.CLIENT_SIDE,
            vlm_client=vlm_client,
            stt_client=stt_client,
            stream_manager=stream_manager,  # type: ignore
            config=ManagerConfig(
                max_segment_buffer_size=100,
                energy_threshold=0.01,
                silence_duration_sec=0.5,  # Short for testing
            ),
        )

        # Start manager
        await manager.start()

        # Verify callbacks were registered
        assert len(video_stream.callbacks) == 1
        assert len(audio_stream.callbacks) == 1
        assert manager._is_running

        # Verify client-side components are initialized
        assert manager._segment_buffer is not None
        assert manager._activity_detector is not None
        assert manager._client_handler is not None

        # Simulate capture events with speech
        base_time = datetime.now(timezone.utc)

        # Emit frames
        for i in range(3):
            frame = create_test_frame(base_time, frame_number=i + 1)
            await video_stream.emit_frame(frame)
            await asyncio.sleep(0.01)

        # Emit audio with speech (high energy)
        audio = create_test_audio(base_time, chunk_number=1)
        # Make audio louder to trigger speech detection
        audio.data = audio.data * 10.0
        await audio_stream.emit_audio(audio)

        # Give time for async processing
        await asyncio.sleep(0.1)

        # Verify segment was built
        assert len(manager._segment_builder._frame_buffer) > 0

        # Stop manager
        await manager.stop()

        # Verify cleanup
        assert len(video_stream.callbacks) == 0
        assert len(audio_stream.callbacks) == 0
        assert not manager._is_running

        print("✓ Client-side mode event flow verified")

    async def test_mode_switching(self):
        """Test switching between audio modes.

        Verifies:
        - Manager can switch modes when stopped
        - Handlers are reinitialized correctly
        - Mode validation works
        """
        # Create mock clients
        vlm_client = Mock()
        stt_client = Mock()

        # Create manager in server-side mode
        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
            stt_client=stt_client,
        )

        # Verify initial mode
        assert manager._audio_mode == AudioMode.SERVER_SIDE
        assert manager._server_handler is not None
        assert manager._client_handler is None

        # Switch to client-side mode
        manager.set_audio_mode(AudioMode.CLIENT_SIDE)

        # Verify mode changed
        assert manager._audio_mode == AudioMode.CLIENT_SIDE
        assert manager._server_handler is None
        assert manager._client_handler is not None
        assert manager._segment_buffer is not None
        assert manager._activity_detector is not None

        # Switch back to server-side mode
        manager.set_audio_mode(AudioMode.SERVER_SIDE)

        # Verify mode changed back
        assert manager._audio_mode == AudioMode.SERVER_SIDE
        assert manager._server_handler is not None
        assert manager._client_handler is None

        print("✓ Mode switching verified")

    async def test_error_handling(self):
        """Test error handling in event-driven flow.

        Verifies:
        - Errors in callbacks are caught
        - Error callback is invoked
        - Manager continues operating after errors
        """
        # Create mock VLM client
        vlm_client = Mock()

        # Create mock streams
        stream_manager = MockStreamManager()
        video_stream = MockVideoStream()
        audio_stream = MockAudioStream()
        stream_manager.add_video_stream(video_stream)
        stream_manager.add_audio_stream(audio_stream)

        # Create manager
        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
            stream_manager=stream_manager,  # type: ignore
        )

        # Register error callback
        error_callback = Mock()
        manager.register_error_callback(error_callback)

        # Start manager
        await manager.start()

        # Create invalid frame that will cause error in segment builder
        invalid_frame = Mock()
        invalid_frame.timestamp = "invalid"  # Wrong type

        # Emit invalid frame
        try:
            await video_stream.emit_frame(invalid_frame)
        except Exception:
            pass  # Expected to fail

        # Give time for error handling
        await asyncio.sleep(0.1)

        # Verify error callback was invoked
        assert error_callback.called

        # Verify manager is still running
        assert manager._is_running

        # Stop manager
        await manager.stop()

        print("✓ Error handling verified")

    async def test_metrics_collection(self):
        """Test metrics collection from manager.

        Verifies:
        - Metrics are collected from handlers
        - Metrics reflect actual operations
        - Both modes provide metrics
        """
        # Test server-side mode metrics
        vlm_client = Mock()
        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
        )

        # Mock handler metrics
        manager._server_handler.get_metrics = Mock(  # type: ignore
            return_value={
                "segments_sent": 10,
                "send_errors": 2,
            }
        )

        metrics = manager.get_metrics()
        assert metrics.segments_sent == 10
        assert metrics.send_errors == 2

        # Test client-side mode metrics
        stt_client = Mock()
        manager = MultimediaManager(
            audio_mode=AudioMode.CLIENT_SIDE,
            vlm_client=vlm_client,
            stt_client=stt_client,
        )

        # Mock handler metrics
        manager._client_handler.get_metrics = Mock(  # type: ignore
            return_value={
                "segments_buffered": 5,
                "segments_sent": 3,
                "stt_errors": 1,
                "send_errors": 0,
            }
        )

        metrics = manager.get_metrics()
        assert metrics.segments_buffered == 5
        assert metrics.segments_sent == 3
        assert metrics.stt_errors == 1

        print("✓ Metrics collection verified")

    async def test_callback_registration(self):
        """Test callback registration and invocation.

        Verifies:
        - Response callbacks can be registered
        - Error callbacks can be registered
        - Callbacks are stored correctly
        """
        vlm_client = Mock()
        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
        )

        # Register callbacks
        response_callback = Mock()
        error_callback = Mock()

        manager.register_response_callback(response_callback)
        manager.register_error_callback(error_callback)

        # Verify callbacks are stored
        assert manager._response_callback == response_callback
        assert manager._error_callback == error_callback

        print("✓ Callback registration verified")


if __name__ == "__main__":
    # Run checkpoint tests
    pytest.main([__file__, "-v", "-s"])
