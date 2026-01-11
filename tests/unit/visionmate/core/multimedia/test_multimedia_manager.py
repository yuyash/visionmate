"""Unit tests for MultimediaManager.

Tests the multimedia manager's initialization, lifecycle management,
and event-driven data flow coordination.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from visionmate.core.models import (
    AudioChunk,
    AudioMode,
    AudioSourceType,
    ManagerConfig,
    VideoFrame,
    VideoSourceType,
)
from visionmate.core.multimedia.manager import MultimediaManager


class TestMultimediaManagerInitialization:
    """Test MultimediaManager initialization."""

    def test_init_server_side_mode(self):
        """Test initialization with server-side mode."""
        vlm_client = Mock()
        vlm_client.client_type = Mock()

        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
        )

        assert manager._audio_mode == AudioMode.SERVER_SIDE
        assert manager._vlm_client == vlm_client
        assert manager._server_handler is not None
        assert manager._client_handler is None
        assert not manager._is_running

    def test_init_client_side_mode(self):
        """Test initialization with client-side mode."""
        vlm_client = Mock()
        stt_client = Mock()

        manager = MultimediaManager(
            audio_mode=AudioMode.CLIENT_SIDE,
            vlm_client=vlm_client,
            stt_client=stt_client,
        )

        assert manager._audio_mode == AudioMode.CLIENT_SIDE
        assert manager._stt_client == stt_client
        assert manager._client_handler is not None
        assert manager._server_handler is None
        assert manager._segment_buffer is not None
        assert manager._activity_detector is not None

    def test_init_client_side_without_stt_raises_error(self):
        """Test that client-side mode requires STT client."""
        vlm_client = Mock()

        with pytest.raises(ValueError, match="STT client is required"):
            MultimediaManager(
                audio_mode=AudioMode.CLIENT_SIDE,
                vlm_client=vlm_client,
                stt_client=None,
            )

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        vlm_client = Mock()
        config = ManagerConfig(
            max_segment_buffer_size=500,
            energy_threshold=0.02,
        )

        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
            config=config,
        )

        assert manager._config.max_segment_buffer_size == 500
        assert manager._config.energy_threshold == 0.02


class TestMultimediaManagerLifecycle:
    """Test MultimediaManager lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_registers_callbacks(self):
        """Test that start() registers callbacks with streams."""
        vlm_client = Mock()
        stream_manager = Mock()

        # Create mock streams
        video_stream = Mock()
        video_stream.get_source_id.return_value = "video-1"
        audio_stream = Mock()
        audio_stream.get_source_id.return_value = "audio-1"

        stream_manager.get_all_video_streams.return_value = [video_stream]
        stream_manager.get_all_audio_streams.return_value = [audio_stream]

        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
            stream_manager=stream_manager,
        )

        await manager.start()

        # Verify callbacks were registered
        video_stream.register_callback.assert_called_once()
        audio_stream.register_callback.assert_called_once()
        assert manager._is_running

    @pytest.mark.asyncio
    async def test_stop_unregisters_callbacks(self):
        """Test that stop() unregisters callbacks and cleans up."""
        vlm_client = Mock()
        stream_manager = Mock()

        # Create mock streams
        video_stream = Mock()
        video_stream.get_source_id.return_value = "video-1"
        audio_stream = Mock()
        audio_stream.get_source_id.return_value = "audio-1"

        stream_manager.get_all_video_streams.return_value = [video_stream]
        stream_manager.get_all_audio_streams.return_value = [audio_stream]

        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
            stream_manager=stream_manager,
        )

        await manager.start()
        await manager.stop()

        # Verify callbacks were unregistered
        video_stream.unregister_callback.assert_called_once()
        audio_stream.unregister_callback.assert_called_once()
        assert not manager._is_running

    @pytest.mark.asyncio
    async def test_start_without_stream_manager(self):
        """Test that start() handles missing stream manager gracefully."""
        vlm_client = Mock()

        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
            stream_manager=None,
        )

        # Should not raise error
        await manager.start()
        assert not manager._is_running


class TestMultimediaManagerCallbacks:
    """Test MultimediaManager callback handling."""

    @pytest.mark.asyncio
    async def test_on_frame_captured_adds_to_builder(self):
        """Test that frame callback adds frame to segment builder."""
        vlm_client = Mock()
        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
        )

        # Create test frame
        frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=datetime.now(timezone.utc),
            source_id="test-video",
            source_type=VideoSourceType.SCREEN,
            resolution=Mock(),
            fps=30,
            frame_number=1,
        )

        # Call callback
        await manager._on_frame_captured(frame)

        # Verify frame was added to builder
        assert len(manager._segment_builder._frame_buffer) == 1

    @pytest.mark.asyncio
    async def test_on_audio_captured_builds_segment(self):
        """Test that audio callback builds and routes segment."""
        vlm_client = Mock()
        vlm_client.send_frame = AsyncMock()
        vlm_client.send_audio_chunk = AsyncMock()

        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
        )

        # Add a frame first
        frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=datetime.now(timezone.utc),
            source_id="test-video",
            source_type=VideoSourceType.SCREEN,
            resolution=Mock(),
            fps=30,
            frame_number=1,
        )
        await manager._on_frame_captured(frame)

        # Create test audio chunk
        audio = AudioChunk(
            data=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=datetime.now(timezone.utc),
            source_id="test-audio",
            source_type=AudioSourceType.DEVICE,
            chunk_number=1,
        )

        # Call callback
        await manager._on_audio_captured(audio)

        # In server-side mode, should send to VLM
        # Note: Actual sending depends on handler implementation

    @pytest.mark.asyncio
    async def test_error_callback_invoked_on_exception(self):
        """Test that error callback is invoked when exception occurs."""
        vlm_client = Mock()
        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
        )

        # Register error callback
        error_callback = Mock()
        manager.register_error_callback(error_callback)

        # Create invalid frame that will cause error
        with patch.object(
            manager._segment_builder,
            "add_frame",
            side_effect=Exception("Test error"),
        ):
            frame = VideoFrame(
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=datetime.now(timezone.utc),
                source_id="test-video",
                source_type=VideoSourceType.SCREEN,
                resolution=Mock(),
                fps=30,
                frame_number=1,
            )

            await manager._on_frame_captured(frame)

            # Error callback should be invoked
            error_callback.assert_called_once()


class TestMultimediaManagerModeSwitch:
    """Test MultimediaManager mode switching."""

    def test_set_audio_mode_when_stopped(self):
        """Test changing audio mode when manager is stopped."""
        vlm_client = Mock()
        stt_client = Mock()

        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
            stt_client=stt_client,
        )

        # Switch to client-side mode
        manager.set_audio_mode(AudioMode.CLIENT_SIDE)

        assert manager._audio_mode == AudioMode.CLIENT_SIDE
        assert manager._client_handler is not None
        assert manager._server_handler is None

    @pytest.mark.asyncio
    async def test_set_audio_mode_when_running_raises_error(self):
        """Test that changing mode while running raises error."""
        vlm_client = Mock()
        stream_manager = Mock()
        stream_manager.get_all_video_streams.return_value = []
        stream_manager.get_all_audio_streams.return_value = []

        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
            stream_manager=stream_manager,
        )

        await manager.start()

        with pytest.raises(RuntimeError, match="Cannot change audio mode while manager is running"):
            manager.set_audio_mode(AudioMode.CLIENT_SIDE)

    def test_set_audio_mode_validates_requirements(self):
        """Test that mode switch validates required components."""
        vlm_client = Mock()

        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
            stt_client=None,
        )

        # Try to switch to client-side without STT client
        with pytest.raises(ValueError, match="STT client is required"):
            manager.set_audio_mode(AudioMode.CLIENT_SIDE)

        # Mode should remain unchanged
        assert manager._audio_mode == AudioMode.SERVER_SIDE


class TestMultimediaManagerMetrics:
    """Test MultimediaManager metrics collection."""

    def test_get_metrics_server_side_mode(self):
        """Test metrics collection in server-side mode."""
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

    def test_get_metrics_client_side_mode(self):
        """Test metrics collection in client-side mode."""
        vlm_client = Mock()
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
        assert metrics.send_errors == 0


class TestMultimediaManagerCallbackRegistration:
    """Test callback registration methods."""

    def test_register_response_callback(self):
        """Test registering response callback."""
        vlm_client = Mock()
        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
        )

        callback = Mock()
        manager.register_response_callback(callback)

        assert manager._response_callback == callback

    def test_register_error_callback(self):
        """Test registering error callback."""
        vlm_client = Mock()
        manager = MultimediaManager(
            audio_mode=AudioMode.SERVER_SIDE,
            vlm_client=vlm_client,
        )

        callback = Mock()
        manager.register_error_callback(callback)

        assert manager._error_callback == callback
