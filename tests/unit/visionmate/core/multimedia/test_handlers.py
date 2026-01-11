"""Unit tests for multimedia handlers.

Tests for ServerSideHandler and ClientSideHandler implementations.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from visionmate.core.models import (
    AudioChunk,
    AudioSourceType,
    ManagerConfig,
    MultimediaSegment,
    Resolution,
    VideoFrame,
    VideoSourceType,
)
from visionmate.core.multimedia.handlers import ServerSideHandler


class TestServerSideHandler:
    """Test ServerSideHandler implementation."""

    @pytest.fixture
    def mock_vlm_client(self):
        """Create mock streaming VLM client."""
        client = MagicMock()
        client.send_frame = AsyncMock()
        client.send_audio_chunk = AsyncMock()
        return client

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ManagerConfig(
            max_send_buffer_size=10,
            enable_backpressure=True,
            max_retry_attempts=3,
            retry_backoff_base=0.1,  # Fast retries for testing
        )

    @pytest.fixture
    def sample_segment(self):
        """Create sample multimedia segment."""
        # Create video frame
        frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=datetime.now(timezone.utc),
            source_id="test_video",
            source_type=VideoSourceType.UVC,
            resolution=Resolution(width=640, height=480),
            fps=30,
            frame_number=1,
        )

        # Create audio chunk
        audio = AudioChunk(
            data=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=datetime.now(timezone.utc),
            source_id="test_audio",
            source_type=AudioSourceType.DEVICE,
            chunk_number=1,
        )

        # Create segment
        start_time = datetime.now(timezone.utc)
        return MultimediaSegment(
            audio=audio,
            video_frames=[frame],
            start_time=start_time,
            end_time=start_time,
            source_id="test",
        )

    def test_init(self, mock_vlm_client, config):
        """Test ServerSideHandler initialization."""
        handler = ServerSideHandler(mock_vlm_client, config)

        assert handler._vlm_client == mock_vlm_client
        assert handler._config == config
        assert len(handler._send_buffer) == 0
        assert handler._segments_sent == 0
        assert handler._send_errors == 0

    def test_init_default_config(self, mock_vlm_client):
        """Test initialization with default config."""
        handler = ServerSideHandler(mock_vlm_client)

        assert handler._config is not None
        assert isinstance(handler._config, ManagerConfig)

    @pytest.mark.asyncio
    async def test_send_segment_success(self, mock_vlm_client, config, sample_segment):
        """Test successful segment sending."""
        handler = ServerSideHandler(mock_vlm_client, config)

        await handler.send_segment(sample_segment)

        # Verify frame was sent
        assert mock_vlm_client.send_frame.call_count == len(sample_segment.video_frames)
        mock_vlm_client.send_frame.assert_called_with(sample_segment.video_frames[0])

        # Verify audio was sent
        mock_vlm_client.send_audio_chunk.assert_called_once_with(sample_segment.audio)

        # Verify buffer is empty after successful send
        assert handler.get_buffer_size() == 0
        assert handler._segments_sent == 1

    @pytest.mark.asyncio
    async def test_send_segment_with_retry(self, mock_vlm_client, config, sample_segment):
        """Test segment sending with retry on transient failure."""
        handler = ServerSideHandler(mock_vlm_client, config)

        # Simulate transient failure then success
        mock_vlm_client.send_frame.side_effect = [
            ConnectionError("Network error"),
            None,  # Success on retry
        ]

        await handler.send_segment(sample_segment)

        # Verify retry occurred
        assert mock_vlm_client.send_frame.call_count == 2
        assert handler._segments_sent == 1
        assert handler._send_errors == 0

    @pytest.mark.asyncio
    async def test_send_segment_max_retries_exceeded(self, mock_vlm_client, config, sample_segment):
        """Test segment sending fails after max retries."""
        handler = ServerSideHandler(mock_vlm_client, config)

        # Simulate persistent failure
        mock_vlm_client.send_frame.side_effect = ConnectionError("Network error")

        with pytest.raises(ConnectionError, match="Failed to send segment after .* retries"):
            await handler.send_segment(sample_segment)

        # Verify max retries were attempted
        assert mock_vlm_client.send_frame.call_count == config.max_retry_attempts + 1
        assert handler._send_errors == 1
        # Buffer should be empty (failed segment removed)
        assert handler.get_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_backpressure_handling(self, mock_vlm_client, config, sample_segment):
        """Test backpressure handling when buffer is full."""
        # Configure small buffer for testing
        config.max_send_buffer_size = 2
        handler = ServerSideHandler(mock_vlm_client, config)

        # Create a future that never completes to block sending
        never_complete = asyncio.Future()

        # Block sending to fill buffer
        mock_vlm_client.send_frame.return_value = never_complete

        # Add segments until buffer is full
        tasks = []
        for _ in range(3):  # Try to add more than buffer size
            task = asyncio.create_task(handler.send_segment(sample_segment))
            tasks.append(task)
            await asyncio.sleep(0.01)  # Let task start

        # Buffer should be at capacity (oldest dropped)
        assert handler.get_buffer_size() <= config.max_send_buffer_size

        # Cancel tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def test_get_buffer_size(self, mock_vlm_client, config):
        """Test get_buffer_size method."""
        handler = ServerSideHandler(mock_vlm_client, config)

        assert handler.get_buffer_size() == 0

        # Add segments to buffer (without processing)
        handler._send_buffer.append(MagicMock())
        assert handler.get_buffer_size() == 1

        handler._send_buffer.append(MagicMock())
        assert handler.get_buffer_size() == 2

    def test_get_metrics(self, mock_vlm_client, config):
        """Test get_metrics method."""
        handler = ServerSideHandler(mock_vlm_client, config)

        metrics = handler.get_metrics()

        assert "segments_sent" in metrics
        assert "send_errors" in metrics
        assert "buffer_size" in metrics
        assert "buffer_capacity" in metrics

        assert metrics["segments_sent"] == 0
        assert metrics["send_errors"] == 0
        assert metrics["buffer_size"] == 0
        assert metrics["buffer_capacity"] == config.max_send_buffer_size

    @pytest.mark.asyncio
    async def test_multiple_frames_per_segment(self, mock_vlm_client, config):
        """Test sending segment with multiple video frames."""
        handler = ServerSideHandler(mock_vlm_client, config)

        # Create segment with multiple frames
        frames = [
            VideoFrame(
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=datetime.now(timezone.utc),
                source_id="test_video",
                source_type=VideoSourceType.UVC,
                resolution=Resolution(width=640, height=480),
                fps=30,
                frame_number=i,
            )
            for i in range(3)
        ]

        audio = AudioChunk(
            data=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=datetime.now(timezone.utc),
            source_id="test_audio",
            source_type=AudioSourceType.DEVICE,
            chunk_number=1,
        )

        start_time = datetime.now(timezone.utc)
        segment = MultimediaSegment(
            audio=audio,
            video_frames=frames,
            start_time=start_time,
            end_time=start_time,
            source_id="test",
        )

        await handler.send_segment(segment)

        # Verify all frames were sent
        assert mock_vlm_client.send_frame.call_count == 3
        mock_vlm_client.send_audio_chunk.assert_called_once()


class TestClientSideHandler:
    """Test ClientSideHandler implementation."""

    @pytest.fixture
    def mock_vlm_client(self):
        """Create mock VLM client."""
        client = MagicMock()
        client.send_frame = AsyncMock()
        client.send_text = AsyncMock()
        client.process_multimodal_input = AsyncMock()
        return client

    @pytest.fixture
    def mock_stt_client(self):
        """Create mock STT client."""
        client = MagicMock()
        client.transcribe = AsyncMock(return_value="test transcription")
        client.get_provider = MagicMock(return_value="test_provider")
        return client

    @pytest.fixture
    def segment_buffer(self):
        """Create segment buffer manager."""
        from visionmate.core.multimedia.buffer import SegmentBufferManager

        return SegmentBufferManager(max_capacity=10, max_memory_mb=100)

    @pytest.fixture
    def activity_detector(self):
        """Create audio activity detector."""
        from visionmate.core.multimedia.detector import AudioActivityDetector

        return AudioActivityDetector(
            energy_threshold=0.01,
            silence_duration_sec=0.5,  # Short for testing
        )

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ManagerConfig(
            max_segment_buffer_size=10,
            max_retry_attempts=3,
            retry_backoff_base=0.1,  # Fast retries for testing
        )

    @pytest.fixture
    def sample_segment(self):
        """Create sample multimedia segment with speech energy."""
        # Create video frame
        frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=datetime.now(timezone.utc),
            source_id="test_video",
            source_type=VideoSourceType.UVC,
            resolution=Resolution(width=640, height=480),
            fps=30,
            frame_number=1,
        )

        # Create audio chunk with speech-level energy
        audio_data = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
        audio = AudioChunk(
            data=audio_data,
            sample_rate=16000,
            channels=1,
            timestamp=datetime.now(timezone.utc),
            source_id="test_audio",
            source_type=AudioSourceType.DEVICE,
            chunk_number=1,
        )

        # Create segment
        start_time = datetime.now(timezone.utc)
        return MultimediaSegment(
            audio=audio,
            video_frames=[frame],
            start_time=start_time,
            end_time=start_time,
            source_id="test",
        )

    @pytest.fixture
    def silence_segment(self):
        """Create sample multimedia segment with silence."""
        # Create video frame
        frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=datetime.now(timezone.utc),
            source_id="test_video",
            source_type=VideoSourceType.UVC,
            resolution=Resolution(width=640, height=480),
            fps=30,
            frame_number=1,
        )

        # Create audio chunk with low energy (silence)
        audio_data = np.zeros(16000, dtype=np.float32)
        audio = AudioChunk(
            data=audio_data,
            sample_rate=16000,
            channels=1,
            timestamp=datetime.now(timezone.utc),
            source_id="test_audio",
            source_type=AudioSourceType.DEVICE,
            chunk_number=1,
        )

        # Create segment
        start_time = datetime.now(timezone.utc)
        return MultimediaSegment(
            audio=audio,
            video_frames=[frame],
            start_time=start_time,
            end_time=start_time,
            source_id="test",
        )

    def _create_silence_segment_at_time(self, timestamp: datetime):
        """Helper to create silence segment with specific timestamp."""
        frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=timestamp,
            source_id="test_video",
            source_type=VideoSourceType.UVC,
            resolution=Resolution(width=640, height=480),
            fps=30,
            frame_number=1,
        )

        audio_data = np.zeros(16000, dtype=np.float32)
        audio = AudioChunk(
            data=audio_data,
            sample_rate=16000,
            channels=1,
            timestamp=timestamp,
            source_id="test_audio",
            source_type=AudioSourceType.DEVICE,
            chunk_number=1,
        )

        return MultimediaSegment(
            audio=audio,
            video_frames=[frame],
            start_time=timestamp,
            end_time=timestamp,
            source_id="test",
        )

    def test_init(
        self, mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
    ):
        """Test ClientSideHandler initialization."""
        from visionmate.core.multimedia.handlers import ClientSideHandler

        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
        )

        assert handler._vlm_client == mock_vlm_client
        assert handler._stt_client == mock_stt_client
        assert handler._segment_buffer == segment_buffer
        assert handler._activity_detector == activity_detector
        assert handler._config == config
        assert not handler._is_buffering
        assert handler._speech_start_time is None

    def test_init_default_config(
        self, mock_vlm_client, mock_stt_client, segment_buffer, activity_detector
    ):
        """Test initialization with default config."""
        from visionmate.core.multimedia.handlers import ClientSideHandler

        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector
        )

        assert handler._config is not None
        assert isinstance(handler._config, ManagerConfig)

    @pytest.mark.asyncio
    async def test_process_segment_speech_detected(
        self,
        mock_vlm_client,
        mock_stt_client,
        segment_buffer,
        activity_detector,
        config,
        sample_segment,
    ):
        """Test segment buffering when speech is detected."""
        from visionmate.core.multimedia.handlers import ClientSideHandler

        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
        )

        # Process segment with speech
        await handler.process_segment(sample_segment)

        # Verify buffering started
        assert handler._is_buffering
        assert handler._speech_start_time is not None

        # Verify segment was buffered
        assert segment_buffer.get_size() == 1
        assert handler._segments_buffered == 1

        # Verify no STT or VLM calls yet
        mock_stt_client.transcribe.assert_not_called()
        mock_vlm_client.send_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_segment_speech_ended(
        self,
        mock_vlm_client,
        mock_stt_client,
        segment_buffer,
        activity_detector,
        config,
        sample_segment,
        silence_segment,
    ):
        """Test transcription and sending when speech ends."""
        from datetime import timedelta

        from visionmate.core.multimedia.handlers import ClientSideHandler

        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
        )

        # Process speech segment to start buffering
        await handler.process_segment(sample_segment)
        assert handler._is_buffering
        assert segment_buffer.get_size() == 1

        # Process silence segments to trigger speech end
        # Need to process enough silence to exceed silence_duration_sec (0.5s in fixture)
        # Create silence segments with incrementing timestamps
        base_time = sample_segment.start_time
        for i in range(10):
            # Each segment is 0.1s apart (100ms), need at least 500ms total
            silence_time = base_time + timedelta(milliseconds=(i + 1) * 100)
            silence_seg = self._create_silence_segment_at_time(silence_time)
            await handler.process_segment(silence_seg)

        # Verify STT was called
        mock_stt_client.transcribe.assert_called_once()

        # Verify VLM was called (either streaming or request-response)
        assert mock_vlm_client.send_frame.called or mock_vlm_client.process_multimodal_input.called

        # Verify buffer was cleared
        assert segment_buffer.get_size() == 0
        assert not handler._is_buffering

    @pytest.mark.asyncio
    async def test_process_segment_silence_timeout(
        self,
        mock_vlm_client,
        mock_stt_client,
        segment_buffer,
        activity_detector,
        config,
        sample_segment,
        silence_segment,
    ):
        """Test buffer clearing on silence timeout."""
        from datetime import timedelta

        from visionmate.core.multimedia.handlers import ClientSideHandler

        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
        )

        # Process speech segment to start buffering
        await handler.process_segment(sample_segment)
        assert handler._is_buffering
        initial_buffer_size = segment_buffer.get_size()
        assert initial_buffer_size > 0

        # Process silence segments to trigger speech end (which will process and clear buffer)
        base_time = sample_segment.start_time
        for i in range(10):
            silence_time = base_time + timedelta(milliseconds=(i + 1) * 100)
            silence_seg = self._create_silence_segment_at_time(silence_time)
            await handler.process_segment(silence_seg)

        # Verify buffer was cleared (speech ended triggers processing and clearing)
        assert segment_buffer.get_size() == 0
        assert not handler._is_buffering

        # Note: Speech ended will trigger STT and VLM processing, then clear buffer
        # This is the expected behavior - silence timeout means speech ended

    @pytest.mark.asyncio
    async def test_stt_failure_handling(
        self,
        mock_vlm_client,
        mock_stt_client,
        segment_buffer,
        activity_detector,
        config,
        sample_segment,
        silence_segment,
    ):
        """Test error handling when STT fails."""
        from datetime import timedelta

        from visionmate.core.multimedia.handlers import ClientSideHandler

        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
        )

        # Configure STT to fail
        mock_stt_client.transcribe.side_effect = Exception("STT service unavailable")

        # Process speech segment
        await handler.process_segment(sample_segment)
        assert handler._is_buffering

        # Process silence to trigger speech end
        base_time = sample_segment.start_time
        for i in range(10):
            silence_time = base_time + timedelta(milliseconds=(i + 1) * 100)
            silence_seg = self._create_silence_segment_at_time(silence_time)
            await handler.process_segment(silence_seg)

        # Verify STT error was counted
        assert handler._stt_errors > 0

        # Verify VLM was not called (STT failure prevents sending)
        mock_vlm_client.send_frame.assert_not_called()
        mock_vlm_client.process_multimodal_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_vlm_failure_with_retry(
        self,
        mock_vlm_client,
        mock_stt_client,
        segment_buffer,
        activity_detector,
        config,
        sample_segment,
        silence_segment,
    ):
        """Test VLM failure handling with retry."""
        from datetime import timedelta

        from visionmate.core.multimedia.handlers import ClientSideHandler

        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
        )

        # Configure VLM to fail once then succeed
        # The handler checks for process_multimodal_input first, so configure that
        mock_vlm_client.process_multimodal_input.side_effect = [
            ConnectionError("Network error"),
            AsyncMock(return_value=None),  # Success on retry
        ]

        # Process speech segment
        await handler.process_segment(sample_segment)

        # Process silence to trigger speech end
        base_time = sample_segment.start_time
        for i in range(10):
            silence_time = base_time + timedelta(milliseconds=(i + 1) * 100)
            silence_seg = self._create_silence_segment_at_time(silence_time)
            await handler.process_segment(silence_seg)

        # Verify retry occurred
        assert mock_vlm_client.process_multimodal_input.call_count >= 2

    @pytest.mark.asyncio
    async def test_vlm_failure_max_retries(
        self,
        mock_vlm_client,
        mock_stt_client,
        segment_buffer,
        activity_detector,
        config,
        sample_segment,
        silence_segment,
    ):
        """Test VLM failure after max retries."""
        from datetime import timedelta

        from visionmate.core.multimedia.handlers import ClientSideHandler

        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
        )

        # Configure VLM to always fail
        # The handler checks for process_multimodal_input first, so configure that
        mock_vlm_client.process_multimodal_input.side_effect = ConnectionError("Network error")

        # Process speech segment
        await handler.process_segment(sample_segment)

        # Process silence to trigger speech end (should raise error)
        with pytest.raises(ConnectionError):
            base_time = sample_segment.start_time
            for i in range(10):
                silence_time = base_time + timedelta(milliseconds=(i + 1) * 100)
                silence_seg = self._create_silence_segment_at_time(silence_time)
                await handler.process_segment(silence_seg)

        # Verify max retries were attempted
        assert mock_vlm_client.process_multimodal_input.call_count >= config.max_retry_attempts

        # Verify error was counted
        assert handler._send_errors > 0

    @pytest.mark.asyncio
    async def test_multiple_segments_buffering(
        self,
        mock_vlm_client,
        mock_stt_client,
        segment_buffer,
        activity_detector,
        config,
        sample_segment,
    ):
        """Test buffering multiple segments during speech."""
        from visionmate.core.multimedia.handlers import ClientSideHandler

        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
        )

        # Process multiple speech segments
        num_segments = 5
        for _ in range(num_segments):
            await handler.process_segment(sample_segment)
            await asyncio.sleep(0.05)

        # Verify all segments were buffered
        assert segment_buffer.get_size() == num_segments
        assert handler._segments_buffered == num_segments
        assert handler._is_buffering

    def test_get_metrics(
        self, mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
    ):
        """Test get_metrics method."""
        from visionmate.core.multimedia.handlers import ClientSideHandler

        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
        )

        metrics = handler.get_metrics()

        assert "segments_buffered" in metrics
        assert "segments_sent" in metrics
        assert "stt_errors" in metrics
        assert "send_errors" in metrics
        assert "buffer_size" in metrics
        assert "buffer_capacity" in metrics
        assert "is_buffering" in metrics

        assert metrics["segments_buffered"] == 0
        assert metrics["segments_sent"] == 0
        assert metrics["stt_errors"] == 0
        assert metrics["send_errors"] == 0
        assert metrics["is_buffering"] is False

    @pytest.mark.asyncio
    async def test_request_response_vlm_client(
        self,
        mock_vlm_client,
        mock_stt_client,
        segment_buffer,
        activity_detector,
        config,
        sample_segment,
        silence_segment,
    ):
        """Test with request-response VLM client."""
        from datetime import timedelta

        from visionmate.core.multimedia.handlers import ClientSideHandler

        # Configure as request-response client (has process_multimodal_input)
        handler = ClientSideHandler(
            mock_vlm_client, mock_stt_client, segment_buffer, activity_detector, config
        )

        # Process speech segment
        await handler.process_segment(sample_segment)

        # Process silence to trigger speech end
        base_time = sample_segment.start_time
        for i in range(10):
            silence_time = base_time + timedelta(milliseconds=(i + 1) * 100)
            silence_seg = self._create_silence_segment_at_time(silence_time)
            await handler.process_segment(silence_seg)

        # Verify request-response method was called
        assert mock_vlm_client.process_multimodal_input.called or mock_vlm_client.send_frame.called
