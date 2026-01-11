"""Multimedia manager for orchestrating event-driven data flow.

This module implements the MultimediaManager which coordinates event-driven
data flow between capture sources and VLM clients based on audio processing mode.
"""

import logging
from typing import Callable, Optional

from visionmate.core.capture.stream import AudioStream, StreamManager, VideoStream
from visionmate.core.models import (
    AudioChunk,
    AudioMode,
    ManagerConfig,
    ManagerMetrics,
    VideoFrame,
)
from visionmate.core.multimedia.buffer import SegmentBufferManager
from visionmate.core.multimedia.detector import AudioActivityDetector
from visionmate.core.multimedia.handlers import ClientSideHandler, ServerSideHandler
from visionmate.core.multimedia.segment import MultimediaSegmentBuilder
from visionmate.core.recognition import (
    SpeechToTextInterface,
    StreamingVLMClient,
    VLMClientInterface,
    VLMResponse,
)

logger = logging.getLogger(__name__)


class MultimediaManager:
    """Manages multimedia data flow to VLM based on audio mode.

    This component manages the lifecycle of data transmission to the VLM,
    switching between streaming and buffered modes based on configuration.
    Uses event-driven architecture with callbacks for all operations.

    """

    def __init__(
        self,
        audio_mode: AudioMode,
        vlm_client: VLMClientInterface,
        stt_client: Optional[SpeechToTextInterface] = None,
        stream_manager: Optional[StreamManager] = None,
        config: Optional[ManagerConfig] = None,
    ):
        """Initialize multimedia manager.

        Args:
            audio_mode: Audio processing mode (SERVER_SIDE or CLIENT_SIDE)
            vlm_client: VLM client for sending data
            stt_client: STT client for client-side mode (required if CLIENT_SIDE)
            stream_manager: Stream manager for accessing video/audio streams
            config: Configuration parameters

        Raises:
            ValueError: If required components are missing for the selected mode

        """
        self._audio_mode = audio_mode
        self._vlm_client = vlm_client
        self._stt_client = stt_client
        self._stream_manager = stream_manager
        self._config = config or ManagerConfig()

        # Validate required components for audio mode
        self._validate_mode_requirements()

        # Initialize segment builder
        self._segment_builder = MultimediaSegmentBuilder(
            frame_selection_strategy=self._config.frame_selection_strategy,
            max_time_drift_ms=self._config.max_time_drift_ms,
            change_threshold=self._config.change_threshold,
            max_frames_per_segment=self._config.max_frames_per_segment,
        )

        # Initialize components based on mode
        self._segment_buffer: Optional[SegmentBufferManager] = None
        self._activity_detector: Optional[AudioActivityDetector] = None
        self._server_handler: Optional[ServerSideHandler] = None
        self._client_handler: Optional[ClientSideHandler] = None

        if self._audio_mode in (AudioMode.SERVER_SIDE, AudioMode.DIRECT):
            # Server-side mode: streaming handler
            if not isinstance(vlm_client, StreamingVLMClient):
                logger.warning(
                    "Server-side mode works best with StreamingVLMClient, "
                    f"but got {type(vlm_client).__name__}"
                )
            self._server_handler = ServerSideHandler(
                vlm_client=vlm_client,  # type: ignore[arg-type]
                config=self._config,
                error_callback=self._propagate_error,
            )
        else:
            # Client-side mode: buffering handler
            self._segment_buffer = SegmentBufferManager(
                max_capacity=self._config.max_segment_buffer_size,
                max_memory_mb=self._config.max_buffer_memory_mb,
            )
            self._activity_detector = AudioActivityDetector(
                energy_threshold=self._config.energy_threshold,
                silence_duration_sec=self._config.silence_duration_sec,
            )
            self._client_handler = ClientSideHandler(
                vlm_client=vlm_client,
                stt_client=stt_client,  # type: ignore[arg-type]
                segment_buffer=self._segment_buffer,
                activity_detector=self._activity_detector,
                config=self._config,
                error_callback=self._propagate_error,
            )

        # Callbacks
        self._response_callback: Optional[Callable[[VLMResponse], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None

        # State tracking
        self._is_running = False
        self._registered_streams: list[tuple[VideoStream | AudioStream, Callable]] = []

        logger.info(
            f"MultimediaManager initialized with mode={audio_mode.value}, "
            f"frame_strategy={self._config.frame_selection_strategy.value}"
        )

    def _validate_mode_requirements(self) -> None:
        """Validate that required components are available for the selected mode.

        Raises:
            ValueError: If required components are missing

        """
        if self._audio_mode in (AudioMode.CLIENT_SIDE, AudioMode.TEXT):
            if self._stt_client is None:
                raise ValueError(f"STT client is required for {self._audio_mode.value} mode")

    async def start(self) -> None:
        """Start managing data flow.

        Registers callbacks with video and audio streams to receive
        frames and chunks as they are captured. Also connects streaming
        VLM clients if needed.

        """
        if self._is_running:
            logger.warning("MultimediaManager is already running")
            return

        if self._stream_manager is None:
            logger.warning("No stream manager provided, cannot register callbacks")
            return

        logger.info("Starting MultimediaManager")

        # Connect streaming VLM client if needed
        from visionmate.core.recognition import StreamingVLMClient

        if isinstance(self._vlm_client, StreamingVLMClient):
            try:
                await self._vlm_client.connect()
                logger.info("Connected to streaming VLM client")
            except Exception as e:
                logger.error(f"Failed to connect to streaming VLM client: {e}", exc_info=True)
                raise

        # Register callbacks with all video streams
        video_streams = self._stream_manager.get_all_video_streams()
        for stream in video_streams:
            # Create sync wrapper for async callback with proper closure
            def make_video_callback():
                async def async_callback(frame: VideoFrame) -> None:
                    await self._on_frame_captured(frame)

                def sync_wrapper(frame: VideoFrame) -> None:
                    import asyncio

                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(async_callback(frame))
                        else:
                            loop.run_until_complete(async_callback(frame))
                    except RuntimeError:
                        # No event loop, create one
                        asyncio.run(async_callback(frame))

                return sync_wrapper

            callback = make_video_callback()
            stream.register_callback(callback)
            self._registered_streams.append((stream, callback))
            logger.debug(f"Registered video callback for stream: {stream.get_source_id()}")

        # Register callbacks with all audio streams
        audio_streams = self._stream_manager.get_all_audio_streams()
        for stream in audio_streams:
            # Create sync wrapper for async callback with proper closure
            def make_audio_callback():
                async def async_callback(audio: AudioChunk) -> None:
                    await self._on_audio_captured(audio)

                def sync_wrapper(audio: AudioChunk) -> None:
                    import asyncio

                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(async_callback(audio))
                        else:
                            loop.run_until_complete(async_callback(audio))
                    except RuntimeError:
                        # No event loop, create one
                        asyncio.run(async_callback(audio))

                return sync_wrapper

            callback = make_audio_callback()
            stream.register_callback(callback)
            self._registered_streams.append((stream, callback))
            logger.debug(f"Registered audio callback for stream: {stream.get_source_id()}")

        self._is_running = True
        logger.info(
            f"MultimediaManager started with {len(video_streams)} video "
            f"and {len(audio_streams)} audio streams"
        )

    async def stop(self) -> None:
        """Stop managing and cleanup resources.

        Unregisters all callbacks, disconnects VLM client, and cleans up buffers.

        """
        if not self._is_running:
            logger.warning("MultimediaManager is not running")
            return

        logger.info("Stopping MultimediaManager")

        # Disconnect streaming VLM client if needed
        from visionmate.core.recognition import StreamingVLMClient

        if isinstance(self._vlm_client, StreamingVLMClient):
            try:
                await self._vlm_client.disconnect()
                logger.info("Disconnected from streaming VLM client")
            except Exception as e:
                logger.warning(f"Error disconnecting from streaming VLM client: {e}")

        # Unregister all callbacks
        for stream, callback in self._registered_streams:
            stream.unregister_callback(callback)
            logger.debug(f"Unregistered callback for stream: {stream.get_source_id()}")

        self._registered_streams.clear()

        # Clear buffers
        if self._segment_buffer:
            cleared = self._segment_buffer.clear()
            logger.debug(f"Cleared {cleared} segments from buffer")

        # Reset activity detector
        if self._activity_detector:
            self._activity_detector.reset()

        self._is_running = False
        logger.info("MultimediaManager stopped")

    async def _on_frame_captured(self, frame: VideoFrame) -> None:
        """Callback when video frame is captured.

        Adds frame to segment builder for temporal correlation with audio.
        Propagates errors via error callback.

        Args:
            frame: Captured video frame

        """
        try:
            self._segment_builder.add_frame(frame)
            logger.debug(
                f"Added frame to segment builder: {frame.source_id} "
                f"at {frame.timestamp.isoformat()}"
            )
        except Exception as e:
            logger.error(f"Error in _on_frame_captured: {e}", exc_info=True)
            self._propagate_error(e)

    async def _on_audio_captured(self, audio: AudioChunk) -> None:
        """Callback when audio chunk is captured.

        Builds multimedia segment from audio chunk and routes to appropriate
        handler based on audio mode. Propagates errors via error callback.

        Args:
            audio: Captured audio chunk

        """
        try:
            # Build multimedia segment
            segment = self._segment_builder.build_segment(audio)

            if segment is None:
                logger.debug(
                    f"No suitable frames available for audio chunk at {audio.timestamp.isoformat()}"
                )
                return

            logger.debug(
                f"Built segment with {len(segment.video_frames)} frame(s) "
                f"for audio at {audio.timestamp.isoformat()}"
            )

            # Route to appropriate handler based on mode
            if self._audio_mode in (AudioMode.SERVER_SIDE, AudioMode.DIRECT):
                # Server-side mode: stream immediately
                if self._server_handler:
                    await self._server_handler.send_segment(segment)
            else:
                # Client-side mode: buffer and process based on activity
                if self._client_handler:
                    await self._client_handler.process_segment(segment)

        except Exception as e:
            logger.error(f"Error in _on_audio_captured: {e}", exc_info=True)
            self._propagate_error(e)

    def register_response_callback(
        self,
        callback: Callable[[VLMResponse], None],
    ) -> None:
        """Register callback for VLM responses.

        Args:
            callback: Function to call when response received

        """
        self._response_callback = callback
        logger.debug("Registered response callback")

    def register_error_callback(
        self,
        callback: Callable[[Exception], None],
    ) -> None:
        """Register callback for errors.

        Args:
            callback: Function to call when error occurs

        """
        self._error_callback = callback
        logger.debug("Registered error callback")

    def _propagate_error(self, error: Exception) -> None:
        """Propagate error to registered error callback.

        Logs the error and calls the registered error callback if available.
        This method is used by handlers to propagate errors up to the manager.

        Args:
            error: Exception to propagate

        """
        # Log the error appropriately based on type
        from visionmate.core.multimedia.handlers import RecoveryEvent

        if isinstance(error, RecoveryEvent):
            # Recovery event - log as info
            logger.info(f"Recovery event: {error}")
        elif isinstance(error, ConnectionError):
            # Connection error - log as warning (may be transient)
            logger.warning(f"Connection error: {error}")
        else:
            # Other errors - log as error
            logger.error(f"Error propagated from handler: {error}", exc_info=True)

        # Call registered error callback
        if self._error_callback:
            try:
                self._error_callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}", exc_info=True)

    def set_audio_mode(self, mode: AudioMode) -> None:
        """Change audio processing mode.

        Args:
            mode: New audio mode

        Raises:
            RuntimeError: If manager is running
            ValueError: If required components are missing for the new mode

        """
        if self._is_running:
            raise RuntimeError(
                "Cannot change audio mode while manager is running. Call stop() first."
            )

        # Store old mode for logging
        old_mode = self._audio_mode
        self._audio_mode = mode

        # Validate new mode requirements
        try:
            self._validate_mode_requirements()
        except ValueError:
            # Restore old mode on validation failure
            self._audio_mode = old_mode
            raise

        # Reinitialize handlers for new mode
        self._server_handler = None
        self._client_handler = None
        self._segment_buffer = None
        self._activity_detector = None

        if mode in (AudioMode.SERVER_SIDE, AudioMode.DIRECT):
            # Server-side mode
            self._server_handler = ServerSideHandler(
                vlm_client=self._vlm_client,  # type: ignore[arg-type]
                config=self._config,
                error_callback=self._propagate_error,
            )
        else:
            # Client-side mode
            self._segment_buffer = SegmentBufferManager(
                max_capacity=self._config.max_segment_buffer_size,
                max_memory_mb=self._config.max_buffer_memory_mb,
            )
            self._activity_detector = AudioActivityDetector(
                energy_threshold=self._config.energy_threshold,
                silence_duration_sec=self._config.silence_duration_sec,
            )
            self._client_handler = ClientSideHandler(
                vlm_client=self._vlm_client,
                stt_client=self._stt_client,  # type: ignore[arg-type]
                segment_buffer=self._segment_buffer,
                activity_detector=self._activity_detector,
                config=self._config,
                error_callback=self._propagate_error,
            )

        logger.info(f"Audio mode changed from {old_mode.value} to {mode.value}")

    def get_metrics(self) -> ManagerMetrics:
        """Get current metrics.

        Collects metrics from all components and returns aggregated metrics.

        Returns:
            Metrics object with buffer usage, throughput, errors, and latencies

        """
        metrics = ManagerMetrics()

        # Collect metrics based on active handler
        if self._server_handler:
            handler_metrics = self._server_handler.get_metrics()
            metrics.segments_sent = handler_metrics.get("segments_sent", 0)
            metrics.send_errors = handler_metrics.get("send_errors", 0)
            metrics.connection_errors = handler_metrics.get("connection_errors", 0)
            metrics.avg_segment_latency_ms = handler_metrics.get("avg_segment_latency_ms", 0.0)
            metrics.requests_sent = metrics.segments_sent  # Each segment is a request

        elif self._client_handler:
            handler_metrics = self._client_handler.get_metrics()
            metrics.segments_buffered = handler_metrics.get("segments_buffered", 0)
            metrics.segments_sent = handler_metrics.get("segments_sent", 0)
            metrics.stt_errors = handler_metrics.get("stt_errors", 0)
            metrics.send_errors = handler_metrics.get("send_errors", 0)
            metrics.connection_errors = handler_metrics.get("connection_errors", 0)
            metrics.avg_stt_duration_ms = handler_metrics.get("avg_stt_duration_ms", 0.0)
            # In client-side mode, requests are batched (one request per speech segment)
            # This is an approximation - actual count would need tracking in handler
            metrics.requests_sent = metrics.segments_sent

        # Collect buffer metrics
        if self._segment_buffer:
            metrics.buffer_memory_mb = self._segment_buffer.get_memory_usage_mb()
            metrics.segments_dropped = self._segment_buffer.get_dropped_count()

        return metrics
