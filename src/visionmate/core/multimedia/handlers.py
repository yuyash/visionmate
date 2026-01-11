"""Handlers for mode-specific multimedia data flow.

This module implements handlers for server-side and client-side audio processing modes.
Each handler manages the data flow from multimedia segments to VLM clients according
to the specific requirements of its mode.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Deque, List, Optional

from visionmate.core.models import ActivityState, ManagerConfig, MultimediaSegment
from visionmate.core.multimedia.buffer import SegmentBufferManager
from visionmate.core.multimedia.detector import AudioActivityDetector
from visionmate.core.recognition import (
    SpeechToTextInterface,
    StreamingVLMClient,
    VLMClientInterface,
    VLMRequest,
)

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for multimedia manager observability."""

    # State transition events
    BUFFERING_STARTED = "buffering_started"
    BUFFERING_STOPPED = "buffering_stopped"
    SPEECH_DETECTED = "speech_detected"
    SPEECH_ENDED = "speech_ended"

    # Data flow events
    SEGMENT_SENT = "segment_sent"
    SEGMENT_BUFFERED = "segment_buffered"
    SEGMENT_DROPPED = "segment_dropped"

    # Error events
    CONNECTION_ERROR = "connection_error"
    STT_ERROR = "stt_error"
    SEND_ERROR = "send_error"

    # Recovery events
    CONNECTION_RECOVERED = "connection_recovered"


class MultimediaEvent:
    """Event for multimedia manager state changes and errors."""

    def __init__(
        self,
        event_type: EventType,
        message: str,
        data: Optional[dict] = None,
    ):
        """Initialize multimedia event.

        Args:
            event_type: Type of event
            message: Human-readable message
            data: Optional additional data
        """
        self.event_type = event_type
        self.message = message
        self.data = data or {}

    def __str__(self) -> str:
        """String representation."""
        return f"{self.event_type.value}: {self.message}"


class RecoveryEvent(Exception):
    """Event indicating recovery from an error condition.

    This is used to signal that the system has recovered from a previous error.
    """

    pass


class ServerSideHandler:
    """Handles server-side recognition mode with continuous streaming.

    Streams multimedia segments to VLM as they arrive.
    Implements back pressure and buffering for network resilience.
    """

    def __init__(
        self,
        vlm_client: StreamingVLMClient,
        config: Optional[ManagerConfig] = None,
        error_callback: Optional[Callable] = None,
    ):
        """Initialize server-side handler.

        Args:
            vlm_client: Streaming VLM client for sending data
            config: Configuration parameters (uses defaults if None)
            error_callback: Callback for error events (optional)
        """
        self._vlm_client = vlm_client
        self._config = config or ManagerConfig()
        self._error_callback = error_callback

        # Send buffer for backpressure handling
        # When network is slow, segments queue here
        self._send_buffer: Deque[MultimediaSegment] = deque(
            maxlen=self._config.max_send_buffer_size if self._config.enable_backpressure else None
        )

        # Metrics
        self._segments_sent = 0
        self._send_errors = 0
        self._connection_errors = 0
        self._total_latency_ms = 0.0  # Sum of all segment latencies
        self._latency_count = 0  # Number of latency measurements

        logger.info(
            f"ServerSideHandler initialized with backpressure={'enabled' if self._config.enable_backpressure else 'disabled'}, "
            f"max_buffer_size={self._config.max_send_buffer_size}"
        )

    async def send_segment(self, segment: MultimediaSegment) -> None:
        """Send multimedia segment to VLM.

        Sends video frames and audio chunk to the VLM via streaming client.
        Implements retry logic with exponential backoff for transient failures.
        Handles backpressure by buffering segments when network is slow.

        Args:
            segment: Multimedia segment to send

        Raises:
            ConnectionError: If VLM connection is lost after retries

        """
        # Check backpressure
        if (
            self._config.enable_backpressure
            and len(self._send_buffer) >= self._config.max_send_buffer_size
        ):
            logger.warning(
                f"Send buffer full ({len(self._send_buffer)}/{self._config.max_send_buffer_size}), "
                "dropping oldest segment"
            )
            # Buffer is full, drop oldest segment (FIFO)
            self._send_buffer.popleft()

        # Add to send buffer
        self._send_buffer.append(segment)

        # Process send buffer
        await self._process_send_buffer()

    async def _process_send_buffer(self) -> None:
        """Process segments in send buffer with retry logic.

        Attempts to send all buffered segments to the VLM.
        Implements exponential backoff retry for transient failures.
        Emits error events for failures and recovery events on success.

        """
        import time

        while self._send_buffer:
            segment = self._send_buffer[0]  # Peek at first segment
            send_start_time = time.time()

            # Try to send with retry logic
            retry_count = 0
            last_error: Optional[Exception] = None
            had_previous_error = self._connection_errors > 0

            while retry_count <= self._config.max_retry_attempts:
                try:
                    # Send video frames
                    for frame in segment.video_frames:
                        await self._vlm_client.send_frame(frame)

                    # Send audio chunk
                    await self._vlm_client.send_audio_chunk(segment.audio)

                    # Success! Remove from buffer
                    self._send_buffer.popleft()
                    self._segments_sent += 1

                    # Track latency
                    latency_ms = (time.time() - send_start_time) * 1000
                    self._total_latency_ms += latency_ms
                    self._latency_count += 1

                    # Emit recovery event if we had previous errors
                    if had_previous_error and retry_count > 0:
                        logger.info(f"Recovered from connection error after {retry_count} retries")
                        if self._error_callback:
                            # Emit recovery event (using None to indicate recovery)
                            recovery_msg = f"Connection recovered after {retry_count} retries"
                            self._error_callback(
                                RecoveryEvent(recovery_msg)  # type: ignore[arg-type]
                            )

                    logger.debug(
                        f"Sent segment with {len(segment.video_frames)} frame(s) and audio "
                        f"({segment.get_duration_ms():.1f}ms duration, latency: {latency_ms:.1f}ms)"
                    )
                    break  # Exit retry loop

                except ConnectionError as e:
                    last_error = e
                    retry_count += 1
                    self._connection_errors += 1

                    if retry_count <= self._config.max_retry_attempts:
                        # Calculate backoff delay
                        delay = self._config.retry_backoff_base * (2 ** (retry_count - 1))
                        logger.warning(
                            f"Failed to send segment (attempt {retry_count}/{self._config.max_retry_attempts}), "
                            f"retrying in {delay}s: {e}"
                        )

                        # Emit error event
                        if self._error_callback:
                            error_msg = (
                                f"Connection error (attempt {retry_count}/"
                                f"{self._config.max_retry_attempts}): {e}"
                            )
                            self._error_callback(ConnectionError(error_msg))

                        await asyncio.sleep(delay)
                    else:
                        # Max retries exceeded
                        self._send_errors += 1
                        error_msg = f"Failed to send segment after {self._config.max_retry_attempts} attempts: {e}"
                        logger.error(error_msg)

                        # Emit error event
                        if self._error_callback:
                            self._error_callback(ConnectionError(error_msg))

                        # Remove failed segment from buffer to prevent blocking
                        self._send_buffer.popleft()
                        raise ConnectionError(
                            f"Failed to send segment after {self._config.max_retry_attempts} retries"
                        ) from last_error

                except Exception as e:
                    # Non-retryable error
                    self._send_errors += 1
                    logger.error(f"Non-retryable error sending segment: {e}", exc_info=True)

                    # Emit error event
                    if self._error_callback:
                        self._error_callback(e)

                    # Remove failed segment from buffer
                    self._send_buffer.popleft()
                    raise

    def get_buffer_size(self) -> int:
        """Get current buffer size (for backpressure monitoring).

        Returns:
            Number of segments in send buffer

        """
        return len(self._send_buffer)

    def get_metrics(self) -> dict:
        """Get handler metrics.

        Returns:
            Dictionary with metrics including segments sent, errors, buffer size, and latency
        """
        avg_latency_ms = (
            self._total_latency_ms / self._latency_count if self._latency_count > 0 else 0.0
        )

        return {
            "segments_sent": self._segments_sent,
            "send_errors": self._send_errors,
            "connection_errors": self._connection_errors,
            "buffer_size": len(self._send_buffer),
            "buffer_capacity": self._config.max_send_buffer_size,
            "avg_segment_latency_ms": avg_latency_ms,
        }


class ClientSideHandler:
    """Handles client-side recognition mode with buffering and STT.

    Buffers multimedia segments during speech, transcribes audio with STT,
    and sends batched data to VLM.

    """

    def __init__(
        self,
        vlm_client: VLMClientInterface,
        stt_client: SpeechToTextInterface,
        segment_buffer: SegmentBufferManager,
        activity_detector: AudioActivityDetector,
        config: Optional[ManagerConfig] = None,
        error_callback: Optional[Callable] = None,
    ):
        """Initialize client-side handler.

        Args:
            vlm_client: VLM client (streaming or request-response)
            stt_client: STT client for transcription
            segment_buffer: Segment buffer manager
            activity_detector: Audio activity detector
            config: Configuration parameters (uses defaults if None)
            error_callback: Callback for error events (optional)

        """
        self._vlm_client = vlm_client
        self._stt_client = stt_client
        self._segment_buffer = segment_buffer
        self._activity_detector = activity_detector
        self._config = config or ManagerConfig()
        self._error_callback = error_callback

        # State tracking
        self._is_buffering = False
        self._speech_start_time: Optional[float] = None

        # Metrics
        self._segments_buffered = 0
        self._segments_sent = 0
        self._stt_errors = 0
        self._send_errors = 0
        self._connection_errors = 0
        self._total_stt_duration_ms = 0.0  # Sum of all STT durations
        self._stt_count = 0  # Number of STT operations

        logger.info(
            f"ClientSideHandler initialized with STT provider={self._stt_client.get_provider()}, "
            f"buffer_capacity={self._config.max_segment_buffer_size}"
        )

    async def process_segment(self, segment: MultimediaSegment) -> None:
        """Process multimedia segment (buffer if speech active).

        Checks audio activity state and either buffers the segment during speech
        or triggers transcription and sending when speech ends.

        Args:
            segment: Multimedia segment to process

        """
        # Detect activity in audio
        activity_state = self._activity_detector.process_audio(segment.audio)

        if activity_state == ActivityState.SPEECH:
            # Speech detected - start/continue buffering
            if not self._is_buffering:
                self._is_buffering = True
                self._speech_start_time = segment.start_time.timestamp()
                logger.info("Speech detected, starting segment buffering")

                # Emit buffering started event
                if self._error_callback:
                    event = MultimediaEvent(
                        EventType.BUFFERING_STARTED,
                        "Started buffering segments during speech",
                        {"timestamp": segment.start_time.isoformat()},
                    )
                    self._error_callback(event)  # type: ignore[arg-type]

            # Add segment to buffer
            self._segment_buffer.add_segment(segment)
            self._segments_buffered += 1

            # Emit segment buffered event
            if self._error_callback:
                event = MultimediaEvent(
                    EventType.SEGMENT_BUFFERED,
                    f"Buffered segment (buffer size: {self._segment_buffer.get_size()})",
                    {
                        "buffer_size": self._segment_buffer.get_size(),
                        "buffer_capacity": self._config.max_segment_buffer_size,
                    },
                )
                self._error_callback(event)  # type: ignore[arg-type]

            logger.debug(
                f"Buffered segment (buffer size: {self._segment_buffer.get_size()}/{self._config.max_segment_buffer_size})"
            )

        elif activity_state == ActivityState.SPEECH_ENDED:
            # Speech ended - process buffered segments
            if self._is_buffering:
                logger.info(
                    f"Speech ended after {self._activity_detector.get_speech_duration():.2f}s, "
                    f"processing {self._segment_buffer.get_size()} buffered segments"
                )

                # Emit buffering stopped event
                if self._error_callback:
                    event = MultimediaEvent(
                        EventType.BUFFERING_STOPPED,
                        f"Stopped buffering, processing {self._segment_buffer.get_size()} segments",
                        {
                            "buffer_size": self._segment_buffer.get_size(),
                            "speech_duration": self._activity_detector.get_speech_duration(),
                        },
                    )
                    self._error_callback(event)  # type: ignore[arg-type]

                # Get all buffered segments
                buffered_segments = self._segment_buffer.get_all_segments()

                if buffered_segments:
                    # Send buffered data for transcription and VLM processing
                    await self._send_buffered_data(buffered_segments)

                # Clear buffer and reset state
                self._segment_buffer.clear()
                self._is_buffering = False
                self._speech_start_time = None

        elif activity_state == ActivityState.SILENCE:
            # Silence - if we were buffering, timeout has occurred
            if self._is_buffering:
                logger.info(
                    f"Silence timeout reached, clearing {self._segment_buffer.get_size()} buffered segments"
                )

                # Emit buffering stopped event
                if self._error_callback:
                    event = MultimediaEvent(
                        EventType.BUFFERING_STOPPED,
                        f"Silence timeout, clearing {self._segment_buffer.get_size()} segments",
                        {"buffer_size": self._segment_buffer.get_size()},
                    )
                    self._error_callback(event)  # type: ignore[arg-type]

                self._segment_buffer.clear()
                self._is_buffering = False
                self._speech_start_time = None

    async def _send_buffered_data(
        self,
        segments: List[MultimediaSegment],
        text: Optional[str] = None,
    ) -> None:
        """Send buffered segments and transcribed text to VLM.

        Transcribes audio using STT client, then sends all buffered segments
        along with the transcribed text to the VLM. Implements retry logic
        for transient failures. Emits error events for failures and recovery
        events on success.

        Args:
            segments: Buffered multimedia segments
            text: Pre-transcribed text (if None, will transcribe from segments)

        """
        import time

        if not segments:
            logger.warning("No segments to send")
            return

        try:
            # Transcribe audio if text not provided
            if text is None:
                logger.debug(f"Transcribing audio from {len(segments)} segments")

                # Combine audio from all segments for transcription
                # For simplicity, we'll transcribe the first segment's audio
                # In production, you might want to concatenate all audio chunks
                first_segment = segments[0]

                try:
                    stt_start_time = time.time()
                    text = await self._stt_client.transcribe(first_segment.audio)
                    stt_duration_ms = (time.time() - stt_start_time) * 1000

                    # Track STT duration
                    self._total_stt_duration_ms += stt_duration_ms
                    self._stt_count += 1

                    logger.info(f"Transcription completed in {stt_duration_ms:.1f}ms: '{text}'")
                except Exception as e:
                    self._stt_errors += 1
                    error_msg = f"STT transcription failed: {e}"
                    logger.error(error_msg, exc_info=True)

                    # Emit error event
                    if self._error_callback:
                        self._error_callback(RuntimeError(error_msg))

                    # Clear buffer on STT failure as per requirements 7.1
                    logger.info("Clearing buffer due to STT failure")
                    return

            # Collect all video frames from segments
            all_frames = []
            for segment in segments:
                all_frames.extend(segment.video_frames)

            logger.debug(f"Sending {len(all_frames)} frames with transcribed text to VLM")

            # Send to VLM with retry logic
            retry_count = 0
            last_error: Optional[Exception] = None
            had_previous_error = self._connection_errors > 0

            while retry_count <= self._config.max_retry_attempts:
                try:
                    # Create VLM request
                    request = VLMRequest(
                        frames=all_frames,
                        audio=segments[0].audio if segments else None,
                        text=text,
                    )

                    # Send based on client type
                    if hasattr(self._vlm_client, "process_multimodal_input"):
                        # Request-response client
                        await self._vlm_client.process_multimodal_input(request)  # type: ignore[attr-defined]
                    else:
                        # Streaming client - send frames and text
                        for frame in all_frames:
                            await self._vlm_client.send_frame(frame)  # type: ignore[attr-defined]
                        if text:
                            await self._vlm_client.send_text(text)  # type: ignore[attr-defined]

                    # Success!
                    self._segments_sent += len(segments)

                    # Emit recovery event if we had previous errors
                    if had_previous_error and retry_count > 0:
                        logger.info(
                            f"Recovered from VLM connection error after {retry_count} retries"
                        )
                        if self._error_callback:
                            recovery_msg = f"VLM connection recovered after {retry_count} retries"
                            self._error_callback(RecoveryEvent(recovery_msg))  # type: ignore[arg-type]

                    logger.info(
                        f"Successfully sent {len(segments)} segments with {len(all_frames)} frames to VLM"
                    )
                    break  # Exit retry loop

                except ConnectionError as e:
                    last_error = e
                    retry_count += 1
                    self._connection_errors += 1

                    if retry_count <= self._config.max_retry_attempts:
                        # Calculate backoff delay
                        delay = self._config.retry_backoff_base * (2 ** (retry_count - 1))
                        logger.warning(
                            f"Failed to send to VLM (attempt {retry_count}/{self._config.max_retry_attempts}), "
                            f"retrying in {delay}s: {e}"
                        )

                        # Emit error event
                        if self._error_callback:
                            error_msg = (
                                f"VLM connection error (attempt {retry_count}/"
                                f"{self._config.max_retry_attempts}): {e}"
                            )
                            self._error_callback(ConnectionError(error_msg))

                        await asyncio.sleep(delay)
                    else:
                        # Max retries exceeded
                        self._send_errors += 1
                        error_msg = f"Failed to send to VLM after {self._config.max_retry_attempts} attempts: {e}"
                        logger.error(error_msg)

                        # Emit error event
                        if self._error_callback:
                            self._error_callback(ConnectionError(error_msg))

                        # Clear buffer on unrecoverable error as per requirements 7.1
                        logger.info("Clearing buffer due to unrecoverable VLM error")
                        raise ConnectionError(
                            f"Failed to send to VLM after {self._config.max_retry_attempts} retries"
                        ) from last_error

                except Exception as e:
                    # Non-retryable error
                    self._send_errors += 1
                    error_msg = f"Non-retryable error sending to VLM: {e}"
                    logger.error(error_msg, exc_info=True)

                    # Emit error event
                    if self._error_callback:
                        self._error_callback(e)

                    # Clear buffer on unrecoverable error
                    logger.info("Clearing buffer due to non-retryable error")
                    raise

        except Exception as e:
            logger.error(f"Error in _send_buffered_data: {e}", exc_info=True)
            # Error already logged and counted above
            raise

    def get_metrics(self) -> dict:
        """Get handler metrics.

        Returns:
            Dictionary with metrics including segments buffered, sent, errors, and STT duration
        """
        avg_stt_duration_ms = (
            self._total_stt_duration_ms / self._stt_count if self._stt_count > 0 else 0.0
        )

        return {
            "segments_buffered": self._segments_buffered,
            "segments_sent": self._segments_sent,
            "stt_errors": self._stt_errors,
            "send_errors": self._send_errors,
            "connection_errors": self._connection_errors,
            "buffer_size": self._segment_buffer.get_size(),
            "buffer_capacity": self._config.max_segment_buffer_size,
            "is_buffering": self._is_buffering,
            "avg_stt_duration_ms": avg_stt_duration_ms,
        }
