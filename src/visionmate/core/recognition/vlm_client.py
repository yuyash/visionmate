"""VLM (Vision Language Model) client interfaces and implementations.

This module provides abstract interfaces and concrete implementations for
interacting with various VLM providers including OpenAI Realtime API and
OpenAI-compatible HTTP APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from visionmate.core.models import AudioChunk, VideoFrame


class VLMClientType(Enum):
    """Type of VLM client."""

    STREAMING = "streaming"  # WebSocket-based, continuous streaming
    REQUEST_RESPONSE = "request_response"  # HTTP-based, batch processing


@dataclass
class VLMRequest:
    """Request to VLM (for request-response clients)."""

    frames: List[VideoFrame]
    audio: Optional[AudioChunk] = None
    text: Optional[str] = None
    context: Optional[str] = None
    system_prompt: Optional[str] = None


@dataclass
class VLMResponse:
    """Response from VLM."""

    question: Optional[str]
    direct_answer: Optional[str]
    follow_up_questions: List[str] = field(default_factory=list)
    supplementary_info: str = ""
    confidence: float = 0.0
    timestamp: float = 0.0
    is_partial: bool = False  # True for streaming partial responses


class VLMClientInterface(ABC):
    """Abstract interface for VLM clients."""

    @property
    @abstractmethod
    def client_type(self) -> VLMClientType:
        """Get client type (streaming or request-response)."""

    @abstractmethod
    def set_model(self, model_name: str) -> None:
        """Set the VLM model to use.

        Args:
            model_name: Name of the model to use
        """

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models.

        Returns:
            List of model names
        """


class StreamingVLMClient(VLMClientInterface):
    """Base class for streaming VLM clients (WebSocket-based).

    Streaming clients maintain a persistent connection and send/receive
    data continuously. They are suitable for real-time applications where
    low latency is important.
    """

    @property
    def client_type(self) -> VLMClientType:
        """Get client type."""
        return VLMClientType.STREAMING

    @abstractmethod
    async def connect(self) -> None:
        """Establish WebSocket connection.

        Raises:
            ConnectionError: If connection fails
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Close WebSocket connection."""

    @abstractmethod
    async def send_frame(self, frame: VideoFrame) -> None:
        """Send video frame to VLM (streaming).

        Args:
            frame: Video frame to send
        """

    @abstractmethod
    async def send_audio_chunk(self, audio: AudioChunk) -> None:
        """Send audio chunk to VLM (streaming).

        Args:
            audio: Audio chunk to send
        """

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send text message to VLM (streaming).

        Args:
            text: Text message to send
        """

    @abstractmethod
    def register_response_callback(
        self,
        callback: Callable[[VLMResponse], None],
    ) -> None:
        """Register callback for receiving streaming responses.

        Args:
            callback: Function to call when response is received
        """

    @abstractmethod
    async def notify_topic_change(self) -> None:
        """Notify VLM of topic change (for Reset operation).

        This signals the VLM to reset conversation context while
        maintaining the connection.
        """


class RequestResponseVLMClient(VLMClientInterface):
    """Base class for request-response VLM clients (HTTP-based).

    Request-response clients process batches of input and return complete
    responses. They are suitable for applications where batch processing
    is acceptable and real-time streaming is not required.
    """

    @property
    def client_type(self) -> VLMClientType:
        """Get client type."""
        return VLMClientType.REQUEST_RESPONSE

    @abstractmethod
    async def process_multimodal_input(
        self,
        request: VLMRequest,
    ) -> VLMResponse:
        """Process multimodal input and return response.

        Args:
            request: VLM request containing frames, audio, and/or text

        Returns:
            VLM response with answer and supplementary information

        Raises:
            ValueError: If request is invalid
            ConnectionError: If API call fails
        """
