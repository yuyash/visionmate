"""Unit tests for VLM client interfaces and implementations."""

from datetime import datetime, timezone

import numpy as np
import pytest

from visionmate.core.models import (
    AudioChunk,
    AudioSourceType,
    Resolution,
    VideoFrame,
    VideoSourceType,
)
from visionmate.core.recognition import (
    OpenAICompatibleClient,
    OpenAIRealtimeClient,
    VLMClientType,
    VLMRequest,
    VLMResponse,
)


class TestVLMClientInterfaces:
    """Test VLM client interfaces."""

    def test_openai_realtime_client_type(self):
        """Test that OpenAI Realtime client has correct type."""
        client = OpenAIRealtimeClient(api_key="test-key")
        assert client.client_type == VLMClientType.STREAMING

    def test_openai_compatible_client_type(self):
        """Test that OpenAI Compatible client has correct type."""
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="http://localhost:8000",
        )
        assert client.client_type == VLMClientType.REQUEST_RESPONSE

    def test_openai_realtime_available_models(self):
        """Test that OpenAI Realtime client returns available models."""
        client = OpenAIRealtimeClient(api_key="test-key")
        models = client.get_available_models()
        assert len(models) > 0
        assert "gpt-4o-realtime-preview" in models

    def test_openai_compatible_available_models(self):
        """Test that OpenAI Compatible client returns available models."""
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="http://localhost:8000",
        )
        models = client.get_available_models()
        assert len(models) > 0
        assert "gpt-4o" in models

    def test_set_model(self):
        """Test setting model on clients."""
        realtime_client = OpenAIRealtimeClient(api_key="test-key")
        realtime_client.set_model("gpt-4o-realtime-preview-2024-12-17")
        assert realtime_client.model == "gpt-4o-realtime-preview-2024-12-17"

        compatible_client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="http://localhost:8000",
        )
        compatible_client.set_model("gpt-4o-mini")
        assert compatible_client.model == "gpt-4o-mini"


class TestVLMDataModels:
    """Test VLM data models."""

    def test_vlm_request_creation(self):
        """Test creating a VLM request."""
        frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=datetime.now(timezone.utc),
            source_id="test-source",
            source_type=VideoSourceType.SCREEN,
            resolution=Resolution(640, 480),
            fps=30,
            frame_number=1,
        )

        request = VLMRequest(
            frames=[frame],
            text="What do you see?",
        )

        assert len(request.frames) == 1
        assert request.text == "What do you see?"
        assert request.audio is None

    def test_vlm_response_creation(self):
        """Test creating a VLM response."""
        response = VLMResponse(
            question="What is shown in the image?",
            direct_answer="The image shows a blank screen.",
            follow_up_questions=["What color is the screen?"],
            supplementary_info="Additional context here.",
            confidence=0.9,
            timestamp=1234567890.0,
            is_partial=False,
        )

        assert response.question == "What is shown in the image?"
        assert response.direct_answer == "The image shows a blank screen."
        assert len(response.follow_up_questions) == 1
        assert response.confidence == 0.9
        assert not response.is_partial


class TestOpenAIRealtimeClient:
    """Test OpenAI Realtime client."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connecting and disconnecting."""
        client = OpenAIRealtimeClient(api_key="test-key")

        # Connect
        await client.connect()
        assert client._connected

        # Disconnect
        await client.disconnect()
        assert not client._connected

    @pytest.mark.asyncio
    async def test_send_frame_requires_connection(self):
        """Test that sending frame requires connection."""
        client = OpenAIRealtimeClient(api_key="test-key")

        frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=datetime.now(timezone.utc),
            source_id="test-source",
            source_type=VideoSourceType.SCREEN,
            resolution=Resolution(640, 480),
            fps=30,
            frame_number=1,
        )

        with pytest.raises(ConnectionError):
            await client.send_frame(frame)

    @pytest.mark.asyncio
    async def test_send_audio_requires_connection(self):
        """Test that sending audio requires connection."""
        client = OpenAIRealtimeClient(api_key="test-key")

        audio = AudioChunk(
            data=np.zeros(1000, dtype=np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=datetime.now(timezone.utc),
            source_id="test-source",
            source_type=AudioSourceType.DEVICE,
            chunk_number=1,
        )

        with pytest.raises(ConnectionError):
            await client.send_audio_chunk(audio)

    @pytest.mark.asyncio
    async def test_register_callback(self):
        """Test registering response callback."""
        client = OpenAIRealtimeClient(api_key="test-key")

        callback_called = False

        def callback(response: VLMResponse):
            nonlocal callback_called
            callback_called = True

        client.register_response_callback(callback)
        assert client.response_callback is not None


class TestOpenAICompatibleClient:
    """Test OpenAI Compatible client."""

    def test_frame_to_base64(self):
        """Test converting frame to base64."""
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="http://localhost:8000",
        )

        frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=datetime.now(timezone.utc),
            source_id="test-source",
            source_type=VideoSourceType.SCREEN,
            resolution=Resolution(640, 480),
            fps=30,
            frame_number=1,
        )

        base64_str = client._frame_to_base64(frame)
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

    def test_extract_question(self):
        """Test extracting question from content."""
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="http://localhost:8000",
        )

        content = "Question:\nWhat is the weather?\nAnswer: It's sunny."
        question = client._extract_question(content)
        assert question == "What is the weather?"

    def test_extract_answer(self):
        """Test extracting answer from content."""
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="http://localhost:8000",
        )

        content = "Answer: It's sunny today."
        answer = client._extract_answer(content)
        assert answer is not None
        assert "sunny" in answer

    def test_extract_follow_ups(self):
        """Test extracting follow-up questions."""
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="http://localhost:8000",
        )

        content = "Follow-up questions:\n- What time is it?\n- Where are you?"
        follow_ups = client._extract_follow_ups(content)
        assert len(follow_ups) == 2
        assert "What time is it?" in follow_ups
        assert "Where are you?" in follow_ups

    @pytest.mark.asyncio
    async def test_process_multimodal_input_validation(self):
        """Test that process_multimodal_input validates input."""
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="http://localhost:8000",
        )

        # Empty request should raise ValueError
        request = VLMRequest(frames=[])
        with pytest.raises(ValueError, match="at least one input type"):
            await client.process_multimodal_input(request)
