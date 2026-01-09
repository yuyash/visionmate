"""OpenAI-compatible HTTP API client implementation.

This module provides a request-response VLM client that uses OpenAI-compatible
HTTP APIs for batch processing of multimodal input. Supports any API that
follows the OpenAI API specification.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import List, Optional

from openai import AsyncOpenAI
from PIL import Image

from visionmate.core.models import VideoFrame
from visionmate.core.recognition.vlm_client import (
    RequestResponseVLMClient,
    VLMRequest,
    VLMResponse,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleClient(RequestResponseVLMClient):
    """OpenAI-compatible HTTP API client (request-response).

    Uses the OpenAI Python library with a custom base_url to support any
    OpenAI-compatible API endpoint. This includes local models, other cloud
    providers, and custom implementations that follow the OpenAI API spec.

    Batches frames and audio into single requests and returns complete responses.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gpt-4o",
    ):
        """Initialize OpenAI-compatible client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            model: Model name to use (default: gpt-4o)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info(f"Initialized OpenAI-compatible client: base_url={base_url}, model={model}")

    def set_model(self, model_name: str) -> None:
        """Set the VLM model to use.

        Args:
            model_name: Name of the model to use
        """
        self.model = model_name
        logger.info(f"Model set to: {model_name}")

    def get_available_models(self) -> List[str]:
        """Get list of available models.

        Note: This returns common OpenAI model names. The actual available
        models depend on the API endpoint being used.

        Returns:
            List of common model names
        """
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]

    async def process_multimodal_input(
        self,
        request: VLMRequest,
    ) -> VLMResponse:
        """Process multimodal input and return response.

        Converts frames to base64 images, builds a messages array with
        multimodal content, and calls the OpenAI-compatible API.

        Args:
            request: VLM request containing frames, audio, and/or text

        Returns:
            VLM response with answer and supplementary information

        Raises:
            ValueError: If request is invalid
            ConnectionError: If API call fails
        """
        try:
            # Validate request
            if not request.frames and not request.text and not request.audio:
                raise ValueError("Request must contain at least one input type")

            # Build messages array
            messages = []

            # Add system prompt if provided
            if request.system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": request.system_prompt,
                    }
                )

            # Build user message content
            content = []

            # Add context if provided
            if request.context:
                content.append(
                    {
                        "type": "text",
                        "text": f"Context: {request.context}",
                    }
                )

            # Add frames as images
            for frame in request.frames:
                image_base64 = self._frame_to_base64(frame)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    }
                )

            # Add audio if provided
            # Note: Audio handling depends on the API's capabilities
            # Some APIs may not support audio input directly
            if request.audio:
                audio_info = (
                    f"Audio chunk: {request.audio.sample_rate}Hz, "
                    f"{request.audio.channels} channels, "
                    f"{len(request.audio.data)} samples"
                )
                content.append(
                    {
                        "type": "text",
                        "text": f"[Audio input: {audio_info}]",
                    }
                )
                logger.debug(f"Audio included in request: {audio_info}")

            # Add text if provided
            if request.text:
                content.append(
                    {
                        "type": "text",
                        "text": request.text,
                    }
                )

            # Add user message
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )

            logger.info(
                f"Sending request to {self.base_url}: "
                f"{len(request.frames)} frames, "
                f"audio={'yes' if request.audio else 'no'}, "
                f"text={'yes' if request.text else 'no'}"
            )

            # Call API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
            )

            # Parse response
            if not response.choices:
                raise ValueError("API returned no choices")

            content_text = response.choices[0].message.content or ""

            # Create VLM response
            # Note: This is a simple parsing. In production, you might want
            # to use structured output or parse the response more carefully
            vlm_response = VLMResponse(
                question=self._extract_question(content_text),
                direct_answer=self._extract_answer(content_text),
                follow_up_questions=self._extract_follow_ups(content_text),
                supplementary_info=self._extract_supplementary(content_text),
                confidence=0.8,  # Default confidence
                timestamp=time.time(),
                is_partial=False,
            )

            logger.info("Response received and parsed successfully")
            return vlm_response

        except Exception as e:
            logger.error(f"Error processing multimodal input: {e}")
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                raise ConnectionError(f"API call failed: {e}") from e
            raise

    def _frame_to_base64(self, frame: VideoFrame) -> str:
        """Convert video frame to base64-encoded JPEG.

        Args:
            frame: Video frame to convert

        Returns:
            Base64-encoded JPEG string
        """
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame.image)

        # Convert to JPEG and encode as base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return image_base64

    def _extract_question(self, content: str) -> Optional[str]:
        """Extract question from response content.

        This is a simple implementation that looks for question markers.
        In production, you might want more sophisticated parsing.

        Args:
            content: Response content text

        Returns:
            Extracted question or None
        """
        # Look for common question markers
        markers = ["Question:", "Q:", "User asked:"]
        for marker in markers:
            if marker in content:
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if marker in line:
                        # Return the line after the marker
                        if i + 1 < len(lines):
                            return lines[i + 1].strip()
                        # Or the rest of the line
                        return line.split(marker, 1)[1].strip()
        return None

    def _extract_answer(self, content: str) -> Optional[str]:
        """Extract direct answer from response content.

        Args:
            content: Response content text

        Returns:
            Extracted answer or None
        """
        # Look for answer markers
        markers = ["Answer:", "A:", "Response:"]
        for marker in markers:
            if marker in content:
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if marker in line:
                        # Collect lines until next section
                        answer_lines = []
                        for j in range(i + 1, len(lines)):
                            if any(m in lines[j] for m in ["Follow-up:", "Supplementary:", "---"]):
                                break
                            answer_lines.append(lines[j])
                        if answer_lines:
                            return "\n".join(answer_lines).strip()
                        # Or the rest of the line
                        return line.split(marker, 1)[1].strip()

        # If no marker found, return the whole content as answer
        return content.strip()

    def _extract_follow_ups(self, content: str) -> List[str]:
        """Extract follow-up questions from response content.

        Args:
            content: Response content text

        Returns:
            List of follow-up questions
        """
        follow_ups = []
        markers = ["Follow-up questions:", "Follow-up:", "Additional questions:"]

        for marker in markers:
            if marker in content:
                lines = content.split("\n")
                in_follow_up_section = False
                for line in lines:
                    if marker in line:
                        in_follow_up_section = True
                        continue
                    if in_follow_up_section:
                        # Stop at next section
                        if any(m in line for m in ["Supplementary:", "---", "Answer:"]):
                            break
                        # Extract question (remove bullet points, numbers, etc.)
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Remove common prefixes
                            for prefix in ["- ", "* ", "• ", "1. ", "2. ", "3. "]:
                                if line.startswith(prefix):
                                    line = line[len(prefix) :]
                            if line:
                                follow_ups.append(line)

        return follow_ups

    def _extract_supplementary(self, content: str) -> str:
        """Extract supplementary information from response content.

        Args:
            content: Response content text

        Returns:
            Supplementary information
        """
        markers = ["Supplementary information:", "Supplementary:", "Additional context:"]

        for marker in markers:
            if marker in content:
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if marker in line:
                        # Collect remaining lines
                        supplementary_lines = []
                        for j in range(i + 1, len(lines)):
                            supplementary_lines.append(lines[j])
                        if supplementary_lines:
                            return "\n".join(supplementary_lines).strip()
                        # Or the rest of the line
                        return line.split(marker, 1)[1].strip()

        return ""
