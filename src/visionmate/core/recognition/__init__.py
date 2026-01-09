"""Recognition module for VLM and speech-to-text integration."""

from visionmate.core.recognition.openai_compatible import OpenAICompatibleClient
from visionmate.core.recognition.openai_realtime import OpenAIRealtimeClient
from visionmate.core.recognition.question_segmenter import (
    Question,
    QuestionSegmenter,
    QuestionState,
)
from visionmate.core.recognition.stt import CloudSTT, SpeechToTextInterface, WhisperSTT
from visionmate.core.recognition.vlm_client import (
    RequestResponseVLMClient,
    StreamingVLMClient,
    VLMClientInterface,
    VLMClientType,
    VLMRequest,
    VLMResponse,
)

__all__ = [
    "VLMClientInterface",
    "VLMClientType",
    "VLMRequest",
    "VLMResponse",
    "StreamingVLMClient",
    "RequestResponseVLMClient",
    "OpenAIRealtimeClient",
    "OpenAICompatibleClient",
    "SpeechToTextInterface",
    "WhisperSTT",
    "CloudSTT",
    "Question",
    "QuestionSegmenter",
    "QuestionState",
]
