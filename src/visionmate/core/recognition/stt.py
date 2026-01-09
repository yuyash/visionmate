"""Speech-to-Text module for audio transcription.

This module provides interfaces and implementations for converting audio to text
using various speech recognition providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import whisper

from ..models import AudioChunk, STTProvider

logger = logging.getLogger(__name__)


class SpeechToTextInterface(ABC):
    """Abstract interface for speech-to-text transcription.

    This interface defines the contract for all STT implementations,
    supporting both local and cloud-based speech recognition.
    """

    @abstractmethod
    async def transcribe(self, audio: AudioChunk) -> str:
        """Transcribe audio chunk to text.

        Args:
            audio: Audio chunk to transcribe

        Returns:
            Transcribed text

        Raises:
            RuntimeError: If transcription fails
        """
        pass

    @abstractmethod
    def set_language(self, language: str) -> None:
        """Set transcription language.

        Args:
            language: Language code (e.g., 'en', 'ja', 'es')
        """
        pass

    @abstractmethod
    def get_provider(self) -> STTProvider:
        """Get the STT provider type.

        Returns:
            STT provider enum value
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the STT provider is available and ready.

        Returns:
            True if provider is available, False otherwise
        """
        pass


class WhisperSTT(SpeechToTextInterface):
    """Local Whisper speech recognition implementation.

    Uses OpenAI's Whisper model for local, offline speech-to-text transcription.
    Supports multiple languages and model sizes.
    """

    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """Initialize Whisper STT.

        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run model on ('cpu', 'cuda', or None for auto-detect)
        """
        self._language = "en"
        self._model_name = model_name
        self._device = device
        self._model: Optional[whisper.Whisper] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self._model_name}")
            self._model = whisper.load_model(self._model_name, device=self._device)
            logger.info(f"Whisper model loaded successfully on device: {self._model.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self._model = None

    async def transcribe(self, audio: AudioChunk) -> str:
        """Transcribe audio chunk to text using Whisper.

        Args:
            audio: Audio chunk to transcribe

        Returns:
            Transcribed text

        Raises:
            RuntimeError: If transcription fails or model not loaded
        """
        if self._model is None:
            raise RuntimeError("Whisper model not loaded")

        try:
            # Convert audio data to float32 format expected by Whisper
            # Whisper expects audio in [-1.0, 1.0] range
            audio_data = audio.data.astype(np.float32)

            # Normalize if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data / 2147483648.0

            # Convert stereo to mono if needed (Whisper expects mono)
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)

            # Ensure 1D array
            audio_data = audio_data.flatten()

            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if audio.sample_rate != 16000:
                # Simple resampling - for production, use librosa or scipy
                logger.warning(
                    f"Audio sample rate is {audio.sample_rate}Hz, "
                    "Whisper expects 16kHz. Results may be suboptimal."
                )

            # Transcribe
            logger.debug(f"Transcribing audio chunk {audio.chunk_number}")
            result = self._model.transcribe(
                audio_data,
                language=self._language if self._language != "auto" else None,
                fp16=False,  # Use FP32 for better compatibility
            )

            text = result["text"].strip()
            logger.debug(f"Transcription result: {text}")
            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Whisper transcription failed: {e}") from e

    def set_language(self, language: str) -> None:
        """Set transcription language.

        Args:
            language: Language code (e.g., 'en', 'ja', 'es') or 'auto' for auto-detection
        """
        self._language = language
        logger.info(f"Whisper language set to: {language}")

    def get_provider(self) -> STTProvider:
        """Get the STT provider type.

        Returns:
            STTProvider.WHISPER
        """
        return STTProvider.WHISPER

    def is_available(self) -> bool:
        """Check if Whisper model is loaded and available.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._model is not None

    def get_model_name(self) -> str:
        """Get the current Whisper model name.

        Returns:
            Model name (e.g., 'base', 'small')
        """
        return self._model_name

    def reload_model(self, model_name: Optional[str] = None) -> None:
        """Reload the Whisper model, optionally with a different size.

        Args:
            model_name: New model name, or None to reload current model
        """
        if model_name:
            self._model_name = model_name
        self._model = None
        self._load_model()


class CloudSTT(SpeechToTextInterface):
    """Cloud-based speech recognition implementation.

    Uses OpenAI's Whisper API for cloud-based speech-to-text transcription.
    Requires an API key and internet connection.
    """

    def __init__(self, api_key: str, model: str = "whisper-1"):
        """Initialize Cloud STT.

        Args:
            api_key: OpenAI API key
            model: Whisper model to use (default: 'whisper-1')
        """
        self._api_key = api_key
        self._model = model
        self._language: Optional[str] = None
        self._client: Optional[object] = None  # type: ignore[type-arg]
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
            logger.info("OpenAI client initialized for Cloud STT")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self._client = None

    async def transcribe(self, audio: AudioChunk) -> str:
        """Transcribe audio chunk to text using OpenAI Whisper API.

        Args:
            audio: Audio chunk to transcribe

        Returns:
            Transcribed text

        Raises:
            RuntimeError: If transcription fails or client not initialized
        """
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")

        try:
            # Convert audio data to bytes
            # OpenAI API expects audio in various formats (mp3, wav, etc.)
            # We'll convert to WAV format in memory
            import io
            import wave

            # Convert audio data to int16 if needed
            if audio.data.dtype == np.float32 or audio.data.dtype == np.float64:
                # Convert from [-1.0, 1.0] to int16
                audio_int16 = (audio.data * 32767).astype(np.int16)
            else:
                audio_int16 = audio.data.astype(np.int16)

            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(audio.channels)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(audio.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            # Reset buffer position
            wav_buffer.seek(0)

            # Prepare file tuple for API
            # OpenAI expects a file-like object with a name attribute
            wav_buffer.name = "audio.wav"

            # Transcribe using OpenAI API
            logger.debug(f"Transcribing audio chunk {audio.chunk_number} via OpenAI API")

            # Build transcription parameters
            transcribe_params = {
                "model": self._model,
                "file": wav_buffer,
            }

            # Add language if specified
            if self._language:
                transcribe_params["language"] = self._language

            # Call API
            # Type checker doesn't understand the dynamic client initialization
            response = self._client.audio.transcriptions.create(**transcribe_params)  # type: ignore[union-attr]

            text = response.text.strip()
            logger.debug(f"Cloud transcription result: {text}")
            return text

        except Exception as e:
            logger.error(f"Cloud transcription failed: {e}")
            raise RuntimeError(f"OpenAI Whisper API transcription failed: {e}") from e

    def set_language(self, language: str) -> None:
        """Set transcription language.

        Args:
            language: ISO-639-1 language code (e.g., 'en', 'ja', 'es')
        """
        self._language = language
        logger.info(f"Cloud STT language set to: {language}")

    def get_provider(self) -> STTProvider:
        """Get the STT provider type.

        Returns:
            STTProvider.CLOUD
        """
        return STTProvider.CLOUD

    def is_available(self) -> bool:
        """Check if OpenAI client is initialized and available.

        Returns:
            True if client is initialized, False otherwise
        """
        return self._client is not None

    def get_model(self) -> str:
        """Get the current Whisper model name.

        Returns:
            Model name (e.g., 'whisper-1')
        """
        return self._model

    def set_model(self, model: str) -> None:
        """Set the Whisper model to use.

        Args:
            model: Model name (e.g., 'whisper-1')
        """
        self._model = model
        logger.info(f"Cloud STT model set to: {model}")
