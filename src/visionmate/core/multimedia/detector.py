"""Audio activity detection for speech recognition.

This module provides audio activity detection (VAD) functionality to determine
when speech starts and stops in an audio stream. Used for triggering segment
buffering in client-side recognition mode.
"""

from datetime import datetime
from typing import Optional

import numpy as np

from visionmate.core.models import ActivityState, AudioChunk


class AudioActivityDetector:
    """Detects speech activity in audio stream.

    Uses energy-based voice activity detection (VAD) to determine when
    speech starts and stops. Configurable thresholds and silence duration.
    """

    def __init__(
        self,
        energy_threshold: float = 0.01,
        silence_duration_sec: float = 1.5,
        sample_rate: int = 16000,
    ):
        """Initialize activity detector.

        Args:
            energy_threshold: Energy threshold for speech detection (0.0-1.0)
            silence_duration_sec: Duration of silence before considering speech ended
            sample_rate: Audio sample rate in Hz
        """
        self.energy_threshold = energy_threshold
        self.silence_duration_sec = silence_duration_sec
        self.sample_rate = sample_rate

        # State tracking
        self._current_state = ActivityState.SILENCE
        self._speech_start_time: Optional[datetime] = None
        self._last_speech_time: Optional[datetime] = None
        self._silence_start_time: Optional[datetime] = None

    def process_audio(self, audio: AudioChunk) -> ActivityState:
        """Process audio chunk and detect activity.

        Calculates audio energy and compares with threshold to detect speech.
        Tracks silence duration to determine when speech has ended.

        Args:
            audio: Audio chunk to analyze

        Returns:
            Current activity state (SILENCE, SPEECH, SPEECH_ENDED)
        """
        # Calculate audio energy (RMS - Root Mean Square)
        energy = self._calculate_energy(audio.data)

        # Determine if current chunk contains speech
        is_speech = energy > self.energy_threshold

        # State machine for activity detection
        if self._current_state == ActivityState.SILENCE:
            if is_speech:
                # Transition to SPEECH state
                self._current_state = ActivityState.SPEECH
                self._speech_start_time = audio.timestamp
                self._last_speech_time = audio.timestamp
                self._silence_start_time = None

        elif self._current_state == ActivityState.SPEECH:
            if is_speech:
                # Continue in SPEECH state
                self._last_speech_time = audio.timestamp
                self._silence_start_time = None
            else:
                # Start tracking silence
                if self._silence_start_time is None:
                    self._silence_start_time = audio.timestamp

                # Check if silence duration exceeded
                silence_duration = (audio.timestamp - self._silence_start_time).total_seconds()
                if silence_duration >= self.silence_duration_sec:
                    # Transition to SPEECH_ENDED state
                    self._current_state = ActivityState.SPEECH_ENDED
                    return self._current_state

        elif self._current_state == ActivityState.SPEECH_ENDED:
            # Reset to SILENCE after one cycle in SPEECH_ENDED
            self._current_state = ActivityState.SILENCE
            self._speech_start_time = None
            self._last_speech_time = None
            self._silence_start_time = None

        return self._current_state

    def reset(self) -> None:
        """Reset detector state.

        Resets state to SILENCE and clears all timing information.
        """
        self._current_state = ActivityState.SILENCE
        self._speech_start_time = None
        self._last_speech_time = None
        self._silence_start_time = None

    def get_speech_duration(self) -> float:
        """Get duration of current speech segment in seconds.

        Returns:
            Duration in seconds, or 0.0 if no speech segment active
        """
        if self._speech_start_time is None or self._last_speech_time is None:
            return 0.0

        duration = (self._last_speech_time - self._speech_start_time).total_seconds()
        return max(0.0, duration)

    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate audio energy using RMS (Root Mean Square).

        Args:
            audio_data: Audio samples as numpy array

        Returns:
            Energy value normalized to 0.0-1.0 range
        """
        # Handle empty or invalid data
        if audio_data.size == 0:
            return 0.0

        # Calculate RMS energy
        # RMS = sqrt(mean(x^2))
        rms = np.sqrt(np.mean(audio_data**2))

        # Normalize to 0-1 range (assuming audio is in -1 to 1 range)
        # If audio is int16, we need to normalize differently
        if audio_data.dtype == np.int16:
            rms = rms / 32768.0  # Normalize int16 range

        return float(rms)
