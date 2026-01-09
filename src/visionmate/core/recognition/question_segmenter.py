"""Question segmentation module for identifying question boundaries.

This module provides the QuestionSegmenter class which manages question
detection and segmentation in continuous input streams. It maintains a
state machine for question understanding and identifies when new questions
are detected.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from visionmate.core.models import AudioChunk, VideoFrame


class QuestionState(Enum):
    """State of question understanding."""

    LISTENING = "listening"  # Waiting for question
    QUESTION_DETECTED = "question_detected"  # Question identified
    GATHERING_INFO = "gathering_info"  # Collecting context
    ANSWERING = "answering"  # Generating response


@dataclass
class Question:
    """Detected question with context."""

    text: str
    frames: List[VideoFrame] = field(default_factory=list)
    audio: Optional[AudioChunk] = None
    timestamp: datetime = field(default_factory=datetime.now)
    state: QuestionState = QuestionState.LISTENING
    confidence: float = 0.0


class QuestionSegmenter:
    """Manages question detection and segmentation.

    The QuestionSegmenter maintains a state machine for question understanding
    and identifies question boundaries in continuous input streams. It uses
    a sliding window approach to maintain context and timeout-based segmentation
    to detect question boundaries.

    Attributes:
        context_window_seconds: Duration of context window (default 5 seconds)
        segmentation_timeout_seconds: Timeout for question segmentation (default 3 seconds)
    """

    def __init__(
        self,
        context_window_seconds: float = 5.0,
        segmentation_timeout_seconds: float = 3.0,
    ):
        """Initialize QuestionSegmenter.

        Args:
            context_window_seconds: Duration of sliding context window in seconds
            segmentation_timeout_seconds: Timeout for question segmentation in seconds
        """
        self._state = QuestionState.LISTENING
        self._current_question: Optional[Question] = None
        self._context_window_seconds = context_window_seconds
        self._segmentation_timeout_seconds = segmentation_timeout_seconds

        # Sliding window for context
        self._frame_buffer: deque[VideoFrame] = deque()
        self._audio_buffer: deque[AudioChunk] = deque()
        self._text_buffer: deque[tuple[str, datetime]] = deque()

        # Timing for segmentation
        self._last_input_time: Optional[datetime] = None
        self._question_start_time: Optional[datetime] = None

    def update(
        self,
        frames: Optional[List[VideoFrame]] = None,
        audio: Optional[AudioChunk] = None,
        text: Optional[str] = None,
    ) -> Optional[Question]:
        """Update segmenter with new input and detect questions.

        This method processes new input (frames, audio, text) and maintains
        the sliding context window. It uses timeout-based segmentation to
        detect question boundaries.

        Args:
            frames: List of video frames (optional)
            audio: Audio chunk (optional)
            text: Text input (optional)

        Returns:
            Question object if a new question is detected, None otherwise
        """
        current_time = datetime.now()
        self._last_input_time = current_time

        # Add input to buffers
        if frames:
            for frame in frames:
                self._frame_buffer.append(frame)
        if audio:
            self._audio_buffer.append(audio)
        if text:
            self._text_buffer.append((text, current_time))

        # Clean old data from buffers (sliding window)
        self._clean_buffers(current_time)

        # Check for question detection based on state
        detected_question = self._process_state(current_time, text)

        return detected_question

    def reset(self) -> None:
        """Reset segmentation state.

        Clears all buffers and resets the state machine to LISTENING.
        This is typically called when starting a new session or when
        the user explicitly resets the system.
        """
        self._state = QuestionState.LISTENING
        self._current_question = None
        self._frame_buffer.clear()
        self._audio_buffer.clear()
        self._text_buffer.clear()
        self._last_input_time = None
        self._question_start_time = None

    def notify_topic_change(self) -> None:
        """Notify that topic has changed (for Reset operation).

        This method is called when the user triggers a Reset operation.
        It resets the question understanding state while maintaining
        the capture session. This allows the system to start fresh with
        a new topic without stopping capture.
        """
        # Reset question understanding but keep buffers for context
        self._state = QuestionState.LISTENING
        self._current_question = None
        self._question_start_time = None
        # Note: We keep the buffers to maintain some context

    def get_current_state(self) -> QuestionState:
        """Get current question understanding state.

        Returns:
            Current QuestionState
        """
        return self._state

    def get_current_question(self) -> Optional[Question]:
        """Get current question being processed.

        Returns:
            Current Question object or None if no question is active
        """
        return self._current_question

    def _clean_buffers(self, current_time: datetime) -> None:
        """Remove old data from buffers based on context window.

        Args:
            current_time: Current timestamp for comparison
        """
        cutoff_time = current_time - timedelta(seconds=self._context_window_seconds)

        # Clean frame buffer
        while self._frame_buffer and self._frame_buffer[0].timestamp < cutoff_time:
            self._frame_buffer.popleft()

        # Clean audio buffer
        while self._audio_buffer and self._audio_buffer[0].timestamp < cutoff_time:
            self._audio_buffer.popleft()

        # Clean text buffer
        while self._text_buffer and self._text_buffer[0][1] < cutoff_time:
            self._text_buffer.popleft()

    def _process_state(
        self,
        current_time: datetime,
        text: Optional[str],
    ) -> Optional[Question]:
        """Process current state and detect questions.

        This is a simplified implementation that uses timeout-based segmentation.
        In a production system, this would integrate with VLM for semantic
        question detection.

        Args:
            current_time: Current timestamp
            text: Text input (if any)

        Returns:
            Question object if detected, None otherwise
        """
        # Simple heuristic: detect questions based on text input and timeout
        if self._state == QuestionState.LISTENING:
            if text and self._looks_like_question(text):
                # Question detected
                self._state = QuestionState.QUESTION_DETECTED
                self._question_start_time = current_time
                self._current_question = Question(
                    text=text,
                    frames=list(self._frame_buffer),
                    audio=self._audio_buffer[-1] if self._audio_buffer else None,
                    timestamp=current_time,
                    state=QuestionState.QUESTION_DETECTED,
                    confidence=0.8,  # Simple heuristic confidence
                )
                return self._current_question

        elif self._state == QuestionState.QUESTION_DETECTED:
            # Transition to gathering info
            self._state = QuestionState.GATHERING_INFO

        elif self._state == QuestionState.GATHERING_INFO:
            # Check if we should segment (timeout-based)
            if self._question_start_time:
                elapsed = (current_time - self._question_start_time).total_seconds()
                if elapsed >= self._segmentation_timeout_seconds:
                    # Timeout reached, move to answering
                    self._state = QuestionState.ANSWERING

        elif self._state == QuestionState.ANSWERING:
            # Check for new question
            if text and self._looks_like_question(text):
                # New question detected, create new question
                self._state = QuestionState.QUESTION_DETECTED
                self._question_start_time = current_time
                self._current_question = Question(
                    text=text,
                    frames=list(self._frame_buffer),
                    audio=self._audio_buffer[-1] if self._audio_buffer else None,
                    timestamp=current_time,
                    state=QuestionState.QUESTION_DETECTED,
                    confidence=0.8,
                )
                return self._current_question

        return None

    def _looks_like_question(self, text: str) -> bool:
        """Simple heuristic to detect if text looks like a question.

        This is a placeholder implementation. In production, this would
        use VLM-based semantic analysis.

        Args:
            text: Text to analyze

        Returns:
            True if text appears to be a question
        """
        if not text:
            return False

        text_lower = text.lower().strip()

        # Check for question marks
        if "?" in text:
            return True

        # Check for question words at the start
        question_words = [
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "is",
            "are",
            "can",
            "could",
            "would",
            "should",
            "do",
            "does",
            "did",
        ]

        for word in question_words:
            if text_lower.startswith(word + " "):
                return True

        return False
