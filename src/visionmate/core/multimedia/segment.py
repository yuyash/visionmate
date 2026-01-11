"""Multimedia segment builder for temporal correlation of audio and video.

This module implements intelligent frame selection strategies to create
multimedia segments that correlate audio chunks with representative video frames.
"""

from collections import deque
from datetime import datetime
from typing import Deque, List, Optional

import cv2
import numpy as np

from visionmate.core.models import (
    AudioChunk,
    FrameSelectionStrategy,
    MultimediaSegment,
    VideoFrame,
)


class MultimediaSegmentBuilder:
    """Builds multimedia segments by correlating audio and video by timestamp.

    An audio chunk represents audio from time t1 to t2. This builder selects
    representative video frame(s) from that time period using intelligent
    selection strategies that prioritize frames with meaningful visual changes.

    Frame Selection Strategies:
    - MIDDLE: Select middle frame (simple, fast)
    - MOST_DIFFERENT: Select frame most different from previous sent frame
    - ADAPTIVE: Select multiple frames if significant changes detected
    - KEYFRAME: Select frames with high information content

    """

    def __init__(
        self,
        frame_selection_strategy: FrameSelectionStrategy = FrameSelectionStrategy.MOST_DIFFERENT,
        max_time_drift_ms: float = 100.0,
        change_threshold: float = 0.15,
        max_frames_per_segment: int = 3,
    ):
        """Initialize segment builder.

        Args:
            frame_selection_strategy: Strategy for selecting representative frame(s)
            max_time_drift_ms: Maximum allowed time drift between audio and video
            change_threshold: Threshold for detecting significant visual change (0-1)
            max_frames_per_segment: Maximum frames to include per segment

        """
        self.frame_selection_strategy = frame_selection_strategy
        self.max_time_drift_ms = max_time_drift_ms
        self.change_threshold = change_threshold
        self.max_frames_per_segment = max_frames_per_segment

        # Frame buffer: stores frames with timestamp indexing
        self._frame_buffer: Deque[VideoFrame] = deque(maxlen=1000)

        # Track last sent frame for difference calculation
        self._last_sent_frame: Optional[VideoFrame] = None

    def add_frame(self, frame: VideoFrame) -> None:
        """Add video frame to builder.

        Args:
            frame: Video frame to add

        """
        self._frame_buffer.append(frame)

    def build_segment(self, audio: AudioChunk) -> Optional[MultimediaSegment]:
        """Build multimedia segment for audio chunk.

        Selects representative frame(s) from the time period covered by
        the audio chunk using the configured selection strategy.

        Args:
            audio: Audio chunk to build segment for

        Returns:
            MultimediaSegment with correlated audio and video, or None if
            no suitable frames available

        """
        # Calculate audio time range
        audio_start = audio.timestamp
        # Estimate audio duration based on sample count
        audio_duration_sec = len(audio.data) / audio.sample_rate
        audio_end = datetime.fromtimestamp(
            audio_start.timestamp() + audio_duration_sec,
            tz=audio_start.tzinfo,
        )

        # Find frames within audio time range (with drift tolerance)
        drift_tolerance_sec = self.max_time_drift_ms / 1000.0
        candidate_frames = [
            frame
            for frame in self._frame_buffer
            if (
                audio_start.timestamp() - drift_tolerance_sec
                <= frame.timestamp.timestamp()
                <= audio_end.timestamp() + drift_tolerance_sec
            )
        ]

        if not candidate_frames:
            return None

        # Apply selection strategy
        selected_frames: List[VideoFrame] = []

        if self.frame_selection_strategy == FrameSelectionStrategy.MIDDLE:
            # Select middle frame
            middle_idx = len(candidate_frames) // 2
            selected_frames = [candidate_frames[middle_idx]]

        elif self.frame_selection_strategy == FrameSelectionStrategy.MOST_DIFFERENT:
            # Select frame most different from last sent
            selected_frames = [self._select_most_different_frame(candidate_frames)]

        elif self.frame_selection_strategy == FrameSelectionStrategy.ADAPTIVE:
            # Select multiple frames at change points
            selected_frames = self._select_adaptive_frames(candidate_frames)

        elif self.frame_selection_strategy == FrameSelectionStrategy.KEYFRAME:
            # Select frames with high information content
            selected_frames = self._select_keyframes(candidate_frames)

        if not selected_frames:
            return None

        # Update last sent frame
        self._last_sent_frame = selected_frames[-1]

        # Create multimedia segment
        return MultimediaSegment(
            audio=audio,
            video_frames=selected_frames,
            start_time=audio_start,
            end_time=audio_end,
            source_id=audio.source_id,
        )

    def _calculate_frame_difference(
        self,
        frame1: VideoFrame,
        frame2: VideoFrame,
    ) -> float:
        """Calculate visual difference between two frames.

        Uses histogram comparison for fast difference calculation.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Difference score (0 = identical, 1 = completely different)

        """
        # Convert images to grayscale for faster comparison
        gray1 = cv2.cvtColor(frame1.image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2.image, cv2.COLOR_BGR2GRAY)

        # Resize to standard size for consistent comparison
        size = (256, 256)
        gray1 = cv2.resize(gray1, size)
        gray2 = cv2.resize(gray2, size)

        # Calculate histograms
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Normalize histograms
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Compare histograms using correlation method
        # Returns value in [-1, 1] where 1 = identical
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Convert to difference score [0, 1] where 0 = identical
        difference = (1.0 - correlation) / 2.0

        return max(0.0, min(1.0, difference))

    def _calculate_frame_importance(self, frame: VideoFrame) -> float:
        """Calculate importance score for a frame.

        Uses edge density and color variance to determine frame information content.

        Args:
            frame: Frame to score

        Returns:
            Importance score (0-1, higher = more important)

        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

        # Resize for faster processing
        gray = cv2.resize(gray, (256, 256))

        # Calculate edge density using Sobel operator
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.mean(edge_magnitude) / 255.0  # Normalize to [0, 1]

        # Calculate color variance
        # Convert to float for variance calculation
        img_float = frame.image.astype(np.float32)
        # Resize for faster processing
        img_resized: np.ndarray = cv2.resize(img_float, (256, 256))
        color_variance = float(np.var(img_resized)) / (255.0**2)  # type: ignore[no-matching-overload]  # Normalize to [0, 1]

        # Combine metrics (weighted average)
        importance = 0.6 * edge_density + 0.4 * color_variance

        return max(0.0, min(1.0, importance))

    def _select_most_different_frame(
        self,
        frames: List[VideoFrame],
    ) -> VideoFrame:
        """Select frame most different from last sent frame.

        Args:
            frames: Candidate frames

        Returns:
            Frame with maximum difference from last sent frame

        """
        if not self._last_sent_frame:
            # No previous frame, return middle frame
            return frames[len(frames) // 2]

        # Calculate difference for each frame
        max_diff = -1.0
        most_different = frames[0]

        for frame in frames:
            diff = self._calculate_frame_difference(self._last_sent_frame, frame)
            if diff > max_diff:
                max_diff = diff
                most_different = frame

        return most_different

    def _select_adaptive_frames(
        self,
        frames: List[VideoFrame],
    ) -> List[VideoFrame]:
        """Select multiple frames if significant changes detected.

        Analyzes frame sequence and selects frames at change points.

        Args:
            frames: Candidate frames

        Returns:
            List of selected frames (1 to max_frames_per_segment)

        """
        if len(frames) <= 1:
            return frames

        selected = [frames[0]]  # Always include first frame

        # Detect change points
        for i in range(1, len(frames)):
            if len(selected) >= self.max_frames_per_segment:
                break

            # Calculate difference from last selected frame
            diff = self._calculate_frame_difference(selected[-1], frames[i])

            # If significant change detected, add frame
            if diff >= self.change_threshold:
                selected.append(frames[i])

        # If no changes detected, return middle frame
        if len(selected) == 1 and len(frames) > 1:
            middle_idx = len(frames) // 2
            return [frames[middle_idx]]

        return selected

    def _select_keyframes(
        self,
        frames: List[VideoFrame],
    ) -> List[VideoFrame]:
        """Select frames with high information content.

        Uses importance scoring to select most informative frames.

        Args:
            frames: Candidate frames

        Returns:
            List of selected keyframes

        """
        if len(frames) <= self.max_frames_per_segment:
            return frames

        # Score all frames by importance
        scored_frames = [(frame, self._calculate_frame_importance(frame)) for frame in frames]

        # Sort by importance (descending)
        scored_frames.sort(key=lambda x: x[1], reverse=True)

        # Select top N frames
        selected = [frame for frame, _ in scored_frames[: self.max_frames_per_segment]]

        # Sort selected frames by timestamp to maintain temporal order
        selected.sort(key=lambda f: f.timestamp)

        return selected

    def clear_old_frames(self, before_timestamp: datetime) -> int:
        """Clear frames older than timestamp.

        Args:
            before_timestamp: Timestamp threshold

        Returns:
            Number of frames cleared

        """
        initial_count = len(self._frame_buffer)

        # Filter out old frames
        self._frame_buffer = deque(
            (frame for frame in self._frame_buffer if frame.timestamp >= before_timestamp),
            maxlen=self._frame_buffer.maxlen,
        )

        cleared_count = initial_count - len(self._frame_buffer)
        return cleared_count
