"""Segment buffer management for multimedia data.

This module provides buffering capabilities for multimedia segments with
capacity and memory limits, implementing FIFO eviction policy.
"""

import logging
from collections import deque
from typing import List

from visionmate.core.models import MultimediaSegment

logger = logging.getLogger(__name__)


class SegmentBufferManager:
    """Manages multimedia segment buffering with capacity limits and eviction.

    Implements FIFO eviction when capacity is reached. Maintains timestamps
    for temporal ordering.
    """

    def __init__(
        self,
        max_capacity: int = 300,
        max_memory_mb: float = 500.0,
    ):
        """Initialize segment buffer.

        Args:
            max_capacity: Maximum number of segments to buffer
            max_memory_mb: Maximum memory usage in MB
        """
        self._max_capacity = max_capacity
        self._max_memory_mb = max_memory_mb
        self._buffer: deque[MultimediaSegment] = deque(maxlen=max_capacity)
        self._current_memory_mb: float = 0.0
        self._dropped_count: int = 0

        logger.info(
            f"Initialized SegmentBufferManager with capacity={max_capacity}, "
            f"max_memory={max_memory_mb}MB"
        )

    def add_segment(self, segment: MultimediaSegment) -> None:
        """Add segment to buffer.

        Args:
            segment: Multimedia segment to add

        Note:
            If buffer is full, oldest segment is evicted (FIFO).
        """
        segment_size_mb = segment.get_memory_size_mb()

        # Check memory limit and evict if needed
        while (
            self._current_memory_mb + segment_size_mb > self._max_memory_mb
            and len(self._buffer) > 0
        ):
            evicted = self._buffer.popleft()
            evicted_size = evicted.get_memory_size_mb()
            self._current_memory_mb -= evicted_size
            self._dropped_count += 1
            logger.warning(
                f"Evicted segment due to memory limit. Dropped count: {self._dropped_count}"
            )

        # Add segment (deque handles capacity limit automatically with FIFO)
        if len(self._buffer) == self._max_capacity:
            # About to evict due to capacity
            evicted = self._buffer[0]  # Will be evicted by append
            evicted_size = evicted.get_memory_size_mb()
            self._current_memory_mb -= evicted_size
            self._dropped_count += 1
            logger.warning(
                f"Evicted segment due to capacity limit. Dropped count: {self._dropped_count}"
            )

        self._buffer.append(segment)
        self._current_memory_mb += segment_size_mb

        logger.debug(
            f"Added segment. Buffer size: {len(self._buffer)}/{self._max_capacity}, "
            f"Memory: {self._current_memory_mb:.2f}/{self._max_memory_mb}MB"
        )

    def get_all_segments(self) -> List[MultimediaSegment]:
        """Get all buffered segments in temporal order.

        Returns:
            List of multimedia segments ordered by timestamp
        """
        return list(self._buffer)

    def clear(self) -> int:
        """Clear all buffered segments.

        Returns:
            Number of segments that were cleared
        """
        count = len(self._buffer)
        self._buffer.clear()
        self._current_memory_mb = 0.0

        logger.info(f"Cleared {count} segments from buffer")
        return count

    def get_size(self) -> int:
        """Get current number of buffered segments.

        Returns:
            Number of segments in buffer
        """
        return len(self._buffer)

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        return self._current_memory_mb

    def is_full(self) -> bool:
        """Check if buffer is at capacity.

        Returns:
            True if buffer is full
        """
        return len(self._buffer) >= self._max_capacity

    def get_dropped_count(self) -> int:
        """Get total number of segments dropped due to eviction.

        Returns:
            Number of segments dropped
        """
        return self._dropped_count
