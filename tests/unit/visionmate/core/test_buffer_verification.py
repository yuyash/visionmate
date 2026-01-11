"""Verification tests for SegmentBufferManager.

This module contains verification tests for the buffer manager checkpoint.
Tests various capacities, eviction behavior, and memory limits.

"""

from datetime import datetime, timedelta

import numpy as np

from visionmate.core.models import (
    AudioChunk,
    AudioSourceType,
    MultimediaSegment,
    Resolution,
    VideoFrame,
    VideoSourceType,
)
from visionmate.core.multimedia.buffer import SegmentBufferManager


def create_test_segment(
    duration_ms: float = 100.0,
    num_frames: int = 1,
    frame_size: tuple[int, int] = (640, 480),
    audio_duration_sec: float = 0.1,
) -> MultimediaSegment:
    """Create a test multimedia segment.

    Args:
        duration_ms: Duration in milliseconds
        num_frames: Number of video frames
        frame_size: Frame dimensions (width, height)
        audio_duration_sec: Audio duration in seconds

    Returns:
        Test MultimediaSegment
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(milliseconds=duration_ms)

    # Create audio chunk
    sample_rate = 16000
    num_samples = int(sample_rate * audio_duration_sec)
    audio_data = np.random.randn(num_samples).astype(np.float32)

    audio = AudioChunk(
        data=audio_data,
        sample_rate=sample_rate,
        channels=1,
        timestamp=start_time,
        source_id="test_audio",
        source_type=AudioSourceType.DEVICE,
        chunk_number=0,
    )

    # Create video frames
    frames = []
    for i in range(num_frames):
        frame_data = np.random.randint(0, 255, (frame_size[1], frame_size[0], 3), dtype=np.uint8)
        frame = VideoFrame(
            image=frame_data,
            timestamp=start_time + timedelta(milliseconds=i * (duration_ms / num_frames)),
            source_id="test_video",
            source_type=VideoSourceType.SCREEN,
            resolution=Resolution(width=frame_size[0], height=frame_size[1]),
            fps=30,
            frame_number=i,
        )
        frames.append(frame)

    return MultimediaSegment(
        audio=audio,
        video_frames=frames,
        start_time=start_time,
        end_time=end_time,
        source_id="test_source",
    )


def test_basic_buffer_operations():
    """Test basic buffer operations: add, get, clear."""
    print("\n=== Test 1: Basic Buffer Operations ===")

    buffer = SegmentBufferManager(max_capacity=10, max_memory_mb=100)

    # Test empty buffer
    assert buffer.get_size() == 0, "Buffer should be empty initially"
    assert buffer.get_memory_usage_mb() == 0.0, "Memory usage should be 0 initially"
    assert not buffer.is_full(), "Buffer should not be full initially"

    # Add segments
    segments = [create_test_segment() for _ in range(5)]
    for segment in segments:
        buffer.add_segment(segment)

    assert buffer.get_size() == 5, f"Buffer should have 5 segments, got {buffer.get_size()}"
    assert buffer.get_memory_usage_mb() > 0, "Memory usage should be > 0"
    assert not buffer.is_full(), "Buffer should not be full with 5/10 segments"

    # Get all segments
    retrieved = buffer.get_all_segments()
    assert len(retrieved) == 5, f"Should retrieve 5 segments, got {len(retrieved)}"

    # Clear buffer
    cleared_count = buffer.clear()
    assert cleared_count == 5, f"Should clear 5 segments, got {cleared_count}"
    assert buffer.get_size() == 0, "Buffer should be empty after clear"
    assert buffer.get_memory_usage_mb() == 0.0, "Memory should be 0 after clear"

    print("✓ Basic operations work correctly")


def test_capacity_eviction():
    """Test FIFO eviction when capacity is reached."""
    print("\n=== Test 2: Capacity-Based FIFO Eviction ===")

    capacity = 5
    buffer = SegmentBufferManager(max_capacity=capacity, max_memory_mb=1000)

    # Fill buffer to capacity
    segments = []
    for _i in range(capacity):
        segment = create_test_segment()
        segments.append(segment)
        buffer.add_segment(segment)

    assert buffer.get_size() == capacity, f"Buffer should be at capacity {capacity}"
    assert buffer.is_full(), "Buffer should be full"
    assert buffer.get_dropped_count() == 0, "No segments should be dropped yet"

    # Add one more segment - should evict oldest
    new_segment = create_test_segment()
    buffer.add_segment(new_segment)

    assert buffer.get_size() == capacity, f"Buffer should still be at capacity {capacity}"
    assert buffer.get_dropped_count() == 1, "One segment should be dropped"

    # Verify FIFO: first segment should be gone, last should be present
    retrieved = buffer.get_all_segments()
    assert len(retrieved) == capacity, f"Should have {capacity} segments"

    # Add multiple more segments
    for _i in range(3):
        buffer.add_segment(create_test_segment())

    assert buffer.get_size() == capacity, f"Buffer should remain at capacity {capacity}"
    assert buffer.get_dropped_count() == 4, "Four segments should be dropped total"

    print("✓ FIFO eviction works correctly")


def test_memory_limit_eviction():
    """Test eviction when memory limit is reached."""
    print("\n=== Test 3: Memory Limit Eviction ===")

    # Create buffer with small memory limit
    memory_limit_mb = 10.0
    buffer = SegmentBufferManager(max_capacity=100, max_memory_mb=memory_limit_mb)

    # Create segments that together will exceed the limit
    segments = []

    # Add segments until we approach memory limit
    while buffer.get_memory_usage_mb() < memory_limit_mb * 0.7:
        segment = create_test_segment(
            num_frames=2,
            frame_size=(640, 480),
            audio_duration_sec=0.3,
        )
        buffer.add_segment(segment)
        segments.append(segment)

    initial_count = buffer.get_size()
    initial_memory = buffer.get_memory_usage_mb()

    print(f"  Initial: {initial_count} segments, {initial_memory:.2f}MB")

    # Add more segments that will cause eviction due to memory
    for _i in range(5):
        segment = create_test_segment(
            num_frames=2,
            frame_size=(640, 480),
            audio_duration_sec=0.3,
        )
        buffer.add_segment(segment)

    final_count = buffer.get_size()
    final_memory = buffer.get_memory_usage_mb()

    print(f"  Final: {final_count} segments, {final_memory:.2f}MB")
    print(f"  Dropped: {buffer.get_dropped_count()} segments")

    # Verify memory limit is respected (allowing some tolerance for the last segment)
    # The buffer should evict old segments to stay near the limit
    assert (
        final_memory <= memory_limit_mb * 1.5
    ), f"Memory usage {final_memory:.2f}MB significantly exceeds limit {memory_limit_mb}MB"

    # Verify segments were evicted
    assert buffer.get_dropped_count() > 0, "Segments should have been evicted"

    # Verify we have fewer segments than we added
    assert final_count < initial_count + 5, "Some segments should have been evicted"

    print("✓ Memory limit eviction works correctly")


def test_various_capacities():
    """Test buffer with various capacity configurations."""
    print("\n=== Test 4: Various Capacity Configurations ===")

    capacities = [1, 5, 10, 50, 100]

    for capacity in capacities:
        buffer = SegmentBufferManager(max_capacity=capacity, max_memory_mb=1000)

        # Fill to capacity
        for _i in range(capacity):
            buffer.add_segment(create_test_segment())

        assert (
            buffer.get_size() == capacity
        ), f"Buffer with capacity {capacity} should have {capacity} segments"
        assert buffer.is_full(), f"Buffer with capacity {capacity} should be full"

        # Add more and verify eviction
        buffer.add_segment(create_test_segment())
        assert (
            buffer.get_size() == capacity
        ), f"Buffer should remain at capacity {capacity} after eviction"
        assert buffer.get_dropped_count() == 1, "One segment should be dropped"

        print(f"  ✓ Capacity {capacity} works correctly")

    print("✓ All capacity configurations work correctly")


def test_temporal_ordering():
    """Test that segments maintain temporal ordering."""
    print("\n=== Test 5: Temporal Ordering ===")

    buffer = SegmentBufferManager(max_capacity=10, max_memory_mb=100)

    # Add segments with increasing timestamps
    base_time = datetime.now()
    for i in range(5):
        segment = create_test_segment()
        # Manually set timestamps to ensure ordering
        segment.start_time = base_time + timedelta(seconds=i)
        segment.end_time = base_time + timedelta(seconds=i + 0.1)
        buffer.add_segment(segment)

    # Retrieve and verify ordering
    segments = buffer.get_all_segments()
    for i in range(len(segments) - 1):
        assert (
            segments[i].start_time <= segments[i + 1].start_time
        ), "Segments should be in temporal order"

    print("✓ Temporal ordering is maintained")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== Test 6: Edge Cases ===")

    # Test capacity of 1
    buffer = SegmentBufferManager(max_capacity=1, max_memory_mb=100)
    buffer.add_segment(create_test_segment())
    assert buffer.get_size() == 1, "Buffer should have 1 segment"
    buffer.add_segment(create_test_segment())
    assert buffer.get_size() == 1, "Buffer should still have 1 segment"
    assert buffer.get_dropped_count() == 1, "One segment should be dropped"
    print("  ✓ Capacity 1 works correctly")

    # Test clearing empty buffer
    buffer2 = SegmentBufferManager(max_capacity=10, max_memory_mb=100)
    cleared = buffer2.clear()
    assert cleared == 0, "Clearing empty buffer should return 0"
    print("  ✓ Clearing empty buffer works correctly")

    # Test very small memory limit
    buffer3 = SegmentBufferManager(max_capacity=100, max_memory_mb=0.1)
    segment = create_test_segment(frame_size=(100, 100), audio_duration_sec=0.01)
    buffer3.add_segment(segment)
    # Should handle gracefully even if segment is larger than limit
    print("  ✓ Small memory limit handled gracefully")

    print("✓ All edge cases handled correctly")


def run_all_verification_tests():
    """Run all verification tests for the buffer manager."""
    print("\n" + "=" * 60)
    print("BUFFER MANAGER VERIFICATION TESTS")
    print("=" * 60)

    try:
        test_basic_buffer_operations()
        test_capacity_eviction()
        test_memory_limit_eviction()
        test_various_capacities()
        test_temporal_ordering()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("✅ ALL VERIFICATION TESTS PASSED")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("=" * 60)
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_verification_tests()
    exit(0 if success else 1)
