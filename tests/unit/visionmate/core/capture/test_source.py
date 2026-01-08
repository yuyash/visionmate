"""Unit tests for VideoSourceManager."""

from typing import Optional

import pytest

from visionmate.core.capture.source import VideoSourceManager
from visionmate.core.capture.video import VideoCaptureInterface
from visionmate.core.models import DeviceMetadata, DeviceType, VideoFrame


class MockVideoCapture(VideoCaptureInterface):
    """Mock video capture for testing."""

    def __init__(self, source_id: str):
        self.source_id = source_id
        self._capturing = False

    def start_capture(
        self,
        device_id: str,
        fps: int = 30,
        resolution: Optional[tuple[int, int]] = None,
        enable_window_detection: bool = False,
    ) -> None:
        """Start capture."""
        self._capturing = True

    def stop_capture(self) -> None:
        """Stop capture."""
        self._capturing = False

    def is_capturing(self) -> bool:
        """Check if capturing."""
        return self._capturing

    def get_frame(self) -> Optional[VideoFrame]:
        """Get frame."""
        return None

    def get_source_info(self) -> DeviceMetadata:
        """Get source info."""
        return DeviceMetadata(
            device_id=self.source_id,
            name="Mock Device",
            device_type=DeviceType.SCREEN,
        )

    def set_window_detection(self, enabled: bool) -> None:
        """Set window detection."""
        pass

    def is_window_detection_enabled(self) -> bool:
        """Check if window detection is enabled."""
        return False


class TestVideoSourceManager:
    """Test cases for VideoSourceManager."""

    def test_init(self):
        """Test VideoSourceManager initialization."""
        manager = VideoSourceManager()
        assert manager.get_source_count() == 0
        assert len(manager) == 0

    def test_add_source(self):
        """Test adding a video source."""
        manager = VideoSourceManager()
        capture = MockVideoCapture("test_source")

        manager.add_source("test_source", capture)

        assert manager.get_source_count() == 1
        assert "test_source" in manager
        assert manager.get_source("test_source") == capture

    def test_add_duplicate_source(self):
        """Test adding a duplicate source raises ValueError."""
        manager = VideoSourceManager()
        capture = MockVideoCapture("test_source")

        manager.add_source("test_source", capture)

        with pytest.raises(ValueError, match="Source already exists"):
            manager.add_source("test_source", capture)

    def test_remove_source(self):
        """Test removing a video source."""
        manager = VideoSourceManager()
        capture = MockVideoCapture("test_source")
        capture.start_capture("device_1")

        manager.add_source("test_source", capture)
        assert manager.get_source_count() == 1

        manager.remove_source("test_source")
        assert manager.get_source_count() == 0
        assert "test_source" not in manager
        assert not capture.is_capturing()  # Should be stopped

    def test_remove_nonexistent_source(self):
        """Test removing a nonexistent source raises KeyError."""
        manager = VideoSourceManager()

        with pytest.raises(KeyError, match="Source not found"):
            manager.remove_source("nonexistent")

    def test_get_source(self):
        """Test getting a video source."""
        manager = VideoSourceManager()
        capture = MockVideoCapture("test_source")

        manager.add_source("test_source", capture)

        retrieved = manager.get_source("test_source")
        assert retrieved == capture

    def test_get_nonexistent_source(self):
        """Test getting a nonexistent source returns None."""
        manager = VideoSourceManager()

        result = manager.get_source("nonexistent")
        assert result is None

    def test_get_all_sources(self):
        """Test getting all video sources."""
        manager = VideoSourceManager()
        capture1 = MockVideoCapture("source1")
        capture2 = MockVideoCapture("source2")

        manager.add_source("source1", capture1)
        manager.add_source("source2", capture2)

        all_sources = manager.get_all_sources()
        assert len(all_sources) == 2
        assert all_sources["source1"] == capture1
        assert all_sources["source2"] == capture2

    def test_get_source_ids(self):
        """Test getting source IDs."""
        manager = VideoSourceManager()
        capture1 = MockVideoCapture("source1")
        capture2 = MockVideoCapture("source2")

        manager.add_source("source1", capture1)
        manager.add_source("source2", capture2)

        source_ids = manager.get_source_ids()
        assert len(source_ids) == 2
        assert "source1" in source_ids
        assert "source2" in source_ids

    def test_collect_frames(self):
        """Test collecting frames from all sources."""
        manager = VideoSourceManager()
        capture1 = MockVideoCapture("source1")
        capture2 = MockVideoCapture("source2")

        manager.add_source("source1", capture1)
        manager.add_source("source2", capture2)

        frames = manager.collect_frames()
        assert len(frames) == 2
        assert "source1" in frames
        assert "source2" in frames
        # Mock returns None for frames
        assert frames["source1"] is None
        assert frames["source2"] is None

    def test_stop_all(self):
        """Test stopping all sources."""
        manager = VideoSourceManager()
        capture1 = MockVideoCapture("source1")
        capture2 = MockVideoCapture("source2")

        capture1.start_capture("device_1")
        capture2.start_capture("device_2")

        manager.add_source("source1", capture1)
        manager.add_source("source2", capture2)

        assert capture1.is_capturing()
        assert capture2.is_capturing()

        manager.stop_all()

        assert not capture1.is_capturing()
        assert not capture2.is_capturing()

    def test_clear_all(self):
        """Test clearing all sources."""
        manager = VideoSourceManager()
        capture1 = MockVideoCapture("source1")
        capture2 = MockVideoCapture("source2")

        manager.add_source("source1", capture1)
        manager.add_source("source2", capture2)

        assert manager.get_source_count() == 2

        manager.clear_all()

        assert manager.get_source_count() == 0
        assert "source1" not in manager
        assert "source2" not in manager

    def test_is_capturing(self):
        """Test checking if a source is capturing."""
        manager = VideoSourceManager()
        capture = MockVideoCapture("test_source")

        manager.add_source("test_source", capture)

        assert not manager.is_capturing("test_source")

        capture.start_capture("device_1")
        assert manager.is_capturing("test_source")

        capture.stop_capture()
        assert not manager.is_capturing("test_source")

    def test_is_capturing_nonexistent_source(self):
        """Test checking if a nonexistent source is capturing."""
        manager = VideoSourceManager()

        assert not manager.is_capturing("nonexistent")

    def test_contains(self):
        """Test __contains__ method."""
        manager = VideoSourceManager()
        capture = MockVideoCapture("test_source")

        assert "test_source" not in manager

        manager.add_source("test_source", capture)
        assert "test_source" in manager

    def test_len(self):
        """Test __len__ method."""
        manager = VideoSourceManager()

        assert len(manager) == 0

        manager.add_source("source1", MockVideoCapture("source1"))
        assert len(manager) == 1

        manager.add_source("source2", MockVideoCapture("source2"))
        assert len(manager) == 2

    def test_repr(self):
        """Test __repr__ method."""
        manager = VideoSourceManager()

        repr_str = repr(manager)
        assert "VideoSourceManager" in repr_str
        assert "sources=0" in repr_str

        manager.add_source("source1", MockVideoCapture("source1"))
        repr_str = repr(manager)
        assert "sources=1" in repr_str
