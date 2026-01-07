"""Unit tests for CaptureManager."""

from typing import Optional

import pytest

from visionmate.core.capture.manager import CaptureManager
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


class TestCaptureManager:
    """Test cases for CaptureManager."""

    def test_init(self):
        """Test CaptureManager initialization."""
        coordinator = CaptureManager()
        assert coordinator.get_video_source_count() == 0
        assert len(coordinator) == 0

    def test_enumerate_screens(self):
        """Test screen enumeration."""
        coordinator = CaptureManager()
        screens = coordinator.enumerate_screens()

        # Should return at least one screen
        assert len(screens) >= 1
        assert all(screen.device_id.startswith("screen_") for screen in screens)

    def test_enumerate_uvc_devices(self):
        """Test UVC device enumeration."""
        coordinator = CaptureManager()
        devices = coordinator.enumerate_uvc_devices()

        # May return 0 or more devices depending on system
        assert isinstance(devices, list)

    def test_enumerate_audio_devices(self):
        """Test audio device enumeration."""
        coordinator = CaptureManager()
        devices = coordinator.enumerate_audio_devices()

        # Should return at least one audio device
        assert len(devices) >= 1

    def test_get_device_metadata(self):
        """Test getting device metadata."""
        coordinator = CaptureManager()
        screens = coordinator.enumerate_screens()

        if screens:
            device_id = screens[0].device_id
            metadata = coordinator.get_device_metadata(device_id)

            assert metadata.device_id == device_id
            assert metadata.name is not None

    def test_add_video_source(self):
        """Test adding a video source."""
        coordinator = CaptureManager()
        capture = MockVideoCapture("test_source")

        coordinator.add_video_source("test_source", capture)

        assert coordinator.get_video_source_count() == 1
        assert "test_source" in coordinator
        assert coordinator.get_video_source("test_source") == capture

    def test_add_duplicate_video_source(self):
        """Test adding a duplicate video source raises ValueError."""
        coordinator = CaptureManager()
        capture = MockVideoCapture("test_source")

        coordinator.add_video_source("test_source", capture)

        with pytest.raises(ValueError, match="Source already exists"):
            coordinator.add_video_source("test_source", capture)

    def test_remove_video_source(self):
        """Test removing a video source."""
        coordinator = CaptureManager()
        capture = MockVideoCapture("test_source")
        capture.start_capture("device_1")

        coordinator.add_video_source("test_source", capture)
        assert coordinator.get_video_source_count() == 1

        coordinator.remove_video_source("test_source")
        assert coordinator.get_video_source_count() == 0
        assert "test_source" not in coordinator
        assert not capture.is_capturing()

    def test_get_all_video_sources(self):
        """Test getting all video sources."""
        coordinator = CaptureManager()
        capture1 = MockVideoCapture("source1")
        capture2 = MockVideoCapture("source2")

        coordinator.add_video_source("source1", capture1)
        coordinator.add_video_source("source2", capture2)

        all_sources = coordinator.get_all_video_sources()
        assert len(all_sources) == 2
        assert all_sources["source1"] == capture1
        assert all_sources["source2"] == capture2

    def test_collect_frames(self):
        """Test collecting frames from all sources."""
        coordinator = CaptureManager()
        capture1 = MockVideoCapture("source1")
        capture2 = MockVideoCapture("source2")

        coordinator.add_video_source("source1", capture1)
        coordinator.add_video_source("source2", capture2)

        frames = coordinator.collect_frames()
        assert len(frames) == 2
        assert "source1" in frames
        assert "source2" in frames

    def test_stop_all_video_sources(self):
        """Test stopping all video sources."""
        coordinator = CaptureManager()
        capture1 = MockVideoCapture("source1")
        capture2 = MockVideoCapture("source2")

        capture1.start_capture("device_1")
        capture2.start_capture("device_2")

        coordinator.add_video_source("source1", capture1)
        coordinator.add_video_source("source2", capture2)

        assert capture1.is_capturing()
        assert capture2.is_capturing()

        coordinator.stop_all_video_sources()

        assert not capture1.is_capturing()
        assert not capture2.is_capturing()

    def test_clear_all_video_sources(self):
        """Test clearing all video sources."""
        coordinator = CaptureManager()
        capture1 = MockVideoCapture("source1")
        capture2 = MockVideoCapture("source2")

        coordinator.add_video_source("source1", capture1)
        coordinator.add_video_source("source2", capture2)

        assert coordinator.get_video_source_count() == 2

        coordinator.clear_all_video_sources()

        assert coordinator.get_video_source_count() == 0

    def test_has_video_source(self):
        """Test checking if video source exists."""
        coordinator = CaptureManager()
        capture = MockVideoCapture("test_source")

        assert not coordinator.has_video_source("test_source")

        coordinator.add_video_source("test_source", capture)
        assert coordinator.has_video_source("test_source")

    def test_contains(self):
        """Test __contains__ method."""
        coordinator = CaptureManager()
        capture = MockVideoCapture("test_source")

        assert "test_source" not in coordinator

        coordinator.add_video_source("test_source", capture)
        assert "test_source" in coordinator

    def test_len(self):
        """Test __len__ method."""
        coordinator = CaptureManager()

        assert len(coordinator) == 0

        coordinator.add_video_source("source1", MockVideoCapture("source1"))
        assert len(coordinator) == 1

        coordinator.add_video_source("source2", MockVideoCapture("source2"))
        assert len(coordinator) == 2

    def test_repr(self):
        """Test __repr__ method."""
        coordinator = CaptureManager()

        repr_str = repr(coordinator)
        assert "CaptureManager" in repr_str
        assert "video_sources=0" in repr_str

        coordinator.add_video_source("source1", MockVideoCapture("source1"))
        repr_str = repr(coordinator)
        assert "video_sources=1" in repr_str
