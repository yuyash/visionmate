"""Integration tests for video capture with real devices.

These tests use actual hardware devices and should be run on systems
with available capture devices.
"""

import time

import numpy as np
import pytest

from visionmate.core.capture.device import DeviceManager
from visionmate.core.capture.video import ScreenCapture, WindowDetector
from visionmate.core.models import VideoSourceType, WindowRegion


class TestWindowDetectorIntegration:
    """Integration tests for WindowDetector with real system."""

    def test_platform_libraries_available(self):
        """Test if platform-specific libraries are available."""
        detector = WindowDetector()

        print(f"\n✓ Platform: {detector._platform}")
        print(f"  Libraries available: {detector._platform_available}")

        if detector._platform_available:
            print("  ✓ Platform-specific window detection libraries are installed")
        else:
            print("  ⚠ Platform-specific libraries not installed")
            if detector._platform == "Darwin":
                print("    Install with: uv add 'pyobjc-framework-Quartz>=10.3.1'")
            elif detector._platform == "Windows":
                print("    Install with: uv add 'pywin32>=308'")
            elif detector._platform == "Linux":
                print("    Install with: uv add 'python-xlib>=0.33'")

    def test_detect_active_window_real_system(self):
        """Test detecting active window on real system."""
        detector = WindowDetector()

        # Try to detect active window
        region = detector.detect_active_window()

        # On systems with proper libraries installed, should detect a window
        # On systems without libraries, will return None
        if region is not None:
            assert isinstance(region, WindowRegion)
            assert region.width > 0
            assert region.height > 0
            assert region.x >= 0
            assert region.y >= 0
            assert 0.0 <= region.confidence <= 1.0
            assert region.area == region.width * region.height
            print("\n✓ Detected active window:")
            print(f"  Position: ({region.x}, {region.y})")
            print(f"  Size: {region.width}x{region.height}")
            print(f"  Area: {region.area} pixels")
            print(f"  Confidence: {region.confidence}")
        else:
            if detector._platform_available:
                print("\n⚠ No active window detected (may be no window in focus)")
            else:
                print("\n⚠ Window detection not available (missing platform libraries)")


class TestScreenCaptureIntegration:
    """Integration tests for ScreenCapture with real devices."""

    @pytest.fixture
    def device_manager(self):
        """Create a DeviceManager instance."""
        return DeviceManager()

    @pytest.fixture
    def screen_device_id(self, device_manager):
        """Get the first available screen device ID."""
        screens = device_manager.get_screens()
        if not screens:
            pytest.skip("No screen devices available")
        return screens[0].device_id

    def test_get_screens(self, device_manager):
        """Test enumerating real screen devices."""
        screens = device_manager.get_screens()

        assert len(screens) > 0, "At least one screen should be available"

        for screen in screens:
            print(f"\n✓ Found screen: {screen.name}")
            print(f"  Device ID: {screen.device_id}")
            print(f"  Resolution: {screen.resolution}")
            print(f"  FPS: {screen.fps}Hz")

            assert screen.device_id.startswith("screen_")
            assert screen.resolution is not None
            assert screen.resolution.width > 0
            assert screen.resolution.height > 0
            assert screen.fps > 0

    def test_capture_single_frame(self, device_manager, screen_device_id):
        """Test capturing a single frame from real screen."""
        capture = ScreenCapture(device_manager=device_manager)

        # Start capture at 1 FPS
        capture.start_capture(screen_device_id, fps=1)

        try:
            assert capture.is_capturing()

            # Wait for first frame (up to 2 seconds)
            frame = None
            for _ in range(20):
                frame = capture.get_frame()
                if frame is not None:
                    break
                time.sleep(0.1)

            assert frame is not None, "Should capture at least one frame"

            print("\n✓ Captured frame:")
            print(f"  Source ID: {frame.source_id}")
            print(f"  Source Type: {frame.source_type}")
            print(f"  Resolution: {frame.resolution}")
            print(f"  FPS: {frame.fps}")
            print(f"  Frame Number: {frame.frame_number}")
            print(f"  Image Shape: {frame.image.shape}")
            print(f"  Image Dtype: {frame.image.dtype}")
            print(f"  Timestamp: {frame.timestamp}")
            print(f"  Is Cropped: {frame.is_cropped}")
            if frame.active_region:
                print(f"  Active Region: {frame.active_region.to_tuple()}")

            # Verify frame properties
            assert frame.source_id == screen_device_id
            assert frame.source_type == VideoSourceType.SCREEN
            assert frame.fps == 1
            assert frame.frame_number >= 0
            assert isinstance(frame.image, np.ndarray)
            assert frame.image.ndim == 3  # Height x Width x Channels
            assert frame.image.shape[2] == 3  # RGB
            assert frame.image.dtype == np.uint8
            assert frame.resolution.width == frame.image.shape[1]
            assert frame.resolution.height == frame.image.shape[0]

            # Verify timestamp is recent
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            time_diff = (now - frame.timestamp).total_seconds()
            assert time_diff < 5.0, "Frame timestamp should be recent"

        finally:
            capture.stop_capture()
            assert not capture.is_capturing()

    def test_capture_multiple_frames(self, device_manager, screen_device_id):
        """Test capturing multiple frames at different FPS rates."""
        capture = ScreenCapture(device_manager=device_manager)

        # Test at 5 FPS
        capture.start_capture(screen_device_id, fps=5)

        try:
            # Wait and collect frames
            time.sleep(1.0)  # Wait 1 second

            frame = capture.get_frame()
            assert frame is not None

            first_frame_number = frame.frame_number
            print(f"\n✓ Captured {first_frame_number + 1} frames in 1 second at 5 FPS")

            # Should have captured approximately 5 frames (±2 for timing variance)
            assert (
                3 <= first_frame_number + 1 <= 7
            ), f"Expected ~5 frames, got {first_frame_number + 1}"

        finally:
            capture.stop_capture()

    def test_capture_with_window_detection(self, device_manager, screen_device_id):
        """Test capture with window detection enabled."""
        capture = ScreenCapture(device_manager=device_manager)

        # Window detection is disabled by default
        assert not capture.is_window_detection_enabled()

        # Enable window detection
        capture.start_capture(screen_device_id, fps=1, enable_window_detection=True)

        try:
            # Window detection should now be enabled
            assert capture.is_window_detection_enabled()

            # Wait for frame
            time.sleep(0.5)
            frame = capture.get_frame()

            if frame is not None:
                print("\n✓ Window detection status:")
                print(f"  Enabled: {capture.is_window_detection_enabled()}")
                print(f"  Detected Regions: {len(frame.detected_regions)}")
                print(f"  Is Cropped: {frame.is_cropped}")

                if frame.detected_regions:
                    for i, region in enumerate(frame.detected_regions):
                        print(f"  Region {i}: {region.to_tuple()}")
                        print(f"    Confidence: {region.confidence}")
                        print(f"    Area: {region.area} pixels")

                if frame.active_region:
                    print(f"  Active Region: {frame.active_region.to_tuple()}")

                    # If cropped, verify image matches active region
                    if frame.is_cropped:
                        assert frame.resolution.width <= frame.active_region.width
                        assert frame.resolution.height <= frame.active_region.height
                else:
                    print("  No active window detected (may need platform libraries)")

        finally:
            capture.stop_capture()

    def test_capture_fps_accuracy(self, device_manager, screen_device_id):
        """Test FPS timing accuracy."""
        capture = ScreenCapture(device_manager=device_manager)

        target_fps = 10
        capture.start_capture(screen_device_id, fps=target_fps)

        try:
            # Collect frame timestamps
            timestamps = []
            start_time = time.time()

            while time.time() - start_time < 2.0:  # Collect for 2 seconds
                frame = capture.get_frame()
                if frame is not None:
                    current_frame_num = frame.frame_number
                    if not timestamps or current_frame_num > len(timestamps) - 1:
                        timestamps.append(time.time())
                time.sleep(0.01)

            # Calculate actual FPS
            if len(timestamps) >= 2:
                time_span = timestamps[-1] - timestamps[0]
                actual_fps = (len(timestamps) - 1) / time_span

                print("\n✓ FPS accuracy test:")
                print(f"  Target FPS: {target_fps}")
                print(f"  Actual FPS: {actual_fps:.2f}")
                print(f"  Frames captured: {len(timestamps)}")
                print(f"  Time span: {time_span:.2f}s")

                # Allow 20% tolerance for FPS accuracy
                tolerance = target_fps * 0.2
                assert (
                    abs(actual_fps - target_fps) <= tolerance
                ), f"FPS {actual_fps:.2f} outside tolerance of {target_fps}±{tolerance}"

        finally:
            capture.stop_capture()

    def test_capture_different_fps_rates(self, device_manager, screen_device_id):
        """Test capture at different FPS rates."""
        capture = ScreenCapture(device_manager=device_manager)

        fps_rates = [1, 5, 10, 30]

        for fps in fps_rates:
            print(f"\n✓ Testing {fps} FPS...")

            capture.start_capture(screen_device_id, fps=fps)

            try:
                # Wait for a frame
                time.sleep(0.5)
                frame = capture.get_frame()

                assert frame is not None
                assert frame.fps == fps
                print(f"  Successfully captured at {fps} FPS")

            finally:
                capture.stop_capture()
                time.sleep(0.1)  # Brief pause between tests

    def test_capture_start_stop_multiple_times(self, device_manager, screen_device_id):
        """Test starting and stopping capture multiple times."""
        capture = ScreenCapture(device_manager=device_manager)

        for i in range(3):
            print(f"\n✓ Capture cycle {i + 1}...")

            # Start
            capture.start_capture(screen_device_id, fps=5)
            assert capture.is_capturing()

            # Wait for frames
            time.sleep(0.3)
            frame = capture.get_frame()
            assert frame is not None
            print(f"  Captured frame {frame.frame_number}")

            # Stop
            capture.stop_capture()
            assert not capture.is_capturing()
            time.sleep(0.1)

    def test_frame_buffer_behavior(self, device_manager, screen_device_id):
        """Test frame buffer behavior."""
        capture = ScreenCapture(device_manager=device_manager)

        capture.start_capture(screen_device_id, fps=10)

        try:
            # Wait for buffer to fill
            time.sleep(1.0)

            # Get multiple frames - should return the same (latest) frame
            frame1 = capture.get_frame()
            frame2 = capture.get_frame()

            assert frame1 is not None
            assert frame2 is not None

            # Should be the same frame object (latest in buffer)
            assert frame1.frame_number == frame2.frame_number

            print("\n✓ Frame buffer test:")
            print(f"  Latest frame number: {frame1.frame_number}")
            print("  Buffer returns consistent latest frame")

        finally:
            capture.stop_capture()

    def test_get_source_info(self, device_manager, screen_device_id):
        """Test getting source device information."""
        capture = ScreenCapture(device_manager=device_manager)
        capture._device_id = screen_device_id

        info = capture.get_source_info()

        print("\n✓ Source info:")
        print(f"  Device ID: {info.device_id}")
        print(f"  Name: {info.name}")
        print(f"  Device Type: {info.device_type}")
        print(f"  Resolution: {info.resolution}")
        print(f"  FPS: {info.fps}Hz")

        assert info.device_id == screen_device_id
        assert info.resolution is not None

    def test_capture_image_quality(self, device_manager, screen_device_id):
        """Test captured image quality and properties."""
        capture = ScreenCapture(device_manager=device_manager)

        capture.start_capture(screen_device_id, fps=1)

        try:
            time.sleep(0.5)
            frame = capture.get_frame()

            assert frame is not None

            img = frame.image

            print("\n✓ Image quality test:")
            print(f"  Shape: {img.shape}")
            print(f"  Dtype: {img.dtype}")
            print(f"  Min value: {img.min()}")
            print(f"  Max value: {img.max()}")
            print(f"  Mean value: {img.mean():.2f}")

            # Verify image properties
            assert img.ndim == 3
            assert img.shape[2] == 3  # RGB
            assert img.dtype == np.uint8
            assert 0 <= img.min() <= 255
            assert 0 <= img.max() <= 255

            # Image should have some variation (not all black or all white)
            assert img.std() > 0, "Image should have some variation"

        finally:
            capture.stop_capture()

    def test_window_detection_with_multiple_captures(self, device_manager, screen_device_id):
        """Test window detection across multiple frame captures."""
        capture = ScreenCapture(device_manager=device_manager)

        capture.start_capture(screen_device_id, fps=5)

        try:
            # Collect multiple frames
            frames_with_windows = 0
            frames_cropped = 0

            time.sleep(1.0)  # Capture for 1 second

            # Check last few frames
            for _ in range(5):
                frame = capture.get_frame()
                if frame and frame.detected_regions:
                    frames_with_windows += 1
                if frame and frame.is_cropped:
                    frames_cropped += 1
                time.sleep(0.1)

            print("\n✓ Window detection across multiple frames:")
            print(f"  Frames with detected windows: {frames_with_windows}/5")
            print(f"  Frames cropped: {frames_cropped}/5")

            detector = WindowDetector()
            if detector._platform_available:
                # If libraries are available, we should detect windows
                # (unless no window is in focus)
                print("  Platform libraries available: Yes")
                if frames_with_windows > 0:
                    print("  ✓ Successfully detecting windows!")
            else:
                print("  Platform libraries available: No")

        finally:
            capture.stop_capture()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
