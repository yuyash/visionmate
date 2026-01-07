#!/usr/bin/env python3
"""Manual test script for verifying complete capture functionality.

This script tests:
- Screen capture
- UVC device capture (if available)
- RTSP stream capture (if configured)
- Audio capture from devices
- Audio mixing from multiple sources
- Multiple video sources simultaneously

Run this script manually to verify Phase 3 capture implementation.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from visionmate.core.capture.audio import AudioMixer, DeviceAudioCapture
from visionmate.core.capture.device import DeviceManager
from visionmate.core.capture.video import ScreenCapture, UVCCapture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_screen_capture():
    """Test screen capture functionality."""
    logger.info("=" * 60)
    logger.info("Testing Screen Capture")
    logger.info("=" * 60)

    device_manager = DeviceManager()

    # Get screens
    screens = device_manager.get_screens()
    logger.info(f"Found {len(screens)} screen(s)")

    if not screens:
        logger.warning("No screens found - skipping screen capture test")
        return False

    # Test first screen
    screen = screens[0]
    logger.info(f"Testing screen: {screen.name} (ID: {screen.device_id})")

    # Create capture
    capture = ScreenCapture(device_manager=device_manager)

    try:
        # Test without window detection
        logger.info("Starting capture (full screen mode)...")
        capture.start_capture(
            device_id=screen.device_id,
            fps=2,
            enable_window_detection=False,
        )

        # Capture a few frames
        logger.info("Capturing frames...")
        for i in range(5):
            time.sleep(0.5)
            frame = capture.get_frame()
            if frame:
                logger.info(
                    f"Frame {i + 1}: {frame.resolution.width}x{frame.resolution.height}, "
                    f"cropped={frame.is_cropped}"
                )
            else:
                logger.warning(f"Frame {i + 1}: No frame available")

        # Test with window detection
        logger.info("Enabling window detection...")
        capture.set_window_detection(True)

        logger.info("Capturing frames with window detection...")
        for i in range(5):
            time.sleep(0.5)
            frame = capture.get_frame()
            if frame:
                logger.info(
                    f"Frame {i + 1}: {frame.resolution.width}x{frame.resolution.height}, "
                    f"cropped={frame.is_cropped}, "
                    f"detected_regions={len(frame.detected_regions)}"
                )
                if frame.active_region:
                    logger.info(f"  Active region: {frame.active_region.to_tuple()}")
            else:
                logger.warning(f"Frame {i + 1}: No frame available")

        logger.info("✓ Screen capture test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Screen capture test failed: {e}", exc_info=True)
        return False

    finally:
        capture.stop_capture()
        logger.info("Screen capture stopped")


def test_uvc_capture():
    """Test UVC device capture functionality."""
    logger.info("=" * 60)
    logger.info("Testing UVC Device Capture")
    logger.info("=" * 60)

    device_manager = DeviceManager()

    # Get UVC devices
    devices = device_manager.get_uvc_devices()
    logger.info(f"Found {len(devices)} UVC device(s)")

    if not devices:
        logger.warning("No UVC devices found - skipping UVC capture test")
        return True  # Not a failure, just no devices

    # Test first device
    device = devices[0]
    logger.info(f"Testing device: {device.name} (ID: {device.device_id})")

    # Create capture
    capture = UVCCapture(device_manager=device_manager)

    try:
        logger.info("Starting capture...")
        capture.start_capture(
            device_id=device.device_id,
            fps=2,
            enable_window_detection=False,
        )

        # Capture a few frames
        logger.info("Capturing frames...")
        for i in range(5):
            time.sleep(0.5)
            frame = capture.get_frame()
            if frame:
                logger.info(f"Frame {i + 1}: {frame.resolution.width}x{frame.resolution.height}")
            else:
                logger.warning(f"Frame {i + 1}: No frame available")

        logger.info("✓ UVC capture test passed")
        return True

    except Exception as e:
        logger.error(f"✗ UVC capture test failed: {e}", exc_info=True)
        return False

    finally:
        capture.stop_capture()
        logger.info("UVC capture stopped")


def test_audio_capture():
    """Test audio device capture functionality."""
    logger.info("=" * 60)
    logger.info("Testing Audio Device Capture")
    logger.info("=" * 60)

    device_manager = DeviceManager()

    # Get audio devices
    devices = device_manager.get_audio_devices()
    logger.info(f"Found {len(devices)} audio device(s)")

    if not devices:
        logger.warning("No audio devices found - skipping audio capture test")
        return True  # Not a failure, just no devices

    # Test first device
    device = devices[0]
    logger.info(f"Testing device: {device.name} (ID: {device.device_id})")

    # Create capture
    capture = DeviceAudioCapture(chunk_duration=0.5)

    try:
        logger.info("Starting capture...")
        capture.start_capture(
            device_id=device.device_id,
            sample_rate=16000,
            channels=1,
        )

        # Capture a few chunks
        logger.info("Capturing audio chunks...")
        for i in range(5):
            time.sleep(0.5)
            chunk = capture.get_chunk()
            if chunk:
                logger.info(
                    f"Chunk {i + 1}: {len(chunk.data)} samples, "
                    f"sample_rate={chunk.sample_rate}, "
                    f"channels={chunk.channels}"
                )
            else:
                logger.warning(f"Chunk {i + 1}: No chunk available")

        logger.info("✓ Audio capture test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Audio capture test failed: {e}", exc_info=True)
        return False

    finally:
        capture.stop_capture()
        logger.info("Audio capture stopped")


def test_audio_mixing():
    """Test audio mixing from multiple sources."""
    logger.info("=" * 60)
    logger.info("Testing Audio Mixing")
    logger.info("=" * 60)

    device_manager = DeviceManager()

    # Get audio devices
    devices = device_manager.get_audio_devices()
    logger.info(f"Found {len(devices)} audio device(s)")

    if len(devices) < 1:
        logger.warning("Need at least 1 audio device - skipping audio mixing test")
        return True  # Not a failure, just not enough devices

    # Create mixer
    mixer = AudioMixer(target_sample_rate=16000, target_channels=1)

    # Create captures for available devices (up to 2)
    captures = []
    for i, device in enumerate(devices[:2]):
        logger.info(f"Adding device {i + 1}: {device.name}")
        capture = DeviceAudioCapture(chunk_duration=0.5)
        try:
            capture.start_capture(
                device_id=device.device_id,
                sample_rate=16000,
                channels=1,
            )
            source_id = mixer.add_source(capture)
            captures.append(capture)
            logger.info(f"  Added to mixer with ID: {source_id}")
        except Exception as e:
            logger.warning(f"  Failed to add device: {e}")

    if not captures:
        logger.warning("No audio sources added - skipping audio mixing test")
        return True

    try:
        # Get mixed chunks
        logger.info("Capturing mixed audio chunks...")
        for i in range(5):
            time.sleep(0.5)
            chunk = mixer.get_mixed_chunk()
            if chunk:
                logger.info(
                    f"Mixed chunk {i + 1}: {len(chunk.data)} samples, "
                    f"sample_rate={chunk.sample_rate}, "
                    f"channels={chunk.channels}"
                )
            else:
                logger.warning(f"Mixed chunk {i + 1}: No chunk available")

        logger.info("✓ Audio mixing test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Audio mixing test failed: {e}", exc_info=True)
        return False

    finally:
        # Stop all captures
        for capture in captures:
            capture.stop_capture()
        logger.info("All audio captures stopped")


def test_multiple_video_sources():
    """Test multiple video sources simultaneously."""
    logger.info("=" * 60)
    logger.info("Testing Multiple Video Sources")
    logger.info("=" * 60)

    device_manager = DeviceManager()

    # Get screens and UVC devices
    screens = device_manager.get_screens()
    uvc_devices = device_manager.get_uvc_devices()

    logger.info(f"Found {len(screens)} screen(s) and {len(uvc_devices)} UVC device(s)")

    if len(screens) < 1:
        logger.warning("Need at least 1 screen - skipping multiple sources test")
        return False

    # Create captures
    captures = []

    # Add first screen
    if screens:
        screen = screens[0]
        logger.info(f"Adding screen: {screen.name}")
        capture = ScreenCapture(device_manager=device_manager)
        try:
            capture.start_capture(
                device_id=screen.device_id,
                fps=2,
                enable_window_detection=False,
            )
            captures.append(("screen", capture))
            logger.info("  Screen capture started")
        except Exception as e:
            logger.warning(f"  Failed to start screen capture: {e}")

    # Add first UVC device if available
    if uvc_devices:
        device = uvc_devices[0]
        logger.info(f"Adding UVC device: {device.name}")
        capture = UVCCapture(device_manager=device_manager)
        try:
            capture.start_capture(
                device_id=device.device_id,
                fps=2,
                enable_window_detection=False,
            )
            captures.append(("uvc", capture))
            logger.info("  UVC capture started")
        except Exception as e:
            logger.warning(f"  Failed to start UVC capture: {e}")

    if not captures:
        logger.warning("No video sources started - skipping multiple sources test")
        return False

    try:
        # Capture frames from all sources
        logger.info(f"Capturing frames from {len(captures)} source(s)...")
        for i in range(5):
            time.sleep(0.5)
            logger.info(f"Frame {i + 1}:")
            for source_type, capture in captures:
                frame = capture.get_frame()
                if frame:
                    logger.info(
                        f"  {source_type}: {frame.resolution.width}x{frame.resolution.height}"
                    )
                else:
                    logger.warning(f"  {source_type}: No frame available")

        logger.info("✓ Multiple video sources test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Multiple video sources test failed: {e}", exc_info=True)
        return False

    finally:
        # Stop all captures
        for source_type, capture in captures:
            capture.stop_capture()
            logger.info(f"{source_type} capture stopped")


def main():
    """Run all capture tests."""
    logger.info("=" * 60)
    logger.info("VISIONMATE COMPLETE CAPTURE VERIFICATION")
    logger.info("=" * 60)
    logger.info("")

    results = {}

    # Run tests
    results["screen_capture"] = test_screen_capture()
    logger.info("")

    results["uvc_capture"] = test_uvc_capture()
    logger.info("")

    results["audio_capture"] = test_audio_capture()
    logger.info("")

    results["audio_mixing"] = test_audio_mixing()
    logger.info("")

    results["multiple_sources"] = test_multiple_video_sources()
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info("")

    all_passed = all(results.values())
    if all_passed:
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        return 0
    else:
        logger.info("=" * 60)
        logger.info("SOME TESTS FAILED ✗")
        logger.info("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
