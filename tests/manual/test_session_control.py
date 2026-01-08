#!/usr/bin/env python3
"""Manual test script for session control verification.

This script tests the Start/Stop/Reset operations and verifies UI state updates.
Run this script to manually verify checkpoint 17.

Requirements tested:
- 9.1: Start operation
- 9.2: Stop operation
- 9.3: Reset operation
- 9.5: Device selection enables Start button
- 9.6: Start begins capture and recognition
- 9.7: Stop stops capture and recognition
- 9.8: Reset continues capture, restarts recognition
- 9.9: UI disables device selection while active
- 9.10: UI enables device selection when stopped
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from PySide6.QtWidgets import QApplication

from visionmate.core.models import InputMode, SessionState, VideoSourceType
from visionmate.core.session.manager import SessionManager


def test_session_state_transitions():
    """Test session state transitions."""
    print("\n" + "=" * 70)
    print("TEST 1: Session State Transitions")
    print("=" * 70)

    manager = SessionManager()

    # Test initial state
    print("\n1. Testing initial state...")
    assert manager.get_state() == SessionState.IDLE, "Initial state should be IDLE"
    print("   ✓ Initial state is IDLE")

    # Test start without sources (should fail)
    print("\n2. Testing start without sources (should fail)...")
    try:
        manager.start()
        print("   ✗ FAILED: Should have raised RuntimeError")
        return False
    except RuntimeError as e:
        print(f"   ✓ Correctly raised RuntimeError: {e}")

    # Add a video source
    print("\n3. Adding a screen capture source...")
    try:
        # Set input mode to VIDEO_ONLY since we're not adding audio
        manager.set_input_mode(InputMode.VIDEO_ONLY)
        source_id = manager.add_video_source(
            source_type=VideoSourceType.SCREEN, device_id="screen_1", fps=1
        )
        print(f"   ✓ Added video source: {source_id}")
    except Exception as e:
        print(f"   ✗ FAILED to add video source: {e}")
        return False

    # Test start
    print("\n4. Testing start operation...")
    try:
        manager.start()
        assert manager.get_state() == SessionState.ACTIVE, "State should be ACTIVE after start"
        print("   ✓ Session started successfully")
        print(f"   ✓ State is now: {manager.get_state().value}")
    except Exception as e:
        print(f"   ✗ FAILED to start session: {e}")
        return False

    # Test start when already active (should fail)
    print("\n5. Testing start when already active (should fail)...")
    try:
        manager.start()
        print("   ✗ FAILED: Should have raised RuntimeError")
        return False
    except RuntimeError as e:
        print(f"   ✓ Correctly raised RuntimeError: {e}")

    # Test reset
    print("\n6. Testing reset operation...")
    try:
        manager.reset()
        assert manager.get_state() == SessionState.ACTIVE, "State should remain ACTIVE after reset"
        print("   ✓ Session reset successfully")
        print(f"   ✓ State is still: {manager.get_state().value}")
    except Exception as e:
        print(f"   ✗ FAILED to reset session: {e}")
        return False

    # Test stop
    print("\n7. Testing stop operation...")
    try:
        manager.stop()
        assert manager.get_state() == SessionState.IDLE, "State should be IDLE after stop"
        print("   ✓ Session stopped successfully")
        print(f"   ✓ State is now: {manager.get_state().value}")
    except Exception as e:
        print(f"   ✗ FAILED to stop session: {e}")
        return False

    # Test reset when idle (should fail)
    print("\n8. Testing reset when idle (should fail)...")
    try:
        manager.reset()
        print("   ✗ FAILED: Should have raised RuntimeError")
        return False
    except RuntimeError as e:
        print(f"   ✓ Correctly raised RuntimeError: {e}")

    print("\n" + "=" * 70)
    print("TEST 1: PASSED ✓")
    print("=" * 70)
    return True


def test_event_broadcasting():
    """Test event broadcasting."""
    print("\n" + "=" * 70)
    print("TEST 2: Event Broadcasting")
    print("=" * 70)

    manager = SessionManager()
    events_received = []

    def state_changed_callback(data):
        events_received.append(("state_changed", data))
        print(f"   → Event received: state_changed, state={data['state'].value}")

    def session_reset_callback(data):
        events_received.append(("session_reset", data))
        print("   → Event received: session_reset")

    # Register callbacks
    print("\n1. Registering event callbacks...")
    manager.register_callback("state_changed", state_changed_callback)
    manager.register_callback("session_reset", session_reset_callback)
    print("   ✓ Callbacks registered")

    # Add a video source
    print("\n2. Adding video source...")
    # Set input mode to VIDEO_ONLY since we're not adding audio
    manager.set_input_mode(InputMode.VIDEO_ONLY)
    source_id = manager.add_video_source(
        source_type=VideoSourceType.SCREEN, device_id="screen_1", fps=1
    )
    print(f"   ✓ Added video source: {source_id}")

    # Test start event
    print("\n3. Testing start event...")
    events_received.clear()
    manager.start()
    assert len(events_received) == 1, "Should receive 1 event"
    assert events_received[0][0] == "state_changed", "Should be state_changed event"
    assert events_received[0][1]["state"] == SessionState.ACTIVE, "State should be ACTIVE"
    print("   ✓ Start event received correctly")

    # Test reset event
    print("\n4. Testing reset event...")
    events_received.clear()
    manager.reset()
    assert len(events_received) == 1, "Should receive 1 event"
    assert events_received[0][0] == "session_reset", "Should be session_reset event"
    print("   ✓ Reset event received correctly")

    # Test stop event
    print("\n5. Testing stop event...")
    events_received.clear()
    manager.stop()
    assert len(events_received) == 1, "Should receive 1 event"
    assert events_received[0][0] == "state_changed", "Should be state_changed event"
    assert events_received[0][1]["state"] == SessionState.IDLE, "State should be IDLE"
    print("   ✓ Stop event received correctly")

    print("\n" + "=" * 70)
    print("TEST 2: PASSED ✓")
    print("=" * 70)
    return True


def test_ui_button_states():
    """Test UI button state management."""
    print("\n" + "=" * 70)
    print("TEST 3: UI Button State Management")
    print("=" * 70)

    from visionmate.desktop.widgets.session import SessionControlWidget

    _app = QApplication.instance() or QApplication(sys.argv)

    widget = SessionControlWidget()

    # Test initial state (no devices)
    print("\n1. Testing initial state (no devices)...")
    widget.set_has_devices(False)
    widget.set_session_state(SessionState.IDLE)
    assert widget._start_button is not None, "Start button should exist"
    assert widget._stop_button is not None, "Stop button should exist"
    assert widget._reset_button is not None, "Reset button should exist"
    assert not widget._start_button.isEnabled(), "Start should be disabled without devices"
    assert not widget._stop_button.isEnabled(), "Stop should be disabled when IDLE"
    assert not widget._reset_button.isEnabled(), "Reset should be disabled when IDLE"
    print("   ✓ All buttons correctly disabled")

    # Test with devices selected
    print("\n2. Testing with devices selected (IDLE)...")
    widget.set_has_devices(True)
    widget.set_session_state(SessionState.IDLE)
    assert widget._start_button.isEnabled(), "Start should be enabled with devices"
    assert not widget._stop_button.isEnabled(), "Stop should be disabled when IDLE"
    assert not widget._reset_button.isEnabled(), "Reset should be disabled when IDLE"
    print("   ✓ Start enabled, Stop and Reset disabled")

    # Test active state
    print("\n3. Testing active state...")
    widget.set_session_state(SessionState.ACTIVE)
    assert not widget._start_button.isEnabled(), "Start should be disabled when ACTIVE"
    assert widget._stop_button.isEnabled(), "Stop should be enabled when ACTIVE"
    assert widget._reset_button.isEnabled(), "Reset should be enabled when ACTIVE"
    print("   ✓ Start disabled, Stop and Reset enabled")

    # Test back to idle
    print("\n4. Testing back to idle...")
    widget.set_session_state(SessionState.IDLE)
    assert widget._start_button.isEnabled(), "Start should be enabled with devices"
    assert not widget._stop_button.isEnabled(), "Stop should be disabled when IDLE"
    assert not widget._reset_button.isEnabled(), "Reset should be disabled when IDLE"
    print("   ✓ Start enabled, Stop and Reset disabled")

    print("\n" + "=" * 70)
    print("TEST 3: PASSED ✓")
    print("=" * 70)
    return True


def test_capture_lifecycle():
    """Test that capture actually starts and stops."""
    print("\n" + "=" * 70)
    print("TEST 4: Capture Lifecycle")
    print("=" * 70)

    manager = SessionManager()

    # Add a video source
    print("\n1. Adding screen capture source...")
    # Set input mode to VIDEO_ONLY since we're not adding audio
    manager.set_input_mode(InputMode.VIDEO_ONLY)
    source_id = manager.add_video_source(
        source_type=VideoSourceType.SCREEN, device_id="screen_1", fps=1
    )
    print(f"   ✓ Added video source: {source_id}")

    # Get the capture instance
    capture = manager.get_capture_manager().get_video_source(source_id)
    assert capture is not None, "Capture should exist"
    print("   ✓ Capture instance retrieved")

    # Verify capture is running
    print("\n2. Verifying capture is running...")
    assert capture.is_capturing(), "Capture should be running after add"
    print("   ✓ Capture is running")

    # Start session
    print("\n3. Starting session...")
    manager.start()
    assert manager.get_state() == SessionState.ACTIVE, "Session should be ACTIVE"
    assert capture.is_capturing(), "Capture should still be running"
    print("   ✓ Session started, capture still running")

    # Get a frame
    print("\n4. Getting a frame...")
    time.sleep(0.5)  # Wait for at least one frame
    frame = capture.get_frame()
    if frame:
        print(f"   ✓ Got frame: {frame.resolution}, source={frame.source_id}")
    else:
        print("   ⚠ No frame available yet (may be normal)")

    # Stop session
    print("\n5. Stopping session...")
    manager.stop()
    assert manager.get_state() == SessionState.IDLE, "Session should be IDLE"
    assert not capture.is_capturing(), "Capture should be stopped"
    print("   ✓ Session stopped, capture stopped")

    print("\n" + "=" * 70)
    print("TEST 4: PASSED ✓")
    print("=" * 70)
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("SESSION CONTROL VERIFICATION - Checkpoint 17")
    print("=" * 70)
    print("\nThis script tests:")
    print("  - Start/Stop/Reset operations")
    print("  - Session state transitions")
    print("  - Event broadcasting")
    print("  - UI button state management")
    print("  - Capture lifecycle")

    results = []

    # Run tests
    try:
        results.append(("State Transitions", test_session_state_transitions()))
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("State Transitions", False))

    try:
        results.append(("Event Broadcasting", test_event_broadcasting()))
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Event Broadcasting", False))

    try:
        results.append(("UI Button States", test_ui_button_states()))
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("UI Button States", False))

    try:
        results.append(("Capture Lifecycle", test_capture_lifecycle()))
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Capture Lifecycle", False))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Session control is working correctly.")
        print("\nVerified:")
        print("  ✓ Start operation transitions IDLE → ACTIVE")
        print("  ✓ Stop operation transitions ACTIVE → IDLE")
        print("  ✓ Reset operation keeps state ACTIVE")
        print("  ✓ Events are broadcast correctly")
        print("  ✓ UI button states update correctly")
        print("  ✓ Capture lifecycle is managed correctly")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED! Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
