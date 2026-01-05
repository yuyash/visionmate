"""Property-based tests for capture method switching.

Feature: real-time-qa-assistant, Property 1: Configuration Switching Preserves State
Validates: Requirements 1.3
"""

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from visionmate.capture.audio import SoundDeviceAudioCapture
from visionmate.capture.screen import MSSScreenCapture, UVCScreenCapture
from visionmate.core.input import CaptureMethod
from visionmate.core.session import SessionManager

# Strategy for generating CaptureMethod values
capture_method_strategy = st.sampled_from(
    [
        CaptureMethod.OS_NATIVE,
        CaptureMethod.UVC_DEVICE,
    ]
)


def create_session_manager() -> SessionManager:
    """Create a fresh session manager for testing."""
    os_native_capture = MSSScreenCapture()
    uvc_capture = UVCScreenCapture(device_id=0)
    audio_capture = SoundDeviceAudioCapture()
    return SessionManager(
        screen_capture=os_native_capture,
        audio_capture=audio_capture,
        uvc_capture=uvc_capture,
    )


def create_session_manager_with_mock_uvc() -> SessionManager:
    """Create a session manager with mocked UVC capture for testing."""
    os_native_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()

    # Create a mock UVC capture that doesn't require hardware
    mock_uvc_capture = MagicMock(spec=UVCScreenCapture)
    mock_uvc_capture.start_capture = MagicMock()
    mock_uvc_capture.stop_capture = MagicMock()
    mock_uvc_capture.set_fps = MagicMock()
    mock_uvc_capture.get_fps = MagicMock(return_value=30)
    mock_uvc_capture.get_frame = MagicMock(return_value=None)
    mock_uvc_capture.get_frame_with_highlight = MagicMock(return_value=None)

    return SessionManager(
        screen_capture=os_native_capture,
        audio_capture=audio_capture,
        uvc_capture=mock_uvc_capture,
    )


@settings(max_examples=100)
@given(
    initial_method=capture_method_strategy,
    new_method=capture_method_strategy,
)
def test_capture_method_switching_preserves_state_when_not_capturing(
    initial_method: CaptureMethod,
    new_method: CaptureMethod,
) -> None:
    """Property 1: Configuration Switching Preserves State (not capturing).

    For any system configuration change (capture method), switching to a new
    configuration should preserve the current session state when not capturing.

    Feature: real-time-qa-assistant, Property 1: Configuration Switching Preserves State
    Validates: Requirements 1.3
    """
    session_manager = create_session_manager()

    # Set initial method
    session_manager.set_capture_method(initial_method)

    # Verify initial state
    assert session_manager.get_capture_method() == initial_method
    assert not session_manager.is_capturing()

    # Switch to new method
    session_manager.set_capture_method(new_method)

    # Verify state is preserved (not capturing)
    assert session_manager.get_capture_method() == new_method
    assert not session_manager.is_capturing()


@settings(max_examples=100, deadline=None)
@given(
    initial_method=capture_method_strategy,
    new_method=capture_method_strategy,
    fps=st.integers(min_value=1, max_value=60),
)
def test_capture_method_switching_preserves_state_when_capturing(
    initial_method: CaptureMethod,
    new_method: CaptureMethod,
    fps: int,
) -> None:
    """Property 1: Configuration Switching Preserves State (capturing).

    For any system configuration change (capture method), switching to a new
    configuration should preserve the current session state and allow the
    system to continue operating with the new configuration.

    Feature: real-time-qa-assistant, Property 1: Configuration Switching Preserves State
    Validates: Requirements 1.3
    """
    # Use mocked UVC capture to avoid hardware dependency
    session_manager = create_session_manager_with_mock_uvc()

    # Set initial method
    session_manager.set_capture_method(initial_method)

    # Start capture
    try:
        session_manager.start_capture(fps=fps)

        # Verify capturing state
        assert session_manager.is_capturing()
        assert session_manager.get_capture_method() == initial_method
        assert session_manager.get_capture_fps() == fps

        # Switch to new method (should restart capture with new method)
        session_manager.set_capture_method(new_method)

        # Verify state is preserved
        assert session_manager.get_capture_method() == new_method
        assert session_manager.is_capturing()  # Should still be capturing
        assert session_manager.get_capture_fps() == fps  # FPS should be preserved

    finally:
        # Clean up
        session_manager.stop_capture()


@settings(max_examples=100)
@given(
    method_sequence=st.lists(capture_method_strategy, min_size=2, max_size=5),
)
def test_capture_method_switching_sequence_preserves_state(
    method_sequence: list[CaptureMethod],
) -> None:
    """Property 1: Configuration Switching Preserves State (sequence).

    For any sequence of capture method changes, the system should preserve
    state through all transitions.

    Feature: real-time-qa-assistant, Property 1: Configuration Switching Preserves State
    Validates: Requirements 1.3
    """
    session_manager = create_session_manager()

    # Apply sequence of method changes
    for method in method_sequence:
        session_manager.set_capture_method(method)

        # Verify method is applied
        assert session_manager.get_capture_method() == method

        # Verify not capturing (since we never started)
        assert not session_manager.is_capturing()


@settings(max_examples=100)
@given(
    initial_method=capture_method_strategy,
    new_method=capture_method_strategy,
)
def test_capture_method_switching_triggers_callback(
    initial_method: CaptureMethod,
    new_method: CaptureMethod,
) -> None:
    """Property 1: Configuration Switching Preserves State (callbacks).

    For any capture method change, the system should trigger the appropriate
    callback with the new method.

    Feature: real-time-qa-assistant, Property 1: Configuration Switching Preserves State
    Validates: Requirements 1.3
    """
    session_manager = create_session_manager()

    # Track callback invocations
    callback_invoked = []

    def method_changed_callback(method: CaptureMethod) -> None:
        callback_invoked.append(method)

    # Register callback
    session_manager.register_callback("capture_method_changed", method_changed_callback)

    # Set initial method
    session_manager.set_capture_method(initial_method)

    # Clear callback history
    callback_invoked.clear()

    # Switch to new method
    session_manager.set_capture_method(new_method)

    # Verify callback was invoked if method actually changed
    if initial_method != new_method:
        assert len(callback_invoked) == 1
        assert callback_invoked[0] == new_method
    else:
        # If method didn't change, callback should not be invoked
        assert len(callback_invoked) == 0


def test_capture_method_string_representation() -> None:
    """Test CaptureMethod string representation."""
    assert str(CaptureMethod.OS_NATIVE) == "OS-Native (MSS)"
    assert str(CaptureMethod.UVC_DEVICE) == "UVC Device"
