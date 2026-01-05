"""Property-based tests for input mode switching.

Feature: real-time-qa-assistant, Property 1: Configuration Switching Preserves State
Validates: Requirements 3.2
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from visionmate.capture.audio import SoundDeviceAudioCapture
from visionmate.capture.screen import MSSScreenCapture
from visionmate.core.input import InputMode
from visionmate.core.session import SessionManager

# Strategy for generating InputMode values
input_mode_strategy = st.sampled_from(
    [
        InputMode.VIDEO_AUDIO,
        InputMode.VIDEO_ONLY,
        InputMode.AUDIO_ONLY,
    ]
)


def create_session_manager() -> SessionManager:
    """Create a fresh session manager for testing."""
    screen_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()
    return SessionManager(
        screen_capture=screen_capture,
        audio_capture=audio_capture,
    )


@settings(max_examples=100)
@given(
    initial_mode=input_mode_strategy,
    new_mode=input_mode_strategy,
)
def test_input_mode_switching_preserves_state_when_not_capturing(
    initial_mode: InputMode,
    new_mode: InputMode,
) -> None:
    """Property 1: Configuration Switching Preserves State (not capturing).

    For any system configuration change (input mode), switching to a new
    configuration should preserve the current session state when not capturing.

    Feature: real-time-qa-assistant, Property 1: Configuration Switching Preserves State
    Validates: Requirements 3.2
    """
    session_manager = create_session_manager()

    # Set initial mode
    session_manager.set_input_mode(initial_mode)

    # Verify initial state
    assert session_manager.get_input_mode() == initial_mode
    assert not session_manager.is_capturing()

    # Switch to new mode
    session_manager.set_input_mode(new_mode)

    # Verify state is preserved (not capturing)
    assert session_manager.get_input_mode() == new_mode
    assert not session_manager.is_capturing()


@settings(max_examples=100, deadline=None)
@given(
    initial_mode=input_mode_strategy,
    new_mode=input_mode_strategy,
    fps=st.integers(min_value=1, max_value=60),
)
def test_input_mode_switching_preserves_state_when_capturing(
    initial_mode: InputMode,
    new_mode: InputMode,
    fps: int,
) -> None:
    """Property 1: Configuration Switching Preserves State (capturing).

    For any system configuration change (input mode), switching to a new
    configuration should preserve the current session state and allow the
    system to continue operating with the new configuration.

    Feature: real-time-qa-assistant, Property 1: Configuration Switching Preserves State
    Validates: Requirements 3.2
    """
    session_manager = create_session_manager()

    # Set initial mode
    session_manager.set_input_mode(initial_mode)

    # Start capture
    try:
        session_manager.start_capture(fps=fps)

        # Verify capturing state
        assert session_manager.is_capturing()
        assert session_manager.get_input_mode() == initial_mode
        assert session_manager.get_capture_fps() == fps

        # Switch to new mode (should restart capture with new mode)
        session_manager.set_input_mode(new_mode)

        # Verify state is preserved
        assert session_manager.get_input_mode() == new_mode
        assert session_manager.is_capturing()  # Should still be capturing
        assert session_manager.get_capture_fps() == fps  # FPS should be preserved

    finally:
        # Clean up
        session_manager.stop_capture()


@settings(max_examples=100)
@given(
    mode_sequence=st.lists(input_mode_strategy, min_size=2, max_size=5),
)
def test_input_mode_switching_sequence_preserves_state(
    mode_sequence: list[InputMode],
) -> None:
    """Property 1: Configuration Switching Preserves State (sequence).

    For any sequence of input mode changes, the system should preserve
    state through all transitions.

    Feature: real-time-qa-assistant, Property 1: Configuration Switching Preserves State
    Validates: Requirements 3.2
    """
    session_manager = create_session_manager()

    # Apply sequence of mode changes
    for mode in mode_sequence:
        session_manager.set_input_mode(mode)

        # Verify mode is applied
        assert session_manager.get_input_mode() == mode

        # Verify not capturing (since we never started)
        assert not session_manager.is_capturing()


@settings(max_examples=100)
@given(
    initial_mode=input_mode_strategy,
    new_mode=input_mode_strategy,
)
def test_input_mode_switching_triggers_callback(
    initial_mode: InputMode,
    new_mode: InputMode,
) -> None:
    """Property 1: Configuration Switching Preserves State (callbacks).

    For any input mode change, the system should trigger the appropriate
    callback with the new mode.

    Feature: real-time-qa-assistant, Property 1: Configuration Switching Preserves State
    Validates: Requirements 3.2
    """
    session_manager = create_session_manager()

    # Track callback invocations
    callback_invoked = []

    def mode_changed_callback(mode: InputMode) -> None:
        callback_invoked.append(mode)

    # Register callback
    session_manager.register_callback("input_mode_changed", mode_changed_callback)

    # Set initial mode
    session_manager.set_input_mode(initial_mode)

    # Clear callback history
    callback_invoked.clear()

    # Switch to new mode
    session_manager.set_input_mode(new_mode)

    # Verify callback was invoked if mode actually changed
    if initial_mode != new_mode:
        assert len(callback_invoked) == 1
        assert callback_invoked[0] == new_mode
    else:
        # If mode didn't change, callback should not be invoked
        assert len(callback_invoked) == 0


def test_input_mode_has_video_property() -> None:
    """Test InputMode.has_video property."""
    assert InputMode.VIDEO_AUDIO.has_video is True
    assert InputMode.VIDEO_ONLY.has_video is True
    assert InputMode.AUDIO_ONLY.has_video is False


def test_input_mode_has_audio_property() -> None:
    """Test InputMode.has_audio property."""
    assert InputMode.VIDEO_AUDIO.has_audio is True
    assert InputMode.VIDEO_ONLY.has_audio is False
    assert InputMode.AUDIO_ONLY.has_audio is True


def test_input_mode_string_representation() -> None:
    """Test InputMode string representation."""
    assert str(InputMode.VIDEO_AUDIO) == "Video + Audio"
    assert str(InputMode.VIDEO_ONLY) == "Video Only"
    assert str(InputMode.AUDIO_ONLY) == "Audio Only"
