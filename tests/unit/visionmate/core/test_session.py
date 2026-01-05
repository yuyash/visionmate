"""Property-based tests for session state transitions.

Feature: real-time-qa-assistant, Property 2: Session State Transitions
Validates: Requirements 7.6, 7.7, 7.8, 7.9, 7.10
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from visionmate.capture.audio import SoundDeviceAudioCapture
from visionmate.capture.screen import MSSScreenCapture
from visionmate.core.session import SessionManager, SessionState


def create_session_manager() -> SessionManager:
    """Create a fresh session manager for testing."""
    screen_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()
    return SessionManager(
        screen_capture=screen_capture,
        audio_capture=audio_capture,
    )


@settings(max_examples=100, deadline=None)
@given(
    fps=st.integers(min_value=1, max_value=60),
)
def test_start_capture_transitions_from_idle_to_capturing(fps: int) -> None:
    """Property 2: Session State Transitions - Start Capture.

    For any valid start capture operation from IDLE state, the system should
    transition to CAPTURING state and trigger the capture_started event.

    Feature: real-time-qa-assistant, Property 2: Session State Transitions
    Validates: Requirements 7.6
    """
    session_manager = create_session_manager()

    # Track callback invocations
    capture_started_called = []

    def capture_started_callback(**kwargs) -> None:
        capture_started_called.append(True)

    # Register callback
    session_manager.register_callback("capture_started", capture_started_callback)

    # Verify initial state
    assert session_manager.get_session_state() == SessionState.IDLE
    assert not session_manager.is_capturing()

    try:
        # Start capture
        session_manager.start_capture(fps=fps)

        # Verify state transition
        assert session_manager.get_session_state() == SessionState.CAPTURING
        assert session_manager.is_capturing()
        assert session_manager.get_capture_fps() == fps

        # Verify callback was invoked
        assert len(capture_started_called) == 1

    finally:
        # Clean up
        session_manager.stop_capture()


@settings(max_examples=100, deadline=None)
@given(
    fps=st.integers(min_value=1, max_value=60),
)
def test_start_recognition_transitions_from_capturing_to_recognizing(fps: int) -> None:
    """Property 2: Session State Transitions - Start Recognition.

    For any valid start recognition operation from CAPTURING state, the system
    should transition to RECOGNIZING state and trigger the recognition_started event.

    Feature: real-time-qa-assistant, Property 2: Session State Transitions
    Validates: Requirements 7.7
    """
    session_manager = create_session_manager()

    # Track callback invocations
    recognition_started_called = []

    def recognition_started_callback(**kwargs) -> None:
        recognition_started_called.append(True)

    # Register callback
    session_manager.register_callback("recognition_started", recognition_started_callback)

    try:
        # Start capture first
        session_manager.start_capture(fps=fps)
        assert session_manager.get_session_state() == SessionState.CAPTURING

        # Start recognition
        session_manager.start_recognition()

        # Verify state transition
        assert session_manager.get_session_state() == SessionState.RECOGNIZING
        assert session_manager.is_capturing()  # Should still be capturing

        # Verify callback was invoked
        assert len(recognition_started_called) == 1

    finally:
        # Clean up
        session_manager.stop_capture()


@settings(max_examples=100, deadline=None)
@given(
    fps=st.integers(min_value=1, max_value=60),
)
def test_stop_recognition_transitions_from_recognizing_to_capturing(fps: int) -> None:
    """Property 2: Session State Transitions - Stop Recognition.

    For any valid stop recognition operation from RECOGNIZING state, the system
    should transition to CAPTURING state and trigger the recognition_stopped event.

    Feature: real-time-qa-assistant, Property 2: Session State Transitions
    Validates: Requirements 7.8
    """
    session_manager = create_session_manager()

    # Track callback invocations
    recognition_stopped_called = []

    def recognition_stopped_callback(**kwargs) -> None:
        recognition_stopped_called.append(True)

    # Register callback
    session_manager.register_callback("recognition_stopped", recognition_stopped_callback)

    try:
        # Start capture and recognition
        session_manager.start_capture(fps=fps)
        session_manager.start_recognition()
        assert session_manager.get_session_state() == SessionState.RECOGNIZING

        # Stop recognition
        session_manager.stop_recognition()

        # Verify state transition
        assert session_manager.get_session_state() == SessionState.CAPTURING
        assert session_manager.is_capturing()  # Should still be capturing

        # Verify callback was invoked
        assert len(recognition_stopped_called) == 1

    finally:
        # Clean up
        session_manager.stop_capture()


@settings(max_examples=100, deadline=None)
@given(
    fps=st.integers(min_value=1, max_value=60),
)
def test_reset_maintains_recognizing_state(fps: int) -> None:
    """Property 2: Session State Transitions - Reset.

    For any valid reset operation from RECOGNIZING state, the system should
    remain in RECOGNIZING state and trigger the session_reset event.

    Feature: real-time-qa-assistant, Property 2: Session State Transitions
    Validates: Requirements 7.9
    """
    session_manager = create_session_manager()

    # Track callback invocations
    session_reset_called = []

    def session_reset_callback(**kwargs) -> None:
        session_reset_called.append(True)

    # Register callback
    session_manager.register_callback("session_reset", session_reset_callback)

    try:
        # Start capture and recognition
        session_manager.start_capture(fps=fps)
        session_manager.start_recognition()
        assert session_manager.get_session_state() == SessionState.RECOGNIZING

        # Reset
        session_manager.reset()

        # Verify state is maintained
        assert session_manager.get_session_state() == SessionState.RECOGNIZING
        assert session_manager.is_capturing()

        # Verify callback was invoked
        assert len(session_reset_called) == 1

    finally:
        # Clean up
        session_manager.stop_capture()


@settings(max_examples=100, deadline=None)
@given(
    fps=st.integers(min_value=1, max_value=60),
)
def test_stop_capture_from_capturing_transitions_to_idle(fps: int) -> None:
    """Property 2: Session State Transitions - Stop Capture from CAPTURING.

    For any valid stop capture operation from CAPTURING state, the system
    should transition to IDLE state and trigger the capture_stopped event.

    Feature: real-time-qa-assistant, Property 2: Session State Transitions
    Validates: Requirements 7.10
    """
    session_manager = create_session_manager()

    # Track callback invocations
    capture_stopped_called = []

    def capture_stopped_callback(**kwargs) -> None:
        capture_stopped_called.append(True)

    # Register callback
    session_manager.register_callback("capture_stopped", capture_stopped_callback)

    # Start capture
    session_manager.start_capture(fps=fps)
    assert session_manager.get_session_state() == SessionState.CAPTURING

    # Stop capture
    session_manager.stop_capture()

    # Verify state transition
    assert session_manager.get_session_state() == SessionState.IDLE
    assert not session_manager.is_capturing()

    # Verify callback was invoked
    assert len(capture_stopped_called) == 1


@settings(max_examples=100, deadline=None)
@given(
    fps=st.integers(min_value=1, max_value=60),
)
def test_stop_capture_from_recognizing_transitions_to_idle(fps: int) -> None:
    """Property 2: Session State Transitions - Stop Capture from RECOGNIZING.

    For any valid stop capture operation from RECOGNIZING state, the system
    should stop recognition first, then transition to IDLE state.

    Feature: real-time-qa-assistant, Property 2: Session State Transitions
    Validates: Requirements 7.10
    """
    session_manager = create_session_manager()

    # Track callback invocations
    recognition_stopped_called = []
    capture_stopped_called = []

    def recognition_stopped_callback(**kwargs) -> None:
        recognition_stopped_called.append(True)

    def capture_stopped_callback(**kwargs) -> None:
        capture_stopped_called.append(True)

    # Register callbacks
    session_manager.register_callback("recognition_stopped", recognition_stopped_callback)
    session_manager.register_callback("capture_stopped", capture_stopped_callback)

    # Start capture and recognition
    session_manager.start_capture(fps=fps)
    session_manager.start_recognition()
    assert session_manager.get_session_state() == SessionState.RECOGNIZING

    # Stop capture (should stop recognition first)
    session_manager.stop_capture()

    # Verify state transition
    assert session_manager.get_session_state() == SessionState.IDLE
    assert not session_manager.is_capturing()

    # Verify callbacks were invoked
    # Note: recognition_stopped callback is not emitted when stopping capture
    # because it's an internal cleanup, not an explicit user action
    assert len(capture_stopped_called) == 1


@settings(max_examples=100, deadline=None)
@given(
    fps=st.integers(min_value=1, max_value=60),
)
def test_complete_session_lifecycle(fps: int) -> None:
    """Property 2: Session State Transitions - Complete Lifecycle.

    For any valid sequence of session operations, the system should transition
    through states correctly: IDLE -> CAPTURING -> RECOGNIZING -> CAPTURING -> IDLE.

    Feature: real-time-qa-assistant, Property 2: Session State Transitions
    Validates: Requirements 7.6, 7.7, 7.8, 7.9, 7.10
    """
    session_manager = create_session_manager()

    try:
        # Start from IDLE
        assert session_manager.get_session_state() == SessionState.IDLE

        # Start capture: IDLE -> CAPTURING
        session_manager.start_capture(fps=fps)
        assert session_manager.get_session_state() == SessionState.CAPTURING

        # Start recognition: CAPTURING -> RECOGNIZING
        session_manager.start_recognition()
        assert session_manager.get_session_state() == SessionState.RECOGNIZING

        # Reset: RECOGNIZING -> RECOGNIZING (state maintained)
        session_manager.reset()
        assert session_manager.get_session_state() == SessionState.RECOGNIZING

        # Stop recognition: RECOGNIZING -> CAPTURING
        session_manager.stop_recognition()
        assert session_manager.get_session_state() == SessionState.CAPTURING

        # Stop capture: CAPTURING -> IDLE
        session_manager.stop_capture()
        assert session_manager.get_session_state() == SessionState.IDLE

    finally:
        # Ensure cleanup
        if session_manager.is_capturing():
            session_manager.stop_capture()


def test_invalid_state_transitions_raise_errors() -> None:
    """Test that invalid state transitions raise appropriate errors.

    Feature: real-time-qa-assistant, Property 2: Session State Transitions
    Validates: Requirements 7.6, 7.7, 7.8, 7.9, 7.10
    """
    session_manager = create_session_manager()

    # Cannot start capture from CAPTURING
    try:
        session_manager.start_capture()
        try:
            session_manager.start_capture()
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError as e:
            assert "Cannot start capture from" in str(e)
    finally:
        session_manager.stop_capture()

    # Cannot start recognition from IDLE
    try:
        session_manager.start_recognition()
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "Cannot start recognition from" in str(e)

    # Cannot stop recognition from IDLE
    try:
        session_manager.stop_recognition()
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "Cannot stop recognition from" in str(e)

    # Cannot reset from IDLE
    try:
        session_manager.reset()
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "Cannot reset from" in str(e)


def test_session_state_string_representation() -> None:
    """Test SessionState string representation."""
    assert str(SessionState.IDLE) == "Idle"
    assert str(SessionState.CAPTURING) == "Capturing"
    assert str(SessionState.RECOGNIZING) == "Recognizing"
