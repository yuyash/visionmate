"""Main application entry point for VisionMate Desktop UI."""

import argparse
import logging
import os
import sys

# Ensure we're using the correct Qt platform plugin
if sys.platform == "darwin":  # macOS
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

from PySide6.QtWidgets import QApplication, QPushButton

from visionmate.capture.audio import SoundDeviceAudioCapture
from visionmate.capture.screen import MSSScreenCapture
from visionmate.core.input import InputMode
from visionmate.core.logging import LoggingConfig
from visionmate.core.session import SessionManager
from visionmate.core.version import __version__
from visionmate.ui import (
    MainWindow,
)
from visionmate.ui.widgets import (
    AudioWaveformWidget,
    DeviceControlsWidget,
    FPSControlWidget,
    InputModeWidget,
    VideoPreviewWidget,
)

logger = logging.getLogger(__name__)


class VisionMateApp:
    """Main application class for VisionMate."""

    def __init__(self):
        """Initialize the application."""
        logger.info("Initializing VisionMate application...")
        self.app = QApplication(sys.argv)
        logger.info("QApplication created")

        # Create capture instances
        self.screen_capture = MSSScreenCapture()
        logger.info("Screen capture created")
        self.audio_capture = SoundDeviceAudioCapture()
        logger.info("Audio capture created")

        # Create session manager
        self.session_manager = SessionManager(
            screen_capture=self.screen_capture,
            audio_capture=self.audio_capture,
        )
        logger.info("Session manager created")

        # Create main window
        self.main_window = MainWindow()
        logger.info("Main window created")

        # Create UI components
        self._setup_ui()
        logger.info("UI setup complete")

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        # Get layouts
        control_layout = self.main_window.get_control_panel_layout()
        preview_layout = self.main_window.get_preview_panel_layout()

        # Create input mode controls
        self.input_mode_widget = InputModeWidget(initial_mode=self.session_manager.get_input_mode())
        self.input_mode_widget.input_mode_changed.connect(self._on_input_mode_changed)
        control_layout.addWidget(self.input_mode_widget)

        # Create device controls
        self.device_controls = DeviceControlsWidget(
            screen_capture=self.screen_capture,
            audio_capture=self.audio_capture,
        )
        control_layout.addWidget(self.device_controls)

        # Create FPS control
        self.fps_control = FPSControlWidget(initial_fps=self.session_manager.get_capture_fps())
        self.fps_control.fps_changed.connect(self._on_fps_changed)
        control_layout.addWidget(self.fps_control)

        # Create start/stop buttons
        self.start_button = QPushButton("Start Capture")
        self.start_button.clicked.connect(self._start_capture)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Capture")
        self.stop_button.clicked.connect(self._stop_capture)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        # Add stretch to push controls to top
        control_layout.addStretch()

        # Create video preview
        self.video_preview = VideoPreviewWidget(capture=self.screen_capture)
        preview_layout.addWidget(self.video_preview)

        # Create audio waveform
        self.audio_waveform = AudioWaveformWidget(capture=self.audio_capture)
        preview_layout.addWidget(self.audio_waveform)

    def _on_input_mode_changed(self, mode: InputMode) -> None:
        """Handle input mode change.

        Args:
            mode: New input mode
        """
        logger.info(f"Input mode changed to {mode}")
        self.session_manager.set_input_mode(mode)

    def _on_fps_changed(self, fps: int) -> None:
        """Handle FPS change.

        Args:
            fps: New FPS value
        """
        logger.info(f"FPS changed to {fps}")
        self.session_manager.set_capture_fps(fps)

        # Update video preview FPS if it's running
        if self.video_preview.is_preview_active():
            self.video_preview.set_preview_fps(fps)

    def _start_capture(self) -> None:
        """Start capture and preview."""
        try:
            logger.info("Starting capture...")

            # Disable controls during capture
            self.input_mode_widget.set_capture_active(True)
            self.device_controls.set_capture_active(True)
            self.fps_control.set_capture_active(True)

            # Get current FPS from control
            fps = self.fps_control.get_fps()

            # Start capture through session manager
            self.session_manager.start_capture(fps=fps)
            logger.info("Capture started through session manager")

            # Start previews based on input mode
            current_mode = self.session_manager.get_input_mode()

            if current_mode.has_video:
                self.video_preview.start_preview(fps=fps)
                logger.info("Video preview started")

            if current_mode.has_audio:
                self.audio_waveform.start_preview(fps=30)
                logger.info("Audio waveform started")

            # Update button states
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

        except Exception as e:
            logger.error(f"Error starting capture: {e}", exc_info=True)
            # Re-enable controls on error
            self.input_mode_widget.set_capture_active(False)
            self.device_controls.set_capture_active(False)
            self.fps_control.set_capture_active(False)

    def _stop_capture(self) -> None:
        """Stop capture and preview."""
        try:
            logger.info("Stopping capture...")

            # Stop previews
            self.video_preview.stop_preview()
            self.audio_waveform.stop_preview()
            logger.info("Previews stopped")

            # Stop capture through session manager
            self.session_manager.stop_capture()
            logger.info("Capture stopped through session manager")

            # Re-enable controls
            self.input_mode_widget.set_capture_active(False)
            self.device_controls.set_capture_active(False)
            self.fps_control.set_capture_active(False)

            # Update button states
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

        except Exception as e:
            logger.error(f"Error stopping capture: {e}", exc_info=True)

    def run(self) -> int:
        """Run the application.

        Returns:
            Exit code
        """
        logger.info("Showing main window...")
        self.main_window.show()
        logger.info("Main window shown, starting event loop...")
        return self.app.exec()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="VisionMate - Real-time QA Assistant Desktop UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (console logging at INFO level)
  python -m visionmate.app

  # Run with DEBUG logging to console
  python -m visionmate.app --log-level DEBUG

  # Run with logging to file
  python -m visionmate.app --log-to-file

  # Run with custom log file path
  python -m visionmate.app --log-to-file --log-file /path/to/logfile.log

  # Run with both console and file logging
  python -m visionmate.app --log-to-file --log-to-console

  # Run with file logging only (no console output)
  python -m visionmate.app --log-to-file --no-log-to-console
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"VisionMate {__version__}",
        help="Show version information and exit",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )

    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="Enable logging to file (default: logs/visionmate.log)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: logs/visionmate.log)",
    )

    parser.add_argument(
        "--log-to-console",
        action="store_true",
        default=True,
        help="Enable logging to console (default: enabled)",
    )

    parser.add_argument(
        "--no-log-to-console",
        action="store_false",
        dest="log_to_console",
        help="Disable logging to console",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set up logging
    LoggingConfig.setup_logging(
        level=args.log_level,
        log_to_file=args.log_to_file,
        log_file_path=args.log_file,
        log_to_console=args.log_to_console,
    )

    logger.info(f"Starting VisionMate Desktop UI v{__version__}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Log to file: {args.log_to_file}")
    logger.info(f"Log to console: {args.log_to_console}")

    try:
        app = VisionMateApp()
        logger.info("Application initialized successfully")
        return app.run()
    except Exception as e:
        logger.critical(f"Error starting application: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
