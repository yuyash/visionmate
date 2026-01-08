"""
Visionmate CLI entry point.

This module provides the command-line interface for the Visionmate application.
"""

import argparse
import logging
import os
import sys
from importlib.metadata import metadata
from pathlib import Path
from typing import Optional

from visionmate.core.logging import setup_logging

# Load package metadata from pyproject.toml
_metadata = metadata("visionmate")
APP_NAME = _metadata["Name"]
APP_VERSION = _metadata["Version"]
APP_DESCRIPTION = _metadata["Summary"]


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} - {APP_DESCRIPTION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with desktop UI
  python -m visionmate
  
  # Launch in server mode with specific video source
  python -m visionmate --server-mode --video-source screen:0
  
  # Launch with custom VLM provider
  python -m visionmate --vlm-provider openai
  
  # Launch with debug logging
  python -m visionmate --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--server-mode",
        action="store_true",
        help="Run in server mode without desktop UI (web interface only)",
    )

    parser.add_argument(
        "--video-source",
        type=str,
        metavar="SOURCE",
        help=(
            "Video source specification. Format: TYPE:ID where TYPE is "
            "'screen', 'uvc', or 'rtsp'. Examples: 'screen:0', "
            "'uvc:device_id', 'rtsp:rtsp://example.com/stream'"
        ),
    )

    parser.add_argument(
        "--audio-source", type=str, metavar="DEVICE_ID", help="Audio source device ID"
    )

    parser.add_argument(
        "--vlm-provider",
        type=str,
        choices=["openai", "openai-compatible", "local"],
        metavar="PROVIDER",
        help=(
            "VLM provider to use. Choices: 'openai' (OpenAI Realtime API), "
            "'openai-compatible' (OpenAI-compatible HTTP API), 'local' (local model)"
        ),
    )

    parser.add_argument("--config", type=str, metavar="PATH", help="Path to configuration file")

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        metavar="LEVEL",
        help="Logging level (default: INFO)",
    )

    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")

    return parser


def main() -> int:
    """Main entry point for the Visionmate application.

    Returns:
        Exit code using os.EX_* constants:
        - os.EX_OK (0): Success
        - os.EX_NOINPUT (66): Cannot open input file
        - os.EX_SOFTWARE (70): Internal software error
    """
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Visionmate application")
    logger.debug(f"Command-line arguments: {args}")

    try:
        # Load configuration if specified
        config_path: Optional[Path] = None
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return os.EX_NOINPUT
            logger.info(f"Loading configuration from: {config_path}")

        # Launch application based on mode
        if args.server_mode:
            logger.info("Launching in server mode (web interface only)")
            # TODO: Implement server mode with web interface
            logger.warning("Server mode not yet implemented")
            return os.EX_SOFTWARE
        else:
            logger.info("Launching desktop UI")
            # Import here to avoid loading Qt when not needed
            try:
                from PySide6.QtWidgets import QApplication
            except ImportError:
                logger.error(
                    "PySide6 not installed. Desktop UI requires PySide6. "
                    "Install with: pip install PySide6"
                )
                return os.EX_SOFTWARE

            # Create Qt application
            app = QApplication(sys.argv)
            app.setApplicationName(APP_NAME)
            app.setOrganizationName(APP_NAME)
            app.setApplicationVersion(APP_VERSION)

            # Apply global stylesheet for flat design
            from visionmate.desktop.styles import get_global_stylesheet

            app.setStyleSheet(get_global_stylesheet())
            logger.debug("Global stylesheet applied")

            # Create and show main window
            from visionmate.desktop import MainWindow

            window = MainWindow()
            window.show()

            logger.info("Desktop UI launched successfully")

            # Start event loop
            return app.exec()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return os.EX_OK

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return os.EX_SOFTWARE


if __name__ == "__main__":
    sys.exit(main())
