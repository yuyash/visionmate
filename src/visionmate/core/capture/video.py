"""Video capture module for capturing video from various sources.

This module provides abstract interfaces and concrete implementations for
capturing video from screens, UVC devices, and RTSP streams. It includes
support for active window detection and frame cropping.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

from visionmate.core.models import (
    DeviceMetadata,
    DeviceType,
    Resolution,
    VideoFrame,
    VideoSourceType,
    WindowRegion,
)

logger = logging.getLogger(__name__)


class WindowCaptureMode(Enum):
    """Window capture mode for screen capture."""

    FULL_SCREEN = "full_screen"  # Capture entire screen
    ACTIVE_WINDOW = "active_window"  # Crop to active window only
    SELECTED_WINDOWS = "selected_windows"  # Crop to selected windows


class VideoCaptureInterface(ABC):
    """Abstract interface for video capture.

    This interface defines the contract for all video capture implementations,
    supporting screen capture, UVC devices, and RTSP streams with optional
    active window detection and cropping.
    """

    @abstractmethod
    def start_capture(
        self,
        device_id: str,
        fps: int = 30,
        resolution: Optional[tuple[int, int]] = None,
        enable_window_detection: bool = False,
    ) -> None:
        """Start capturing video.

        Args:
            device_id: Device identifier
            fps: Frame rate (1-60)
            resolution: Optional resolution override as (width, height)
            enable_window_detection: Enable active window detection and cropping

        Requirements: 1.1, 1.2, 1.3, 28.1-28.3
        """
        pass

    @abstractmethod
    def stop_capture(self) -> None:
        """Stop capturing video.

        Requirements: 1.1, 1.2, 1.3
        """
        pass

    @abstractmethod
    def get_frame(self) -> Optional[VideoFrame]:
        """Get the latest captured frame (cropped if window detection enabled).

        Returns:
            VideoFrame object with the latest frame, or None if no frame available.

        Requirements: 1.1, 1.2, 1.3, 28.5
        """
        pass

    @abstractmethod
    def is_capturing(self) -> bool:
        """Check if currently capturing.

        Returns:
            True if capture is active, False otherwise.

        Requirements: 1.1, 1.2, 1.3
        """
        pass

    @abstractmethod
    def get_source_info(self) -> DeviceMetadata:
        """Get source device metadata.

        Returns:
            DeviceMetadata object with device information.

        Requirements: 27.1-27.6
        """
        pass

    @abstractmethod
    def set_window_detection(self, enabled: bool) -> None:
        """Enable or disable active window detection.

        Args:
            enabled: True to enable window detection, False to disable

        Requirements: 28.6, 28.8
        """
        pass

    @abstractmethod
    def is_window_detection_enabled(self) -> bool:
        """Check if window detection is enabled.

        Returns:
            True if window detection is enabled, False otherwise.

        Requirements: 28.6, 28.8
        """
        pass


class WindowDetector:
    """Detects window regions in frames using platform-specific APIs.

    For screen capture, uses platform-specific APIs to detect the active window.
    For UVC/RTSP, uses computer vision techniques (future implementation).
    """

    def __init__(self):
        """Initialize the WindowDetector."""
        self._platform = platform.system()
        self._platform_available = self._check_platform_libraries()
        logger.info(
            f"WindowDetector initialized on platform: {self._platform}, "
            f"libraries available: {self._platform_available}"
        )

    def _check_platform_libraries(self) -> bool:
        """Check if platform-specific libraries are available.

        Returns:
            True if required libraries are installed, False otherwise.
        """
        try:
            if self._platform == "Darwin":
                import Quartz  # noqa: F401

                return True
            elif self._platform == "Windows":
                import win32gui  # noqa: F401

                return True
            elif self._platform == "Linux":
                import Xlib  # noqa: F401

                return True
            return False
        except ImportError:
            return False

    def detect_active_window(self) -> Optional[WindowRegion]:
        """Detect the active window region using platform-specific APIs.

        This method is used for screen capture to detect the currently active
        window and return its bounds. Excludes Visionmate's own window.

        Returns:
            WindowRegion object for the active window, or None if detection fails.

        Requirements: 28.1, 28.4, 28.7
        """
        try:
            if self._platform == "Darwin":
                return self._detect_active_window_macos()
            elif self._platform == "Windows":
                return self._detect_active_window_windows()
            elif self._platform == "Linux":
                return self._detect_active_window_linux()
            else:
                logger.warning(f"Unsupported platform for window detection: {self._platform}")
                return None
        except Exception as e:
            logger.error(f"Error detecting active window: {e}", exc_info=True)
            return None

    def detect_windows(
        self,
        frame: np.ndarray,
        min_area_percent: float = 10.0,
        min_confidence: float = 0.7,
    ) -> list[WindowRegion]:
        """Detect all window regions in frame using computer vision.

        Uses edge detection and contour analysis to find rectangular regions
        that likely represent windows or application interfaces.

        Args:
            frame: Input frame (RGB format)
            min_area_percent: Minimum region area as percentage of frame (default 10%)
            min_confidence: Minimum confidence threshold (default 0.7)

        Returns:
            List of detected window regions, sorted by area (largest first)

        Requirements: 28.2, 28.3, 28.4, 28.5
        """
        try:
            import cv2

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate minimum area threshold
            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_width * frame_height
            min_area = frame_area * (min_area_percent / 100.0)

            regions = []

            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate area
                area = w * h

                # Filter by minimum area
                if area < min_area:
                    continue

                # Calculate aspect ratio
                aspect_ratio = w / h if h > 0 else 0

                # Filter by aspect ratio (reasonable window shapes)
                # Most windows have aspect ratios between 0.5 and 3.0
                if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                    continue

                # Calculate rectangularity (how close the contour is to a rectangle)
                contour_area = cv2.contourArea(contour)
                rectangularity = contour_area / area if area > 0 else 0

                # Filter by rectangularity (should be close to 1.0 for rectangles)
                if rectangularity < 0.7:
                    continue

                # Calculate confidence based on rectangularity and relative size
                size_factor = min(area / frame_area, 1.0)
                confidence = (rectangularity * 0.7) + (size_factor * 0.3)

                # Filter by confidence
                if confidence < min_confidence:
                    continue

                # Create WindowRegion
                region = WindowRegion(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=confidence,
                )
                regions.append(region)

            # Sort by area (largest first)
            regions.sort(key=lambda r: r.area, reverse=True)

            logger.debug(f"Detected {len(regions)} window regions using computer vision")
            return regions

        except ImportError as e:
            logger.warning(f"OpenCV not installed: {e}. Install with: uv add 'opencv-python>=4.10'")
            return []
        except Exception as e:
            logger.error(f"Error detecting windows with computer vision: {e}", exc_info=True)
            return []

    def get_largest_region(
        self,
        regions: list[WindowRegion],
    ) -> Optional[WindowRegion]:
        """Get largest region from list.

        Args:
            regions: List of WindowRegion objects

        Returns:
            Largest WindowRegion by area, or None if list is empty
        """
        if not regions:
            return None
        return max(regions, key=lambda r: r.area)

    def filter_regions_by_confidence(
        self,
        regions: list[WindowRegion],
        min_confidence: float = 0.7,
    ) -> list[WindowRegion]:
        """Filter regions by confidence threshold.

        Args:
            regions: List of WindowRegion objects
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            Filtered list of WindowRegion objects
        """
        return [r for r in regions if r.confidence >= min_confidence]

    def detect_all_windows(self, monitor_bounds: Optional[dict] = None) -> list[WindowRegion]:
        """Detect all windows on screen using platform-specific APIs.

        Args:
            monitor_bounds: Optional monitor bounds dict with 'left', 'top', 'width', 'height'
                          to filter windows to a specific screen

        Returns:
            List of WindowRegion objects for all windows (excluding Visionmate),
            sorted by area (largest first)
        """
        try:
            if self._platform == "Darwin":
                return self._detect_all_windows_macos(monitor_bounds)
            elif self._platform == "Windows":
                return self._detect_all_windows_windows(monitor_bounds)
            elif self._platform == "Linux":
                return self._detect_all_windows_linux(monitor_bounds)
            else:
                logger.warning(f"Unsupported platform for window detection: {self._platform}")
                return []
        except Exception as e:
            logger.error(f"Error detecting all windows: {e}", exc_info=True)
            return []

    def capture_window_by_id(self, window_id: int) -> Optional[np.ndarray]:
        """Capture a specific window by its ID.

        This method captures only the specified window, regardless of whether
        it's visible or hidden behind other windows.

        Args:
            window_id: Platform-specific window ID

        Returns:
            Numpy array (RGB format) of the window content, or None if capture fails
        """
        try:
            if self._platform == "Darwin":
                return self._capture_window_macos(window_id)
            elif self._platform == "Windows":
                return self._capture_window_windows(window_id)
            elif self._platform == "Linux":
                return self._capture_window_linux(window_id)
            else:
                logger.warning(f"Unsupported platform for window capture: {self._platform}")
                return None
        except Exception as e:
            logger.error(f"Error capturing window: {e}", exc_info=True)
            return None

    def _capture_window_macos(self, window_id: int) -> Optional[np.ndarray]:
        """Capture a specific window on macOS using CGWindowListCreateImage.

        Args:
            window_id: CGWindowID

        Returns:
            Numpy array (RGB format) of the window content, or None if capture fails
        """
        try:
            from Quartz import (
                CGWindowListCreateImage,
                kCGWindowImageBoundsIgnoreFraming,
                kCGWindowListOptionIncludingWindow,
            )
            from Quartz.CoreGraphics import CGRectNull

            # Capture the specific window
            cg_image = CGWindowListCreateImage(
                CGRectNull,
                kCGWindowListOptionIncludingWindow,
                window_id,
                kCGWindowImageBoundsIgnoreFraming,
            )

            if cg_image is None:
                logger.warning(f"Failed to capture window {window_id}")
                return None

            # Convert CGImage to numpy array
            from Quartz import (
                CGImageGetBytesPerRow,
                CGImageGetDataProvider,
                CGImageGetHeight,
                CGImageGetWidth,
            )

            width = CGImageGetWidth(cg_image)
            height = CGImageGetHeight(cg_image)
            bytes_per_row = CGImageGetBytesPerRow(cg_image)

            # Get image data
            from Quartz import CGDataProviderCopyData

            data_provider = CGImageGetDataProvider(cg_image)
            data = CGDataProviderCopyData(data_provider)

            # Convert to numpy array
            # CGImage is in BGRA format
            bytes_data = bytes(data)
            img = np.frombuffer(bytes_data, dtype=np.uint8)

            # Reshape using bytes_per_row
            img = img.reshape((int(height), int(bytes_per_row)))

            # Extract only the actual image data (4 bytes per pixel for BGRA)
            bytes_per_pixel = 4
            img = img[:, : int(width) * bytes_per_pixel]
            img = img.reshape((int(height), int(width), bytes_per_pixel))

            # Convert BGRA to RGB
            img = img[:, :, :3]  # Drop alpha
            img = img[:, :, ::-1]  # BGR to RGB

            logger.debug(f"Captured window {window_id}: {width}x{height}")
            return img

        except ImportError as e:
            logger.warning(
                f"pyobjc-framework-Quartz not installed: {e}. "
                "Install with: uv add 'pyobjc-framework-Quartz>=10.3.1'"
            )
            return None
        except Exception as e:
            logger.error(f"Error capturing window on macOS: {e}", exc_info=True)
            return None

    def _capture_window_windows(self, window_id: int) -> Optional[np.ndarray]:
        """Capture a specific window on Windows using PrintWindow.

        Args:
            window_id: Window handle (HWND)

        Returns:
            Numpy array (RGB format) of the window content, or None if capture fails
        """
        try:
            import win32gui
            import win32ui

            # Get window dimensions
            rect = win32gui.GetWindowRect(window_id)
            x, y, right, bottom = rect
            width = right - x
            height = bottom - y

            if width <= 0 or height <= 0:
                logger.warning(f"Invalid window dimensions: {width}x{height}")
                return None

            # Create device context
            hwnd_dc = win32gui.GetWindowDC(window_id)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            # Create bitmap
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)

            # Print window to bitmap
            result = win32gui.PrintWindow(window_id, save_dc.GetSafeHdc(), 0)

            if result == 0:
                logger.warning(f"PrintWindow failed for window {window_id}")
                return None

            # Convert bitmap to numpy array
            bmp_str = save_bitmap.GetBitmapBits(True)

            img = np.frombuffer(bmp_str, dtype=np.uint8)
            img = img.reshape((height, width, 4))

            # Convert BGRA to RGB
            img = img[:, :, :3]
            img = img[:, :, ::-1]  # BGR to RGB

            # Cleanup
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(window_id, hwnd_dc)

            logger.debug(f"Captured window {window_id}: {width}x{height}")
            return img

        except ImportError as e:
            logger.warning(f"pywin32 not installed: {e}. Install with: uv add 'pywin32>=308'")
            return None
        except Exception as e:
            logger.error(f"Error capturing window on Windows: {e}", exc_info=True)
            return None

    def _capture_window_linux(self, window_id: int) -> Optional[np.ndarray]:
        """Capture a specific window on Linux using XGetImage.

        Args:
            window_id: X11 window ID

        Returns:
            Numpy array (RGB format) of the window content, or None if capture fails
        """
        try:
            from Xlib import X, display

            d = display.Display()
            window = d.create_resource_object("window", window_id)

            # Get window geometry
            geom = window.get_geometry()
            width, height = geom.width, geom.height

            if width <= 0 or height <= 0:
                logger.warning(f"Invalid window dimensions: {width}x{height}")
                return None

            # Capture window image
            raw_image = window.get_image(0, 0, width, height, X.ZPixmap, 0xFFFFFFFF)

            # Convert to numpy array
            img = np.frombuffer(raw_image.data, dtype=np.uint8)

            # X11 typically returns BGRA format
            img = img.reshape((height, width, 4))

            # Convert BGRA to RGB
            img = img[:, :, :3]
            img = img[:, :, ::-1]  # BGR to RGB

            logger.debug(f"Captured window {window_id}: {width}x{height}")
            return img

        except ImportError as e:
            logger.warning(
                f"python-xlib not installed: {e}. Install with: uv add 'python-xlib>=0.33'"
            )
            return None
        except Exception as e:
            logger.error(f"Error capturing window on Linux: {e}", exc_info=True)
            return None

    def _detect_active_window_linux(self) -> Optional[WindowRegion]:
        """Detect active window on Linux using X11.

        Returns:
            WindowRegion object for the active window, or None if detection fails.
        """
        try:
            from Xlib import X, display

            d = display.Display()
            root = d.screen().root

            # Get the active window
            active_window = root.get_full_property(
                d.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType
            )

            if active_window and active_window.value:
                window_id = active_window.value[0]
                window = d.create_resource_object("window", window_id)

                # Get window geometry
                geom = window.get_geometry()
                x, y = geom.x, geom.y
                width, height = geom.width, geom.height

                # Translate coordinates to root window
                coords = window.translate_coords(root, 0, 0)
                x, y = coords.x, coords.y

                if width > 0 and height > 0:
                    region = WindowRegion(x=x, y=y, width=width, height=height, confidence=1.0)
                    logger.debug(f"Detected active window on Linux: {region.to_tuple()}")
                    return region

            logger.debug("No active window found on Linux")
            return None

        except ImportError as e:
            logger.warning(
                f"python-xlib not installed: {e}. Install with: uv add 'python-xlib>=0.33'"
            )
            return None
        except Exception as e:
            logger.error(f"Error detecting active window on Linux: {e}", exc_info=True)
            return None

    def _detect_all_windows_macos(
        self, monitor_bounds: Optional[dict] = None
    ) -> list[WindowRegion]:
        """Detect all windows on macOS using Quartz API.

        Args:
            monitor_bounds: Optional monitor bounds to filter windows

        Returns:
            List of WindowRegion objects for all windows (excluding Visionmate)
        """
        try:
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowBounds,
                kCGWindowLayer,
                kCGWindowListOptionOnScreenOnly,
                kCGWindowName,
                kCGWindowNumber,
                kCGWindowOwnerName,
            )

            # Get list of on-screen windows
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, kCGNullWindowID
            )

            regions = []

            for window in window_list:
                # Only consider normal windows (layer 0)
                if window.get(kCGWindowLayer, -1) != 0:
                    continue

                # Get window title, owner name, and window ID
                window_name = window.get(kCGWindowName, "")
                owner_name = window.get(kCGWindowOwnerName, "")
                window_id = window.get(kCGWindowNumber, 0)

                bounds = window.get(kCGWindowBounds, {})
                if bounds:
                    x = int(bounds.get("X", 0))
                    y = int(bounds.get("Y", 0))
                    width = int(bounds.get("Width", 0))
                    height = int(bounds.get("Height", 0))

                    if width > 0 and height > 0:
                        # Filter by monitor bounds if provided
                        if monitor_bounds:
                            mon_left = monitor_bounds.get("left", 0)
                            mon_top = monitor_bounds.get("top", 0)
                            mon_width = monitor_bounds.get("width", 0)
                            mon_height = monitor_bounds.get("height", 0)

                            # Check if window is within monitor bounds
                            if not (
                                x >= mon_left
                                and y >= mon_top
                                and x + width <= mon_left + mon_width
                                and y + height <= mon_top + mon_height
                            ):
                                continue

                        region = WindowRegion(
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                            confidence=1.0,
                            title=window_name or owner_name,
                            window_id=window_id,
                        )
                        regions.append(region)

            # Sort by area (largest first)
            regions.sort(key=lambda r: r.area, reverse=True)

            logger.debug(f"Detected {len(regions)} windows on macOS")
            return regions

        except ImportError as e:
            logger.warning(
                f"pyobjc-framework-Quartz not installed: {e}. "
                "Install with: uv add 'pyobjc-framework-Quartz>=10.3.1'"
            )
            return []
        except Exception as e:
            logger.error(f"Error detecting all windows on macOS: {e}", exc_info=True)
            return []

    def _detect_all_windows_windows(
        self, monitor_bounds: Optional[dict] = None
    ) -> list[WindowRegion]:
        """Detect all windows on Windows using Win32 API.

        Args:
            monitor_bounds: Optional monitor bounds to filter windows

        Returns:
            List of WindowRegion objects for all windows (excluding Visionmate)
        """
        try:
            import win32gui

            regions = []

            def enum_callback(hwnd, _):
                # Only visible windows
                if not win32gui.IsWindowVisible(hwnd):
                    return

                # Get window title
                window_title = win32gui.GetWindowText(hwnd)

                # Skip windows without title
                if not window_title:
                    return

                rect = win32gui.GetWindowRect(hwnd)
                x, y, right, bottom = rect
                width = right - x
                height = bottom - y

                if width > 0 and height > 0:
                    # Filter by monitor bounds if provided
                    if monitor_bounds:
                        mon_left = monitor_bounds.get("left", 0)
                        mon_top = monitor_bounds.get("top", 0)
                        mon_width = monitor_bounds.get("width", 0)
                        mon_height = monitor_bounds.get("height", 0)

                        # Check if window is within monitor bounds
                        if not (
                            x >= mon_left
                            and y >= mon_top
                            and x + width <= mon_left + mon_width
                            and y + height <= mon_top + mon_height
                        ):
                            return

                    region = WindowRegion(
                        x=x, y=y, width=width, height=height, confidence=1.0, title=window_title
                    )
                    regions.append(region)

            win32gui.EnumWindows(enum_callback, None)

            # Sort by area (largest first)
            regions.sort(key=lambda r: r.area, reverse=True)

            logger.debug(f"Detected {len(regions)} windows on Windows")
            return regions

        except ImportError as e:
            logger.warning(f"pywin32 not installed: {e}. Install with: uv add 'pywin32>=308'")
            return []
        except Exception as e:
            logger.error(f"Error detecting all windows on Windows: {e}", exc_info=True)
            return []

    def _detect_all_windows_linux(
        self, monitor_bounds: Optional[dict] = None
    ) -> list[WindowRegion]:
        """Detect all windows on Linux using X11.

        Args:
            monitor_bounds: Optional monitor bounds to filter windows

        Returns:
            List of WindowRegion objects for all windows (excluding Visionmate)
        """
        try:
            from Xlib import X, display

            d = display.Display()
            root = d.screen().root

            regions = []

            # Get all windows
            window_ids = root.get_full_property(
                d.intern_atom("_NET_CLIENT_LIST"), X.AnyPropertyType
            )

            if window_ids and window_ids.value:
                for window_id in window_ids.value:
                    try:
                        window = d.create_resource_object("window", window_id)

                        # Get window title
                        window_name = window.get_wm_name()

                        # Skip windows without title
                        if not window_name:
                            continue

                        # Get window geometry
                        geom = window.get_geometry()
                        x, y = geom.x, geom.y
                        width, height = geom.width, geom.height

                        # Translate coordinates to root window
                        coords = window.translate_coords(root, 0, 0)
                        x, y = coords.x, coords.y

                        if width > 0 and height > 0:
                            # Filter by monitor bounds if provided
                            if monitor_bounds:
                                mon_left = monitor_bounds.get("left", 0)
                                mon_top = monitor_bounds.get("top", 0)
                                mon_width = monitor_bounds.get("width", 0)
                                mon_height = monitor_bounds.get("height", 0)

                                # Check if window is within monitor bounds
                                if not (
                                    x >= mon_left
                                    and y >= mon_top
                                    and x + width <= mon_left + mon_width
                                    and y + height <= mon_top + mon_height
                                ):
                                    continue

                            region = WindowRegion(
                                x=x,
                                y=y,
                                width=width,
                                height=height,
                                confidence=1.0,
                                title=window_name,
                            )
                            regions.append(region)

                    except Exception as e:
                        logger.debug(f"Error processing window: {e}")
                        continue

            # Sort by area (largest first)
            regions.sort(key=lambda r: r.area, reverse=True)

            logger.debug(f"Detected {len(regions)} windows on Linux")
            return regions

        except ImportError as e:
            logger.warning(
                f"python-xlib not installed: {e}. Install with: uv add 'python-xlib>=0.33'"
            )
            return []
        except Exception as e:
            logger.error(f"Error detecting all windows on Linux: {e}", exc_info=True)
            return []

    def _detect_active_window_macos(self) -> Optional[WindowRegion]:
        """Detect active window on macOS using Quartz API.

        Returns:
            WindowRegion object for the active window, or None if detection fails.
        """
        try:
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowBounds,
                kCGWindowLayer,
                kCGWindowListOptionOnScreenOnly,
                kCGWindowNumber,
            )

            # Get list of on-screen windows
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, kCGNullWindowID
            )

            # Find the frontmost window (layer 0)
            for window in window_list:
                if window.get(kCGWindowLayer, -1) == 0:
                    # Get window ID
                    window_id = window.get(kCGWindowNumber, 0)

                    bounds = window.get(kCGWindowBounds, {})
                    if bounds:
                        x = int(bounds.get("X", 0))
                        y = int(bounds.get("Y", 0))
                        width = int(bounds.get("Width", 0))
                        height = int(bounds.get("Height", 0))

                        if width > 0 and height > 0:
                            region = WindowRegion(
                                x=x,
                                y=y,
                                width=width,
                                height=height,
                                confidence=1.0,
                                window_id=window_id,
                            )
                            logger.debug(f"Detected active window on macOS: {region.to_tuple()}")
                            return region

            logger.debug("No active window found on macOS")
            return None

        except ImportError as e:
            logger.warning(
                f"pyobjc-framework-Quartz not installed: {e}. "
                "Install with: uv add 'pyobjc-framework-Quartz>=10.3.1'"
            )
            return None
        except Exception as e:
            logger.error(f"Error detecting active window on macOS: {e}", exc_info=True)
            return None

    def _detect_active_window_windows(self) -> Optional[WindowRegion]:
        """Detect active window on Windows using Win32 API.

        Returns:
            WindowRegion object for the active window, or None if detection fails.
        """
        try:
            import win32gui

            # Get the foreground window
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                rect = win32gui.GetWindowRect(hwnd)
                x, y, right, bottom = rect
                width = right - x
                height = bottom - y

                if width > 0 and height > 0:
                    region = WindowRegion(x=x, y=y, width=width, height=height, confidence=1.0)
                    logger.debug(f"Detected active window on Windows: {region.to_tuple()}")
                    return region

            logger.debug("No active window found on Windows")
            return None

        except ImportError as e:
            logger.warning(f"pywin32 not installed: {e}. Install with: uv add 'pywin32>=308'")
            return None
        except Exception as e:
            logger.error(f"Error detecting active window on Windows: {e}", exc_info=True)
            return None


class ScreenCapture(VideoCaptureInterface):
    """OS-native screen capture using mss library.

    Features:
    - Cross-platform screen capture using mss
    - Configurable FPS with precise timing control
    - Frame buffering for smooth playback
    - Automatic window detection enabled for screen capture
    - Platform-specific active window detection and cropping

    Requirements: 1.1, 1.4, 4.1, 4.2, 28.1, 28.7
    """

    def __init__(self, device_manager=None):
        """Initialize the ScreenCapture.

        Args:
            device_manager: Optional DeviceManager instance for metadata retrieval
        """
        self._device_manager = device_manager
        self._device_id: Optional[str] = None
        self._fps: int = 1
        self._resolution: Optional[Resolution] = None
        self._window_detection_enabled: bool = False  # Default: disabled until explicitly set
        self._window_capture_mode: WindowCaptureMode = WindowCaptureMode.FULL_SCREEN
        self._selected_window_titles: list[str] = []  # For SELECTED_WINDOWS mode
        self._capturing: bool = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_buffer: deque = deque(
            maxlen=5
        )  # Buffer last 5 frames. # TODO: The buffer size should be parameterized.
        self._frame_number: int = 0
        self._window_detector = WindowDetector()
        self._monitor_index: int = 0
        self._lock = threading.Lock()

        logger.info("ScreenCapture initialized")

    def start_capture(
        self,
        device_id: str,
        fps: int = 30,
        resolution: Optional[tuple[int, int]] = None,
        enable_window_detection: bool = False,
    ) -> None:
        """Start capturing video from screen.

        Args:
            device_id: Device identifier (e.g., "screen_1")
            fps: Frame rate (1-60)
            resolution: Optional resolution override (not used for screen capture)
            enable_window_detection: If True, crop to active window; if False, capture full screen

        Requirements: 1.1, 1.4, 4.1, 4.2, 28.1, 28.7
        """
        if self._capturing:
            logger.warning("Capture already in progress")
            return

        # Parse device_id to get monitor index
        try:
            self._monitor_index = int(device_id.split("_")[1])
        except (IndexError, ValueError) as e:
            logger.error(f"Invalid device_id format: {device_id}")
            raise ValueError(f"Invalid device_id format: {device_id}") from e

        self._device_id = device_id
        self._fps = max(1, min(fps, 60))
        self._resolution = Resolution.from_tuple(resolution) if resolution else None
        self._window_detection_enabled = enable_window_detection
        self._frame_number = 0
        self._frame_buffer.clear()

        # Start capture thread
        self._capturing = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        window_mode = "active window" if enable_window_detection else "full screen"
        logger.info(
            f"Started screen capture: device={device_id}, fps={self._fps}, mode={window_mode}"
        )

    def stop_capture(self) -> None:
        """Stop capturing video.

        Requirements: 1.1
        """
        if not self._capturing:
            logger.warning("Capture not in progress")
            return

        self._capturing = False

        # Wait for capture thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        logger.info(f"Stopped screen capture: device={self._device_id}")

    def get_frame(self) -> Optional[VideoFrame]:
        """Get the latest captured frame (cropped if window detection enabled).

        Returns:
            VideoFrame object with the latest frame, or None if no frame available.

        Requirements: 1.1, 28.5
        """
        with self._lock:
            if self._frame_buffer:
                return self._frame_buffer[-1]
            return None

    def is_capturing(self) -> bool:
        """Check if currently capturing.

        Returns:
            True if capture is active, False otherwise.

        Requirements: 1.1
        """
        return self._capturing

    def get_source_info(self) -> DeviceMetadata:
        """Get source device metadata.

        Returns:
            DeviceMetadata object with device information.

        Requirements: 27.1-27.6
        """
        if self._device_manager and self._device_id:
            return self._device_manager.get_device_metadata(self._device_id)
        else:
            # Return minimal metadata if device_manager not available
            return DeviceMetadata(
                device_id=self._device_id or "unknown",
                name="Screen",
                device_type=DeviceType.SCREEN,
            )

    def set_window_detection(self, enabled: bool) -> None:
        """Enable or disable active window detection.

        Args:
            enabled: True to enable window detection (crop to active window),
                    False to disable (capture full screen)

        Requirements: 28.6, 28.7
        """
        self._window_detection_enabled = enabled
        if enabled:
            self._window_capture_mode = WindowCaptureMode.ACTIVE_WINDOW
        else:
            self._window_capture_mode = WindowCaptureMode.FULL_SCREEN
        mode = "active window" if enabled else "full screen"
        logger.info(f"Window detection set to: {mode}")

    def is_window_detection_enabled(self) -> bool:
        """Check if window detection is enabled.

        Returns:
            True if window detection is enabled, False otherwise.

        Requirements: 28.6, 28.7
        """
        return self._window_detection_enabled

    def set_window_capture_mode(
        self, mode: WindowCaptureMode, selected_titles: Optional[list[str]] = None
    ) -> None:
        """Set window capture mode.

        Args:
            mode: Window capture mode
            selected_titles: List of window titles to capture (for SELECTED_WINDOWS mode)
        """
        self._window_capture_mode = mode
        self._window_detection_enabled = mode != WindowCaptureMode.FULL_SCREEN

        if mode == WindowCaptureMode.SELECTED_WINDOWS:
            self._selected_window_titles = selected_titles or []
            logger.info(
                f"Window capture mode set to: SELECTED_WINDOWS "
                f"({len(self._selected_window_titles)} windows)"
            )
        else:
            self._selected_window_titles = []
            logger.info(f"Window capture mode set to: {mode.value}")

    def get_window_capture_mode(self) -> WindowCaptureMode:
        """Get current window capture mode.

        Returns:
            Current WindowCaptureMode
        """
        return self._window_capture_mode

    def get_selected_window_titles(self) -> list[str]:
        """Get list of selected window titles.

        Returns:
            List of window titles (for SELECTED_WINDOWS mode)
        """
        return self._selected_window_titles.copy()

    def get_available_windows(self) -> list[WindowRegion]:
        """Get list of available windows on the current screen.

        Returns:
            List of WindowRegion objects for all windows on the screen
        """
        try:
            import mss

            with mss.mss() as sct:
                # Get monitor bounds
                if self._monitor_index > 0 and self._monitor_index < len(sct.monitors):
                    monitor = sct.monitors[self._monitor_index]
                    return self._window_detector.detect_all_windows(monitor)
                else:
                    return []
        except Exception as e:
            logger.error(f"Error getting available windows: {e}", exc_info=True)
            return []

    def _crop_to_region(
        self, img: np.ndarray, region: WindowRegion, monitor: dict
    ) -> tuple[np.ndarray, bool]:
        """Crop image to window region.

        Args:
            img: Input image
            region: Window region to crop to
            monitor: Monitor bounds dict

        Returns:
            Tuple of (cropped_image, is_cropped)
        """
        # Adjust coordinates relative to monitor
        x = max(0, region.x - monitor["left"])
        y = max(0, region.y - monitor["top"])
        x2 = min(img.shape[1], x + region.width)
        y2 = min(img.shape[0], y + region.height)

        if x < x2 and y < y2:
            cropped_img = img[y:y2, x:x2]
            logger.debug(f"Cropped frame to window region: {region.to_tuple()}")
            return cropped_img, True
        else:
            logger.warning(f"Invalid crop region: {region.to_tuple()}")
            return img, False

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread.

        Captures frames at the specified FPS rate, detects active window,
        and crops frames to the window region.
        """
        import mss

        frame_interval = 1.0 / self._fps
        next_capture_time = time.time()

        try:
            with mss.mss() as sct:
                # Get monitor (add 1 because monitor 0 is "all monitors")
                monitor = sct.monitors[self._monitor_index]

                logger.debug(f"Capturing from monitor: {monitor}")

                while self._capturing:
                    current_time = time.time()

                    # Wait until next capture time
                    if current_time < next_capture_time:
                        time.sleep(0.001)  # Sleep 1ms to avoid busy waiting
                        continue

                    # Capture frame
                    try:
                        # Capture full screen
                        screenshot = sct.grab(monitor)
                        img = np.array(screenshot)

                        # Convert BGRA to RGB
                        img = img[:, :, :3]  # Drop alpha channel
                        img = img[:, :, ::-1]  # BGR to RGB

                        # Get current timestamp
                        timestamp = datetime.now(timezone.utc)

                        # Detect windows based on capture mode
                        detected_regions = []
                        active_region = None
                        is_cropped = False

                        if self._window_capture_mode == WindowCaptureMode.ACTIVE_WINDOW:
                            # Detect and crop to active window only
                            window_region = self._window_detector.detect_active_window()
                            if window_region:
                                detected_regions = [window_region]
                                active_region = window_region

                                # Crop image to window region
                                img, is_cropped = self._crop_to_region(img, window_region, monitor)

                        elif self._window_capture_mode == WindowCaptureMode.SELECTED_WINDOWS:
                            # Capture specific windows directly (not cropped from screen)
                            all_windows = self._window_detector.detect_all_windows(monitor)
                            selected_windows = [
                                w for w in all_windows if w.title in self._selected_window_titles
                            ]

                            if selected_windows:
                                detected_regions = selected_windows
                                # Use first selected window as active region
                                active_region = selected_windows[0]

                                # Capture window directly by ID (no other windows visible)
                                if active_region.window_id is not None:
                                    window_img = self._window_detector.capture_window_by_id(
                                        active_region.window_id
                                    )
                                    if window_img is not None:
                                        img = window_img
                                        is_cropped = True
                                        logger.debug(
                                            f"Captured window directly: {active_region.title}"
                                        )
                                    else:
                                        # Fallback to cropping if direct capture fails
                                        img, is_cropped = self._crop_to_region(
                                            img, active_region, monitor
                                        )
                                else:
                                    # No window ID, fallback to cropping
                                    img, is_cropped = self._crop_to_region(
                                        img, active_region, monitor
                                    )

                        # FULL_SCREEN mode: no cropping, use full image

                        # Get resolution
                        height, width = img.shape[:2]
                        resolution = Resolution(width=width, height=height)

                        # Ensure device_id is set (should always be true in capture loop)
                        if self._device_id is None:
                            logger.error("Device ID is None during capture")
                            break

                        # Create VideoFrame
                        frame = VideoFrame(
                            image=img,
                            timestamp=timestamp,
                            source_id=self._device_id,
                            source_type=VideoSourceType.SCREEN,
                            resolution=resolution,
                            fps=self._fps,
                            frame_number=self._frame_number,
                            detected_regions=detected_regions,
                            active_region=active_region,
                            is_cropped=is_cropped,
                        )

                        # Add to buffer
                        with self._lock:
                            self._frame_buffer.append(frame)

                        self._frame_number += 1

                        # Schedule next capture
                        next_capture_time += frame_interval

                        # If we're falling behind, reset timing
                        if next_capture_time < current_time:
                            next_capture_time = current_time + frame_interval

                    except Exception as e:
                        logger.error(f"Error capturing frame: {e}", exc_info=True)
                        time.sleep(frame_interval)

        except Exception as e:
            logger.error(f"Error in capture loop: {e}", exc_info=True)
        finally:
            self._capturing = False
            logger.debug("Capture loop ended")


class UVCCapture(VideoCaptureInterface):
    """UVC device capture using OpenCV.

    Features:
    - Cross-platform UVC device capture using OpenCV
    - Configurable FPS with precise timing control
    - Frame buffering for smooth playback
    - User-controlled window detection (optional)
    - Computer vision-based window detection and cropping

    Requirements: 1.2, 1.5, 28.2, 28.8
    """

    def __init__(self, device_manager=None):
        """Initialize the UVCCapture.

        Args:
            device_manager: Optional DeviceManager instance for metadata retrieval
        """
        self._device_manager = device_manager
        self._device_id: Optional[str] = None
        self._fps: int = 1
        self._resolution: Optional[Resolution] = None
        self._window_detection_enabled: bool = False  # User-controlled
        self._capturing: bool = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_buffer: deque = deque(maxlen=5)
        self._frame_number: int = 0
        self._window_detector = WindowDetector()
        self._lock = threading.Lock()
        self._cap = None  # cv2.VideoCapture object (type: ignore for dynamic import)

        logger.info("UVCCapture initialized")

    def start_capture(
        self,
        device_id: str,
        fps: int = 30,
        resolution: Optional[tuple[int, int]] = None,
        enable_window_detection: bool = False,
    ) -> None:
        """Start capturing video from UVC device.

        Args:
            device_id: Device identifier (e.g., "uvc_0")
            fps: Frame rate (1-60)
            resolution: Optional resolution override as (width, height)
            enable_window_detection: Enable computer vision window detection and cropping

        Requirements: 1.2, 1.5, 28.2, 28.8
        """
        if self._capturing:
            logger.warning("Capture already in progress")
            return

        # Parse device_id to get device index
        try:
            device_index = int(device_id.split("_")[1])
        except (IndexError, ValueError) as e:
            logger.error(f"Invalid device_id format: {device_id}")
            raise ValueError(f"Invalid device_id format: {device_id}") from e

        try:
            import cv2

            # Open video capture device
            self._cap = cv2.VideoCapture(device_index)

            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open UVC device: {device_id}")

            # Set resolution if specified
            if resolution:
                width, height = resolution
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                logger.debug(f"Set resolution to: {width}x{height}")

            # Set FPS
            self._cap.set(cv2.CAP_PROP_FPS, fps)

            self._device_id = device_id
            self._fps = max(1, min(fps, 60))
            self._resolution = Resolution.from_tuple(resolution) if resolution else None
            self._window_detection_enabled = enable_window_detection
            self._frame_number = 0
            self._frame_buffer.clear()

            # Start capture thread
            self._capturing = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

            window_mode = "with window detection" if enable_window_detection else "full frame"
            logger.info(
                f"Started UVC capture: device={device_id}, fps={self._fps}, mode={window_mode}"
            )

        except Exception as e:
            logger.error(f"Error starting UVC capture: {e}", exc_info=True)
            if self._cap:
                self._cap.release()
                self._cap = None
            raise

    def stop_capture(self) -> None:
        """Stop capturing video.

        Requirements: 1.2
        """
        if not self._capturing:
            logger.warning("Capture not in progress")
            return

        self._capturing = False

        # Wait for capture thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        # Release video capture
        if self._cap:
            self._cap.release()
            self._cap = None

        logger.info(f"Stopped UVC capture: device={self._device_id}")

    def get_frame(self) -> Optional[VideoFrame]:
        """Get the latest captured frame (cropped if window detection enabled).

        Returns:
            VideoFrame object with the latest frame, or None if no frame available.

        Requirements: 1.2, 28.5
        """
        with self._lock:
            if self._frame_buffer:
                return self._frame_buffer[-1]
            return None

    def is_capturing(self) -> bool:
        """Check if currently capturing.

        Returns:
            True if capture is active, False otherwise.

        Requirements: 1.2
        """
        return self._capturing

    def get_source_info(self) -> DeviceMetadata:
        """Get source device metadata.

        Returns:
            DeviceMetadata object with device information.

        Requirements: 27.1-27.6
        """
        if self._device_manager and self._device_id:
            return self._device_manager.get_device_metadata(self._device_id)
        else:
            # Return minimal metadata if device_manager not available
            return DeviceMetadata(
                device_id=self._device_id or "unknown",
                name="UVC Device",
                device_type=DeviceType.UVC,
            )

    def set_window_detection(self, enabled: bool) -> None:
        """Enable or disable window detection.

        Args:
            enabled: True to enable window detection (crop to detected windows),
                    False to disable (capture full frame)

        Requirements: 28.6, 28.8
        """
        self._window_detection_enabled = enabled
        mode = "enabled" if enabled else "disabled"
        logger.info(f"Window detection set to: {mode}")

    def is_window_detection_enabled(self) -> bool:
        """Check if window detection is enabled.

        Returns:
            True if window detection is enabled, False otherwise.

        Requirements: 28.6, 28.8
        """
        return self._window_detection_enabled

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread.

        Captures frames at the specified FPS rate, optionally detects windows,
        and crops frames to the largest detected window region.
        """
        import cv2

        frame_interval = 1.0 / self._fps
        next_capture_time = time.time()

        try:
            while self._capturing and self._cap and self._cap.isOpened():
                current_time = time.time()

                # Wait until next capture time
                if current_time < next_capture_time:
                    time.sleep(0.001)  # Sleep 1ms to avoid busy waiting
                    continue

                # Capture frame
                try:
                    ret, frame = self._cap.read()

                    if not ret or frame is None:
                        logger.warning("Failed to read frame from UVC device")
                        time.sleep(frame_interval)
                        continue

                    # Convert BGR to RGB
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Get current timestamp
                    timestamp = datetime.now(timezone.utc)

                    # Detect windows if enabled
                    detected_regions = []
                    active_region = None
                    is_cropped = False

                    if self._window_detection_enabled:
                        # Use computer vision to detect windows
                        detected_regions = self._window_detector.detect_windows(img)

                        if detected_regions:
                            # Use largest detected region
                            active_region = self._window_detector.get_largest_region(
                                detected_regions
                            )

                            if active_region:
                                # Crop image to window region
                                x, y, w, h = active_region.to_tuple()
                                x2 = min(img.shape[1], x + w)
                                y2 = min(img.shape[0], y + h)

                                if x < x2 and y < y2:
                                    img = img[y:y2, x:x2]
                                    is_cropped = True
                                    logger.debug(
                                        f"Cropped UVC frame to window region: {active_region.to_tuple()}"
                                    )

                    # Get resolution
                    height, width = img.shape[:2]
                    resolution = Resolution(width=width, height=height)

                    # Ensure device_id is set
                    if self._device_id is None:
                        logger.error("Device ID is None during capture")
                        break

                    # Create VideoFrame
                    video_frame = VideoFrame(
                        image=img,
                        timestamp=timestamp,
                        source_id=self._device_id,
                        source_type=VideoSourceType.UVC,
                        resolution=resolution,
                        fps=self._fps,
                        frame_number=self._frame_number,
                        detected_regions=detected_regions,
                        active_region=active_region,
                        is_cropped=is_cropped,
                    )

                    # Add to buffer
                    with self._lock:
                        self._frame_buffer.append(video_frame)

                    self._frame_number += 1

                    # Schedule next capture
                    next_capture_time += frame_interval

                    # If we're falling behind, reset timing
                    if next_capture_time < current_time:
                        next_capture_time = current_time + frame_interval

                except Exception as e:
                    logger.error(f"Error capturing frame: {e}", exc_info=True)
                    time.sleep(frame_interval)

        except Exception as e:
            logger.error(f"Error in capture loop: {e}", exc_info=True)
        finally:
            self._capturing = False
            logger.debug("Capture loop ended")


class RTSPCapture(VideoCaptureInterface):
    """RTSP stream capture using OpenCV.

    Features:
    - Cross-platform RTSP stream capture using OpenCV
    - Configurable FPS with precise timing control
    - Frame buffering for smooth playback
    - User-controlled window detection (optional)
    - Computer vision-based window detection and cropping
    - Connection error handling and retry logic

    Requirements: 1.3, 1.9, 28.3, 28.8
    """

    def __init__(self, device_manager=None):
        """Initialize the RTSPCapture.

        Args:
            device_manager: Optional DeviceManager instance for metadata retrieval
        """
        self._device_manager = device_manager
        self._device_id: Optional[str] = None
        self._rtsp_url: Optional[str] = None
        self._fps: int = 1
        self._resolution: Optional[Resolution] = None
        self._window_detection_enabled: bool = False  # User-controlled
        self._capturing: bool = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_buffer: deque = deque(maxlen=5)
        self._frame_number: int = 0
        self._window_detector = WindowDetector()
        self._lock = threading.Lock()
        self._cap = None  # cv2.VideoCapture object (type: ignore for dynamic import)

        logger.info("RTSPCapture initialized")

    def start_capture(
        self,
        device_id: str,
        fps: int = 30,
        resolution: Optional[tuple[int, int]] = None,
        enable_window_detection: bool = False,
    ) -> None:
        """Start capturing video from RTSP stream.

        Args:
            device_id: Device identifier (e.g., "rtsp_rtsp://...")
            fps: Frame rate (1-60)
            resolution: Optional resolution override as (width, height)
            enable_window_detection: Enable computer vision window detection and cropping

        Requirements: 1.3, 1.9, 28.3, 28.8
        """
        if self._capturing:
            logger.warning("Capture already in progress")
            return

        # Parse device_id to extract RTSP URL
        # Format: "rtsp_<url>" where <url> is the RTSP URL
        try:
            if not device_id.startswith("rtsp_"):
                raise ValueError(f"Invalid RTSP device_id format: {device_id}")

            self._rtsp_url = device_id[5:]  # Remove "rtsp_" prefix

            if not self._rtsp_url.startswith("rtsp://"):
                raise ValueError(f"Invalid RTSP URL: {self._rtsp_url}")

        except (IndexError, ValueError) as e:
            logger.error(f"Invalid device_id format: {device_id}")
            raise ValueError(f"Invalid device_id format: {device_id}") from e

        try:
            import cv2

            # Open RTSP stream
            logger.info(f"Connecting to RTSP stream: {self._rtsp_url}")
            self._cap = cv2.VideoCapture(self._rtsp_url)

            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to connect to RTSP stream: {self._rtsp_url}")

            # Set resolution if specified
            if resolution:
                width, height = resolution
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                logger.debug(f"Set resolution to: {width}x{height}")

            # Set FPS
            self._cap.set(cv2.CAP_PROP_FPS, fps)

            self._device_id = device_id
            self._fps = max(1, min(fps, 60))
            self._resolution = Resolution.from_tuple(resolution) if resolution else None
            self._window_detection_enabled = enable_window_detection
            self._frame_number = 0
            self._frame_buffer.clear()

            # Start capture thread
            self._capturing = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

            window_mode = "with window detection" if enable_window_detection else "full frame"
            logger.info(
                f"Started RTSP capture: url={self._rtsp_url}, fps={self._fps}, mode={window_mode}"
            )

        except Exception as e:
            logger.error(f"Error starting RTSP capture: {e}", exc_info=True)
            if self._cap:
                self._cap.release()
                self._cap = None
            raise

    def stop_capture(self) -> None:
        """Stop capturing video.

        Requirements: 1.3
        """
        if not self._capturing:
            logger.warning("Capture not in progress")
            return

        self._capturing = False

        # Wait for capture thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        # Release video capture
        if self._cap:
            self._cap.release()
            self._cap = None

        logger.info(f"Stopped RTSP capture: url={self._rtsp_url}")

    def get_frame(self) -> Optional[VideoFrame]:
        """Get the latest captured frame (cropped if window detection enabled).

        Returns:
            VideoFrame object with the latest frame, or None if no frame available.

        Requirements: 1.3, 28.5
        """
        with self._lock:
            if self._frame_buffer:
                return self._frame_buffer[-1]
            return None

    def is_capturing(self) -> bool:
        """Check if currently capturing.

        Returns:
            True if capture is active, False otherwise.

        Requirements: 1.3
        """
        return self._capturing

    def get_source_info(self) -> DeviceMetadata:
        """Get source device metadata.

        Returns:
            DeviceMetadata object with device information.

        Requirements: 27.1-27.6
        """
        if self._device_manager and self._device_id:
            return self._device_manager.get_device_metadata(self._device_id)
        else:
            # Return minimal metadata if device_manager not available
            return DeviceMetadata(
                device_id=self._device_id or "unknown",
                name=f"RTSP Stream: {self._rtsp_url}" if self._rtsp_url else "RTSP Stream",
                device_type=DeviceType.RTSP,
            )

    def set_window_detection(self, enabled: bool) -> None:
        """Enable or disable window detection.

        Args:
            enabled: True to enable window detection (crop to detected windows),
                    False to disable (capture full frame)

        Requirements: 28.6, 28.8
        """
        self._window_detection_enabled = enabled
        mode = "enabled" if enabled else "disabled"
        logger.info(f"Window detection set to: {mode}")

    def is_window_detection_enabled(self) -> bool:
        """Check if window detection is enabled.

        Returns:
            True if window detection is enabled, False otherwise.

        Requirements: 28.6, 28.8
        """
        return self._window_detection_enabled

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread.

        Captures frames at the specified FPS rate, optionally detects windows,
        and crops frames to the largest detected window region.
        Handles connection errors and attempts reconnection.
        """
        import cv2

        frame_interval = 1.0 / self._fps
        next_capture_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 10

        try:
            while self._capturing:
                current_time = time.time()

                # Wait until next capture time
                if current_time < next_capture_time:
                    time.sleep(0.001)  # Sleep 1ms to avoid busy waiting
                    continue

                # Check if capture is still open
                if not self._cap or not self._cap.isOpened():
                    logger.error("RTSP stream connection lost")
                    consecutive_failures += 1

                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            f"Too many consecutive failures ({consecutive_failures}), stopping capture"
                        )
                        break

                    # Try to reconnect
                    logger.info("Attempting to reconnect to RTSP stream...")
                    time.sleep(1.0)  # Wait before reconnecting

                    if self._cap:
                        self._cap.release()

                    # Type assertion: self._rtsp_url is guaranteed to be set in start_capture
                    assert self._rtsp_url is not None
                    self._cap = cv2.VideoCapture(self._rtsp_url)

                    if not self._cap.isOpened():
                        logger.error("Failed to reconnect to RTSP stream")
                        time.sleep(frame_interval)
                        continue

                    logger.info("Successfully reconnected to RTSP stream")
                    consecutive_failures = 0

                # Capture frame
                try:
                    ret, frame = self._cap.read()

                    if not ret or frame is None:
                        logger.warning("Failed to read frame from RTSP stream")
                        consecutive_failures += 1
                        time.sleep(frame_interval)
                        continue

                    # Reset failure counter on successful read
                    consecutive_failures = 0

                    # Convert BGR to RGB
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Get current timestamp
                    timestamp = datetime.now(timezone.utc)

                    # Detect windows if enabled
                    detected_regions = []
                    active_region = None
                    is_cropped = False

                    if self._window_detection_enabled:
                        # Use computer vision to detect windows
                        detected_regions = self._window_detector.detect_windows(img)

                        if detected_regions:
                            # Use largest detected region
                            active_region = self._window_detector.get_largest_region(
                                detected_regions
                            )

                            if active_region:
                                # Crop image to window region
                                x, y, w, h = active_region.to_tuple()
                                x2 = min(img.shape[1], x + w)
                                y2 = min(img.shape[0], y + h)

                                if x < x2 and y < y2:
                                    img = img[y:y2, x:x2]
                                    is_cropped = True
                                    logger.debug(
                                        f"Cropped RTSP frame to window region: {active_region.to_tuple()}"
                                    )

                    # Get resolution
                    height, width = img.shape[:2]
                    resolution = Resolution(width=width, height=height)

                    # Ensure device_id is set
                    if self._device_id is None:
                        logger.error("Device ID is None during capture")
                        break

                    # Create VideoFrame
                    video_frame = VideoFrame(
                        image=img,
                        timestamp=timestamp,
                        source_id=self._device_id,
                        source_type=VideoSourceType.RTSP,
                        resolution=resolution,
                        fps=self._fps,
                        frame_number=self._frame_number,
                        detected_regions=detected_regions,
                        active_region=active_region,
                        is_cropped=is_cropped,
                    )

                    # Add to buffer
                    with self._lock:
                        self._frame_buffer.append(video_frame)

                    self._frame_number += 1

                    # Schedule next capture
                    next_capture_time += frame_interval

                    # If we're falling behind, reset timing
                    if next_capture_time < current_time:
                        next_capture_time = current_time + frame_interval

                except Exception as e:
                    logger.error(f"Error capturing frame: {e}", exc_info=True)
                    consecutive_failures += 1
                    time.sleep(frame_interval)

        except Exception as e:
            logger.error(f"Error in capture loop: {e}", exc_info=True)
        finally:
            self._capturing = False
            logger.debug("Capture loop ended")
