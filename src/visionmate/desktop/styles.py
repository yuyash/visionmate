"""
Global stylesheet definitions for Visionmate desktop application.

This module provides centralized style definitions that can be applied
application-wide, ensuring consistent flat design across all components.
"""

# Global application stylesheet
# This is applied to QApplication and affects all widgets
GLOBAL_STYLESHEET = """
/* Flat design baseline for all components */

QLabel {
    background-color: transparent;
    border: none;
}

QGroupBox {
    font-weight: bold;
    border: none;
    margin-top: 10px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
}

QRadioButton {
    border: none;
    background-color: transparent;
    spacing: 5px;
}

QComboBox {
    border: 1px solid palette(mid);
    border-radius: 3px;
    padding: 5px;
    background-color: white;
    min-height: 20px;
}

QComboBox:focus {
    border: 1px solid palette(highlight);
}

QComboBox::drop-down {
    border: none;
    background-color: transparent;
}

QComboBox::down-arrow {
    image: none;
    border: none;
    background-color: transparent;
}

QComboBox QAbstractItemView {
    background-color: white;
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
    outline: none;
    border: 1px solid palette(mid);
    padding: 2px;
}

QComboBox QAbstractItemView::item {
    height: 28px;
    padding: 4px 8px;
    border: none;
}

QComboBox QAbstractItemView::item:hover {
    background-color: palette(light);
}

QComboBox QAbstractItemView::item:selected {
    background-color: palette(highlight);
    color: palette(highlighted-text);
}

QPushButton {
    border: 1px solid palette(mid);
    border-radius: 3px;
    padding: 5px;
    background-color: palette(button);
}

QPushButton:hover {
    background-color: palette(light);
    border: 1px solid palette(dark);
}

QPushButton:pressed {
    background-color: palette(mid);
}

QListWidget {
    border: 1px solid palette(mid);
    border-radius: 3px;
    background-color: palette(base);
}

QListWidget::item:hover {
    background-color: palette(light);
}

QListWidget::item:selected {
    background-color: palette(highlight);
    color: palette(highlighted-text);
}

QListWidget::item:selected:hover {
    background-color: palette(highlight);
    color: palette(highlighted-text);
}
"""


def get_global_stylesheet() -> str:
    """Get the global application stylesheet.

    Returns:
        Global stylesheet string
    """
    return GLOBAL_STYLESHEET
