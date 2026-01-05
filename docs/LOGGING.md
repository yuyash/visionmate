# VisionMate Logging Configuration

## Overview

VisionMate uses Python's built-in `logging` module for comprehensive logging across all components. Logging can be configured to output to console (stdout), file, or both.

## Command Line Options

### Basic Usage

```bash
# Run with default settings (INFO level, console only)
python -m visionmate.app

# Run with DEBUG logging
python -m visionmate.app --log-level DEBUG

# Run with logging to file
python -m visionmate.app --log-to-file

# Run with custom log file path
python -m visionmate.app --log-to-file --log-file /path/to/custom.log

# Run with both console and file logging
python -m visionmate.app --log-to-file --log-to-console

# Run with file logging only (no console output)
python -m visionmate.app --log-to-file --no-log-to-console
```

### Available Options

| Option                | Description                                              | Default             |
| --------------------- | -------------------------------------------------------- | ------------------- |
| `--version`           | Show version information and exit                        | -                   |
| `--log-level`         | Set logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL | INFO                |
| `--log-to-file`       | Enable logging to file                                   | False               |
| `--log-file`          | Path to log file                                         | logs/visionmate.log |
| `--log-to-console`    | Enable logging to console                                | True                |
| `--no-log-to-console` | Disable logging to console                               | -                   |

### Help and Version

```bash
# Show help
python -m visionmate.app --help

# Show version
python -m visionmate.app --version
```

## Log Levels

### DEBUG

Most detailed logging. Use for development and troubleshooting.

- All function calls and state changes
- Frame capture details
- Audio buffer operations
- UI events

### INFO (Default)

General informational messages about application flow.

- Application startup/shutdown
- Capture start/stop
- Device selection changes
- Major state transitions

### WARNING

Potentially problematic situations that don't prevent operation.

- Device not found
- Frame drops
- Buffer overflows
- Performance issues

### ERROR

Error events that might still allow the application to continue.

- Capture failures
- Device errors
- UI rendering issues

### CRITICAL

Severe errors that may cause the application to terminate.

- Initialization failures
- Unrecoverable errors

## Log Format

All log messages follow this structured format:

```
YYYY-MM-DD HH:MM:SS.mmm [ThreadName     ] LEVEL  filename.py:line - message
```

Example:

```
2026-01-04 22:08:42.644 [MainThread     ] INFO  app.py:235 - Starting VisionMate Desktop UI v0.3.0
2026-01-04 22:08:42.644 [MainThread     ] INFO  app.py:236 - Log level: INFO
2026-01-04 22:08:42.644 [MainThread     ] INFO  app.py:33 - Initializing VisionMate application...
2026-01-04 22:08:42.735 [MainThread     ] INFO  app.py:35 - QApplication created
2026-01-04 22:08:42.743 [Thread-1       ] DEBUG screen_capture.py:245 - Frame captured: 1920x1080
```

The format includes:

- **Timestamp**: Date and time with millisecond precision (for performance analysis)
- **Thread Name**: Name of the thread (fixed-width 15 chars, left-aligned)
- **Level**: Log level (fixed-width 5 chars: DEBUG, INFO, WARN, ERROR, CRIT)
- **Source**: Filename and line number where the log was generated
- **Message**: The actual log message

### Format Benefits

- **Fixed-width fields**: Thread names and log levels align vertically for easy scanning
- **Millisecond precision**: Useful for performance analysis and timing issues
- **Thread identification**: Essential for debugging multi-threaded capture operations
- **Compact source location**: Filename and line number for quick navigation

## Log File Location

### Default Location

```
logs/visionmate.log
```

The `logs/` directory is created automatically in the current working directory if it doesn't exist.

### Custom Location

Specify a custom path with `--log-file`:

```bash
python -m visionmate.app --log-to-file --log-file ~/my-logs/app.log
```

## Using Logging in Code

### Getting a Logger

```python
import logging

logger = logging.getLogger(__name__)
```

### Logging Messages

```python
# Debug level - detailed information
logger.debug("Frame captured: %dx%d", width, height)

# Info level - general information
logger.info("Capture started at %d FPS", fps)

# Warning level - potential issues
logger.warning("Frame drop detected, buffer full")

# Error level - errors that don't stop execution
logger.error("Failed to capture frame: %s", error)

# Critical level - severe errors
logger.critical("Cannot initialize capture device")
```

### Logging Exceptions

```python
try:
    # Some operation
    pass
except Exception as e:
    logger.error("Operation failed: %s", e, exc_info=True)
```

The `exc_info=True` parameter includes the full stack trace in the log.

## Examples

### Development Mode

Maximum verbosity for debugging:

```bash
python -m visionmate.app --log-level DEBUG --log-to-file --log-to-console
```

### Production Mode

Minimal console output, detailed file logging:

```bash
python -m visionmate.app --log-level INFO --log-to-file --no-log-to-console
```

### Troubleshooting Mode

Debug level with file logging:

```bash
python -m visionmate.app --log-level DEBUG --log-to-file --log-file debug.log
```

### Silent Mode

Only errors to file:

```bash
python -m visionmate.app --log-level ERROR --log-to-file --no-log-to-console
```

## Log Rotation

Currently, log files are not automatically rotated. For production use, consider implementing log rotation using:

1. **Python's RotatingFileHandler**:

   ```python
   from logging.handlers import RotatingFileHandler

   handler = RotatingFileHandler(
       'logs/visionmate.log',
       maxBytes=10*1024*1024,  # 10MB
       backupCount=5
   )
   ```

2. **External tools** (Linux/macOS):
   - `logrotate` utility
   - System log management tools

## Platform-Specific Notes

### macOS

- Logs directory created in application working directory
- Console output uses UTF-8 encoding
- File output uses UTF-8 encoding

### Windows

- Logs directory created in application working directory
- Console output uses system encoding
- File output uses UTF-8 encoding

### Linux

- Logs directory created in application working directory
- Console output uses UTF-8 encoding
- File output uses UTF-8 encoding

## Troubleshooting

### Log File Not Created

- Check write permissions for the logs directory
- Verify the path specified with `--log-file` is valid
- Ensure parent directories exist

### No Console Output

- Check if `--no-log-to-console` was specified
- Verify log level is appropriate (DEBUG shows more than ERROR)

### Too Much Output

- Increase log level: `--log-level WARNING` or `--log-level ERROR`
- Disable console logging: `--no-log-to-console`

### Missing Log Messages

- Decrease log level: `--log-level DEBUG`
- Check if logging is configured before the code runs

## Best Practices

1. **Use appropriate log levels**:

   - DEBUG: Development only
   - INFO: Normal operation
   - WARNING: Potential issues
   - ERROR: Actual errors
   - CRITICAL: Fatal errors

2. **Include context in messages**:

   ```python
   logger.info("Capture started: device=%s, fps=%d", device_name, fps)
   ```

3. **Use structured logging**:

   ```python
   logger.info("Operation completed", extra={
       'duration': elapsed_time,
       'items_processed': count
   })
   ```

4. **Don't log sensitive information**:

   - Avoid logging passwords, API keys, personal data
   - Sanitize user input before logging

5. **Use exc_info for exceptions**:
   ```python
   logger.error("Failed to process", exc_info=True)
   ```

## Future Enhancements

Planned logging improvements:

- [ ] Automatic log rotation
- [ ] JSON structured logging option
- [ ] Remote logging support (syslog, cloud services)
- [ ] Performance metrics logging
- [ ] Log filtering by component
- [ ] Real-time log viewer in UI
