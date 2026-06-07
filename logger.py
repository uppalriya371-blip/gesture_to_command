"""
logger.py
---------
Writes gesture events to gesture_log.txt and exposes the last entry
for on-screen display.
"""

import os
from datetime import datetime
from typing import Optional

LOG_FILENAME = "gesture_log.txt"


class Logger:
    """
    Appends timestamped gesture events to a plain-text log file.

    Log format (one entry per line)
    --------------------------------
    YYYY-MM-DD HH:MM:SS | N Finger(s) | <Command>
    """

    def __init__(self, log_path: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        log_path : full path for the log file.  Defaults to
                   gesture_log.txt in the same directory as this module.
        """
        if log_path is None:
            log_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), LOG_FILENAME
            )
        self._log_path   : str           = log_path
        self._last_entry : Optional[str] = None
        self._last_count : int           = -1   # avoid duplicate consecutive logs
        self._last_cmd   : str           = ""

        # Ensure the file exists (touch it)
        if not os.path.exists(self._log_path):
            with open(self._log_path, "w", encoding="utf-8") as fh:
                fh.write("# Gesture-to-Command Mapping System — Log File\n")
                fh.write("# Format: Timestamp | Finger Count | Command\n")
                fh.write("-" * 60 + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, finger_count: int, command: str) -> None:
        """
        Write an entry only when the command actually changes.

        Parameters
        ----------
        finger_count : number of raised fingers
        command      : resolved command string
        """
        if finger_count == self._last_count and command == self._last_cmd:
            return   # no change – skip

        finger_label = f"{finger_count} Finger{'s' if finger_count != 1 else ''}"
        timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry        = f"{timestamp} | {finger_label} | {command}"

        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(entry + "\n")

        self._last_entry = entry
        self._last_count = finger_count
        self._last_cmd   = command

    def last_entry(self) -> str:
        """Return the last written log entry, or a placeholder."""
        return self._last_entry or "No entries yet."

    def log_path(self) -> str:
        """Return the resolved absolute path of the log file."""
        return self._log_path
