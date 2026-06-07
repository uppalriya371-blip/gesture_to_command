"""
command_mapper.py
-----------------
Maps finger counts to drone commands and maintains command history.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CommandEntry:
    """A single resolved command record."""
    finger_count : int
    command      : str
    confidence   : float   # 0.0 – 1.0


class CommandMapper:
    """
    Converts a finger count (0–5) into a named drone command.

    Provides the mapping table, resolves commands, and exposes the
    current command along with a short history for display purposes.
    """

    COMMAND_MAP: Dict[int, str] = {
        0: "Idle",
        1: "Takeoff",
        2: "Hover",
        3: "Move Forward",
        4: "Move Backward",
        5: "Land",
    }

    # BGR colours used by the UI for each command
    COMMAND_COLORS: Dict[str, tuple] = {
        "Idle"          : (100, 100, 100),
        "Takeoff"       : (0,   220,  80),
        "Hover"         : (0,   200, 255),
        "Move Forward"  : (0,   160, 255),
        "Move Backward" : (0,    80, 255),
        "Land"          : (0,   255, 160),
        "Unknown"       : (60,  60,  200),
    }

    def __init__(self, history_size: int = 20) -> None:
        self._history   : List[CommandEntry] = []
        self._max_hist  : int                = history_size
        self.current    : CommandEntry       = CommandEntry(0, "Idle", 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, finger_count: int, confidence: float) -> CommandEntry:
        """
        Resolve *finger_count* to a CommandEntry and update internal state.

        Parameters
        ----------
        finger_count : number of raised fingers (0–5)
        confidence   : detection confidence from HandDetector (0.0–1.0)

        Returns
        -------
        CommandEntry with the resolved command string.
        """
        command_str  = self.COMMAND_MAP.get(finger_count, "Unknown")
        entry        = CommandEntry(finger_count, command_str, confidence)
        self.current = entry

        # Append to history (bounded)
        self._history.append(entry)
        if len(self._history) > self._max_hist:
            self._history.pop(0)

        return entry

    def get_last(self) -> Optional[CommandEntry]:
        """Return the most recently logged command, or None."""
        return self._history[-1] if self._history else None

    def command_color(self, command: Optional[str] = None) -> tuple:
        """Return BGR colour for *command* (defaults to current command)."""
        cmd = command or self.current.command
        return self.COMMAND_COLORS.get(cmd, self.COMMAND_COLORS["Unknown"])

    def format_command_received(self) -> str:
        """Human-readable 'Command Received: …' string."""
        return f"Command Received: {self.current.command}"

    @property
    def all_commands(self) -> Dict[int, str]:
        """Expose the full mapping for UI reference panels."""
        return dict(self.COMMAND_MAP)
