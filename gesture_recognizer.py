"""
gesture_recognizer.py
---------------------
Counts raised fingers from MediaPipe hand landmarks and applies
a temporal stability filter to eliminate command flickering.
"""

import time
from collections import deque
from typing import Optional
import numpy as np


# MediaPipe landmark indices
WRIST         = 0
THUMB_CMC     = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP     = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP    = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP      = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP     = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20


class GestureRecognizer:
    """
    Converts a MediaPipe NormalizedLandmarkList into a stable finger count.

    Uses a sliding-window majority vote plus a hold-timer to ensure the
    same count is seen continuously for `stability_seconds` before it
    is committed as the *stable* count.
    """

    def __init__(
        self,
        stability_seconds: float = 1.0,
        smoothing_window: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        stability_seconds : how long the same count must persist before
                            it becomes the confirmed stable count.
        smoothing_window  : number of recent frames used for majority vote.
        """
        self._stability_seconds  = stability_seconds
        self._history: deque[int] = deque(maxlen=smoothing_window)

        self._candidate_count : int   = 0
        self._candidate_start : float = time.time()

        self.raw_count    : int   = 0   # instantaneous (per frame)
        self.stable_count : int   = 0   # temporally filtered
        self.changed      : bool  = False  # True for exactly one frame on change

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, landmarks) -> int:
        """
        Compute raw & stable finger counts from *landmarks*.

        Parameters
        ----------
        landmarks : MediaPipe NormalizedLandmarkList (or None if no hand)

        Returns
        -------
        stable_count : the confirmed (flicker-free) finger count
        """
        self.changed = False

        if landmarks is None:
            self.raw_count = 0
            self._push_to_stability(0)
            return self.stable_count

        lm = landmarks.landmark

        # ── 1. Count fingers ─────────────────────────────────────────
        count = self._count_fingers(lm)
        self.raw_count = count
        self._history.append(count)

        # ── 2. Majority vote over recent frames ───────────────────────
        smoothed = self._majority(self._history)

        # ── 3. Stability timer ────────────────────────────────────────
        self._push_to_stability(smoothed)

        return self.stable_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _push_to_stability(self, value: int) -> None:
        """Commit *value* to stable_count only after it holds long enough."""
        now = time.time()
        if value != self._candidate_count:
            self._candidate_count = value
            self._candidate_start = now
        else:
            elapsed = now - self._candidate_start
            if elapsed >= self._stability_seconds and value != self.stable_count:
                self.stable_count = value
                self.changed      = True

    @staticmethod
    def _majority(history: deque) -> int:
        """Return the most common value in *history*."""
        from collections import Counter
        if not history:
            return 0
        return Counter(history).most_common(1)[0][0]

    @staticmethod
    def _count_fingers(lm) -> int:
        """
        Count extended fingers using tip-vs-PIP landmark comparison.

        Thumb is handled separately via X-axis comparison to account
        for orientation; fingers 2-5 use Y-axis (tip above PIP = raised).
        """
        fingers = 0

        # ── Thumb ─────────────────────────────────────────────────────
        # Compare tip.x to IP.x; direction depends on handedness.
        # We check both directions (works for either hand).
        if abs(lm[THUMB_TIP].x - lm[THUMB_MCP].x) > 0.04:
            fingers += 1

        # ── Fingers 2-5 ───────────────────────────────────────────────
        tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP,  PINKY_PIP]
        for tip_idx, pip_idx in zip(tips, pips):
            if lm[tip_idx].y < lm[pip_idx].y:   # y increases downward
                fingers += 1

        return fingers
