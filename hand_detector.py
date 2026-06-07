"""
hand_detector.py
----------------
Handles webcam capture and MediaPipe hand landmark detection.
Encapsulates all MediaPipe Hands setup and per-frame processing.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple


class HandDetector:
    """
    Detects a single hand in a video frame using MediaPipe Hands.

    Attributes
    ----------
    results         : latest MediaPipe detection results
    confidence      : normalized landmark detection confidence (0.0–1.0)
    """

    # MediaPipe landmark drawing utilities (shared across instances)
    _mp_drawing       = mp.solutions.drawing_utils
    _mp_drawing_styles = mp.solutions.drawing_styles
    _mp_hands         = mp.solutions.hands

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ) -> None:
        """
        Parameters
        ----------
        max_hands              : maximum number of hands to detect (1 for speed)
        detection_confidence   : minimum confidence for initial detection
        tracking_confidence    : minimum confidence to keep tracking
        """
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.results    = None
        self.confidence = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run MediaPipe on *frame_bgr* (in-place BGR → RGB conversion).

        Returns the original BGR frame unchanged so callers can keep
        drawing on it.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # minor perf gain
        self.results = self._hands.process(rgb)
        rgb.flags.writeable = True

        # Derive a single confidence scalar from the first hand's landmarks
        if self.results and self.results.multi_handedness:
            self.confidence = self.results.multi_handedness[0].classification[0].score
        else:
            self.confidence = 0.0

        return frame_bgr

    def draw_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Overlay hand skeleton and bounding box onto *frame_bgr*.

        Returns the annotated frame.
        """
        if not self.hand_detected:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        for hand_landmarks in self.results.multi_hand_landmarks:
            # ── Skeleton ─────────────────────────────────────────────
            self._mp_drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style(),
            )
            # ── Bounding box ──────────────────────────────────────────
            frame_bgr = self._draw_bounding_box(frame_bgr, hand_landmarks, w, h)

        return frame_bgr

    def get_landmarks(self) -> Optional[object]:
        """Return the first detected hand's NormalizedLandmarkList, or None."""
        if self.hand_detected:
            return self.results.multi_hand_landmarks[0]
        return None

    @property
    def hand_detected(self) -> bool:
        """True if at least one hand was found in the last processed frame."""
        return (
            self.results is not None
            and self.results.multi_hand_landmarks is not None
        )

    def release(self) -> None:
        """Free MediaPipe resources."""
        self._hands.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_bounding_box(
        frame: np.ndarray,
        hand_landmarks,
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        """Draw a cyan bounding rectangle around the detected hand."""
        xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
        ys = [lm.y * frame_h for lm in hand_landmarks.landmark]

        pad   = 18
        x_min = max(int(min(xs)) - pad, 0)
        y_min = max(int(min(ys)) - pad, 0)
        x_max = min(int(max(xs)) + pad, frame_w)
        y_max = min(int(max(ys)) + pad, frame_h)

        # Outer glow effect (thick, semi-transparent rectangle)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_min - 3, y_min - 3), (x_max + 3, y_max + 3),
                      (0, 255, 220), 4)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Solid border
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                      (0, 255, 220), 2, cv2.LINE_AA)
        return frame
