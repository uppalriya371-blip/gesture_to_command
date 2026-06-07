"""
main.py
-------
Entry point for the Gesture-to-Command Mapping System (Simulation).

Run
---
    python main.py

Keyboard controls
-----------------
    Q  — Quit
    R  — Reset command state
    L  — Print last logged command to terminal
"""

import sys
import time
import cv2
import numpy as np

from hand_detector      import HandDetector
from gesture_recognizer import GestureRecognizer
from command_mapper     import CommandMapper
from logger             import Logger


# ══════════════════════════════════════════════════════════════════════
# UI constants
# ══════════════════════════════════════════════════════════════════════

WINDOW_TITLE    = "Gesture-to-Command Mapping Simulator"
FONT            = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO       = cv2.FONT_HERSHEY_PLAIN

# Palette (BGR)
COL_WHITE   = (255, 255, 255)
COL_BLACK   = (0,   0,   0  )
COL_CYAN    = (0,   255, 220)
COL_GREEN   = (0,   230,  80)
COL_RED     = (60,  60,  220)
COL_DARK    = (18,  18,  24 )   # near-black panel bg
COL_PANEL   = (28,  28,  36 )   # slightly lighter panel
COL_ACCENT  = (0,   210, 255)

DASHBOARD_W = 340    # width of right-hand info panel (pixels)
TOP_BAR_H   = 64     # height of command banner


class MainApplication:
    """
    Orchestrates webcam capture, detection, command mapping, logging,
    and all UI rendering.
    """

    def __init__(self, camera_index: int = 0) -> None:
        # ── Sub-systems ───────────────────────────────────────────────
        self._detector   = HandDetector(
            detection_confidence=0.72,
            tracking_confidence=0.72,
        )
        self._recognizer = GestureRecognizer(
            stability_seconds=1.0,
            smoothing_window=10,
        )
        self._mapper     = CommandMapper()
        self._logger     = Logger()

        # ── Webcam ────────────────────────────────────────────────────
        self._cap = self._open_camera(camera_index)

        # ── Runtime state ─────────────────────────────────────────────
        self._fps           : float = 0.0
        self._prev_tick     : float = time.perf_counter()
        self._show_last_log : bool  = False
        self._last_log_ts   : float = 0.0    # timestamp to auto-hide overlay

        print(f"\n{'='*55}")
        print("  Gesture-to-Command Mapping System  — Simulation")
        print(f"{'='*55}")
        print(f"  Log file : {self._logger.log_path()}")
        print("  Controls : Q=Quit  R=Reset  L=Last Log")
        print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Blocking main loop; returns when the user presses Q or closes."""
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    print("[ERROR] Frame capture failed — camera disconnected?")
                    break

                frame = cv2.flip(frame, 1)   # mirror mode feels natural
                frame = self._process_frame(frame)

                cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF

                if   key == ord('q') or key == ord('Q'): break
                elif key == ord('r') or key == ord('R'): self._reset()
                elif key == ord('l') or key == ord('L'): self._show_log()

        except Exception as exc:
            print(f"[FATAL] Unexpected error: {exc}")
            raise
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Full detection → render pipeline for one frame."""
        # 1. Detection
        self._detector.process_frame(frame)

        # 2. Finger count + stability
        landmarks    = self._detector.get_landmarks()
        stable_count = self._recognizer.update(landmarks)
        confidence   = self._detector.confidence

        # 3. Command resolution
        entry = self._mapper.resolve(stable_count, confidence)

        # 4. Log on change
        if self._recognizer.changed:
            self._logger.log(stable_count, entry.command)

        # 5. FPS
        self._update_fps()

        # 6. Draw everything
        frame = self._detector.draw_landmarks(frame)
        frame = self._render_ui(frame, entry, confidence)

        return frame

    # ------------------------------------------------------------------
    # UI rendering
    # ------------------------------------------------------------------

    def _render_ui(self, frame: np.ndarray, entry, confidence: float) -> np.ndarray:
        fh, fw = frame.shape[:2]
        is_tracking = self._detector.hand_detected

        # ── 1. Extend canvas to include right dashboard ───────────────
        canvas = np.full((fh, fw + DASHBOARD_W, 3), COL_DARK, dtype=np.uint8)
        canvas[:, :fw] = frame

        # ── 2. Top command banner ─────────────────────────────────────
        self._draw_top_banner(canvas, fw, entry, is_tracking)

        # ── 3. Right-side dashboard panel ────────────────────────────
        self._draw_dashboard(canvas, fw, fh, entry, confidence, is_tracking)

        # ── 4. Watermark / title corner ───────────────────────────────
        cv2.putText(canvas, "DRONE CTRL SIM", (10, fh - 12),
                    FONT_MONO, 1.0, (60, 60, 80), 1, cv2.LINE_AA)

        # ── 5. Last-log overlay ───────────────────────────────────────
        if self._show_last_log and time.time() - self._last_log_ts < 4.0:
            self._draw_log_overlay(canvas, fw, fh)
        elif self._show_last_log:
            self._show_last_log = False

        return canvas

    # ── Banner ────────────────────────────────────────────────────────

    def _draw_top_banner(self, canvas, fw, entry, is_tracking):
        h = TOP_BAR_H
        color = self._mapper.command_color() if is_tracking else COL_RED

        # Background strip
        cv2.rectangle(canvas, (0, 0), (fw + DASHBOARD_W, h), COL_DARK, -1)

        # Accent left bar
        cv2.rectangle(canvas, (0, 0), (6, h), color, -1)

        # Icon circles
        for i, (fc, cmd) in enumerate(self._mapper.all_commands.items()):
            cx = 30 + i * 52
            active = (fc == entry.finger_count and is_tracking)
            c_fill = self._mapper.command_color(cmd) if active else (45, 45, 55)
            cv2.circle(canvas, (cx, h // 2), 18, c_fill, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, h // 2), 18, color if active else (70, 70, 80), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(fc), (cx - 6, h // 2 + 6),
                        FONT, 0.55, COL_WHITE if active else (120, 120, 130),
                        1, cv2.LINE_AA)

        # Command text
        cmd_txt = f"Command Received: {entry.command}"
        cv2.putText(canvas, cmd_txt, (360, 22),
                    FONT, 0.75, color, 1, cv2.LINE_AA)

        status_txt = "TRACKING" if is_tracking else "NO HAND DETECTED"
        status_col = COL_GREEN if is_tracking else COL_RED
        cv2.putText(canvas, status_txt, (360, 50),
                    FONT, 0.55, status_col, 1, cv2.LINE_AA)

        # Separator line
        cv2.line(canvas, (0, h), (fw + DASHBOARD_W, h), (50, 50, 60), 1)

    # ── Dashboard panel ───────────────────────────────────────────────

    def _draw_dashboard(self, canvas, fw, fh, entry, confidence, is_tracking):
        px = fw + 12   # left edge of panel content
        cmd_color = self._mapper.command_color()

        # Panel background (slightly lighter than canvas bg)
        cv2.rectangle(canvas, (fw, TOP_BAR_H), (fw + DASHBOARD_W, fh),
                      COL_PANEL, -1)
        cv2.line(canvas, (fw, TOP_BAR_H), (fw, fh), (50, 50, 65), 2)

        y = TOP_BAR_H + 30

        # ── Title ─────────────────────────────────────────────────────
        cv2.putText(canvas, "SYSTEM MONITOR", (px, y),
                    FONT, 0.65, COL_ACCENT, 1, cv2.LINE_AA)
        y += 6
        cv2.line(canvas, (px, y), (fw + DASHBOARD_W - 12, y), (50, 50, 70), 1)
        y += 22

        # ── Metrics ───────────────────────────────────────────────────
        metrics = [
            ("Finger Count", str(entry.finger_count),                  COL_CYAN),
            ("Command",      entry.command,                             cmd_color),
            ("Confidence",   f"{confidence * 100:.1f}%",               COL_GREEN if confidence > 0.8 else (80, 180, 255)),
            ("FPS",          f"{self._fps:.1f}",                        COL_WHITE),
            ("Status",       "Tracking" if is_tracking else "Idle",    COL_GREEN if is_tracking else COL_RED),
        ]
        for label, value, col in metrics:
            # Label
            cv2.putText(canvas, label, (px, y),
                        FONT_MONO, 1.15, (140, 140, 160), 1, cv2.LINE_AA)
            y += 20
            # Value (larger)
            cv2.putText(canvas, value, (px + 8, y),
                        FONT, 0.72, col, 1, cv2.LINE_AA)
            y += 28
            cv2.line(canvas, (px, y), (fw + DASHBOARD_W - 12, y), (40, 40, 52), 1)
            y += 12

        # ── Confidence bar ────────────────────────────────────────────
        y += 8
        cv2.putText(canvas, "Detection Confidence", (px, y),
                    FONT_MONO, 1.0, (140, 140, 160), 1, cv2.LINE_AA)
        y += 16
        bar_w     = DASHBOARD_W - 28
        bar_fill  = int(bar_w * confidence)
        bar_col   = COL_GREEN if confidence > 0.8 else (0, 180, 255) if confidence > 0.5 else COL_RED
        cv2.rectangle(canvas, (px, y), (px + bar_w, y + 14), (40, 40, 52), -1)
        cv2.rectangle(canvas, (px, y), (px + bar_fill, y + 14), bar_col, -1)
        cv2.rectangle(canvas, (px, y), (px + bar_w, y + 14), (70, 70, 85), 1)
        y += 30

        # ── Command reference table ───────────────────────────────────
        y += 8
        cv2.putText(canvas, "COMMAND REFERENCE", (px, y),
                    FONT_MONO, 1.0, (140, 140, 160), 1, cv2.LINE_AA)
        y += 6
        cv2.line(canvas, (px, y), (fw + DASHBOARD_W - 12, y), (50, 50, 70), 1)
        y += 18

        for fc, cmd in self._mapper.all_commands.items():
            active   = (fc == entry.finger_count and is_tracking)
            row_col  = self._mapper.command_color(cmd) if active else (100, 100, 115)
            prefix   = "▶ " if active else "  "
            row_txt  = f"{prefix}{fc}  {cmd}"
            cv2.putText(canvas, row_txt, (px, y),
                        FONT_MONO, 1.15, row_col, 1, cv2.LINE_AA)
            y += 22

        # ── Controls hint ─────────────────────────────────────────────
        y = fh - 52
        cv2.line(canvas, (px, y), (fw + DASHBOARD_W - 12, y), (40, 40, 52), 1)
        y += 16
        cv2.putText(canvas, "Q Quit  R Reset  L Log", (px, y),
                    FONT_MONO, 1.0, (80, 80, 100), 1, cv2.LINE_AA)
        y += 20
        cv2.putText(canvas, f"Log: {self._logger.log_path()[-28:]}", (px, y),
                    FONT_MONO, 0.85, (60, 60, 78), 1, cv2.LINE_AA)

    # ── Last-log overlay ──────────────────────────────────────────────

    def _draw_log_overlay(self, canvas, fw, fh):
        last = self._logger.last_entry()
        ow, oh = 620, 56
        ox = (fw - ow) // 2
        oy = fh - oh - 20

        overlay = canvas.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + ow, oy + oh), (20, 20, 28), -1)
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
        cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), COL_ACCENT, 1)
        cv2.putText(canvas, "LAST LOG", (ox + 10, oy + 18),
                    FONT_MONO, 1.0, COL_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(canvas, last, (ox + 10, oy + 42),
                    FONT_MONO, 1.05, COL_WHITE, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _update_fps(self):
        now         = time.perf_counter()
        delta       = now - self._prev_tick
        self._fps   = 1.0 / delta if delta > 0 else 0.0
        self._prev_tick = now

    def _reset(self):
        self._recognizer.stable_count = 0
        self._recognizer._candidate_count = -1
        print("[INFO] Command state reset.")

    def _show_log(self):
        last = self._logger.last_entry()
        print(f"[LOG] {last}")
        self._show_last_log = True
        self._last_log_ts   = time.time()

    def _shutdown(self):
        print("\n[INFO] Shutting down…")
        self._detector.release()
        self._cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Goodbye.")

    # ------------------------------------------------------------------
    # Camera initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _open_camera(index: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(
                f"[ERROR] Cannot open camera at index {index}. "
                "Check that a webcam is connected and not in use."
            )
        # Prefer higher resolution & framerate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS,          30)
        return cap


# ══════════════════════════════════════════════════════════════════════

def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        app = MainApplication(camera_index=cam_index)
        app.run()
    except RuntimeError as err:
        print(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
