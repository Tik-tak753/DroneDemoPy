from __future__ import annotations

from PySide6.QtCore import QSize

from app.widgets import OverlayDetection, ScreenOverlayWidget


class ScreenCaptureService:
    def __init__(self) -> None:
        self._screen_overlay: ScreenOverlayWidget | None = None
        self._saved_non_screen_window_size = QSize()

    def enter_screen_mode(self, image_view, window) -> None:
        self._saved_non_screen_window_size = window.size()
        image_view.hide()
        compact_width = 300
        compact_height = max(window.height(), 600)
        window.resize(compact_width, compact_height)

    def exit_screen_mode(self, image_view, window) -> None:
        image_view.show()
        if self._saved_non_screen_window_size.isValid():
            window.resize(self._saved_non_screen_window_size)

    def ensure_overlay(self) -> ScreenOverlayWidget:
        if self._screen_overlay is None:
            self._screen_overlay = ScreenOverlayWidget()
        return self._screen_overlay

    def sync_overlay_region(self, screen_region: dict[str, int] | None) -> None:
        if screen_region is None:
            return
        overlay = self.ensure_overlay()
        overlay.set_region_geometry(screen_region)

    def show_overlay_prediction(self, screen_region: dict[str, int] | None, prediction) -> None:
        if screen_region is None:
            return

        overlay = self.ensure_overlay()
        overlay.set_region_geometry(screen_region)
        detections = [
            OverlayDetection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=label,
                confidence=confidence,
            )
            for x1, y1, x2, y2, label, confidence in prediction.detections
        ]
        overlay.set_detections(detections)
        overlay.show()

    def hide_overlay(self) -> None:
        if self._screen_overlay is None:
            return
        self._screen_overlay.clear_detections()
        self._screen_overlay.hide()

    def clear_and_close_overlay(self) -> None:
        if self._screen_overlay is None:
            return
        self._screen_overlay.close()
        self._screen_overlay = None
