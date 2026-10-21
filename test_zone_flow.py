import os
import sys
import tempfile
import types

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

import gesture_qt as gq
import gesture_app as ga

gq.CONFIG_PATH = os.path.join(tempfile.gettempdir(), "rufp_test_gq_config.json")
if os.path.exists(gq.CONFIG_PATH):
    os.remove(gq.CONFIG_PATH)

app = QApplication(sys.argv)

gq.MainWindow.start_worker = lambda self: None
win = gq.MainWindow(dict(gq.DEFAULT_CONFIG))

win.video._pm_geom = (10, 20, 800, 450)


def mouse_ev(ev_type, x, y, button, buttons):
    return QMouseEvent(
        ev_type, QPointF(x, y), QPointF(x, y), button, buttons, Qt.KeyboardModifier.NoModifier
    )


L = Qt.MouseButton.LeftButton
NO = Qt.MouseButton.NoButton

win.start_zone_selection()
assert win.video.selecting is True, "select mode not entered"

win.video.mousePressEvent(mouse_ev(QEvent.Type.MouseButtonPress, 100, 120, L, L))
assert win.video._origin is not None, "origin not captured on press"
assert not win.video._rubber.isHidden(), "rubber band not shown"

win.video.mouseMoveEvent(mouse_ev(QEvent.Type.MouseMove, 400, 300, L, L))
g = win.video._rubber.geometry()
print("rubber rect:", g.x(), g.y(), g.width(), g.height())

win.video.mouseReleaseEvent(mouse_ev(QEvent.Type.MouseButtonRelease, 400, 300, L, NO))
expected = ((100 - 10) / 800, (120 - 20) / 450, (400 - 10) / 800, (300 - 20) / 450)
got = win.roi
print("roi got:", got)
print("roi exp:", expected)
assert got is not None, "ROI was not set after release"
assert all(abs(a - b) < 1e-6 for a, b in zip(got, expected)), "ROI math wrong"
assert win.video.selecting is False
assert win.config["roi"] == list(expected)
assert os.path.exists(gq.CONFIG_PATH), "config not saved"

win.worker = types.SimpleNamespace(roi=None, roi_enabled=False)
win.on_roi_toggled(False)
assert win.worker.roi_enabled is False
win.on_roi_toggled(True)
assert win.worker.roi_enabled is True
win.reset_zone()
assert win.roi is None and win.worker.roi is None and win.worker.roi_enabled is False

worker = gq.VideoWorker(0, 1)
worker.roi = (0.1, 0.1, 0.9, 0.9)
worker.roi_enabled = True
frame = np.full((720, 1280, 3), 200, dtype=np.uint8)


class FakeResult:
    hand_landmarks = []
    handedness = []


class FakeRec:
    analyzer = None

    def detect(self, frame):
        return FakeResult()


f2, hands, fps = worker._process_frame(frame.copy(), FakeRec(), 0.0, 0.0)
assert f2.shape == frame.shape
assert not np.array_equal(f2, frame), "ROI dimming not applied"
print("process_frame ROI ok, hands:", hands)

dlg = gq.SettingsDialog(dict(gq.DEFAULT_CONFIG), None, [0, 2], win.bindings)
cfg_out = dlg.result_config()
assert cfg_out["actions_enabled"] is True
assert cfg_out["camera"] == 0
assert cfg_out["bindings"]["Кулак"] == "audio:mute_all"

import time as _t
win.actions_enabled = False
win.apply_actions(
    [{"hand": "Right", "gesture": "Кулак", "stable": "Кулак", "x": 0.5, "y": 0.5}],
    _t.time(),
)
assert win.action_gate.gesture is None, "action gate must be reset when disabled"

gesture0 = dlg.tbl_bindings.item(0, 0).text()
combo0 = dlg.tbl_bindings.cellWidget(0, 1)
assert gesture0 == "Открытая ладонь"
assert combo0.currentData() == "audio:unmute_all"

combo0.setCurrentIndex(combo0.findData("key:0xB3"))
assert dlg.bindings[gesture0] == "key:0xB3"
combo0.setCurrentIndex(combo0.findData(""))
assert gesture0 not in dlg.bindings
assert dlg.result_config()["bindings"] == dlg.bindings

matched = ga.match_binding(win.bindings, ["Указательный"])
assert matched == "key:0xB0"

print("ALL OFFSCREEN ZONE TESTS PASSED")
