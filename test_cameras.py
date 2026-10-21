"""Тест поиска камер.

Камеру, которую уже держит приложение, открывать повторно нельзя: второй
захват того же устройства ломает текущий (кадры перестают идти). Поэтому
detect_cameras принимает skip — индексы занятых камер пропускаются.
"""

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QCoreApplication

import gesture_app as ga

app = QCoreApplication(sys.argv)


class FakeCapture:
    """Заглушка cv2.VideoCapture: запоминает открытые индексы."""

    opened = []
    working = (0, 2)

    def __init__(self, index, backend=None):
        FakeCapture.opened.append(index)
        self._ok = index in FakeCapture.working

    def isOpened(self):
        return self._ok

    def read(self):
        return (self._ok, None)

    def release(self):
        pass


orig_capture = ga.cv2.VideoCapture
ga.cv2.VideoCapture = FakeCapture

try:
    # --- 1. Без skip перебираются все индексы ---
    FakeCapture.opened = []
    found = ga.detect_cameras(max_index=4)
    assert FakeCapture.opened == [0, 1, 2, 3], FakeCapture.opened
    assert found == [0, 2], found

    # --- 2. Занятые камеры не открываются вообще ---
    FakeCapture.opened = []
    found = ga.detect_cameras(max_index=4, skip={0, 2})
    assert FakeCapture.opened == [1, 3], FakeCapture.opened
    assert found == [], found

    # --- 3. CameraDetectWorker прокидывает skip в ядро ---
    from gesture_qt import CameraDetectWorker

    FakeCapture.opened = []
    got = []
    worker = CameraDetectWorker(skip={0})
    worker.resultReady.connect(got.append)
    worker.run()
    assert FakeCapture.opened == [1, 2, 3, 4, 5], FakeCapture.opened
    assert got == [[2]], got

    # --- 4. skip задаётся и списком, и кортежем ---
    FakeCapture.opened = []
    found = ga.detect_cameras(max_index=3, skip=[1])
    assert FakeCapture.opened == [0, 2], FakeCapture.opened
    assert found == [0, 2], found
finally:
    ga.cv2.VideoCapture = orig_capture

print("ALL CAMERA TESTS PASSED")
