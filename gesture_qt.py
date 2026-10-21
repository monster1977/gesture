import os
import sys
import json
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QRect, Signal
from PySide6.QtGui import QAction, QImage, QPixmap, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QRubberBand,
)

import gesture_app as ga

def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


CONFIG_PATH = os.path.join(_app_dir(), "gesture_qt_config.json")

DEFAULT_CONFIG = {
    "camera": 0,
    "hands": 2,
    "delay": 0.0,
    "volume_step": 10.0,
    "roi": None,
    "roi_enabled": False,
    "post_url": "",
    "post_enabled": False,
    "actions_enabled": True,
    # None = привязки не заданы (используются DEFAULT_BINDINGS),
    # {} = явно «без команд»
    "bindings": None,
}


def load_config():
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in DEFAULT_CONFIG:
                if key in data:
                    config[key] = data[key]
        except (OSError, ValueError):
            pass
    return config


def save_config(config):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


class VideoWorker(QThread):
    stateReady = Signal(object)
    errorOccurred = Signal(str)

    def __init__(self, camera_index=0, num_hands=2, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.num_hands = num_hands
        self.roi = None
        self.roi_enabled = False
        self.reporter = None
        self._running = True

    def stop(self):
        self._running = False
        self.wait(3000)

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.errorOccurred.emit(str(e))

    def _open_capture(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _run(self):
        cap = self._open_capture()
        if cap is None:
            self.errorOccurred.emit(
                f"Не удалось открыть камеру {self.camera_index}."
            )
            return
        recognizer = ga.GestureRecognizer(num_hands=self.num_hands)
        prev_time = time.time()
        fps = 0.0
        failures = 0

        while self._running:
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            if not ok:
                # Кадр не пришёл: камеру могло занять другое приложение.
                # Пробуем переподключиться и только затем сдаёмся.
                failures += 1
                cap.release()
                cap = self._open_capture()
                if cap is None or failures >= 5:
                    self.errorOccurred.emit("Кадр не получен, камера отключена.")
                    break
                time.sleep(0.3)
                continue
            failures = 0
            frame = cv2.flip(frame, 1)
            frame, hands_info, fps = self._process_frame(
                frame, recognizer, prev_time, fps
            )
            prev_time = time.time()

            if self.reporter is not None:
                self.reporter.update(hands_info, fps)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            image = QImage(
                np.ascontiguousarray(rgb).data,
                w,
                h,
                3 * w,
                QImage.Format.Format_RGB888,
            ).copy()
            self.stateReady.emit({"image": image, "hands": hands_info, "fps": fps})

        cap.release()
        recognizer.close()

    def _process_frame(self, frame, recognizer, prev_time, fps):
        h, w = frame.shape[:2]
        result = recognizer.detect(frame)
        analyzer = recognizer.analyzer

        hands_info = []
        texts = []

        roi_px = None
        if self.roi_enabled and self.roi:
            x1, y1, x2, y2 = self.roi
            roi_px = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))

        if result.hand_landmarks and result.handedness:
            for idx, (lms, handedness) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                side = handedness[0].category_name
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

                if roi_px is not None:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    cx = (min(xs) + max(xs)) // 2
                    cy = (min(ys) + max(ys)) // 2
                    rx1, ry1, rx2, ry2 = roi_px
                    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                        continue

                color = (0, 220, 0) if side == "Right" else (0, 160, 255)
                for a, b in ga.HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], color, 2)
                for i, p in enumerate(pts):
                    r = 6 if i in ga.TIP_IDS else 4
                    cv2.circle(frame, p, r, color, -1)

                fingers, palm = analyzer.fingers_state(pts)
                raw = analyzer.classify(pts, fingers, palm)
                stable = analyzer.stable_gesture((side, idx), raw)
                hands_info.append(
                    {
                        "hand": side,
                        "gesture": stable or raw,
                        "stable": stable,
                        "x": lms[0].x,
                        "y": lms[0].y,
                    }
                )
                label = stable if stable else "..."
                text = f"{side}: {label}"
                lx = min(max(pts[0][0] - 20, 5), w - 10)
                ly = max(pts[0][1] - 70, 5)
                texts.append(((lx, ly), text, (255, 255, 255), 24, (30, 30, 30)))

        if roi_px is not None:
            rx1, ry1, rx2, ry2 = roi_px
            mask = np.zeros((h, w), dtype=bool)
            mask[ry1:ry2, rx1:rx2] = True
            dimmed = (frame * 0.35).astype(np.uint8)
            frame = np.where(mask[..., None], frame, dimmed)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 180), 2)
            texts.append(
                (
                    (rx1 + 8, ry1 + 8),
                    "Зона распознавания",
                    (0, 255, 180),
                    20,
                    (30, 30, 30),
                )
            )
        elif self.roi_enabled:
            texts.append(
                ((10, 10), "Зона не задана", (0, 200, 255), 20, (30, 30, 30))
            )

        now = time.time()
        if now > prev_time:
            instant = 1.0 / (now - prev_time)
            fps = instant if fps == 0 else fps * 0.9 + instant * 0.1

        frame = ga.GestureRecognizer.draw_texts(frame, texts)
        return frame, hands_info, fps


class CameraDetectWorker(QThread):
    """Поиск камер в фоновом потоке, чтобы не блокировать окно."""

    resultReady = Signal(object)

    def __init__(self, skip=(), parent=None):
        super().__init__(parent)
        self.skip = set(skip)

    def run(self):
        try:
            cameras = ga.detect_cameras(skip=self.skip)
        except Exception:
            cameras = []
        self.resultReady.emit(cameras)


class VideoLabel(QLabel):
    zoneSelected = Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(800, 450)
        self.setStyleSheet("background: #101010;")
        self._pm_geom = None
        self.selecting = False
        self._origin = None
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)

    def set_pm_geom(self, geom):
        self._pm_geom = geom

    def _to_normalized(self, pos):
        if not self._pm_geom:
            return None
        ox, oy, pw, ph = self._pm_geom
        if pw <= 0 or ph <= 0:
            return None
        nx = (pos.x() - ox) / pw
        ny = (pos.y() - oy) / ph
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def mousePressEvent(self, event):
        if self.selecting and event.button() == Qt.MouseButton.LeftButton:
            n = self._to_normalized(event.position())
            if n:
                self._origin = n
                self._rubber.setGeometry(QRect(event.position().toPoint(), event.position().toPoint()))
                self._rubber.show()
            return
        if event.button() == Qt.MouseButton.RightButton and self.selecting:
            self._cancel()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selecting and self._origin:
            n = self._to_normalized(event.position())
            if n:
                self._rubber.setGeometry(
                    self._norm_to_rect(self._origin, n)
                )
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.selecting and event.button() == Qt.MouseButton.LeftButton and self._origin:
            n = self._to_normalized(event.position())
            origin = self._origin
            self._rubber.hide()
            self.selecting = False
            self._origin = None
            self.unsetCursor()
            if n:
                x1, y1, x2, y2 = self._ordered(origin, n)
                if (x2 - x1) >= 0.03 and (y2 - y1) >= 0.03:
                    self.zoneSelected.emit(x1, y1, x2, y2)
                    return
        super().mouseReleaseEvent(event)

    def _cancel(self):
        self._rubber.hide()
        self.selecting = False
        self._origin = None
        self.unsetCursor()

    @staticmethod
    def _ordered(a, b):
        return min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])

    def _norm_to_rect(self, a, b):
        x1, y1, x2, y2 = self._ordered(a, b)
        ox, oy, pw, ph = self._pm_geom
        return QRect(
            int(ox + x1 * pw), int(oy + y1 * ph),
            int((x2 - x1) * pw), int((y2 - y1) * ph),
        )


class SettingsDialog(QDialog):
    CUSTOM_KEY = "#custom"
    CUSTOM_WIN = "#custom_win"

    def __init__(self, config, parent=None, cameras=None, bindings=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры")
        self.config = dict(config)
        self.bindings = (
            dict(bindings) if bindings is not None else dict(ga.DEFAULT_BINDINGS)
        )

        self.combo_camera = QComboBox()
        cam_list = list(cameras) if cameras else list(range(10))
        current = int(self.config["camera"])
        if current not in cam_list:
            cam_list.append(current)
            cam_list.sort()
        for i in cam_list:
            self.combo_camera.addItem(f"Камера {i}", int(i))
        pos = self.combo_camera.findData(current)
        self.combo_camera.setCurrentIndex(pos if pos >= 0 else 0)

        self.spin_hands = QSpinBox()
        self.spin_hands.setRange(1, 4)
        self.spin_hands.setValue(int(self.config["hands"]))

        self.spin_delay = QDoubleSpinBox()
        self.spin_delay.setRange(0.0, 5.0)
        self.spin_delay.setSingleStep(0.1)
        self.spin_delay.setSuffix(" с")
        self.spin_delay.setValue(float(self.config["delay"]))

        self.spin_volume = QDoubleSpinBox()
        self.spin_volume.setRange(1.0, 50.0)
        self.spin_volume.setSingleStep(1.0)
        self.spin_volume.setSuffix(" %")
        self.spin_volume.setValue(float(self.config["volume_step"]))

        self.chk_roi = QCheckBox("Использовать зону распознавания")
        self.chk_roi.setChecked(bool(self.config["roi_enabled"]))

        self.edit_post_url = QLineEdit(str(self.config.get("post_url") or ""))
        self.edit_post_url.setPlaceholderText("http://192.168.1.10:8080/report")
        self.chk_post = QCheckBox("Отправлять жесты по HTTP POST (JSON)")
        self.chk_post.setChecked(bool(self.config.get("post_enabled")))

        self.chk_actions = QCheckBox(
            "Отправлять команды системе (звук, медиа-клавиши)"
        )
        self.chk_actions.setChecked(bool(self.config.get("actions_enabled", True)))

        self.tbl_bindings = self._build_bindings_table()

        form = QFormLayout()
        form.addRow("Камера:", self.combo_camera)
        form.addRow("Максимум рук:", self.spin_hands)
        form.addRow("Задержка срабатывания:", self.spin_delay)
        form.addRow("Шаг громкости:", self.spin_volume)
        form.addRow(self.chk_roi)
        form.addRow("URL POST:", self.edit_post_url)
        form.addRow(self.chk_post)
        form.addRow(self.chk_actions)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )

        lbl_bindings = QLabel("Жесты и действия:")
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(lbl_bindings)
        layout.addWidget(self.tbl_bindings)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def _build_bindings_table(self):
        table = QTableWidget(len(ga.ALL_GESTURES), 2)
        table.setHorizontalHeaderLabels(["Жест", "Действие"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        table.setColumnWidth(0, 230)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)

        for row, gesture in enumerate(ga.ALL_GESTURES):
            item = QTableWidgetItem(gesture)
            table.setItem(row, 0, item)
            combo = self._make_action_combo(gesture)
            table.setCellWidget(row, 1, combo)
        return table

    def _make_action_combo(self, gesture):
        combo = QComboBox()
        combo.addItem("Нет команды", "")
        for spec, label in ga.COMMAND_CATALOG:
            combo.addItem(label, spec)
        combo.addItem("Своя клавиша (VK)…", self.CUSTOM_KEY)
        combo.addItem("Win + клавиша (VK)…", self.CUSTOM_WIN)
        combo.setProperty("baseItemCount", combo.count())
        spec = self.bindings.get(gesture, "")
        self._select_spec(combo, spec)
        combo.currentIndexChanged.connect(
            lambda _ix, cb=combo, g=gesture: self._on_action_selected(cb, g)
        )
        return combo

    @staticmethod
    def _remove_custom_items(combo):
        """Убрать динамические пункты «Своя: …», чтобы список не разрастался."""
        base = int(combo.property("baseItemCount") or 0)
        combo.blockSignals(True)
        for index in range(combo.count() - 1, base - 1, -1):
            combo.removeItem(index)
        combo.blockSignals(False)

    @staticmethod
    def _select_spec(combo, spec):
        pos = combo.findData(spec)
        if pos < 0 and spec:
            combo.blockSignals(True)
            combo.addItem(f"Своя: {spec}", spec)
            pos = combo.count() - 1
            combo.setCurrentIndex(pos)
            combo.blockSignals(False)
        elif pos >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(pos)
            combo.blockSignals(False)

    def _on_action_selected(self, combo, gesture):
        data = combo.currentData()
        if data in (None, ""):
            self._remove_custom_items(combo)
            self.bindings.pop(gesture, None)
            return
        if data in (self.CUSTOM_KEY, self.CUSTOM_WIN):
            previous = self.bindings.get(gesture, "")
            title = (
                "Своя клавиша: VK-код"
                if data == self.CUSTOM_KEY
                else "Win + клавиша: VK-код"
            )
            text, ok = QInputDialog.getText(
                self, title,
                "Введите VK-код (например 0xB3 или 179):",
            )
            spec = ""
            if ok:
                try:
                    vk = int(text.strip(), 0)
                except ValueError:
                    vk = 0
                if 0 < vk < 256:
                    spec = (
                        f"key:{vk}" if data == self.CUSTOM_KEY else f"combo:{vk}"
                    )
            self._remove_custom_items(combo)
            if spec:
                self.bindings[gesture] = spec
            elif ok:
                self.bindings.pop(gesture, None)
            else:
                # отмена ввода — возвращаем прежнюю привязку
                spec = previous
                if spec:
                    self.bindings[gesture] = spec
                else:
                    self.bindings.pop(gesture, None)
            self._select_spec(combo, spec)
            return
        self._remove_custom_items(combo)
        self.bindings[gesture] = data

    def result_config(self):
        return {
            "camera": int(self.combo_camera.currentData()),
            "hands": self.spin_hands.value(),
            "delay": self.spin_delay.value(),
            "volume_step": self.spin_volume.value(),
            "roi_enabled": self.chk_roi.isChecked(),
            "post_url": self.edit_post_url.text().strip(),
            "post_enabled": self.chk_post.isChecked(),
            "actions_enabled": self.chk_actions.isChecked(),
            "bindings": dict(self.bindings),
        }


class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle(f"Жесты рук v{ga.APP_VERSION}")
        self.config = config
        self.roi = tuple(config["roi"]) if config.get("roi") else None
        config["roi_enabled"] = bool(config.get("roi_enabled")) and self.roi is not None
        self.audio = None
        if ga.PYCAW_AVAILABLE:
            try:
                self.audio = ga.AudioController()
            except Exception as e:
                print("Аудиоуправление недоступно:", e)
        self.keys = ga.KeyController()
        self.action_gate = ga.ActionGate(config["delay"])
        self.actions_enabled = bool(self.config.get("actions_enabled", True))
        saved_bindings = self.config.get("bindings")
        self.bindings = (
            dict(saved_bindings)
            if saved_bindings is not None
            else dict(ga.DEFAULT_BINDINGS)
        )

        self.video = VideoLabel()
        self.video.zoneSelected.connect(self.on_zone_selected)
        self.setCentralWidget(self.video)

        self._build_menus()
        self._build_statusbar()

        self.worker = None
        self._cam_thread = None
        self.start_worker()

    def _build_menus(self):
        m_settings = self.menuBar().addMenu("Настройки")
        act_params = QAction("Параметры…", self)
        act_params.setShortcut(QKeySequence("Ctrl+P"))
        act_params.triggered.connect(self.open_settings)
        m_settings.addAction(act_params)

        m_zone = self.menuBar().addMenu("Зона")
        self.act_roi_enabled = QAction("Использовать зону", self)
        self.act_roi_enabled.setCheckable(True)
        self.act_roi_enabled.setChecked(bool(self.config.get("roi_enabled")))
        self.act_roi_enabled.triggered.connect(self.on_roi_toggled)
        act_roi_select = QAction("Выбрать зону мышью", self)
        act_roi_select.triggered.connect(self.start_zone_selection)
        act_roi_reset = QAction("Сбросить зону", self)
        act_roi_reset.triggered.connect(self.reset_zone)
        m_zone.addAction(self.act_roi_enabled)
        m_zone.addAction(act_roi_select)
        m_zone.addAction(act_roi_reset)

        m_help = self.menuBar().addMenu("Справка")
        act_gestures = QAction("Жесты и действия…", self)
        act_gestures.triggered.connect(self.show_gestures_help)
        m_help.addAction(act_gestures)

    def _build_statusbar(self):
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_sound = QLabel("Звук: -")
        self.lbl_volume = QLabel("Громкость: -")
        self.lbl_hands = QLabel("Руки: -")
        self.lbl_post = QLabel("POST: выкл")
        for w in (
            self.lbl_hands,
            self.lbl_volume,
            self.lbl_sound,
            self.lbl_post,
            self.lbl_fps,
        ):
            self.statusBar().addPermanentWidget(w)
        self.statusBar().showMessage("Зона: не задана. Меню «Зона» → «Выбрать зону мышью».")

    def start_worker(self):
        if self.worker is not None:
            self.worker.stop()
        worker = VideoWorker(self.config["camera"], self.config["hands"])
        worker.roi = self.roi
        worker.roi_enabled = bool(self.config.get("roi_enabled")) and self.roi is not None
        post_url = str(self.config.get("post_url") or "").strip()
        if self.config.get("post_enabled") and post_url:
            worker.reporter = ga.GestureReporter(post_url)
        worker.stateReady.connect(self.on_state)
        worker.errorOccurred.connect(self.on_worker_error)
        self.worker = worker
        worker.start()

    def on_state(self, payload):
        img = payload["image"]
        label_size = self.video.size()
        pm = QPixmap.fromImage(img).scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        ox = (label_size.width() - pm.width()) // 2
        oy = (label_size.height() - pm.height()) // 2
        self.video.set_pm_geom((ox, oy, pm.width(), pm.height()))
        self.video.setPixmap(pm)

        hands = payload["hands"]
        now = time.time()
        self.apply_actions(hands, now)
        self.update_status(hands, payload["fps"])

    def apply_actions(self, hands, now):
        if not self.actions_enabled:
            self.action_gate.update(None, now)
            return
        gestures = [h["stable"] for h in hands]
        spec = ga.match_binding(self.bindings, gestures)
        if self.action_gate.update(spec, now):
            action = ga.parse_action(spec)
            if action is not None:
                ga.execute_action(
                    action,
                    self.audio,
                    self.keys,
                    float(self.config["volume_step"]) / 100.0,
                )
                print(ga.COMMAND_LABELS.get(spec, spec))

    def update_status(self, hands, fps):
        self.lbl_fps.setText(f"FPS: {fps:.0f}")
        if self.audio is not None:
            muted = self.audio.get_muted()
            self.lbl_sound.setText("Звук: выкл" if muted else "Звук: вкл")
            self.lbl_sound.setStyleSheet(
                "color: #d9534f;" if muted else "color: #5cb85c;"
            )
            self.lbl_volume.setText(
                f"Громкость: {int(round(self.audio.get_volume() * 100))}%"
            )
        if hands:
            self.lbl_hands.setText(
                " | ".join(f"{h['hand']}: {h['gesture']}" for h in hands)
            )
        else:
            self.lbl_hands.setText("Руки: нет")
        rep = self.worker.reporter if self.worker is not None else None
        if rep is None:
            self.lbl_post.setText("POST: выкл")
            self.lbl_post.setStyleSheet("")
        else:
            info = f"POST: {rep.sent}"
            if rep.failed:
                info += f"/ошибок: {rep.failed}"
                self.lbl_post.setStyleSheet("color: #d9534f;")
            else:
                self.lbl_post.setStyleSheet("color: #5cb85c;")
            self.lbl_post.setText(info)

    def on_worker_error(self, message):
        self.statusBar().showMessage(f"Ошибка: {message}", 10000)
        QMessageBox.critical(self, "Ошибка", message)

    def open_settings(self):
        if self._cam_thread is not None:
            self.statusBar().showMessage("Поиск камер уже выполняется…")
            return
        self.statusBar().showMessage("Поиск камер…")
        # Камеру, которую уже держит приложение, открывать нельзя:
        # второй захват того же устройства ломает текущий.
        busy = {int(self.config["camera"])}
        thread = CameraDetectWorker(skip=busy, parent=self)
        thread.resultReady.connect(self._on_cameras_detected)
        # Удалять поток можно только после его завершения: иначе Qt
        # прервётся с "QThread: Destroyed while thread is still running".
        thread.finished.connect(self._on_camera_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._cam_thread = thread
        thread.start()

    def _on_camera_thread_finished(self):
        self._cam_thread = None

    def _on_cameras_detected(self, cameras):
        self.statusBar().clearMessage()
        cameras = list(cameras)
        current = int(self.config["camera"])
        if current not in cameras:
            cameras.append(current)
        dialog = SettingsDialog(self.config, self, cameras, self.bindings)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.apply_settings(dialog.result_config())

    def apply_settings(self, new_cfg):
        restart = (
            new_cfg["camera"] != self.config["camera"]
            or new_cfg["hands"] != self.config["hands"]
        )
        self.config.update(new_cfg)
        self.actions_enabled = bool(self.config.get("actions_enabled", True))
        self.bindings = dict(new_cfg.get("bindings") or {})
        roi_on = bool(self.config["roi_enabled"]) and self.roi is not None
        self.config["roi_enabled"] = roi_on
        self.act_roi_enabled.setChecked(roi_on)
        if self.worker is not None:
            self.worker.roi_enabled = roi_on
        if self.worker is not None:
            post_url = str(self.config.get("post_url") or "").strip()
            if self.config.get("post_enabled") and post_url:
                if self.worker.reporter is None:
                    self.worker.reporter = ga.GestureReporter(post_url)
                else:
                    self.worker.reporter.set_url(post_url)
            else:
                self.worker.reporter = None
        self.action_gate.delay = float(self.config["delay"])
        self.save()
        if restart:
            self.start_worker()
        if not self.actions_enabled:
            self.statusBar().showMessage(
                "Команды системе отключены: только распознавание и POST.", 5000
            )
        else:
            self.statusBar().showMessage("Параметры применены.", 5000)

    def start_zone_selection(self):
        self.video.selecting = True
        self.video.setCursor(Qt.CursorShape.CrossCursor)
        self.statusBar().showMessage(
            "Выделите зону мышью по кадру. Правая кнопка - отмена.", 10000
        )

    def on_zone_selected(self, x1, y1, x2, y2):
        self.roi = (x1, y1, x2, y2)
        self.config["roi"] = [x1, y1, x2, y2]
        self.config["roi_enabled"] = True
        self.act_roi_enabled.setChecked(True)
        if self.worker is not None:
            self.worker.roi = self.roi
            self.worker.roi_enabled = True
        self.save()
        self.statusBar().showMessage(
            f"Зона задана: ({x1:.2f}, {y1:.2f}) - ({x2:.2f}, {y2:.2f}).", 8000
        )

    def on_roi_toggled(self, checked):
        if checked and self.roi is None:
            self.act_roi_enabled.setChecked(False)
            self.statusBar().showMessage(
                "Сначала задайте зону: «Зона» → «Выбрать зону мышью».", 8000
            )
            return
        self.config["roi_enabled"] = bool(checked)
        if self.worker is not None:
            self.worker.roi_enabled = bool(checked) and self.roi is not None
        self.save()

    def reset_zone(self):
        self.roi = None
        self.config["roi"] = None
        self.config["roi_enabled"] = False
        self.act_roi_enabled.setChecked(False)
        if self.worker is not None:
            self.worker.roi = None
            self.worker.roi_enabled = False
        self.save()
        self.statusBar().showMessage("Зона сброшена, распознаётся весь кадр.", 8000)

    def show_gestures_help(self):
        lines = []
        for gesture in ga.ALL_GESTURES:
            spec = self.bindings.get(gesture)
            if spec:
                lines.append(f"{gesture} - {ga.COMMAND_LABELS.get(spec, spec)}")
        free = [
            g for g in ga.RESERVED_GESTURES if g not in self.bindings
        ]
        if free:
            lines.append("")
            lines.append(
                "Распознаются без команды: " + ", ".join(free) + "."
            )
        QMessageBox.information(
            self,
            "Жесты и действия",
            "\n".join(lines)
            + "\n\n"
            "Медиа-клавиши работают глобально, плеер может быть в фоне.\n\n"
            "Назначение действий: «Настройки» → «Параметры…», таблица\n"
            "«Жесты и действия»; там же можно задать любой VK-код.\n\n"
            "POST-отправка распознанных жестов в JSON настраивается\n"
            "в тех же «Параметрах» (URL и включение).\n\n"
            "Команды работают только с руками в зоне распознавания (если она задана).",
        )

    def save(self):
        save_config(self.config)

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
        if self._cam_thread is not None:
            self._cam_thread.wait(3000)
        super().closeEvent(event)


def main():
    args = ga.parse_args()
    config = load_config()
    if not os.path.exists(CONFIG_PATH):
        config.update(
            {
                "camera": args.camera,
                "hands": args.hands,
                "delay": args.delay,
                "volume_step": args.volume_step,
            }
        )
    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.resize(1000, 640)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
