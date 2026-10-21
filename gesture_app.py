import os
import sys
import time
import math
import json
import argparse
import ctypes
import threading
import urllib.request
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

try:
    import comtypes
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.constants import CLSID_MMDeviceEnumerator, DEVICE_STATE, EDataFlow
    from pycaw.pycaw import (
        AudioUtilities,
        IAudioEndpointVolume,
        IMMDeviceEnumerator,
    )
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False

APP_VERSION = "1.7.0"

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

def model_asset_path():
    base = getattr(sys, "_MEIPASS", "")
    if base:
        bundled = os.path.join(base, MODEL_PATH)
        if os.path.exists(bundled):
            return bundled
    return MODEL_PATH


def detect_cameras(max_index=6):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            found.append(i)
    return found


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

TIP_IDS = {4, 8, 12, 16, 20}
GESTURE_TABLE = {
    (1, 1, 1, 1, 1): "Открытая ладонь",
    (0, 1, 0, 0, 0): "Указательный",
    (0, 1, 1, 0, 0): "Победа (V)",
    (1, 1, 0, 0, 0): "Буква L",
    (0, 1, 1, 1, 0): "Три пальца",
    (1, 1, 1, 0, 0): "Три пальца (с большим)",
    (0, 1, 1, 1, 1): "Четыре пальца",
    (0, 0, 0, 0, 1): "Мизинец",
    (0, 0, 1, 0, 0): "Средний палец",
    (0, 0, 0, 1, 0): "Безымянный",
    (1, 0, 0, 0, 1): "Shaka",
    (0, 1, 0, 0, 1): "Коза",
    (1, 1, 0, 0, 1): "Коза (ILY)",
    (0, 0, 0, 1, 1): "Безымянный и мизинец",
    (0, 0, 1, 0, 1): "Средний и мизинец",
    (1, 1, 1, 1, 0): "Четыре (с большим)",
}

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

GESTURE_ACTIONS = {
    "Указательный": ("key", VK_MEDIA_NEXT_TRACK, "Следующий трек"),
    "Победа (V)": ("key", VK_MEDIA_PREV_TRACK, "Предыдущий трек"),
    "Коза": ("key", VK_MEDIA_PLAY_PAUSE, "Play/Pause"),
    "Четыре пальца": ("combo", (0x44,), "Рабочий стол"),
}

_font_cache = {}


def ensure_model(path=MODEL_PATH):
    if not os.path.exists(path):
        print("Скачивание модели MediaPipe...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Модель загружена.")
    return path


def get_font(size):
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype("arial.ttf", size)
        except OSError:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_at(p1, p2, p3):
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos_a))

class KeyController:

    KEYEVENTF_KEYUP = 2
    KEYEVENTF_EXTENDEDKEY = 1
    VK_LWIN = 0x5B

    def __init__(self):
        self.user32 = ctypes.windll.user32

    def press(self, vk):
        flags = self.KEYEVENTF_EXTENDEDKEY if 0xA6 <= vk <= 0xB5 else 0
        self.user32.keybd_event(vk, 0, flags, 0)
        self.user32.keybd_event(vk, 0, flags | self.KEYEVENTF_KEYUP, 0)

    def combo(self, *vks):
        self.user32.keybd_event(self.VK_LWIN, 0, 0, 0)
        for vk in vks:
            self.user32.keybd_event(vk, 0, 0, 0)
            self.user32.keybd_event(vk, 0, self.KEYEVENTF_KEYUP, 0)
        self.user32.keybd_event(self.VK_LWIN, 0, self.KEYEVENTF_KEYUP, 0)

    def execute(self, action):
        kind, payload = action
        if kind == "key":
            self.press(payload)
        elif kind == "combo":
            self.combo(*payload)


class GestureReporter:
    def __init__(self, url, timeout=1.5, min_interval=0.2):
        self.url = url
        self.timeout = timeout
        self.min_interval = min_interval
        self.enabled = True
        self.sent = 0
        self.failed = 0
        self.last_error = None
        self._last_key = None
        self._last_send = 0.0
        self._lock = threading.Lock()

    def set_url(self, url):
        self.url = url
        self._last_key = None

    def make_key(self, hands):
        return tuple(
            (h["hand"], h["gesture"], round(h["x"], 3), round(h["y"], 3))
            for h in hands
        )

    def update(self, hands, fps):
        if not self.enabled or not self.url:
            return False
        key = self.make_key(hands)
        now = time.monotonic()
        if key == self._last_key:
            return False
        if now - self._last_send < self.min_interval:
            return False
        if not self._lock.acquire(blocking=False):
            return False
        self._last_key = key
        self._last_send = now
        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "fps": round(float(fps), 1),
            "hands": [
                {
                    "side": h["hand"],
                    "gesture": h["gesture"],
                    "x": round(float(h["x"]), 3),
                    "y": round(float(h["y"]), 3),
                }
                for h in hands
            ],
        }
        threading.Thread(target=self._post, args=(payload,), daemon=True).start()
        return True

    def _post(self, payload):
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            urllib.request.urlopen(request, timeout=self.timeout).close()
            self.sent += 1
            self.last_error = None
        except Exception as e:
            self.failed += 1
            self.last_error = str(e)
        finally:
            self._lock.release()


class ActionGate:
    def __init__(self, delay=0.0):
        self.delay = max(0.0, delay)
        self.gesture = None
        self.since = 0.0
        self.fired = False

    def update(self, gesture, now):
        if gesture is None or gesture != self.gesture:
            self.gesture = gesture
            self.since = now
            self.fired = False
        if self.gesture is None or self.fired:
            return False
        if now - self.since >= self.delay:
            self.fired = True
            return True
        return False


class AudioController:
    REFRESH_INTERVAL = 2.0

    def __init__(self):
        self._endpoints = None
        self._cache_time = 0.0

    def _endpoints_list(self):
        now = time.time()
        if self._endpoints is None or now - self._cache_time > self.REFRESH_INTERVAL:
            enumerator = comtypes.CoCreateInstance(
                CLSID_MMDeviceEnumerator,
                IMMDeviceEnumerator,
                comtypes.CLSCTX_INPROC_SERVER,
            )
            collection = enumerator.EnumAudioEndpoints(
                EDataFlow.eRender.value, DEVICE_STATE.ACTIVE.value
            )
            endpoints = []
            for i in range(collection.GetCount()):
                device = collection.Item(i)
                wrapper = AudioUtilities.CreateDevice(device)
                endpoints.append((device.GetId(), wrapper.EndpointVolume))
            self._endpoints = endpoints
            self._cache_time = now
        return self._endpoints

    def get_muted(self):
        return any(bool(ep.GetMute()) for _, ep in self._endpoints_list())

    def mute_all(self):
        changed = False
        for _, ep in self._endpoints_list():
            if not ep.GetMute():
                ep.SetMute(True, None)
                changed = True
        return changed

    def unmute_all(self):
        changed = False
        for _, ep in self._endpoints_list():
            if ep.GetMute():
                ep.SetMute(False, None)
                changed = True
        return changed

    def get_volume(self):
        levels = [ep.GetMasterVolumeLevelScalar() for _, ep in self._endpoints_list()]
        return sum(levels) / len(levels) if levels else 0.0

    def change_volume_all(self, delta):
        changed = False
        for _, ep in self._endpoints_list():
            current = ep.GetMasterVolumeLevelScalar()
            new_level = min(1.0, max(0.0, current + delta))
            if abs(new_level - current) > 1e-4:
                ep.SetMasterVolumeLevelScalar(new_level, None)
                changed = True
        return changed


class HandAnalyzer:
    def __init__(self, smooth_window=8, min_votes=3):
        self.smooth_window = smooth_window
        self.min_votes = min_votes
        self.history = {}

    def fingers_state(self, pts):
        wrist = pts[0]
        palm = dist(wrist, pts[9])
        if palm == 0:
            return [0, 0, 0, 0, 0], 0.0
        thumb_angle = angle_at(pts[2], pts[3], pts[4])
        thumb_spread = dist(pts[4], pts[17]) > dist(pts[3], pts[17]) * 1.05
        thumb = 1 if thumb_angle > 150 and thumb_spread else 0
        fingers = [thumb]
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            fingers.append(
                1 if dist(pts[tip], wrist) > dist(pts[pip], wrist) * 1.05 else 0
            )
        return fingers, palm

    def classify(self, pts, fingers, palm):
        if palm == 0:
            return "Неизвестный"
        if dist(pts[4], pts[8]) < palm * 0.4 and all(fingers[2:]):
            return "OK"
        if fingers == [0, 0, 0, 0, 0]:
            return "Кулак"
        if fingers == [1, 0, 0, 0, 0]:
            if pts[4][1] < pts[0][1] - palm * 0.3:
                return "Большой палец вверх"
            if pts[4][1] > pts[0][1] + palm * 0.3:
                return "Большой палец вниз"
            return "Большой палец (вбок)"
        return GESTURE_TABLE.get(tuple(fingers), "Неизвестный")

    def stable_gesture(self, hand_key, gesture):
        dq = self.history.setdefault(hand_key, deque(maxlen=self.smooth_window))
        dq.append(gesture)
        best, votes = Counter(dq).most_common(1)[0]
        return best if votes >= self.min_votes else None


class GestureRecognizer:
    def __init__(self, model_path=None, num_hands=2):
        model_path = model_path or model_asset_path()
        ensure_model(model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.analyzer = HandAnalyzer()
        self._timestamp_ms = 0

    def detect(self, frame_bgr):
        self._timestamp_ms += 33
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(image, self._timestamp_ms)

    def process(self, frame_bgr):
        result = self.detect(frame_bgr)
        return self.annotate(frame_bgr, result)

    def annotate(self, img, result):
        h, w = img.shape[:2]
        texts = []
        hands_info = []

        if result.hand_landmarks and result.handedness:
            for idx, (lms, handedness) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                side = handedness[0].category_name
                color = (0, 220, 0) if side == "Right" else (0, 160, 255)
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

                for a, b in HAND_CONNECTIONS:
                    cv2.line(img, pts[a], pts[b], color, 2)
                for i, p in enumerate(pts):
                    r = 6 if i in TIP_IDS else 4
                    cv2.circle(img, p, r, color, -1)

                fingers, palm = self.analyzer.fingers_state(pts)
                raw = self.analyzer.classify(pts, fingers, palm)
                stable = self.analyzer.stable_gesture((side, idx), raw)
                label = stable if stable else "..."
                hands_info.append(
                    {
                        "hand": side,
                        "raw": raw,
                        "gesture": stable or raw,
                        "stable": stable,
                        "fingers": fingers,
                        "x": lms[0].x,
                        "y": lms[0].y,
                    }
                )

                text = f"{side}: {label}"
                lx = min(max(pts[0][0] - 20, 5), w - 10)
                ly = max(pts[0][1] - 70, 5)
                texts.append(((lx, ly), text, (255, 255, 255), 24, (30, 30, 30)))

        texts.append(
            ((10, 10), "Q - выход", (200, 200, 200), 20, (30, 30, 30))
        )
        img = self.draw_texts(img, texts)
        return img, hands_info

    @staticmethod
    def draw_texts(img, texts):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for pos, text, color, size, bg in texts:
            font = get_font(size)
            box = draw.textbbox(pos, text, font=font)
            if bg is not None:
                draw.rectangle(
                    [box[0] - 6, box[1] - 3, box[2] + 6, box[3] + 3], fill=bg
                )
            draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def close(self):
        self.landmarker.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Управление звуком и треками жестами руки через камеру."
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Индекс камеры (по умолчанию: 0)"
    )
    parser.add_argument(
        "--hands", type=int, default=2,
        help="Максимум рук в кадре (по умолчанию: 2)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Сколько секунд нужно держать жест до срабатывания команды (по умолчанию: 0)"
    )
    parser.add_argument(
        "--volume-step", type=float, default=10.0,
        help="Шаг изменения громкости в процентах (по умолчанию: 10)"
    )
    parser.add_argument(
        "--post-url", default="",
        help="URL для POST-отправки распознанных жестов в JSON "
             "(например, http://192.168.1.10:8080/report)"
    )
    parser.add_argument(
        "--no-actions", action="store_true",
        help="Без команд системе: только распознавание и POST-отправка"
    )
    parser.add_argument(
        "--version", action="version",
        version=f"Жесты рук v{APP_VERSION}"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    recognizer = GestureRecognizer(num_hands=args.hands)

    audio = None
    if PYCAW_AVAILABLE:
        try:
            audio = AudioController()
        except Exception as e:
            print("Аудиоуправление недоступно:", e)
    else:
        print("pycaw не установлен, управление звуком отключено (pip install pycaw).")

    volume_actions = {
        "Большой палец вверх": args.volume_step / 100.0,
        "Большой палец вниз": -args.volume_step / 100.0,
    }

    keys = KeyController()
    mute_gate = ActionGate(args.delay)
    media_gate = ActionGate(args.delay)
    volume_gate = ActionGate(args.delay)

    reporter = GestureReporter(args.post_url) if args.post_url else None
    actions_on = not args.no_actions

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру {args.camera}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps_smooth = 0.0
    print("Управление: Кулак - звук выкл, Ладонь - звук вкл,")
    print("Указательный - след. трек, Победа (V) - пред. трек, Коза - Play/Pause,")
    print(
        f"Большой палец вверх/вниз - громкость +{args.volume_step:g}%/-{args.volume_step:g}%."
    )
    if reporter is not None:
        print(f"POST-отправка жестов: {args.post_url}")
    if not actions_on:
        print("Команды системе отключены (--no-actions).")
    if args.delay > 0:
        print(f"Задержка срабатывания: {args.delay:g} с.")
    print("Распознавание жестов запущено. Нажмите Q в окне для выхода.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Кадр не получен, завершение работы.")
            break
        frame = cv2.flip(frame, 1)

        frame, hands_info = recognizer.process(frame)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        fps_smooth = fps if fps_smooth == 0 else fps_smooth * 0.9 + fps * 0.1

        if reporter is not None:
            reporter.update(hands_info, fps_smooth)

        gestures = [info["stable"] for info in hands_info]

        if actions_on and audio is not None:
            mute_gesture = next(
                (g for g in gestures if g in ("Кулак", "Открытая ладонь")), None
            )
            if mute_gate.update(mute_gesture, now):
                if mute_gesture == "Кулак":
                    if audio.mute_all():
                        print("Звук выключен")
                else:
                    if audio.unmute_all():
                        print("Звук включен")

            volume_gesture = next(
                (g for g in gestures if g in volume_actions), None
            )
            if volume_gate.update(volume_gesture, now):
                delta = volume_actions[volume_gesture]
                if audio.change_volume_all(delta):
                    percent = int(round(audio.get_volume() * 100))
                    sign = "+" if delta > 0 else "-"
                    print(f"Громкость {percent}% ({sign}{args.volume_step:g}%)")

        if actions_on:
            media_gesture = next((g for g in gestures if g in GESTURE_ACTIONS), None)
            if media_gate.update(media_gesture, now):
                kind, payload, message = GESTURE_ACTIONS[media_gesture]
                keys.execute((kind, payload))
                print(message)

        status = "Рука не найдена" if not hands_info else f"FPS: {fps_smooth:.0f}"
        overlay = [((10, 40), status, (0, 255, 120), 20, (30, 30, 30))]
        if audio is not None:
            snd = "Звук: выкл" if audio.get_muted() else "Звук: вкл"
            snd_color = (0, 0, 255) if audio.get_muted() else (0, 255, 120)
            overlay.append(((10, 70), snd, snd_color, 20, (30, 30, 30)))
            vol = int(round(audio.get_volume() * 100))
            overlay.append(
                ((10, 100), f"Громкость: {vol}%", (0, 255, 120), 20, (30, 30, 30))
            )
        frame = recognizer.draw_texts(frame, overlay)

        cv2.imshow("Hand Gestures", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    main()
