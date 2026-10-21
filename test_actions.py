import sys

sys.path.insert(0, r"C:\i\gesture")

import gesture_app as ga


class Recorder:
    def __init__(self):
        self.calls = []

    def keybd_event(self, vk, scan, flags, extra):
        self.calls.append((vk, flags))


kc = ga.KeyController()
kc.user32 = Recorder()

kc.press(ga.VK_MEDIA_PLAY_PAUSE)
assert kc.user32.calls == [(0xB3, 1), (0xB3, 3)], kc.user32.calls
kc.user32.calls.clear()

kc.press(0x44)
assert kc.user32.calls == [(0x44, 0), (0x44, 2)], kc.user32.calls
kc.user32.calls.clear()

kc.execute(("key", ga.VK_MEDIA_NEXT_TRACK))
assert kc.user32.calls == [(0xB0, 1), (0xB0, 3)], kc.user32.calls
kc.user32.calls.clear()

kc.execute(("combo", (0x44,)))
assert kc.user32.calls == [
    (0x5B, 0), (0x44, 0), (0x44, 2), (0x5B, 2)
], kc.user32.calls

assert ga.GESTURE_ACTIONS["Четыре пальца"] == ("combo", (0x44,), "Рабочий стол")
for gesture, entry in ga.GESTURE_ACTIONS.items():
    assert entry[0] in ("key", "combo"), gesture

labels = set(ga.GESTURE_TABLE.values()) | {"OK"}
for gesture in ga.RESERVED_GESTURES:
    assert gesture in labels, f"reserved gesture not recognized: {gesture}"
assert not (set(ga.RESERVED_GESTURES) & set(ga.GESTURE_ACTIONS)), "reserved must have no command"

print("ALL ACTION TESTS PASSED")
